"""Branch A production driver: one molecule -> one basin list.

PRODUCTION. This is the only supported entry point for branch A. Everything else
under scripts/ that touches conformers is either a calibration (it produces a
number used to decide something, not a deliverable) or lives in scripts/_superseded/.

What comes out
--------------
A basin list. Each basin carries: the geometry converged on MACE-OFF23_medium, the
electronic energy, an analytic Hessian with zero imaginary frequencies, its own
external symmetry number, the RRHO thermodynamic terms, and a Boltzmann weight.

It is the ONLY structure source for the other two branches -- branch B starts its
unbiased trajectories from these geometries, and branch C computes its
RI-MP2/RIJK/cc-pVTZ reference labels on them.

The six steps (plan_A section 3)
--------------------------------
  1. gates F0-F7                       -> is this molecule in scope at all
  2. CREST iMTD-GC, gfn2 workhorse     -> raw ensemble
     + MACE refine="sp" over a socket (was "opt" until 2026-09-04; criterion 3)
  3. pool in the reference geometry    -> an independent starting point
  4. tighten to fmax = 1e-4 eV/A       -> tighter than CREST's optlev="tight"
  5. all-atom best-RMSD dedup, 0.30 A  -> graph automorphism solves the atom mapping
  6. analytic Hessian                  -> imaginary screen, sigma, RRHO, weights

Steps 4-6 exist because CREST's conformer count is NOT a basin count. Measured
(defect 54): acetone, CREST reports 2 conformers 0.8118 kcal/mol apart; tightened
to fmax = 1e-4 they are identical to the last digit. That 0.8118 was CREST's
convergence residual, not an energy difference.

Two settings that are NOT this script's to choose
-------------------------------------------------
The dynamics package (SHAKE on all bonds, 5 fs, hydrogen mass 2 amu) is the
published protocol of Grimme, JCTC 2019, 15, 2847, and it belongs to BRANCH A
ONLY. It buys barrier crossing at the price of a trajectory that is no longer real
thermal motion, which is a fair trade when the product is a geometry and a ruinous
one when the product is a fluctuation. Branch B restarts from these geometries with
dt = 1 fs, no constraints, real hydrogen mass, and uses no CREST frame.

Usage
-----
    python scripts/production/s0_A_pipeline.py --species dsgdb9nsd_000018
    python scripts/production/s0_A_pipeline.py --species dsgdb9nsd_000018 --skip-crest \\
        --reuse ~/runs/openQHA/branchA/dsgdb9nsd_000018
"""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

def _repo_root():
    """Directory holding the openQHA package, found by walking up from this file.

    Depth-independent on purpose: this file keeps working wherever under the
    repository it is moved to. The earlier move into `scripts/_superseded/` broke
    every `parents[1]` in the moved files silently, which is what this replaces.
    """
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


sys.path.insert(0, str(_repo_root()))

import numpy as np  # noqa: E402
from ase.io import read  # noqa: E402

from openqha import (S0_ROOT, config, conformers, crest, crest_census,  # noqa: E402
                     engine, filters, hessian, symmetry, thermo)

EV_TO_KCAL = conformers.EV_TO_KCAL


# ======================================================================================
# Provenance -- every product carries the conditions it was made under
# ======================================================================================
def machine_load():
    """Load average and core count. A cost number without these is not reportable.

    D0-P3-6: two earlier benchmarks that differed by 4.5x were both contaminated by
    a competing job nobody had recorded.
    """
    try:
        one, five, fifteen = os.getloadavg()
    except (OSError, AttributeError):
        one = five = fifteen = None
    return dict(loadavg_1min=one, loadavg_5min=five, loadavg_15min=fifteen,
                cpu_count=os.cpu_count(), node=platform.node())


# ======================================================================================
# Step 1 -- the gates
# ======================================================================================
def gate(qid, cfg, smiles=None):
    """Run filters F0-F7 on this molecule. Returns the screening record.

    `filters.screen` short-circuits, so the failing gate is unique and is the whole
    attribution: it names one gate, never a list.

    F7 asks whether a molecule is on QM9's official "uncharacterized" list, and that
    list is published BY QM9 INDEX. A molecule specified by SMILES has no index, so
    F7 has no input. It is then dropped from the gate list and the reason is written
    into the record -- `filters.screen` itself raises rather than let an enabled gate
    pass silently on missing input, and that rule is right; this is the caller
    declaring the gate inapplicable in the open, which is a different thing.
    """
    spec = cfg["species"].get(qid) if qid else None
    if smiles is None:
        # config.qm9_smiles falls back to the geometry file's own SMILES line when the
        # 119 MB index table is not present, so a molecule can be run from curatedQM9
        # alone.
        smiles = spec["smiles"] if spec else config.qm9_smiles(qid, cfg)

    gates = list(filters.enabled_gates(cfg))
    skipped = None
    if qid is None and "F7" in gates:
        gates.remove("F7")
        skipped = ("F7 needs a QM9 index (the uncharacterized list is published by "
                   "index) and this molecule was given as SMILES. The gate is "
                   "declared inapplicable here rather than silently passed.")

    passed, failed_gate, reason = filters.screen(
        smiles, cfg=cfg, gates=gates, identifier=qid)
    return dict(smiles=smiles, passed=bool(passed), failed_gate=failed_gate,
                reason=reason, gates_enabled=gates, gate_skipped=skipped,
                smiles_source=("declared_in_config" if spec
                               else ("given on the command line" if qid is None
                                     else "QM9 index row")))


# ======================================================================================
# Step 1b -- a starting geometry for a molecule this repo does not ship
# ======================================================================================
def seed_geometry_from_smiles(smiles, calc, dest, seed=20260903, fmax=1e-3):
    """Build ONE starting structure from SMILES and relax it on the potential.

    ############################################################################
    # Read this before assuming ETKDG has been un-retired. It has not.         #
    ############################################################################
    The 2026-09-03 ruling retired ETKDG as a CONFORMER SEARCH METHOD -- as the thing
    that produces the ensemble whose members become basins. That is not what happens
    here. Here it produces ONE structure, which is then thrown at CREST, and **every
    conformer in the result comes from CREST's metadynamics**, not from this seed.
    Upstream's own example is `crest struc.xyz --gfn2`; `struc.xyz` has to come from
    somewhere, and for a molecule with no deposited geometry this is where.

    Concretely, what would break the ruling and is NOT done: embedding many
    conformers and letting them into the basin list, or pooling extra embeddings
    alongside the CREST ensemble. `n_embed=1`.

    **What is genuinely lost, and must be said rather than hidden.** For the seven
    shipped species, step 3 pools the deposited QM9 geometry as an INDEPENDENT
    starting point, which is the only thing left that can bound the systematic error
    plan_A section 1.2 records (CREST-only conformational correction: mean +0.1209,
    max +0.5824 kcal/mol). A SMILES-specified molecule has no such independent
    geometry, so for it that error is **unbounded**, and acceptance criterion 6 is
    reported as not-applicable rather than passed.
    """
    from ase import Atoms

    mol, rec = conformers.embed(smiles, n_embed=1, seed=seed, mmff_prune=True)
    cid = int(mol.GetConformers()[0].GetId())
    atoms = conformers._mol_to_atoms(mol, cid)
    energy, residual, converged, steps = conformers.optimise(
        atoms, calc, fmax=fmax, steps=2000)
    conformers.write_xyz(
        atoms, dest,
        comment="{} seed geometry, single ETKDG embedding relaxed on the "
                "potential to fmax={:g} eV/A".format(smiles, fmax))
    return dict(
        smiles=smiles, path=str(dest), embed_seed=int(seed),
        n_embeddings=1,
        embed_record={k: v for k, v in rec.items() if k != "smiles"},
        relaxed_energy_eV=float(energy),
        residual_force_eV_A=float(residual),
        converged=bool(converged), opt_steps=int(steps),
        role=("STARTING STRUCTURE ONLY. Every conformer in the result comes from "
              "CREST's metadynamics. ETKDG is not used as a conformer search here "
              "and remains retired for that purpose."))


# ======================================================================================
# Step 2 -- CREST
# ======================================================================================
def crest_scratch_dir(name):
    """Where CREST RUNS: node-local, never the shared filesystem (ADR 0002, Q20).

    `$S0_SCRATCH/openqha_crest/<name>/` -- hpc/env/tianhe.sh sets S0_SCRATCH to
    /tmp/<user>/<jobid>; off-cluster the system temp directory, per process. CREST
    writes dozens of small files per molecule into parallel `_N` subdirectories, and
    doing that on Lustre was measured slow for the job and for everyone else on the
    machine (docs/tianhe_runbook.md section 6). The finished directory is copied once
    into the molecule directory by `move_crest_dir`.
    """
    import tempfile
    base = os.environ.get("S0_SCRATCH") or os.path.join(
        tempfile.gettempdir(), "openqha_{}".format(os.getpid()))
    return Path(base) / "openqha_crest" / str(name)


def move_crest_dir(scratch, final):
    """Copy a finished CREST directory into the molecule directory, once, then remove
    the node-local copy. A destination that already exists (a failed earlier run) is
    overwritten file by file, never left half old and half new."""
    import shutil
    scratch, final = Path(scratch), Path(final)
    if not scratch.is_dir():
        return False
    final.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(scratch, final, dirs_exist_ok=True)
    shutil.rmtree(scratch, ignore_errors=True)
    return True


def run_crest(qid, workdir, cfg, args, start_xyz=None):
    """CREST iMTD-GC under the published protocol, with the pointwise fallback.

    `start_xyz` is the structure CREST starts from. For a shipped species it is the
    deposited QM9 geometry; for a SMILES-specified molecule the caller has already
    built one (see `seed_geometry_from_smiles`). Either way it is a STARTING point,
    not a conformer -- CREST's metadynamics produces the conformers.

    `workdir` is the FINAL directory, `<molecule>/crest/` (ADR 0001). CREST itself runs
    in `crest_scratch_dir(name)` and the finished directory is moved here once, the
    SHAKE fallback's into `crest_shake<N>/` beside it (ticket 06, 2026-09-14). A CREST
    that fails is moved too, so its crest.out is where a reader looks.
    """
    c = cfg["crest"]
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    xyz = workdir / "{}.xyz".format(qid or "molecule")
    src = Path(start_xyz) if start_xyz else config.qm9_xyz(qid, cfg)
    if src.resolve() != xyz.resolve():
        xyz.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    wanted = dict(runtype=c["runtype"], optlev=c["optlev"], refine=c["refine"],
                  shake=c["shake"], workhorse=c["workhorse"], tstep=c["tstep_fs"])
    existing = crest.read_input_settings(workdir / "input.toml")
    if existing is not None and (workdir / "crest_conformers.xyz").exists():
        ok, why = crest.settings_match(existing, wanted)
        if ok and not args.force_crest:
            rec = crest.record_from_dir(workdir, settings=wanted)
            rec["reused_scratch"] = True
            # D0-P1-12: a reused directory yields no cost measurement for THIS run.
            # CREST's own printed wall-time is carried across so the number is not
            # lost, but it is flagged as not a cost of this invocation.
            rec["wall_is_valid_cost"] = False
            rec["wall_seconds"] = rec.get("seconds")
            rec["shake_used"] = c["shake"]
            rec["used_shake_fallback"] = False
            return rec
        if not ok:
            raise RuntimeError(
                "scratch directory {} was made under different settings ({}). "
                "Refusing to reuse it: mixing two sampling conditions into one "
                "dataset is defect 57, and it is not visible in the products. "
                "Delete it or pass --force-crest.".format(workdir, why))

    fb = c.get("shake_fallback") or {}
    fallback_to = int(fb.get("to", 1))
    scratch = crest_scratch_dir(workdir.name if qid is None else qid)
    scratch_shake = Path("{}_shake{}".format(scratch, fallback_to))
    final_shake = Path("{}_shake{}".format(workdir, fallback_to))
    if scratch.exists():
        import shutil
        shutil.rmtree(scratch, ignore_errors=True)       # a dead job's leftovers
    started = time.time()
    try:
        rec = crest.run_with_shake_fallback(
            scratch, xyz,
            shake=int(c["shake"]),
            fallback_to=fallback_to,
            enabled=bool(fb.get("enabled", True)),
            runtype=c["runtype"], threads=int(args.threads), optlev=c["optlev"],
            refine=c["refine"], backend=c["backend"],
            engine_client=S0_ROOT / c["engine_client"],
            workhorse=c["workhorse"], tstep_fs=float(c["tstep_fs"]),
            calcspace=getattr(args, "keep_calcspace", None),
            timeout_s=int(args.timeout_s))
    finally:
        # Moved whatever CREST left -- a failure's crest.out included -- and the
        # fallback's directory when there is one. Once; then the node-local copy goes.
        move_crest_dir(scratch, workdir)
        move_crest_dir(scratch_shake, final_shake)
    rec["wall_seconds"] = time.time() - started
    rec["reused_scratch"] = False
    rec["wall_is_valid_cost"] = True
    # The record names the FINAL places, and says where the run actually happened.
    rec["ran_in"] = str(scratch_shake if rec.get("used_shake_fallback") else scratch)
    rec["workdir"] = str(final_shake if rec.get("used_shake_fallback") else workdir)
    if rec.get("first_attempt"):
        rec["first_attempt"]["workdir"] = str(workdir)
        rec["first_attempt"]["ran_in"] = str(scratch)
    return rec


# ======================================================================================
# Steps 3-6 -- the repo's own criteria
# ======================================================================================
def basin_list(qid, smiles, frames, comments, calc, cfg, args, reference_xyz=None,
               molecule_dir=None):
    """Tighten, dedup, Hessian-screen, and label. Returns (record, basins, mol).

    `molecule_dir`: where MACE's engine files go (`mace/confNN`, `mace/basinNN`; ADR 0001).
    """
    pkg1, pkg2 = cfg["package1"], cfg["package2"]
    fmax = float(pkg2["fmax_hessian_eV_A"])
    thr = float(pkg1["dedup_rmsd_A"])
    temperature = config.temperature(cfg)

    frames = list(frames)
    comments = list(comments)
    n_from_crest = len(frames)

    # ---- step 3: the reference geometry is pooled in as an independent start --------
    #
    # With ETKDG retired (user ruling 2026-09-03) this is the ONLY starting point that
    # does not come from CREST. Without it there is nothing left that could bound the
    # systematic error plan_A section 1.2 records (+0.1209 mean, +0.5824 max kcal/mol
    # for CREST-only), so it is not optional.
    if reference_xyz is not None and Path(reference_xyz).exists():
        # config.read_qm9_xyz, not ase.io.read: QM9 files carry frequency, SMILES and
        # InChI lines after the atoms, and ASE reads those as a malformed second frame.
        ref = config.read_qm9_xyz(reference_xyz)
        frames.append(ref)
        comments.append("reference geometry (QM9 native), pooled by branch A step 3")
    n_reference_added = len(frames) - n_from_crest

    spec = cfg["species"].get(qid)
    # CREGEN three-fold criterion (plan_A section 9). `rmsd_only` restores the pre
    # 2026-09-04 behaviour, which is kept reproducible rather than deleted.
    three_fold = pkg1.get("dedup_criterion", "cregen_three_fold") == "cregen_three_fold"
    ethr = float(pkg1["dedup_ethr_kcal"]) if three_fold else None
    bthr = float(pkg1["dedup_bthr_relative"]) if three_fold else None

    rec, basins, mol = crest_census.census_from_frames(
        smiles, frames, calc, name=qid, fmax=fmax, threshold_A=thr,
        temperature_K=temperature, do_hessian=True, reject_imaginary=True,
        species=None,                       # sigma is done per basin below, not here
        comments=comments,
        hessian_mode=args.hessian_mode,
        ethr_kcal=ethr, bthr_rel=bthr,
        molecule_dir=molecule_dir)

    rec["n_frames_from_crest"] = n_from_crest
    rec["n_reference_geometries_pooled"] = n_reference_added

    # ---- correct a count that pooling would otherwise corrupt ----------------------
    #
    # `census_from_frames` fills `n_conformers_reported_by_crest` with the number of
    # frames it was handed, which was true before this step existed. Step 3 pools the
    # reference geometry into the same list, so that field silently became
    # "CREST's conformers PLUS ours" -- measured on acetone 2026-09-03: it printed 3
    # where CREST had reported 2.
    #
    # That is precisely the confusion acceptance criterion 7 exists to prevent
    # (D0-P1-1, D0-95: CREST's conformer count is not the basin count), arriving
    # from the opposite direction -- not the basin count borrowing CREST's number,
    # but CREST's number quietly absorbing ours. The three counts are now separate
    # and none of them is inferred from another.
    cv = rec["crest_vs_repo"]
    cv["n_input_frames_total"] = int(n_from_crest + n_reference_added)
    cv["n_conformers_reported_by_crest"] = int(n_from_crest)
    cv["n_reference_geometries_pooled"] = int(n_reference_added)
    cv["n_collapsed"] = int(cv["n_input_frames_total"]
                            - cv["n_basins_by_repo_criteria"])

    # ---- which basin did the reference geometry land in ----------------------------
    # plan_A acceptance criterion 6. Reported as a basin index, or None if the
    # reference collapsed into something else during tightening.
    if n_reference_added:
        rec["reference_geometry_basin"] = _locate_reference(mol, rec, n_from_crest)

    return rec, basins, mol


def _locate_reference(mol, rec, ref_frame_index):
    """Which surviving basin the pooled reference geometry ended up in."""
    dup = {int(k): int(v) for k, v in rec["duplicate_map"].items()}
    kept = [int(c) for c in rec["basin_conformer_ids"]]
    cid = ref_frame_index                   # conformer ids are assigned in frame order
    home = dup.get(cid, cid)
    if home in kept:
        return dict(basin_index=kept.index(home), conformer_id=home,
                    was_merged=bool(home != cid))
    return dict(basin_index=None, conformer_id=home,
                note="the reference geometry did not survive the imaginary-frequency "
                     "screen, or merged into a rejected structure")


# ======================================================================================
# Step 6b -- symmetry number and thermodynamics, per basin
# ======================================================================================
def label_basins(qid, basins, rec, cfg, args):
    """Attach sigma and the RRHO terms to every basin.

    sigma is computed PER BASIN, not per molecule, because the point group is a
    property of a conformer: planar oxetane is C2v with sigma = 2 while the puckered
    ring is Cs with sigma = 1, and the two differ by kT ln 2 = 0.41 kcal/mol. The
    config says this in as many words for dsgdb9nsd_000048.
    """
    spec = cfg["species"].get(qid)
    declared = None
    conditional = None
    if spec:
        declared = int(spec["symmetry_number"])
        conditional = spec.get("conditional")
    temperature = config.temperature(cfg)

    out = []
    for i, atoms in enumerate(basins):
        # A declaration that the config itself marks CONDITIONAL is not a constant.
        # For those species the geometry decides and the declared value is recorded
        # as what an ideal geometry would give -- otherwise every puckered oxetane
        # basin would silently inherit the planar sigma = 2.
        use_declared = declared if (declared is not None and not conditional) else None
        srec = symmetry.analyse_atoms(atoms, declared=use_declared)
        if conditional:
            srec["declared_sigma_for_ideal_geometry"] = declared
            srec["declaration_is_conditional"] = True
            srec["conditional_note"] = spec.get("symmetry_reason")

        nu = np.asarray(rec["hessian"][str(rec["basin_conformer_ids"][i])]
                        ["frequencies_cm_inv"], dtype=float)
        g = thermo.g_minus_eel(
            atoms.get_masses(), atoms.get_positions(), nu,
            symmetry_number=int(srec["sigma"]),
            degeneracy=int(spec["electronic_degeneracy"]) if spec else 1,
            temperature_K=temperature)
        out.append(dict(
            basin_index=i,
            energy_eV=rec["basin_energies_eV"][i],
            relative_kcal=rec["basin_relative_kcal"][i],
            symmetry=srec,
            electronic_degeneracy=(int(spec["electronic_degeneracy"]) if spec else 1),
            electronic_degeneracy_source=("declared_in_config" if spec
                                          else "assumed_singlet"),
            thermo={k: v for k, v in g.items()},
            lowest_frequency_cm_inv=float(nu.min()),
            n_imaginary=int(rec["hessian"][str(rec["basin_conformer_ids"][i])]
                            ["n_imaginary"])))

    if not spec:
        for b in out:
            b["free_energy_warning"] = (
                "electronic degeneracy is not declared for this molecule; a closed-"
                "shell singlet was assumed. sigma came from the geometry and is "
                "sound, but g0 is an assumption and is flagged as one.")
    return out


# ======================================================================================
# Acceptance criteria, evaluated on the product itself
# ======================================================================================
def check_criteria(record, cfg_shake_fallback=None):
    """plan_A section 4. Every one of these can fail, and says so if it does."""
    c = record["crest"]
    rec = record["census"]
    basins = record["basins"]
    checks = []

    def add(n, name, passed, detail):
        checks.append(dict(number=n, criterion=name, passed=bool(passed),
                           detail=detail))

    # 1 -- workhorse identity: what was asked for is what CREST was actually given.
    #
    # Read back off the input.toml in the work directory, NOT off the settings dict
    # this script passed in. Comparing a variable with itself is a criterion that
    # cannot fail, and a criterion that cannot fail is not one (skills 2.4(a)).
    asked = record["settings"]["workhorse"]
    on_disk = crest.read_input_settings(Path(c["workdir"]) / "input.toml") or {}
    ran = on_disk.get("workhorse")
    add(1, "workhorse identity", ran == asked,
        "asked {!r}, {}/input.toml records {!r}".format(asked, c["workdir"], ran))

    # 1b -- the dynamics package went in whole. The three settings are one protocol,
    # so a run with two of the three is not the published protocol.
    # A molecule that FELL BACK deliberately did not run the published protocol, so the
    # expected shake for it is the fallback value, not the configured one. Checking it
    # against the published value would fail every legitimate fallback -- and a
    # criterion that fails on expected behaviour is one people learn to ignore.
    #
    # The fallback is not thereby excused: criterion 10 reports it, the record carries
    # `used_shake_fallback`, and the config states that such a molecule is reported
    # separately. What criterion 11 checks is that the package on disk is WHOLE and
    # matches what this run intended -- two of three settings is not a protocol.
    fb_to = (cfg_shake_fallback or {}).get("to", 1)
    fell_back = bool(c.get("used_shake_fallback"))
    # The published package has THREE members -- SHAKE all bonds, 5 fs, hydrogen mass
    # 2 amu (Grimme, JCTC 2019, 15, 2847) -- so all three are checked. hmass used to be
    # left to CREST's default with a comment saying the default was right. It IS right
    # (a plain MD prints `hydrogen mass /u : 2.00000`), but a criterion that trusts a
    # default in someone else's source tree cannot fail when that default changes, and
    # this one is named for checking the package whole.
    want_shake = float(fb_to) if fell_back else float(record["settings"]["shake"])
    want_hmass = float(record["settings"].get("hydrogen_mass_amu", 2.0))
    got_hmass, hmass_source = crest.effective_hmass(on_disk)
    whole = (on_disk.get("shake") is not None
             and float(on_disk["shake"]) == want_shake
             and on_disk.get("tstep") is not None
             and float(on_disk["tstep"]) == float(record["settings"]["tstep_fs"])
             and got_hmass == want_hmass)
    add(11, "dynamics package written as one" + (" (fallback)" if fell_back else ""),
        whole,
        "input.toml has shake={} tstep={} hmass={} ({}); this run intended "
        "shake={} tstep={} hmass={}{}".format(
            on_disk.get("shake"), on_disk.get("tstep"), got_hmass, hmass_source,
            want_shake, record["settings"]["tstep_fs"], want_hmass,
            "  [fallback: NOT the published protocol, reported separately]"
            if fell_back else ""))

    # 2 -- the record makes an unambiguous claim about cost.
    #
    # Passing does NOT mean "this was fast". It means the record says either "this
    # is a measured cost of this run" or "this reused a directory, so it is not a
    # cost of this run" -- and never leaves a reader to guess which. A wall clock
    # with no such flag beside it is the thing defect 34 was made of.
    # Two ways to be unambiguous, and BOTH pass:
    #   valid is True  -> the number is a cost of this run, so the number must be there
    #   valid is False -> this reused a directory, so there is no cost to state, and a
    #                     missing wall time is the honest record rather than a gap
    # Requiring a wall time in both cases made every reused run fail for being correct
    # (multibasin/dihydroxybutanone was recorded 22/23 on this alone). What still fails
    # is silence: `wall_is_valid_cost` absent, or claimed valid with no number.
    wall, valid = c.get("wall_seconds"), c.get("wall_is_valid_cost")
    ok2 = valid is not None and (wall is not None if valid else True)
    add(2, "cost claim is unambiguous", ok2,
        "wall {} s, reused_scratch={}, wall_is_valid_cost={}{}".format(
            None if wall is None else round(wall, 1),
            c.get("reused_scratch"), valid,
            "  [reused: no cost to state, and none claimed]"
            if valid is False else ""))

    # 4 -- zero imaginary frequencies, exactly 6 rigid modes removed, clean gap.
    bad = [b["basin_index"] for b in basins if b["n_imaginary"]]
    h = [rec["hessian"][str(cid)] for cid in rec["basin_conformer_ids"]]
    rigid_ok = all(x["n_rigid_modes_removed"] == 6 for x in h)
    gaps = [x["separation_gap_ratio"] for x in h if x["separation_gap_ratio"]]
    add(4, "zero imaginary, 6 rigid modes removed, separation > 1e8",
        not bad and rigid_ok and (not gaps or min(gaps) > 1e8),
        "imaginary in basins {}; rigid modes {}; min separation ratio {:.3g}".format(
            bad or "none",
            sorted({x["n_rigid_modes_removed"] for x in h}),
            min(gaps) if gaps else float("nan")))

    # 6 -- the reference geometry was pooled in and located.
    #
    # A molecule given by SMILES has no deposited geometry, so there is nothing to
    # pool. That is reported as NOT APPLICABLE with the consequence spelled out --
    # never as a pass. A criterion that quietly passes when its subject is absent is
    # worse than not having it.
    ref = rec.get("reference_geometry_basin")
    if rec.get("reference_geometry_note"):
        checks.append(dict(
            number=6, criterion="reference geometry pooled and located",
            passed=True, not_applicable=True,
            detail="NOT APPLICABLE: " + rec["reference_geometry_note"]))
    else:
        add(6, "reference geometry pooled and located",
            rec.get("n_reference_geometries_pooled", 0) > 0,
            "pooled {}; landed in basin {}".format(
                rec.get("n_reference_geometries_pooled"),
                None if ref is None else ref.get("basin_index")))

    # 7 -- CREST's conformer count is reported next to the basin count, never as it.
    # The three counts must all be present AND must add up: CREST's own conformers,
    # the geometries we pooled in, and the frames that actually went through the
    # tightening. If they do not add up, one of them has absorbed another.
    cv = rec["crest_vs_repo"]
    counts_consistent = (
        cv.get("n_conformers_reported_by_crest") is not None
        and cv.get("n_reference_geometries_pooled") is not None
        and cv.get("n_input_frames_total")
        == cv["n_conformers_reported_by_crest"]
        + cv["n_reference_geometries_pooled"])
    add(7, "CREST, pooled and basin counts are three separate numbers that add up",
        counts_consistent,
        "CREST {} + pooled {} = {} frames -> {} basins".format(
            cv.get("n_conformers_reported_by_crest"),
            cv.get("n_reference_geometries_pooled"),
            cv.get("n_input_frames_total"),
            cv["n_basins_by_repo_criteria"]))

    # 9 (new, this session) -- sigma is sound: the accepted proper operations form a
    # group, and the answer does not depend on where inside the working range the
    # tolerance was put.
    #
    # NOT "constant across the whole sweep". The loose end of the sweep exists to
    # show where the method breaks, and on acetone it does: sigma goes 2 -> 14 at
    # 0.40 A as the methyl rotor starts being admitted -- the D0-9 failure mode
    # appearing exactly where it should. Demanding a flat sweep would report that
    # evidence as a failure. What is required is a plateau containing the working
    # tolerance, plus a positive margin below the nearest rejected operation.
    bad_sigma = [b["basin_index"] for b in basins
                 if b["symmetry"].get("group_closure_defect")
                 or not b["symmetry"].get("sigma_stable_below_tolerance")
                 or (b["symmetry"].get("margin_A") is not None
                     and b["symmetry"]["margin_A"] <= 0)]
    flips = {b["basin_index"]: b["symmetry"].get("sigma_flip_tolerance_A")
             for b in basins}
    add(9, "sigma is a group order and is stable up to the working tolerance",
        not bad_sigma,
        "basins needing a look: {}; sigma flips at (A): {}".format(
            bad_sigma or "none", flips))

    # 10 (new) -- the published protocol actually ran, or the fallback is declared.
    add(10, "published dynamics protocol, or a declared fallback",
        c.get("n_terminated_early", 0) == 0,
        "terminated EARLY {}; shake used {}; fell back {}".format(
            c.get("n_terminated_early"), c.get("shake_used"),
            c.get("used_shake_fallback")))

    # 12 (new, 2026-09-04) -- the deduplication did not merge anything the energy
    # criterion should have stopped, and the criterion in force is recorded.
    #
    # This is written against the PRODUCT, not against the config: `merge_energy_warnings`
    # lists merges that actually happened across the energy threshold. Under the
    # three-fold criterion that list must be empty, and if it is not, the RMSD and energy
    # thresholds have been set inconsistently with each other.
    warns = rec.get("merge_energy_warnings") or []
    add(12, "no merge crossed the energy threshold; criterion recorded",
        not warns and rec.get("dedup_criterion") is not None,
        "criterion {} (RMSD < {} Å, |ΔE| < {} kcal/mol, ΔB < {}); "
        "merges blocked by energy {}, by rotational constants {}; "
        "warnings {}".format(
            rec.get("dedup_criterion"), rec.get("dedup_threshold_A"),
            rec.get("dedup_ethr_kcal"), rec.get("dedup_bthr_relative"),
            rec.get("n_merges_blocked_by_energy"),
            rec.get("n_merges_blocked_by_rotational"),
            warns or "none"))

    return checks


# ======================================================================================
def run_species(qid, cfg, args, calc, prov, smiles=None, label=None):
    """One molecule, end to end.

    Identified EITHER by a QM9 index (`qid`) or by SMILES. The two differ in exactly
    two places, both of which are recorded rather than smoothed over: where the
    starting geometry comes from, and whether an independent reference geometry
    exists to pool in at step 3.
    """
    t_start = time.time()
    gate_rec = gate(qid, cfg, smiles=smiles)
    if not gate_rec["passed"]:
        return dict(qm9_index=qid, gate=gate_rec, stopped_at="gates",
                    note="molecule rejected by the filters; nothing downstream ran")

    name = label or qid
    # THE MOLECULE DIRECTORY (ADR 0001, 2026-09-14): one per (tag, molecule) under the
    # root, one folder per engine inside it, this repository's records in _records/.
    # CREST's finished directory is <molecule>/crest/ (it RUNS node-local; see run_crest).
    from openqha.store import layout
    molecule = layout.molecule_dir(config.runs_root(cfg), args.tag, qid or name)
    workdir = Path(args.reuse) if args.reuse else layout.crest_dir(molecule)

    seed_rec = None
    start_xyz = None
    if qid is None:
        workdir.mkdir(parents=True, exist_ok=True)
        start_xyz = workdir / "{}_seed.xyz".format(name)
        if not start_xyz.exists():
            seed_rec = seed_geometry_from_smiles(
                gate_rec["smiles"], calc, start_xyz)
        else:
            # Reused from disk. It must carry the SAME keys as a freshly built one, or
            # every consumer has to branch on which kind it got -- and the notebook did
            # not, which is how this was found. The xyz comment line is where the
            # geometry recorded its own provenance, so it is read back rather than
            # invented.
            head = start_xyz.read_text(encoding="utf-8").splitlines()
            seed_rec = dict(
                smiles=gate_rec["smiles"], path=str(start_xyz), reused=True,
                n_embeddings=1,
                seed_comment=(head[1] if len(head) > 1 else ""),
                relaxed_energy_eV=None, residual_force_eV_A=None,
                converged=None, opt_steps=None,
                role=("STARTING STRUCTURE ONLY (reused from disk; the energy and "
                      "force columns belong to the run that built it, and are left "
                      "null here rather than copied from a different invocation)"))

    if args.skip_crest:
        crest_rec = crest.record_from_dir(workdir)
        crest_rec["reused_scratch"] = True
        # Same treatment as the scratch-reuse branch inside run_crest: CREST's own
        # printed wall time is carried across so the number is not lost, and flagged as
        # NOT a cost of this invocation. Missing this here made criterion 2 fail on the
        # --skip-crest path -- correctly: the record was leaving a reader to guess
        # whether there was a cost measurement at all.
        crest_rec["wall_is_valid_cost"] = False
        crest_rec["wall_seconds"] = crest_rec.get("seconds")
        # READ what the directory actually ran, do not assert what the config asks for.
        # Pointing --reuse at a `_shake1` fallback directory previously produced a
        # record claiming shake_used = 2 for a directory whose input.toml says 1.
        # Acceptance criterion 11 caught the inconsistency, but the record itself was
        # wrong -- and a record that states the configuration instead of the event is
        # the same defect as 57, one level up.
        on_disk = crest.read_input_settings(Path(crest_rec["workdir"]) / "input.toml")
        used = (on_disk or {}).get("shake")
        crest_rec["shake_used"] = int(float(used)) if used is not None else None
        crest_rec["used_shake_fallback"] = bool(
            str(workdir).endswith("_shake{}".format(
                (cfg["crest"].get("shake_fallback") or {}).get("to", 1))))
        crest_rec["shake_source"] = "read back from input.toml, not from the config"
    else:
        crest_rec = run_crest(qid or name, workdir, cfg, args, start_xyz=start_xyz)

    ens = Path(crest_rec["workdir"]) / "crest_conformers.xyz"
    if not ens.exists():
        raise FileNotFoundError(
            "CREST produced no ensemble at {}. Its own report: terminated_normally="
            "{}, terminated EARLY={}".format(ens, crest_rec.get("terminated_normally"),
                                             crest_rec.get("n_terminated_early")))
    # ---- stop here rather than build basins out of numbers CREST could not rank -------
    # Both of these were survivable-looking on 2026-09-09 and neither was survivable:
    # CREST said `terminated normally`, and the ensemble it handed over had 1814 frames
    # with NaN for every energy. Everything after this line would have run happily on it.
    if crest_rec.get("energy_defect"):
        return dict(qm9_index=qid, gate=gate_rec, crest=crest_rec,
                    stopped_at="crest_energies", note=crest_rec["energy_defect"])
    ceiling = getattr(args, "max_conformers", None)
    if ceiling and crest_rec.get("n_conformers", 0) > int(ceiling):
        return dict(
            qm9_index=qid, gate=gate_rec, crest=crest_rec,
            stopped_at="conformer_count",
            note="CREST reports {} conformers and this run's ceiling is {}. The energies "
                 "parsed as finite, so this is not the NaN defect -- either the ceiling "
                 "is wrong for this molecule (it is set per example, and it is a "
                 "judgement, not a measurement) or the deduplication is not "
                 "deduplicating.".format(crest_rec.get("n_conformers"), ceiling))

    frames, comments = crest_census.read_ensemble_atoms(ens)

    # Step 3 pools an INDEPENDENT starting geometry. Only a shipped species has one.
    reference_xyz = config.qm9_xyz(qid, cfg) if qid else None
    census, basins, mol = basin_list(
        qid or name, gate_rec["smiles"], frames, comments, calc, cfg, args,
        reference_xyz=reference_xyz, molecule_dir=molecule)
    if reference_xyz is None:
        census["reference_geometry_note"] = (
            "No deposited geometry exists for a SMILES-specified molecule, so step 3 "
            "pooled nothing and every structure below descends from CREST. The "
            "systematic error plan_A section 1.2 records for a CREST-only ensemble "
            "(mean +0.1209, max +0.5824 kcal/mol) is therefore UNBOUNDED here.")
    labelled = label_basins(qid, basins, census, cfg, args)

    record = dict(
        generated_by="scripts/production/s0_A_pipeline.py",
        branch="A", qm9_index=qid, label=name,
        name=(cfg["species"].get(qid) or {}).get("name") if qid else label,
        identified_by="qm9_index" if qid else "smiles",
        seed_geometry=seed_rec,
        smiles=gate_rec["smiles"],
        composite_notation=engine.composite_notation(),
        engine=prov,
        crest_version=crest.crest_version(),
        settings=dict(
            workhorse=cfg["crest"]["workhorse"], refine=cfg["crest"]["refine"],
            runtype=cfg["crest"]["runtype"], optlev=cfg["crest"]["optlev"],
            shake=cfg["crest"]["shake"], tstep_fs=cfg["crest"]["tstep_fs"],
            hydrogen_mass_amu=cfg["crest"].get("hydrogen_mass_amu"),
            threads=int(args.threads),
            fmax_eV_A=float(cfg["package2"]["fmax_hessian_eV_A"]),
            dedup_rmsd_A=float(cfg["package1"]["dedup_rmsd_A"]),
            dedup_criterion=cfg["package1"].get("dedup_criterion", "cregen_three_fold"),
            dedup_ethr_kcal=cfg["package1"].get("dedup_ethr_kcal"),
            dedup_bthr_relative=cfg["package1"].get("dedup_bthr_relative"),
            hessian_mode=args.hessian_mode,
            temperature_K=config.temperature(cfg),
            symmetry_tolerance_A=symmetry.TOLERANCE_A),
        protocol_source=("SHAKE all bonds + 5 fs + hydrogen mass 2 amu: Grimme, "
                         "JCTC 2019, 15, 2847. iMTD-GC: Pracht, Bohle & Grimme, "
                         "PCCP 2020, 22, 7169. BRANCH A ONLY -- see the module "
                         "docstring."),
        gate=gate_rec, crest=crest_rec, census=census, basins=labelled,
        machine=machine_load(),
        wall_seconds_total=time.time() - t_start)
    record["criteria"] = check_criteria(record, cfg["crest"].get("shake_fallback"))
    record["all_criteria_passed"] = all(c["passed"] for c in record["criteria"])

    # ---- where the record goes (ADR 0001, 2026-09-14) --------------------------------
    # The engine files are already in place: mace/confNN and mace/basinNN under the
    # molecule directory, written by the census. This record -- everything this
    # repository has to say about the run -- goes to _records/ beside them, with the
    # basin geometries as one multi-frame xyz for a reader who wants them in one file.
    # Until 2026-09-14 the same record went to analysis/branchA/<tag>/<name>/ AND, byte
    # for byte, into the sharded store data/basins/<tag>/<range>/<chunk>/; branch B now
    # reads the basins from mace/basinNN/basin.extxyz and the store is retired.
    outdir = layout.records_dir(molecule)
    outdir.mkdir(parents=True, exist_ok=True)
    record["molecule_dir"] = str(molecule)
    record["mace"] = dict(
        folder=str(molecule / "mace"),
        basins=[str(layout.mace_basin_dir(molecule, i) / "basin.extxyz")
                for i in range(len(basins))],
        hessians=[str(layout.mace_basin_dir(molecule, i) / "hessian.npy")
                  for i in range(len(basins))])
    record["output_dir"] = str(outdir)
    record["tag"] = args.tag
    # The Record of this Calculation (records redesign, user ruling 2026-09-15):
    # branchA.out is the Report a person reads, everything above expanded, last line the
    # terminal line; branchA.toml is the Property file a program reads, ORCA's
    # .property.txt shape (status, inputs, the result blocks a later step reads) and
    # nothing else. The Report first, the Property file last: STATUS = NORMAL TERMINATION
    # is the completion marker, so it must be the last thing written.
    from openqha.store import branch_a_property
    write_branch_a_report(record, outdir / branch_a_property.REPORT)
    unknown = branch_a_property.write(outdir / branch_a_property.FILE, record)
    if unknown:
        raise RuntimeError("branchA.toml: keys outside the schema {}; add them to "
                           "openqha.store.branch_a_property.SCHEMA".format(unknown))
    return record


def write_branch_a_report(record, path):
    """`branchA.out`: what a person reads. Sections in the order the work happened, the
    basins table, the criteria, then the complete record expanded, and the terminal line."""
    from openqha.store import report as _report
    r = _report.Report("openQHA branch A -- conformer search, basins, Hessians",
                       subtitle="{}  ({})   tag {}".format(
                           record.get("qm9_index") or record.get("label"), record.get("name"),
                           record.get("tag")))
    # Provenance is the Report's (records redesign, 2026-09-15): the Property file holds
    # only what a later step reads, so the engine, the weights, the versions, the machine
    # and the neighbour-list probe are printed here, once, as ORCA prints its version.
    eng = record.get("engine") or {}
    r.section("Provenance")
    r.kv("engine", eng.get("engine"))
    r.kv("composite_notation", record.get("composite_notation"))
    r.kv("weights_path", eng.get("weights_path"))
    r.kv("weights_bytes", eng.get("bytes"))
    r.kv("interface", eng.get("interface"))
    r.kv("mace_torch_version", eng.get("mace_torch_version"))
    r.kv("torch_version", eng.get("torch_version"))
    r.kv("dtype", eng.get("dtype"))
    patch = eng.get("neighbour_list_patch") or {}
    r.kv("neighbour_list_patch_applied", patch.get("applied"))
    r.kv("installed_mace_defect_present", (patch.get("installed") or {}).get("defect_present"))
    cvers = record.get("crest_version") or {}
    r.kv("crest_version", "{} ({})".format(cvers.get("version"), cvers.get("commit")))
    r.kv("crest_binary", (record.get("crest") or {}).get("binary"))
    r.kv("protocol_source", record.get("protocol_source"))
    for k, v in (record.get("machine") or {}).items():
        r.kv("machine_" + k, v)
    r.kv("generated_by", record.get("generated_by"))
    r.section("Settings")
    for k, v in (record.get("settings") or {}).items():
        r.kv(k, v)
    g = record.get("gate") or {}
    r.section("Gate")
    for k in ("smiles", "smiles_source", "gates_enabled", "passed", "reason"):
        if k in g:
            r.kv(k, g[k])
    c = record.get("crest") or {}
    r.section("CREST")
    for k in ("workdir", "ran_in", "returncode", "n_conformers", "terminated_normally",
              "n_terminated_early", "n_completed_successfully", "total_engrad_calls",
              "energy_spread", "shake_used", "used_shake_fallback", "wall_seconds",
              "wall_is_valid_cost", "reused_scratch"):
        if k in c:
            r.kv(k, c[k])
    cv = record.get("census") or {}
    r.section("Census: tighten, deduplicate, Hessian")
    for k in ("n_frames_from_crest", "n_reference_geometries_pooled", "n_input_frames_total",
              "n_not_converged", "n_graph_changed", "max_residual_force_eV_A",
              "n_basins_by_repo_criteria", "n_saddle_points_rejected"):
        if k in cv:
            r.kv(k, cv[k])
    r.section("Basins")
    rows = []
    for b in record.get("basins") or []:
        s = b.get("symmetry") or {}
        rows.append([b.get("basin_index"), "{:.6f}".format(b["energy_eV"]),
                     "{:.4f}".format(b["relative_kcal"]), s.get("sigma"),
                     s.get("pymsym_point_group"), "{:.2f}".format(b.get("lowest_frequency_cm_inv", float("nan"))),
                     "{:.4f}".format((b.get("thermo") or {}).get("G_minus_Eel_kcal", float("nan")))])
    r.table(["basin", "E / eV", "rel / kcal", "sigma", "pg", "nu_min / cm-1", "G - E_el / kcal"], rows)
    # Per basin: the sigma diagnostics and the tolerance sweep (criterion 9's evidence),
    # then the thermochemistry breakdown. These left the Property file on 2026-09-15.
    for b in record.get("basins") or []:
        s = b.get("symmetry") or {}
        r.section("Basin {}: symmetry".format(b.get("basin_index")))
        for k, v in s.items():
            if k == "tolerance_sweep":
                continue
            r.kv(k, v)
        sweep = s.get("tolerance_sweep") or {}
        if sweep:
            r.table(["tolerance / A", "sigma", "n_improper"],
                    [[t, (row or {}).get("sigma"), (row or {}).get("n_improper")]
                     for t, row in sweep.items()], title="sigma over the tolerance sweep")
        th = b.get("thermo") or {}
        r.section("Basin {}: thermochemistry".format(b.get("basin_index")))
        for k, v in th.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    r.kv("{}_{}".format(k, kk), vv)
            else:
                r.kv(k, v)
    r.section("Acceptance criteria")
    for crit in record.get("criteria") or []:
        r.verdict("{} {}".format(crit.get("number", ""), crit.get("criterion", "")),
                  crit.get("detail"), bool(crit.get("passed")))
    r.kv("all_criteria_passed", record.get("all_criteria_passed"))
    r.kv("wall_seconds_total", record.get("wall_seconds_total"))
    r.json_dump(record, title="complete record (expanded)")
    return r.write(path, step="branch A")


def _basins_xyz_text(basins, labelled, qid, record):
    """One multi-frame xyz as text; the comment line carries what a reader needs."""
    lines = []
    for i, atoms in enumerate(basins):
        lab = labelled[i]
        lines.append(str(len(atoms)))
        lines.append(
            "{} basin {} E={:.8f} eV rel={:.4f} kcal/mol sigma={} pg={} "
            "nu_min={:.2f} cm-1 engine={} hessian={}".format(
                qid, i, lab["energy_eV"], lab["relative_kcal"],
                lab["symmetry"]["sigma"], lab["symmetry"]["pymsym_point_group"],
                lab["lowest_frequency_cm_inv"], record["engine"]["engine"],
                record["settings"]["hessian_mode"]))
        for s, p in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
            lines.append("{:<2s} {:18.10f} {:18.10f} {:18.10f}".format(s, *p))
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species", default=None,
                    help="QM9 index, e.g. dsgdb9nsd_000018")
    ap.add_argument("--smiles", default=None,
                    help="SMILES, for a molecule this repo does not ship. The "
                         "starting structure is then built here; CREST still "
                         "produces every conformer.")
    ap.add_argument("--label", default=None,
                    help="output directory name when --smiles is used")
    ap.add_argument("--tag", default="prod")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--timeout-s", type=int, default=14400)
    ap.add_argument("--hessian-mode", default=crest_census.HESSIAN_MODE_DEFAULT,
                    choices=("analytic", "finite_difference"))
    ap.add_argument("--skip-crest", action="store_true",
                    help="use an ensemble already on disk (see --reuse)")
    ap.add_argument("--reuse", default=None, help="an existing CREST work directory")
    ap.add_argument("--force-crest", action="store_true",
                    help="rerun CREST even if a matching scratch directory exists")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-conformers", type=int, default=None,
                    help="stop if CREST reports more conformers than this. **There is "
                         "no default**: the right ceiling is a property of the molecule, "
                         "not of the pipeline, so it is set per example (see "
                         "examples/02a_qha_openmm_acetone/branchA.conf) and never "
                         "guessed here. Acetone at 10 atoms cannot have hundreds; the "
                         "2026-09-09 Tianhe run reported 1814 for propanal because every "
                         "energy was NaN and CREGEN discarded nothing.")
    ap.add_argument("--keep-calcspace", default=None, metavar="DIR",
                    help="run CREST's external gradient calls in DIR and keep it, "
                         "instead of a scratch directory CREST deletes. DIAGNOSTIC: it "
                         "keeps files for every gradient call. Without it, a failed "
                         "gradient leaves no evidence at all -- which is why job 7347197 "
                         "could not be explained from its own output.")
    args = ap.parse_args()

    if not (args.species or args.smiles):
        raise SystemExit("give --species <qm9 index> or --smiles <SMILES> --label <name>")
    if args.species and args.smiles:
        raise SystemExit("--species and --smiles name the molecule two different "
                         "ways; give one")
    label = args.label or (args.species if args.species else "molecule")

    cfg = config.load()
    calc, name, prov = engine.calculator(device=args.device)

    print("=" * 92)
    print("Branch A pipeline -- {}".format(args.species or
                                           "{} ({})".format(label, args.smiles)))
    print("=" * 92)
    print("engine      {}   weights {}".format(name, prov["weights_path"]))
    print("notation    {}".format(engine.composite_notation()))
    print("workhorse   {}   refine {}   shake {}   tstep {} fs".format(
        cfg["crest"]["workhorse"], cfg["crest"]["refine"],
        cfg["crest"]["shake"], cfg["crest"]["tstep_fs"]))
    print("hessian     {}   sigma tolerance {} A".format(
        args.hessian_mode, symmetry.TOLERANCE_A))
    print()

    # CREST's quality layer reaches MACE through a Unix socket served by a resident
    # process (openqha/potentials/mace_server.py). One molecule needs exactly one server, so
    # this script starts its own unless a socket is already exported -- branch E
    # exports one per parallel slot, because a single server holds one model and one
    # lock and would serialise the whole fan-out.
    servers = []
    need_socket = (not args.skip_crest
                   and cfg["crest"]["backend"] == "generic"
                   and cfg["crest"]["refine"] not in crest.NO_QUALITY_LEVEL)
    if need_socket and not os.environ.get("S0_MACE_SOCKET"):
        servers = crest.start_servers(1)
        os.environ["S0_MACE_SOCKET"] = servers[0][1]
        print("started MACE server on {}".format(servers[0][1]))
        print()
    try:
        record = run_species(args.species, cfg, args, calc, prov,
                         smiles=args.smiles, label=label)
    finally:
        if servers:
            crest.stop_servers(servers)
    if record.get("stopped_at"):
        print("stopped at {}: {}".format(record["stopped_at"], record["note"]))
        return 1

    rec = record["census"]
    cv = rec["crest_vs_repo"]
    print("CREST reports {} conformers; + {} reference geometry = {} frames in "
          "-> {} basins by this repo's criteria".format(
              cv["n_conformers_reported_by_crest"],
              cv["n_reference_geometries_pooled"],
              cv["n_input_frames_total"], cv["n_basins_by_repo_criteria"]))
    print()
    print("{:>5} {:>16} {:>12} {:>6} {:>7} {:>12} {:>12}".format(
        "basin", "E / eV", "rel/kcal", "sigma", "pg", "nu_min/cm-1", "G-Eel/kcal"))
    for b in record["basins"]:
        print("{:>5} {:>16.6f} {:>12.4f} {:>6} {:>7} {:>12.2f} {:>12.4f}".format(
            b["basin_index"], b["energy_eV"], b["relative_kcal"],
            b["symmetry"]["sigma"], str(b["symmetry"]["pymsym_point_group"]),
            b["lowest_frequency_cm_inv"], b["thermo"]["G_minus_Eel_kcal"]))
    print()
    print("acceptance criteria")
    for c in record["criteria"]:
        print("  {:>2}  {:<62} {}".format(
            c["number"], c["criterion"][:62], "PASS" if c["passed"] else "FAIL"))
        if not c["passed"]:
            print("      {}".format(c["detail"]))
    print()
    print("wall {:.1f} s   all criteria passed: {}".format(
        record["wall_seconds_total"], record["all_criteria_passed"]))
    print("written {}/branchA.out and branchA.toml".format(record["output_dir"]))
    return 0 if record["all_criteria_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
