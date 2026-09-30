"""Branch B analysis: trajectories -> quasi-harmonic entropy, with every criterion tested.

PRODUCTION. It reads what `scripts/production/s0_B_qha_trajectory.py` wrote and produces the branch B
deliverable: `S_QH`, `S_Schlitter` and the assembled `(G - E_el)` for each basin, together
with the diagnostics that decide whether those numbers mean anything.

The diagnostics are the deliverable too
---------------------------------------
Quasi-harmonic entropy rises monotonically with sampling time and saturates slowly, so a
trajectory that has not run long enough returns a number that looks perfectly stable and
is too low. The density-of-states route this branch replaced was killed by exactly one
missing experiment -- a blank control -- and the lesson recorded from that was
that without a blank control you can only guess. So this script always runs:

  * the SATURATION CURVE on leading prefixes, and reports the increment over the last
    doubling (criterion 1, budget 0.3 kcal/mol on T*S);
  * the BLANK CONTROL: the same potential, the same basin, different seeds. Their spread
    is the noise floor. If an edge-level signal is smaller than that, branch B is
    reported as unable to measure it, in those words (criterion 5);
  * the PER-BATCH convergence of the modes, which separates "the values have not settled"
    from "the modes have not all appeared" (criterion 10);
  * the RANK check, 3N-6 against the number of non-zero eigenvalues (criterion 9);
  * the RIGID-MODE check and its separation ratio (criterion 4);
  * `S_QH <= S_Schlitter`, which is analytic and therefore catches implementation
    errors only (criterion 6);
  * the GROMACS cross-check (criterion 2, now an EXTENSION -- see
    openqha/extensions/gromacs.py and openqha/capabilities.py);
  * the trajectory identity assertion, which refuses biased, constrained or
    mass-repartitioned input (criterion 11).

    python scripts/production/s0_B_qha_analyse.py --species dsgdb9nsd_000018 --tag prod
    python scripts/production/s0_B_qha_analyse.py --species X --tag smoke --no-gmx
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _repo_root():
    """Directory holding the openQHA package, found by walking up from this file."""
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


sys.path.insert(0, str(_repo_root()))

from openqha import capabilities, config, mdtraj_io, qha, report, thermo  # noqa: E402
from openqha.extensions import gromacs as gmx_io                    # noqa: E402

#: Criterion 1's budget on T*S, in kcal/mol. It is 30 percent of the 1.0 kcal/mol target
#: accuracy: the saturation increment is the main term of the error bar, and a term that
#: is allowed to eat the whole budget leaves nothing for the rest of the chain.
SATURATION_BUDGET_KCAL = 0.3

#: Criterion 2's budget on T*S, in kcal/mol, against the independent GROMACS spectrum.
GMX_BUDGET_KCAL = 0.05

#: Below this production length, criteria 1 and 5 are reported as SMOKE rather than
#: passed. Not a scientific threshold -- it is the line under which a pass is meaningless,
#: because quasi-harmonic entropy rises monotonically and a trajectory that has barely
#: started has barely started rising. The real length is set from the
#: saturation curve, and 20 ps is simply far short of any answer that curve could give.
SMOKE_LENGTH_PS = 20.0


def discover(molecule, setting, route="auto"):
    """Every trajectory of one setting: (basin label, seed label, engine_dir, records_dir).

    The labels are directory names, as before (`basin00`, and `seed00` for the one
    trajectory per basin), because collect's tables and the
    ensemble report key on them. The trajectory itself is `traj.dcd`.
    """
    from openqha.quasi_harmonic import trajectory_reader
    return [("basin{:02d}".format(b), "seed00", eng, rec)
            for b, eng, rec in trajectory_reader.trajectory_dirs(molecule, setting, route)]


def analyse_one(engine_dir, records_dir, temperature_K, fractions, n_batches,
                limit_frames=None, setting="default"):
    """One trajectory: the identity assertion first, then the whole chain.

    Frames come from the engine folder (traj.dcd + start.pdb); the record beside them
    carries what the identity assertion needs (thermostat, timestep, hydrogen mass, the
    masses the driver integrated with). A trajectory without its record is refused: the
    assertion cannot be made from a DCD alone.
    """
    from openqha.quasi_harmonic import trajectory_reader
    tr = trajectory_reader.read_trajectory(engine_dir, records_dir=records_dir, setting=setting)
    meta = tr["meta"]
    if meta is None:
        raise SystemExit(
            "no meta.json in {}: the trajectory in {} has no record, and the identity "
            "assertion (thermostat, timestep, hydrogen mass) cannot be made from the DCD "
            "alone".format(records_dir, engine_dir))
    frames = tr["positions_A"]
    if limit_frames:
        frames = frames[:int(limit_frames)]
    masses = np.asarray(meta.get("masses_amu") or tr["masses_amu"], dtype=float)

    rec = qha.analyse(frames, masses, temperature_K, meta=meta)
    sat = qha.saturation_curve(frames, masses, temperature_K, fractions=fractions)
    batches = qha.mode_batch_convergence(frames, masses, temperature_K,
                                         n_batches=n_batches,
                                         fractions=tuple(f for f in fractions if f >= 0.25))
    # `path` is the RECORDS folder: what else this step writes about the trajectory
    # (the GROMACS cross-check's working files) goes beside the record, never into the
    # engine folder.
    return dict(path=str(records_dir), engine_dir=str(engine_dir), meta=meta,
                frames=frames, masses=masses, analysis=rec, saturation=sat,
                mode_batches=batches)


def assembly_consistency(masses, positions, eigenvalues, symmetry_number, degeneracy,
                         temperature_K):
    """Criterion 7, as a check that can fail rather than a claim in a docstring.

    The translational, rotational and electronic terms must be bit-for-bit identical
    between this branch and the Hessian route, because the whole point of the assembly is
    that the difference between the two routes is the VIBRATIONAL term and nothing else.
    `qha.g_minus_eel` calls `thermo.g_minus_eel`, so the guarantee is structural -- and a
    structural guarantee that nobody ever tests is a comment. Here the same three terms
    are recomputed from a deliberately DIFFERENT vibrational input and compared: they must
    not move.
    """
    quasi = qha.g_minus_eel(masses, positions, eigenvalues, symmetry_number, degeneracy,
                            temperature_K=temperature_K)
    other = thermo.g_minus_eel(masses, positions, [500.0] * (3 * len(masses) - 6),
                               symmetry_number, degeneracy, temperature_K=temperature_K)
    same = {}
    for term in ("rotational", "translational", "electronic"):
        a, b = quasi[term], other[term]
        keys = [k for k in a if isinstance(a[k], float)]
        same[term] = all(a[k] == b[k] for k in keys)
    return quasi, dict(identical_terms=same, all_identical=all(same.values()),
                       compared_against="thermo.g_minus_eel with a flat 500 cm^-1 spectrum",
                       symmetry_number=int(symmetry_number),
                       electronic_degeneracy=int(degeneracy))


def table_sections(species, per_traj, blank, assembly):
    """The rows of collect's Table, `{section: rows}` for `trajectories`, `blank`,
    `assembly` -- the columns `chain_records.COLUMNS` explains, in that order. The writer
    reports a column added here without its explanation, and `main` refuses to go on
    (the rule `md_record` applies to `md.toml` keys): the Table is on disk but no Property
    file, so the collect Batch redoes the molecule once COLUMNS is fixed."""
    return dict(
        trajectories=[dict(
            species=species, basin=t["basin"], seed=t["seed"],
            n_frames=t["analysis"]["rank_check"]["n_frames"],
            TS_QH_kcal=t["analysis"]["entropy"]["TS_QH_kcal"],
            TS_Schlitter_kcal=t["analysis"]["entropy"]["TS_Schlitter_kcal"],
            S_QH_kcal_per_K=t["analysis"]["entropy"]["S_QH_kcal_per_K"],
            A_vib_kcal=t["analysis"]["entropy"]["A_vib_kcal"],
            lowest_frequency_cm_inv=t["analysis"]["entropy"]["lowest_frequency_cm_inv"],
            highest_frequency_cm_inv=t["analysis"]["entropy"]["highest_frequency_cm_inv"],
            n_nonzero=t["analysis"]["spectrum"]["n_nonzero_eigenvalues"],
            expected_modes=t["analysis"]["spectrum"]["expected_vibrational_modes"],
            rigid_ratio=t["analysis"]["spectrum"]["rigid_to_first_vibrational_ratio"],
            saturation_last_doubling_kcal=t["saturation"][
                "increment_over_last_doubling_kcal"],
            wall_seconds=t["meta"]["production"].get("wall_seconds"),
            seconds_per_ps=t["meta"]["production"].get("seconds_per_ps_this_run"),
        ) for t in per_traj],
        blank=[dict(species=species, **{k: v for k, v in b.items() if k != "values_kcal"})
               for b in blank],
        assembly=[dict(species=species, basin=a["basin"],
                       G_minus_Eel_kcal=a["G_minus_Eel_kcal"],
                       A_vib_kcal=a["A_vib_kcal"],
                       S_vib_kcal_per_K=a["S_vib_kcal_per_K"],
                       terms_match_hessian_route=a["consistency"]["all_identical"])
                  for a in assembly],
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species", required=True)
    ap.add_argument("--tag", default="prod")
    ap.add_argument("--basin-tag", default=None,
                    help="the tag the molecule directory is under (branch A's; default --tag)")
    ap.add_argument("--setting", default="default",
                    help="which setting's trajectories to analyse")
    ap.add_argument("--route", default="auto", choices=("auto", "openmm", "ase"),
                    help="which engine's trajectories to read: md_openmm/basinNN (traj.dcd) "
                         "or md_ase/basinNN (md.traj); auto takes openmm when present, else ase")

    ap.add_argument("--molecule-dir", default=None,
                    help="override the molecule directory (default: from S0_RUNS_ROOT)")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--fractions", default="0.05,0.1,0.25,0.5,1.0")
    ap.add_argument("--batches", type=int, default=5)
    ap.add_argument("--limit-frames", type=int, default=None)
    ap.add_argument("--no-gmx", action="store_true",
                    help="skip the GROMACS cross-check (criterion 2 then has no "
                         "independent implementation behind it, and the report says so)")
    ap.add_argument("--gmx-all", action="store_true",
                    help="cross-check every trajectory, not just the first")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # (Until 2026-09-15 a parquet-engine preflight stood here: the tables were parquet and
    # pyarrow's absence on an113 threw a finished analysis away on its last line. The
    # tables are whitespace .dat now and need nothing beyond the standard library.)

    cfg = config.load()
    temperature = args.temperature or config.temperature(cfg)
    spec = config.species(args.species, cfg)      # raises if sigma or g is undeclared
    fractions = tuple(float(x) for x in args.fractions.split(","))

    from openqha.store import layout
    molecule = (Path(args.molecule_dir) if args.molecule_dir
                else layout.molecule_dir(config.runs_root(cfg), args.basin_tag or args.tag,
                                         args.species))
    from openqha.quasi_harmonic import trajectory_reader as _tr
    found = discover(molecule, args.setting, args.route)
    if not found:
        raise SystemExit("no trajectories under {} for route {!r}, setting {!r} (no "
                         "md_openmm/basinNN/traj.dcd or md_ase/basinNN/md.traj)\n"
                         "Run a branch B trajectory driver first."
                         .format(molecule, args.route, args.setting))
    route = _tr.route_of(found[0][2])       # (basin label, seed label, engine_dir, records_dir)
    root = molecule / layout.md_folder(route)

    print("=" * 92)
    print("Branch B -- quasi-harmonic analysis   {}  ({})".format(
        args.species, spec.get("name")))
    print("=" * 92)
    print("trajectories   {} under {}".format(len(found), root))
    print("temperature    {} K     symmetry number {} ({})".format(
        temperature, spec["symmetry_number"], spec.get("symmetry_reason")))
    print()

    per_traj, by_basin = [], {}
    for basin, seed, engine_dir, records_dir in found:
        one = analyse_one(engine_dir, records_dir, temperature, fractions, args.batches,
                          args.limit_frames, setting=args.setting)
        ent = one["analysis"]["entropy"]
        spec_rec = one["analysis"]["spectrum"]
        print("{:<9} {:<8} {:>7} frames   T*S_QH = {:9.4f}   T*S_Schl = {:9.4f}   "
              "modes {}/{}   last-doubling {:+.4f}".format(
                  basin, seed, one["analysis"]["rank_check"]["n_frames"],
                  ent["TS_QH_kcal"], ent["TS_Schlitter_kcal"],
                  spec_rec["n_nonzero_eigenvalues"],
                  spec_rec["expected_vibrational_modes"],
                  one["saturation"]["increment_over_last_doubling_kcal"]))
        one["basin"], one["seed"] = basin, seed
        per_traj.append(one)
        by_basin.setdefault(basin, []).append(one)

    # ---- the blank control: same potential, same basin, different seed ---------------
    blank = []
    for basin, group in sorted(by_basin.items()):
        ts = np.array([g["analysis"]["entropy"]["TS_QH_kcal"] for g in group])
        blank.append(dict(
            basin=basin, n_seeds=int(len(ts)),
            TS_QH_mean_kcal=float(ts.mean()),
            TS_QH_spread_kcal=float(ts.max() - ts.min()) if len(ts) > 1 else 0.0,
            TS_QH_rms_about_mean_kcal=(float(np.sqrt(((ts - ts.mean()) ** 2).mean()))
                                       if len(ts) > 1 else 0.0),
            standard_error_kcal=(float(ts.std(ddof=1) / np.sqrt(len(ts)))
                                 if len(ts) > 1 else None),
            values_kcal=[float(x) for x in ts],
            note=("one seed per basin: no noise floor is measured "
                  "here; --seeds 2 or more would give one" if len(ts) < 2 else None)))

    # ---- the assembly, per basin, on the seed-averaged spectrum ----------------------
    assembly = []
    for basin, group in sorted(by_basin.items()):
        best = max(group, key=lambda g: g["analysis"]["rank_check"]["n_frames"])
        lam = np.asarray(best["analysis"]["spectrum"]["eigenvalues_amu_A2"], dtype=float)
        mean_structure = np.asarray(best["analysis"]["mean_structure_A"], dtype=float)
        g_rec, consistency = assembly_consistency(
            best["masses"], mean_structure, lam, spec["symmetry_number"],
            spec["electronic_degeneracy"], temperature)
        assembly.append(dict(basin=basin, from_trajectory=best["path"],
                             G_minus_Eel_kcal=g_rec["G_minus_Eel_kcal"],
                             A_vib_kcal=g_rec["vibrational"]["A_vib_kcal"],
                             S_vib_kcal_per_K=g_rec["vibrational"]["S_vib_kcal_per_K"],
                             rotational_kcal=g_rec["rotational"]["A_rot_kcal"],
                             translational_kcal=g_rec["translational"]["value_kcal"],
                             electronic_kcal=g_rec["electronic"]["A_elec_kcal"],
                             consistency=consistency, full=g_rec))

    # ---- criterion 2 -----------------------------------------------------------------
    cross = []
    if not args.no_gmx:
        targets = per_traj if args.gmx_all else per_traj[:1]
        for one in targets:
            work = Path(one["path"]) / "gmx"
            print("\n-- GROMACS cross-check on {} {}".format(one["basin"], one["seed"]))
            try:
                c = gmx_io.cross_check(one["frames"], one["meta"]["symbols"],
                                       one["masses"], work, temperature_K=temperature)
                c["basin"], c["seed"] = one["basin"], one["seed"]
                c.pop("ours", None)
                cross.append(c)
                print("   gmx {}  precision {}".format(
                    c["gmx_version"].get("version"), c["gmx_version"].get("precision")))
                print("   eigenvalue max relative difference {:.3e}   "
                      "T*S difference {:+.6f} kcal/mol".format(
                          c["eigenvalue_max_relative_difference_vs_unprojected"],
                          c["TS_QH_difference_kcal"]))
            except Exception as exc:                    # record it; do not lose the run
                import traceback
                cross.append(dict(basin=one["basin"], seed=one["seed"],
                                  error_type=type(exc).__name__, error=str(exc),
                                  traceback=traceback.format_exc()))
                print("   FAILED: {}: {}".format(type(exc).__name__, exc))

    # ---- criterion 2, second independent implementation -------------------------------
    # MDAnalysis, fitted MASS-WEIGHTED and ITERATED to the mean, which is our protocol
    # exactly. What remains between the two is then the implementation alone -- our
    # Kabsch/SVD against its Theobald QCP quaternion -- and that is the comparison worth
    # making. Measured on a real 25 ps trajectory: 8.0e-09 kcal/mol.
    #
    # Run unconditionally where the library is present, because this one needs no
    # external binary and no PATH.
    superposition = None
    try:
        one = per_traj[0]
        superposition = mdtraj_io.cross_check(
            one["frames"], one["masses"], one["meta"]["symbols"],
            temperature_K=temperature)
        superposition["basin"], superposition["seed"] = one["basin"], one["seed"]
        print("\n-- superposition cross-check on {} {}".format(
            one["basin"], one["seed"]))
        for c in superposition["comparisons"]:
            if "TS_QH_kcal" in c:
                print("   {:<24} T*S diff {:+.3e} kcal/mol   mass-weighted={}"
                      .format(c["library"], c["TS_difference_kcal"],
                              c["mass_weighted"]))
            else:
                print("   {:<24} {}".format(
                    c["library"], c.get("skipped") or c.get("failed")))
    except Exception as exc:                        # record it; do not lose the run
        superposition = dict(error_type=type(exc).__name__, error=str(exc))
        print("   superposition cross-check FAILED: {}: {}".format(
            type(exc).__name__, exc))

    #: The one comparison that isolates the implementation: same weighting, same
    #: reference protocol. A difference here is OUR arithmetic, and nothing else.
    mda_worst = None
    if superposition and "comparisons" in superposition:
        for c in superposition["comparisons"]:
            if c.get("library") == "MDAnalysis (iterated)" and "TS_difference_kcal" in c:
                mda_worst = abs(c["TS_difference_kcal"])

    # ---- verdicts --------------------------------------------------------------------
    sat_worst = max(abs(t["saturation"]["increment_over_last_doubling_kcal"])
                    for t in per_traj)
    rank_ok = all(t["analysis"]["rank_check"]["rank_is_full"] for t in per_traj)
    rigid_ok = all(t["analysis"]["spectrum"]["n_rigid_modes_removed"]
                   == t["analysis"]["spectrum"]["rigid_subspace_rank"] for t in per_traj)
    rigid_ratio = min(t["analysis"]["spectrum"]["rigid_to_first_vibrational_ratio"]
                      for t in per_traj)
    bound_ok = all(t["analysis"]["entropy"]["schlitter_bounds_qh"] for t in per_traj)
    identity_ok = all(t["analysis"]["identity_check"] is not None for t in per_traj)
    assembly_ok = all(a["consistency"]["all_identical"] for a in assembly)
    noise_floor = max((b["TS_QH_rms_about_mean_kcal"] for b in blank), default=0.0)
    # One trajectory per basin is the production setting. Criterion
    # 5 then has nothing to measure and says so; it must not read as a failure, or every
    # production molecule would stop at collect by construction.
    single_seed = bool(blank) and all(b["n_seeds"] == 1 for b in blank)
    modes_grew = any(t["mode_batches"]["mode_count_grew"] for t in per_traj)
    gmx_worst = None
    for c in cross:
        if c.get("TS_QH_difference_kcal") is not None:
            gmx_worst = max(gmx_worst or 0.0, abs(c["TS_QH_difference_kcal"]))

    # A short trajectory passes criterion 1 for the wrong reason: quasi-harmonic entropy
    # rises monotonically and saturates slowly, so a trajectory that has barely started
    # has barely started rising. The length therefore travels WITH the verdict, and
    # anything under SMOKE_LENGTH_PS is called what it is.
    longest_ps = max(t["analysis"]["rank_check"]["n_frames"]
                     * t["meta"]["frame_spacing_fs"] / 1000.0 for t in per_traj)
    smoke = longest_ps < SMOKE_LENGTH_PS

    verdicts = [
        ("1  the saturation curve is drawn and the increment over the last doubling of "
         "trajectory length is below {} kcal/mol on T*S".format(SATURATION_BUDGET_KCAL),
         "{:+.4f} kcal/mol (worst trajectory), longest trajectory {:.2f} ps{}".format(
             sat_worst, longest_ps,
             "  -- SMOKE LENGTH: a trajectory this short has barely begun to rise, so "
             "this pass says nothing about production" if smoke else ""),
         sat_worst < SATURATION_BUDGET_KCAL and not smoke),
        ("2  an independent MASS-WEIGHTED implementation reproduces T*S to within "
         "{} kcal/mol (gmx covar -mwa, or MDAnalysis fitted our way)"
         .format(GMX_BUDGET_KCAL),
         "  ".join(filter(None, [
             ("gmx: not run (--no-gmx)" if args.no_gmx else
              "gmx: no comparison produced" if gmx_worst is None else
              "gmx: {:.2e}".format(gmx_worst)),
             ("MDAnalysis: not available" if mda_worst is None else
              "MDAnalysis: {:.2e}".format(mda_worst)),
         ])) or "no comparison produced",
         ((gmx_worst is not None and gmx_worst < GMX_BUDGET_KCAL)
          or (mda_worst is not None and mda_worst < GMX_BUDGET_KCAL))),
        ("4  exactly the rigid-body modes are removed, and they separate cleanly from "
         "the first vibrational mode",
         "{} removed; smallest separation ratio {:.3e}".format(
             per_traj[0]["analysis"]["spectrum"]["n_rigid_modes_removed"], rigid_ratio),
         bool(rigid_ok and rigid_ratio > 1e6)),
        ("5  the blank control (same potential, same basin, different seed) gives a "
         "noise floor smaller than the 1.0 kcal/mol target accuracy",
         ("NOT APPLICABLE: one seed per basin, no noise floor "
          "measured here; see docs/branchB_seeds_and_length.md" if single_seed else
          "{:.4f} kcal/mol RMS between seeds, longest trajectory {:.2f} ps{}".format(
              noise_floor, longest_ps,
              "  -- SMOKE LENGTH: the noise floor of a production trajectory is a "
              "different number" if smoke else "")),
         bool(single_seed
              or (noise_floor < 1.0 and any(b["n_seeds"] > 1 for b in blank) and not smoke))),
        ("6  S_QH <= S_Schlitter for every trajectory (analytic, so a failure is an "
         "implementation error)",
         "all {} trajectories".format(len(per_traj)) if bound_ok else "violated",
         bound_ok),
        ("7  the translational, rotational and electronic terms are identical to the "
         "Hessian route's, so only the vibrational term differs",
         "identical" if assembly_ok else "NOT identical",
         assembly_ok),
        ("8  the symmetry number is read from the declaration, never derived",
         "sigma = {} ({})".format(spec["symmetry_number"], spec.get("symmetry_reason")),
         True),
        ("9  the number of non-zero eigenvalues equals 3N-6",
         "{} of {} expected{}".format(
             per_traj[0]["analysis"]["spectrum"]["n_nonzero_eigenvalues"],
             per_traj[0]["analysis"]["spectrum"]["expected_vibrational_modes"],
             # A covariance from T frames has at most T-1 non-zero eigenvalues, so a
             # smoke-length trajectory CANNOT pass this; say that it could not, rather
             # than letting it read like a defect (an113, 2026-09-13: 5 frames, 24 wanted).
             "  -- SMOKE LENGTH: {} frames cannot span {} modes; rank is not measurable here"
             .format(per_traj[0]["analysis"]["rank_check"]["n_frames"],
                     per_traj[0]["analysis"]["spectrum"]["expected_vibrational_modes"])
             if smoke and not rank_ok else ""),
         rank_ok),
        ("10 the per-batch convergence separates unsettled values from missing modes: "
         "the mode COUNT must not still be growing",
         "count still growing" if modes_grew else "count stable across prefixes",
         not modes_grew),
        ("11 every trajectory passes the identity assertion (no bias, no constraints, "
         "real hydrogen mass, 1 fs, not from CREST or the sampling loop)",
         "all {} trajectories".format(len(per_traj)) if identity_ok else "rejected",
         identity_ok),
    ]

    # ---- product ---------------------------------------------------------------------
    # collect.out, collect.toml and the Table collect.dat in _records/md_<route>/, the setting
    # in the stem (collect_s2.out); no setting level (records redesign, 2026-09-15).
    from openqha.quasi_harmonic import chain_records
    out_stem = Path(args.out) if args.out else chain_records.stem(molecule, route, args.setting, chain_records.COLLECT)
    r = report.Report(
        "openQHA branch B -- quasi-harmonic analysis",
        subtitle="{}  ({})   tag {}".format(args.species, spec.get("name"), args.tag))
    r.section("What produced these numbers")
    r.kv("species", args.species)
    r.kv("engine", per_traj[0]["meta"].get("engine_name"))
    r.kv("temperature_K", temperature)
    r.kv("symmetry_number", spec["symmetry_number"], note=spec.get("symmetry_reason"))
    r.kv("electronic_degeneracy", spec["electronic_degeneracy"])
    r.kv("trajectories", len(per_traj))
    r.kv("frame_spacing_fs", per_traj[0]["meta"]["frame_spacing_fs"])
    r.kv("timestep_fs", per_traj[0]["meta"]["timestep_fs"])
    r.note("The potential is MACE and the trajectories come from ASE. GROMACS evaluates "
           "no energy anywhere in this branch; it only diagonalises the same covariance "
           "independently. Reading these numbers as sitting on a classical force field "
           "would be wrong.")

    r.section("Per trajectory")
    r.table(["basin", "seed", "frames", "T*S_QH", "T*S_Schlitter", "modes", "3N-6",
             "last doubling"],
            [[t["basin"], t["seed"], t["analysis"]["rank_check"]["n_frames"],
              round(t["analysis"]["entropy"]["TS_QH_kcal"], 5),
              round(t["analysis"]["entropy"]["TS_Schlitter_kcal"], 5),
              t["analysis"]["spectrum"]["n_nonzero_eigenvalues"],
              t["analysis"]["spectrum"]["expected_vibrational_modes"],
              round(t["saturation"]["increment_over_last_doubling_kcal"], 5)]
             for t in per_traj],
            units=[None, None, None, "kcal/mol", "kcal/mol", None, None, "kcal/mol"])

    r.section("Saturation curve (criterion 1)")
    r.note("Quasi-harmonic entropy is a lower bound at any finite length. The increment "
           "over the last doubling is the main term of the error bar, so the criterion "
           "is written on the trajectory length rather than the length being tuned to "
           "pass it.")
    for t in per_traj:
        r.table(["fraction", "frames", "T*S_QH", "T*S_Schlitter", "non-zero modes"],
                [[p["fraction"], p["n_frames"], round(p["TS_QH_kcal"], 5),
                  round(p["TS_Schlitter_kcal"], 5), p["n_nonzero_eigenvalues"]]
                 for p in t["saturation"]["points"]],
                units=[None, None, "kcal/mol", "kcal/mol", None],
                title="{} {}   monotonic = {}".format(
                    t["basin"], t["seed"], t["saturation"]["monotonic"]))

    r.section("Blank control -- the noise floor (criterion 5)")
    r.note("Same potential, same basin, different seed. The density-of-states "
           "route was abandoned because this experiment showed its noise floor exceeded "
           "its signal. If that happens here it is reported in the same words.")
    r.table(["basin", "seeds", "T*S mean", "spread", "RMS", "standard error"],
            [[b["basin"], b["n_seeds"], round(b["TS_QH_mean_kcal"], 5),
              round(b["TS_QH_spread_kcal"], 5), round(b["TS_QH_rms_about_mean_kcal"], 5),
              "-" if b["standard_error_kcal"] is None
              else round(b["standard_error_kcal"], 5)] for b in blank],
            units=[None, None, "kcal/mol", "kcal/mol", "kcal/mol", "kcal/mol"])

    r.section("Assembly: (G - E_el) per basin")
    r.note("Only the vibrational term comes from this branch. Translation, rotation and "
           "electronic degeneracy go through thermo.g_minus_eel, the same function the "
           "Hessian route uses, with the same declared symmetry number.")
    r.table(["basin", "G - E_el", "A_vib", "A_rot", "A_trans", "A_elec", "terms match"],
            [[a["basin"], round(a["G_minus_Eel_kcal"], 4), round(a["A_vib_kcal"], 4),
              round(a["rotational_kcal"], 4), round(a["translational_kcal"], 4),
              round(a["electronic_kcal"], 4), a["consistency"]["all_identical"]]
             for a in assembly],
            units=[None, "kcal/mol", "kcal/mol", "kcal/mol", "kcal/mol", "kcal/mol",
                   None])

    if cross:
        r.section("GROMACS cross-check (criterion 2)")
        r.note("gmx anaeig -entropy is NOT the reference: it refuses mass-weighted "
               "eigenvalues, its formula expects them anyway, and it drops the six "
               "softest modes instead of the six rigid ones. See openqha/gmx_io.py. "
               "What is compared here is the mass-weighted covariance spectrum from "
               "gmx covar -mwa, which is independent and correct.")
        for c in cross:
            if c.get("error"):
                r.warn("{} {}: {}: {}".format(c["basin"], c["seed"], c["error_type"],
                                              c["error"]))
                continue
            r.kv("{} {} gmx".format(c["basin"], c["seed"]),
                 "{}  ({} precision)".format(c["gmx_version"].get("version"),
                                             c["gmx_version"].get("precision")))
            r.kv("eigenvalue max relative difference",
                 c["eigenvalue_max_relative_difference_vs_unprojected"])
            r.kv("eigenvalue sum relative difference",
                 c["eigenvalue_sum_relative_difference"])
            r.kv("T*S_QH difference_kcal", c["TS_QH_difference_kcal"])
            r.kv("T*S_Schlitter difference_kcal", c["TS_Schlitter_difference_kcal"])
            r.kv("fit protocol difference on T*S_kcal",
                 c["fit_protocol_difference"]["TS_QH_single_fit_kcal"]
                 - c["fit_protocol_difference"]["TS_QH_iterated_fit_kcal"],
                 note="gmx fits once to the -s structure; openQHA iterates to the mean. "
                      "Removed here by giving gmx our mean as -s; this is how big it was.")
            r.kv("Eckart projection effect on T*S_kcal",
                 c["projection_effect"]["TS_QH_projected_kcal"]
                 - c["projection_effect"]["TS_QH_unprojected_kcal"])

    r.section("Criteria")
    for criterion, measured, passed in verdicts:
        r.verdict(criterion, measured, passed)

    # The dump holds what nothing else holds: the per-trajectory analysis, saturation and
    # mode batches, the declaration and the cross-checks. The blank control, the assembly
    # and the criteria are in collect.dat and the Criteria section above (2026-09-16);
    # repeating them here made a 400-column table nobody could read.
    r.json_dump(dict(
        species=args.species, tag=args.tag, temperature_K=temperature,
        declaration=spec,
        trajectories=[dict(basin=t["basin"], seed=t["seed"], path=t["path"],
                           meta={k: v for k, v in t["meta"].items()
                                 if k not in ("masses_amu", "symbols")},
                           analysis=t["analysis"], saturation=t["saturation"],
                           mode_batches=t["mode_batches"]) for t in per_traj],
        gromacs_cross_check=cross,
        superposition_cross_check=superposition,
    ))

    # The Table (collect.dat, sections trajectories / blank / assembly, every column
    # commented from chain_records.COLUMNS), then the Report, then the Property file.
    sections = table_sections(args.species, per_traj, blank, assembly)
    drift = chain_records.write_collect_table(out_stem, sections)
    if drift:
        raise RuntimeError("collect.dat: columns and their explanations have drifted apart "
                           "{}; fix openqha.quasi_harmonic.chain_records.COLUMNS".format(drift))
    written = [(chain_records.collect_paths(out_stem)["dat"],
                ", ".join("{} {}".format(len(sections[s]), s) for s in chain_records.SECTIONS))]
    out_path = r.write(str(out_stem) + ".out", step="collect")
    # The Property file LAST: its STATUS is what the collect Batch reads to say this molecule
    # is done, so a kill before this line leaves the molecule to be redone.
    _m0 = per_traj[0]["meta"]
    unknown = chain_records.write_collect(out_stem, dict(
        species=args.species, tag=args.tag, basin_tag=args.basin_tag or args.tag, setting=args.setting,
        route=route, engine=_m0.get("engine_name"), temperature_K=temperature,
        sigma=spec["symmetry_number"], g0=spec["electronic_degeneracy"],
        n_trajectories=len(per_traj), n_basins=len({t["basin"] for t in per_traj}),
        frame_spacing_fs=_m0.get("frame_spacing_fs"), timestep_fs=_m0.get("timestep_fs")), verdicts)
    if unknown:
        raise RuntimeError("collect.toml: keys outside the schema {}".format(unknown))

    print()
    for criterion, measured, passed in verdicts:
        print("[{}] {}".format("PASS" if passed else "FAIL", criterion.splitlines()[0]))
        print("       measured: {}".format(measured))
    print("\nwritten:\n  {}".format(out_path))
    for p, n in written:
        print("  {}  ({} rows)".format(p, n))
    return 0 if all(p for _c, _m, p in verdicts) else 1


if __name__ == "__main__":
    raise SystemExit(main())
