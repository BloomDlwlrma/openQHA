"""Measure what `refine="opt"` costs against `refine="sp"` under the GFN2 workhorse.

CALIBRATION. Branch A acceptance criterion 3. It produces the number the criterion asks
for; it produces nothing that enters a deliverable.

The debt this pays
------------------
The choice of `refine="opt"` rests on a user ruling. The measurement that had argued
*against* it was taken with the **GFNFF** workhorse
(`scripts/_superseded/one-off/s0_crest_refine_compare.py`):

    propanal CCC=O (3 basins globally)
      opt   CREST  833 s -> 2 basins   conformational correction error +0.1054 kcal/mol
      sp    CREST  136 s -> 3 basins   error +0.0000
      none  CREST    3 s -> 1 basin    error +0.2336

    OCCC(=O)CO (34 basins globally)
      opt   CREST 2926 s -> 11 basins  error +0.0103
      sp    CREST  496 s -> 28 basins  error +0.0003
      none  CREST    9 s -> 20 basins  error +0.0045

On those numbers `sp` was both faster and more complete, with no trade-off.

**The numbers are not wrong. They no longer apply.** The mechanism is that `opt`
re-optimises and merges the ensemble BEFORE cregen deduplicates, and how much it merges
depends on how far the workhorse geometries sit from the MACE minimum. GFN2 geometries
are far closer than GFNFF ones, so the size of the effect is unmeasured under the
workhorse actually in use. **The ruling stands whatever this shows; the measured miss
rate is a deliverable either way.**

What is compared
----------------
For each molecule and each refinement level, the FOUR numbers that matter:

    CREST wall clock          what it costs
    conformers CREST reports  what the search says it found
    basins by our criteria    what survives tightening, CREGEN dedup and the Hessian
    conformational correction -RT ln sum_i exp(-dG_i/RT), relative to the union

The fourth is the one the criterion is about: **how much free energy is lost by the
basins a setting misses**. It is computed against the UNION of the two ensembles, so
neither setting is scored against itself.

Usage
-----
    python scripts/calibration/s0_A_refine_compare.py --species dsgdb9nsd_000035
    python scripts/calibration/s0_A_refine_compare.py --smiles "OCCC(=O)CO" --label dhb
"""
import argparse
import json
import sys
import time
from pathlib import Path


def _repo_root():
    for p in Path(__file__).resolve().parents:
        if (p / "openqha" / "__init__.py").is_file():
            return p
    raise RuntimeError("openqha package not found above " + __file__)


sys.path.insert(0, str(_repo_root()))

import numpy as np  # noqa: E402

from openqha import (S0_ROOT, conformers, config, crest, crest_census,  # noqa: E402
                     engine)

EV_TO_KCAL = conformers.EV_TO_KCAL
LEVELS = ("opt", "sp")


def run_crest_at(level, workdir, start_xyz, cfg, threads, timeout_s):
    """One CREST run at one refinement level. Returns the record."""
    c = cfg["crest"]
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    xyz = workdir / Path(start_xyz).name
    if Path(start_xyz).resolve() != xyz.resolve():
        xyz.write_text(Path(start_xyz).read_text(encoding="utf-8"), encoding="utf-8")

    wanted = dict(runtype=c["runtype"], optlev=c["optlev"], refine=level,
                  shake=c["shake"], workhorse=c["workhorse"], tstep=c["tstep_fs"])
    existing = crest.read_input_settings(workdir / "input.toml")
    if existing is not None and (workdir / "crest_conformers.xyz").exists():
        ok, why = crest.settings_match(existing, wanted)
        if ok:
            rec = crest.record_from_dir(workdir, settings=wanted)
            rec["reused_scratch"] = True
            rec["wall_is_valid_cost"] = False
            rec["wall_seconds"] = rec.get("seconds")
            return rec
        raise RuntimeError("{} was built under different settings ({}) -- delete it "
                           "rather than mixing two conditions".format(workdir, why))

    t0 = time.time()
    rec = crest.run(workdir, xyz, runtype=c["runtype"], threads=int(threads),
                    optlev=c["optlev"], refine=level, backend=c["backend"],
                    engine_client=S0_ROOT / c["engine_client"],
                    workhorse=c["workhorse"], shake=int(c["shake"]),
                    tstep_fs=float(c["tstep_fs"]), timeout_s=int(timeout_s))
    rec["wall_seconds"] = time.time() - t0
    rec["reused_scratch"] = False
    rec["wall_is_valid_cost"] = True
    return rec


def census_of(smiles, ens_path, calc, cfg, args):
    """Tighten, deduplicate and Hessian-screen one ensemble. Returns (record, atoms)."""
    frames, comments = crest_census.read_ensemble_atoms(ens_path)
    pkg1, pkg2 = cfg["package1"], cfg["package2"]
    rec, basins, _mol = crest_census.census_from_frames(
        smiles, frames, calc,
        fmax=float(pkg2["fmax_hessian_eV_A"]),
        threshold_A=float(pkg1["dedup_rmsd_A"]),
        temperature_K=config.temperature(cfg),
        do_hessian=True, reject_imaginary=True, comments=comments,
        hessian_mode=args.hessian_mode,
        ethr_kcal=float(pkg1["dedup_ethr_kcal"]),
        bthr_rel=float(pkg1["dedup_bthr_relative"]))
    return rec, basins


def correction_kcal(energies_kcal, temperature_K):
    """-RT ln sum_i exp(-dE_i/RT), relative to the lowest of the SAME list."""
    r = 0.0019872041
    kt = r * temperature_K
    e = np.asarray(energies_kcal, dtype=float)
    e = e - e.min()
    return float(-kt * np.log(np.exp(-e / kt).sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species", default=None)
    ap.add_argument("--smiles", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--timeout-s", type=int, default=21600)
    ap.add_argument("--hessian-mode", default="analytic")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if not (args.species or args.smiles):
        raise SystemExit("give --species or --smiles")

    cfg = config.load()
    label = args.label or args.species
    calc, name, prov = engine.calculator(device=args.device)
    temperature = config.temperature(cfg)

    print("=" * 92)
    print("Branch A acceptance criterion 3 -- refine=opt vs sp, workhorse {}".format(
        cfg["crest"]["workhorse"]))
    print("=" * 92)
    print("molecule  {}".format(label))
    print("engine    {}".format(name))
    print("dedup     {} (RMSD {} A, dE {} kcal/mol, dB {})".format(
        cfg["package1"]["dedup_criterion"], cfg["package1"]["dedup_rmsd_A"],
        cfg["package1"]["dedup_ethr_kcal"], cfg["package1"]["dedup_bthr_relative"]))
    print()

    runs = config.runs_dir("branchA", cfg) / "refine_compare" / label

    # start geometry: shipped/QM9 for a species, one relaxed embedding for a SMILES
    if args.species:
        start = config.qm9_xyz(args.species, cfg)
        smiles = config.qm9_smiles(args.species, cfg)
    else:
        sys.path.insert(0, str(S0_ROOT / "scripts" / "production"))
        import s0_A_pipeline as P
        runs.mkdir(parents=True, exist_ok=True)
        start = runs / "{}_seed.xyz".format(label)
        smiles = args.smiles
        if not start.exists():
            P.seed_geometry_from_smiles(smiles, calc, start)
    print("smiles    {}".format(smiles))
    print("start     {}".format(start))
    print()

    results = {}
    for level in LEVELS:
        print("-" * 92)
        print("refine = {}".format(level))
        crec = run_crest_at(level, runs / level, start, cfg, args.threads,
                            args.timeout_s)
        ens = Path(crec["workdir"]) / "crest_conformers.xyz"
        if not ens.exists():
            raise FileNotFoundError("CREST produced no ensemble at {}".format(ens))
        cen, basins = census_of(smiles, ens, calc, cfg, args)
        results[level] = dict(
            crest=crec, census=cen, basins=basins,
            n_conformers=cen["crest_vs_repo"]["n_conformers_reported_by_crest"],
            n_basins=cen["n_basins"],
            wall_seconds=crec.get("wall_seconds"),
            wall_is_valid_cost=crec.get("wall_is_valid_cost"),
            terminated_early=crec.get("n_terminated_early"),
            engrad=crec.get("total_engrad_calls"),
            basin_energies_kcal=[e * EV_TO_KCAL for e in cen["basin_energies_eV"]])
        print("  CREST wall {:.1f} s | EARLY {} | engrad {} | conformers {} | basins {}"
              .format(crec.get("wall_seconds") or float("nan"),
                      crec.get("n_terminated_early"), crec.get("total_engrad_calls"),
                      results[level]["n_conformers"], results[level]["n_basins"]))

    # ---- the union, so neither setting is scored against itself ---------------------
    #
    # Built by POOLING GEOMETRIES and applying the same CREGEN criterion that defines a
    # basin everywhere else. The obvious shortcut -- concatenating the two energy lists
    # -- is wrong, and wrong in a way that looks like a result: when both settings find
    # the same basins, every basin is counted twice and -RT ln(sum exp) gains exactly
    # RT ln 2 = 0.4107 kcal/mol. The first run of this script reported that number as
    # the "error" of BOTH settings, which is how the defect was noticed: a value equal
    # to something structural is usually measuring the structure (D0-81).
    pooled = []
    for level in LEVELS:
        pooled.extend(results[level]["basins"])
    union_rec, union_basins, _m = crest_census.census_from_frames(
        smiles, pooled, calc,
        fmax=float(cfg["package2"]["fmax_hessian_eV_A"]),
        threshold_A=float(cfg["package1"]["dedup_rmsd_A"]),
        temperature_K=temperature, do_hessian=True, reject_imaginary=True,
        hessian_mode=args.hessian_mode,
        ethr_kcal=float(cfg["package1"]["dedup_ethr_kcal"]),
        bthr_rel=float(cfg["package1"]["dedup_bthr_relative"]))
    all_e = [e * EV_TO_KCAL for e in union_rec["basin_energies_eV"]]
    ref = min(all_e)
    union_corr = correction_kcal([e - ref for e in all_e], temperature)
    print()
    print("union: %d structures pooled -> %d basins by the same criterion"
          % (len(pooled), union_rec["n_basins"]))

    print()
    print("=" * 92)
    print("%-8s %10s %8s %12s %12s %14s %14s" % (
        "refine", "wall/s", "EARLY", "conformers", "basins", "corr/kcal", "error/kcal"))
    summary_rows = []
    for level in LEVELS:
        r = results[level]
        corr = correction_kcal([e - ref for e in r["basin_energies_kcal"]], temperature)
        err = corr - union_corr
        print("%-8s %10.1f %8s %12d %12d %14.4f %14.4f" % (
            level, r["wall_seconds"] or float("nan"), r["terminated_early"],
            r["n_conformers"], r["n_basins"], corr, err))
        summary_rows.append(dict(refine=level, wall_seconds=r["wall_seconds"],
                                 terminated_early=r["terminated_early"],
                                 n_conformers=r["n_conformers"],
                                 n_basins=r["n_basins"],
                                 conformational_correction_kcal=corr,
                                 error_vs_union_kcal=err,
                                 engrad_calls=r["engrad"],
                                 wall_is_valid_cost=r["wall_is_valid_cost"]))
    print()
    print("union of both ensembles: %d basins, correction %.4f kcal/mol"
          % (len(all_e), union_corr))
    print("(the union is a re-deduplicated pool of both basin sets, not a "
          "concatenation -- see the comment in the source)")

    out = dict(
        generated_by="scripts/calibration/s0_A_refine_compare.py",
        label=label, smiles=smiles, engine=prov,
        workhorse=cfg["crest"]["workhorse"],
        dedup=dict(criterion=cfg["package1"]["dedup_criterion"],
                   rmsd_A=cfg["package1"]["dedup_rmsd_A"],
                   ethr_kcal=cfg["package1"]["dedup_ethr_kcal"],
                   bthr_relative=cfg["package1"]["dedup_bthr_relative"]),
        temperature_K=temperature,
        rows=summary_rows,
        union=dict(n_basins=len(all_e), correction_kcal=union_corr,
                   n_pooled_structures=len(pooled),
                   built_by=("pooling both basin sets and re-applying the CREGEN "
                             "criterion; NOT concatenating energy lists, which "
                             "double-counts shared basins and adds RT ln 2")),
        note=("The error column is measured against the UNION of both ensembles, so "
              "neither setting is scored against itself. The prior measurement that "
              "favoured sp was taken with the GFNFF workhorse and does not transfer: "
              "opt's merging depends on how far the workhorse geometries sit from the "
              "MACE minimum, and GFN2's are much closer."))

    dest = (S0_ROOT / "analysis" / "branchA" /
            "refine_compare_{}.json".format(label))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("written %s" % dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
