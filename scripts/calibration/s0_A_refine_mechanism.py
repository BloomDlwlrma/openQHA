# -*- coding: utf-8 -*-
"""Why `refine="opt"` returns fewer basins than `refine="sp"`.

CALIBRATION. Branch A. It explains the measurement behind acceptance criterion 3; it
produces an explanation, not a number that enters a deliverable.

Acceptance criterion 3 measured THAT it is worse (OCCC(=O)CO, gfn2: 23 basins against
37, 7.7x the wall clock, 0.4023 kcal/mol of error). This asks WHY, because a parameter
that was changed on a measurement nobody can explain is a parameter that will be changed
back by the next person with a different molecule.

    python scripts/calibration/s0_A_refine_mechanism.py \\
        --ensemble ~/runs/openQHA/branchA/refine_compare/dihydroxybutanone/sp/crest_conformers.xyz \\
        --smiles "OCCC(=O)CO" --label dihydroxybutanone

THE HYPOTHESIS
    `opt` has CREST relax the ensemble on MACE and then hand it to CREGEN, whose
    structural condition is an RMSD threshold. Two distinct minima still closer than that
    AT CREST'S CONVERGENCE are merged, and the merge is final -- the discarded structure
    never reaches us. `sp` merges nothing on the MACE surface, so this repository can
    tighten FIRST (fmax = 1e-4 eV/A) and deduplicate SECOND, by which time those minima
    have separated.

WHAT WOULD REFUTE IT
    A basin count that does not depend on convergence. If one fixed ensemble yields the
    same number of basins whether it was deduplicated loose or tight, then merging before
    convergence costs nothing and the difference lies somewhere else -- in the sampling,
    or in CREST's energy window. The run prints the curve either way.

The `sp` ensemble is the input on purpose: it is the one that was NOT merged on the MACE
surface, so it still contains the structures whose fate is in question.
"""
import argparse
import json
import sys
import time
from pathlib import Path


def _repo_root():
    """Search upward for the package, rather than counting directory levels.

    `parents[2]` is banned here and t_repo_bootstrap enforces it: this repository has
    been broken four times by a hard-coded depth (scripts/ reorganised, hpc/ moved twice,
    source-code/ moved). A name written into a path does not move with the thing it names.
    """
    for p in Path(__file__).resolve().parents:
        if (p / "openqha" / "__init__.py").is_file():
            return p
    raise RuntimeError("openQHA package not found above " + __file__)


sys.path.insert(0, str(_repo_root()))

from openqha import S0_ROOT, engine, refine_analysis  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble", required=True,
                    help="crest_conformers.xyz from the refine=sp run")
    ap.add_argument("--smiles", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--limit", type=int, default=0,
                    help="use only the first N structures (a smoke test, not a result)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from ase.io import read

    frames = read(args.ensemble, index=":")
    if args.limit:
        frames = frames[:args.limit]
    calc, engine_name, prov = engine.calculator()

    print("=" * 92)
    print("Why refine=opt loses basins -- basin count against convergence tightness")
    print("=" * 92)
    print("ensemble    {}".format(args.ensemble))
    print("structures  {}{}".format(
        len(frames), "  (LIMITED -- smoke test, not a result)" if args.limit else ""))
    print("engine      {}  sha256 {}".format(engine_name, prov["sha256"][:16]))
    print("dedup       CREGEN three-fold, held FIXED at every rung:")
    print("            RMSD {} A, dE {} kcal/mol, dB {}".format(
        refine_analysis.conformers.DEDUP_RMSD_A,
        refine_analysis.conformers.CREGEN_ETHR_KCAL,
        refine_analysis.conformers.CREGEN_BTHR_REL))
    print()

    t0 = time.time()
    rows = refine_analysis.basins_vs_convergence(
        frames, calc, args.smiles,
        progress=lambda m: print("  " + m, end="\r", flush=True))
    wall = time.time() - t0
    print(" " * 60, end="\r")

    print("{:>12}  {:>8}  {:>8}  {:>8}  {:>14}  {:>12}".format(
        "fmax /eV/A", "input", "basins", "merges", "correction/kcal", "spread/kcal"))
    print("-" * 92)
    for r in rows:
        if r["n_basins"] is None:
            print("{:>12.1e}  {:>8}  {:>8}".format(r["fmax_eV_A"], r["n_input"], "--"))
            continue
        print("{:>12.1e}  {:>8}  {:>8}  {:>8}  {:>14.4f}  {:>12.4f}".format(
            r["fmax_eV_A"], r["n_input"], r["n_basins"], r["merges"],
            r["correction_kcal"], r["spread_kcal"]))

    got = [r for r in rows if r["n_basins"]]
    verdict = "INCONCLUSIVE"
    if len(got) >= 2:
        loose, tight = got[0]["n_basins"], got[-1]["n_basins"]
        if tight > loose:
            verdict = ("SUPPORTED: {} basins at fmax {:.0e} against {} at {:.0e}. "
                       "Deduplicating before convergence destroys minima that have not "
                       "yet separated.".format(loose, got[0]["fmax_eV_A"],
                                               tight, got[-1]["fmax_eV_A"]))
        elif tight == loose:
            verdict = ("REFUTED: the count does not depend on convergence ({} at both "
                       "ends). The `opt` deficit is not merging-before-convergence; look "
                       "at the sampling or CREST's energy window.".format(tight))
        else:
            verdict = ("REFUTED, and backwards: tighter convergence gave FEWER basins "
                       "({} against {}). That needs explaining before anything here is "
                       "used.".format(tight, loose))
    print()
    print("verdict: {}".format(verdict))
    print("wall {:.1f} s".format(wall))

    out = Path(args.out) if args.out else (
        S0_ROOT / "analysis" / "branchA" /
        "refine_mechanism_{}.json".format(args.label))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        generated_by="scripts/calibration/s0_A_refine_mechanism.py",
        ensemble=str(args.ensemble), smiles=args.smiles, label=args.label,
        n_structures=len(frames), limited=bool(args.limit),
        engine=prov, dedup=dict(
            rmsd_A=refine_analysis.conformers.DEDUP_RMSD_A,
            ethr_kcal=refine_analysis.conformers.CREGEN_ETHR_KCAL,
            bthr_relative=refine_analysis.conformers.CREGEN_BTHR_REL),
        ladder=list(refine_analysis.FMAX_LADDER_EV_A),
        rows=rows, verdict=verdict, wall_seconds=wall), indent=2), encoding="utf-8")
    print("written {}".format(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
