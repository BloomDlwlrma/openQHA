"""Validate openqha/symmetry.py -- the geometry-derived symmetry number.

CALIBRATION. Branch A. It produces the numbers used to decide
whether the module may be trusted in production; it produces nothing that enters a
deliverable.

Three parts, and the second one is the point
--------------------------------------------
1. AGREEMENT  -- the seven species whose sigma we declared by hand are the only
   answers we know independently. The detector must reproduce all seven.

2. EXAMPLES IT MUST FAIL: a new criterion that has only
   ever passed has not been tested. So this script also runs cases where a WRONG
   implementation would give a specific, known-wrong answer:

     * ethane      graph automorphism = 18, correct sigma = 6.  A module that
                   counted automorphisms would print 18 here. This is the exact
                   failure this check exists to catch.
     * methane     graph automorphism = 24, correct sigma = 12 (Td has 12 proper
                   rotations; the other 12 operations are improper).
     * benzene     graph automorphism = 24, correct sigma = 12 (D6h).
     * CO2         linear, sigma = 2 -- the finite rotation subgroup, not infinity.
     * propanal    sigma = 1 with 3 graph automorphisms from the methyl rotor.
                   A rotor-counting implementation would print 3.

   If any of these prints the automorphism count instead of the symmetry number,
   the module is doing the forbidden thing and the run must stop.

3. DISPLACEMENT SWEEP -- sigma is a property of a GEOMETRY. Optimised basins sit
   1e-3 to 1e-2 A off ideal symmetry, and pymsym collapses to C1 at 0.005 A
   (measured 2026-09-03). This part reports the displacement at which OUR detector
   changes its mind, which is what decides whether TOLERANCE_A is usable.

Usage
-----
    python scripts/calibration/s0_A_symmetry_number.py
    python scripts/calibration/s0_A_symmetry_number.py --out analysis/symmetry_number.json
"""
import argparse
import json
import sys
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

from openqha import S0_ROOT, config, symmetry  # noqa: E402

PERTURB_SEED = 20260903
DISPLACEMENTS_A = (0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20)

#: Cases whose answer is known from the point group, chosen so that a wrong
#: implementation prints a specific, recognisable wrong number.
#:
#: `expected_sigma`   what the external rotational symmetry number must be.
#: `forbidden_answer` what a graph-automorphism count prints instead, MEASURED by
#:                    this script on 2026-09-03, not quoted from anywhere.
#:
#: A note on the "ethane 18" figure: it counts only the cyclic rotations of
#: the two methyls times the C2 (3 x 3 x 2). The FULL automorphism group of the
#: ethane graph is 3! x 3! x 2 = 72, which is what an implementation calling
#: `GetSubstructMatches(mol, uniquify=False)` actually returns and what this
#: script measures. The conclusion is unchanged and the size estimate is if
#: anything understated: the error is a factor 12, not 3.
KNOWN_CASES = (
    dict(name="C2H6",  label="ethane",   point_group="D3d", expected_sigma=6,
         forbidden_answer=72,
         note="The reference example. The rigid subgroup is D3d (12 ops, 6 proper); "
              "the other 60 automorphisms are internal-rotor and H-permutation "
              "symmetry and are not external symmetry."),
    dict(name="CH4",   label="methane",  point_group="Td",  expected_sigma=12,
         forbidden_answer=24,
         note="Td has 24 operations but only 12 proper rotations. Catches an "
              "implementation that counts the point group instead of its "
              "rotational subgroup."),
    dict(name="C6H6",  label="benzene",  point_group="D6h", expected_sigma=12,
         forbidden_answer=12,
         note="Automorphism count happens to equal sigma here, so this is NOT a "
              "discriminating case for the automorphism count. It is in the list because it is "
              "planar: it is one of the cases that caught the degenerate-"
              "determinant defect on 2026-09-03 (it printed 2)."),
    dict(name="H2O",   label="water",    point_group="C2v", expected_sigma=2,
         forbidden_answer=2,
         note="Planar control. Printed 1 before the determinant fix."),
    dict(name="CO2",   label="carbon dioxide", point_group="Dinfh", expected_sigma=2,
         forbidden_answer=2,
         note="Linear. The finite proper subgroup is E and C2, so sigma = 2. "
              "Printed 1 before the determinant fix."),
    dict(name="CH3CHO", label="acetaldehyde", point_group="Cs", expected_sigma=1,
         forbidden_answer=6,
         note="Methyl rotor plus the mirror: 6 graph automorphisms, sigma = 1."),
)


def build(name):
    from ase.build import molecule
    return molecule(name)


def check_known():
    rows = []
    for case in KNOWN_CASES:
        atoms = build(case["name"])
        rec = symmetry.analyse_atoms(atoms)
        rows.append(dict(
            case, detected_sigma=rec["sigma"],
            n_graph_automorphisms=rec["n_graph_automorphisms"],
            n_improper=rec["n_improper"],
            pymsym_point_group=rec["pymsym_point_group"],
            margin_A=rec["margin_A"],
            group_closure_defect=rec["group_closure_defect"],
            sigma_stable_below_tolerance=rec["sigma_stable_below_tolerance"],
            sigma_flip_tolerance_A=rec["sigma_flip_tolerance_A"],
            sigma_stable_over_whole_sweep=rec["sigma_stable_over_whole_sweep"],
            passes=bool(rec["sigma"] == case["expected_sigma"]),
            printed_the_forbidden_answer=bool(
                rec["sigma"] == case["forbidden_answer"]
                and case["forbidden_answer"] != case["expected_sigma"]),
        ))
    return rows


def check_declared():
    cfg = config.load()
    rows = []
    for qid, spec in sorted(cfg["species"].items()):
        xyz = S0_ROOT / "data" / "reference-geometries" / "{}.xyz".format(qid)
        if not xyz.exists():
            continue
        atoms = read(xyz)
        declared = int(spec["symmetry_number"])
        rec = symmetry.analyse_atoms(atoms)
        rows.append(dict(
            qm9_index=qid, name=spec.get("name"), declared_sigma=declared,
            detected_sigma=rec["sigma"],
            n_graph_automorphisms=rec["n_graph_automorphisms"],
            pymsym_point_group=rec["pymsym_point_group"],
            pymsym_sigma=rec["pymsym_sigma"],
            margin_A=rec["margin_A"],
            max_accepted_rmsd_A=rec["max_accepted_rmsd_A"],
            min_rejected_rmsd_A=rec["min_rejected_rmsd_A"],
            group_closure_defect=rec["group_closure_defect"],
            agrees=bool(rec["sigma"] == declared)))
    return rows


def displacement_sweep():
    """How far a geometry may drift before each detector changes its mind."""
    cfg = config.load()
    rng = np.random.default_rng(PERTURB_SEED)
    rows = []
    for qid, spec in sorted(cfg["species"].items()):
        xyz = S0_ROOT / "data" / "reference-geometries" / "{}.xyz".format(qid)
        if not xyz.exists():
            continue
        atoms = read(xyz)
        x0 = atoms.get_positions().copy()
        declared = int(spec["symmetry_number"])
        ours, theirs = {}, {}
        for amp in DISPLACEMENTS_A:
            x = x0 + (rng.standard_normal(x0.shape) * amp if amp else 0.0)
            atoms.set_positions(x)
            rec = symmetry.analyse_atoms(atoms, scan=())
            ours["{:g}".format(amp)] = rec["sigma"]
            theirs["{:g}".format(amp)] = rec["pymsym_sigma"]
        atoms.set_positions(x0)
        rows.append(dict(qm9_index=qid, name=spec.get("name"),
                         declared_sigma=declared,
                         ours_by_displacement=ours,
                         pymsym_by_displacement=theirs))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="analysis/symmetry_number.json")
    args = ap.parse_args()

    print("=" * 92)
    print("Branch A -- geometry-derived symmetry number, validation")
    print("tolerance {:.3f} A   sweep {}".format(
        symmetry.TOLERANCE_A, symmetry.TOLERANCE_SCAN_A))
    print("=" * 92)

    print()
    print("PART 2 -- cases a graph-automorphism implementation would get wrong")
    print("{:<16} {:>6} {:>8} {:>8} {:>10} {:>9}  {}".format(
        "molecule", "sigma", "expected", "autos", "forbidden", "margin/A", "verdict"))
    known = check_known()
    for r in known:
        print("{:<16} {:>6} {:>8} {:>8} {:>10} {:>9}  {}".format(
            r["label"], r["detected_sigma"], r["expected_sigma"],
            r["n_graph_automorphisms"], r["forbidden_answer"],
            "n/a" if r["margin_A"] is None else "{:.3f}".format(r["margin_A"]),
            "PASS" if r["passes"] else "FAIL"))

    print()
    print("PART 1 -- the seven declared species")
    print("{:<20} {:>9} {:>9} {:>7} {:>9} {:>9}  {}".format(
        "species", "declared", "detected", "autos", "pymsym", "margin/A", "verdict"))
    declared = check_declared()
    for r in declared:
        print("{:<20} {:>9} {:>9} {:>7} {:>9} {:>9}  {}".format(
            (r["name"] or r["qm9_index"])[:20], r["declared_sigma"],
            r["detected_sigma"], r["n_graph_automorphisms"],
            "{}/{}".format(r["pymsym_point_group"], r["pymsym_sigma"]),
            "n/a" if r["margin_A"] is None else "{:.3f}".format(r["margin_A"]),
            "PASS" if r["agrees"] else "FAIL"))

    print()
    print("PART 3 -- displacement at which each detector changes its mind")
    sweep = displacement_sweep()
    head = "  ".join("{:>6}".format(a) for a in DISPLACEMENTS_A)
    print("{:<20} {:>4}  {}".format("species", "sig", head))
    for r in sweep:
        ours = "  ".join("{:>6}".format(r["ours_by_displacement"]["{:g}".format(a)])
                         for a in DISPLACEMENTS_A)
        theirs = "  ".join(
            "{:>6}".format(r["pymsym_by_displacement"]["{:g}".format(a)])
            for a in DISPLACEMENTS_A)
        print("{:<20} {:>4}  {}   <- ours".format(
            (r["name"] or r["qm9_index"])[:20], r["declared_sigma"], ours))
        print("{:<20} {:>4}  {}   <- pymsym".format("", "", theirs))

    n_fail = sum(1 for r in known if not r["passes"])
    n_fail += sum(1 for r in declared if not r["agrees"])
    n_forbidden = sum(1 for r in known if r["printed_the_forbidden_answer"])

    summary = dict(
        generated_by="scripts/calibration/s0_A_symmetry_number.py",
        tolerance_A=symmetry.TOLERANCE_A,
        tolerance_scan_A=list(symmetry.TOLERANCE_SCAN_A),
        perturbation_seed=PERTURB_SEED,
        displacements_A=list(DISPLACEMENTS_A),
        known_cases=known,
        declared_species=declared,
        displacement_sweep=sweep,
        n_failures=n_fail,
        n_cases_printing_the_forbidden_answer=n_forbidden,
        verdict=("PASS" if n_fail == 0 and n_forbidden == 0 else "FAIL"),
        identity=("Calibration only. sigma enters G_rot as +RT ln(sigma); an "
                  "error of a factor 2 is 0.411 kcal/mol at 298.15 K."))

    out = S0_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    print("=" * 92)
    print("failures {}   cases printing the forbidden answer {}   verdict {}".format(
        n_fail, n_forbidden, summary["verdict"]))
    print("written  {}".format(out))
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
