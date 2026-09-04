"""Regression: the symmetry number must not collapse on planar or linear molecules.

REGRESSION. Branch A, checkpoint 3 criterion 3. Seconds; no potential, no CREST.

The defect this pins down (2026-09-03)
--------------------------------------
`openqha/symmetry.py` decides whether a candidate atom permutation is a real
symmetry operation by superposing the geometry on its permuted copy. The first
version solved ONE unconstrained Kabsch fit and read the sign of its determinant to
decide proper against improper.

That is degenerate whenever the geometry is planar or linear: the third singular
value of the correlation matrix is zero, so the determinant's sign is arbitrary and
numpy's SVD picks one. Measured, first version:

    water     1   (correct 2)
    benzene   2   (correct 12)
    methane   4   (correct 12)
    ethane    4   (correct 6)

**And all seven production species passed on that same code.** Acetone, oxetane and
the five C1 molecules were all correct, so the seven species alone would have shipped
this. That is the whole argument of plan_A section 2.5.3 and skills section 2.4(a) --
a criterion has to be shown an example it should fail -- and this test is that
argument made permanent.

The fix: solve BOTH constrained fits (det = +1 and det = -1) for every permutation
and ask "does a PROPER rotation realise this permutation", rather than asking which
unconstrained fit happened to win.

Why sigma being wrong matters: G_rot carries +RT ln(sigma). Water reported as
sigma = 1 instead of 2 puts G low by RT ln 2 = 0.411 kcal/mol -- the same error and
the same direction as the one ORCA makes on acetone (D0-P2-15), against a 1.0
kcal/mol target, and it does not cancel between isomers.

Run::  python tests/regression/t_symmetry_planar_determinant.py
"""
import sys
from pathlib import Path


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


sys.path.insert(0, str(_repo_root()))

from openqha import symmetry  # noqa: E402

#: (ase.build name, correct sigma, why it is in this list)
CASES = (
    ("H2O", 2, "planar; printed 1 before the fix"),
    ("C6H6", 12, "planar; printed 2 before the fix"),
    ("CO2", 2, "linear; printed 1 before the fix"),
    ("CH4", 12, "printed 4 before the fix; also separates the point group (24 ops) "
                "from its rotational subgroup (12)"),
    ("C2H6", 6, "printed 4 before the fix; D0-9's own example, 72 graph "
               "automorphisms against a true sigma of 6"),
    ("CH3CHO", 1, "methyl rotor: 6 graph automorphisms, sigma = 1"),
)


def main():
    from ase.build import molecule

    fails = []
    print("{:<10} {:>6} {:>9} {:>7}  {}".format(
        "molecule", "sigma", "expected", "autos", "verdict"))
    for name, expected, _why in CASES:
        atoms = molecule(name)
        rec = symmetry.analyse_atoms(atoms)
        ok = rec["sigma"] == expected
        if not ok:
            fails.append("{}: sigma {}, expected {}".format(
                name, rec["sigma"], expected))
        print("{:<10} {:>6} {:>9} {:>7}  {}".format(
            name, rec["sigma"], expected, rec["n_graph_automorphisms"],
            "PASS" if ok else "FAIL"))

    # The proper operations must also form a group. A count that is not a group
    # order is not a symmetry number, whatever it happens to equal.
    for name, expected, _why in CASES:
        rec = symmetry.analyse_atoms(molecule(name))
        if rec["group_closure_defect"]:
            fails.append("{}: accepted proper operations are not closed under "
                         "composition ({} products fall outside)".format(
                             name, rec["group_closure_defect"]))

    print()
    if fails:
        print("{} failure(s):".format(len(fails)))
        for f in fails:
            print("  - " + f)
        return 1
    print("planar and linear cases hold; proper operations form a group in all "
          "{} cases".format(len(CASES)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
