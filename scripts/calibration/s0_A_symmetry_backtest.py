"""Back-test pymsym's 3D point-group symmetry numbers against our declared values.

CALIBRATION. Branch A, step 5 gate. It produces a number used to decide whether
pymsym may be trusted; it produces nothing that enters a deliverable.

Why this exists
---------------
D0-9 forbids deriving the external rotational symmetry number sigma from the
molecular GRAPH: graph automorphism counts internal-rotor and H-permutation
symmetry, which are not external symmetry, and overcounts badly for flexible
molecules (ethane 18 against a true sigma of 6, worth kT ln 3 = 0.651 kcal/mol --
the same order as the 1.0 kcal/mol target, and it does not cancel between isomers).

pymsym detects the point group from 3D COORDINATES (it is the Python binding of
libmsym), so it does not have that failure mode. That is why it is allowed where
`GetSubstructMatches(mol, uniquify=False)` is not. See plan_A section 2.5.

But an automatic detector can still be wrong: ORCA calls acetone `C1, Symmetry
Number: 1` when the correct value is 2 (D0-P2-15), costing RT ln 2 = 0.411
kcal/mol. So pymsym has to earn trust on the seven species whose sigma we declared
by hand -- those are the only answers we know independently.

Two things this checks
----------------------
1. Agreement with the declared sigma on all seven species.
2. Displacement sensitivity: sigma is a property of a GEOMETRY, and a geometry
   that sits slightly off ideal symmetry can fall to C1. pymsym 0.3.5 exposes no
   tolerance knob, so the probe is applied from the other side -- random
   displacements of growing amplitude, reporting the amplitude at which sigma
   first changes.

Usage
-----
    python scripts/calibration/s0_A_symmetry_backtest.py
    python scripts/calibration/s0_A_symmetry_backtest.py --geometry-dir analysis/package1/.../xyz
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
from openqha import S0_ROOT, config  # noqa: E402

try:
    import pymsym
except ImportError:  # pragma: no cover - the message IS the useful output
    raise SystemExit(
        "pymsym is not installed.\n"
        "    pip install pymsym\n"
        "Source vendored at ../source-code/final-workflow-design/symmetry/"
        "pymsym-master.zip (MIT, Marcus Johansson / Corin Wagen).")

import numpy as np  # noqa: E402
from ase.data import atomic_numbers  # noqa: E402
from ase.io import read  # noqa: E402

#: Random-displacement amplitudes (Angstrom) used to probe how close a geometry
#: sits to a symmetry flip. modifiable_convention.
#:
#: WHY DISPLACEMENT AND NOT A TOLERANCE SWEEP: pymsym 0.3.5 exposes
#: `get_symmetry_number(*args) -> int` with NO tolerance argument (verified by
#: inspect.signature on 2026-09-03). An earlier version of this script passed
#: `tolerance=` anyway, caught the resulting TypeError, and reported the arm as
#: "no result" -- which the summary line then filtered out, so the output LOOKED
#: like a sweep that agreed at every tolerance. It had checked nothing.
#: That is exactly the failure mode of defects 4 and 8 (skills section 2.4(a):
#: a new criterion must first be shown an example it should fail).
#:
#: Perturbing the geometry measures the same thing from the other side: how far
#: this structure is from the point where the detector changes its mind.
DISPLACEMENTS_A = (0.0, 0.005, 0.02, 0.05, 0.15)
PERTURB_SEED = 20260903        # fixed so the sweep is reproducible


def detect(symbols, positions):
    """Return (point_group, sigma), or an ERROR tag and None if pymsym refuses."""
    z = [int(atomic_numbers[s]) for s in symbols]
    pos = [[float(c) for c in p] for p in positions]
    try:
        return pymsym.get_point_group(z, pos), int(pymsym.get_symmetry_number(z, pos))
    except Exception as exc:            # noqa: BLE001 - third-party failure is data
        return "ERROR:{}".format(type(exc).__name__), None


def supports_tolerance():
    """True only if pymsym actually accepts a tolerance argument.

    Reported in the product so nobody later assumes a sweep was done.
    """
    import inspect
    try:
        sig = inspect.signature(pymsym.get_symmetry_number)
    except (TypeError, ValueError):
        return False
    return "tolerance" in sig.parameters


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--geometry-dir", default=None,
                    help="directory of <qm9_index>.xyz; defaults to the shipped reference geometries")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = config.load()
    species = cfg["species"]
    geo_dir = Path(args.geometry_dir) if args.geometry_dir else (
        S0_ROOT / "data" / "reference-geometries")

    print("=" * 100)
    print("Branch A -- pymsym back-test against the hand-declared symmetry numbers")
    print("=" * 100)
    print("pymsym      {}".format(getattr(pymsym, "__version__", "(no __version__)")))
    print("geometries  {}".format(geo_dir))
    print("criterion   every declared species must match; a mismatch means pymsym")
    print("            may NOT be used to fill in the undeclared molecules.")
    print()

    rows = []
    for qm9_index, meta in species.items():
        xyz = geo_dir / "{}.xyz".format(qm9_index)
        if not xyz.exists():
            print("  missing geometry: {}".format(xyz))
            continue
        atoms = read(str(xyz))
        symbols = atoms.get_chemical_symbols()
        declared = int(meta["symmetry_number"])
        pg, sigma = detect(symbols, atoms.get_positions())
        ok = sigma == declared

        rng = np.random.default_rng(PERTURB_SEED)
        sweep = {}
        for amp in DISPLACEMENTS_A:
            pos = atoms.get_positions()
            if amp > 0.0:
                pos = pos + rng.normal(scale=amp, size=pos.shape)
            p_pg, p_sigma = detect(symbols, pos)
            sweep["{:g}".format(amp)] = {"point_group": p_pg, "sigma": p_sigma}

        # The flip amplitude is the smallest displacement at which sigma changes.
        flip_at = None
        for amp in DISPLACEMENTS_A:
            if amp > 0.0 and sweep["{:g}".format(amp)]["sigma"] != sigma:
                flip_at = amp
                break

        rows.append({
            "qm9_index": qm9_index,
            "name": meta["name"],
            "declared_sigma": declared,
            "declared_reason": meta.get("symmetry_reason", "").strip(),
            "pymsym_point_group": pg,
            "pymsym_sigma": sigma,
            "match": ok,
            "displacement_sweep_A": sweep,
            "sigma_flips_at_A": flip_at,
        })
        print("  {:18s} {:22s} declared={}  pymsym={} ({})  {:12s} sigma flips at {}".format(
            qm9_index, meta["name"], declared, sigma, pg,
            "MATCH" if ok else "**MISMATCH**",
            "{} A".format(flip_at) if flip_at is not None
            else "no flip up to {} A".format(max(DISPLACEMENTS_A))))

    n_match = sum(1 for r in rows if r["match"])
    verdict = (n_match == len(rows)) and len(rows) > 0
    print()
    print("-" * 100)
    print("matched {} / {}   ->   {}".format(
        n_match, len(rows),
        "PASS: pymsym may be used for the undeclared molecules"
        if verdict else
        "FAIL: pymsym must NOT be used to fill in sigma"))

    out = Path(args.out) if args.out else (S0_ROOT / "analysis" / "symmetry_backtest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_by": "scripts/calibration/s0_A_symmetry_backtest.py",
        "pymsym_version": getattr(pymsym, "__version__", None),
        "geometry_dir": str(geo_dir),
        "pymsym_exposes_tolerance_arg": supports_tolerance(),
        "displacement_amplitudes_A": list(DISPLACEMENTS_A),
        "perturb_seed": PERTURB_SEED,
        "n_species": len(rows),
        "n_match": n_match,
        "verdict_pass": verdict,
        "criterion": ("Every species with a hand-declared sigma must be reproduced. "
                      "These seven are the only independently known answers we have "
                      "(D0-9, plan_A section 2.5.3)."),
        "rows": rows,
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("written {}".format(out))


if __name__ == "__main__":
    main()
