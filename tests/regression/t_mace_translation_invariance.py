"""Translation invariance: the patched path must be exact, and the install must be named.

REGRESSION. It needs an engine, so it is slow by this suite's standards (a few seconds).

Why this test was rewritten
---------------------------
Its previous version asserted that the installed MACE HAS an origin-anchored neighbour-list
defect, and failed loudly when that turned out not to be so. The assertion was wrong. The
defect is real, but it lives in the MACE **develop** tree vendored under stage 2, not in
the `mace_torch 0.3.16` that openQHA imports -- and nothing in the products could have
caught the mix-up, because provenance recorded a VERSION and both trees answer 0.3.x.

So the test no longer asserts which MACE is installed. It asserts something that is true
either way and that fails if either half of the record is wrong:

  1. the patched path is exactly translation invariant, in the energy AND in the edge list
     -- this is what production runs on and it must never fail;
  2. the patch is bound at all three call sites, including `mace.data.atomic_data`, which
     holds its own reference and is the one that actually runs;
  3. the SOURCE FINGERPRINT of the installed neighbour list agrees with a direct
     MEASUREMENT of whether it is defective. Either alone can be wrong; disagreeing is the
     signal that the record has drifted from the code.

Point 3 is the one that would have caught the original error. It fails if the source says
origin-anchored and the probe finds no defect, and it fails the other way too.

The install is REPORTED rather than required, with the consequence spelled out: on a
defective install the patch is load-bearing, on a clean one it is insurance against being
pointed at the develop tree (which stage 2 uses).
"""
import sys
from pathlib import Path


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


sys.path.insert(0, str(_repo_root()))
from openqha import config, engine, mace_patch              # noqa: E402

#: Shifts measured to be safe and unsafe on the ORIGIN-ANCHORED variant (see
#: openqha/mace_patch.py for the reproduction).
SAFE_A = -5.5
UNSAFE_A = -8.0
FAR_A = 50.0
#: Below this counts as "the energy did not change".
EXACT_KCAL = 1e-6


def main():
    from ase.io import read
    from ase.optimize import BFGS

    cfg = config.load()
    try:
        calc, name, prov = engine.calculator(device="cpu")
    except Exception as exc:
        print("SKIP: no engine available ({}: {})".format(type(exc).__name__, exc))
        return 0

    atoms = read(str(config.qm9_xyz("dsgdb9nsd_000018", cfg)))
    atoms.calc = calc
    BFGS(atoms, logfile=None).run(fmax=0.005, steps=300)
    positions = atoms.get_positions().copy()

    print("=" * 96)
    print("Regression -- a rigid translation must not change the energy")
    print("=" * 96)
    patch = prov.get("neighbour_list_patch") or {}
    installed = patch.get("installed") or mace_patch.installed_variant()
    print("engine        {}".format(name))
    print("patch applied {}   sites {}".format(patch.get("applied"),
                                               len(patch.get("sites") or [])))
    print("mace module   {}".format(installed.get("module_path")))
    print("  sizing      {}".format(installed.get("sizing")))
    print("  probe at {:+.1f} A: defect present = {}   ({} edges)".format(
        installed.get("probe_shift_A", 0.0), installed.get("defect_present"),
        installed.get("n_edges")))
    print()

    # ---- the patched path, which is what production uses -----------------------------
    rec = engine.translation_invariance(
        atoms, shifts_A=(0.0, SAFE_A, UNSAFE_A, FAR_A, -1000.0), name=name)
    print("{:>12} {:>24} {:>16}".format("shift/A", "dE/(kcal/mol)", "max|F|/(eV/A)"))
    for p in rec["points"]:
        print("{:>12.1f} {:>24.9f} {:>16.4f}".format(
            p["shift_A"], p["delta_energy_kcal"], p["max_force_eV_per_A"]))
    edges = mace_patch.verify(positions, shifts_A=(0.0, UNSAFE_A, FAR_A, -1000.0))
    print()
    print("neighbour list: {} edges, identical at every shift: {}".format(
        edges["points"][0]["n_edges"], edges["invariant"]))
    print()

    fingerprint_says_defective = "defective" in (installed.get("sizing") or "")
    fingerprint_known = (installed.get("sizing") or "unknown") != "unknown"
    measured_defective = installed.get("defect_present")

    checks = [
        ("PATCHED: the energy is unchanged at every shift, including {} A and {} A "
         "where the origin-anchored variant is wrong".format(UNSAFE_A, FAR_A),
         "max {:.2e} kcal/mol over 5 shifts".format(rec["max_abs_delta_energy_kcal"]),
         rec["max_abs_delta_energy_kcal"] < EXACT_KCAL),
        ("PATCHED: the neighbour list itself is identical at every shift, so the "
         "invariance is structural and not a cancellation in the energy",
         "{} edges, invariant = {}".format(edges["points"][0]["n_edges"],
                                           edges["invariant"]),
         bool(edges["invariant"])),
        ("the patch is actually applied, at every call site -- including "
         "mace.data.atomic_data, which holds its own reference and is the one that runs",
         "applied = {}, {} sites".format(patch.get("applied"),
                                         len(patch.get("sites") or [])),
         bool(patch.get("applied")) and len(patch.get("sites") or []) >= 3),
        ("the source fingerprint of the installed neighbour list AGREES with a direct "
         "measurement of it. This is the check that would have caught the wrong claim "
         "this test used to make",
         "sizing = {!r}, measured defect = {}".format(installed.get("sizing"),
                                                      measured_defective),
         bool(fingerprint_known and measured_defective is not None
              and fingerprint_says_defective == bool(measured_defective))),
    ]

    bad = 0
    for text, measured, ok in checks:
        print("[{}] {}".format("PASS" if ok else "FAIL", text))
        print("       measured: {}".format(measured))
        bad += 0 if ok else 1

    print()
    if measured_defective:
        print("NOTE: the installed MACE IS defective. openqha/mace_patch.py is "
              "load-bearing here and every product depends on it.")
    else:
        print("NOTE: the installed MACE is translation invariant on its own. The patch is "
              "insurance, not a repair: it costs one centroid subtraction per force call "
              "and it is what stops the defect returning if openQHA is ever pointed at "
              "the develop tree, which stage 2 uses.")
    print()
    print("{} of {} checks failed".format(bad, len(checks)))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
