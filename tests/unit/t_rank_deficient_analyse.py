"""A trajectory with too few frames must yield a rank verdict, not an OverflowError.

UNIT. NumPy only, a synthetic 10-atom trajectory of 5 frames; under a second.

The defect (an113, 2026-09-13)
------------------------------
The first branch B chain to reach `collect` on a card did so with 1+5 ps test
trajectories: 5 frames each, 30 coordinates. A covariance estimated from T frames has at
most T-1 non-zero eigenvalues, so 20 of the 24 vibrational eigenvalues were zero -- but
zero only to rounding, so `lam > 0` let them through, nu = sqrt(kT/lambda) came out
around 1e15 cm^-1, x = h nu / kT around 1e17, and `math.expm1(x)` raised

    OverflowError: math range error

inside the entropy sum. The chain had a perfectly good verdict to give -- criterion 9,
"the number of non-zero eigenvalues equals 3N-6", fails and says why -- and crashed
instead of giving it.

What is asserted
----------------
  A. thermo's three per-mode functions are finite for any frequency, including the
     absurd ones, and continuous across the guard (a frozen mode: S = 0, E = A = ZPE).
  B. `qha.analyse` on 5 frames of a 10-atom molecule RETURNS, its entropies are finite,
     and its rank_check says frames_are_the_limit -- the sentence collect needed.
  C. The same molecule with plenty of frames (a harmonic sample, 3000 frames) reports
     full rank, so the deficient case is recognised as such and not as the norm.
"""
import math
import sys
from pathlib import Path


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))
import numpy as np                                          # noqa: E402
from openqha.quasi_harmonic import qha                      # noqa: E402
from openqha.thermochem import thermo                       # noqa: E402

FAIL = []
N_ATOMS = 10
MASSES = np.array([12.011, 12.011, 12.011, 15.999] + [1.008] * 6)
T_K = 298.15


def check_thermo():
    print("A. per-mode thermodynamics stay finite and continuous")
    for nu in (1.0, 100.0, 3000.0, 1e6, 1e15, 1e300):
        vals = (thermo.s_mode_kcal_per_K(nu, T_K), thermo.e_mode_kcal(nu, T_K),
                thermo.a_mode_kcal(nu, T_K))
        ok = all(math.isfinite(v) for v in vals)
        print("   {}  nu {:>8.3g} cm^-1   S {:.3e}  E {:.3e}  A {:.3e}".format(
            "ok  " if ok else "FAIL", nu, *vals))
        if not ok:
            FAIL.append("non-finite thermodynamics at nu={}".format(nu))
    # Continuity means: just past the guard, the guarded value equals what the full
    # formula gives AT THE SAME POINT (x ~ 700.0007 is still representable; expm1 only
    # overflows near 709.78). Comparing two different nu would measure the slope of the
    # zero-point term, ~4e-4 kcal/mol here, and call it a jump -- the first version did.
    nu_just_past = thermo.X_FROZEN * (1 + 1e-6) * thermo.KB_KCAL * T_K / thermo.HC_KCAL
    x = thermo.HC_KCAL * nu_just_past / (thermo.KB_KCAL * T_K)
    kt = thermo.KB_KCAL * T_K
    full = dict(
        s_mode_kcal_per_K=thermo.KB_KCAL * (x / math.expm1(x) - math.log1p(-math.exp(-x))),
        e_mode_kcal=kt * (0.5 * x + x / math.expm1(x)),
        a_mode_kcal=kt * (0.5 * x + math.log1p(-math.exp(-x))))
    for fn in (thermo.s_mode_kcal_per_K, thermo.e_mode_kcal, thermo.a_mode_kcal):
        got, want = fn(nu_just_past, T_K), full[fn.__name__]
        if abs(got - want) > 1e-12 * max(1.0, abs(want)):
            FAIL.append("{} jumps at the guard: guarded {} vs full formula {}".format(
                fn.__name__, got, want))
    print("   {}  continuous across X_FROZEN = {}".format(
        "ok  " if not any("jumps" in f for f in FAIL) else "FAIL", thermo.X_FROZEN))


def _fake_frames(n_frames, seed=20260913, amplitude=0.05):
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.2, (N_ATOMS, 3))
    return base[None] + rng.normal(0.0, amplitude, (n_frames, N_ATOMS, 3))


def check_deficient():
    print("\nB. 5 frames of a 10-atom molecule: a verdict, not an exception")
    try:
        rec = qha.analyse(_fake_frames(5), MASSES, T_K, meta=None)
    except Exception as exc:                                # noqa: BLE001
        FAIL.append("qha.analyse raised on 5 frames: {}: {}".format(type(exc).__name__, exc))
        print("   FAIL  raised {}: {}".format(type(exc).__name__, exc))
        return
    rc = rec["rank_check"]
    ent = rec["entropy"]
    finite = all(math.isfinite(float(ent[k])) for k in ("S_QH_kcal_per_K", "S_schlitter_kcal_per_K")
                 if k in ent)
    print("   frames {}  frame rank limit {}  3N-6 {}  nonzero eigenvalues {}  "
          "frames_are_the_limit {}".format(
              rc["n_frames"], rc["frame_rank_limit"], rc["degrees_of_freedom_3N_minus_6"],
              rc["n_nonzero_eigenvalues"], rc["frames_are_the_limit"]))
    print("   entropy keys finite: {}   rank_is_full: {}".format(finite, rc["rank_is_full"]))
    if not rc["frames_are_the_limit"]:
        FAIL.append("5 frames were not reported as the limiting factor")
    if rc["rank_is_full"]:
        FAIL.append("5 frames reported as full rank")
    if not finite:
        FAIL.append("entropy from a rank-deficient spectrum is not finite")


def check_sufficient():
    print("\nC. 3000 frames: full rank, so B is recognised as the exception")
    rec = qha.analyse(_fake_frames(3000), MASSES, T_K, meta=None)
    rc = rec["rank_check"]
    print("   nonzero eigenvalues {} of {}   rank_is_full {}   frames_are_the_limit {}".format(
        rc["n_nonzero_eigenvalues"], rc["degrees_of_freedom_3N_minus_6"],
        rc["rank_is_full"], rc["frames_are_the_limit"]))
    if not rc["rank_is_full"] or rc["frames_are_the_limit"]:
        FAIL.append("3000 frames did not give full rank")


def main():
    check_thermo()
    check_deficient()
    check_sufficient()
    print()
    if FAIL:
        print("{} problem(s):".format(len(FAIL)))
        for f in FAIL:
            print("  - " + f)
        return 1
    print("too few frames is reported as such, and nothing overflows on the way")
    return 0


if __name__ == "__main__":
    sys.exit(main())
