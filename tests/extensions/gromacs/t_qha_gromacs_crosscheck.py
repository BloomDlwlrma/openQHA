"""Branch B acceptance criterion 2: an independent implementation reproduces the spectrum.

INTEGRATION. Needs a GROMACS binary on PATH (or `S0_GMX_BIN`). No potential is called: the
trajectory is the analytic harmonic one, so this test costs a few seconds and its answer is
known in closed form.

What is being cross-checked, and what is not
--------------------------------------------
`gmx anaeig -entropy` was meant to be the independent implementation of the whole chain
(plan_B revision C). It cannot be. Measured on 2026-09-03 and confirmed against upstream
source, it refuses mass-weighted eigenvalues, uses a formula that expects them anyway, and
drops the six SOFTEST modes rather than the six rigid ones -- a factor of 452 on a case
whose answer is known. `openqha/gmx_io.py:anaeig` carries the three findings.

So criterion 2 splits, and this test covers the half that GROMACS can still answer:
`gmx covar -mwa` is an independent implementation of the superposition, the mass weighting
and the diagonalisation -- the part that could plausibly be wrong. The entropy sum on top
is closed form and is checked to machine precision by `t_qha_harmonic.py`. Between the two,
every step has an independent check; no single command covers the whole chain, and this
test says so rather than implying otherwise by passing.

The comparison is made fair before it is made
----------------------------------------------
`gmx covar` fits every frame once, to the structure in `-s`; `openqha/qha.py` iterates the
fit to the mean. That difference alone moves T*S by about 0.009 kcal/mol, more than
anything else here, so `cross_check` hands GROMACS our converged mean as `-s` and then
reports separately how large the effect it removed was. A difference that has been arranged
away still has to be visible.
"""
import sys
from pathlib import Path


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


sys.path.insert(0, str(_repo_root()))
import numpy as np                                          # noqa: E402
from openqha import gmx_io, hessian, qha                    # noqa: E402

#: Criterion 2's budget on T*S, kcal/mol.
BUDGET_KCAL = 0.05
#: The eigenvalues themselves are compared as well, because a T*S agreement built out of
#: two cancelling eigenvalue errors would pass a test on T*S alone.
EIGENVALUE_BUDGET_RELATIVE = 1e-2

N_FRAMES = 20000
SEED = 11


def main():
    # The capability contract (openqha/capabilities.py): locally a missing extension is a
    # SKIP, and a run that DECLARED it via S0_REQUIRE_CAPS has already failed in
    # check_declared() before reaching here. So a skip below is always a local skip and
    # never a job quietly going green.
    from openqha import capabilities
    if not capabilities.available("gromacs"):
        print("SKIP: {}".format(capabilities.missing_extension_note("gromacs")))
        print("      install with: conda install -c conda-forge gromacs")
        print("      to make this an ERROR instead: S0_REQUIRE_CAPS=gromacs")
        return 0

    try:
        exe = gmx_io.gmx_binary()
    except RuntimeError as exc:
        print("SKIP: {}".format(str(exc).splitlines()[0]))
        return 0

    masses = np.array([12.011, 15.999, 1.008, 1.008, 1.008])
    symbols = ["C", "O", "H", "H", "H"]
    positions = np.array([[0.0, 0.0, 0.0], [1.42, 0.0, 0.0], [-0.5, 1.0, 0.1],
                          [-0.5, -0.6, 0.9], [1.9, 0.9, -0.2]])
    rng = np.random.RandomState(7)
    n = 3 * len(masses)
    v, _s, rank = hessian.rigid_body_vectors(masses, positions)
    p = np.eye(n) - v @ v.T
    a = rng.normal(size=(n, n))
    a = p @ (a + a.T) / 2 @ p
    _w, u = np.linalg.eigh(a)
    keep = np.argsort(np.linalg.norm(v.T @ u, axis=0) ** 2)[:n - rank]
    u = u[:, keep]
    lam = (np.linspace(120.0, 3200.0, n - rank) / qha.CM_INV_PER_SQRT_EV_A2_AMU) ** 2
    m3 = np.repeat(masses, 3)
    h = (u @ np.diag(lam) @ u.T) * np.sqrt(np.outer(m3, m3))
    frames, ref = qha.synthetic_harmonic_trajectory(h, masses, positions,
                                                    n_frames=N_FRAMES, seed=SEED)

    import tempfile
    with tempfile.TemporaryDirectory(prefix="openqha_gmx_") as tmp:
        out = gmx_io.cross_check(frames, symbols, masses, tmp)

    print("=" * 92)
    print("Branch B criterion 2 -- gmx covar -mwa against openqha/qha.py")
    print("=" * 92)
    print("binary       {}".format(exe))
    print("gmx          {}   precision {}".format(out["gmx_version"].get("version"),
                                                  out["gmx_version"].get("precision")))
    print("ours         {}".format(out["our_precision"]))
    print("frames       {}   eigenvalues {} (gmx) / {} (ours)".format(
        out["n_frames"], out["n_eigenvalues_gmx"], out["n_eigenvalues_ours"]))
    print("reference    {}".format(out["reference_structure_for_fit"]))
    print()
    print("closed-form T*S              {:.6f} kcal/mol".format(
        ref["closed_form_TS_kcal"]))
    print("ours                         {:.6f}".format(out["TS_QH_ours_kcal"]))
    print("from gmx's eigenvalues       {:.6f}".format(
        out["TS_QH_from_gmx_eigenvalues_kcal"]))
    print("difference                   {:+.3e}".format(out["TS_QH_difference_kcal"]))
    print("eigenvalue max rel. diff     {:.3e}".format(
        out["eigenvalue_max_relative_difference_vs_unprojected"]))
    print("eigenvalue sum rel. diff     {:.3e}".format(
        out["eigenvalue_sum_relative_difference"]))
    fp = out["fit_protocol_difference"]
    print("fit protocol (single vs iterated fit)  {:+.6f} kcal/mol".format(
        fp["TS_QH_single_fit_kcal"] - fp["TS_QH_iterated_fit_kcal"]))
    pe = out["projection_effect"]
    print("Eckart projection                      {:+.3e} kcal/mol".format(
        pe["TS_QH_projected_kcal"] - pe["TS_QH_unprojected_kcal"]))
    ev = out["anaeig_evidence"]
    print()
    print("gmx anaeig -entropy (EVIDENCE, is_reference = {}):".format(ev["is_reference"]))
    print("   Schlitter it printed  {} J/(mol K)".format(
        ev["parsed"]["S_Schlitter_J_per_mol_K"]))
    print("   physically correct    {:.4f} J/(mol K)".format(
        out["TS_Schlitter_ours_kcal"] / out["temperature_K"] * 4184.0))
    print()

    checks = [
        ("criterion 2: T*S from gmx covar -mwa's spectrum agrees with ours to better "
         "than {} kcal/mol".format(BUDGET_KCAL),
         "{:+.3e} kcal/mol".format(out["TS_QH_difference_kcal"]),
         abs(out["TS_QH_difference_kcal"]) < BUDGET_KCAL),
        ("the eigenvalues agree individually, not just in the sum -- two cancelling "
         "errors would pass a test on T*S alone",
         "{:.3e} relative".format(
             out["eigenvalue_max_relative_difference_vs_unprojected"]),
         out["eigenvalue_max_relative_difference_vs_unprojected"]
         < EIGENVALUE_BUDGET_RELATIVE),
        ("the Schlitter estimator agrees too, on the same spectrum",
         "{:+.3e} kcal/mol".format(out["TS_Schlitter_difference_kcal"]),
         abs(out["TS_Schlitter_difference_kcal"]) < BUDGET_KCAL),
        ("gmx anaeig -entropy is recorded as evidence and NOT as a reference value",
         "is_reference = {}".format(ev["is_reference"]),
         ev["is_reference"] is False),
        ("the precision of both sides is reported, so ruling E1 (measure the mixed vs "
         "float64 difference rather than building a double-precision GROMACS) has "
         "something to stand on",
         "{} vs {}".format(out["gmx_version"].get("precision"), out["our_precision"]),
         bool(out["gmx_version"].get("precision") and out["our_precision"])),
    ]

    bad = 0
    for text, measured, ok in checks:
        print("[{}] {}".format("PASS" if ok else "FAIL", text))
        print("       measured: {}".format(measured))
        bad += 0 if ok else 1
    print()
    print("{} of {} checks failed".format(bad, len(checks)))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
