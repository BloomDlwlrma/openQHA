"""Branch B acceptance criterion 3: the quasi-harmonic chain recovers the harmonic limit.

UNIT. Pure NumPy, no potential, no engine, a few seconds.

Why this test is the sharp one
------------------------------
For a classical harmonic oscillator in mass-weighted normal coordinates,
<q_k^2> = k_B T / omega_k^2. So the quasi-harmonic reading nu_k = sqrt(k_B T / lambda_k)
returns the Hessian frequency EXACTLY. That identity is the entire content of the
quasi-harmonic approximation, which means a synthetic harmonic trajectory has a known
closed-form entropy and the whole implementation -- superposition, mass weighting, rigid
subspace removal, diagonalisation, the entropy sum -- can be checked against it without a
single force evaluation.

There is a precedent for the scale: the same style of check on the retired
density-of-states chain came back at 0.0011 kcal/mol.

Two numbers, because they fail for different reasons
----------------------------------------------------
  * EXACT COVARIANCE feeds lambda_k = k_B T / omega_k^2 straight in, with no sampling.
    It isolates the algebra and must sit at machine precision. A failure here is a bug in
    `openqha/qha.py` and nothing else.
  * SAMPLED runs finite frames through the production path, so it carries the estimator's
    statistical error. This is the number criterion 3 is written on, budget 0.02 kcal/mol
    on T*S.

Along the way the same trajectory tests criteria 4 (the rigid modes separate), 6
(S_QH <= S_Schlitter, which is analytic) and 9 (rank == 3N-6).
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
from openqha import hessian, qha                            # noqa: E402

#: Criterion 3's budget on T*S, kcal/mol.
SAMPLED_BUDGET_KCAL = 0.02
#: The algebra has no sampling error, so it gets a machine-precision budget.
EXACT_BUDGET_KCAL = 1e-9
#: Criterion 4: the rigid eigenvalues must separate from the first vibrational one.
RIGID_SEPARATION_MIN = 1e6

N_FRAMES = 60000
SEED = 20260903


def build_hessian(masses, positions, target_cm, seed=7):
    """A Hessian with prescribed frequencies, positive definite on the internal subspace.

    Built by picking an arbitrary orthonormal basis of the internal subspace and assigning
    the wanted eigenvalues to it, rather than by taking a real molecule's Hessian: the
    point of this test is that the ANSWER is known exactly, and a synthetic spectrum spans
    the soft-to-stiff range (120 to 3200 cm^-1) that a real molecule would only sample
    accidentally. Free energy is most sensitive at the soft end, so it has to be covered.
    """
    rng = np.random.RandomState(seed)
    n = 3 * len(masses)
    v, _s, rank = hessian.rigid_body_vectors(masses, positions)
    p = np.eye(n) - v @ v.T
    a = rng.normal(size=(n, n))
    a = p @ (a + a.T) / 2 @ p
    _w, u = np.linalg.eigh(a)
    keep = np.argsort(np.linalg.norm(v.T @ u, axis=0) ** 2)[:n - rank]
    u = u[:, keep]
    lam = (np.asarray(target_cm, dtype=float) / qha.CM_INV_PER_SQRT_EV_A2_AMU) ** 2
    m3 = np.repeat(masses, 3)
    return (u @ np.diag(lam) @ u.T) * np.sqrt(np.outer(m3, m3)), rank


def main():
    masses = np.array([12.011, 15.999, 1.008, 1.008, 1.008])
    positions = np.array([[0.0, 0.0, 0.0], [1.42, 0.0, 0.0], [-0.5, 1.0, 0.1],
                          [-0.5, -0.6, 0.9], [1.9, 0.9, -0.2]])
    n_modes = 3 * len(masses) - 6
    target_cm = np.linspace(120.0, 3200.0, n_modes)
    h, rank = build_hessian(masses, positions, target_cm)

    out = qha.harmonic_limit_check(h, masses, positions, n_frames=N_FRAMES, seed=SEED)
    ref = out["reference"]
    spec = out["sampled"]["spectrum"]
    ent = out["sampled"]["entropy"]

    print("=" * 92)
    print("Branch B criterion 3 -- harmonic limit of the quasi-harmonic chain")
    print("=" * 92)
    print("atoms {}   rigid subspace rank {}   modes {}   frames {}".format(
        len(masses), rank, ref["n_modes"], N_FRAMES))
    print("frequencies put in   {:.1f} .. {:.1f} cm^-1".format(
        target_cm.min(), target_cm.max()))
    print("closed-form T*S      {:.8f} kcal/mol".format(ref["closed_form_TS_kcal"]))
    print("exact covariance     {:.8f} kcal/mol   error {:+.3e}".format(
        out["exact_covariance"]["TS_QH_kcal"], out["exact_covariance_TS_error_kcal"]))
    print("sampled              {:.8f} kcal/mol   error {:+.6f}".format(
        ent["TS_QH_kcal"], out["sampled_TS_error_kcal"]))
    print("frequency error      max {:.3f}   rms {:.3f} cm^-1".format(
        out["max_frequency_error_cm_inv"], out["rms_frequency_error_cm_inv"]))
    print("rigid modes removed  {} of rank {}   separation ratio {:.3e}".format(
        spec["n_rigid_modes_removed"], spec["rigid_subspace_rank"],
        spec["rigid_to_first_vibrational_ratio"]))
    print("non-zero eigenvalues {} of 3N-6 = {}".format(
        spec["n_nonzero_eigenvalues"], spec["expected_vibrational_modes"]))
    print("T*S_Schlitter        {:.8f} kcal/mol   (must be >= T*S_QH)".format(
        ent["TS_Schlitter_kcal"]))
    print()

    checks = [
        ("criterion 3, exact covariance: the algebra reproduces the closed-form entropy "
         "to machine precision",
         "{:+.3e} kcal/mol".format(out["exact_covariance_TS_error_kcal"]),
         abs(out["exact_covariance_TS_error_kcal"]) < EXACT_BUDGET_KCAL),
        ("criterion 3, sampled: the whole chain reproduces the closed-form entropy to "
         "better than {} kcal/mol on T*S".format(SAMPLED_BUDGET_KCAL),
         "{:+.6f} kcal/mol".format(out["sampled_TS_error_kcal"]),
         abs(out["sampled_TS_error_kcal"]) < SAMPLED_BUDGET_KCAL),
        ("criterion 4: exactly the rigid-body modes are removed",
         "{} removed, rank {}".format(spec["n_rigid_modes_removed"],
                                      spec["rigid_subspace_rank"]),
         spec["n_rigid_modes_removed"] == spec["rigid_subspace_rank"] == rank),
        ("criterion 4: the rigid eigenvalues separate from the first vibrational one by "
         "more than {:.0e}".format(RIGID_SEPARATION_MIN),
         "{:.3e}".format(spec["rigid_to_first_vibrational_ratio"]),
         spec["rigid_to_first_vibrational_ratio"] > RIGID_SEPARATION_MIN),
        ("criterion 6: S_QH <= S_Schlitter (analytic, so a failure is an implementation "
         "error and never a sampling one)",
         "difference {:+.6f} kcal/mol".format(ent["schlitter_minus_qh_kcal"]),
         ent["schlitter_bounds_qh"]),
        ("criterion 9: the number of non-zero eigenvalues equals 3N-6",
         "{} of {}".format(spec["n_nonzero_eigenvalues"],
                           spec["expected_vibrational_modes"]),
         spec["n_nonzero_eigenvalues"] == spec["expected_vibrational_modes"] == n_modes),
        ("every recovered frequency is within 15 cm^-1 of the one put in, at this "
         "trajectory length",
         "max {:.3f} cm^-1".format(out["max_frequency_error_cm_inv"]),
         out["max_frequency_error_cm_inv"] < 15.0),
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
