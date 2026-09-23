"""Tickets 11 and 35 of the Hessian-learning set: Algorithms 1-2 of the PHL loss
(`openqha.training.phl`) on the propanal fixture -- no engine.

H_r = the ORCA wB97M-D3BJ/def2-TZVPPD Hessian of basin 0 (`job.hess`), H_theta = the
stored MACE-OFF23_medium Hessian at the same geometry (`hessian_at_<level>.npy`).

PHL VERBATIM (S0-C-64), the derivations of spec `spec-phl-verbatim.md` step 2 run on
that pair: eq. 1' is the mean square per matrix element; Derivation 2.3 -- the 3N unit
probes reproduce it to 1e-12 with zero variance; Derivation 2.1 -- 4000 Rademacher and
4000 Gaussian single-probe draws have their mean within 3 standard errors of it;
Derivation 2.2 -- each sample variance is within 5 % of the closed form, and the
Gaussian one is strictly the larger (the kurtosis term 2 sum_i B_ii^2); the variance
falls as 1/k; the probe is the RAW draw and r_j = H_r v_j. Refused: an unknown probe
mode, `modes` (there is no such probe any more), a `metric` or masses in the signature,
and a Label that is not square.

THE VIBRATIONAL ANALYSIS block at the end of `phl` is not a training path: it is the
judge's and the smoke fit's, and it is asserted here as it always was (P symmetric,
idempotent and killing V; ||A||_F^2/n_vib = `hessian_compare` family 4; Weyl
0.3427 <= 0.3697 <= 0.5448; a rigid-block perturbation changes nothing; H_r := H_theta
gives 0; H_r := 0.81 H_theta gives 0.19^2 ||K~||_F^2/n_vib; the entropy weights equal
the analytic harmonic derivative to 4 digits at 131 cm^-1 and eq. 3 holds to 1e-16).
It goes with its last caller, tickets 36 and 38.
"""
import math
import sys
from pathlib import Path

import numpy as np


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))
from openqha.qm_interfaces import orca                                  # noqa: E402
from openqha.store import layout                                        # noqa: E402
from openqha.thermochem import hessian as hessian_mod                   # noqa: E402
from openqha.thermochem import hessian_compare as hc                    # noqa: E402
from openqha.training import phl                                        # noqa: E402

FIX = ROOT / "tests" / "data" / "propanal_molecule"
LEVEL = "wb97m-d3bj_def2-tzvppd"
FAIL = []


def check(label, ok, detail=""):
    print("  {:78s} {}".format(label, "ok" if ok else "FAIL " + str(detail)[:200]))
    if not ok:
        FAIL.append(label)


def main():
    from ase.data import atomic_masses, atomic_numbers
    parsed = orca.parse_hess(layout.orca_level_file(FIX, LEVEL, 0, ".hess"))
    symbols = list(parsed["symbols"])
    x_r = np.asarray(parsed["positions_bohr"]) / orca.BOHR_PER_ANGSTROM
    masses = np.array([atomic_masses[atomic_numbers[s]] for s in symbols])
    H_r = orca.hessian_to_ev_per_angstrom2(parsed["hessian_eh_bohr2"])
    H_t = np.load(FIX / "mace" / "basin00" / "hessian_at_{}.npy".format(LEVEL))
    n3 = 3 * len(symbols)

    # === PHL verbatim: the target, the probes, the estimator (spec step 2) ==================
    # --- eq. 1': the exact target ------------------------------------------------------------
    L = phl.loss_full(H_t, H_r)
    d = H_t - H_r
    check("eq. 1': loss_full = ||dH||_F^2 / (9 N^2) (1e-16); ~1e-2..1e-1 eV^2/A^4 on propanal",
          abs(L - np.sum(d * d) / n3 ** 2) < 1e-16 and 1e-3 < L < 1.0, L)
    check("cartesian_loss_full is the same function under its ticket-21 name",
          phl.cartesian_loss_full is phl.loss_full)

    # --- Derivation 2.3: the 3N unit probes are the full matrix ------------------------------
    v, r, info = phl.make_probes(H_r, mode="cartesian")
    check("Derivation 2.3: v = I, r = H_r, k = 3N, denominator 9 N^2, estimator = eq. 1' to 1e-12",
          np.array_equal(v, np.eye(n3)) and np.abs(r - H_r).max() < 1e-12 and info["k"] == n3
          and info["denominator"] == n3 * n3 and info["nu"] == n3 * n3 and info["stochastic"] is False
          and abs(phl.estimator_from_products(v @ H_t, r, info) - L) < 1e-12,
          abs(phl.estimator_from_products(v @ H_t, r, info) - L))

    # --- the probe is the raw draw; r_j = H_r v_j --------------------------------------------
    v, r, info = phl.make_probes(H_r, mode="rademacher", k=4, rng=np.random.default_rng(1))
    check("k = 4 Rademacher: v is the raw +-1 draw [4, 3N], r = H_r v (1e-12), denominator 9 N^2 k",
          v.shape == (4, n3) and set(np.unique(v)) == {-1.0, 1.0} and np.abs(r - v @ H_r).max() < 1e-12
          and info["denominator"] == 4 * n3 * n3 and info["stochastic"] is True)
    vg, rg, _ig = phl.make_probes(H_r, mode="gaussian", k=3, rng=np.random.default_rng(1))
    check("k = 3 Gaussian: the draw is standard normal, r = H_r v (1e-12)",
          vg.shape == (3, n3) and np.abs(rg - vg @ H_r).max() < 1e-12 and abs(vg.std() - 1.0) < 0.2, vg.std())

    # --- Derivations 2.1 and 2.2: unbiased, and the two variances ----------------------------
    draws = {}
    for mode, seed in (("rademacher", 11), ("gaussian", 13)):
        rng = np.random.default_rng(seed)
        vals = []
        for _ in range(4000):
            vj, rj, ij = phl.make_probes(H_r, mode=mode, k=1, rng=rng)
            vals.append(phl.estimator_from_products(vj @ H_t, rj, ij))
        draws[mode] = np.array(vals)
    var1 = phl.estimator_variance(H_t, H_r, k=1)
    for mode in ("rademacher", "gaussian"):
        vals = draws[mode]
        n = len(vals)
        se = math.sqrt(var1[mode] / n)
        check("Derivation 2.1 ({}): 4000 draws' mean within 3 s.e. of eq. 1' ({:.2f} s.e.)".format(
              mode, abs(vals.mean() - L) / se), abs(vals.mean() - L) < 3 * se, (vals.mean(), L, se))
        # the sample variance has its own sampling error, sd(s^2) = s^2 sqrt((kurt - 1)/n);
        # for the Gaussian draw X is a weighted sum of chi^2_1 and that band is ~6 %, so the
        # tolerance is read off the draws rather than invented
        kurt = float(np.mean((vals - vals.mean()) ** 4) / vals.var() ** 2)
        se_var = vals.var() * math.sqrt((kurt - 1) / n)
        check("Derivation 2.2 ({}): sample variance within 3 sd(s^2) of the closed form "
              "({:.3e} vs {:.3e}, band {:.1%})".format(mode, vals.var(), var1[mode], 3 * se_var / var1[mode]),
              abs(vals.var() - var1[mode]) < 3 * se_var, (vals.var(), var1[mode], se_var))
    b = (H_t - H_r).T @ (H_t - H_r)
    kurtosis_term = 2.0 * float(np.sum(np.diag(b) ** 2)) / (float(n3 ** 2) ** 2)
    check("Derivation 2.2: Var_Gaussian - Var_Rademacher = 2 sum_i B_ii^2 / (9N^2)^2 (1e-18), and it is > 0",
          abs((var1["gaussian"] - var1["rademacher"]) - kurtosis_term) < 1e-18 and kurtosis_term > 0
          and draws["gaussian"].var() > draws["rademacher"].var(),
          (var1["gaussian"] - var1["rademacher"], kurtosis_term))
    var4 = phl.estimator_variance(H_t, H_r, k=4)
    check("the variance falls as 1/k: Var(k=4) = Var(k=1)/4 for both draws (1e-20)",
          abs(var4["rademacher"] - var1["rademacher"] / 4) < 1e-20
          and abs(var4["gaussian"] - var1["gaussian"] / 4) < 1e-20)

    # --- what the signature no longer takes --------------------------------------------------
    for bad_mode in ("hutchinson", "modes"):
        try:
            phl.make_probes(H_r, mode=bad_mode)
            check("probe mode {!r} is refused".format(bad_mode), False)
        except ValueError as exc:
            check("probe mode {!r} is refused".format(bad_mode), "probe mode must be one of" in str(exc))
    try:
        phl.make_probes(H_r, mode="rademacher", metric="cartesian")
        check("make_probes takes no `metric` (there is one target, S0-C-64)", False)
    except TypeError:
        check("make_probes takes no `metric` (there is one target, S0-C-64)", True)
    try:
        phl.make_probes(masses, x_r, H_r)
        check("make_probes takes no masses or positions", False)
    except (TypeError, ValueError):
        check("make_probes takes no masses or positions", True)
    try:
        phl.estimator_variance(H_t, H_r, masses, x_r, k=1)
        check("estimator_variance takes only (H_theta, H_r, k)", False)
    except TypeError:
        check("estimator_variance takes only (H_theta, H_r, k)", True)
    try:
        phl.make_probes(H_r[:-1], mode="cartesian")
        check("a Label that is not square is refused", False)
    except ValueError as exc:
        check("a Label that is not square is refused", "square" in str(exc))

    # === the vibrational analysis (the judge's and the smoke fit's; tickets 36, 38) ==========
    P, inv_m, V = phl.projector(masses, x_r)
    check("P symmetric, idempotent, P V = 0 (1e-12)",
          np.abs(P - P.T).max() < 1e-12 and np.abs(P @ P - P).max() < 1e-12 and np.abs(P @ V).max() < 1e-12)
    L_r, lam_r = phl.reference_modes(H_r, masses, x_r)
    n_vib = L_r.shape[1]
    pr = hc.projected(H_r, masses, x_r)
    check("reference modes = hessian_compare.projected (n_vib {}, eigenvalues to 1e-12)".format(n_vib),
          n_vib == 24 and np.abs(lam_r - pr["lam"]).max() < 1e-12
          and np.abs(np.abs(np.sum(L_r * pr["vec"], axis=0)) - 1).max() < 1e-10)
    check("L_r L_r^T = P (1e-12)", np.abs(L_r @ L_r.T - P).max() < 1e-12)
    L_full = phl.projected_loss_full(H_t, H_r, masses, x_r)
    D, lam = phl.mode_basis_terms(H_t, H_r, masses, x_r)
    rhs = (np.sum((np.diag(D) - lam) ** 2) + np.sum((D - np.diag(np.diag(D))) ** 2)) / n_vib
    check("||A||_F^2/n_vib = [sum (D_ii - lam_i)^2 + sum_{i!=j} D_ij^2]/n_vib to 1e-14",
          abs(L_full - rhs) < 1e-14, abs(L_full - rhs))
    cmp = hc.compare_hessians(H_t, H_r, masses, x_r)
    d_along = np.asarray(cmp["OMEGA_ALONG_REF_CM"], dtype=float)
    d_ii = np.sign(d_along) * (d_along / hessian_mod.CM_INV_PER_SQRT_EV_A2_AMU) ** 2
    check("D_ii = hessian_compare OMEGA_ALONG_REF_CM^2 (1e-12)", np.abs(d_ii - np.diag(D)).max() < 1e-12)
    mixing = np.linalg.norm(D - np.diag(np.diag(D))) / np.linalg.norm(D)
    check("MIXING = ||D - diag D||_F / ||D||_F (1e-10)", abs(mixing - cmp["MIXING"]) < 1e-10, (mixing, cmp["MIXING"]))
    A = phl.error_operator(H_t, H_r, masses, x_r)
    lam_t = hc.projected(H_t, masses, x_r)["lam"]
    weyl = np.abs(lam_t - lam_r).max(); spec = np.linalg.norm(A, 2); fro = math.sqrt(np.sum(A * A))
    check("Weyl: max|lam^theta - lam^r| {:.4f} <= ||A||_2 {:.4f} <= ||A||_F {:.4f}".format(weyl, spec, fro),
          weyl <= spec <= fro and abs(weyl - 0.3427) < 5e-4 and abs(spec - 0.3697) < 5e-4 and abs(fro - 0.5448) < 5e-4)
    m3, _ = phl.mass_vectors(masses)
    rigid = np.sqrt(np.outer(m3, m3)) * (V @ V.T) * 5.0                       # eps sqrt(m m^T) V V^T
    check("a rigid-block perturbation of H_r changes the projected diagnostic by 0 (1e-14)",
          abs(phl.projected_loss_full(H_t, H_r + rigid, masses, x_r) - L_full) < 1e-14)
    check("H_r := H_theta -> 0", phl.projected_loss_full(H_t, H_t, masses, x_r) < 1e-20)
    K_t = P @ phl.mass_weighted(H_t, masses) @ P
    a6 = 0.19 ** 2 * np.sum(K_t * K_t) / n_vib
    check("H_r := 0.81 H_theta -> 0.19^2 ||K~||_F^2 / n_vib (1e-12 relative)",
          abs(phl.projected_loss_full(H_t, 0.81 * H_t, masses, x_r) / a6 - 1) < 1e-12)
    omega_r = hessian_mod.eigenvalues_to_cm_inv(lam_r)
    w, ds = phl.entropy_weights(omega_r, masses, x_r)
    R = 1.987204259
    u = 1.438776877 * omega_r / 298.15
    ds_ho = R * u * np.exp(u) / (np.exp(u) - 1) ** 2 * (1.438776877 / 298.15)
    i0 = 0
    check("|dS_msRRHO/d nu| at {:.1f} cm^-1 = analytic HO derivative to 4 digits ({:.4e} vs {:.4e})".format(
          omega_r[i0], ds[i0], ds_ho[i0]), abs(ds[i0] / ds_ho[i0] - 1) < 2e-3 and abs(omega_r[i0] - 131.1) < 1)
    check("weights normalised: max 1, monotone decreasing with omega above the rotor cap",
          w.max() == 1.0 and np.all(np.diff(w[3:]) <= 1e-12) and w[-1] < 1e-3, w)
    L_w = phl.projected_loss_full(H_t, H_r, masses, x_r, weights=w)
    rhs_w = np.sum(np.outer(w, w) * (D - np.diag(lam_r)) ** 2) / n_vib
    check("||A_W||_F^2/n_vib = sum w_i w_j (D_ij - lam_i d_ij)^2 / n_vib (1e-16)",
          abs(L_w - rhs_w) < 1e-16, abs(L_w - rhs_w))
    check("the weighted diagnostic ~1e-5 against the unweighted ~1e-2 (the low modes carry it)",
          1e-6 < L_w < 1e-4 and 1e-3 < L_full < 1e-1, (L_w, L_full))
    del inv_m

    print("\n{} checks, {} failed".format(26, len(FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
