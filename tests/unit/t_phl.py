"""Ticket 11 of the Hessian-learning set: Algorithm 1 and the exact projected loss
(`openqha.training.phl`) against T03's numbers on the propanal fixture -- no engine.

H_r = the ORCA wB97M-D3BJ/def2-TZVPPD Hessian of basin 0 (`job.hess`), H_theta = the
stored MACE-OFF23_medium Hessian at the same geometry (`hessian_at_<level>.npy`).
Asserted: P is symmetric, idempotent and kills V; the exact loss equals
`hessian_compare` family 4 (eq. 2: D_ii = OMEGA_ALONG_REF_CM^2, MIXING) to 1e-14; Weyl
(0.3427 <= 0.3697 <= 0.5448); `probe = modes` and `probe = cartesian` reproduce the exact
loss to 1e-14 (A2); 4000 Rademacher draws give the mean within 1 % and the variance
within 5 % of eq. 8, the Gaussian variance being larger (A1); a rigid-block perturbation
of H_r changes nothing (A3); H_r := H_theta gives 0 (A5); H_r := 0.81 H_theta gives
0.19^2 ||K~||_F^2 / n_vib (A6); the entropy weights equal the analytic harmonic
derivative to 4 digits at 131 cm^-1 and eq. 3 holds to 1e-16; the weighted loss is ~1e-5
against the unweighted ~1e-2.
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

    # --- projector and reference modes -----------------------------------------------------
    P, inv_m, V = phl.projector(masses, x_r)
    check("P symmetric, idempotent, P V = 0 (1e-12)",
          np.abs(P - P.T).max() < 1e-12 and np.abs(P @ P - P).max() < 1e-12 and np.abs(P @ V).max() < 1e-12)
    L_r, lam_r = phl.reference_modes(H_r, masses, x_r)
    n_vib = L_r.shape[1]
    pr = hc.projected(H_r, masses, x_r)
    check("reference modes = hessian_compare.projected (n_vib {}, eigenvalues to 1e-12)".format(n_vib),
          n_vib == 24 and np.abs(lam_r - pr["lam"]).max() < 1e-12 and np.abs(np.abs(np.sum(L_r * pr["vec"], axis=0)) - 1).max() < 1e-10)
    check("L_r L_r^T = P (1e-12)", np.abs(L_r @ L_r.T - P).max() < 1e-12)

    # --- eq. 1 and eq. 2 = family 4 -----------------------------------------------------------
    L_full = phl.projected_loss_full(H_t, H_r, masses, x_r)
    D, lam = phl.mode_basis_terms(H_t, H_r, masses, x_r)
    rhs = (np.sum((np.diag(D) - lam) ** 2) + np.sum((D - np.diag(np.diag(D))) ** 2)) / n_vib
    check("eq. 2: ||A||_F^2/n_vib = [sum (D_ii - lam_i)^2 + sum_{i!=j} D_ij^2]/n_vib to 1e-14", abs(L_full - rhs) < 1e-14,
          abs(L_full - rhs))
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

    # --- A2: deterministic probe sets are exact ---------------------------------------------
    for mode in ("modes", "cartesian"):
        vt, r, info = phl.make_probes(masses, x_r, H_r, mode=mode)
        est = phl.estimator_from_products(vt @ H_t, r, info)
        check("A2 probe={} (k={}) reproduces the exact loss to 1e-14".format(mode, info["k"]),
              abs(est - L_full) < 1e-14 and info["k"] == (n_vib if mode == "modes" else n3), abs(est - L_full))

    # --- A1: Hutchinson -------------------------------------------------------------------------
    rng = np.random.default_rng(7)
    draws = {}
    for mode in ("rademacher", "gaussian"):
        vals = []
        for _ in range(4000):
            vt, r, info = phl.make_probes(masses, x_r, H_r, mode=mode, k=1, rng=rng)
            vals.append(phl.estimator_from_products(vt @ H_t, r, info))
        draws[mode] = np.array(vals)
    var = phl.estimator_variance(H_t, H_r, masses, x_r, k=1)
    rad, gau = draws["rademacher"], draws["gaussian"]
    check("A1 Rademacher: mean within 1 % of the exact loss over 4000 draws ({:.2%})".format(abs(rad.mean() / L_full - 1)),
          abs(rad.mean() / L_full - 1) < 0.01)
    check("A1 Rademacher: variance within 5 % of eq. 8 ({:.3e} vs {:.3e})".format(rad.var(), var["rademacher"]),
          abs(rad.var() / var["rademacher"] - 1) < 0.05)
    check("Gaussian: mean within 3 %, variance larger than Rademacher's (eq. 8's remark)",
          abs(gau.mean() / L_full - 1) < 0.03 and var["gaussian"] > var["rademacher"] and gau.var() > rad.var())
    vt, r, info = phl.make_probes(masses, x_r, H_r, mode="rademacher", k=4, rng=np.random.default_rng(1))
    check("k = 4 probes -> [4, 3N] and r = P M^-1/2 H_r v~ by definition",
          vt.shape == (4, n3) and np.abs(r - ((vt @ H_r) * inv_m[None, :]) @ P).max() < 1e-9)   # P vs L_r L_r^T: 1e-13 x |H_r|

    # --- A3, A5, A6 ---------------------------------------------------------------------------------
    m3, _ = phl.mass_vectors(masses)
    rigid = np.sqrt(np.outer(m3, m3)) * (V @ V.T) * 5.0                       # eps sqrt(m m^T) V V^T
    check("A3: a rigid-block perturbation of H_r changes the loss by 0 (1e-14)",
          abs(phl.projected_loss_full(H_t, H_r + rigid, masses, x_r) - L_full) < 1e-14)
    check("A5: H_r := H_theta -> 0", phl.projected_loss_full(H_t, H_t, masses, x_r) < 1e-20)
    K_t = P @ phl.mass_weighted(H_t, masses) @ P
    a6 = 0.19 ** 2 * np.sum(K_t * K_t) / n_vib
    check("A6: H_r := 0.81 H_theta -> 0.19^2 ||K~||_F^2 / n_vib (1e-12 relative)",
          abs(phl.projected_loss_full(H_t, 0.81 * H_t, masses, x_r) / a6 - 1) < 1e-12)

    # --- eq. 3: entropy weights ----------------------------------------------------------------------
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
    check("eq. 3: ||A_W||_F^2/n_vib = sum w_i w_j (D_ij - lam_i d_ij)^2 / n_vib (1e-16)", abs(L_w - rhs_w) < 1e-16, abs(L_w - rhs_w))
    check("weighted loss ~1e-5 against unweighted ~1e-2 (the low modes carry it)", 1e-6 < L_w < 1e-4 and 1e-3 < L_full < 1e-1, (L_w, L_full))
    vt, r, info = phl.make_probes(masses, x_r, H_r, mode="modes", weights=w)
    check("A2 with weights: probe=modes reproduces eq. 3 to 1e-14",
          abs(phl.estimator_from_products(vt @ H_t, r, info) - L_w) < 1e-14)
    try:
        phl.make_probes(masses, x_r, H_r, mode="hutchinson")
        check("an unknown probe mode is refused", False)
    except ValueError:
        check("an unknown probe mode is refused", True)

    print("\n{} checks, {} failed".format(21, len(FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
