"""Algorithms 1 and 2 of the PHL loss: the frame's constants, the random probes, the
reference matvec and the exact Cartesian loss (Pengmei, Han, Liu et al., "Probing the
Hessian", eqs. 6 and 2.1').

PRODUCTION. Ticket 11 of the Hessian-learning set, reduced to PHL verbatim by ticket 35
(S0-C-64, 2026-09-23). numpy only; no mace import.

Everything here is per STRUCTURE and needs no autograd: it is what a data loader or a
loss computes from the Label `H_r` alone, before the model is asked for anything.
Derivations and the numbers the tests hold these functions to:
`docs/tutorials/T04_openQHA_Theory_Hessian_Learning.ipynb` sections 2 and 6
(Algorithms 1-2) and T05 section 2.

THE TARGET (S0-C-53, S0-C-64). The raw Cartesian matrix as the reference program wrote
it -- no mass weighting, no Eckart projection, no reference modes, nothing projected but
the random vector itself:

    L_H  = ||H_theta - H_r||_F^2 / (3N)^2                                         (1')
    v_j i.i.d., E[v] = 0, E[v v^T] = I ;   r_j = H_r v_j                          (2')
    L^(K) = sum_j ||H_theta v_j - r_j||^2 / (9 N^2 K)                             (6')

`L^(K)` is unbiased for every K (Derivation 2.1), its Rademacher variance is the
smallest of the unit-variance draws (Derivation 2.2), and the 3N unit probes make it
exact -- the deterministic limit of the estimator IS the full matrix (Derivation 2.3).

Units: `H` in eV/A^2. The Label is used AS STORED: never symmetrised, projected or
mass-weighted here (the fork's data-loading check refuses a non-symmetric or wrongly
sized Hessian -- that is a data check, not a transformation).

The block at the end of this module is the standard VIBRATIONAL ANALYSIS (mass
weighting, the Eckart projector, the reference modes, the msRRHO entropy weights). It
is no longer on any training path: what is left of it serves the smoke fit's diagnostic
and the judge's `loss_exact` row, and it goes with its last caller (tickets 36, 38).
"""
import numpy as np

from ..thermochem import hessian as hessian_mod
from ..thermochem import thermo

#: The three probe sets of `make_probes`. `rademacher` and `gaussian` are stochastic
#: (Hutchinson, eq. 6'); `cartesian` is the deterministic unit-vector set that makes the
#: estimator exact (Derivation 2.3) at k = 3N HVPs -- the cost of `get_hessian` itself.
PROBE_MODES = ("rademacher", "gaussian", "cartesian")

#: Finite-difference step for the entropy derivative, cm^-1 (the analysis block below).
ENTROPY_FD_STEP_CM = 0.5


def loss_full(hessian_theta, hessian_r):
    """The EXACT target, eq. 1': ||H_theta - H_r||_F^2 / (9 N^2) -- the mean squared
    error per matrix element, and the oracle every estimator here is held to."""
    d = np.asarray(hessian_theta, dtype=float) - np.asarray(hessian_r, dtype=float)
    n3 = d.shape[0]
    return float(np.sum(d * d)) / (n3 * n3)


#: The name ticket 21 gave `loss_full` while two targets existed; kept for one release
#: so that a caller written then still reads. Same function, same number.
cartesian_loss_full = loss_full


def make_probes(hessian_r, mode="rademacher", k=4, rng=None):
    """Algorithm 2: the probes the model sees and the reference side of each.

    Returns (v [k, 3N], r [k, 3N], info) with

        v_j           -- the probe, the raw draw: nothing is projected or weighted
        r_j = H_r v_j -- a matvec on the Label (PHL's `bmm(H_ref, v)`)

    and info = dict(n3, mode, k, stochastic, nu, denominator), so that the loss forms
    rho_j = H_theta v_j - r_j and returns sum_j ||rho_j||^2 / denominator. The
    denominator is `nu k` = 9 N^2 k for a STOCHASTIC probe -- every draw estimates the
    whole ||dH||_F^2 (Derivation 2.1), so the draws are averaged -- and `nu` = 9 N^2 for
    the DETERMINISTIC set, where probe j is column j of dH and the columns are summed
    (Derivation 2.3). That is the one place the two kinds of probe differ.

    mode: "rademacher" (v_i in {-1, +1}, the default: the smallest variance of the
    unit-variance draws, Derivation 2.2) or "gaussian" (standard normal, PHL's
    Algorithm 1), k draws from `rng` (a numpy Generator; the caller seeds it);
    "cartesian" (v_j = e_j, k := 3N, exact).
    """
    if mode not in PROBE_MODES:
        raise ValueError("probe mode must be one of {}; got {!r}".format(PROBE_MODES, mode))
    hr = np.asarray(hessian_r, dtype=float)
    if hr.ndim != 2 or hr.shape[0] != hr.shape[1]:
        raise ValueError("the Label must be a square [3N, 3N] matrix; got shape {}".format(hr.shape))
    n3 = int(hr.shape[0])
    stochastic = mode in ("rademacher", "gaussian")
    if stochastic:
        if k < 1:
            raise ValueError("k must be >= 1 for a stochastic probe")
        rng = np.random.default_rng() if rng is None else rng
        v = rng.choice([-1.0, 1.0], size=(k, n3)) if mode == "rademacher" else rng.standard_normal((k, n3))
    else:
        v = np.eye(n3)
    v = np.asarray(v, dtype=float)
    r = v @ hr.T                                                      # H_r v_j, row by row
    nu = n3 * n3                                                      # PHL's (3N)^2
    info = dict(n3=n3, mode=mode, k=int(v.shape[0]), stochastic=stochastic, nu=nu,
                denominator=nu * (int(v.shape[0]) if stochastic else 1))
    return v, r, info


def estimator_from_products(hvp_theta, r, info):
    """Eq. 6' from the model's products: sum_j ||H_theta v_j - r_j||^2 over
    `info["denominator"]`, numpy side (the torch side lives in `phl_loss`).
    `hvp_theta` [k, 3N]."""
    hv = np.asarray(hvp_theta, dtype=float).reshape(np.shape(r))
    rho = hv - np.asarray(r, dtype=float)
    return float(np.sum(rho * rho)) / info["denominator"]


def estimator_variance(hessian_theta, hessian_r, k=1):
    """Derivation 2.2 with A = H_theta - H_r and B = A^T A:

        Var_Rademacher[L^(k)] = 2 (||B||_F^2 - sum_i B_ii^2) / ((9 N^2)^2 k)
        Var_Gaussian  [L^(k)] = 2 ||B||_F^2                  / ((9 N^2)^2 k)

    returned as a dict, the Gaussian one always the larger (the kurtosis term).
    """
    a = np.asarray(hessian_theta, dtype=float) - np.asarray(hessian_r, dtype=float)
    nu = float(a.shape[0]) ** 2                                       # 9 N^2
    b = a.T @ a
    fro2 = float(np.sum(b * b))
    diag2 = float(np.sum(np.diag(b) ** 2))
    return dict(rademacher=2.0 * (fro2 - diag2) / (nu ** 2 * k), gaussian=2.0 * fro2 / (nu ** 2 * k))


# ---------------------------------------------------------------------------------------
# THE STANDARD VIBRATIONAL ANALYSIS -- not a training path (S0-C-64). Mass weighting and
# the Eckart projection are what a FREQUENCY is; they live here only until their last
# caller is gone: the smoke fit's diagnostic (ticket 36) and the judge's `loss_exact`
# row (ticket 38). Nothing below is reachable from `phl_loss`.
#
#     K   = M^-1/2 H M^-1/2                      mass-weighted Hessian, eV A^-2 amu^-1
#     P   = I - V V^T                            Eckart projector (V: rigid, mass-weighted)
#     K~  = P K P = L Lambda L^T                 reference modes L_r, eigenvalues lambda_r
#     A   = P_W M^-1/2 (H_theta - H_r) M^-1/2 P_W
# ---------------------------------------------------------------------------------------


def mass_vectors(masses):
    """m3 = masses repeated per Cartesian component [3N], and M^-1/2 as a vector."""
    m3 = np.repeat(np.asarray(masses, dtype=float), 3)
    return m3, 1.0 / np.sqrt(m3)


def mass_weighted(hessian_eV_A2, masses):
    """K = M^-1/2 H M^-1/2, symmetrised."""
    m3, _inv = mass_vectors(masses)
    k = np.asarray(hessian_eV_A2, dtype=float) / np.sqrt(np.outer(m3, m3))
    return 0.5 * (k + k.T)


def projector(masses, positions):
    """(P, M^-1/2 as a vector, V): the Eckart projector in mass-weighted coordinates at
    `positions`, from `hessian.rigid_body_vectors` (orthonormal, rank 6 or 5)."""
    m3, inv_sqrt_m = mass_vectors(masses)
    v, _sing, _rank = hessian_mod.rigid_body_vectors(masses, positions)
    p = np.eye(len(m3)) - v @ v.T
    return p, inv_sqrt_m, v


def reference_modes(hessian_r_eV_A2, masses, positions):
    """(L_r [3N, n_vib], lambda_r [n_vib]) of the projected reference Hessian, ascending;
    the rigid modes are identified by overlap with the rigid subspace, as
    `hessian.project_and_diagonalise` does, never as 'the six smallest'."""
    p, _inv, v = projector(masses, positions)
    k = p @ mass_weighted(hessian_r_eV_A2, masses) @ p
    k = 0.5 * (k + k.T)
    lam, vec = np.linalg.eigh(k)
    keep = np.linalg.norm(v.T @ vec, axis=0) ** 2 <= 0.5
    lam, vec = lam[keep], vec[:, keep]
    order = np.argsort(lam)
    return vec[:, order], lam[order]


def entropy_weights(omega_r_cm, masses, positions, temperature_K=thermo.T_REF, preset="crest",
                    step_cm=ENTROPY_FD_STEP_CM):
    """w_i = |dS_msRRHO/d omega_i| at T, normalised to max 1.

    Central finite difference of the msRRHO vibrational entropy under the preset, mode
    by mode (the per-mode terms are separable; only the rotor cap is molecular, and it
    is evaluated with the given masses and positions). A mode below `thermo.VIBTHR_CM`
    has no harmonic entropy to differentiate: it keeps the largest weight, 1.0 -- the
    limit of the harmonic-oscillator derivative as omega -> 0.
    Returns (w normalised, |dS/d omega| in cal/mol/K per cm^-1).
    """
    omega = np.asarray(omega_r_cm, dtype=float)
    ds = np.zeros(len(omega))
    below = omega - step_cm < thermo.VIBTHR_CM
    for i in np.flatnonzero(~below):
        s = []
        for f in (omega[i] + step_cm, omega[i] - step_cm):
            r = thermo.msrrho([float(f)], masses, positions, preset=preset, temperature_K=temperature_K,
                              imaginary_policy="refuse")
            s.append(r["S_vib_kcal_per_K"] * 1000.0)
        ds[i] = abs(s[0] - s[1]) / (2.0 * step_cm)
    top = ds.max() if ds.size and ds.max() > 0 else 1.0
    w = ds / top
    w[below] = 1.0
    return w, ds


def weighted_projector(modes_r, weights=None):
    """P_W = L_r W^1/2 L_r^T (symmetric). With `weights` None this is P itself,
    because L_r L_r^T = P."""
    l = np.asarray(modes_r, dtype=float)
    if weights is None:
        return l @ l.T
    w = np.asarray(weights, dtype=float)
    if w.shape != (l.shape[1],) or (w < 0).any():
        raise ValueError("weights must be one non-negative number per reference mode ({}); got {}".format(
            l.shape[1], w.shape))
    return (l * np.sqrt(w)[None, :]) @ l.T


def error_operator(hessian_theta, hessian_r, masses, positions, weights=None):
    """A = P_W M^-1/2 (H_theta - H_r) M^-1/2 P_W, symmetrised."""
    l_r, _lam = reference_modes(hessian_r, masses, positions)
    p_w = weighted_projector(l_r, weights)
    d = mass_weighted(hessian_theta, masses) - mass_weighted(hessian_r, masses)
    a = p_w @ d @ p_w
    return 0.5 * (a + a.T)


def projected_loss_full(hessian_theta, hessian_r, masses, positions, weights=None):
    """||A||_F^2 / n_vib: the projected, mass-weighted DIAGNOSTIC. Not the training
    target (S0-C-64) -- the judge's `loss_exact` row and the smoke fit's diagnostic."""
    l_r, _lam = reference_modes(hessian_r, masses, positions)
    a = error_operator(hessian_theta, hessian_r, masses, positions, weights)
    return float(np.sum(a * a)) / l_r.shape[1]


def mode_basis_terms(hessian_theta, hessian_r, masses, positions):
    """The two terms in the reference-mode basis: (D, lambda_r) with
    D = L_r^T K~_theta L_r, so that ||A||_F^2 = sum_i (D_ii - lambda_i)^2 + sum_{i!=j} D_ij^2
    -- `hessian_compare` family 4 (OMEGA_ALONG_REF_CM^2 on the diagonal, MIXING off it)."""
    l_r, lam_r = reference_modes(hessian_r, masses, positions)
    p, _inv, _v = projector(masses, positions)
    k_t = p @ mass_weighted(hessian_theta, masses) @ p
    return l_r.T @ (0.5 * (k_t + k_t.T)) @ l_r, lam_r
