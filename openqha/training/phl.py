"""Algorithm 1 of the projected Hessian loss: projector, reference modes, entropy
weights, probes, and the exact loss (eqs. 1, 3, 5, 10 of design-phl-loss.md).

PRODUCTION. Ticket 11 of the Hessian-learning set. numpy + torch; no mace import.

Everything here is per STRUCTURE and needs no autograd: it is what a data loader or a
loss computes from the Label `H_r`, the masses and the positions before the model is
asked for anything. Derivations and the numbers the tests hold these functions to:
`docs/tutorials/T03_openQHA_Theory_Projected_Hessian_Loss.ipynb` sections 1-4 and 6.

    K   = M^-1/2 H M^-1/2                      mass-weighted Hessian, eV A^-2 amu^-1
    P   = I - V V^T                            Eckart projector (V: rigid vectors, mass-weighted)
    K~  = P K P = L Lambda L^T                 reference modes L_r, eigenvalues lambda_r
    A   = P M^-1/2 (H_theta - H_r) M^-1/2 P    the error operator
    L_H = ||A||_F^2 / n_vib                                                        (1)
        = [ sum_i (D_ii - lambda_i)^2 + sum_{i!=j} D_ij^2 ] / n_vib,  D = L_r^T K~_theta L_r   (2)
    P_W = L_r W^1/2 L_r^T ;  L_H^W = ||P_W M^-1/2 dH M^-1/2 P_W||_F^2 / n_vib      (3)
    v~  = M^-1/2 P v ;  A v = P M^-1/2 (H_theta v~ - H_r v~)                      (5)
    probe = modes (v_j = L_r,j, k = n_vib)  ->  the estimator IS (1), exactly     (10)

THE CARTESIAN TARGET (S0-C-53, 2026-09-21; T05 section 3, PHL eq. 2.1'). The training
target is the raw Cartesian matrix: no mass weighting, no Eckart projection, no
reference modes --

    L_H^cart = ||H_theta - H_r||_F^2 / (9 N^2)                                    (1')
    v~ = v ;  r = H_r v ;  rho = H_theta v - r ;  L^ = sum_j ||rho_j||^2 / (9 N^2 k)   (6')

`make_probes(..., metric="cartesian")` is that path: `projector = I`, `inv_sqrt_m = 1`,
denominator `9 N^2 k` (stochastic) or `9 N^2` (a deterministic set). It is the DEFAULT
of the loss module; the projected quantities above stay as diagnostics (the judge's
frequency rows, the entropy weighting of T03).

Units: `H` in eV/A^2, masses in amu, positions in A. The Label `H_r` is the raw
Cartesian Hessian at the frame's geometry, never pre-projected (design section 1): the
same `P` is applied to both sides here, which is what makes the comparison meaningful.
"""
import numpy as np

from ..thermochem import hessian as hessian_mod
from ..thermochem import thermo

#: The four probe modes of `make_probes`. `rademacher` and `gaussian` are stochastic
#: (Hutchinson, eq. 4); `modes` and `cartesian` are the deterministic sets that make the
#: estimator exact (eq. 10) at k = n_vib and k = 3N HVPs.
PROBE_MODES = ("rademacher", "gaussian", "modes", "cartesian")

#: The two metrics of `make_probes`: `projected` (P_W M^-1/2 on both sides, eqs. 1-6)
#: and `cartesian` (the raw matrix, eq. 1'). Not to be confused with the PROBE MODE
#: `cartesian` (the deterministic unit-vector set), which either metric may use.
METRICS = ("projected", "cartesian")

#: Finite-difference step for the entropy derivative, cm^-1. The msRRHO entropy is
#: smooth in omega away from the rotor cap; 0.5 cm^-1 reproduces the analytic harmonic
#: derivative to four digits (T03 section 3).
ENTROPY_FD_STEP_CM = 0.5


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
    """w_i = |dS_msRRHO/d omega_i| at T, normalised to max 1 (design section 2.2).

    Central finite difference of the msRRHO vibrational entropy under the preset, mode
    by mode (the per-mode terms are separable; only the rotor cap is molecular, and it
    is evaluated with the given masses and positions). The msRRHO interpolation weight
    is inside dS/d omega, so a mode under tau is not over-driven. A mode below
    `thermo.VIBTHR_CM` (non-positive or inside CREST's threshold) has no harmonic
    entropy to differentiate: it keeps the largest weight, 1.0 -- the limit of the
    harmonic-oscillator derivative as omega -> 0 -- because its Label is kept (round-1
    Q3: never refused) and the judge, not the loss, sets such modes aside (round-2 Q6).
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
    """A = P_W M^-1/2 (H_theta - H_r) M^-1/2 P_W (eq. 1 / eq. 3's operator), symmetrised."""
    l_r, _lam = reference_modes(hessian_r, masses, positions)
    p_w = weighted_projector(l_r, weights)
    d = mass_weighted(hessian_theta, masses) - mass_weighted(hessian_r, masses)
    a = p_w @ d @ p_w
    return 0.5 * (a + a.T)


def projected_loss_full(hessian_theta, hessian_r, masses, positions, weights=None):
    """The EXACT loss, eq. 1 (or eq. 3 with weights): ||A||_F^2 / n_vib. The
    evaluation-side quantity, and the oracle every estimator is held to."""
    l_r, _lam = reference_modes(hessian_r, masses, positions)
    a = error_operator(hessian_theta, hessian_r, masses, positions, weights)
    return float(np.sum(a * a)) / l_r.shape[1]


def mode_basis_terms(hessian_theta, hessian_r, masses, positions):
    """Eq. 2's two terms in the reference-mode basis: (D, lambda_r) with
    D = L_r^T K~_theta L_r, so that ||A||_F^2 = sum_i (D_ii - lambda_i)^2 + sum_{i!=j} D_ij^2
    -- `hessian_compare` family 4 (OMEGA_ALONG_REF_CM^2 on the diagonal, MIXING off it)."""
    l_r, lam_r = reference_modes(hessian_r, masses, positions)
    p, _inv, _v = projector(masses, positions)
    k_t = p @ mass_weighted(hessian_theta, masses) @ p
    return l_r.T @ (0.5 * (k_t + k_t.T)) @ l_r, lam_r


def cartesian_loss_full(hessian_theta, hessian_r):
    """The EXACT Cartesian target, eq. 1': ||H_theta - H_r||_F^2 / (9 N^2) -- what the
    `cartesian` metric's estimator is held to (T05 section 3)."""
    d = np.asarray(hessian_theta, dtype=float) - np.asarray(hessian_r, dtype=float)
    n3 = d.shape[0]
    return float(np.sum(d * d)) / (n3 * n3)


def make_probes(masses, positions, hessian_r, mode="rademacher", k=4, weights=None, rng=None,
                metric="projected"):
    """Algorithm 1: the probes the model sees and the reference side of each.

    Returns (v~ [k, 3N], r [k, 3N], info) with, for metric = "projected" (eqs. 1-6),
        v~_j = M^-1/2 P_W v_j                 -- the vector for the model's HVP (eq. 5)
        r_j  = P_W M^-1/2 H_r v~_j            -- a matvec on the Label
    and info = dict(n_vib, mode, k, `denominator`, P_W as `projector`, M^-1/2 as
    `inv_sqrt_m`), so that the loss forms rho_j = P_W M^-1/2 (H_theta v~_j) - r_j and
    returns sum_j ||rho_j||^2 / denominator. The denominator is n_vib k for a STOCHASTIC
    probe (every draw estimates the whole ||A||_F^2, eq. 4, so the draws are averaged)
    and n_vib for a DETERMINISTIC set (each probe is one column of A in that basis and
    the columns are summed, eq. 10) -- the one place the two kinds of probe differ.

    For metric = "cartesian" (eq. 1', S0-C-53) nothing is projected or weighted:
        v~_j = v_j ;  r_j = H_r v_j ;  projector = I ;  inv_sqrt_m = 1
    and the denominator is 9 N^2 k (stochastic) or 9 N^2 (deterministic); `n_vib` in
    `info` is then 3N (the dimension the norm is averaged over) and no reference modes
    are computed at all. `mode = "modes"` needs the reference modes and is refused there.

    mode: "rademacher" (v_i in {-1, +1}) or "gaussian" (standard normal), k draws from
    `rng` (a numpy Generator; the caller seeds it); "modes" (v_j = L_r,j, k := n_vib,
    exact by eq. 10); "cartesian" (v_j = e_j, k := 3N, exact likewise). `weights` (one per
    reference mode) switches P to P_W (eq. 3); ignored by the cartesian metric.
    """
    if mode not in PROBE_MODES:
        raise ValueError("probe mode must be one of {}; got {!r}".format(PROBE_MODES, mode))
    if metric not in METRICS:
        raise ValueError("metric must be one of {}; got {!r}".format(METRICS, metric))
    n3 = 3 * len(masses)
    hr = np.asarray(hessian_r, dtype=float)
    stochastic = mode in ("rademacher", "gaussian")
    if stochastic:
        if k < 1:
            raise ValueError("k must be >= 1 for a stochastic probe")
        rng = np.random.default_rng() if rng is None else rng
        v = rng.choice([-1.0, 1.0], size=(k, n3)) if mode == "rademacher" else rng.standard_normal((k, n3))
    if metric == "cartesian":
        if mode == "modes":
            raise ValueError("probe mode 'modes' needs the reference modes; the cartesian metric has none "
                             "(use 'cartesian' for the exact set)")
        if mode == "cartesian":
            v = np.eye(n3)
        vt = np.asarray(v, dtype=float)
        r = vt @ hr.T                                                     # H_r v  (H_r symmetric)
        info = dict(n_vib=int(n3), mode=mode, k=int(vt.shape[0]), stochastic=stochastic,
                    denominator=int(n3) * int(n3) * (int(vt.shape[0]) if stochastic else 1),
                    projector=np.eye(n3), inv_sqrt_m=np.ones(n3), metric=metric)
        return vt, r, info
    l_r, _lam = reference_modes(hessian_r, masses, positions)
    n_vib = l_r.shape[1]
    p_w = weighted_projector(l_r, weights)
    _m3, inv_sqrt_m = mass_vectors(masses)
    if mode == "modes":
        v = l_r.T
    elif mode == "cartesian":
        v = np.eye(n3)
    vt = (v @ p_w) * inv_sqrt_m[None, :]                                  # M^-1/2 P_W v  (P_W symmetric)
    r = ((vt @ hr) * inv_sqrt_m[None, :]) @ p_w                           # P_W M^-1/2 H_r v~
    info = dict(n_vib=int(n_vib), mode=mode, k=int(vt.shape[0]), stochastic=stochastic,
                denominator=int(n_vib) * (int(vt.shape[0]) if stochastic else 1),
                projector=p_w, inv_sqrt_m=inv_sqrt_m, metric=metric)
    return vt, r, info


def estimator_from_products(hvp_theta, r, info):
    """Eq. 6 (or eq. 10) from the model's products: sum_j ||P_W M^-1/2 (H_theta v~_j) - r_j||^2
    over `info["denominator"]`, numpy side (the torch side lives in `phl_loss`).
    `hvp_theta` [k, 3N]."""
    hv = np.asarray(hvp_theta, dtype=float).reshape(r.shape)
    rho = ((hv * info["inv_sqrt_m"][None, :]) @ info["projector"]) - r
    return float(np.sum(rho * rho)) / info["denominator"]


def estimator_variance(hessian_theta, hessian_r, masses, positions, k=1, weights=None, metric="projected"):
    """Eq. 8, Rademacher: Var[L^_H^(k)] = 2 (||B||_F^2 - sum_i B_ii^2) / (n_vib^2 k),
    B = A^T A; and the Gaussian value 2 ||B||_F^2 / (n_vib^2 k) beside it. For the
    cartesian metric A = H_theta - H_r and the estimator divides by 9 N^2 instead of
    n_vib, so the variance carries (9 N^2)^2 (the same algebra, eq. 6')."""
    if metric == "cartesian":
        a = np.asarray(hessian_theta, dtype=float) - np.asarray(hessian_r, dtype=float)
        denominator = float(a.shape[0]) ** 2                # 9 N^2
    else:
        l_r, _lam = reference_modes(hessian_r, masses, positions)
        denominator = float(l_r.shape[1])                   # n_vib
        a = error_operator(hessian_theta, hessian_r, masses, positions, weights)
    b = a.T @ a
    fro2 = float(np.sum(b * b))
    diag2 = float(np.sum(np.diag(b) ** 2))
    return dict(rademacher=2.0 * (fro2 - diag2) / (denominator ** 2 * k), gaussian=2.0 * fro2 / (denominator ** 2 * k))
