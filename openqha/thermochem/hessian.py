"""Package 2 -- finite-difference Hessian, Eckart projection, normal frequencies.

**The projection must come before the diagonalisation** (plan section 4, package 2).
The reason: if one diagonalises first and then "picks the 6 smallest and discards
them", those 6 modes are not the rigid-body modes -- where the residual forces are
non-zero, or where a low-frequency mode is close in energy to the rigid-body ones (the
methyl rotor in this project goes down to 42 cm^-1), what gets picked can be a real
vibrational mode. Projection **removes** the rigid-body subspace from the matrix; it
does not **pick it out** of the result.

The chain of units, every step written out, with no bare coefficient left anywhere:

    Hessian             eV / A^2
    mass-weighted       eV / (A^2 * amu)          H~_ij = H_ij / sqrt(m_i m_j)
    eigenvalue lambda   eV / (A^2 * amu)
    omega^2             s^-2                      lambda * e / (1e-20 * amu_kg)
    wavenumber nu~      cm^-1                     omega / (2 pi c)
"""
import numpy as np

# ---- physical constants (the same set as the lecture notes and every script) -------
E_CHARGE = 1.602176634e-19          # J per eV
AMU_KG = 1.66053906660e-27          # kg per amu
ANG_M = 1.0e-10                     # m per A
C_CM_S = 2.99792458e10              # cm/s
# nu~[cm^-1] = CM_INV_PER_SQRT_EV_A2_AMU * sqrt(lambda)
CM_INV_PER_SQRT_EV_A2_AMU = (np.sqrt(E_CHARGE / (ANG_M ** 2 * AMU_KG))
                             / (2.0 * np.pi * C_CM_S))

# ---- parameter classification ------------------------------------------------------
DELTA_A = 0.01          # A, the finite-difference displacement. A convention that may be
                        # changed -- its convergence is measured by delta_convergence
RIGID_SINGULAR_TOL = 1e-8   # rank criterion for the rigid-body vectors: a singular value
                            # below this counts as linearly dependent (a linear molecule
                            # has only 2 rotations, so the 3rd singular value is exactly 0)


def finite_difference_hessian(atoms, calc, delta=DELTA_A, progress=None):
    """Central-difference Hessian, in eV/A^2.

        H_ij = -( F_j(x + delta e_i) - F_j(x - delta e_i) ) / (2 delta)

    6N force evaluations in all. Returns (the symmetrised H, the asymmetry residual).

    The asymmetry residual max|H - H^T| is **a free diagnostic of the finite-difference
    error itself** -- an exact Hessian is necessarily symmetric, so the whole residual
    comes from truncation of the difference and from numerical noise in the potential.
    """
    atoms = atoms.copy()
    atoms.calc = calc
    x0 = atoms.get_positions().copy()
    n = len(atoms)
    h = np.zeros((3 * n, 3 * n))
    for i in range(3 * n):
        a, c = divmod(i, 3)
        for sign in (+1, -1):
            x = x0.copy()
            x[a, c] += sign * delta
            atoms.set_positions(x)
            f = atoms.get_forces().reshape(-1)
            h[i] += -sign * f / (2.0 * delta)
        if progress is not None:
            progress(i + 1, 3 * n)
    atoms.set_positions(x0)
    asym = float(np.abs(h - h.T).max())
    return 0.5 * (h + h.T), asym


def analytic_hessian(atoms, calc, progress=None):
    """Analytic (automatic-differentiation) Hessian, eV/A^2. Returns (H, asymmetry).

    THIS IS THE PRODUCTION PATH. `finite_difference_hessian` is kept for the
    comparison and for calculators that cannot differentiate twice.

    Defect 64 / D0-P1-47 -- why the default changed (measured 2026-08-31,
    scripts/_superseded/closed-defects/s0_hessian_autodiff_probe.py, mace 0.3.17):

        quantity                       finite difference    analytic
        max|H - H^T|                   4.785e-02 eV/A^2     4.263e-14
        dsgdb9nsd_000506 mode 0        -9.17 cm^-1          +13.94 cm^-1
        cost                           3.93 s               2.24 s

    More accurate AND cheaper -- there is no trade-off to weigh. The middle row is
    the reason it matters: at delta = 0.01 A the finite-difference resolution is
    +-9 to 20 cm^-1, while the modes it was condemning as imaginary had absolute
    values of only 9 to 24 cm^-1. **The criterion was being applied below the
    resolution of the measurement**, so a true minimum was thrown away as a saddle
    and the molecule was lost. Nothing about the criterion changed; the instrument
    did.

    `progress` is accepted and ignored: an analytic Hessian is one call, so there
    is nothing to report progress against. It is in the signature so callers can
    swap the two implementations without a branch.
    """
    getter = getattr(calc, "get_hessian", None)
    if getter is None:
        raise NotImplementedError(
            "{} has no get_hessian; use finite_difference_hessian, and record "
            "in the product that the finite-difference path was used "
            "(defect 64: its resolution is +-9 to 20 cm^-1)".format(
                type(calc).__name__))
    atoms = atoms.copy()
    atoms.calc = calc
    h = np.asarray(getter(atoms=atoms), dtype=float)
    n = len(atoms)
    # mace 0.3.17's MACECalculator returns (3N, N, 3) -- measured 2026-09-03, not
    # assumed: the first version of this function guessed (N, 3, 3N) and the shape
    # check below rejected the real thing, which is what it is for. Both index
    # groups are atom-major, so flattening the trailing (N, 3) gives the same
    # ordering as the leading 3N axis and `reshape(3N, 3N)` is the right move.
    # `verify_analytic_against_finite_difference` checks the result against an
    # independent computation rather than taking the layout on faith.
    #
    # Measured 2026-09-03: the transposed reshape gives the SAME frequencies here,
    # because the Hessian is symmetric under exchange of the two index groups
    # anyway (d2E/dx_i dx_j = d2E/dx_j dx_i). So the two orderings are not
    # distinguishable from the result, and no inspection of the matrix -- symmetry,
    # rigid-mode count, plausibility of the spectrum -- could separate them. Only
    # the finite-difference comparison is evidence.
    if h.shape in ((3 * n, n, 3), (n, 3, 3 * n)):
        h = h.reshape(3 * n, 3 * n)
    elif h.shape != (3 * n, 3 * n):
        raise ValueError(
            "unexpected Hessian shape {} for {} atoms -- expected {}, {} or {}. "
            "Refusing to reshape something whose layout has not been established: "
            "a wrong reshape stays symmetric and produces plausible frequencies, "
            "so it would not announce itself.".format(
                h.shape, n, (3 * n, n, 3), (n, 3, 3 * n), (3 * n, 3 * n)))
    asym = float(np.abs(h - h.T).max())
    return 0.5 * (h + h.T), asym


def verify_analytic_against_finite_difference(atoms, calc, delta=DELTA_A,
                                              tol_cm_inv=5.0):
    """Check the analytic Hessian against finite differences on this molecule.

    Exists because the analytic path involves a RESHAPE whose correctness cannot be
    read off the result. Measured on acetone: the transposed reshape is symmetric,
    removes exactly six rigid modes, and returns the identical spectrum -- because
    the Hessian is symmetric under exchange of the two index groups. Nothing about
    the matrix distinguishes a right reshape from a wrong one, so the only evidence
    available is an independent computation.

    Measured agreement, MACE-OFF23_medium, delta = 0.01 A:

        acetone   24 modes   max deviation 1.0407 cm^-1   mean 0.2396
        oxetane   24 modes   max deviation 0.5707 cm^-1   mean 0.2148

    and the asymmetry that motivated the switch, on the same structures:
    analytic 7e-15 to 1e-14 eV/A^2 against finite difference 8e-3 to 2e-2.

    Returns a record. Raises if the two disagree by more than `tol_cm_inv` on any
    mode, because at that point one of them is wrong and neither should be used.
    """
    ha, asym_a = analytic_hessian(atoms, calc)
    hf, asym_f = finite_difference_hessian(atoms, calc, delta=delta)
    m, x = atoms.get_masses(), atoms.get_positions()
    nu_a = np.asarray(project_and_diagonalise(ha, m, x)["frequencies_cm_inv"])
    nu_f = np.asarray(project_and_diagonalise(hf, m, x)["frequencies_cm_inv"])
    d = np.abs(nu_a - nu_f)
    rec = dict(
        n_modes=int(len(nu_a)),
        max_deviation_cm_inv=float(d.max()),
        mean_deviation_cm_inv=float(d.mean()),
        asymmetry_analytic_eV_A2=asym_a,
        asymmetry_finite_difference_eV_A2=asym_f,
        finite_difference_delta_A=float(delta),
        tolerance_cm_inv=float(tol_cm_inv),
        frequencies_analytic_cm_inv=[float(v) for v in nu_a],
        frequencies_finite_difference_cm_inv=[float(v) for v in nu_f],
        agrees=bool(d.max() <= tol_cm_inv))
    if not rec["agrees"]:
        raise ValueError(
            "analytic and finite-difference Hessians disagree by up to {:.2f} cm^-1 "
            "(tolerance {:.2f}). One of them is wrong -- most likely the analytic "
            "reshape. Do not use either until this is resolved.\n"
            "  analytic         {}\n  finite difference {}".format(
                d.max(), tol_cm_inv,
                np.round(nu_a, 2)[:8], np.round(nu_f, 2)[:8]))
    return rec


def hessian(atoms, calc, mode="analytic", delta=DELTA_A, progress=None):
    """Hessian by the named method. `mode` is 'analytic' or 'finite_difference'.

    The mode goes into every product record: which instrument produced a number is
    part of the number (defect 57 -- a condition that cannot be read off the
    product has already cost this repo one dataset).
    """
    if mode == "analytic":
        return analytic_hessian(atoms, calc, progress=progress)
    if mode in ("finite_difference", "fd"):
        return finite_difference_hessian(atoms, calc, delta=delta, progress=progress)
    raise ValueError("unknown Hessian mode {!r} -- 'analytic' or "
                     "'finite_difference'".format(mode))


def rigid_body_vectors(masses, positions):
    """Translation and rotation vectors in mass-weighted coordinates (3N-dimensional),
    orthonormalised.

    Returns (V, singular values, rank). The rank is 6 (non-linear) or 5 (linear) --
    **decided by the singular values, not by an assumption**.
    """
    m = np.asarray(masses, dtype=float)
    r = np.asarray(positions, dtype=float)
    com = (m[:, None] * r).sum(0) / m.sum()
    d = r - com
    sm = np.sqrt(m)
    n = len(m)
    cols = []
    for a in range(3):                      # translations
        v = np.zeros((n, 3))
        v[:, a] = sm
        cols.append(v.reshape(-1))
    for a in range(3):                      # rotations: sqrt(m_i) * (e_a x d_i)
        e = np.zeros(3); e[a] = 1.0
        v = sm[:, None] * np.cross(np.broadcast_to(e, d.shape), d)
        cols.append(v.reshape(-1))
    a_mat = np.stack(cols, axis=1)
    u, s, _ = np.linalg.svd(a_mat, full_matrices=False)
    rank = int((s > RIGID_SINGULAR_TOL * max(s[0], 1.0)).sum())
    return u[:, :rank], s, rank


def rigid_block_floor_cm(hessian_eV_A2, masses, positions):
    """The noise floor of a Hessian in cm^-1: the largest |eigenvalue| of the rigid-body
    block V^T M^-1/2 H M^-1/2 V of the UNPROJECTED mass-weighted Hessian (V: the
    orthonormal translation / rotation vectors). Translational and rotational invariance
    put this block at zero for an exact Hessian; an analytic ORCA Hessian leaves a few
    cm^-1 there, a numerical one (NumFreq of numerical gradients) the size of its
    finite-difference noise. It must be read BEFORE projection: after P H P the block is
    in the kernel by construction and says nothing (ticket 32 review)."""
    m = np.repeat(np.asarray(masses, dtype=float), 3)
    hm = np.asarray(hessian_eV_A2, dtype=float) / np.sqrt(np.outer(m, m))
    v, _sing, _rank = rigid_body_vectors(masses, positions)
    block = v.T @ hm @ v
    block = 0.5 * (block + block.T)
    lam = np.linalg.eigvalsh(block)
    return float(np.abs(eigenvalues_to_cm_inv(lam)).max()) if lam.size else 0.0


def project_and_diagonalise(hessian_eV_A2, masses, positions):
    """Eckart projection -> diagonalisation -> frequencies. Returns a full record dict.

    After projection the rigid-body subspace lies **in the kernel** and its eigenvalues
    are zero to machine precision; that is not an adjustable threshold, so what is
    reported here is the **gap** (the largest "zero" eigenvalue against the smallest
    genuine vibrational eigenvalue), leaving the reader to see for themselves how clean
    the separation is.
    """
    m = np.repeat(np.asarray(masses, dtype=float), 3)
    hm = hessian_eV_A2 / np.sqrt(np.outer(m, m))

    v, sing, rank = rigid_body_vectors(masses, positions)
    p = np.eye(len(m)) - v @ v.T
    hp = p @ hm @ p
    hp = 0.5 * (hp + hp.T)

    lam_raw = np.linalg.eigvalsh(hm)
    lam, vec = np.linalg.eigh(hp)

    # A rigid mode is an eigenvector that LIES IN the rigid-body subspace, decided by
    # overlap rather than by "take the 6 smallest"
    overlap = np.linalg.norm(v.T @ vec, axis=0) ** 2      # fraction of each eigenvector
                                                          # inside the rigid subspace
    is_rigid = overlap > 0.5
    n_rigid = int(is_rigid.sum())

    lam_vib = lam[~is_rigid]
    lam_rig = lam[is_rigid]
    order = np.argsort(lam_vib)
    lam_vib = lam_vib[order]

    return dict(
        n_atoms=len(masses),
        rigid_singular_values=[float(x) for x in sing],
        rigid_subspace_rank=rank,
        n_rigid_modes_removed=n_rigid,
        n_vibrational_modes=int(len(lam_vib)),
        expected_vibrational_modes=int(3 * len(masses) - rank),
        rigid_eigenvalues_eV_A2_amu=[float(x) for x in np.sort(lam_rig)],
        max_abs_rigid_eigenvalue=float(np.abs(lam_rig).max()) if n_rigid else 0.0,
        min_vibrational_eigenvalue=float(lam_vib.min()),
        separation_gap_ratio=(float(abs(lam_vib.min()) / np.abs(lam_rig).max())
                              if n_rigid and np.abs(lam_rig).max() > 0 else float("inf")),
        frequencies_cm_inv=[float(x) for x in eigenvalues_to_cm_inv(lam_vib)],
        frequencies_unprojected_cm_inv=[float(x) for x in eigenvalues_to_cm_inv(lam_raw)],
        n_imaginary=int((lam_vib < 0).sum()),
        overlap_with_rigid_subspace=[float(x) for x in np.sort(overlap)[::-1][:rank + 3]],
    )


def eigenvalues_to_cm_inv(lam):
    """A negative eigenvalue returns a negative wavenumber, the usual notation for an
    imaginary frequency."""
    lam = np.asarray(lam, dtype=float)
    return np.sign(lam) * CM_INV_PER_SQRT_EV_A2_AMU * np.sqrt(np.abs(lam))


def delta_convergence(atoms, calc, deltas=(0.005, 0.01, 0.02, 0.04)):
    """Measured convergence in the finite-difference displacement -- what turns DELTA_A
    from "a convention" into "a convention with a stated consequence".

    Reports the frequencies at each delta, together with the per-mode and maximum
    deviation relative to 0.01 A.
    """
    ref = None
    out = []
    for d in deltas:
        h, asym = finite_difference_hessian(atoms, calc, delta=d)
        rec = project_and_diagonalise(h, atoms.get_masses(), atoms.get_positions())
        nu = np.asarray(rec["frequencies_cm_inv"])
        if abs(d - DELTA_A) < 1e-12:
            ref = nu
        out.append(dict(delta_A=float(d), asymmetry_eV_A2=asym,
                        frequencies_cm_inv=[float(x) for x in nu],
                        n_imaginary=rec["n_imaginary"]))
    if ref is not None:
        for rec in out:
            nu = np.asarray(rec["frequencies_cm_inv"])
            rec["max_deviation_from_0.01A_cm_inv"] = float(np.abs(nu - ref).max())
            rec["mean_abs_deviation_from_0.01A_cm_inv"] = float(np.abs(nu - ref).mean())
    return out


# ---- thermal displacement sampling -------------------------------------------------
# **Why this is needed**: comparing two sets of forces AT a potential's own minimum is
# **degenerate** -- that potential's forces are approximately zero by construction, so
# the "deviation" is identically the reference force itself, and what gets measured is
# **the difference between two minimum geometries**, not the accuracy of the forces. To
# measure the accuracy of forces one must compare on structures that are **away from the
# minimum**. Allen et al. do exactly this in the original: their test set is
# configurations from 300 K molecular dynamics.
HBAR_SI = 1.054571817e-34
KB_SI_ = 1.380649e-23
AMU_KG_ = 1.66053906660e-27


MAX_RMS_DISPLACEMENT_A = 0.15   # this function's own default; the Frame set passes None
                                # since 2026-09-23 (data/frames.py FILTER). Its consequence
                                # is documented in thermal_displacements


def thermal_displacements(atoms, calc=None, temperature_K=298.15, n_samples=4,
                          seed=0, delta=DELTA_A,
                          max_rms_displacement_A=MAX_RMS_DISPLACEMENT_A,
                          max_draws_per_sample=50, return_hessian=False, hessian=None,
                          seeds=None, distribution="quantum"):
    """Sample displacements along the normal modes, returning n_samples structures away
    from the minimum. Three draws (`distribution=`): `nms` (NORMAL-MODE SAMPLING, a random
    partition of a bounded thermal energy over the modes -- what the Hessian-learning Frame
    set uses), `quantum` (the harmonic-oscillator positional fluctuation, this function's
    own default) and `classical` (equipartition). The last two are for calibration and for
    branch B; the Frame-set workflow draws only `nms`.

    The positional fluctuation of each vibrational mode takes the exact quantum
    harmonic-oscillator value

        <q_k^2> = (hbar / 2 omega_k) * coth(hbar omega_k / 2 k_B T)

    (in mass-weighted coordinates). In the high-temperature limit this reduces to the
    classical k_B T / omega^2, and in the low-temperature limit to the zero-point
    fluctuation -- **both limits are right**, so a low-frequency mode is not
    overestimated and a high-frequency one is not treated as frozen.

    Returns (a list of structures, a record). The record carries the displacement
    standard deviation of every mode, so it can be inspected.

    **A whole Hessian is computed inside this function anyway** -- the sampling has to
    know the normal modes. That Hessian used to be thrown away, so the same work was
    then repeated elsewhere. `return_hessian=True` hands it back: the record then also
    carries `hessian_eV_A2` (a 3N x 3N list) and `mode_record` (the full frequency
    record after Eckart projection, on exactly the same footing as
    `project_and_diagonalise`). This is half of the user's instruction of the afternoon
    of 2026-08-29, "change it so that it can produce something"; the other half is
    `openqha.orca.composite_hessian`, which swaps the source of the forces for the
    composite reference.

    `distribution="nms"` -- NORMAL-MODE SAMPLING. Mode k is given the harmonic energy
    E_k = c_k (3/2) N_a k_B T with a random partition c_k >= 0, sum_k c_k = s <= 1, and a
    random sign:

        q_k = ±sqrt(2 E_k) / omega_k ,    E_h = sum_k E_k = (3/2) s N_a k_B T

    so the TOTAL harmonic energy of a frame is bounded by (3/2) N_a k_B T with mean
    (3/4) N_a k_B T (s uniform on [0, 1]), against equipartition's mean (3N-6)/2 k_B T and
    unbounded chi-square tail at the same temperature -- which is why this draw is used at
    450 K where equipartition was used at 298 K. Per mode the energies are a uniform
    partition, not Boltzmann. The record carries `c_sum` (s per frame),
    `harmonic_energy_kcal` (per frame) and `harmonic_energy_cap_kcal`. The scheme is the
    normal-mode sampling published with the ANI-1 data set (Smith, Isayev, Roitberg,
    Sci. Data 4, 170193 (2017), section "Normal mode sampling", eq. 1), here in
    mass-weighted coordinates.

    `distribution="classical"` replaces the quantum fluctuation by equipartition,
    <q_k^2> = k_B T / omega_k^2 -- the distribution a classical 298 K trajectory (branch
    B) samples. Below 300 cm^-1 the two agree to 1-6 % in amplitude at 298 K; above
    1000 cm^-1 the quantum draw is 1.6-2.8x wider (zero-point motion) and carries 88 % of
    a quantum frame's ~27 kcal/mol (propanal), against ~7 kcal/mol classical -- measured
    2026-09-18, Hessian-learning note 4.

    `max_rms_displacement_A=` rejects and redraws a displacement whose RMS over the 3N
    coordinates exceeds it, `max_draws_per_sample` times, then raises. **A basin with a
    near-zero mode cannot satisfy any ceiling**: the classical amplitude is
    sqrt(k_B T)/omega, so an eigenvalue of 4-6 cm^-1 that survives the Eckart projection
    (measured on dsgdb9nsd_013068 and 025659, 2026-09-23) puts the median draw at 1.4-2.0 A
    RMS with 97-98 % of draws over 0.15 A. Pass None -- the Frame set does -- and the first
    draw is taken as it comes; the filtering is then done on the engine's own energy, where
    such a draw shows up as hundreds of kcal/mol above the basin (data/frames.py FILTER).

    `hessian=` (3N x 3N, eV/A^2, raw Cartesian) skips that computation and samples on
    the modes of the given matrix -- the Frame set (ticket 02) hands in the basin's
    stored `hessian.npy` so the frames are drawn along exactly the modes the basin
    record holds. `seeds=` gives one integer per sample (then `seed` is unused): each
    sample has its own generator, so a frame is reproducible from its own seed alone.
    The record's `seeds` lists what each sample was drawn from.
    """
    from ase import Atoms
    atoms = atoms.copy()
    m = atoms.get_masses()
    x0 = atoms.get_positions()

    if hessian is not None:
        h = np.asarray(hessian, dtype=float)
        if h.shape != (3 * len(atoms), 3 * len(atoms)):
            raise ValueError("hessian= has shape {}, expected {}".format(h.shape, (3 * len(atoms),) * 2))
        asym = float(np.abs(h - h.T).max())
        h = 0.5 * (h + h.T)
    else:
        if calc is None:
            raise ValueError("thermal_displacements needs a calculator or hessian=")
        h, asym = finite_difference_hessian(atoms, calc, delta=delta)
    mrep = np.repeat(np.asarray(m, dtype=float), 3)
    hm = h / np.sqrt(np.outer(mrep, mrep))
    v, sing, rank = rigid_body_vectors(m, x0)
    p = np.eye(len(mrep)) - v @ v.T
    hp = 0.5 * (p @ hm @ p + (p @ hm @ p).T)
    lam, vec = np.linalg.eigh(hp)
    overlap = np.linalg.norm(v.T @ vec, axis=0) ** 2
    keep = (overlap <= 0.5) & (lam > 0)          # sample only on real vibrational modes
    lam_v, vec_v = lam[keep], vec[:, keep]

    nu_cm = eigenvalues_to_cm_inv(lam_v)
    omega = 2.0 * np.pi * C_CM_S * nu_cm          # rad/s
    x = HBAR_SI * omega / (2.0 * KB_SI_ * temperature_K)
    if distribution == "quantum":
        q2_si = (HBAR_SI / (2.0 * omega)) / np.tanh(x)      # kg·m²
    elif distribution in ("classical", "nms"):
        q2_si = KB_SI_ * temperature_K / omega ** 2         # equipartition
    else:
        raise ValueError("distribution must be 'nms', 'quantum' or 'classical', not {!r}".format(distribution))
    sigma_q = np.sqrt(q2_si) / (np.sqrt(AMU_KG_) * 1.0e-10)  # amu^½·Å
    # normal-mode sampling does not use sigma_q -- it partitions a bounded total energy
    # instead (below); the equipartition sigma is still reported, as the scale of the modes.
    e_cap_si = 1.5 * len(atoms) * KB_SI_ * temperature_K     # (3/2) N_a k_B T, the NMS bound

    per_sample = seeds is not None
    if per_sample:
        seeds = [int(s) for s in seeds]
        if len(seeds) != int(n_samples):
            raise ValueError("seeds= has {} entries for n_samples = {}".format(len(seeds), n_samples))
    else:
        seeds = [int(seed)] * int(n_samples)
    def _draw_q(rng):
        """(q in amu^½·Å, the energy share s = sum_k c_k of this frame) for one frame."""
        if distribution != "nms":
            return rng.normal(0.0, sigma_q), float("nan")
        # NORMAL-MODE SAMPLING: mode k gets the harmonic energy E_k = c_k (3/2) N_a k_B T
        # from a random partition (Dirichlet(1,…,1) direction, uniform radius s), with a
        # random sign; q_k = ±sqrt(2 E_k)/omega_k in mass-weighted coordinates. The frame's
        # total harmonic energy is then (3/2) s N_a k_B T <= (3/2) N_a k_B T.
        c = rng.dirichlet(np.ones(len(omega)))
        s = float(rng.random())
        sign = np.where(rng.random(len(omega)) < 0.5, -1.0, 1.0)
        q_si = sign * np.sqrt(2.0 * c * s * e_cap_si) / omega
        return q_si / (np.sqrt(AMU_KG_) * 1.0e-10), s

    kcal_per_j = 6.02214076e23 / 4184.0
    rng = np.random.default_rng(seed)
    out, rejected, shares = [], 0, []
    for i_sample in range(int(n_samples)):
        if per_sample:
            rng = np.random.default_rng(seeds[i_sample])       # one generator per frame
        for _draw in range(int(max_draws_per_sample)):
            q, s_share = _draw_q(rng)
            dq = vec_v @ q                                   # mass-weighted displacement
            dx = (dq.reshape(-1, 3) / np.sqrt(np.asarray(m))[:, None])
            rms = float(np.sqrt((dx ** 2).mean()))
            if max_rms_displacement_A is None or rms <= max_rms_displacement_A:
                break
            rejected += 1
        else:
            raise RuntimeError(
                "{} consecutive draws all exceeded the displacement ceiling of {} A -- "
                "either the ceiling is too tight or the system has an extremely soft "
                "mode".format(max_draws_per_sample, max_rms_displacement_A))
        out.append(Atoms(numbers=atoms.numbers, positions=x0 + dx))
        shares.append(s_share)
    rec = dict(temperature_K=float(temperature_K), n_samples=int(n_samples),
               seed=int(seed), seeds=list(seeds), n_modes_sampled=int(keep.sum()),
               hessian_source="given" if hessian is not None else "finite_difference",
               distribution=distribution,
               hessian_asymmetry_eV_A2=float(asym),
               frequencies_cm_inv=[float(t) for t in nu_cm],
               sigma_q_amu_half_A=[float(t) for t in sigma_q],
               rms_displacement_A=[float(np.sqrt((np.asarray(s.get_positions()
                                                             - x0) ** 2).mean()))
                                   for s in out],
               max_rms_displacement_A=max_rms_displacement_A,
               n_draws_rejected=int(rejected),
               c_sum=[float(t) for t in shares],
               harmonic_energy_kcal=[float(t * e_cap_si * kcal_per_j) for t in shares],
               harmonic_energy_cap_kcal=float(e_cap_si * kcal_per_j),
               delta_A=float(delta),
               note=("`nms` (normal-mode sampling): a random partition of s (3/2) N_a k_B T "
                     "over the modes with random signs, s <= 1 -- the total harmonic ENERGY "
                     "of a frame is bounded (`harmonic_energy_cap_kcal`, `c_sum` and "
                     "`harmonic_energy_kcal` per frame), the GEOMETRY is not: the amplitude "
                     "is sqrt(2 E_k)/omega_k, so a near-zero mode (4-6 cm^-1 surviving the "
                     "Eckart projection: dsgdb9nsd_013068 / 025659, 2026-09-23) displaces by "
                     "angstroms under every draw here. `quantum`: the harmonic-oscillator "
                     "positional fluctuation, classical in the high-temperature limit and "
                     "zero-point in the low-temperature limit. `classical`: equipartition, "
                     "<q²> = k_B T / omega². `max_rms_displacement_A` rejects and redraws a "
                     "frame over the ceiling (None -- what the Frame set passes since "
                     "2026-09-23 -- takes the first draw and lets the energy window judge it). "
                     "**A better source is a real trajectory snapshot from package 3** -- at "
                     "that point this sampler should be replaced."))
    if return_hessian:
        rec["hessian_eV_A2"] = [[float(v) for v in row] for row in h]
        rec["mode_record"] = project_and_diagonalise(h, m, x0)
    return out, rec
