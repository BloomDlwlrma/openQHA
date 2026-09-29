"""Quasi-harmonic analysis: position covariance -> quasi-harmonic frequencies -> entropy.

This is branch B. It replaces the velocity-autocorrelation / vibrational-density-of-
states route, which was killed by a blank control: with the potential held
fixed and only the seed changed, three trajectories gave A_vib spreads of 10.04 and
16.74 kcal/mol, larger than the signal they were supposed to resolve. Quasi-harmonic
analysis is a VARIANCE estimator rather than a SPECTRAL one, so it converges
differently -- "differently" is not "better", and acceptance criterion 5 (the blank
control repeated here) is what decides whether this route survives.

The three formulas, and where each comes from
---------------------------------------------
1. Mass-weighted position covariance, after least-squares superposition:

       sigma_ij = < (x_i - <x_i>) (x_j - <x_j>) >          i, j = 1 .. 3N
       C        = M^(1/2) sigma M^(1/2)                    M = diag(m_1,m_1,m_1,m_2,...)

2. Quasi-harmonic frequency of each eigenvalue lambda_k of C (units mass * length^2):

       nu_k = (1 / 2 pi c) * sqrt(k_B T / lambda_k)

   Andricioaei & Karplus, J. Chem. Phys. 2001, 115, 6289; quoted verbatim in
   Rinaldo & Field, Biophys. J. 2003, 85, 3485, eq. 6.

3. Entropy of each mode, from the QUANTUM harmonic oscillator (Rinaldo & Field eq. 6):

       S = k_B sum_i [ (h nu_i / k_B T) / (exp(h nu_i / k_B T) - 1)
                       - ln(1 - exp(-h nu_i / k_B T)) ]

   and the Schlitter upper bound (Chem. Phys. Lett. 1993, 215, 617; Rinaldo & Field
   eq. 7, whose form already absorbs the mass weighting into lambda_i):

       S' = (1/2) k_B sum_i ln[ 1 + (k_B T e^2 / hbar^2) lambda_i ]

   S_QH <= S_Schlitter holds analytically, which makes it a free implementation check
   (acceptance criterion 6).

What this module refuses to do
------------------------------
It will not analyse a biased, constrained or mass-repartitioned trajectory. Those three
things do not crash anything; they produce an entropy that looks entirely normal and is
systematically too high (see `assert_trajectory_identity`, acceptance criterion 11, and
the header of configs/mdp/s0_qha_production.mdp). Branch A hands branch B GEOMETRIES;
not one CREST frame is ever reused.

Units throughout: positions angstrom, masses amu, eigenvalues amu*angstrom^2,
frequencies cm^-1, entropies kcal/(mol*K), temperature kelvin.
"""
import numpy as np

from ..thermochem import thermo
from ..thermochem.hessian import rigid_body_vectors

#: Physical constants. They are imported from thermo where they already exist, so the
#: package holds exactly one value of each.
HBAR_SI = 1.054571817e-34
KB_SI = thermo.KB_SI
AMU_KG = thermo.AMU_KG
ANG_M = 1.0e-10
C_CM_S = 2.99792458e10
EV_J = 1.602176634e-19

#: nu[cm^-1] = CM_INV_PER_SQRT_K_PER_AMU_A2 * sqrt(T[K] / lambda[amu*A^2])
CM_INV_PER_SQRT_K_PER_AMU_A2 = (
    np.sqrt(KB_SI / (AMU_KG * ANG_M ** 2)) / (2.0 * np.pi * C_CM_S))

#: cm^-1 per sqrt(eV / (A^2 amu)) -- the Hessian-side conversion, used only by the
#: harmonic-limit check so that its reference frequencies come from the same algebra
#: the branch C route uses.
CM_INV_PER_SQRT_EV_A2_AMU = (
    np.sqrt(EV_J / (AMU_KG * ANG_M ** 2)) / (2.0 * np.pi * C_CM_S))

#: Eigenvalues below this fraction of the largest are counted as zero. It exists so
#: acceptance criterion 9 (rank == 3N-6) can be MEASURED; the full spectrum is always
#: reported next to it so the effect of this number is visible rather than hidden.
RANK_TOLERANCE = 1e-12

#: A trajectory whose metadata does not satisfy every one of these is rejected. This is
#: acceptance criterion 11: the trajectory-admissibility prohibitions written as code
#: that can fail, because none of the three mistakes it guards against would ever raise
#: on its own.
FORBIDDEN_SOURCES = ("crest", "metadynamics", "alf_sampling", "alf", "biased")
#: The longest Nose-Hoover coupling time a branch B trajectory may be produced with.
#:
#: MEASURED, not conventional -- and the conventional choice is what fails. On the
#: harmonic surface built from the production potential's own Hessian, where the answer is
#: known in closed form (scripts/calibration/s0_B_thermostat_choice.py, 200 ps x 4 seeds,
#: acetone, lowest true mode 79.69 cm^-1):
#:
#:     tdamp / fs      T*S error / kcal      lowest recovered mode / cm^-1
#:         20              -0.093                      78.6
#:         50              -0.442                      78.4
#:        100              -1.150                     179.0
#:        419              -1.228                     229.0
#:       1000              -1.221                     227.9
#:
#: The KINETIC temperature is correct in every row (2<KE>/kT = 29.9 to 30.2 against 3N =
#: 30), so nothing about the run looks wrong; it is the CONFIGURATIONAL distribution that
#: collapses (2<U>/kT = 18.1 to 19.1 against 3N-3 = 27), and quasi-harmonic analysis reads
#: exactly that. The softest mode -- which dominates the entropy -- comes back three times
#: too stiff. Lengthening the chain (3 to 5) and the propagation resolution (tloop 1 to 5)
#: changed nothing, so this is the thermostat's ergodicity on a small stiff molecule and
#: not a discretisation error.
NOSE_HOOVER_MAX_TDAMP_FS = 20.0
#: Chains were invented because plain Nose-Hoover is not ergodic for a harmonic
#: oscillator, which is the regime a 10-atom molecule near its minimum sits in. Measured
#: on the same surface, chain length 1 returned a configurational variance ratio of 2.591
#: against a canonical value of 1.
NOSE_HOOVER_MIN_CHAIN_LENGTH = 3
HYDROGEN_MASS_AMU = 1.00794
HYDROGEN_MASS_TOLERANCE_AMU = 1e-3
REQUIRED_TIMESTEP_FS = 1.0


# ======================================================================================
# Acceptance criterion 11 -- the trajectory identity assertion
# ======================================================================================
def assert_trajectory_identity(meta, timestep_fs=REQUIRED_TIMESTEP_FS):
    """Raise unless the trajectory is admissible for quasi-harmonic analysis.

    `meta` is the metadata dict written next to the frames by
    `scripts/production/s0_B_qha_trajectory.py`. Every check below corresponds to one of the three
    independent reasons a CREST metadynamics trajectory is inadmissible:

      * a bias potential makes the sampled distribution not Boltzmann, so lambda_i comes
        out too large, nu_i too low and the entropy too HIGH -- a fixed direction, not
        noise;
      * SHAKE REMOVES a degree of freedom rather than softening it, so the covariance is
        identically zero along every constrained bond and the count of non-zero
        eigenvalues cannot reach 3N-6;
      * m_H = 2 amu changes the mass weighting itself, and nu ~ 1/sqrt(mu) turns a
        3000 cm^-1 C-H stretch into roughly 2200 cm^-1.

    A FIFTH applies only under Nose-Hoover, and it replaces the fourth rather than
    joining it: `fixcm` is an ASE Langevin option and means nothing there. What takes its
    place is the COUPLING TIME, because Nose-Hoover fails in exactly the same shape -- the
    kinetic temperature stays perfect while the configurational distribution collapses,
    and the entropy comes out wrong in a fixed direction with nothing looking wrong. See
    NOSE_HOOVER_MAX_TDAMP_FS for the table that fixes the limit.

    A fourth was added on 2026-09-03 after being measured rather than reasoned about:
    ASE's Langevin defaults to `fixcm=True`, which on a 10-atom molecule thermostats to
    429 K when asked for 298.15 K. Same failure mode as the other three -- nothing raises,
    and the entropy comes out high, here by 0.70 kcal/mol on T*S.

    Warnings are deliberately not an option here. Every one of these produces a perfectly
    healthy-looking number.
    """
    problems = []
    if meta.get("bias_potential") is not None:
        problems.append("bias_potential is {!r}, must be None -- a metadynamics or "
                        "well-tempered trajectory does not sample the Boltzmann "
                        "distribution of the true surface"
                        .format(meta.get("bias_potential")))
    if meta.get("constraints") is not None:
        problems.append("constraints is {!r}, must be None -- a constrained degree of "
                        "freedom is removed, not softened, and its covariance is "
                        "identically zero".format(meta.get("constraints")))
    m_h = meta.get("hydrogen_mass_amu")
    if m_h is None or abs(float(m_h) - HYDROGEN_MASS_AMU) > HYDROGEN_MASS_TOLERANCE_AMU:
        problems.append("hydrogen_mass_amu is {!r}, must be {} +- {} -- lambda_i is the "
                        "eigenvalue of the MASS-WEIGHTED covariance"
                        .format(m_h, HYDROGEN_MASS_AMU, HYDROGEN_MASS_TOLERANCE_AMU))
    dt = meta.get("timestep_fs")
    if dt is None or abs(float(dt) - timestep_fs) > 1e-9:
        problems.append("timestep_fs is {!r}, must be {}".format(dt, timestep_fs))
    name = str(meta.get("thermostat", "")).lower()
    is_nose_hoover = "nose" in name and "hoover" in name
    if is_nose_hoover:
        tdamp = meta.get("thermostat_tdamp_fs")
        if tdamp is None or float(tdamp) > NOSE_HOOVER_MAX_TDAMP_FS + 1e-9:
            problems.append(
                "thermostat_tdamp_fs is {!r}, must be present and <= {} fs. Nose-Hoover "
                "holds the KINETIC temperature perfectly at any coupling time while the "
                "CONFIGURATIONAL distribution collapses, and quasi-harmonic analysis "
                "reads only the configurational one. Measured on the closed-form surface: "
                "at 100 fs the softest mode comes back at 179 cm^-1 and at 419 fs at 229, "
                "against a true 79.7, costing 1.15 to 1.23 kcal/mol on T*S with nothing "
                "in the run looking wrong".format(tdamp, NOSE_HOOVER_MAX_TDAMP_FS))
        chain = meta.get("thermostat_chain_length")
        if chain is None or int(chain) < NOSE_HOOVER_MIN_CHAIN_LENGTH:
            problems.append(
                "thermostat_chain_length is {!r}, must be present and >= {}. A chain of "
                "length 1 is plain Nose-Hoover, which is not ergodic for a harmonic "
                "oscillator -- the regime this branch works in. Note that GROMACS with "
                "the default leap-frog integrator SILENTLY resets the chain to 1 "
                "('leapfrog does not yet support Nose-Hoover chains, nhchainlength reset "
                "to 1') while mdout.mdp still records the length that was asked for"
                .format(chain, NOSE_HOOVER_MIN_CHAIN_LENGTH))
    elif meta.get("thermostat_fixcm") is not False:
        fixcm = meta.get("thermostat_fixcm")
        problems.append(
            "thermostat_fixcm is {!r}, must be False. ASE's Langevin defaults to "
            "fixcm=True and upstream warns that it 'does not strictly sample the correct "
            "NVT distributions ... more pronounced for small systems'. Measured on a "
            "10-atom molecule targeting 298.15 K: fixcm=True gives 429.3 +- 10.1 K, "
            "+44.0 per cent, and costs +0.6991 kcal/mol on T*S -- 70 per cent of this "
            "repo's entire 1.0 kcal/mol accuracy target. A hot trajectory inflates every "
            "lambda, lowers every nu and raises the entropy, in a fixed direction, with "
            "nothing in the run looking wrong".format(fixcm))
    src = str(meta.get("source", "")).lower()
    if any(bad in src for bad in FORBIDDEN_SOURCES):
        problems.append("source is {!r} -- sampling trajectories from branch A and from "
                        "the branch C active-learning loop are deliberately driven away "
                        "from equilibrium and must never reach a thermodynamic number"
                        .format(meta.get("source")))
    if problems:
        raise ValueError("trajectory rejected by the branch B identity assertion "
                         "(acceptance criterion 11):\n  - "
                         + "\n  - ".join(problems))
    return dict(bias_potential=None, constraints=None,
                hydrogen_mass_amu=float(m_h), timestep_fs=float(dt),
                thermostat=meta.get("thermostat"),
                thermostat_fixcm=meta.get("thermostat_fixcm"),
                thermostat_tdamp_fs=meta.get("thermostat_tdamp_fs"),
                thermostat_chain_length=meta.get("thermostat_chain_length"),
                source=meta.get("source"), checked=True)


# ======================================================================================
# Superposition -- translation and rotation are removed here, and ONLY here
# ======================================================================================
def _kabsch(frames, reference, weights):
    """Weighted optimal rotations taking every frame onto `reference`, all already centred.

    Batched over frames: `np.linalg.svd` takes a stack of 3x3 matrices, and a production
    trajectory has tens of thousands of frames while the saturation curve and the batch
    diagnostic each run the fit again over prefixes. A per-frame Python loop turns that
    into minutes for no reason.

    The determinant correction is not optional: without it the "optimal rotation" can come
    back as a reflection, which fits a mirror image and quietly deflates the covariance.
    """
    corr = np.einsum("a,tai,aj->tij", weights, frames, reference)
    u, _s, vt = np.linalg.svd(corr)
    d = np.sign(np.linalg.det(np.einsum("tij,tjk->tik", u, vt)))
    u = u.copy()
    u[:, :, 2] *= d[:, None]
    return np.einsum("tij,tjk->tik", u, vt)


def superimpose(positions, masses, reference=None, max_iterations=10, tolerance_A=1e-8):
    """Mass-weighted least-squares fit of every frame onto an iteratively refined mean.

    Rinaldo & Field state plainly that "translational and rotational contributions were
    neglected" and superimpose every structure on a reference before taking the
    covariance. This function is that step, and it is the reason the assembly in
    `thermo.g_minus_eel` can add analytic translation and rotation without double
    counting: whatever is removed here is exactly what those terms put back.

    The reference is refined to the mean of the fitted frames, because fitting to one
    arbitrary frame leaves a bias whose size depends on which frame was picked.
    Returns (fitted positions, mean structure, record).
    """
    x = np.asarray(positions, dtype=float)
    m = np.asarray(masses, dtype=float)
    if x.ndim != 3 or x.shape[1] != len(m):
        raise ValueError("positions must be (n_frames, n_atoms, 3) matching masses; "
                         "got {} and {} masses".format(x.shape, len(m)))
    w = m / m.sum()
    com = (x * w[None, :, None]).sum(axis=1, keepdims=True)
    centred = x - com
    ref = (np.asarray(reference, dtype=float) if reference is not None
           else centred[0].copy())
    ref = ref - (ref * w[:, None]).sum(axis=0)

    history = []
    fitted = centred
    for it in range(int(max_iterations)):
        rot = _kabsch(centred, ref, w)
        fitted = np.einsum("tai,tij->taj", centred, rot)
        new_ref = fitted.mean(axis=0)
        new_ref = new_ref - (new_ref * w[:, None]).sum(axis=0)
        shift = float(np.sqrt((w[:, None] * (new_ref - ref) ** 2).sum()))
        history.append(dict(iteration=int(it), reference_shift_A=shift))
        ref = new_ref
        if shift < tolerance_A:
            break

    rec = dict(n_frames=int(x.shape[0]), n_atoms=int(x.shape[1]),
               n_iterations=len(history),
               converged=bool(history[-1]["reference_shift_A"] < tolerance_A),
               reference_shift_A=[h["reference_shift_A"] for h in history],
               tolerance_A=float(tolerance_A),
               rms_fluctuation_A=float(np.sqrt(
                   ((fitted - ref[None]) ** 2).sum(axis=2).mean())))
    return fitted, ref, rec


# ======================================================================================
# Covariance and its spectrum
# ======================================================================================

# =========================================================================================
# The production protocol: how long, how often, and what the frames must satisfy.
# =========================================================================================
#: Fallbacks, used only when configs/branchB_protocol.yaml is absent. They are the
#: published values, so a run without the config file is still the right protocol -- but
#: `protocol()` says which source it used, because "the same numbers by luck" and "the
#: same numbers because one file defines them" are different situations.
PROTOCOL_FALLBACK = dict(
    timestep_fs=1.0,
    equilibration_ps=520.0,
    production_ps=1500.0,
    sampling_interval_ps=0.5,
    expected_frames=3000,
    saturation_budget_kcal=0.3,
    chunk_frames=250,
    _source="Rinaldo & Field, Biophysical Journal 2003",
    _status="module fallback -- configs/branchB_protocol.yaml was not readable",
)


def protocol(cfg=None):
    """The branch B production protocol, from configs/branchB_protocol.yaml.

    Returns a dict with `sample_every_steps` DERIVED rather than stored: the interval is
    the physical quantity the paper states, and the step count is what an integrator
    needs. Writing both into the config would let them disagree, and a sampling interval
    that disagrees with its own step count is not detectable in any product.

    Raises if the derived step count is not a whole number -- an interval that is not an
    integer multiple of the timestep would silently be rounded, and every frame spacing
    in the record would then be wrong by that rounding.
    """
    from .. import config as _config

    #: `branch_b` is a CONFIG KEY, not a package name. On 2026-09-07 a subpackage rename
    #: (`branch_b/` -> `quasi_harmonic/`) rewrote this string literal too, and the broad
    #: `except Exception` below turned the resulting KeyError into a silent fall back to
    #: PROTOCOL_FALLBACK -- so every driver quietly ran the SUPERSEDED protocol while the
    #: verification reported "OK", because it compared the drivers against the same
    #: fallback. Hence the narrow except and the loud raise.
    KEY = "branch_b"
    if cfg is None:
        try:
            cfg = _config.load()
        except FileNotFoundError:
            # A checkout with no configuration is a legitimate state -- the module
            # constants ARE the published values. Anything else is not.
            return _derive(dict(PROTOCOL_FALLBACK))
    if KEY not in cfg:
        raise KeyError(
            "configs/branchB_protocol.yaml defines no {!r} section, so the branch B "
            "protocol cannot be read. "
            "Falling back silently is not an option here: the fallback is a "
            "DIFFERENT protocol, and a run using it would report the wrong length "
            "and sampling interval while looking healthy. "
            "Config loaded from: {}".format(KEY, cfg.get("_path")))
    p = dict(cfg[KEY])
    p.setdefault("_status", "configs/branchB_protocol.yaml")
    return _derive(p)


def _derive(p):
    """Fill in the quantities that are computed rather than stored. See `protocol`."""

    dt = float(p["timestep_fs"])
    interval_fs = float(p["sampling_interval_ps"]) * 1000.0
    steps = interval_fs / dt
    if abs(steps - round(steps)) > 1e-9:
        raise ValueError(
            "sampling_interval_ps {} is not a whole number of {} fs timesteps "
            "({} steps). Rounding it would make every frame_spacing_fs in every "
            "product wrong by the rounding.".format(
                p["sampling_interval_ps"], dt, steps))
    p["sample_every_steps"] = int(round(steps))
    p["frame_spacing_fs"] = float(interval_fs)
    return p


def min_frames_for(n_atoms):
    """Frames needed before the covariance can even have full rank.

    The constraint that defeated the source paper: a covariance estimated from T frames
    has at most T non-zero eigenvalues, and 3N-6 are wanted. Rinaldo & Field had 7758
    degrees of freedom and 3000 frames, so 4758 modes were missing by construction and
    the entropy could not converge however long they ran.

    For 10-19 atoms this returns 30-57 against the 3000 frames the protocol produces, so
    it is satisfied by a factor of ~59. It is checked anyway -- the cheap check that
    cannot fire is the one that catches the day something changes.
    """
    return 3 * int(n_atoms)

def mass_weighted_covariance(fitted_positions, masses, mean_structure=None):
    """C = M^(1/2) sigma M^(1/2), shape (3N, 3N), units amu * angstrom^2."""
    x = np.asarray(fitted_positions, dtype=float)
    m = np.asarray(masses, dtype=float)
    mean = (x.mean(axis=0) if mean_structure is None
            else np.asarray(mean_structure, dtype=float))
    d = (x - mean[None]).reshape(len(x), -1)
    sm = np.repeat(np.sqrt(m), 3)
    y = d * sm[None, :]
    return (y.T @ y) / len(y)


def spectrum(covariance, masses, mean_structure, rank_tolerance=RANK_TOLERANCE):
    """Diagonalise the covariance, remove the rigid-body subspace, and measure both.

    Two spectra are produced and both are kept:

      * UNPROJECTED. Superposition removes translation and rotation to the accuracy of a
        finite rotation, not exactly, so the six rigid eigenvalues come out small rather
        than zero. Their separation from the seventh MEASURES how clean the fit was
        (acceptance criterion 4), and it is reported as a ratio rather than compared
        against a tunable threshold.
      * PROJECTED. The Eckart projector built at the mean structure removes the rigid
        subspace exactly, and this is the spectrum the entropies are computed from. It
        uses `hessian.rigid_body_vectors`, the same code the branch C Hessian route
        uses, so the two routes cannot disagree about what "rigid" means.

    Rigid eigenvectors are identified by their overlap with the rigid subspace, not by
    "take the six smallest" -- the latter is an assumption dressed up as a computation.
    """
    c = np.asarray(covariance, dtype=float)
    c = 0.5 * (c + c.T)
    m = np.asarray(masses, dtype=float)
    n_atoms = len(m)

    lam_raw = np.sort(np.linalg.eigvalsh(c))          # ascending: rigid modes first
    v, sing, rank = rigid_body_vectors(m, mean_structure)
    p = np.eye(3 * n_atoms) - v @ v.T
    cp = p @ c @ p
    cp = 0.5 * (cp + cp.T)
    lam, vec = np.linalg.eigh(cp)

    overlap = np.linalg.norm(v.T @ vec, axis=0) ** 2
    is_rigid = overlap > 0.5
    n_rigid = int(is_rigid.sum())
    lam_vib = np.sort(lam[~is_rigid])[::-1]           # descending: softest mode first
    lam_rigid = np.sort(lam[is_rigid])

    lam_max = float(lam_vib.max()) if lam_vib.size else 0.0
    n_nonzero = int((lam_vib > rank_tolerance * lam_max).sum()) if lam_max > 0 else 0
    expected = 3 * n_atoms - rank

    # Acceptance criterion 4 is a ratio on the UNPROJECTED spectrum: the rigid
    # eigenvalues against the first vibrational one. Reported, never used as a filter.
    gap = (float(lam_raw[rank] / lam_raw[rank - 1])
           if rank > 0 and len(lam_raw) > rank and lam_raw[rank - 1] > 0
           else float("inf"))

    return dict(
        n_atoms=int(n_atoms),
        rigid_subspace_rank=int(rank),
        rigid_singular_values=[float(x) for x in sing],
        n_rigid_modes_removed=n_rigid,
        n_vibrational_modes=int(lam_vib.size),
        expected_vibrational_modes=int(expected),
        mode_count_matches_expected=bool(lam_vib.size == expected),
        n_nonzero_eigenvalues=n_nonzero,
        rank_tolerance=float(rank_tolerance),
        eigenvalues_amu_A2=[float(x) for x in lam_vib],
        rigid_eigenvalues_projected_amu_A2=[float(x) for x in lam_rigid],
        eigenvalues_unprojected_ascending_amu_A2=[float(x) for x in lam_raw],
        rigid_to_first_vibrational_ratio=gap,
        max_abs_projected_rigid_eigenvalue=(float(np.abs(lam_rigid).max())
                                            if n_rigid else 0.0),
        overlap_with_rigid_subspace_sorted=[
            float(x) for x in np.sort(overlap)[::-1][:rank + 3]],
        eigenvalues=lam_vib,                          # array, for internal use
    )


def frequencies_cm_inv(eigenvalues_amu_A2, temperature_K=thermo.T_REF):
    """nu = (1/2 pi c) sqrt(k_B T / lambda).

    A non-positive eigenvalue returns NaN rather than a silently enormous frequency:
    lambda -> 0 sends nu -> infinity, which is a divergence and not a small error.
    """
    lam = np.asarray(eigenvalues_amu_A2, dtype=float)
    out = np.full(lam.shape, np.nan)
    ok = lam > 0.0
    out[ok] = CM_INV_PER_SQRT_K_PER_AMU_A2 * np.sqrt(temperature_K / lam[ok])
    return out


# ======================================================================================
# Entropies
# ======================================================================================
def schlitter_entropy_kcal_per_K(eigenvalues_amu_A2, temperature_K=thermo.T_REF):
    """Rinaldo & Field eq. 7: S' = (1/2) k_B sum ln[1 + (k_B T e^2 / hbar^2) lambda]."""
    lam = np.asarray(eigenvalues_amu_A2, dtype=float)
    lam_si = lam * AMU_KG * ANG_M ** 2
    alpha = KB_SI * temperature_K * (np.e ** 2) / HBAR_SI ** 2
    return float(0.5 * thermo.KB_KCAL * np.log1p(alpha * lam_si[lam_si > 0]).sum())


def entropy(eigenvalues_amu_A2, temperature_K=thermo.T_REF):
    """Both estimators from one spectrum, plus the free check that one bounds the other.

    Both are reported, following the convention of reporting both members of a
    low-frequency pair rather than choosing silently: their difference is the direct
    price of the estimator choice on the soft modes, which is where free energy is most
    sensitive (dG/dnu at 100 cm^-1 is 4.2x its value at 3000 cm^-1).
    """
    lam = np.asarray(eigenvalues_amu_A2, dtype=float)
    nu = frequencies_cm_inv(lam, temperature_K)
    good = np.isfinite(nu)
    s_modes = np.array([thermo.s_mode_kcal_per_K(float(x), temperature_K)
                        for x in nu[good]])
    a_modes = np.array([thermo.a_mode_kcal(float(x), temperature_K) for x in nu[good]])
    e_modes = np.array([thermo.e_mode_kcal(float(x), temperature_K) for x in nu[good]])
    s_qh = float(s_modes.sum())
    s_sc = schlitter_entropy_kcal_per_K(lam, temperature_K)
    return dict(
        temperature_K=float(temperature_K),
        n_modes=int(good.sum()),
        n_modes_discarded_non_positive=int((~good).sum()),
        frequencies_cm_inv=[float(x) for x in nu],
        S_QH_kcal_per_K=s_qh,
        S_Schlitter_kcal_per_K=s_sc,
        TS_QH_kcal=float(s_qh * temperature_K),
        TS_Schlitter_kcal=float(s_sc * temperature_K),
        schlitter_minus_qh_kcal=float((s_sc - s_qh) * temperature_K),
        # Analytic, so a violation is an implementation error and nothing else.
        schlitter_bounds_qh=bool(s_sc >= s_qh),
        A_vib_kcal=float(a_modes.sum()),
        E_vib_kcal=float(e_modes.sum()),
        S_per_mode_kcal_per_K=[float(x) for x in s_modes],
        lowest_frequency_cm_inv=float(np.nanmin(nu)) if good.any() else float("nan"),
        highest_frequency_cm_inv=float(np.nanmax(nu)) if good.any() else float("nan"),
    )


def analyse(positions, masses, temperature_K=thermo.T_REF, meta=None,
            rank_tolerance=RANK_TOLERANCE):
    """The whole chain on one trajectory: fit -> covariance -> spectrum -> entropies.

    `meta`, when given, is checked by `assert_trajectory_identity` FIRST. Passing None
    skips that check and is legitimate only for synthetic trajectories built inside this
    package (the harmonic-limit test), which is why the record says which happened.
    """
    identity = assert_trajectory_identity(meta) if meta is not None else None
    fitted, mean, fit_rec = superimpose(positions, masses)
    cov = mass_weighted_covariance(fitted, masses, mean)
    spec = spectrum(cov, masses, mean, rank_tolerance=rank_tolerance)
    ent = entropy(spec["eigenvalues"], temperature_K)

    n_frames = int(len(fitted))
    n_expected = spec["expected_vibrational_modes"]
    return dict(
        identity_check=identity,
        identity_check_skipped_reason=(None if identity is not None else
                                       "meta not supplied -- synthetic trajectory"),
        superposition=fit_rec,
        spectrum={k: v for k, v in spec.items() if k != "eigenvalues"},
        entropy=ent,
        mean_structure_A=[[float(c) for c in row] for row in mean],
        # Acceptance criterion 9. Rinaldo & Field lost 4758 of 7758 modes to exactly
        # this limit and said so; the two numbers are therefore always printed together.
        rank_check=dict(
            n_frames=n_frames,
            frame_rank_limit=n_frames - 1,
            degrees_of_freedom_3N_minus_6=int(n_expected),
            n_nonzero_eigenvalues=int(spec["n_nonzero_eigenvalues"]),
            rank_is_full=bool(spec["n_nonzero_eigenvalues"] == n_expected),
            frames_are_the_limit=bool(n_frames - 1 < n_expected),
        ),
    )


# ======================================================================================
# Convergence diagnostics -- acceptance criteria 1 and 10
# ======================================================================================
def saturation_curve(positions, masses, temperature_K=thermo.T_REF,
                     fractions=(0.05, 0.1, 0.25, 0.5, 0.75, 1.0), meta=None):
    """S_QH computed on leading prefixes of one trajectory (acceptance criterion 1).

    Quasi-harmonic entropy rises monotonically with sampling time and saturates slowly.
    Any finite trajectory therefore gives a LOWER BOUND, and a trajectory that has not
    run long enough returns a value that looks perfectly stable and is wrong. The
    increment over the last doubling is the main term of the error bar, so it is
    computed rather than assumed, and the criterion is written on the trajectory length
    instead of the length being tuned until the criterion passes.
    """
    x = np.asarray(positions, dtype=float)
    if meta is not None:
        assert_trajectory_identity(meta)
    rows = []
    for f in fractions:
        n = max(3 * len(masses), int(round(f * len(x))))
        if n > len(x):
            n = len(x)
        rec = analyse(x[:n], masses, temperature_K)
        rows.append(dict(fraction=float(f), n_frames=int(n),
                         TS_QH_kcal=rec["entropy"]["TS_QH_kcal"],
                         TS_Schlitter_kcal=rec["entropy"]["TS_Schlitter_kcal"],
                         n_nonzero_eigenvalues=rec["spectrum"]["n_nonzero_eigenvalues"],
                         rank_is_full=rec["rank_check"]["rank_is_full"]))
    last_doubling = float("nan")
    for i in range(len(rows) - 1, 0, -1):
        if rows[i]["n_frames"] >= 2 * rows[i - 1]["n_frames"]:
            last_doubling = rows[i]["TS_QH_kcal"] - rows[i - 1]["TS_QH_kcal"]
            break
    if not np.isfinite(last_doubling) and len(rows) >= 2:
        last_doubling = rows[-1]["TS_QH_kcal"] - rows[-2]["TS_QH_kcal"]
    ts = [r["TS_QH_kcal"] for r in rows]
    return dict(points=rows,
                increment_over_last_doubling_kcal=float(last_doubling),
                monotonic=bool(all(b >= a - 1e-9 for a, b in zip(ts, ts[1:]))),
                total_rise_kcal=float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0)


def mode_batch_convergence(positions, masses, temperature_K=thermo.T_REF,
                           n_batches=5, fractions=(0.25, 0.5, 0.75, 1.0)):
    """Entropy of each batch of modes against trajectory length (acceptance criterion 10).

    Rinaldo & Field found their total entropy failing to converge while EVERY BATCH of
    modes converged quickly: what was still growing was the NUMBER of non-zero modes,
    not the value of any of them. The two diagnostics separate "the values have not
    settled" from "the modes have not all appeared", and they cost nothing extra. Our
    3N-6 is 18..51 against tens of thousands of frames, so that mechanism should not
    reach us -- which is a prediction, and this is how it gets tested.
    """
    x = np.asarray(positions, dtype=float)
    rows = []
    for f in fractions:
        n = min(len(x), max(3 * len(masses), int(round(f * len(x)))))
        rec = analyse(x[:n], masses, temperature_K)
        s = np.asarray(rec["entropy"]["S_per_mode_kcal_per_K"])
        batches = np.array_split(np.arange(len(s)), n_batches)   # sorted soft -> stiff
        rows.append(dict(fraction=float(f), n_frames=int(n),
                         n_nonzero_eigenvalues=rec["spectrum"]["n_nonzero_eigenvalues"],
                         batch_TS_kcal=[float(s[b].sum() * temperature_K)
                                        for b in batches]))
    drift = []
    if len(rows) >= 2:
        for k in range(n_batches):
            drift.append(float(rows[-1]["batch_TS_kcal"][k]
                               - rows[-2]["batch_TS_kcal"][k]))
    return dict(n_batches=int(n_batches), points=rows,
                last_step_drift_per_batch_kcal=drift,
                mode_count_grew=bool(len(rows) >= 2 and
                                     rows[-1]["n_nonzero_eigenvalues"]
                                     > rows[-2]["n_nonzero_eigenvalues"]))


# ======================================================================================
# The harmonic limit -- acceptance criterion 3, and it costs zero force evaluations
# ======================================================================================
def synthetic_harmonic_trajectory(hessian_eV_A2, masses, positions,
                                  temperature_K=thermo.T_REF, n_frames=100000,
                                  seed=20260903):
    """Sample the CLASSICAL canonical distribution of a harmonic well, analytically.

    For a classical harmonic oscillator in mass-weighted normal coordinates,
    <q_k^2> = k_B T / omega_k^2, so lambda_k = k_B T / omega_k^2 and
    nu_k = sqrt(k_B T / lambda_k) / (2 pi c) returns the Hessian frequency exactly. That
    identity is the whole content of the quasi-harmonic approximation, which makes this
    the sharpest available end-to-end test of the implementation: the answer is known in
    closed form and no potential is ever called.

    Note which distribution is sampled: CLASSICAL. The entropy is then evaluated with the
    QUANTUM oscillator formula, exactly as the production path does -- the quasi-harmonic
    method reads quantum entropies off classical fluctuations, and the test has to
    reproduce that seam rather than paper over it.
    """
    m = np.asarray(masses, dtype=float)
    x0 = np.asarray(positions, dtype=float)
    h = np.asarray(hessian_eV_A2, dtype=float)
    m3 = np.repeat(m, 3)
    hm = h / np.sqrt(np.outer(m3, m3))
    v, _s, rank = rigid_body_vectors(m, x0)
    p = np.eye(len(m3)) - v @ v.T
    hp = p @ hm @ p
    hp = 0.5 * (hp + hp.T)
    lam, vec = np.linalg.eigh(hp)

    overlap = np.linalg.norm(v.T @ vec, axis=0) ** 2
    keep = overlap <= 0.5
    if (lam[keep] <= 0).any():
        raise ValueError("the Hessian has {} non-positive vibrational eigenvalues -- "
                         "this is not a minimum".format(int((lam[keep] <= 0).sum())))
    lam_vib = lam[keep]
    vec_vib = vec[:, keep]

    # An eigenvalue of the mass-weighted Hessian is omega^2 in eV/(A^2 amu); k_B T in the
    # same unit system is (KB_SI / EV_J) * T.
    kb_ev = KB_SI / EV_J
    var = kb_ev * temperature_K / lam_vib               # <q^2>, amu * A^2
    rng = np.random.RandomState(int(seed))
    q = rng.normal(size=(int(n_frames), len(lam_vib))) * np.sqrt(var)[None, :]
    disp_mw = q @ vec_vib.T                             # (T, 3N), mass-weighted
    disp = disp_mw / np.sqrt(m3)[None, :]
    frames = x0[None] + disp.reshape(-1, len(m), 3)

    nu_exact = np.sqrt(lam_vib) * CM_INV_PER_SQRT_EV_A2_AMU
    s_exact = float(sum(thermo.s_mode_kcal_per_K(float(x), temperature_K)
                        for x in nu_exact))
    return frames, dict(n_frames=int(n_frames), seed=int(seed),
                        rigid_subspace_rank=int(rank),
                        n_modes=int(len(lam_vib)),
                        frequencies_cm_inv=[float(x) for x in np.sort(nu_exact)],
                        closed_form_S_kcal_per_K=s_exact,
                        closed_form_TS_kcal=float(s_exact * temperature_K),
                        temperature_K=float(temperature_K),
                        distribution="classical canonical, sampled exactly")


def harmonic_limit_check(hessian_eV_A2, masses, positions,
                         temperature_K=thermo.T_REF, n_frames=100000, seed=20260903):
    """Run the synthetic trajectory through the production chain and report the error.

    Two numbers come back and they answer different questions:

      * `exact_covariance_TS_error_kcal` uses lambda_k = k_B T / omega_k^2 directly, with
        no sampling at all. It isolates the ALGEBRA and should sit at machine precision.
        If it does not, the bug is in this module.
      * `sampled_TS_error_kcal` runs the fitted-covariance chain over finite frames. It
        carries the statistical error of the estimator, and it is the number acceptance
        criterion 3 is written on (< 0.02 kcal/mol).
    """
    frames, ref = synthetic_harmonic_trajectory(
        hessian_eV_A2, masses, positions, temperature_K, n_frames, seed)

    nu = np.asarray(ref["frequencies_cm_inv"], dtype=float)
    omega_si = 2 * np.pi * C_CM_S * nu
    lam_exact = (KB_SI * temperature_K / omega_si ** 2) / (AMU_KG * ANG_M ** 2)
    ent_exact = entropy(lam_exact, temperature_K)

    rec = analyse(frames, masses, temperature_K)
    nu_sampled = np.sort(np.asarray(rec["entropy"]["frequencies_cm_inv"], dtype=float))
    n = min(len(nu), len(nu_sampled))
    return dict(
        reference=ref,
        exact_covariance=ent_exact,
        exact_covariance_TS_error_kcal=float(ent_exact["TS_QH_kcal"]
                                             - ref["closed_form_TS_kcal"]),
        sampled=rec,
        sampled_TS_error_kcal=float(rec["entropy"]["TS_QH_kcal"]
                                    - ref["closed_form_TS_kcal"]),
        max_frequency_error_cm_inv=float(np.abs(nu_sampled[:n] - nu[:n]).max()),
        rms_frequency_error_cm_inv=float(np.sqrt(
            ((nu_sampled[:n] - nu[:n]) ** 2).mean())),
        n_modes_reference=int(len(nu)), n_modes_recovered=int(len(nu_sampled)),
    )


# ======================================================================================
# Assembly -- the only difference from branch C is the vibrational term
# ======================================================================================
def g_minus_eel(masses, positions, eigenvalues_amu_A2, symmetry_number, degeneracy,
                temperature_K=thermo.T_REF, pressure_Pa=thermo.P_STD):
    """(G - E_el) with the vibrational term taken from quasi-harmonic frequencies.

    It calls `thermo.g_minus_eel`, the same function branch C calls, with the same
    explicit symmetry number and electronic degeneracy. That is not a convenience: it is
    what makes acceptance criterion 7 true by construction -- the translational,
    rotational and electronic terms are bit-for-bit identical between the two routes, so
    the difference between them is the vibrational treatment and nothing else.
    """
    nu = frequencies_cm_inv(eigenvalues_amu_A2, temperature_K)
    nu = nu[np.isfinite(nu)]
    out = thermo.g_minus_eel(masses, positions, nu, symmetry_number, degeneracy,
                             temperature_K=temperature_K, pressure_Pa=pressure_Pa,
                             qrrho=False)
    out["vibrational_term_source"] = ("quasi-harmonic frequencies from the position "
                                      "covariance (branch B), NOT a Hessian")
    out["quasiharmonic_frequencies_cm_inv"] = [float(x) for x in np.sort(nu)]
    return out
