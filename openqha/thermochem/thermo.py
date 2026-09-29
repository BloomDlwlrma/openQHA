"""Rigid-rotor harmonic-oscillator thermodynamics, and the conversion from a frequency
error to a free-energy error.

All four terms (lecture notes chapters 1 and 4):

    A = A_trans + A_rot + A_vib + A_elec
    G = A + pV = A + k_B T   (ideal gas, per mole)

In an isomerisation with Delta n = 0, `A_trans` and `pV` cancel **digit for digit**
between the two sides, so whether they enter the final difference does not change the
result -- but they must be COMPUTED and SHOWN to cancel, not quietly left out.

**The symmetry number and the electronic degeneracy are always declared explicitly and
never derived automatically** (getting the value wrong is worth 0.65-1.06 kcal/mol).
"""
import math

import numpy as np

KB_KCAL = 1.987204259e-3            # kcal/(mol*K)
HC_KCAL = 2.85914308e-3             # kcal/mol per cm^-1
H_SI = 6.62607015e-34
KB_SI = 1.380649e-23
NA = 6.02214076e23
AMU_KG = 1.66053906660e-27
P_STD = 1.0e5                       # Pa, 1 bar
J_TO_KCAL = 1.0 / 4184.0
T_REF = 298.15
# With I in amu*A^2:  B[cm^-1] = ROT_CONST_AMU_A2 / I
ROT_CONST_AMU_A2 = 16.857629046

# Interpolation frequency of Grimme's quasi-rigid-rotor harmonic oscillator. A literature
# value (J. Chem. Eur. 2012, 18, 9955), and a convention that may be changed.
QRRHO_NU0_CM = 100.0


# ------------------------------------------------------------------ vibrational
#: Above this, exp(x) overflows a double (math.expm1 raises at x ~ 709.78) and the
#: mode is frozen to every digit anyway: x/expm1(x) and log1p(-exp(-x)) are both below
#: 1e-300. Measured 2026-09-13 on an113: a covariance from 30 frames for 30 coordinates
#: has eigenvalues that are positive only by rounding, nu of order 1e15 cm^-1, and the
#: entropy sum raised OverflowError instead of reporting a rank-deficient spectrum.
X_FROZEN = 700.0


def _x(nu_cm, temperature_K):
    return HC_KCAL * nu_cm / (KB_KCAL * temperature_K)


def a_mode_kcal(nu_cm, temperature_K=T_REF):
    """Helmholtz free energy of one quantum harmonic oscillator, zero-point energy
    included, with the zero taken at the bottom of the well."""
    kt = KB_KCAL * temperature_K
    x = _x(nu_cm, temperature_K)
    if x > X_FROZEN:
        return kt * 0.5 * x                 # log1p(-exp(-x)) is -0.0 to double precision
    return kt * (0.5 * x + math.log1p(-math.exp(-x)))


def e_mode_kcal(nu_cm, temperature_K=T_REF):
    kt = KB_KCAL * temperature_K
    x = _x(nu_cm, temperature_K)
    if x > X_FROZEN:
        return kt * 0.5 * x                 # x/expm1(x) is 0.0 to double precision
    return kt * (0.5 * x + x / math.expm1(x))


def s_mode_kcal_per_K(nu_cm, temperature_K=T_REF):
    x = _x(nu_cm, temperature_K)
    if x > X_FROZEN:
        return 0.0                          # a frozen mode carries no entropy
    return KB_KCAL * (x / math.expm1(x) - math.log1p(-math.exp(-x)))


def vibrational(frequencies_cm, temperature_K=T_REF, qrrho=False,
                nu0_cm=QRRHO_NU0_CM, moments_amu_A2=None):
    """A / E / S / zero-point energy over all vibrational modes.

    This helper is strict on purpose: a non-positive wavenumber is rejected here, and it
    is never "fixed" by taking an absolute value. The imaginary-mode policies live one
    level up (`msrrho`, `g_minus_eel`), which apply the inversion / sub-1 cm^-1 drop
    rule to the spectrum BEFORE calling this function -- thermochemistry is never
    computed with projection off.
    """
    nu = np.asarray(frequencies_cm, dtype=float)
    bad = nu[nu <= 0.0]
    if bad.size:
        raise ValueError("{} non-positive frequency/frequencies (lowest {:.2f} cm^-1) "
                         "-- this is not a minimum, refusing to compute thermodynamics "
                         "on it".format(bad.size, float(nu.min())))
    a = float(sum(a_mode_kcal(x, temperature_K) for x in nu))
    e = float(sum(e_mode_kcal(x, temperature_K) for x in nu))
    s = float(sum(s_mode_kcal_per_K(x, temperature_K) for x in nu))
    zpe = float(0.5 * HC_KCAL * nu.sum())
    out = dict(A_vib_kcal=a, E_vib_kcal=e, S_vib_kcal_per_K=s, ZPE_kcal=zpe,
               n_modes=int(nu.size), lowest_frequency_cm_inv=float(nu.min()))
    if qrrho:
        out.update(_qrrho(nu, temperature_K, nu0_cm, moments_amu_A2))
    return out


def _qrrho(nu, temperature_K, nu0_cm, moments_amu_A2):
    """Grimme's quasi-rigid-rotor harmonic oscillator: interpolates **the entropy only**
    between a free rotor and a harmonic oscillator.

    The weight is w(nu) = 1 / (1 + (nu0/nu)^4). The internal energy is not interpolated,
    so in the low-frequency limit it still differs from the exact solution by k_B T / 2
    (measured in this project: largest residual 0.2919 against the analytic
    kT/2 = 0.2962 kcal/mol).
    """
    if moments_amu_A2 is None:
        b_av = 1.0e-44
    else:
        b_av = float(np.mean(moments_amu_A2)) * AMU_KG * 1e-20
    kt_si = KB_SI * temperature_K
    s_free = []
    for x in nu:
        mu = H_SI / (8.0 * math.pi ** 2 * (x * 2.99792458e10))
        mu_eff = mu * b_av / (mu + b_av)
        s = KB_SI * (0.5 + math.log(math.sqrt(
            8.0 * math.pi ** 3 * mu_eff * kt_si) / H_SI))
        s_free.append(s * NA * J_TO_KCAL)
    w = 1.0 / (1.0 + (nu0_cm / nu) ** 4)
    s_ho = np.array([s_mode_kcal_per_K(x, temperature_K) for x in nu])
    s_q = float((w * s_ho + (1 - w) * np.array(s_free)).sum())
    e = float(sum(e_mode_kcal(x, temperature_K) for x in nu))
    return dict(S_vib_qrrho_kcal_per_K=s_q,
                A_vib_qrrho_kcal=e - temperature_K * s_q,
                qrrho_nu0_cm=float(nu0_cm),
                qrrho_note="entropy interpolated, internal energy not -- residual in "
                           "the low-frequency limit = kT/2")


# ------------------------------------------------------------------ msRRHO presets
#: The three published parameterisations of the rigid-rotor / harmonic-oscillator
#: interpolation for low modes. `tau_cm` is the crossover frequency of the Chai /
#: Head-Gordon weight w = 1 / (1 + (tau/omega)^4); `ithr_cm` the threshold below which
#: an imaginary mode may be inverted under the `invert_below` policy (None: never);
#: `rotor_cap` is what the free-rotor moment of inertia is capped by -- the molecule's
#: mean principal moment (CREST `axis_module.f90:173`) or Grimme's fixed 1e-44 kg m^2;
#: `interpolate` names the quantities that take the weight. Enthalpy and zero-point
#: energy are never interpolated by any of them.
#:
#:   crest       Pracht & Grimme, Chem. Sci. 2021, 12, 6551; `classes.f90:276-278`
#:   xtb         the xtb `$thermo` defaults
#:   grimme2012  Grimme, Chem. Eur. J. 2012, 18, 9955
MSRRHO_PRESETS = {
    "crest": dict(tau_cm=25.0, ithr_cm=-50.0, rotor_cap="mean_principal_moment",
                  interpolate=("S", "Cp")),
    "xtb": dict(tau_cm=50.0, ithr_cm=-20.0, rotor_cap="mean_principal_moment",
                interpolate=("S", "Cp")),
    "grimme2012": dict(tau_cm=100.0, ithr_cm=None, rotor_cap=1.0e-44,
                       interpolate=("S",)),
}
#: The sub-1 cm^-1 rule: CREST's `vibthr` and ORCA's `CutOffFreq 1.0` drop
#: |omega| < 1 cm^-1 from every thermochemistry sum. One or two such modes in one
#: spectrum are dropped ORCA-style and recorded (`N_BELOW_FLOOR` plus the values); three
#: or more raise -- that count has no plausible physical explanation at a converged,
#: projected minimum and is the signature of a spectrum without projection.
VIBTHR_CM = 1.0
#: Per-mode table cut-off in the report (CREST prints up to max(300, w=0.99 point)).
MODE_TABLE_MAX_CM = 300.0
C_CM_PER_S = 2.99792458e10


def floor_verdict(lowest_cm, ithr_cm):
    """A spectrum's standing against the frequency floor `ithr` -- the one rule the
    census and the reference level call, and the rule the thermochemistry's own policy
    layer applies:

    `below_floor`  the lowest mode is below ithr: not invertible. The census ejects the
                   candidate, the reference level lists it as a saddle, the ensemble
                   excludes the basin.
    `window`       the lowest mode is in [ithr, 0): inverted under the production policy.
    `clean`        every mode is positive: the policy has nothing to do.
    """
    if lowest_cm < ithr_cm:
        return "below_floor"
    if lowest_cm < 0.0:
        return "window"
    return "clean"


def n_below_ithr(frequencies_cm, ithr_cm):
    """Modes below the floor (strictly: a mode exactly on the line is in the window and
    invertible, as `floor_verdict` and the policy layer both have it); 0 for an admitted
    basin."""
    nu = np.asarray(frequencies_cm, dtype=float)
    return int((nu < ithr_cm).sum())


def n_in_window(frequencies_cm, ithr_cm):
    """Modes in [ithr, 0): the geometric inversion window. A mode with |omega| < 1 cm^-1
    inside it is in the window by value but the sub-1 rule drops it rather than inverting
    it; the drop has its own counter (`n_below_floor`)."""
    nu = np.asarray(frequencies_cm, dtype=float)
    return int(((nu >= ithr_cm) & (nu < 0.0)).sum())


def _free_rotor_s_kcal_per_K(nu_cm, rotor_cap_kg_m2, temperature_K):
    """Entropy of one free rotor with sigma = 1 whose moment is the oscillator's
    mu = h / (8 pi^2 c nu), capped as mu B / (mu + B). CREST `thermodyn` line for line."""
    mu = H_SI / (8.0 * math.pi ** 2 * C_CM_PER_S * nu_cm)
    mu_eff = mu * rotor_cap_kg_m2 / (mu + rotor_cap_kg_m2)
    s = KB_SI * (0.5 + math.log(math.sqrt(
        8.0 * math.pi ** 3 * mu_eff * KB_SI * temperature_K) / H_SI))
    return s * NA * J_TO_KCAL


def _cp_ho_kcal_per_K(nu_cm, temperature_K):
    x = _x(nu_cm, temperature_K)
    if x > X_FROZEN:
        return 0.0
    ex = math.exp(-x)
    return KB_KCAL * x * x * ex / (1.0 - ex) ** 2


#: The imaginary-mode policies (one production policy plus the GFN2 seam's):
#:
#: `invert_below`  PRODUCTION, the default everywhere. A mode in [ithr, 0) takes |omega|
#:                 (the floor line itself is inside the window); a mode below ithr is not
#:                 invertible, so the structure has no thermochemistry under this policy
#:                 and the caller excludes it with the reason recorded. `ithr = -50 cm^-1`
#:                 is CREST's `-ithr` default (the `crest` preset's, see MSRRHO_PRESETS).
#: `crest_native`  the GFN2 seam only. CREST 3.0.2 line for line (`thermocalc.f90:207-216`,
#:                 `thermo.f90:135-138`; measured on propanal 2026-09-16): a mode in
#:                 [ithr, 0) is inverted; a mode below ithr is KEPT negative, carries
#:                 zero entropy, and still enters the zero-point energy (0.5 sum nu),
#:                 H(T)-H(0) and Cp with its negative frequency.
#:
#: `refuse` (any imaginary mode excluded the basin) was deleted on 2026-09-25:
#: the spread over the three policies on real data showed no molecule needed it. Records
#: written before that date that name it stay readable as data.
#: The one production policy: the default of every entry point in this package, taken
#: as a named constant so the signatures cannot drift apart.
PRODUCTION_POLICY = "invert_below"
IMAGINARY_POLICIES = (PRODUCTION_POLICY, "crest_native")


def _apply_imaginary_policy(nu, policy, ithr_cm):
    """Return `(frequencies, counters)` or raise.

    A negative mode the sub-1 rule does not remove needs a policy that can act on it, so
    the preset's applicability is decided first: no ithr at all raises (nothing can be
    said about a mode below zero), and under `invert_below` a mode below ithr raises --
    the basin has no thermochemistry under the production floor.

    The sub-1 cm^-1 rule then applies for every policy: a mode with |omega| < VIBTHR_CM
    is removed from every thermochemistry sum (S, Cp, H, ZPE) and counted, ORCA's
    `CutOffFreq`-style drop. THREE or more such modes in one spectrum raise instead --
    a converged, projected minimum has no plausible spectrum with three rigid-body
    leftovers, and silent dropping must not absorb the signature of a spectrum that was
    never projected.

    Then the policy (see IMAGINARY_POLICIES): `invert_below` takes |omega| for modes in
    [ithr, 0); `crest_native` takes |omega| in [ithr, 0) and KEEPS modes below ithr
    negative, exactly as CREST does.

    `counters` is a dict: n_inverted, n_kept_negative, n_below_floor,
    inverted_frequencies_cm, dropped_frequencies_cm.
    """
    nu = np.asarray(nu, dtype=float)
    if policy not in IMAGINARY_POLICIES:
        raise ValueError("imaginary_policy must be one of {}, received {!r}".format(
            ", ".join(IMAGINARY_POLICIES), policy))
    tiny = np.abs(nu) < VIBTHR_CM
    stays = ~tiny
    needs_policy = bool((nu[stays] < 0.0).any())   # a negative the drop does not remove
    if needs_policy and ithr_cm is None:
        raise ValueError("this preset defines no ithr; '{}' is not available".format(policy))
    if policy == "invert_below" and needs_policy and float(nu[stays].min()) < ithr_cm:
        raise ValueError("imaginary mode {:.2f} cm^-1 is below ithr = {:.1f} cm^-1: not "
                         "invertible".format(float(nu[stays].min()), ithr_cm))
    n_tiny = int(tiny.sum())
    if n_tiny >= 3:
        raise ValueError("unprojected signature: {} mode(s) with |omega| < {} cm^-1 in "
                         "one spectrum (|omega| = {}) -- refusing to compute "
                         "thermochemistry on it".format(
                             n_tiny, VIBTHR_CM,
                             ", ".join("{:.2f}".format(abs(x)) for x in nu[tiny])))
    counts = dict(n_inverted=0, n_kept_negative=0, n_below_floor=n_tiny,
                  inverted_frequencies_cm=[],
                  dropped_frequencies_cm=[float(x) for x in nu[tiny]])
    out = nu[stays].copy()
    neg = out < 0.0
    if not neg.any():
        return out, counts
    # the floor line itself is inside the window (floor_verdict says the same): below the
    # floor is strictly lower than ithr, and the min check above has already refused it
    invertible = neg & (out >= ithr_cm)
    below = neg & ~invertible
    counts["inverted_frequencies_cm"] = [float(x) for x in out[invertible]]
    counts["n_inverted"] = int(invertible.sum())
    counts["n_kept_negative"] = int(below.sum())
    out[invertible] = np.abs(out[invertible])
    return out, counts


def msrrho(frequencies_cm, masses, positions, preset="crest", temperature_K=T_REF,
           imaginary_policy=PRODUCTION_POLICY, fscal=1.0):
    """The modified (and scaled) RRHO vibrational term of one basin under a preset.

    Per mode: S = w S_HO + (1 - w) S_FR and, for presets that say so, Cp likewise;
    H(T) - H(0) and the zero-point energy are always harmonic. The free rotor has
    sigma = 1 and a capped moment; see MSRRHO_PRESETS. Returns the totals, the per-mode
    table (ascending omega) and every convention that produced the number -- including
    the policy's counters: `n_inverted`, `n_kept_negative`, `n_below_floor` and the
    inverted / dropped frequencies themselves.
    """
    if preset not in MSRRHO_PRESETS:
        raise ValueError("unknown msRRHO preset {!r}; known: {}".format(
            preset, ", ".join(sorted(MSRRHO_PRESETS))))
    p = MSRRHO_PRESETS[preset]
    nu = np.asarray(frequencies_cm, dtype=float) * float(fscal)
    nu, counts = _apply_imaginary_policy(nu, imaginary_policy, p["ithr_cm"])
    nu = np.sort(nu)
    if p["rotor_cap"] == "mean_principal_moment":
        cap = float(np.mean(principal_moments(masses, positions))) * AMU_KG * 1e-20
    else:
        cap = float(p["rotor_cap"])
    tau = float(p["tau_cm"])
    modes, s_tot, s_ho_tot, cp_tot, h_tot, zpe = [], 0.0, 0.0, 0.0, 0.0, 0.0
    for w_nu in nu:
        # CREST's switching function takes (tau/omega)^4, the same for a negative omega
        w_ho = 1.0 / (1.0 + (tau / w_nu) ** 4)
        if w_nu > 0.0:
            s_ho = s_mode_kcal_per_K(w_nu, temperature_K)
            s_fr = _free_rotor_s_kcal_per_K(w_nu, cap, temperature_K)
        else:
            # a kept negative mode (crest_native): `thermo.f90:135-138`, no entropy
            s_ho = s_fr = 0.0
        s = w_ho * s_ho + (1.0 - w_ho) * s_fr
        # Cp, H(T)-H(0) and the zero-point energy take the frequency as it is, negative
        # included, exactly as CREST evaluates them (exp(-beta omega) > 1 for omega < 0)
        cp_ho = _cp_ho_kcal_per_K(w_nu, temperature_K)
        cp = w_ho * cp_ho + (1.0 - w_ho) * 0.5 * KB_KCAL if "Cp" in p["interpolate"] else cp_ho
        h = e_mode_kcal(w_nu, temperature_K) - 0.5 * HC_KCAL * w_nu
        s_tot += s
        s_ho_tot += s_ho
        cp_tot += cp
        h_tot += h
        zpe += 0.5 * HC_KCAL * w_nu
        modes.append(dict(omega_cm=float(w_nu), w_HO=float(w_ho),
                          TS_HO_kcal=float(temperature_K * s_ho),
                          TS_FR_kcal=float(temperature_K * s_fr),
                          TS_kcal=float(temperature_K * s),
                          kept_negative=bool(w_nu < 0.0)))
    return dict(preset=preset, tau_cm=tau, ithr_cm=p["ithr_cm"],
                rotor_cap_kg_m2=cap, rotor_cap_rule=p["rotor_cap"],
                interpolated=list(p["interpolate"]), fscal=float(fscal),
                imaginary_policy=imaginary_policy,
                n_inverted=counts["n_inverted"], n_kept_negative=counts["n_kept_negative"],
                n_below_floor=counts["n_below_floor"],
                inverted_frequencies_cm=counts["inverted_frequencies_cm"],
                dropped_frequencies_cm=counts["dropped_frequencies_cm"],
                temperature_K=float(temperature_K),
                S_vib_kcal_per_K=float(s_tot), S_vib_HO_kcal_per_K=float(s_ho_tot),
                TS_vib_kcal=float(temperature_K * s_tot),
                TS_vib_HO_kcal=float(temperature_K * s_ho_tot),
                Cp_vib_kcal_per_K=float(cp_tot), H_thermal_kcal=float(h_tot),
                ZPE_kcal=float(zpe),
                A_vib_kcal=float(zpe + h_tot - temperature_K * s_tot),
                n_modes=int(nu.size), lowest_frequency_cm_inv=float(nu.min()),
                modes=modes)


def preset_spread(frequencies_cm, masses, positions, temperature_K=T_REF,
                  imaginary_policy=PRODUCTION_POLICY, fscal=1.0):
    """T*S_vib under every preset, and the spread: the error-bar line for the choice of
    convention (on acetone 0.19 kcal/mol between `crest` and `grimme2012`)."""
    ts, absent = {}, {}
    for name in MSRRHO_PRESETS:
        try:
            r = msrrho(frequencies_cm, masses, positions, preset=name,
                       temperature_K=temperature_K, imaginary_policy=imaginary_policy,
                       fscal=fscal)
        except ValueError as exc:
            # a preset that cannot apply this policy to this spectrum (grimme2012 defines
            # no ithr; crest / xtb call a below-floor mode not invertible) is absent from
            # the spread, and says why, rather than the whole record failing after the
            # assembly succeeded
            absent[name] = str(exc)
            continue
        ts[name] = r["TS_vib_kcal"]
        ts["HO"] = r["TS_vib_HO_kcal"]      # the same for every preset
    if not ts:
        raise ValueError("no preset could evaluate this spectrum under '{}': {}".format(
            imaginary_policy, "; ".join(absent.values())))
    return dict(TS_vib_kcal=ts, max_minus_min_kcal=float(max(ts.values()) - min(ts.values())),
                temperature_K=float(temperature_K), imaginary_policy=imaginary_policy,
                presets_absent=absent)


# ------------------------------------------------------------------ rotational
def principal_moments(masses, positions):
    """Principal moments of inertia in amu*A^2, ascending."""
    m = np.asarray(masses, dtype=float)
    r = np.asarray(positions, dtype=float)
    r = r - (m[:, None] * r).sum(0) / m.sum()
    i = np.zeros((3, 3))
    for mi, ri in zip(m, r):
        i += mi * (np.dot(ri, ri) * np.eye(3) - np.outer(ri, ri))
    return np.sort(np.linalg.eigvalsh(i))


def rotational(masses, positions, symmetry_number, temperature_K=T_REF):
    """Classical rigid rotor.

    The symmetry number **must be supplied explicitly by the caller**; it is neither
    derived here nor defaulted.
    """
    if symmetry_number is None:
        raise ValueError("the external symmetry number is not declared -- refusing to "
                         "compute the rotational term")
    sigma = int(symmetry_number)
    if sigma < 1:
        raise ValueError("the external symmetry number must be a positive integer, "
                         "received {}".format(symmetry_number))
    i_amu = principal_moments(masses, positions)
    if i_amu[0] < 1e-6:                       # linear molecule
        raise NotImplementedError("the rotational term of a linear molecule needs "
                                  "separate treatment; this system has no linear "
                                  "molecules")
    b_cm = ROT_CONST_AMU_A2 / i_amu           # cm^-1
    theta = b_cm * HC_KCAL / KB_KCAL          # K
    q = (math.sqrt(math.pi) / sigma) * math.sqrt(
        temperature_K ** 3 / (theta[0] * theta[1] * theta[2]))
    a = -KB_KCAL * temperature_K * math.log(q)
    return dict(A_rot_kcal=float(a), q_rot=float(q),
                symmetry_number=sigma,
                moments_amu_A2=[float(x) for x in i_amu],
                rotational_constants_cm_inv=[float(x) for x in b_cm],
                rotational_temperatures_K=[float(x) for x in theta],
                E_rot_kcal=1.5 * KB_KCAL * temperature_K,
                S_rot_kcal_per_K=float(KB_KCAL * (math.log(q) + 1.5)),
                T_over_theta_max=float(temperature_K / theta.max()))


# ------------------------------------------------------- translational and electronic
def translational(mass_amu, temperature_K=T_REF, pressure_Pa=P_STD, kind="gibbs"):
    """Sackur-Tetrode.

    `kind="helmholtz"` includes Stirling's +1; `kind="gibbs"` does not (= A + kT).
    """
    if kind not in ("gibbs", "helmholtz"):
        raise ValueError("kind must be 'gibbs' or 'helmholtz', received {!r}".format(kind))
    m = float(mass_amu) * AMU_KG
    lam = H_SI / math.sqrt(2.0 * math.pi * m * KB_SI * temperature_K)
    v = KB_SI * temperature_K / pressure_Pa
    stirling = 1.0 if kind == "helmholtz" else 0.0
    a_j = -KB_SI * temperature_K * (math.log(v / lam ** 3) + stirling)
    return dict(value_kcal=float(a_j * NA * J_TO_KCAL), kind=kind,
                thermal_wavelength_A=float(lam / 1e-10),
                volume_per_molecule_A3=float(v / 1e-30),
                mass_amu=float(mass_amu), pressure_Pa=float(pressure_Pa))


def electronic(degeneracy, temperature_K=T_REF):
    """A_elec = -kT ln g0.

    g0 **must be declared explicitly** and is never guessed from the structure.
    """
    if degeneracy is None:
        raise ValueError("the electronic ground-state degeneracy is not declared -- "
                         "refusing to compute the electronic term")
    g = int(degeneracy)
    if g < 1:
        raise ValueError("the degeneracy must be a positive integer, received {}".format(
            degeneracy))
    return dict(A_elec_kcal=float(-KB_KCAL * temperature_K * math.log(g)),
                degeneracy=g)


# ------------------------------------------------------------------ assembly
def g_minus_eel(masses, positions, frequencies_cm, symmetry_number, degeneracy,
                temperature_K=T_REF, pressure_Pa=P_STD, qrrho=False,
                imaginary_policy=PRODUCTION_POLICY):
    """(G - E_el) for one species, all four terms, each kept in the record.

    The imaginary-mode policy is applied to the spectrum BEFORE the strict sums
    (`vibrational` itself stays strict, so thermochemistry is never computed with
    projection off): under the production `invert_below` -- floor = the `crest` preset's
    `ithr` = -50 cm^-1 -- a mode in [ithr, 0) is inverted, a mode with |omega| < 1 cm^-1
    is dropped ORCA-style and counted, and a mode below the floor raises. A caller that
    gets that raise (the census, branch A's labels) has its answer: this structure has no
    thermochemistry under the production policy, and it records the reason.
    """
    nu, counts = _apply_imaginary_policy(np.asarray(frequencies_cm, dtype=float),
                                         imaginary_policy,
                                         MSRRHO_PRESETS["crest"]["ithr_cm"])
    vib = vibrational(nu, temperature_K, qrrho=qrrho,
                      moments_amu_A2=principal_moments(masses, positions))
    rot = rotational(masses, positions, symmetry_number, temperature_K)
    tr = translational(float(np.sum(masses)), temperature_K, pressure_Pa, kind="helmholtz")
    el = electronic(degeneracy, temperature_K)
    kt = KB_KCAL * temperature_K
    a_total = vib["A_vib_kcal"] + rot["A_rot_kcal"] + tr["value_kcal"] + el["A_elec_kcal"]
    out = dict(temperature_K=temperature_K, pressure_Pa=pressure_Pa,
               vibrational=vib, rotational=rot, translational=tr, electronic=el,
               A_minus_Eel_kcal=float(a_total),
               pV_kcal=float(kt),
               G_minus_Eel_kcal=float(a_total + kt),
               imaginary_policy=imaginary_policy,
               n_inverted=counts["n_inverted"], n_below_floor=counts["n_below_floor"],
               inverted_frequencies_cm=counts["inverted_frequencies_cm"],
               dropped_frequencies_cm=counts["dropped_frequencies_cm"])
    if qrrho and "A_vib_qrrho_kcal" in vib:
        d = vib["A_vib_qrrho_kcal"] - vib["A_vib_kcal"]
        out["G_minus_Eel_qrrho_kcal"] = float(a_total + kt + d)
    return out


# ------------------------------------------------------------------ error conversion
def da_dnu_kcal_per_cm(nu_cm, temperature_K=T_REF):
    """dA_mode/dnu = hc/2 + hc/(exp(beta hc nu) - 1).

    The low-frequency end carries far more weight than the high-frequency end.
    """
    x = HC_KCAL * np.asarray(nu_cm, dtype=float) / (KB_KCAL * temperature_K)
    return HC_KCAL * (0.5 + 1.0 / np.expm1(x))


def error_bar_from_frequency_error(frequencies_cm, sigma_cm, temperature_K=T_REF):
    """Free-energy uncertainty implied by a per-mode frequency error sigma.

    Both extremes are reported:

    * uncorrelated (random):   sqrt( sum_i (dA/dnu_i)^2 ) * sigma
    * fully correlated (same sign, same size, worst case): sum_i |dA/dnu_i| * sigma

    The truth lies between them; **neither of them is "measured"** -- what is actually
    measured is `direct_shift`, which needs no assumption about correlation at all.
    """
    d = da_dnu_kcal_per_cm(frequencies_cm, temperature_K)
    return dict(sigma_cm_inv=float(sigma_cm),
                uncorrelated_kcal=float(np.sqrt((d ** 2).sum()) * sigma_cm),
                fully_correlated_kcal=float(np.abs(d).sum() * sigma_cm),
                per_mode_sensitivity_kcal_per_cm=[float(x) for x in d])


def direct_shift(frequencies_a_cm, frequencies_b_cm, temperature_K=T_REF):
    """Difference in A_vib between two sets of frequencies -- **a measured quantity that
    needs no assumption about correlation**.

    The two sets must be paired mode by mode (same geometry, same mode ordering), or the
    difference means nothing.
    """
    a = np.asarray(frequencies_a_cm, dtype=float)
    b = np.asarray(frequencies_b_cm, dtype=float)
    if a.shape != b.shape:
        raise ValueError("the two sets have different numbers of modes: {} vs {}".format(
            a.shape, b.shape))
    av = sum(a_mode_kcal(x, temperature_K) for x in a)
    bv = sum(a_mode_kcal(x, temperature_K) for x in b)
    dev = a - b
    return dict(A_vib_a_kcal=float(av), A_vib_b_kcal=float(bv),
                delta_A_vib_kcal=float(av - bv),
                mean_absolute_deviation_cm_inv=float(np.abs(dev).mean()),
                root_mean_square_deviation_cm_inv=float(np.sqrt((dev ** 2).mean())),
                max_absolute_deviation_cm_inv=float(np.abs(dev).max()),
                signed_mean_deviation_cm_inv=float(dev.mean()),
                per_mode_deviation_cm_inv=[float(x) for x in dev])
