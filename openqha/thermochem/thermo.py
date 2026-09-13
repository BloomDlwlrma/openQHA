"""Rigid-rotor harmonic-oscillator thermodynamics, and the conversion from a frequency
error to a free-energy error.

All four terms (lecture notes chapters 1 and 4):

    A = A_trans + A_rot + A_vib + A_elec
    G = A + pV = A + k_B T   (ideal gas, per mole)

In an isomerisation with Delta n = 0, `A_trans` and `pV` cancel **digit for digit**
between the two sides, so whether they enter the final difference does not change the
result -- but they must be COMPUTED and SHOWN to cancel, not quietly left out.

**The symmetry number and the electronic degeneracy are always declared explicitly and
never derived automatically** (plan section 7, risk 6: getting the value wrong is worth
0.65-1.06 kcal/mol).
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

    An imaginary frequency (negative wavenumber) is rejected; it is never "fixed" by
    taking an absolute value.
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
                temperature_K=T_REF, pressure_Pa=P_STD, qrrho=False):
    """(G - E_el) for one species, all four terms, each kept in the record."""
    vib = vibrational(frequencies_cm, temperature_K, qrrho=qrrho,
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
               G_minus_Eel_kcal=float(a_total + kt))
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
