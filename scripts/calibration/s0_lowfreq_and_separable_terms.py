"""stage 0 -- explicit assertions for the separable terms, and four treatments of the low-frequency modes compared.

CALIBRATION. Explicit assertions on the separable terms, plus four treatments of a
low-frequency mode. Its output is the size of a contribution to the error bar.

It answers two questions:

  Question 1  Delta G must carry all four of translation / rotation / vibration / electronic. Do the lecture notes have only vibration?
          -> No. Rotation is computed in the lecture notes section 6.6 (rigid rotor + an explicit external symmetry number);
             translation and electronic degeneracy are *claimed* in section 2.1 to "cancel exactly", but that is only a **claim in prose**.
             The standing rule is "it must be asserted explicitly, not merely stated in a comment" --
             part A of this script turns those two claims into assertions that can fail.

  Question 2  A low-frequency mode weighs about 4 times a high-frequency one. How is the low-frequency end solved?
          -> Parts B / C / D: compute the free energy of one methyl rotor under four models
             (harmonic oscillator / free rotor / the exact one-dimensional hindered rotor / Grimme quasi-RRHO),
             with two limiting checks. The spread between them is the contribution of "low-frequency treatment" to the error bar.

Usage:
    python scripts/calibration/s0_lowfreq_and_separable_terms.py

Products:
    analysis/lowfreq_and_separable_terms.json
    analysis/lowfreq_hindered_rotor.png

The parameter classes are in PARAMETER_TABLE at the end of this file.
This script calls no potential, so it is not blocked by Egret-1 being absent.
"""
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")               # this script only writes files and never displays; Agg must NOT be set in the lecture notes
import matplotlib.pyplot as plt

def _repo_root():
    """Directory holding the openQHA package, found by walking up from this file.

    Depth-independent on purpose: this file keeps working wherever under the
    repository it is moved to. The earlier move into `scripts/_superseded/` broke
    every `parents[1]` in the moved files silently, which is what this replaces.
    """
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


# ---------------------------------------------------------------- physical constants
KB_KCAL = 1.987204259e-3            # Boltzmann constant, kcal/(mol*K)
HC_KCAL = 2.85914308e-3             # h*c, kcal/mol per cm^-1  (the energy of 1 cm^-1)
T_REF = 298.15                      # K
KT = KB_KCAL * T_REF                # kcal/mol
H_SI = 6.62607015e-34
KB_SI = 1.380649e-23
NA = 6.02214076e23
AMU_KG = 1.66053906660e-27
P_STD = 1.0e5                       # Pa, the 1 bar standard state
J_TO_KCAL = 1.0 / 4184.0
# With I in amu*A^2, the rotational constant B[cm^-1] = ROT_CONST_AMU_A2 / I
ROT_CONST_AMU_A2 = 16.857629046     # = h / (8 pi^2 c), converted to amu*A^2 and cm^-1

# ---------------------------------------------------------------- system definition
EDGE = "C2H5O1N1_19_36"
SPECIES = {"acetamide": "CC(N)=O", "N-methylformamide": "CNC=O"}

# External symmetry number: declared explicitly, never derived automatically
SYMMETRY_NUMBER = {"acetamide": 1, "N-methylformamide": 1}

# Electronic degeneracy g0 = the spin multiplicity. Same principle as the symmetry number: declared explicitly; a missing one refuses to run.
# Both are closed-shell singlet ground states, so g0 = 1 and -kT ln g0 = 0.
ELECTRONIC_DEGENERACY = {"acetamide": 1, "N-methylformamide": 1}

# Internal symmetry number of a methyl rotor (the three-fold axis of the top itself)
SIGMA_INTERNAL_METHYL = 3


# ================================================================================
# Part A  the separable terms -- turning "cancels exactly" from prose into assertions that can fail
# ================================================================================

def sackur_tetrode_kcal(mass_amu, temperature_K=T_REF, pressure_Pa=P_STD,
                        kind="gibbs"):
    """Ideal-gas translational free energy (per mole, kcal/mol), standard-state volume v = kT/p.

        Lambda   = h / sqrt(2 pi m kB T)                     thermal de Broglie wavelength
        a_trans  = -kT [ ln(kT / (p Lambda^3)) + 1 ]         Helmholtz, including the Stirling term of 1/N!
        g_trans  = a_trans + kT = -kT ln(kT / (p Lambda^3))   Gibbs

    **The two differ by exactly one kT, which is the pV term itself**, that is Delta n * RT.
    Delta n = 0 in this project, so the two conventions are equivalent in a difference; but **the
    absolute value of a single species differs by 0.5925 kcal/mol**, so which one is used must be
    stated -- this function once implemented only the Gibbs convention while being called "Helmholtz", a naming error, since corrected.

    It depends only on mass and temperature/pressure -- not at all on geometry, bonding or the potential.
    """
    if kind not in ("gibbs", "helmholtz"):
        raise ValueError("kind must be 'gibbs' or 'helmholtz', got {!r}".format(kind))
    m = mass_amu * AMU_KG
    lam = H_SI / math.sqrt(2.0 * math.pi * m * KB_SI * temperature_K)
    v = KB_SI * temperature_K / pressure_Pa          # volume per molecule, m^3
    ln_q_over_n = math.log(v / lam ** 3)
    stirling = 0.0 if kind == "gibbs" else 1.0
    a_j = -KB_SI * temperature_K * (ln_q_over_n + stirling)
    return a_j * NA * J_TO_KCAL


def molecular_mass_amu(smiles):
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    return Descriptors.MolWt(Chem.AddHs(Chem.MolFromSmiles(smiles)))


def part_a():
    from rdkit import Chem
    out = {}

    # --- a missing declaration refuses to run; there is no default ------------------------
    for name in SPECIES:
        if name not in SYMMETRY_NUMBER:
            raise KeyError("external symmetry number not declared: {} -- refusing to run".format(name))
        if name not in ELECTRONIC_DEGENERACY:
            raise KeyError("electronic degeneracy not declared: {} -- refusing to run".format(name))

    masses = {n: molecular_mass_amu(s) for n, s in SPECIES.items()}
    formulas = {n: Chem.rdMolDescriptors.CalcMolFormula(
        Chem.AddHs(Chem.MolFromSmiles(s))) for n, s in SPECIES.items()}
    g_trans = {n: sackur_tetrode_kcal(m, kind="gibbs") for n, m in masses.items()}
    a_trans = {n: sackur_tetrode_kcal(m, kind="helmholtz") for n, m in masses.items()}
    a_elec = {n: -KT * math.log(ELECTRONIC_DEGENERACY[n]) for n in SPECIES}

    names = list(SPECIES)
    d_trans = a_trans[names[1]] - a_trans[names[0]]
    d_elec = a_elec[names[1]] - a_elec[names[0]]

    print("=" * 88)
    print("Part A  which of the four terms cancel exactly -- asserted, not claimed")
    print("=" * 88)
    print("edge: {}".format(EDGE))
    for n in names:
        print("  {:20s} formula {:10s} M = {:10.6f} amu   g0 = {:d}   sigma_ext = {:d}".format(
            n, formulas[n], masses[n], ELECTRONIC_DEGENERACY[n], SYMMETRY_NUMBER[n]))
    print()
    print("  1) translation (Sackur-Tetrode, 298.15 K, 1 bar)")
    print("       {:22s} {:>16s} {:>16s} {:>10s}".format(
        "species", "a_trans(Helmholtz)", "g_trans(Gibbs)", "g - a"))
    for n in names:
        print("       {:22s} {:16.8f} {:16.8f} {:10.4f}".format(
            n, a_trans[n], g_trans[n], g_trans[n] - a_trans[n]))
    print("       g - a = kT = {:.4f} kcal/mol -- that is the pV term, Delta n * RT".format(KT))
    print("       Delta a_trans          = {:14.8e} kcal/mol".format(d_trans))
    assert g_trans[names[1]] - g_trans[names[0]] == 0.0, "Delta g_trans is non-zero"

    # Assertion 1: same formula => identical mass digit for digit => the translational term cancels digit for digit in IEEE-754
    assert formulas[names[0]] == formulas[names[1]], \
        "different formulas: the translational term no longer cancels, voiding the cancellation proof in this script"
    assert masses[names[0]] == masses[names[1]], "masses are not identical digit for digit"
    assert d_trans == 0.0, "Delta A_trans is non-zero: {!r}".format(d_trans)
    print("       assertion passed: same formula -> identical mass -> Delta A_trans is identically 0")
    print("       so no separate script for F_trans is needed -- in a Delta it is not small, it is identically 0.")
    print()
    print("  2) electronic (q_el = g0 * exp(-beta E_el); E_el is listed separately, leaving only -kT ln g0 here)")
    for n in names:
        print("       A_elec({:20s}) = {:14.8f} kcal/mol   (g0 = {:d})".format(
            n, a_elec[n], ELECTRONIC_DEGENERACY[n]))
    assert d_elec == 0.0, "Delta A_elec is non-zero: {!r}".format(d_elec)
    print("       assertion passed: both are closed-shell singlets -> g0 = 1 -> the term is identically 0")
    print()
    print("  3) rotation: does not cancel; already computed in the lecture notes section 6.6 as a rigid rotor with an explicit sigma")
    print("  4) vibration/conformers: does not cancel, and is exactly what stage 0 computes (section 6.5, density-of-states weighted)")
    print()

    # --- an example this assertion ought to fail on ---------------------------------------
    #
    # The first version of this check used an invented threshold |Delta| > 0.1 kcal/mol and fired at once.
    # Traced down: what fired was not a real error but **the aperture of the criterion itself** --
    # the translational term depends on mass only as -(3/2) kT ln(m'/m), and 1 amu is worth just 0.0149 kcal/mol.
    # Changed to compare against the analytic form: the counter-example must (i) be non-zero, since the
    # assertion above is digit-for-digit equality, and (ii) equal -(3/2) kT ln(m'/m) exactly. That is a derived criterion, not an invented threshold.
    fake = sackur_tetrode_kcal(masses[names[0]] + 1.0, kind="helmholtz")
    delta_fake = fake - a_trans[names[0]]
    analytic = -1.5 * KT * math.log((masses[names[0]] + 1.0) / masses[names[0]])
    print("  counter-example check (proving the assertion above is not vacuous):")
    print("     if the two sides differ by 1 amu, Delta A_trans = {:+.8f} kcal/mol".format(delta_fake))
    print("     analytic -(3/2) kT ln(m'/m)      = {:+.8f} kcal/mol".format(analytic))
    print("     their difference                 = {:+.3e} kcal/mol".format(
        delta_fake - analytic))
    assert delta_fake != 0.0, "the counter-example failed to break digit-for-digit equality -> the assertion is vacuous"
    assert abs(delta_fake - analytic) < 1e-9, "the Sackur-Tetrode implementation disagrees with the analytic form"
    print("     passed: the assertion has resolving power (non-zero), and the implementation matches the analytic form.")
    print("     note the magnitude: the translational term is very insensitive to mass, 1 amu is only {:.4f} kcal/mol,".format(
        abs(delta_fake)))
    print("               far below the 1.0 kcal/mol target -- but the assertion still demands digit-for-digit equality, because")
    print("               under the same formula it should be exactly 0, and any non-zero means the formula is wrong.")
    print()

    out["formulas"] = formulas
    out["mass_amu"] = masses
    out["a_trans_helmholtz_kcal_per_mol"] = a_trans
    out["g_trans_gibbs_kcal_per_mol"] = g_trans
    out["A_elec_kcal_per_mol"] = a_elec
    out["delta_A_trans_kcal_per_mol"] = d_trans
    out["delta_A_elec_kcal_per_mol"] = d_elec
    out["counterexample_delta_A_trans_mass_plus_1amu"] = delta_fake
    return out


# ================================================================================
# Part B  the reduced moment of inertia of a methyl rotor -- the Pitzer-Gwinn I^(2,3) scheme
# ================================================================================

def embed(smiles, seed=0xC0FFEE):
    """The same recipe as the lecture notes: ETKDGv3 + MMFF94. Used only to obtain a geometry; it produces no energy."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    if AllChem.EmbedMolecule(mol, p) != 0:
        raise RuntimeError("ETKDGv3 embedding failed: {}".format(smiles))
    AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    conf = mol.GetConformer()
    xyz = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    m = np.array([a.GetMass() for a in mol.GetAtoms()])
    return mol, xyz, m


def find_methyl_tops(mol):
    """Returns [(top_carbon_idx, anchor_idx, [top atom idx...]), ...]"""
    from rdkit import Chem
    patt = Chem.MolFromSmarts("[CX4H3]-[!#1]")
    tops = []
    for c_idx, anchor in mol.GetSubstructMatches(patt):
        atoms = [c_idx] + [nb.GetIdx() for nb in mol.GetAtomWithIdx(c_idx).GetNeighbors()
                           if nb.GetIdx() != anchor]
        tops.append((c_idx, anchor, atoms))
    return tops


def reduced_moment_I23(xyz, masses, top_atoms, axis_a, axis_b):
    """The Pitzer-Gwinn I^(2,3) reduced moment of inertia, amu*A^2.

        I_red = I_top * ( 1 - sum_g  lambda_g^2 * I_top / I_g )

    I_top    the moment of inertia of the top about its own axis
    lambda_g direction cosines of that axis against the three principal axes of the whole molecule
    I_g      the three principal moments of inertia of the whole molecule

    Source: K. S. Pitzer, W. D. Gwinn, J. Chem. Phys. 10 (1942) 428, doi:10.1063/1.1723744;
          the scheme numbering and naming are from A. L. L. East, L. Radom, J. Chem. Phys. 106 (1997) 6655,
          doi:10.1063/1.473958 (which writes the Pitzer schemes uniformly as I^(m,n)).
    """
    axis = xyz[axis_b] - xyz[axis_a]
    axis = axis / np.linalg.norm(axis)

    # I_top: mass times the squared perpendicular distance from each top atom to the axis
    d = xyz[top_atoms] - xyz[axis_a]
    perp = d - np.outer(d @ axis, axis)
    i_top = float((masses[top_atoms] * (perp ** 2).sum(1)).sum())

    # principal moments and axes of the whole molecule
    com = (masses[:, None] * xyz).sum(0) / masses.sum()
    r = xyz - com
    r2 = (r ** 2).sum(1)
    inertia = (masses[:, None, None] * (r2[:, None, None] * np.eye(3)
                                        - r[:, :, None] * r[:, None, :])).sum(0)
    i_princ, axes = np.linalg.eigh(inertia)
    lam = axes.T @ axis                                 # direction cosines
    i_red = i_top * (1.0 - float((lam ** 2 * i_top / i_princ).sum()))
    return i_red, i_top, i_princ, lam


def part_b():
    print("=" * 88)
    print("Part B  reduced moment of inertia of a methyl rotor (Pitzer-Gwinn I^(2,3))")
    print("=" * 88)
    out = {}
    for name, smi in SPECIES.items():
        mol, xyz, m = embed(smi)
        tops = find_methyl_tops(mol)
        rec = []
        print("  {:20s} {} methyl top(s) found".format(name, len(tops)))
        for c_idx, anchor, atoms in tops:
            i_red, i_top, i_princ, lam = reduced_moment_I23(xyz, m, atoms, c_idx, anchor)
            b_cm = ROT_CONST_AMU_A2 / i_red
            sym = mol.GetAtomWithIdx(anchor).GetSymbol()
            print("     C{:d}-{}{:d}   I_top = {:6.3f}   I_red = {:6.3f} amu A^2"
                  "   -> B = {:6.3f} cm^-1".format(c_idx, sym, anchor, i_top, i_red, b_cm))
            rec.append(dict(top_carbon=c_idx, anchor=anchor, anchor_element=sym,
                            I_top_amu_A2=i_top, I_red_amu_A2=i_red,
                            B_cm_inv=b_cm,
                            I_principal_amu_A2=[float(v) for v in i_princ]))
        out[name] = rec
    print()
    return out


# ================================================================================
# Part C  four treatments of a low-frequency mode
# ================================================================================

def a_harmonic_kcal(nu_cm):
    """Quantum harmonic-oscillator Helmholtz free energy (including the zero-point energy), zero at the bottom of the well. kcal/mol."""
    x = HC_KCAL * np.asarray(nu_cm, float) / KT
    return KT * (0.5 * x + np.log1p(-np.exp(-x)))


def s_harmonic_kcal_per_k(nu_cm):
    x = HC_KCAL * np.asarray(nu_cm, float) / KT
    return KB_KCAL * (x / np.expm1(x) - np.log1p(-np.exp(-x)))


def q_free_rotor(b_cm, sigma_int, temperature_K=T_REF):
    """One-dimensional free-rotor partition function q = (1/sigma) sqrt(pi kT / (h c B))."""
    kt_cm = KB_KCAL * temperature_K / HC_KCAL
    return math.sqrt(math.pi * kt_cm / b_cm) / sigma_int


def a_free_rotor_kcal(b_cm, sigma_int, temperature_K=T_REF):
    return -KB_KCAL * temperature_K * math.log(q_free_rotor(b_cm, sigma_int, temperature_K))


def s_free_rotor_kcal_per_k(b_cm, sigma_int, temperature_K=T_REF):
    """One-dimensional free-rotor entropy: S = k [ ln q + 1/2 ]  (since E = kT/2)."""
    return KB_KCAL * (math.log(q_free_rotor(b_cm, sigma_int, temperature_K)) + 0.5)


def mathieu_levels(b_cm, v_n_cm, n_fold=3, m_max=200):
    """Eigenvalues of the one-dimensional hindered rotor H = B (-i d/dphi)^2 + (V_n/2)(1 - cos n phi), cm^-1.

    Diagonalised in the free-rotor basis {exp(i m phi)}. The zero of the potential is the bottom of the well (phi = 0), so
        <m|V|m>       = V_n / 2
        <m|V|m ± n>   = -V_n / 4
    This is the standard Mathieu equation; see Pitzer & Gwinn 1942 (doi:10.1063/1.1723744).
    """
    m = np.arange(-m_max, m_max + 1)
    h = np.diag(b_cm * m.astype(float) ** 2 + v_n_cm / 2.0)
    off = -v_n_cm / 4.0
    idx = np.arange(len(m) - n_fold)
    h[idx, idx + n_fold] = off
    h[idx + n_fold, idx] = off
    return np.linalg.eigvalsh(h)


def a_hindered_rotor_kcal(b_cm, v_n_cm, sigma_int=SIGMA_INTERNAL_METHYL, n_fold=3,
                          m_max=200, temperature_K=T_REF):
    """Free energy of the exact hindered rotor, zero at the bottom of the well (as for the harmonic oscillator, including the zero-point energy)."""
    e = mathieu_levels(b_cm, v_n_cm, n_fold, m_max)
    kt_cm = KB_KCAL * temperature_K / HC_KCAL
    q = float(np.exp(-e / kt_cm).sum()) / sigma_int
    return -KB_KCAL * temperature_K * math.log(q), q, e


def torsional_harmonic_frequency_cm(b_cm, v_n_cm, n_fold=3):
    """Torsional frequency of the harmonic approximation at the bottom of the well:  nu = n * sqrt(V_n * B).

    Expanding V = (V_n/2)(1 - cos n phi) about phi=0 gives k = n^2 V_n / 2 and omega = sqrt(k/I);
    substituting B = h/(8 pi^2 c I) and simplifying gives the result. Both sides are in cm^-1.
    """
    return n_fold * math.sqrt(v_n_cm * b_cm)


def a_grimme_qrrho_kcal(nu_cm, b_cm_of_mode, temperature_K=T_REF, nu0_cm=100.0):
    """Grimme quasi-RRHO: the damped interpolation is applied to the **entropy** only, with free energy = E_harm - T*S_qRRHO.

    w(nu) = 1 / (1 + (nu0/nu)^4)                       a Head-Gordon type damping
    S_qRRHO = w * S_HO(nu) + (1-w) * S_FR
    Source: S. Grimme, Chem. Eur. J. 18 (2012) 9955, doi:10.1002/chem.201200497.

    Implementation note: the original scheme uses an "average moment of inertia" mu' = mu*B_av/(mu+B_av)
    to stop the entropy diverging as mu -> inf; here the low-frequency mode really is a rotor whose B was
    computed in part B, so that B is used directly -- closer to this system than the original scheme, but **a modifiable convention**: changing it moves the entropy at the low-frequency end.
    """
    nu = float(nu_cm)
    w = 1.0 / (1.0 + (nu0_cm / nu) ** 4)
    s_ho = float(s_harmonic_kcal_per_k(nu))
    s_fr = s_free_rotor_kcal_per_k(b_cm_of_mode, SIGMA_INTERNAL_METHYL, temperature_K)
    s = w * s_ho + (1.0 - w) * s_fr
    x = HC_KCAL * nu / (KB_KCAL * temperature_K)
    e = KB_KCAL * temperature_K * (0.5 * x + x / math.expm1(x))   # the internal energy is not interpolated
    return e - temperature_K * s, w


def part_c(b_cm):
    print("=" * 88)
    print("Part C  free energy of one methyl rotor: four models, B = {:.4f} cm^-1".format(b_cm))
    print("=" * 88)
    print("  the zero is the bottom of the torsional well throughout; every model includes the zero-point energy; sigma_int = {}".format(
        SIGMA_INTERNAL_METHYL))
    print("  kT = {:.4f} kcal/mol = {:.1f} cm^-1".format(KT, KT / HC_KCAL))
    print()

    # ---- limiting check 1: as V3 -> 0 the hindered rotor must return to the free rotor ---
    a_fr = a_free_rotor_kcal(b_cm, SIGMA_INTERNAL_METHYL)
    a_hr0, q0, _ = a_hindered_rotor_kcal(b_cm, 0.0)
    print("  limiting check 1  V3 -> 0:")
    print("     A_free_rotor            = {:+10.6f} kcal/mol".format(a_fr))
    print("     A_hindered(V3 = 0)      = {:+10.6f} kcal/mol".format(a_hr0))
    print("     difference              = {:+10.3e} kcal/mol".format(a_hr0 - a_fr))
    assert abs(a_hr0 - a_fr) < 5e-3, "did not return to the free rotor at V3=0"
    print("     passed (the free-rotor analytic form is a continuum approximation; the residual comes from summing discrete levels)")
    print()

    # ---- limiting check 2: as V3 -> large the hindered rotor must return to the oscillator
    v_big = 8000.0                                     # cm^-1, about 22.9 kcal/mol
    nu_big = torsional_harmonic_frequency_cm(b_cm, v_big)
    a_hr_big, _, _ = a_hindered_rotor_kcal(b_cm, v_big)
    a_ho_big = float(a_harmonic_kcal(nu_big))
    print("  limiting check 2  V3 = {:.0f} cm^-1 ({:.1f} kcal/mol), nu_harm = {:.1f} cm^-1:".format(
        v_big, v_big * HC_KCAL, nu_big))
    print("     A_hindered              = {:+10.6f} kcal/mol".format(a_hr_big))
    print("     A_harmonic(nu_harm)     = {:+10.6f} kcal/mol".format(a_ho_big))
    print("     difference              = {:+10.6f} kcal/mol".format(a_hr_big - a_ho_big))
    assert abs(a_hr_big - a_ho_big) < 0.05, "did not return to the oscillator in the high-barrier limit"
    print("     passed (the residual is the anharmonicity of a cosine well against a parabolic one, and is expected)")
    print()

    # ---- scan the barrier ---------------------------------------------------------------
    v3_kcal = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
    rows = []
    print("  {:>8s} {:>9s} {:>11s} {:>11s} {:>11s} {:>11s} {:>10s}".format(
        "V3", "nu_harm", "A_harmonic", "A_free", "A_qRRHO", "A_hindered", "HO-HR"))
    print("  {:>8s} {:>9s} {:>11s} {:>11s} {:>11s} {:>11s} {:>10s}".format(
        "kcal/mol", "cm^-1", "kcal/mol", "kcal/mol", "kcal/mol", "kcal/mol", "kcal/mol"))
    print("  " + "-" * 76)
    for v in v3_kcal:
        v_cm = v / HC_KCAL
        nu = torsional_harmonic_frequency_cm(b_cm, v_cm) if v > 0 else 0.0
        a_hr, _, _ = a_hindered_rotor_kcal(b_cm, v_cm)
        if v > 0:
            a_ho = float(a_harmonic_kcal(nu))
            a_qr, _ = a_grimme_qrrho_kcal(nu, b_cm)
            s_ho = "{:+11.4f}".format(a_ho)
            s_qr = "{:+11.4f}".format(a_qr)
            gap = "{:+10.4f}".format(a_ho - a_hr)
        else:
            a_ho = a_qr = None
            s_ho = "{:>11s}".format("-inf")
            s_qr = "{:>11s}".format("-inf")
            gap = "{:>10s}".format("-inf")
        print("  {:8.2f} {:9.1f} {} {:+11.4f} {} {:+11.4f} {}".format(
            v, nu, s_ho, a_fr, s_qr, a_hr, gap))
        rows.append(dict(V3_kcal_per_mol=float(v), nu_harmonic_cm_inv=float(nu),
                         A_harmonic=a_ho, A_free_rotor=float(a_fr),
                         A_qRRHO=a_qr, A_hindered_rotor=float(a_hr)))
    # ---- the zero crossing of the error sign: measured, not eyeballed --------------------
    def gap(v_cm):
        return float(a_harmonic_kcal(torsional_harmonic_frequency_cm(b_cm, v_cm))) \
            - a_hindered_rotor_kcal(b_cm, v_cm)[0]

    lo, hi = 1.0, 4000.0                              # cm^-1, gap(lo) < 0 < gap(hi)
    assert gap(lo) < 0.0 < gap(hi), "the zero crossing is not bracketed, so the bisection below is invalid"
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if gap(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    v_zero = 0.5 * (lo + hi)
    nu_zero = torsional_harmonic_frequency_cm(b_cm, v_zero)

    print()
    print("  reading the table -- the harmonic error has two quite distinct regimes, with the boundary measured as:")
    print("     error zero crossing  V3 = {:.4f} kcal/mol,  nu_harm = {:.1f} cm^-1".format(
        v_zero * HC_KCAL, nu_zero))
    print("     (i)  nu < {:.0f} cm^-1: harmonic **underestimates** A -- because S_HO ~ -k ln(beta h c nu)".format(
        nu_zero))
    print("          diverges to +inf as nu -> 0, so A_HO -> -inf. The entropy of a rotor is in fact finite.")
    print("     (ii) nu > {:.0f} cm^-1: harmonic **overestimates** A, rising then falling, peaking at about 0.10 kcal/mol.".format(
        nu_zero))
    print("     the two regimes mean different things: (i) is a divergence, (ii) is the anharmonicity of the cosine well.")
    print("     only (i) threatens the 1.0 kcal/mol target -- and a methyl rotor falls squarely in (i).")
    print()
    return rows, dict(sign_change_V3_kcal=float(v_zero * HC_KCAL),
                      sign_change_nu_cm_inv=float(nu_zero))


def part_c_figure(b_cm, path):
    v_cm = np.logspace(math.log10(3.0), math.log10(8000.0), 220)     # cm^-1
    a_hr = np.array([a_hindered_rotor_kcal(b_cm, v)[0] for v in v_cm])
    nu = np.array([torsional_harmonic_frequency_cm(b_cm, v) for v in v_cm])
    a_ho = a_harmonic_kcal(nu)
    a_qr = np.array([a_grimme_qrrho_kcal(n, b_cm)[0] for n in nu])
    a_fr = a_free_rotor_kcal(b_cm, SIGMA_INTERNAL_METHYL)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.4))
    ax.plot(nu, a_hr, lw=2.2, color="#1f3b73", label="1D hindered rotor (exact, Mathieu)")
    ax.plot(nu, a_ho, lw=1.6, ls="--", color="#b03030", label="harmonic oscillator")
    ax.plot(nu, a_qr, lw=1.6, ls="-.", color="#2e8b57", label="Grimme quasi-RRHO")
    ax.axhline(a_fr, lw=1.2, ls=":", color="#888888", label="1D free rotor (limit)")
    ax.set_xscale("log")
    ax.set_xlabel("harmonic torsional frequency  $\\tilde\\nu$  /  cm$^{-1}$")
    ax.set_ylabel("$A$ of one methyl torsion  /  kcal mol$^{-1}$")
    ax.set_title("(a) One methyl torsion, four models   (298.15 K, $B$ = "
                 "{:.2f} cm$^{{-1}}$)".format(b_cm), fontsize=10)
    ax.grid(alpha=.25)
    ax.legend(frameon=False, fontsize=8)

    ax2.plot(nu, a_ho - a_hr, lw=2.2, color="#b03030", label="harmonic $-$ exact")
    ax2.plot(nu, a_qr - a_hr, lw=2.0, color="#2e8b57", label="quasi-RRHO $-$ exact")
    ax2.axhline(0, lw=0.8, color="k")
    ax2.axhline(1.0, lw=1.0, ls=":", color="#444444")
    ax2.axhline(-1.0, lw=1.0, ls=":", color="#444444")
    ax2.set_xscale("log")
    ax2.set_xlabel("harmonic torsional frequency  $\\tilde\\nu$  /  cm$^{-1}$")
    ax2.set_ylabel("error in $A$  /  kcal mol$^{-1}$")
    ax2.set_title("(b) Error of each approximation; dotted lines = the 1 kcal/mol target",
                  fontsize=10)
    ax2.grid(alpha=.25)
    ax2.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"harmonic_minus_exact_max_kcal": float(np.abs(a_ho - a_hr).max()),
            "qrrho_minus_exact_max_kcal": float(np.abs(a_qr - a_hr).max())}


# ================================================================================
# Part D  the net effect of the low-frequency treatment on this edge
# ================================================================================

def part_d(tops):
    print("=" * 88)
    print("Part D  the difference between low-frequency treatments does not cancel on this edge")
    print("=" * 88)
    names = list(SPECIES)
    print("  {:20s} {:>8s} {:>14s}".format("species", "methyls", "B (cm^-1)"))
    for n in names:
        bs = ", ".join("{:.3f}".format(t["B_cm_inv"]) for t in tops[n])
        print("  {:20s} {:>8d} {:>14s}".format(n, len(tops[n]), bs if bs else "-"))
    print()
    print("  each side has one methyl rotor, but the anchoring atoms differ ({} vs {}), so the reduced moments differ,".format(
        tops[names[0]][0]["anchor_element"], tops[names[1]][0]["anchor_element"]))
    print("  and so do the barriers. The difference in A_rotor is therefore non-zero and the error of the model choice cancels only **partly**.")
    print()
    print("  the table below: if the true barriers of the two sides are (V_A, V_B), replacing the exact")
    print("        hindered rotor with the harmonic approximation leaves this residual in Delta A (kcal/mol).")
    print()
    grid = [0.05, 0.5, 1.0, 2.0, 3.0]
    b_a = tops[names[0]][0]["B_cm_inv"]
    b_b = tops[names[1]][0]["B_cm_inv"]
    print("       V_B \\ V_A " + "".join("{:>9.2f}".format(v) for v in grid))
    cells = {}
    for vb in grid:
        row = []
        for va in grid:
            ea = float(a_harmonic_kcal(torsional_harmonic_frequency_cm(b_a, va / HC_KCAL))) \
                - a_hindered_rotor_kcal(b_a, va / HC_KCAL)[0]
            eb = float(a_harmonic_kcal(torsional_harmonic_frequency_cm(b_b, vb / HC_KCAL))) \
                - a_hindered_rotor_kcal(b_b, vb / HC_KCAL)[0]
            row.append(eb - ea)
        cells["V_B={:.2f}".format(vb)] = {"V_A={:.2f}".format(va): float(r)
                                          for va, r in zip(grid, row)}
        print("       {:9.2f} ".format(vb) + "".join("{:+9.4f}".format(r) for r in row))
    print()
    worst = max(abs(v) for r in cells.values() for v in r.values())
    print("  on the diagonal (equal barriers) the residual nearly cancels; the further from it, the larger.")
    print("  worst grid point |residual| = {:.4f} kcal/mol.".format(worst))
    print("  conclusion: the same low-frequency model must be used on both sides, and its residual must enter the error bar.")
    print()
    return {"B_cm_inv": {names[0]: b_a, names[1]: b_b},
            "harmonic_minus_hindered_residual_grid_kcal": cells,
            "worst_grid_residual_kcal": worst}


# ================================================================================
PARAMETER_TABLE = [
    dict(name="T_REF", value=298.15, unit="K", classification="modifiable convention",
         note="the reference temperature; changing it changes the whole table, but both sides change together and the change in Delta is physical"),
    dict(name="P_STD", value=1.0e5, unit="Pa", classification="literature value",
         note="the IUPAC 1 bar standard state; it enters only the translational term, which is identically 0 on this edge"),
    dict(name="SIGMA_INTERNAL_METHYL", value=3, unit="1", classification="derived criterion",
         note="the three-fold axis of the methyl top itself; counted separately from the external symmetry number"),
    dict(name="ELECTRONIC_DEGENERACY", value="declared explicitly", unit="1",
         classification="modifiable convention",
         note="the spin multiplicity. A missing one refuses to run and there is no default -- the same principle as the external symmetry number"),
    dict(name="m_max", value=200, unit="1", classification="numerical tolerance",
         note="the free-rotor basis cut-off of the Mathieu matrix; B*200^2 is far above kT and the partition function has converged"),
    dict(name="nu0_cm", value=100.0, unit="cm^-1", classification="literature value",
         note="the turnover frequency of the Grimme damping function; Chem. Eur. J. 18 (2012) 9955"),
    dict(name="v_big", value=8000.0, unit="cm^-1", classification="derived criterion",
         note="the high-barrier limiting check point; V3 must be far above kT (207 cm^-1) before returning to the oscillator is meaningful"),
    dict(name="seed", value="0xC0FFEE", unit="1", classification="resource budget",
         note="the ETKDGv3 random seed; it moves only the third decimal of the reduced moment of inertia and must never be chosen to make a result look better"),
]


def main():
    root = _repo_root()
    outdir = root / "analysis"
    outdir.mkdir(exist_ok=True)

    a = part_a()
    tops = part_b()
    b_cm = tops["acetamide"][0]["B_cm_inv"]
    rows, signchange = part_c(b_cm)
    figpath = outdir / "lowfreq_hindered_rotor.png"
    figstat = part_c_figure(b_cm, figpath)
    print("figure written: {}".format(figpath))
    print("  maximum of |A_harmonic - A_exact| over the scanned range = {:.4f} kcal/mol".format(
        figstat["harmonic_minus_exact_max_kcal"]))
    print("  maximum of |A_qRRHO   - A_exact| over the scanned range = {:.4f} kcal/mol".format(
        figstat["qrrho_minus_exact_max_kcal"]))
    print()
    d = part_d(tops)

    payload = dict(edge=EDGE, temperature_K=T_REF,
                   species=SPECIES, symmetry_number=SYMMETRY_NUMBER,
                   electronic_degeneracy=ELECTRONIC_DEGENERACY,
                   part_a_separable_terms=a,
                   part_b_reduced_moments=tops,
                   part_c_rotor_models=rows,
                   part_c_sign_change=signchange,
                   part_c_figure_summary=figstat,
                   part_d_residual_on_this_edge=d,
                   parameter_table=PARAMETER_TABLE,
                   geometry_source="RDKit ETKDGv3 + MMFF94, seed 0xC0FFEE —— "
                                   "used only for the reduced moment of inertia; it produces no energy",
                   references=[
                       "K. S. Pitzer, W. D. Gwinn, J. Chem. Phys. 10 (1942) 428, "
                       "doi:10.1063/1.1723744",
                       "A. L. L. East, L. Radom, J. Chem. Phys. 106 (1997) 6655, "
                       "doi:10.1063/1.473958",
                       "P. Y. Ayala, H. B. Schlegel, J. Chem. Phys. 108 (1998) 2314, "
                       "doi:10.1063/1.475616",
                       "S. Grimme, Chem. Eur. J. 18 (2012) 9955, "
                       "doi:10.1002/chem.201200497",
                       "S.-T. Lin, M. Blanco, W. A. Goddard III, J. Chem. Phys. 119 (2003) "
                       "11792, doi:10.1063/1.1624057",
                       "S.-T. Lin, P. K. Maiti, W. A. Goddard III, J. Phys. Chem. B 114 "
                       "(2010) 8191, doi:10.1021/jp103120q",
                   ])
    jpath = outdir / "lowfreq_and_separable_terms.json"
    jpath.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("written: {}".format(jpath))


if __name__ == "__main__":
    main()
