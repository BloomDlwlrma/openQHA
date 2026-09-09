"""A **free, independent control** built from the native QM9 data: rigid-rotor harmonic-oscillator thermodynamic terms.

CALIBRATION. The docstring below settles it: a comparison sample at a different
level of theory, explicitly not a stage 0 result.

For every molecule QM9 gives, at B3LYP/6-31G(2df,p):
  * the three rotational constants A, B, C (GHz)  -> the rotational term, without even reading a geometry
  * all 3N-6 harmonic frequencies (cm^-1)         -> the vibrational term (rigid rotor harmonic oscillator)
  * the zero-point energy zpve, and U0 / U / H / G (Hartree)

So a **harmonic reference value** for `(G - E_el)` is available at no cost, to compare
against stage 0's density-of-states route. The difference between them = anharmonicity +
conformers + the difference of potential surfaces, which is exactly what stage 0 measures.

**This is not a stage 0 result, it is a control.** Its level is B3LYP/6-31G(2df,p), a
different surface from the lecture notes' MACE-OFF23-SC, so the difference also contains
the difference of surfaces -- which must be stated.

Usage:
    python scripts/calibration/s0_qm9_native_reference.py
Products:
    analysis/qm9_native_reference.json
"""
import csv
import json
import math
from pathlib import Path

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
KB_KCAL = 1.987204259e-3            # kcal/(mol*K)
HC_KCAL = 2.85914308e-3             # kcal/mol per cm^-1
T_REF = 298.15
KT = KB_KCAL * T_REF
H_SI = 6.62607015e-34
KB_SI = 1.380649e-23
HARTREE_KCAL = 627.5094740631

ROOT = _repo_root()
INDEX = (ROOT.parent / "stage1-qm9-alchemical-reaction-energy" / "source-data"
         / "index_Chem_composition.csv")

EDGE = "C2H5O1N1_19_36"
# species -> QM9 index. The 19 / 36 in an edge name are the QM9 indices (checked against index_Chem_composition.csv)
SPECIES_QM9 = {"acetamide": "dsgdb9nsd_000019",
               "N-methylformamide": "dsgdb9nsd_000036"}
SYMMETRY_NUMBER = {"acetamide": 1, "N-methylformamide": 1}   # declared explicitly, never derived automatically
REACTANT, PRODUCT = list(SPECIES_QM9)


def a_harmonic_kcal(nu_cm, temperature_K=T_REF):
    """Quantum harmonic-oscillator Helmholtz free energy (including the zero-point energy), with the zero at the bottom of the well."""
    kt = KB_KCAL * temperature_K
    x = HC_KCAL * nu_cm / kt
    return kt * (0.5 * x + math.log1p(-math.exp(-x)))


def a_rot_from_constants_kcal(a_ghz, b_ghz, c_ghz, sigma, temperature_K=T_REF):
    """Rigid-rotor rotational free energy, straight from the three rotational constants (GHz).

        Theta_i = h * nu_i / k_B      (nu_i is the rotational constant in Hz)
        q_rot   = (sqrt(pi)/sigma) * sqrt(T^3 / (Theta_A Theta_B Theta_C))
    """
    theta = [H_SI * g * 1e9 / KB_SI for g in (a_ghz, b_ghz, c_ghz)]     # K
    q = (math.sqrt(math.pi) / sigma) * math.sqrt(
        temperature_K ** 3 / (theta[0] * theta[1] * theta[2]))
    return -KB_KCAL * temperature_K * math.log(q), theta, q


def load_qm9():
    rows = {}
    want = set(SPECIES_QM9.values())
    with INDEX.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["qm9_index"] in want:
                rows[r["qm9_index"]] = r
    missing = want - set(rows)
    if missing:
        raise KeyError("missing from the QM9 index table: {} -- refusing to substitute an estimate".format(sorted(missing)))
    return rows


def main():
    qm9 = load_qm9()
    out = dict(edge=EDGE, temperature_K=T_REF,
               level="B3LYP/6-31G(2df,p)  (QM9 native)",
               source=str(INDEX), species={})

    print("=" * 92)
    print("QM9 native B3LYP/6-31G(2df,p) rigid-rotor harmonic-oscillator reference   edge {}".format(EDGE))
    print("=" * 92)
    for name, qid in SPECIES_QM9.items():
        r = qm9[qid]
        freqs = [float(v) for v in r["frequencies"].split()]
        n_at = sum(int(r[k + "_num"]) for k in ("C", "H", "O", "N", "F"))
        expect = 3 * n_at - 6
        if len(freqs) != expect:
            raise ValueError("{}: {} frequencies, expected 3N-6 = {}".format(name, len(freqs), expect))
        n_imag = sum(1 for f in freqs if f <= 0.0)
        a_vib = sum(a_harmonic_kcal(f) for f in freqs)
        a_rot, theta, q_rot = a_rot_from_constants_kcal(
            float(r["A"]), float(r["B"]), float(r["C"]), SYMMETRY_NUMBER[name])
        rec = dict(qm9_index=qid, smiles=r["qm9_smiles"], n_atoms=n_at,
                   n_frequencies=len(freqs), n_imaginary=n_imag,
                   lowest_frequency_cm_inv=min(freqs),
                   rotational_constants_GHz=[float(r[k]) for k in ("A", "B", "C")],
                   rotational_temperatures_K=theta,
                   symmetry_number=SYMMETRY_NUMBER[name],
                   A_vib_kcal=a_vib, A_rot_kcal=a_rot,
                   G_minus_E_el_kcal=a_vib + a_rot,
                   qm9_zpve_hartree=float(r["zpve"]),
                   qm9_U0_hartree=float(r["U0"]), qm9_G_hartree=float(r["G"]))
        out["species"][name] = rec
        print()
        print("  {:20s} {}   {}   {} atom(s)".format(name, qid, r["qm9_smiles"], n_at))
        print("     {} frequencies (3N-6 = {}), {} imaginary, lowest {:.2f} cm^-1".format(
            len(freqs), expect, n_imag, min(freqs)))
        print("     rotational constants (GHz)  {:8.5f} {:8.5f} {:8.5f}".format(*rec["rotational_constants_GHz"]))
        print("     rotational temperatures (K) {:8.5f} {:8.5f} {:8.5f}   T/Theta_max = {:.0f}".format(
            *theta, T_REF / max(theta)))
        print("     A_vib = {:+9.4f}   A_rot = {:+9.4f}   (G - E_el) = {:+9.4f} kcal/mol".format(
            a_vib, a_rot, a_vib + a_rot))
        print("     from QM9 itself: zpve = {:.6f} Eh = {:.4f} kcal/mol".format(
            float(r["zpve"]), float(r["zpve"]) * HARTREE_KCAL))

    s = out["species"]
    d_vib = s[PRODUCT]["A_vib_kcal"] - s[REACTANT]["A_vib_kcal"]
    d_rot = s[PRODUCT]["A_rot_kcal"] - s[REACTANT]["A_rot_kcal"]
    d_therm = s[PRODUCT]["G_minus_E_el_kcal"] - s[REACTANT]["G_minus_E_el_kcal"]
    d_G_qm9 = (s[PRODUCT]["qm9_G_hartree"] - s[REACTANT]["qm9_G_hartree"]) * HARTREE_KCAL
    d_U0_qm9 = (s[PRODUCT]["qm9_U0_hartree"] - s[REACTANT]["qm9_U0_hartree"]) * HARTREE_KCAL

    print()
    print("=" * 92)
    print("difference   {} -> {}".format(REACTANT, PRODUCT))
    print("=" * 92)
    print("   Delta A_vib   (harmonic)         = {:+9.4f} kcal/mol".format(d_vib))
    print("   Delta A_rot   (rigid rotor)      = {:+9.4f} kcal/mol".format(d_rot))
    print("   Delta (G - E_el)  total          = {:+9.4f} kcal/mol   <-- the product of this script".format(
        d_therm))
    print()
    print("   against the complete quantities QM9 ships (also B3LYP/6-31G(2df,p)):")
    print("      Delta G  (the QM9 G column)   = {:+9.4f} kcal/mol".format(d_G_qm9))
    print("      Delta U0 (the QM9 U0 column)  = {:+9.4f} kcal/mol".format(d_U0_qm9))
    print("      their difference = Delta(G - U0) = {:+9.4f} kcal/mol".format(d_G_qm9 - d_U0_qm9))
    print()
    print("   Note: the QM9 G column includes the electronic energy and this script (G - E_el) does not -- they are not directly comparable.")
    print("         What is comparable: Delta G(QM9) - Delta(G-E_el)(this script) should be about Delta E_el(B3LYP),")
    print("         that is {:+.4f} kcal/mol.".format(d_G_qm9 - d_therm))
    out["delta"] = dict(A_vib=d_vib, A_rot=d_rot, G_minus_E_el=d_therm,
                        qm9_G=d_G_qm9, qm9_U0=d_U0_qm9,
                        implied_delta_E_el_b3lyp=d_G_qm9 - d_therm)

    outdir = ROOT / "analysis"
    outdir.mkdir(exist_ok=True)
    p = outdir / "qm9_native_reference.json"
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    print("written:", p)
    print()
    print("**This is a control, not a stage 0 result.** Its level is B3LYP/6-31G(2df,p), a")
    print("different surface from the lecture notes MACE-OFF23-SC; and it is purely harmonic,")
    print("carrying no anharmonic or conformational contribution -- which is precisely what")
    print("the stage 0 density-of-states route captures. The difference between the two =")
    print("anharmonicity + conformers + the difference of surfaces.")


if __name__ == "__main__":
    main()
