"""The `thermo_msrrho` Calculation: a molecule's absolute entropy and free energy from
its basins, assembled the way Pracht & Grimme (Chem. Sci. 2021, 12, 6551) define it,
entirely at one level.

    per basin i (branchA.toml [[Basin]], mace/basinNN/hessian.npy, degeneracy.toml):
      omega_i       Eckart-projected, mass-weighted spectrum of the stored Hessian
      S_vib,i       msRRHO under a preset (thermo.msrrho; `crest` by default)
      S_i           S_vib + S_rot(sigma_i) + S_trans + S_elec(g0)      absolute, cal/mol/K
      H_i           E_el + ZPE + [H(T)-H(0)]_vib + H_rot + H_trans     kcal/mol
      G_i           H_i - T S_i
    ensemble (populations from G_i, degeneracies g'_i):
      p_i           g'_i exp(-G_i/kT) / sum_j g'_j exp(-G_j/kT)
      S'_conf       R [ ln sum_i g'_i e^(-beta dG_i) + beta <dG>_p ]      dG_i = G_i - G_ref
      dS_bar        sum_i p_i S_i - S_ref                                   ref = lowest G_i
      H_conf        sum_i p_i (H_i - H_ref)
      S_abs         S_ref + S'_conf + dS_bar
      G_total       -kT ln sum_i g'_i exp(-G_i/kT)  ==  (H_ref + H_conf) - T S_abs

The last line is an identity, not an approximation: at one level the reference
subtraction cancels exactly (the paper says so of the same-level case), and S'_conf is
the mixing entropy of the populations that G_total is built from. CREST's `--entropy`
evaluates the paper's eq. 10 with the conformers' ELECTRONIC energies instead; that
number is written too, as S_CONF_PRIME_E, for the GFN2 seam.

Conventions are declared, never defaulted silently: preset (tau, rotor cap, what is
interpolated), the imaginary-mode policy (`invert_below`, the production default: a mode
in [ithr, 0) with ithr = -50 cm^-1 is inverted, a mode with |omega| < 1 cm^-1 is dropped
ORCA-style and counted, a mode below the floor excludes the basin, which is listed with
its reason; `crest_native` exists for the GFN2 seam only), fscal = 1.0 for MACE. No
extrapolation to ensemble completeness (the paper's eq. 15): the basins come from one
branch A search. No rotamer factor exists.
"""
import math
from pathlib import Path

import numpy as np

from .. import config
from ..conformer_search import degeneracy
from ..store import basins as basins_mod
from ..store import branch_a_property, layout, property as prop, report
from . import hessian as hessian_mod
from . import thermo

STEP = "thermo_msrrho"
PROGNAME = "openQHA thermo_msrrho"
EV_TO_KCAL = 23.060547830618307
R_CAL = thermo.KB_KCAL * 1000.0          # cal/(mol K)
#: The paper's population mass that gets explicit treatment (CREST `--ptot 0.9`).
PTOT = 0.90

SCHEMA = {
    "Calculation_Info": {
        "MOLECULE_DIR": ("String", None, "the molecule directory"),
        "QM9_INDEX": ("String", None, "the molecule"),
        "TAG": ("String", None, "the branch A tag the basins came from"),
        "LEVEL": ("String", None, "the level of the Hessians"),
        "ENGINE": ("String", None, "the potential that produced the Hessians"),
        "PRESET": ("String", None, "msRRHO preset: crest, xtb or grimme2012"),
        "TAU": ("Double", "cm^-1", "rotor interpolation crossover of the preset"),
        "ROTOR_CAP_RULE": ("String", None, "mean_principal_moment, or a fixed B_av"),
        "ITHR_POLICY": ("String", None, "imaginary-mode policy the [Result] block used: invert_below (production, floor -50 cm^-1) or crest_native (GFN2 seam only)"),
        "FSCAL": ("Double", None, "frequency scaling factor (1.0: declared, none published for MACE)"),
        "TEMPERATURE": ("Double", "K", "temperature"),
        "PRESSURE": ("Double", "Pa", "standard pressure of the translational term"),
        "REFERENCE_BASIN": ("Integer", None, "the basin with the lowest G_i"),
        "PTOT": ("Double", None, "population mass defining N_BASINS_90"),
        "EXTRAPOLATION": ("String", None, "none: one branch A search, no iterative series"),
    },
    "Basin": {
        "INDEX": ("Integer", None, "basin index"),
        "SIGMA": ("Integer", None, "external symmetry number of the basin"),
        "G0": ("Integer", None, "electronic degeneracy"),
        "G_PRIME": ("Integer", None, "enantiomer degeneracy"),
        "G_PRIME_SOURCE": ("String", None, "mirror_pair, class_inherited, achiral, no_mirror_sampled, pooled_frame or mirror_is_basin"),
        "E_EL": ("Double", "kcal/mol", "electronic energy"),
        "ZPE": ("Double", "kcal/mol", "zero-point energy (harmonic)"),
        "H_THERMAL": ("Double", "kcal/mol", "vibrational H(T)-H(0) (harmonic)"),
        "G_ROT": ("Double", "kcal/mol", "rigid-rotor free energy with SIGMA"),
        "G_TRANS": ("Double", "kcal/mol", "Sackur-Tetrode Gibbs term at PRESSURE"),
        "S_VIB": ("Double", "cal/mol/K", "msRRHO vibrational entropy"),
        "S_VIB_HO": ("Double", "cal/mol/K", "harmonic vibrational entropy, for the record"),
        "S_ROT": ("Double", "cal/mol/K", "rigid-rotor entropy with SIGMA"),
        "S_TRANS": ("Double", "cal/mol/K", "Sackur-Tetrode entropy at PRESSURE"),
        "S_MSRRHO": ("Double", "cal/mol/K", "absolute entropy of the basin: vib + rot + trans + elec"),
        "H_I": ("Double", "kcal/mol", "enthalpy of the basin"),
        "G_I": ("Double", "kcal/mol", "free energy of the basin"),
        "POPULATION": ("Double", None, "g' exp(-G_i/kT) normalised over the included basins"),
        "LOWEST_FREQ": ("Double", "cm^-1", "lowest projected frequency"),
        "N_IMAGINARY": ("Integer", None, "imaginary modes of the projected spectrum"),
        "N_INVERTED": ("Integer", None, "modes in [ithr, 0) inverted under the policy"),
        "N_KEPT_NEGATIVE": ("Integer", None, "modes below ithr kept negative (crest_native only)"),
        "N_BELOW_FLOOR": ("Integer", None, "modes with |omega| < 1 cm^-1 dropped from every sum (ORCA CutOffFreq)"),
        "INVERTED_CM": ("ArrayOfDoubles", "cm^-1", "the inverted modes (ithr <= omega < 0); absent when none"),
        "DROPPED_CM": ("ArrayOfDoubles", "cm^-1", "the dropped modes (|omega| < 1 cm^-1); absent when none"),
        "SOFT_SADDLE": ("Boolean", None, "excluded as a soft saddle: the relaxed lowest mode lies in the inversion window [ithr, 0) at the reference level"),
        "EXCLUDED": ("Boolean", None, "left out of the ensemble (lowest mode below ithr, unprojected signature, or a policy the preset cannot apply)"),
        "HESSIAN_ROUTE": ("String", None, "analytic or numerical (reference levels)"),
        "NOISE_FLOOR_CM": ("Double", "cm^-1", "largest |eigenvalue| of the rigid-body block of the unprojected Hessian: 5-30 cm^-1 analytic at a tight minimum (residual gradient in the rotational block), plus the finite-difference noise of a numerical one"),
        "OPT_GRAD_RMS": ("Double", "Eh/bohr", "final RMS gradient of the optimisation (numerical gradient for a numerical level)"),
        "N_SINGLE_POINTS": ("Integer", None, "energy evaluations the ORCA job spent"),
    },
    "Ensemble": {
        "S_CONF_PRIME": ("Double", "cal/mol/K", "mixing entropy of the populations (exact at one level)"),
        "S_CONF_PRIME_E": ("Double", "cal/mol/K", "paper eq. 10 with electronic energies (CREST convention)"),
        "DS_BAR": ("Double", "cal/mol/K", "population average of S_msRRHO minus the reference"),
        "H_CONF": ("Double", "kcal/mol", "population average of H_i minus the reference"),
        "CP_CONF": ("Double", "cal/mol/K", "fluctuation term R beta^2 Var_p(H_i)"),
    },
    "Result": {
        "S_ABS": ("Double", "cal/mol/K", "absolute entropy: S_ref + S_CONF_PRIME + DS_BAR"),
        "H_ABS": ("Double", "kcal/mol", "H_ref + H_CONF"),
        "G_TOTAL": ("Double", "kcal/mol", "-kT ln sum g' exp(-G_i/kT)"),
        "N_BASINS": ("Integer", None, "basins in branchA.toml"),
        "N_INCLUDED": ("Integer", None, "basins in the ensemble"),
        "N_EXCLUDED": ("Integer", None, "basins excluded by the frequency floor (lowest mode below ithr, or the unprojected signature)"),
        "N_BASINS_90": ("ArrayOfIntegers", None, "basins carrying PTOT of the population, most populated first"),
        "PRESET_SPREAD": ("Double", "kcal/mol", "max - min of T*S_vib over the presets, at the reference basin"),
        "S_EXPERIMENT": ("Double", "cal/mol/K", "declared experimental S at 298.15 K"),
        "S_EXPERIMENT_SOURCE": ("String", None, "BibTeX key in docs/cite/cite_openQHA.bib"),
        "S_ABS_MINUS_EXPERIMENT": ("Double", "cal/mol/K", "S_ABS - S_EXPERIMENT"),
    },
}


# ====================================================================== per basin
def basin_thermochemistry(index, hessian_eV_A2, masses, positions, energy_eV, sigma, g0,
                          temperature_K=thermo.T_REF, preset="crest",
                          imaginary_policy=thermo.PRODUCTION_POLICY, fscal=1.0):
    """Project the stored Hessian and hand the spectrum to
    `basin_thermochemistry_from_frequencies`."""
    proj = hessian_mod.project_and_diagonalise(np.asarray(hessian_eV_A2, dtype=float),
                                               masses, positions)
    rec = dict(index=int(index), energy_eV=float(energy_eV),
               E_el_kcal=float(energy_eV) * EV_TO_KCAL, sigma=int(sigma), g0=int(g0),
               masses=[float(m) for m in masses],
               positions=[[float(x) for x in r] for r in positions],
               frequencies_cm=list(proj["frequencies_cm_inv"]),
               n_imaginary=int(proj["n_imaginary"]),
               n_rigid_removed=int(proj["n_rigid_modes_removed"]))
    return basin_thermochemistry_from_frequencies(rec, temperature_K, preset,
                                                  imaginary_policy, fscal)


def basin_thermochemistry_from_frequencies(rec, temperature_K=thermo.T_REF, preset="crest",
                                           imaginary_policy=thermo.PRODUCTION_POLICY, fscal=1.0):
    """S_i, H_i, G_i of one basin from its spectrum, masses, positions, E_el, sigma, g0.

    Under the production policy a mode in [ithr, 0) is inverted and a mode with
    |omega| < 1 cm^-1 dropped (both counted, their values in the record); a basin whose
    lowest mode is below the floor -- or whose spectrum carries the unprojected signature
    -- comes back with `excluded = True`, the reason, and no thermochemistry."""
    rec = dict(rec)
    rec["lowest_frequency_cm"] = float(min(rec["frequencies_cm"]))
    rec.setdefault("n_imaginary", int(sum(1 for f in rec["frequencies_cm"] if f < 0)))
    rec.update(imaginary_policy=imaginary_policy, n_inverted=0, n_kept_negative=0,
               n_below_floor=0, inverted_frequencies_cm=[], dropped_frequencies_cm=[])
    try:
        vib = thermo.msrrho(rec["frequencies_cm"], rec["masses"], rec["positions"],
                            preset=preset, temperature_K=temperature_K,
                            imaginary_policy=imaginary_policy, fscal=fscal)
    except ValueError as exc:
        rec.update(excluded=True, excluded_reason=str(exc))
        return rec
    T = float(temperature_K)
    kt = thermo.KB_KCAL * T
    rot = thermo.rotational(rec["masses"], rec["positions"], rec["sigma"], T)
    tr = thermo.translational(float(np.sum(rec["masses"])), T, kind="gibbs")
    el = thermo.electronic(rec["g0"], T)
    s_trans = (2.5 * kt - tr["value_kcal"]) / T           # Sackur-Tetrode, kcal/mol/K
    s_elec = thermo.KB_KCAL * math.log(rec["g0"])
    s_vib = vib["S_vib_kcal_per_K"]
    s_total = s_vib + rot["S_rot_kcal_per_K"] + s_trans + s_elec
    h_i = (rec["E_el_kcal"] + vib["ZPE_kcal"] + vib["H_thermal_kcal"]
           + rot["E_rot_kcal"] + 2.5 * kt)
    g_i = h_i - T * s_total
    rec.update(
        excluded=False, preset=preset, tau_cm=vib["tau_cm"],
        n_inverted=int(vib["n_inverted"]), n_kept_negative=int(vib["n_kept_negative"]),
        n_below_floor=int(vib["n_below_floor"]),
        inverted_frequencies_cm=list(vib["inverted_frequencies_cm"]),
        dropped_frequencies_cm=list(vib["dropped_frequencies_cm"]),
        rotor_cap_rule=vib["rotor_cap_rule"], fscal=float(fscal),
        ZPE_kcal=vib["ZPE_kcal"], H_thermal_kcal=vib["H_thermal_kcal"],
        G_rot_kcal=rot["A_rot_kcal"], G_trans_kcal=tr["value_kcal"],
        A_elec_kcal=el["A_elec_kcal"],
        S_vib_cal_per_K=s_vib * 1000.0, S_vib_HO_cal_per_K=vib["S_vib_HO_kcal_per_K"] * 1000.0,
        S_rot_cal_per_K=rot["S_rot_kcal_per_K"] * 1000.0, S_trans_cal_per_K=s_trans * 1000.0,
        S_elec_cal_per_K=s_elec * 1000.0, S_msrrho_cal_per_K=s_total * 1000.0,
        Cp_vib_cal_per_K=vib["Cp_vib_kcal_per_K"] * 1000.0,
        H_i_kcal=h_i, G_i_kcal=g_i, G_minus_Eel_kcal=g_i - rec["E_el_kcal"],
        modes=vib["modes"], temperature_K=T)
    return rec


# ====================================================================== the ensemble
def assemble(basins, temperature_K=thermo.T_REF, ptot=PTOT):
    """S'_conf, dS_bar, H_conf, Cp_conf, S_abs, H_abs, G_total from per-basin records
    (each with `G_i_kcal`, `H_i_kcal`, `S_msrrho_cal_per_K`, `g_prime`, `excluded`)."""
    T = float(temperature_K)
    kt = thermo.KB_KCAL * T
    inc = [b for b in basins if not b.get("excluded")]
    if not inc:
        raise ValueError("no basin survived the imaginary-mode policy; nothing to assemble")
    g = np.array([float(b.get("g_prime", 1)) for b in inc])
    G = np.array([b["G_i_kcal"] for b in inc])
    H = np.array([b["H_i_kcal"] for b in inc])
    S = np.array([b["S_msrrho_cal_per_K"] for b in inc])
    E = np.array([b["E_el_kcal"] for b in inc])
    ref = int(np.argmin(G))
    dG = G - G[ref]
    w = g * np.exp(-dG / kt)
    z = float(w.sum())
    p = w / z
    # exact mixing entropy of these populations: S'_conf = R[ln Z + beta <dG>]
    s_conf = R_CAL * (math.log(z) + float((p * dG).sum()) / kt)
    # the paper's eq. 10 with electronic energies (CREST's convention), for the seam
    dE = E - E.min()
    we = g * np.exp(-dE / kt)
    s_conf_e = R_CAL * (math.log(float(we.sum())) + float(((we / we.sum()) * dE).sum()) / kt)
    ds_bar = float((p * S).sum()) - float(S[ref])
    h_conf = float((p * (H - H[ref])).sum())
    cp_conf = R_CAL * (float((p * H * H).sum()) - float((p * H).sum()) ** 2) / kt ** 2
    s_abs = float(S[ref]) + s_conf + ds_bar
    h_abs = float(H[ref]) + h_conf
    g_total = float(G[ref]) - kt * math.log(z)
    g_total_gs = h_abs - T * s_abs / 1000.0
    order = np.argsort(-p)
    cum, n90 = 0.0, []
    for k in order:
        n90.append(int(inc[k]["index"]))
        cum += float(p[k])
        if cum >= ptot - 1e-12:
            break
    for b, pk in zip(inc, p):
        b["population"] = float(pk)
    for b in basins:
        if b.get("excluded"):
            b["population"] = 0.0
    return dict(temperature_K=T, reference_basin=int(inc[ref]["index"]),
                populations={int(b["index"]): float(pk) for b, pk in zip(inc, p)},
                S_ref_cal_per_K=float(S[ref]), H_ref_kcal=float(H[ref]), G_ref_kcal=float(G[ref]),
                S_conf_prime_cal_per_K=s_conf, S_conf_prime_E_cal_per_K=s_conf_e,
                dS_bar_cal_per_K=ds_bar, H_conf_kcal=h_conf, Cp_conf_cal_per_K=cp_conf,
                S_abs_cal_per_K=s_abs, H_abs_kcal=h_abs, G_total_kcal=g_total,
                G_total_gibbs_shannon_kcal=g_total_gs,
                n_basins=len(basins), n_included=len(inc), n_excluded=len(basins) - len(inc),
                basins_90=n90, ptot=float(ptot))


# ====================================================================== the Calculation
def _read_basins(molecule, doc):
    """Per-basin inputs from branchA.toml's [[Basin]] rows and the mace engine folder."""
    from ase.io import read
    rows = branch_a_property.basin_rows(doc)
    files = {int(p.parent.name[len("basin"):]): p for p in basins_mod.basin_files(molecule)}
    out = []
    for r in rows:
        i = int(r["INDEX"])
        if i not in files:
            raise FileNotFoundError("branchA.toml lists basin {} but mace/basin{:02d}/basin.extxyz "
                                    "is absent".format(i, i))
        atoms = read(str(files[i]), format="extxyz")
        hpath = files[i].parent / "hessian.npy"
        if not hpath.is_file():
            raise FileNotFoundError("no hessian.npy for basin {}: {}".format(i, hpath))
        for k in ("SIGMA", "G0", "ENERGY"):
            if k not in r:
                raise KeyError("basin {} of branchA.toml has no {}".format(i, k))
        out.append(dict(index=i, atoms=atoms, hessian=np.load(hpath),
                        sigma=int(r["SIGMA"]), g0=int(r["G0"]), energy_eV=float(r["ENERGY"])))
    return out


def run_calculation(molecule, level, qm9_index=None, cfg=None, preset="crest",
                    temperature_K=None, imaginary_policy=thermo.PRODUCTION_POLICY, fscal=1.0, ptot=PTOT):
    """Run `thermo_msrrho` for one molecule directory at `level`; write the Record to the
    level folder; return the full record (per-basin detail included)."""
    molecule = Path(molecule)
    rec_path = layout.records_dir(molecule) / "branchA.toml"
    if not rec_path.is_file():
        raise FileNotFoundError("no branchA.toml under {}".format(layout.records_dir(molecule)))
    doc = prop.load(rec_path)
    info_a = doc.get("Calculation_Info") or {}
    T = float(temperature_K if temperature_K is not None else info_a.get("TEMPERATURE", thermo.T_REF))
    qid = qm9_index if qm9_index is not None else info_a.get("QM9_INDEX")
    layout.thermo_dir(molecule).mkdir(parents=True, exist_ok=True)

    # g' per basin: the degeneracy Calculation, run here when its record is absent
    deg_path = layout.level_file(molecule, level, "degeneracy.toml")
    if not deg_path.is_file():
        degeneracy.run_calculation(molecule, level)
    deg = {int(r["INDEX"]): r for r in prop.load(deg_path).get("Basin", [])}

    inputs = _read_basins(molecule, doc)
    basins = []
    for b in inputs:
        r = basin_thermochemistry(b["index"], b["hessian"], b["atoms"].get_masses(),
                                  b["atoms"].get_positions(), b["energy_eV"], b["sigma"],
                                  b["g0"], T, preset, imaginary_policy, fscal)
        d = deg.get(b["index"])
        if d is None:
            raise KeyError("degeneracy.toml has no row for basin {}".format(b["index"]))
        r["g_prime"] = int(d["G_PRIME"])
        r["g_prime_source"] = str(d["G_PRIME_SOURCE"])
        basins.append(r)
    ens = assemble(basins, T, ptot)
    ref = next(b for b in basins if b["index"] == ens["reference_basin"])
    spread = thermo.preset_spread(ref["frequencies_cm"], ref["masses"], ref["positions"],
                                  temperature_K=T, imaginary_policy=imaginary_policy, fscal=fscal)
    exp = config.experimental_entropy(qid, cfg) if qid else None

    info = {"MOLECULE_DIR": str(molecule), "QM9_INDEX": qid, "TAG": info_a.get("TAG"),
            "LEVEL": str(level), "ENGINE": info_a.get("ENGINE"), "PRESET": preset,
            "TAU": float(thermo.MSRRHO_PRESETS[preset]["tau_cm"]),
            "ROTOR_CAP_RULE": str(thermo.MSRRHO_PRESETS[preset]["rotor_cap"]),
            "ITHR_POLICY": imaginary_policy, "FSCAL": float(fscal), "TEMPERATURE": T,
            "PRESSURE": float(thermo.P_STD), "REFERENCE_BASIN": ens["reference_basin"],
            "PTOT": float(ptot), "EXTRAPOLATION": "none"}
    write_records(molecule, level, info, basins, ens, spread, exp)
    out = dict(ens)
    out.update(basins=basins, info=info, preset_spread=spread,
               experimental=exp, record=layout.level_file(molecule, level, "thermo_msrrho.toml"))
    return out


def write_records(molecule, level, info, basins, ens, spread, exp, extra_blocks=None, schema=None):
    """Write `msrrho/thermo/<level>.thermo_msrrho.{toml,out}` from the pieces run_calculation
    (or a reference-level Calculation) assembled."""
    layout.thermo_dir(molecule).mkdir(parents=True, exist_ok=True)
    rows = []
    for b in basins:
        row = {"INDEX": b["index"], "SIGMA": b["sigma"], "G0": b["g0"],
               "G_PRIME": b["g_prime"], "G_PRIME_SOURCE": b["g_prime_source"],
               "E_EL": b["E_el_kcal"], "LOWEST_FREQ": b["lowest_frequency_cm"],
               "N_IMAGINARY": b.get("n_imaginary"), "N_INVERTED": b.get("n_inverted", 0),
               "N_KEPT_NEGATIVE": b.get("n_kept_negative", 0),
               "N_BELOW_FLOOR": b.get("n_below_floor", 0),
               "INVERTED_CM": b.get("inverted_frequencies_cm") or None,
               "DROPPED_CM": b.get("dropped_frequencies_cm") or None,
               "EXCLUDED": bool(b["excluded"]), "SOFT_SADDLE": bool(b.get("soft_saddle", False)),
               "POPULATION": b.get("population", 0.0),
               "HESSIAN_ROUTE": b.get("hessian_route"), "NOISE_FLOOR_CM": b.get("noise_floor_cm"),
               "OPT_GRAD_RMS": b.get("opt_grad_rms"), "N_SINGLE_POINTS": b.get("n_single_points")}
        if not b["excluded"]:
            row.update({"ZPE": b["ZPE_kcal"], "H_THERMAL": b["H_thermal_kcal"],
                        "G_ROT": b["G_rot_kcal"], "G_TRANS": b["G_trans_kcal"],
                        "S_VIB": b["S_vib_cal_per_K"], "S_VIB_HO": b["S_vib_HO_cal_per_K"],
                        "S_ROT": b["S_rot_cal_per_K"], "S_TRANS": b["S_trans_cal_per_K"],
                        "S_MSRRHO": b["S_msrrho_cal_per_K"], "H_I": b["H_i_kcal"],
                        "G_I": b["G_i_kcal"]})
        rows.append(row)
    result = {"S_ABS": ens["S_abs_cal_per_K"], "H_ABS": ens["H_abs_kcal"],
              "G_TOTAL": ens["G_total_kcal"], "N_BASINS": ens["n_basins"],
              "N_INCLUDED": ens["n_included"], "N_EXCLUDED": ens["n_excluded"],
              "N_BASINS_90": ens["basins_90"], "PRESET_SPREAD": spread["max_minus_min_kcal"]}
    if exp is not None:
        result.update({"S_EXPERIMENT": exp[0], "S_EXPERIMENT_SOURCE": exp[1],
                       "S_ABS_MINUS_EXPERIMENT": ens["S_abs_cal_per_K"] - exp[0]})
    blocks = {"Calculation_Info": info, "Basin": rows,
              "Ensemble": {"S_CONF_PRIME": ens["S_conf_prime_cal_per_K"],
                           "S_CONF_PRIME_E": ens["S_conf_prime_E_cal_per_K"],
                           "DS_BAR": ens["dS_bar_cal_per_K"], "H_CONF": ens["H_conf_kcal"],
                           "CP_CONF": ens["Cp_conf_cal_per_K"]},
              "Result": result}
    if extra_blocks:
        blocks.update(extra_blocks)
    missing = prop.write(layout.level_file(molecule, level, "thermo_msrrho.toml"), blocks, schema or SCHEMA,
                         prop.NORMAL_TERMINATION, PROGNAME)
    if missing:
        raise RuntimeError("thermo_msrrho.toml keys outside the schema: {}".format(missing))
    _write_report(layout.level_file(molecule, level, "thermo_msrrho.out"), info, basins, ens, spread, exp)
    return layout.level_file(molecule, level, "thermo_msrrho.toml")


def _write_report(path, info, basins, ens, spread, exp):
    rep = report.Report("openQHA thermo_msrrho",
                        "absolute entropy and free energy from the basins, Pracht & Grimme 2021 "
                        "assembly at one level ({})".format(info["LEVEL"]))
    rep.section("conventions")
    for k in ("PRESET", "TAU", "ROTOR_CAP_RULE", "ITHR_POLICY", "FSCAL", "TEMPERATURE",
              "PRESSURE", "PTOT", "EXTRAPOLATION"):
        rep.kv(k, info[k])
    rep.kv("preset spread on T*S_vib at the reference basin",
           "%.4f kcal/mol  (%s)" % (spread["max_minus_min_kcal"],
                                    ", ".join("%s %.4f" % (k, v) for k, v in spread["TS_vib_kcal"].items())))
    for name, why in (spread.get("presets_absent") or {}).items():
        rep.warn("preset {} absent from the spread under '{}': {}".format(name, spread.get("imaginary_policy"), why))
    rep.section("per basin: the six terms (kcal/mol) and the entropies (cal/mol/K)")
    rep.table(["basin", "sigma", "g'", "E_el", "ZPE", "H_therm", "G_rot", "G_trans", "S_vib", "S_msRRHO", "G_i", "p"],
              [[b["index"], b["sigma"], b["g_prime"], "%.4f" % b["E_el_kcal"],
                "%.4f" % b["ZPE_kcal"], "%.4f" % b["H_thermal_kcal"], "%.4f" % b["G_rot_kcal"],
                "%.4f" % b["G_trans_kcal"], "%.3f" % b["S_vib_cal_per_K"],
                "%.3f" % b["S_msrrho_cal_per_K"], "%.4f" % b["G_i_kcal"], "%.4f" % b.get("population", 0.0)]
               if not b["excluded"] else
               [b["index"], b["sigma"], b["g_prime"], "%.4f" % b["E_el_kcal"], "EXCLUDED", "", "", "", "", "", "", "0"]
               for b in basins])
    for b in basins:
        if b["excluded"]:
            rep.warn("basin {} excluded: {}".format(b["index"], b["excluded_reason"]))
        elif b.get("n_kept_negative") or b.get("n_inverted") or b.get("n_below_floor"):
            rep.warn("basin {}: {} mode(s) inverted, {} dropped (|omega| < 1 cm^-1), {} kept "
                     "negative under '{}' (lowest {:.2f} cm^-1)".format(
                         b["index"], b.get("n_inverted", 0), b.get("n_below_floor", 0),
                         b.get("n_kept_negative", 0), info["ITHR_POLICY"],
                         b["lowest_frequency_cm"]))
    for b in basins:
        if b["excluded"]:
            continue
        low = [m for m in b["modes"] if m["omega_cm"] <= thermo.MODE_TABLE_MAX_CM]
        if low:
            rep.table(["mode omega", "T*S(HO)", "T*S(FR)", "w_HO", "T*S(vib)"],
                      [["%.2f" % m["omega_cm"], "%.5f" % m["TS_HO_kcal"], "%.5f" % m["TS_FR_kcal"],
                        "%.4f" % m["w_HO"], "%.5f" % m["TS_kcal"]] for m in low],
                      title="basin {}: modes up to {:.0f} cm^-1 under the modified RRHO".format(
                          b["index"], thermo.MODE_TABLE_MAX_CM))
    rep.section("ensemble (cal/mol/K unless noted)")
    rep.kv("reference basin", ens["reference_basin"])
    rep.kv("S_ref (S_msRRHO of the reference)", "%.4f" % ens["S_ref_cal_per_K"])
    rep.kv("S'_conf (mixing entropy, populations from G_i)", "%.4f" % ens["S_conf_prime_cal_per_K"])
    rep.kv("S'_conf with electronic energies (paper eq. 10, CREST)", "%.4f" % ens["S_conf_prime_E_cal_per_K"])
    rep.kv("dS_bar (population average minus reference)", "%.4f" % ens["dS_bar_cal_per_K"])
    rep.kv("H_conf", "%.4f kcal/mol" % ens["H_conf_kcal"])
    rep.kv("Cp_conf", "%.4f" % ens["Cp_conf_cal_per_K"])
    rep.kv("S_abs", "%.4f" % ens["S_abs_cal_per_K"])
    rep.kv("G_total", "%.4f kcal/mol" % ens["G_total_kcal"])
    rep.kv("G_total via H_abs - T S_abs (identity check)", "%.4f kcal/mol" % ens["G_total_gibbs_shannon_kcal"])
    rep.kv("basins carrying {:.0%} of the population".format(ens["ptot"]), ens["basins_90"])
    rep.note("dS_bar is an identity at one level (the reference subtraction cancels exactly); "
             "it is written out so that a two-level assembly drops into the same file. No "
             "extrapolation to ensemble completeness: the basins come from one search.")
    rep.section("imaginary modes under the frequency floor")
    rep.kv("policy", info["ITHR_POLICY"])
    rep.table(["basin", "lowest cm^-1", "imaginary", "inverted", "dropped |omega|<1", "kept negative", "excluded"],
              [[b["index"], "%.2f" % b["lowest_frequency_cm"], b.get("n_imaginary") or 0,
                b.get("n_inverted", 0), b.get("n_below_floor", 0),
                b.get("n_kept_negative", 0), "yes" if b["excluded"] else "no"]
               for b in basins])
    rep.note("invert_below (production): a mode in [ithr, 0) takes |omega|; a mode below "
             "ithr excludes the basin. crest_native (the GFN2 seam only): CREST 3.0.2 line "
             "for line -- modes in [ithr, 0) inverted, modes below ithr kept negative with "
             "zero entropy but present in ZPE, H(T)-H(0) and Cp (thermocalc.f90:209, "
             "thermo.f90:135). A mode with |omega| < 1 cm^-1 is dropped from every sum "
             "ORCA-style and counted; three or more raise.")
    rep.section("experiment")
    if exp is None:
        rep.text("  no experimental entropy declared for this molecule")
    else:
        rep.kv("S(298.15 K) experiment", "%.2f cal/mol/K  [%s]" % (exp[0], exp[1]))
        rep.kv("S_abs - experiment", "%+.3f cal/mol/K" % (ens["S_abs_cal_per_K"] - exp[0]))
    rep.write(path, step=STEP)
