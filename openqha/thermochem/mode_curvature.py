"""The dry run: the curvature of a higher level along the reference normal
modes, from energies alone.

WHY ENERGIES
------------
ORCA has no analytic gradient for (DLPNO-)CCSD(T). The full Hessian at that level is
`Opt NumGrad + NumFreq`: (6N)^2 single points, doubly numerical, run on hkuhpc (the Batch).
Before 4,000 points are submitted, and as the number the returned NumFreq
must reproduce, the curvature ALONG A CHOSEN REFERENCE MODE needs only a line of energies:

    x(k) = x_r + k * delta_q * M^-1/2 L_r,i        k = -2, -1, 0, +1, +2
    lambda_i = second difference of E(k) in delta_q^2    (3-point and 5-point)
    omega_i  = sign(lambda) sqrt|lambda|  in cm^-1, the same conversion as every spectrum

with L_r,i the mass-weighted Eckart-projected reference mode (unit vector) and delta_q
in amu^1/2 A, chosen so that the harmonic energy rise at k = 1 on the reference surface
is TARGET_RISE_EH, and capped for a near-zero mode. This is family 4's
`D_ii = L_r,i^T K_e L_r,i` measured by finite differences instead of a matrix -- so the
same points evaluated on the reference surface must give back omega_r,i (self-check 1)
and on the engine surface must give back `hessian_compare`'s `OMEGA_ALONG_REF_CM[i]`
(self-check 2).

WHY THE RISE IS 1e-5 Eh AND NOT 1e-4 (measured 2026-09-17, propanal's 131 cm^-1 torsion)
-----------------------------------------------------------------------------------------
A straight Cartesian line along a torsional mode stretches bonds in proportion to
(k delta_q)^2, so the line carries a quartic (and higher) term that is not the mode's
own. At a 1e-4 Eh rise (delta_q 0.29, hydrogens displaced 0.2-0.3 A) that term was 87 %
of the harmonic rise at k = 1 on wB97M, E(2)/E(1) was 8-17 instead of 4, and the 5-point
curvature scattered from 271 (HF) through 107 (RI-MP2) to ~0 cm^-1 (DLPNO-CCSD(T)) on a
mode experiment puts at 135.1 cm^-1 and nearly harmonic (v = 2 at 269.3). The DLPNO
surface itself is smooth to < 1e-7 Eh (E(k = 0.05) - E0 = -7e-9), so the rise can be
ten times smaller: at 1e-5 Eh the stretch term is ~9 % and the noise ~1 %. Two error
bars are recorded: |5-point - 3-point| (anharmonicity on the line) and the 5-point at
delta_q/2 against delta_q (DELTA_CONVERGENCE_CM, two extra points at k = +-1/2), which
is the one the returned NumFreq is held to. A mode below PROFILE_BELOW_CM (oxetane's
ring puckering, 16.7 cm^-1 at wB97M: the near-barrierless double well) additionally
gets a 7-point wide line at DELTA_Q_MAX whose energies are the profile, with the depth
of the deepest point below k = 0 in cm^-1 (negative: a double well at that level).

FILES
-----
Engine files: `msrrho/orca.<level>.basinNN.modeII_dqD.DDDD_kK.{inp,out}` per point (a
file group; full `.out` kept; the tag carries delta_q so a rerun at another rise never
reuses a point), the k = 0 point shared as `.k0`; the reference-surface points likewise
as `orca.<reference level>.basinNN.<tag>.*`. Record:
`msrrho/thermo/<level>.mode_curvature_dryrun.{out,toml}`.
"""
import math
import time
from pathlib import Path

import numpy as np

from ..potentials import engine
from ..qm_interfaces import orca
from ..store import dat, layout, property as prop, report
from . import hessian as hessian_mod
from . import hessian_compare as hc
from . import thermo

STEP = "mode_curvature_dryrun"
PROGNAME = "openQHA mode_curvature"
#: harmonic energy rise at k = 1 on the reference surface that fixes delta_q (see the
#: module docstring for why 1e-5 and not 1e-4)
TARGET_RISE_EH = 1.0e-5
#: delta_q never above this (a flat mode would ask for an absurd displacement)
DELTA_Q_MAX = 0.35            # amu^1/2 A
#: a reference mode below this also gets the 7-point wide profile at DELTA_Q_MAX
PROFILE_BELOW_CM = 30.0
#: half-step points for the delta_q convergence check
HALF_KS = (-0.5, 0.5)
EV_PER_EH = orca.EV_PER_HARTREE
CM_PER_EH = 219474.6313632

SCHEMA = {
    "Calculation_Info": {
        "MOLECULE_DIR": ("String", None, "the molecule directory"),
        "QM9_INDEX": ("String", None, "the molecule"),
        "LEVEL": ("String", None, "the higher level probed along the reference modes"),
        "REFERENCE_LEVEL": ("String", None, "the level whose modes and geometry are used"),
        "ENGINE_LEVEL": ("String", None, "the potential evaluated on the same points"),
        "KEYWORDS": ("String", None, "ORCA keyword line of the higher-level single points"),
        "BLOCKS": ("String", None, "extra ORCA input blocks"),
        "TARGET_RISE_EH": ("Double", "Eh", "harmonic rise at k = 1 that fixes delta_q"),
        "DELTA_Q_MAX": ("Double", "amu^1/2 A", "cap on delta_q"),
        "PROFILE_BELOW_CM": ("Double", "cm^-1", "modes below this also get the 7-point wide profile"),
        "N_SINGLE_POINTS": ("Integer", None, "higher-level single points run or reused"),
        "SECONDS_HIGHER_LEVEL": ("Double", "s", "wall time spent in the higher-level points (new ones)"),
        "MODE_RULE": ("String", None, "which modes: lowest, low (< 300), all, or an explicit list"),
    },
    "Mode": {
        "BASIN": ("Integer", None, "MACE basin index"),
        "REF_BASIN": ("Integer", None, "reference basin index"),
        "MODE": ("Integer", None, "reference mode index (ascending omega)"),
        "OMEGA_R": ("Double", "cm^-1", "reference frequency of the mode (from the .hess)"),
        "DELTA_Q": ("Double", "amu^1/2 A", "displacement step"),
        "POINTS": ("Integer", None, "higher-level points on this mode (5 + 2 half-step, + 6 wide for a profile mode)"),
        "OMEGA_HIGHER_5PT": ("Double", "cm^-1", "higher-level curvature along the mode, 5-point second difference"),
        "OMEGA_HIGHER_3PT": ("Double", "cm^-1", "the same, 3-point"),
        "FD_ERROR_CM": ("Double", "cm^-1", "|5-point - 3-point|: anharmonicity on the line"),
        "OMEGA_HIGHER_5PT_HALF": ("Double", "cm^-1", "higher-level 5-point curvature on the half-step line (k = -1, -1/2, 0, 1/2, 1)"),
        "DELTA_CONVERGENCE_CM": ("Double", "cm^-1", "|5-point at delta_q - 5-point at delta_q/2|: the delta_q error bar the NumFreq is held to"),
        "OMEGA_REF_FD_HALF": ("Double", "cm^-1", "reference surface, 5-point on the half-step line"),
        "OMEGA_REF_FD": ("Double", "cm^-1", "reference surface on the same points, 5-point (self-check: must reproduce OMEGA_R)"),
        "OMEGA_ENGINE_FD": ("Double", "cm^-1", "engine surface on the same points, 5-point (self-check: hessian_compare's D_ii)"),
        "OMEGA_ENGINE_D": ("Double", "cm^-1", "hessian_compare's curvature along this mode from the stored engine Hessian"),
        "MODEL_ERROR_CM": ("Double", "cm^-1", "engine minus reference, both by finite differences"),
        "LEVEL_ERROR_CM": ("Double", "cm^-1", "reference minus higher level, both by finite differences"),
        "TS_REF_KCAL": ("Double", "kcal/mol", "per-mode T*S under the crest preset at OMEGA_REF_FD"),
        "TS_HIGHER_KCAL": ("Double", "kcal/mol", "the same at OMEGA_HIGHER_5PT (0 when not positive)"),
        "TS_ENGINE_KCAL": ("Double", "kcal/mol", "the same at OMEGA_ENGINE_FD"),
        "PROFILE_K": ("ArrayOfIntegers", None, "k of every point on the line"),
        "PROFILE_HIGHER_EH": ("ArrayOfDoubles", "Eh", "higher-level energies relative to k = 0"),
        "PROFILE_REF_EH": ("ArrayOfDoubles", "Eh", "reference energies relative to k = 0"),
        "PROFILE_ENGINE_EH": ("ArrayOfDoubles", "Eh", "engine energies relative to k = 0"),
        "WIDE_DELTA_Q": ("Double", "amu^1/2 A", "step of the wide 7-point profile (profile modes only)"),
        "WIDE_K": ("ArrayOfIntegers", None, "k of every point on the wide profile"),
        "WIDE_HIGHER_EH": ("ArrayOfDoubles", "Eh", "higher-level energies on the wide profile relative to k = 0"),
        "WIDE_REF_EH": ("ArrayOfDoubles", "Eh", "reference energies on the wide profile relative to k = 0"),
        "WIDE_ENGINE_EH": ("ArrayOfDoubles", "Eh", "engine energies on the wide profile relative to k = 0"),
        "WELL_DEPTH_HIGHER_CM": ("Double", "cm^-1", "min over the wide profile of E - E0 at the higher level (negative: double well)"),
        "WELL_DEPTH_REF_CM": ("Double", "cm^-1", "the same on the reference surface"),
        "WELL_DEPTH_ENGINE_CM": ("Double", "cm^-1", "the same on the engine surface"),
        "SECONDS_PER_POINT": ("Double", "s", "mean wall time of the new higher-level points on this mode"),
        "COMPLETE": ("Boolean", None, "every point finished"),
    },
    "Summary": {
        "N_MODES": ("Integer", None, "modes probed"),
        "MAE_MODEL_ERROR_CM": ("Double", "cm^-1", "mean |engine - reference| over the modes"),
        "MAE_LEVEL_ERROR_CM": ("Double", "cm^-1", "mean |reference - higher| over the modes"),
        "MAX_FD_ERROR_CM": ("Double", "cm^-1", "largest finite-difference error bar"),
        "MAX_SELF_CHECK_CM": ("Double", "cm^-1", "largest |OMEGA_REF_FD - OMEGA_R|"),
        "MAX_DELTA_CONVERGENCE_CM": ("Double", "cm^-1", "largest delta_q error bar"),
    },
}


# ====================================================================== geometry line
def displaced_line(positions, masses, mode_vec, delta_q, ks):
    """Geometries x_r + k delta_q M^-1/2 L for the k in `ks` (mode_vec: the mass-weighted
    unit mode as a 3N vector)."""
    m3 = np.repeat(np.asarray(masses, dtype=float), 3)
    step = (np.asarray(mode_vec, dtype=float) / np.sqrt(m3)).reshape(-1, 3)
    return [np.asarray(positions, dtype=float) + k * delta_q * step for k in ks]


def delta_for(omega_cm, target_rise_eh=TARGET_RISE_EH, cap=DELTA_Q_MAX):
    """delta_q so that 1/2 lambda delta_q^2 = target rise on the reference surface, capped."""
    lam = (abs(float(omega_cm)) / hessian_mod.CM_INV_PER_SQRT_EV_A2_AMU) ** 2   # eV/A^2/amu
    rise_ev = target_rise_eh * EV_PER_EH
    if lam <= 0.0:
        return cap
    return float(min(math.sqrt(2.0 * rise_ev / lam), cap))


def curvature_from_line(ks, energies_eh, delta_q):
    """(omega_5pt, omega_3pt) in cm^-1 from energies (Eh) at k in ks (must hold -2..2 or
    at least -1..1 with 0). The second difference is in eV/A^2/amu."""
    e = {float(k): float(v) * EV_PER_EH for k, v in zip(ks, energies_eh)}   # 1 == 1.0 as a key
    d2 = delta_q ** 2
    lam3 = (e[1] - 2.0 * e[0] + e[-1]) / d2
    lam5 = None
    if all(k in e for k in (-2, 2)):
        lam5 = (-e[2] + 16.0 * e[1] - 30.0 * e[0] + 16.0 * e[-1] - e[-2]) / (12.0 * d2)
    to_cm = lambda lam: float(hessian_mod.eigenvalues_to_cm_inv(np.array([lam]))[0])  # noqa: E731
    return (to_cm(lam5) if lam5 is not None else None), to_cm(lam3)


def _ts_kcal(omega_cm, masses, positions, preset, temperature_K):
    if omega_cm is None or omega_cm < thermo.VIBTHR_CM:
        return 0.0
    r = thermo.msrrho([float(omega_cm)], masses, positions, preset=preset, temperature_K=temperature_K)
    return float(r["modes"][0]["TS_kcal"])


# ====================================================================== the dry run
def select_modes(freqs_cm, rule):
    """Indices of the reference modes to probe: 'lowest', 'low' (< LOW_CM of
    hessian_compare), 'all', or a comma-separated list of indices."""
    f = np.asarray(freqs_cm, dtype=float)
    if rule == "lowest":
        return [0]
    if rule == "low":
        return [int(i) for i in np.where(f < hc.LOW_CM)[0]] or [0]
    if rule == "all":
        return list(range(len(f)))
    return [int(x) for x in str(rule).split(",") if x.strip()]


def run_calculation(molecule, level="dlpno-ccsdt_cc-pvtz", reference_level=engine.REFERENCE_LEVEL,
                    qm9_index=None, modes="lowest", basins=None, nprocs=8, maxcore=4000,
                    preset="crest", temperature_K=thermo.T_REF, engine_name=None,
                    higher_keywords=None, higher_blocks=None, target_rise_eh=TARGET_RISE_EH):
    """Probe the higher level along the selected reference modes of every kept basin;
    evaluate the same points on the reference and engine surfaces; write the record."""
    from ase import Atoms
    from ase.data import atomic_masses, atomic_numbers
    molecule = Path(molecule)
    doc_a = prop.load(layout.records_dir(molecule) / "branchA.toml")
    info_a = doc_a.get("Calculation_Info") or {}
    qid = qm9_index if qm9_index is not None else info_a.get("QM9_INDEX")
    engine_level = engine.level_name(info_a.get("ENGINE")) if info_a.get("ENGINE") else None
    hspec = orca.level_spec(level)
    rspec = orca.level_spec(reference_level)
    hkw = higher_keywords or hspec["single_point"]
    hblk = hspec["blocks"] if higher_blocks is None else higher_blocks
    rkw = rspec["single_point"]
    merge = dat.read_table(layout.level_file(molecule, reference_level, "merge_map.dat"))
    kept = [r for r in merge if r["status"] == "kept"]
    if basins is not None:
        kept = [r for r in kept if int(r["mace_basin"]) in {int(b) for b in basins}]
    calc, ename, _prov = engine.calculator(name=engine_name)
    hc_rows = {}
    hc_path = layout.thermo_file(molecule, "hessian_compare.toml")
    if hc_path.is_file():
        hc_rows = {int(r["MACE_BASIN"]): r for r in prop.load(hc_path).get("Basin", [])}

    rows, n_points, secs_higher = [], 0, 0.0
    for r in kept:
        mb, rb = int(r["mace_basin"]), int(r["reference_basin"])
        parsed = orca.parse_hess(layout.orca_level_file(molecule, reference_level, mb, ".hess"))
        symbols = list(parsed["symbols"])
        x_r = np.asarray(parsed["positions_bohr"]) / orca.BOHR_PER_ANGSTROM
        masses = [float(atomic_masses[atomic_numbers[s]]) for s in symbols]
        pr = hc.projected(orca.hessian_to_ev_per_angstrom2(parsed["hessian_eh_bohr2"]), masses, x_r)
        for i in select_modes(pr["freq"], modes):
            omega_r = float(pr["freq"][i])
            dq = delta_for(omega_r, target_rise_eh)
            wide = omega_r < PROFILE_BELOW_CM
            # (k, delta_q) of every point: the 5-point line, the two half-step points, and for a
            # profile mode the wide 7-point line at the cap; k = 0 is one shared point
            pts = [(k, dq) for k in (-2, -1, 0, 1, 2)] + [(k, dq) for k in HALF_KS]
            if wide:
                pts += [(k, DELTA_Q_MAX) for k in (-3, -2, -1, 1, 2, 3)]
            geoms = displaced_line(x_r, masses, pr["vec"][:, i], 1.0, [k * d for k, d in pts])
            e_h, e_r, e_e, secs, complete = {}, {}, {}, [], True
            for (k, d), g in zip(pts, geoms):
                tag = "mode{:02d}_dq{:.4f}_k{}".format(i, d, k) if k != 0 else "k0"
                try:
                    hp = orca.run_single_point(symbols, g, layout.msrrho_dir(molecule), hkw,
                                               stem=layout.orca_level_stem(level, mb) + "." + tag,
                                               nprocs=nprocs, maxcore=maxcore, blocks=hblk)
                except RuntimeError as exc:
                    complete = False
                    report_fail = str(exc)[:200]
                    continue
                n_points += 1
                if hp["seconds"] is not None:
                    secs.append(hp["seconds"]); secs_higher += hp["seconds"]
                rp = orca.run_single_point(symbols, g, layout.msrrho_dir(molecule), rkw,
                                           stem=layout.orca_level_stem(reference_level, mb) + "." + tag,
                                           nprocs=nprocs, maxcore=maxcore)
                atoms = Atoms(symbols=symbols, positions=g)
                atoms.calc = calc
                e_h[(k, d)] = hp["energy_eh"]; e_r[(k, d)] = rp["energy_eh"]
                e_e[(k, d)] = float(atoms.get_potential_energy()) / EV_PER_EH
            ks = [-2, -1, 0, 1, 2]
            row = {"BASIN": mb, "REF_BASIN": rb, "MODE": i, "OMEGA_R": omega_r, "DELTA_Q": dq,
                   "POINTS": len(pts), "PROFILE_K": ks, "COMPLETE": complete,
                   "SECONDS_PER_POINT": float(np.mean(secs)) if secs else None}
            if complete:
                line = lambda e, kk, d: [e[(k, d)] - e[(0, dq)] for k in kk]                        # noqa: E731
                row.update(PROFILE_HIGHER_EH=line(e_h, ks, dq), PROFILE_REF_EH=line(e_r, ks, dq),
                           PROFILE_ENGINE_EH=line(e_e, ks, dq))
                w5h, w3h = curvature_from_line(ks, row["PROFILE_HIGHER_EH"], dq)
                w5r, _ = curvature_from_line(ks, row["PROFILE_REF_EH"], dq)
                w5e, _ = curvature_from_line(ks, row["PROFILE_ENGINE_EH"], dq)
                # the half-step line: k = -1, -1/2, 0, 1/2, 1 at delta_q is -2..2 at delta_q/2
                half = [-1, -0.5, 0, 0.5, 1]
                w5h_half, _ = curvature_from_line(ks, line(e_h, half, dq), dq / 2.0)
                w5r_half, _ = curvature_from_line(ks, line(e_r, half, dq), dq / 2.0)
                row.update(OMEGA_HIGHER_5PT_HALF=w5h_half, DELTA_CONVERGENCE_CM=abs(w5h - w5h_half),
                           OMEGA_REF_FD_HALF=w5r_half)
                if wide:
                    wk = [-3, -2, -1, 0, 1, 2, 3]
                    e_h[(0, DELTA_Q_MAX)], e_r[(0, DELTA_Q_MAX)], e_e[(0, DELTA_Q_MAX)] = e_h[(0, dq)], e_r[(0, dq)], e_e[(0, dq)]
                    prof = {n: line(e, wk, DELTA_Q_MAX) for n, e in (("HIGHER", e_h), ("REF", e_r), ("ENGINE", e_e))}
                    row.update(WIDE_DELTA_Q=DELTA_Q_MAX, WIDE_K=wk,
                               **{"WIDE_%s_EH" % n: v for n, v in prof.items()},
                               **{"WELL_DEPTH_%s_CM" % n: float(min(v)) * CM_PER_EH for n, v in prof.items()})
                row.update(OMEGA_HIGHER_5PT=w5h, OMEGA_HIGHER_3PT=w3h, FD_ERROR_CM=abs(w5h - w3h),
                           OMEGA_REF_FD=w5r, OMEGA_ENGINE_FD=w5e,
                           MODEL_ERROR_CM=w5e - w5r, LEVEL_ERROR_CM=w5r - w5h,
                           TS_REF_KCAL=_ts_kcal(w5r, masses, x_r, preset, temperature_K),
                           TS_HIGHER_KCAL=_ts_kcal(w5h, masses, x_r, preset, temperature_K),
                           TS_ENGINE_KCAL=_ts_kcal(w5e, masses, x_r, preset, temperature_K))
                if mb in hc_rows and i < len(hc_rows[mb]["OMEGA_ALONG_REF_CM"]):
                    row["OMEGA_ENGINE_D"] = float(hc_rows[mb]["OMEGA_ALONG_REF_CM"][i])
            rows.append(row)
    done = [r for r in rows if r["COMPLETE"]]
    summary = {"N_MODES": len(rows)}
    if done:
        summary.update(MAE_MODEL_ERROR_CM=float(np.mean([abs(r["MODEL_ERROR_CM"]) for r in done])),
                       MAE_LEVEL_ERROR_CM=float(np.mean([abs(r["LEVEL_ERROR_CM"]) for r in done])),
                       MAX_FD_ERROR_CM=float(max(r["FD_ERROR_CM"] for r in done)),
                       MAX_SELF_CHECK_CM=float(max(abs(r["OMEGA_REF_FD"] - r["OMEGA_R"]) for r in done)),
                       MAX_DELTA_CONVERGENCE_CM=float(max(r["DELTA_CONVERGENCE_CM"] for r in done)))
    info = {"MOLECULE_DIR": str(molecule), "QM9_INDEX": qid, "LEVEL": level, "REFERENCE_LEVEL": reference_level,
            "ENGINE_LEVEL": engine_level, "KEYWORDS": hkw, "BLOCKS": hblk.replace("\n", " ; "),
            "TARGET_RISE_EH": float(target_rise_eh), "DELTA_Q_MAX": DELTA_Q_MAX, "PROFILE_BELOW_CM": PROFILE_BELOW_CM,
            "N_SINGLE_POINTS": n_points, "SECONDS_HIGHER_LEVEL": secs_higher, "MODE_RULE": str(modes)}
    layout.thermo_dir(molecule).mkdir(parents=True, exist_ok=True)
    rec = layout.level_file(molecule, level, STEP + ".toml")
    missing = prop.write(rec, {"Calculation_Info": info, "Mode": rows, "Summary": summary},
                         SCHEMA, prop.NORMAL_TERMINATION, PROGNAME)
    if missing:
        raise RuntimeError("{}.toml keys outside the schema: {}".format(STEP, missing))
    _write_report(layout.level_file(molecule, level, STEP + ".out"), info, rows, summary)
    return dict(info=info, modes=rows, summary=summary, record=rec)


def _write_report(path, info, rows, summary):
    rep = report.Report("openQHA mode_curvature", "{} curvature along the {} normal modes, from energies "
                        "(the dry run before the numerical Hessian)".format(info["LEVEL"], info["REFERENCE_LEVEL"]))
    rep.section("conventions")
    for k in ("LEVEL", "REFERENCE_LEVEL", "ENGINE_LEVEL", "KEYWORDS", "TARGET_RISE_EH", "DELTA_Q_MAX",
              "PROFILE_BELOW_CM", "MODE_RULE", "N_SINGLE_POINTS", "SECONDS_HIGHER_LEVEL"):
        rep.kv(k, info[k])
    rep.section("per mode (cm^-1; T*S in kcal/mol)")
    rep.table(["basin", "mode", "omega_r", "delta_q", "pts", "higher 5pt", "3pt", "fd err", "5pt dq/2", "dq err",
               "ref fd", "engine fd", "engine D_ii",
               "model err", "level err", "T*S ref", "T*S higher", "T*S engine", "s/pt"],
              [[r["BASIN"], r["MODE"], "%.2f" % r["OMEGA_R"], "%.4f" % r["DELTA_Q"], r["POINTS"]]
               + (["%.2f" % r["OMEGA_HIGHER_5PT"], "%.2f" % r["OMEGA_HIGHER_3PT"], "%.2f" % r["FD_ERROR_CM"],
                   "%.2f" % r["OMEGA_HIGHER_5PT_HALF"], "%.2f" % r["DELTA_CONVERGENCE_CM"],
                   "%.2f" % r["OMEGA_REF_FD"], "%.2f" % r["OMEGA_ENGINE_FD"],
                   "%.2f" % r["OMEGA_ENGINE_D"] if "OMEGA_ENGINE_D" in r else "-",
                   "%+.2f" % r["MODEL_ERROR_CM"], "%+.2f" % r["LEVEL_ERROR_CM"],
                   "%.4f" % r["TS_REF_KCAL"], "%.4f" % r["TS_HIGHER_KCAL"], "%.4f" % r["TS_ENGINE_KCAL"]]
                  if r["COMPLETE"] else ["INCOMPLETE"] + [""] * 12)
               + ["%.0f" % r["SECONDS_PER_POINT"] if r.get("SECONDS_PER_POINT") else "-"] for r in rows])
    for r in rows:
        if r["COMPLETE"] and "WIDE_K" in r:
            rep.table(["k", "higher E - E0 (Eh)", "ref E - E0", "engine E - E0"],
                      [[k, "%.7f" % a, "%.7f" % b, "%.7f" % c] for k, a, b, c in
                       zip(r["WIDE_K"], r["WIDE_HIGHER_EH"], r["WIDE_REF_EH"], r["WIDE_ENGINE_EH"])],
                      title="basin {} mode {} ({:.1f} cm^-1 at the reference): the wide energy profile along the mode, "
                            "delta_q {:.4f} amu^1/2 A; well depth (cm^-1) higher {:+.1f}, ref {:+.1f}, engine {:+.1f}".format(
                                r["BASIN"], r["MODE"], r["OMEGA_R"], r["WIDE_DELTA_Q"], r["WELL_DEPTH_HIGHER_CM"],
                                r["WELL_DEPTH_REF_CM"], r["WELL_DEPTH_ENGINE_CM"]))
    rep.section("summary")
    for k, v in summary.items():
        rep.kv(k, "%.3f" % v if isinstance(v, float) else v)
    rep.note("omega along a mode = sign sqrt|d2E/dq2| with q the mass-weighted normal coordinate; the "
             "reference surface on the same points must give back omega_r (self-check), the engine "
             "surface must give back hessian_compare's D_ii. The higher level has no analytic Hessian, "
             "so this line of energies is what its NumFreq (the hkuhpc Batch) must agree with, mode by "
             "mode, within the delta_q error bar (dq err) and the numerical Hessian's noise floor. fd err "
             "(5-point vs 3-point) measures the anharmonicity of the line, not the error of the 5-point number.")
    rep.write(path, step=STEP)
