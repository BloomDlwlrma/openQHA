"""Ticket 27: the engine's Hessian against the reference Hessian at the reference geometry.

THE QUESTION
------------
How wrong is MACE's curvature where the reference level says the minimum is? Both
Hessians are taken at ONE point, the reference geometry x_r (`$atoms` of the ORCA
`.hess`), so the comparison is between two matrices in one space and no mode assignment
has to be invented. That number -- not S_abs -- is what Hessian learning (plan C) must
reduce, and it is the quantity the loss of Projected Hessian Learning is built on.

FOUR FAMILIES, AS THE MLIP-HESSIAN LITERATURE REPORTS THEM
---------------------------------------------------------
1. Element-wise, Cartesian (HIP `scripts/eval_horm.py`, PFT, Rodriguez 2025, PHL):
   `HESSIAN_MAE`, `HESSIAN_RMSE` in eV/A^2 over the raw 3N x 3N matrices, and
   `ASYMMETRY_MAE` of the engine matrix (0 for an AD Hessian; HIP's metric for a direct
   Hessian head, kept so the record reads against its tables).
2. Basis-free, projected: `K = P M^-1/2 H M^-1/2 P` with the Eckart projector P (the same
   `rigid_body_vectors` every spectrum in this package uses), `HESS_REL_FROBENIUS =
   ||K_e - K_r||_F / ||K_r||_F`.
3. Spectrum at sorted index (Weyl: |lambda_i(A) - lambda_i(B)| <= ||A - B||_2, so sorted
   pairing is the one that inherits the matrix bound; every frequency MAE in the
   literature is this): `EIGVAL_MAE_ECKART`, `EIGVAL1_MAE_ECKART`, `FREQ_MAE_CM`,
   `FREQ_MAX_CM`, the same over reference modes below 300 cm^-1 (`_LOW`, where the
   entropy is decided), `SOFTENING_SLOPE` (least squares omega_e = s omega_r through the
   origin, Deng et al. 2025), lowest modes, imaginary counts, `NEG_NUM_AGREE`.
4. Mode-resolved on the reference eigenbasis, no pairing at all: `D = L_r^T K_e L_r`.
   The diagonal is the engine's curvature along each reference normal mode -- PHL's
   Hessian-vector product v^T H v with the DFT modes as deterministic probe vectors --
   giving `OMEGA_ALONG_REF_CM[i] = sign sqrt|D_ii|`; the off-diagonal norm is the mode
   mixing, `MIXING = ||D - diag D||_F / ||D||_F`. Eigenvector agreement as HIP reports it
   (`EIGVEC1_COS_ECKART`, `EIGVEC2_COS_ECKART` at sorted index; `EIGVEC_OVERLAP_ERROR =
   ||abs(L_e^T L_r) - I||_F`) and made safe on degenerate blocks with
   `mode_match.overlap_matrix` + `degenerate_blocks` (`MIN_BLOCK_OVERLAP`).
   `mode_match.match` and `hybrid_spectrum` are NOT used: their argmax pairing with
   counted collisions was built for two bases from different estimators (covariance vs
   Hessian); here the bases are exact eigenbases of two matrices at one point.

Curvature in the project's units, still a Hessian metric at x_r: the `crest`-preset
per-mode T*S on omega_r and on the curvature along the reference modes ->
`TS_LOW_DELTA` (sum over the low modes), `S_VIB_CURVATURE_DELTA`, `ZPE_CURVATURE_DELTA`.
The engine's residual force at x_r is recorded (`MAX_FORCE_ENGINE_AT_REF_EV_A`): the
projected Hessian at a non-stationary point is the standard treatment, not an exact one.

WHAT IS NOT COMPARED HERE
-------------------------
Thermochemistry proper is compared at each level's OWN geometry: the two levels'
`thermo_msrrho.toml` `[[Basin]]` rows are set side by side (paired by the merge map)
and the `[Ensemble]` / `[Result]` terms likewise -- read, never recomputed at x_r.
`$dipole_derivatives` exist in the ORCA `.hess`; MACE-OFF23 carries no charges, so at
that level `DIPOLE_DERIVATIVES_PRESENT = false` and no dipole block is written.

FILES
-----
Engine files (ADR 0001): `mace/basinNN/hessian_at_<level>.npy` (eV/A^2) and
`forces_at_<level>.npy` (eV/A), the engine evaluated at x_r; reused when present.
Record: `levels/hessian_compare.{out,toml}`.
"""
import math
from pathlib import Path

import numpy as np

from ..potentials import engine
from ..qm_interfaces import orca
from ..quasi_harmonic import mode_match
from ..store import branch_a_property, dat, layout, property as prop, report
from . import hessian as hessian_mod
from . import thermo

STEP = "hessian_compare"
PROGNAME = "openQHA hessian_compare"
#: Reference modes below this carry the entropy; the low-mode statistics are on them.
LOW_CM = 300.0
GAP_CM = mode_match.DEFAULT_DEGENERACY_GAP_CM
#: Per-mode T*S is evaluated with CREST's regime so that a curvature that comes out
#: negative along a reference mode still yields a number (kept, zero entropy) and is
#: reported, never refused.
CURVATURE_POLICY = "crest_native"

SCHEMA = {
    "Calculation_Info": {
        "MOLECULE_DIR": ("String", None, "the molecule directory"),
        "QM9_INDEX": ("String", None, "the molecule"),
        "ENGINE_LEVEL": ("String", None, "the potential's level"),
        "REFERENCE_LEVEL": ("String", None, "the level the Hessian is compared against"),
        "GEOMETRY": ("String", None, "where both Hessians are taken: reference"),
        "PRESET": ("String", None, "msRRHO preset of the per-mode T*S"),
        "TEMPERATURE": ("Double", "K", "temperature of the per-mode T*S"),
        "LOW_CUTOFF": ("Double", "cm^-1", "reference modes below this are the low modes"),
        "DEGENERACY_GAP": ("Double", "cm^-1", "modes closer than this form a degenerate block"),
        "N_BASINS_COMPARED": ("Integer", None, "MACE basins kept at the reference level"),
        "N_MERGED": ("Integer", None, "MACE basins that merged into another (not compared)"),
        "N_SADDLE": ("Integer", None, "MACE basins that became saddles (not compared)"),
        "DIPOLE_DERIVATIVES_PRESENT": ("Boolean", None, "both engines provide dipole derivatives"),
        "ENGINE": ("String", None, "the potential that produced the engine Hessians"),
    },
    "Basin": {
        "MACE_BASIN": ("Integer", None, "branch A basin index"),
        "REF_BASIN": ("Integer", None, "reference basin index"),
        "MAX_FORCE_ENGINE_AT_REF_EV_A": ("Double", "eV/A", "largest engine force component at the reference geometry"),
        "ROUNDTRIP_CM": ("Double", "cm^-1", "our diagonalisation of the .hess against ORCA's own frequencies"),
        "HESSIAN_MAE": ("Double", "eV/A^2", "element-wise MAE of the Cartesian Hessians (HIP hessian_mae)"),
        "HESSIAN_RMSE": ("Double", "eV/A^2", "element-wise RMSE of the Cartesian Hessians"),
        "ASYMMETRY_MAE": ("Double", "eV/A^2", "mean |H - H^T| of the engine Hessian (HIP asymmetry_mae)"),
        "HESS_REL_FROBENIUS": ("Double", None, "||K_e - K_r||_F / ||K_r||_F, mass-weighted Eckart-projected"),
        "EIGVAL_MAE_ECKART": ("Double", "eV/A^2/amu", "sorted-index MAE of the projected eigenvalues (HIP eigval_mae_eckart)"),
        "EIGVAL1_MAE_ECKART": ("Double", "eV/A^2/amu", "error of the lowest projected eigenvalue"),
        "FREQ_MAE_CM": ("Double", "cm^-1", "sorted-index MAE of the frequencies, all modes"),
        "FREQ_MAX_CM": ("Double", "cm^-1", "largest sorted-index frequency error, all modes"),
        "FREQ_MAE_LOW_CM": ("Double", "cm^-1", "sorted-index MAE over reference modes below LOW_CUTOFF"),
        "FREQ_MAX_LOW_CM": ("Double", "cm^-1", "largest error over the low modes"),
        "N_LOW": ("Integer", None, "reference modes below LOW_CUTOFF"),
        "SOFTENING_SLOPE": ("Double", None, "least-squares s in omega_e = s omega_r through the origin, all modes"),
        "SOFTENING_SLOPE_LOW": ("Double", None, "the same over the low modes"),
        "LOWEST_ENGINE_CM": ("Double", "cm^-1", "lowest engine mode at the reference geometry"),
        "LOWEST_REF_CM": ("Double", "cm^-1", "lowest reference mode"),
        "N_IMAGINARY_ENGINE_AT_REF": ("Integer", None, "imaginary engine modes at the reference geometry"),
        "N_IMAGINARY_REF": ("Integer", None, "imaginary reference modes"),
        "NEG_NUM_AGREE": ("Boolean", None, "the imaginary counts agree (HIP neg_num_agree)"),
        "EIGVEC1_COS_ECKART": ("Double", None, "|v1_e . v1_r| of the lowest modes (HIP)"),
        "EIGVEC2_COS_ECKART": ("Double", None, "|v2_e . v2_r| of the second modes (HIP)"),
        "EIGVEC_OVERLAP_ERROR": ("Double", None, "||abs(L_e^T L_r) - I||_F (HIP eigvec_overlap_error)"),
        "MIN_BLOCK_OVERLAP": ("Double", None, "smallest overlap of a reference mode with the engine modes of its degenerate block"),
        "MIXING": ("Double", None, "||D - diag D||_F / ||D||_F, D = L_r^T K_e L_r"),
        "OMEGA_REF_CM": ("ArrayOfDoubles", "cm^-1", "reference frequencies, ascending"),
        "OMEGA_ENGINE_CM": ("ArrayOfDoubles", "cm^-1", "engine frequencies at the reference geometry, ascending"),
        "OMEGA_ALONG_REF_CM": ("ArrayOfDoubles", "cm^-1", "engine curvature along each reference mode, sign sqrt|D_ii|"),
        "N_NEAR_ZERO_ALONG_REF": ("Integer", None, "reference modes along which the engine curvature is inside +-1 cm^-1 (dropped from the T*S terms, as CREST's vibthr drops them)"),
        "TS_LOW_DELTA": ("Double", "kcal/mol", "sum over low modes of T*S(omega along ref) - T*S(omega_r)"),
        "S_VIB_CURVATURE_DELTA": ("Double", "cal/mol/K", "S_vib from the curvature along the reference modes minus S_vib(omega_r)"),
        "ZPE_CURVATURE_DELTA": ("Double", "kcal/mol", "ZPE from the curvature along the reference modes minus ZPE(omega_r)"),
        "D_E_REL": ("Double", "kcal/mol", "own-geometry (E_el - E_el,ref basin): engine minus reference"),
        "D_ZPE": ("Double", "kcal/mol", "own-geometry ZPE: engine minus reference"),
        "D_H_THERMAL": ("Double", "kcal/mol", "own-geometry H(T)-H(0): engine minus reference"),
        "D_S_VIB": ("Double", "cal/mol/K", "own-geometry S_vib: engine minus reference"),
        "D_S_ROT": ("Double", "cal/mol/K", "own-geometry S_rot: engine minus reference"),
        "D_G_I_REL": ("Double", "kcal/mol", "own-geometry (G_i - G_ref): engine minus reference"),
        "D_POPULATION": ("Double", None, "own-geometry population: engine minus reference"),
    },
    "Ensemble": {
        "D_S_REF": ("Double", "cal/mol/K", "S_msRRHO of the reference basin: engine minus reference"),
        "D_S_CONF_PRIME": ("Double", "cal/mol/K", "engine minus reference"),
        "D_DS_BAR": ("Double", "cal/mol/K", "engine minus reference"),
        "D_H_CONF": ("Double", "kcal/mol", "engine minus reference"),
        "D_CP_CONF": ("Double", "cal/mol/K", "engine minus reference"),
        "D_S_ABS": ("Double", "cal/mol/K", "engine minus reference (the model error of level_compare)"),
        "D_G_TOTAL_REL": ("Double", "kcal/mol", "(G_total - G_ref) engine minus reference"),
        "MAX_D_POPULATION": ("Double", None, "largest |population difference| over the compared basins"),
        "MEAN_HESSIAN_MAE": ("Double", "eV/A^2", "mean of HESSIAN_MAE over the compared basins"),
        "MEAN_FREQ_MAE_CM": ("Double", "cm^-1", "mean of FREQ_MAE_CM over the compared basins"),
        "MEAN_FREQ_MAE_LOW_CM": ("Double", "cm^-1", "mean of FREQ_MAE_LOW_CM over the compared basins"),
        "MEAN_SOFTENING_SLOPE": ("Double", None, "mean of SOFTENING_SLOPE over the compared basins"),
    },
}


# ====================================================================== the arithmetic
def projected(hessian_eV_A2, masses, positions):
    """K = P M^-1/2 H M^-1/2 P, its vibrational eigenpairs (ascending) and frequencies.
    The projector and the rigid-mode identification are `hessian.project_and_diagonalise`'s
    (overlap with the rigid subspace, never 'the six smallest')."""
    m3 = np.repeat(np.asarray(masses, dtype=float), 3)
    hm = np.asarray(hessian_eV_A2, dtype=float) / np.sqrt(np.outer(m3, m3))
    v, _s, rank = hessian_mod.rigid_body_vectors(masses, positions)
    p = np.eye(len(m3)) - v @ v.T
    k = p @ hm @ p
    k = 0.5 * (k + k.T)
    lam, vec = np.linalg.eigh(k)
    keep = np.linalg.norm(v.T @ vec, axis=0) ** 2 <= 0.5
    lam, vec = lam[keep], vec[:, keep]
    order = np.argsort(lam)
    lam, vec = lam[order], vec[:, order]
    return dict(K=k, lam=lam, vec=vec, freq=hessian_mod.eigenvalues_to_cm_inv(lam), rank=rank)


def _ts_per_mode(freqs_cm, masses, positions, preset, temperature_K):
    """T*S (kcal/mol), S (cal/mol/K) and ZPE (kcal/mol) contribution of each frequency
    in the given order, under the preset and CURVATURE_POLICY (one call per mode: the
    per-mode terms are separable, only the rotor cap is molecular)."""
    ts, s, zpe, n_dropped = [], [], [], 0
    for f in freqs_cm:
        if abs(float(f)) < thermo.VIBTHR_CM:
            # the engine's curvature along this reference mode crosses zero: CREST's
            # vibthr drops such a mode; it is counted here (N_NEAR_ZERO_ALONG_REF) and
            # contributes nothing, so the comparison is reported rather than refused
            ts.append(0.0); s.append(0.0); zpe.append(0.0); n_dropped += 1
            continue
        r = thermo.msrrho([float(f)], masses, positions, preset=preset,
                          temperature_K=temperature_K, imaginary_policy=CURVATURE_POLICY)
        ts.append(r["modes"][0]["TS_kcal"])
        s.append(r["S_vib_kcal_per_K"] * 1000.0)
        zpe.append(r["ZPE_kcal"])
    return np.array(ts), np.array(s), np.array(zpe), n_dropped


def compare_hessians(h_engine, h_ref, masses, positions, preset="crest",
                     temperature_K=thermo.T_REF, low_cm=LOW_CM, gap_cm=GAP_CM,
                     asymmetry_engine=None):
    """Every metric of the four families for two Cartesian Hessians (eV/A^2) at one
    geometry. `asymmetry_engine` is mean |H - H^T| of the engine matrix as produced
    (the stored matrix is symmetrised); None -> computed from `h_engine`."""
    he = np.asarray(h_engine, dtype=float)
    hr = np.asarray(h_ref, dtype=float)
    if he.shape != hr.shape:
        raise ValueError("Hessians of different shape: {} vs {}".format(he.shape, hr.shape))
    d = he - hr
    out = dict(HESSIAN_MAE=float(np.abs(d).mean()), HESSIAN_RMSE=float(np.sqrt((d ** 2).mean())),
               ASYMMETRY_MAE=float(asymmetry_engine if asymmetry_engine is not None
                                   else np.abs(he - he.T).mean()))
    pe, pr = projected(he, masses, positions), projected(hr, masses, positions)
    if pe["lam"].size != pr["lam"].size:
        raise ValueError("different vibrational mode counts at one geometry: {} vs {}".format(
            pe["lam"].size, pr["lam"].size))
    out["HESS_REL_FROBENIUS"] = float(np.linalg.norm(pe["K"] - pr["K"]) / np.linalg.norm(pr["K"]))

    fe, fr = pe["freq"], pr["freq"]
    dl = np.abs(pe["lam"] - pr["lam"])
    df = fe - fr
    low = fr < low_cm
    out.update(EIGVAL_MAE_ECKART=float(dl.mean()), EIGVAL1_MAE_ECKART=float(dl[0]),
               FREQ_MAE_CM=float(np.abs(df).mean()), FREQ_MAX_CM=float(np.abs(df).max()),
               N_LOW=int(low.sum()),
               FREQ_MAE_LOW_CM=float(np.abs(df[low]).mean()) if low.any() else 0.0,
               FREQ_MAX_LOW_CM=float(np.abs(df[low]).max()) if low.any() else 0.0,
               SOFTENING_SLOPE=float((fe * fr).sum() / (fr * fr).sum()),
               SOFTENING_SLOPE_LOW=(float((fe[low] * fr[low]).sum() / (fr[low] ** 2).sum())
                                    if low.any() else 1.0),
               LOWEST_ENGINE_CM=float(fe[0]), LOWEST_REF_CM=float(fr[0]),
               N_IMAGINARY_ENGINE_AT_REF=int((fe < 0).sum()), N_IMAGINARY_REF=int((fr < 0).sum()))
    out["NEG_NUM_AGREE"] = bool(out["N_IMAGINARY_ENGINE_AT_REF"] == out["N_IMAGINARY_REF"])

    le, lr = pe["vec"], pr["vec"]
    dmat = lr.T @ pe["K"] @ lr
    diag = np.diag(dmat)
    out["OMEGA_ALONG_REF_CM"] = [float(x) for x in hessian_mod.eigenvalues_to_cm_inv(diag)]
    off = dmat - np.diag(diag)
    out["MIXING"] = float(np.linalg.norm(off) / np.linalg.norm(dmat))
    o = mode_match.overlap_matrix(lr, le)                 # rows: reference modes
    blocks = mode_match.degenerate_blocks(fr, gap_cm)
    block_ov = np.zeros(len(fr))
    for blk in blocks:
        for i in blk:
            block_ov[i] = o[i, blk].sum()                  # engine modes of the same indices
    m = np.abs(le.T @ lr)
    out.update(EIGVEC1_COS_ECKART=float(abs(le[:, 0] @ lr[:, 0])),
               EIGVEC2_COS_ECKART=float(abs(le[:, 1] @ lr[:, 1])) if lr.shape[1] > 1 else 1.0,
               EIGVEC_OVERLAP_ERROR=float(np.linalg.norm(m - np.eye(m.shape[0]))),
               MIN_BLOCK_OVERLAP=float(block_ov.min()),
               OMEGA_REF_CM=[float(x) for x in fr], OMEGA_ENGINE_CM=[float(x) for x in fe])

    ts_r, s_r, zpe_r, _n0 = _ts_per_mode(fr, masses, positions, preset, temperature_K)
    ts_a, s_a, zpe_a, n_zero = _ts_per_mode(out["OMEGA_ALONG_REF_CM"], masses, positions, preset, temperature_K)
    out.update(N_NEAR_ZERO_ALONG_REF=int(n_zero),
               TS_LOW_DELTA=float((ts_a[low] - ts_r[low]).sum()) if low.any() else 0.0,
               S_VIB_CURVATURE_DELTA=float((s_a - s_r).sum()),
               ZPE_CURVATURE_DELTA=float((zpe_a - zpe_r).sum()),
               TS_ALONG_REF_KCAL=[float(x) for x in ts_a], TS_REF_KCAL=[float(x) for x in ts_r],
               BLOCK_OVERLAP=[float(x) for x in block_ov])
    return out


# ====================================================================== engine at x_r
def engine_hessian_at(molecule, basin, symbols, positions, level, engine_name=None):
    """The engine's Hessian (eV/A^2) and forces (eV/A) at `positions`, from the engine
    files `mace/basinNN/{hessian,forces}_at_<level>.npy` when present, computed and stored
    otherwise. Returns (hessian, forces, asymmetry_max, computed_now)."""
    from ase import Atoms
    bdir = layout.mace_basin_dir(molecule, basin)
    hp = bdir / "hessian_at_{}.npy".format(level)
    fp = bdir / "forces_at_{}.npy".format(level)
    if hp.is_file() and fp.is_file():
        return np.load(hp), np.load(fp), None, False
    calc, _name, _prov = engine.calculator(name=engine_name)
    atoms = Atoms(symbols=symbols, positions=np.asarray(positions, dtype=float))
    h, asym = hessian_mod.hessian(atoms, calc, mode="analytic")
    atoms.calc = calc
    forces = np.asarray(atoms.get_forces(), dtype=float)
    bdir.mkdir(parents=True, exist_ok=True)
    np.save(hp, h)
    np.save(fp, forces)
    return h, forces, float(asym), True


# ====================================================================== the Calculation
def run_calculation(molecule, reference_level=engine.REFERENCE_LEVEL, engine_level=None,
                    qm9_index=None, preset="crest", temperature_K=None, basins=None,
                    engine_name=None, low_cm=LOW_CM, gap_cm=GAP_CM):
    """Compare the engine Hessian at the reference geometry with the reference Hessian
    for every MACE basin the merge map marks `kept`; set the two levels' own-geometry
    thermochemistry side by side; write `levels/hessian_compare.{out,toml}`."""
    from ase.data import atomic_masses, atomic_numbers
    molecule = Path(molecule)
    doc_a = prop.load(layout.records_dir(molecule) / "branchA.toml")
    info_a = doc_a.get("Calculation_Info") or {}
    T = float(temperature_K if temperature_K is not None else info_a.get("TEMPERATURE", thermo.T_REF))
    qid = qm9_index if qm9_index is not None else info_a.get("QM9_INDEX")
    engine_level = engine_level or (engine.level_name(info_a.get("ENGINE")) if info_a.get("ENGINE") else None)
    if not engine_level:
        raise ValueError("no engine level: branchA.toml names no ENGINE and none was given")
    ref_dir = layout.level_dir(molecule, reference_level)
    mm_path = ref_dir / "merge_map.dat"
    if not mm_path.is_file():
        raise FileNotFoundError("no merge map at {}: run the reference level first".format(mm_path))
    merge = dat.read_table(mm_path)
    kept = [r for r in merge if r["status"] == "kept"]
    if basins is not None:
        kept = [r for r in kept if int(r["mace_basin"]) in {int(b) for b in basins}]
    if not kept:
        raise ValueError("no kept basin to compare in {}".format(mm_path))
    n_merged = sum(1 for r in merge if r["status"] == "merged")
    n_saddle = sum(1 for r in merge if r["status"] == "saddle")

    # the two levels' own thermochemistry, for the side-by-side columns
    thermo_e = _thermo_rows(layout.level_dir(molecule, engine_level) / "thermo_msrrho.toml")
    thermo_r = _thermo_rows(ref_dir / "thermo_msrrho.toml")

    rows, computed = [], 0
    for r in kept:
        mb, rb = int(r["mace_basin"]), int(r["reference_basin"])
        parsed = orca.parse_hess(layout.orca_level_dir(molecule, reference_level, mb) / "job.hess")
        rt = orca.verify_hess_frequencies(parsed)
        symbols = list(parsed["symbols"])
        positions = np.asarray(parsed["positions_bohr"], dtype=float) / orca.BOHR_PER_ANGSTROM
        masses = [float(atomic_masses[atomic_numbers[s]]) for s in symbols]
        h_ref = orca.hessian_to_ev_per_angstrom2(parsed["hessian_eh_bohr2"])
        h_eng, forces, asym, now = engine_hessian_at(molecule, mb, symbols, positions, reference_level, engine_name)
        computed += int(now)
        c = compare_hessians(h_eng, h_ref, masses, positions, preset, T, low_cm, gap_cm,
                             asymmetry_engine=asym)
        row = {"MACE_BASIN": mb, "REF_BASIN": rb,
               "MAX_FORCE_ENGINE_AT_REF_EV_A": float(np.abs(forces).max()),
               "ROUNDTRIP_CM": float(rt["max_deviation_cm_inv"])}
        row.update({k: c[k] for k in SCHEMA["Basin"] if k in c})
        row.update(_thermo_deltas(thermo_e, thermo_r, mb, rb))
        row["_detail"] = c
        rows.append(row)

    ens = _ensemble_deltas(thermo_e, thermo_r, rows)
    info = {"MOLECULE_DIR": str(molecule), "QM9_INDEX": qid, "ENGINE_LEVEL": engine_level,
            "REFERENCE_LEVEL": reference_level, "GEOMETRY": "reference", "PRESET": preset,
            "TEMPERATURE": T, "LOW_CUTOFF": float(low_cm), "DEGENERACY_GAP": float(gap_cm),
            "N_BASINS_COMPARED": len(rows), "N_MERGED": n_merged, "N_SADDLE": n_saddle,
            "DIPOLE_DERIVATIVES_PRESENT": False, "ENGINE": info_a.get("ENGINE")}
    levels_dir = molecule / layout.LEVELS
    levels_dir.mkdir(parents=True, exist_ok=True)
    blocks = {"Calculation_Info": info,
              "Basin": [{k: v for k, v in r.items() if k in SCHEMA["Basin"]} for r in rows],
              "Ensemble": ens}
    missing = prop.write(levels_dir / "hessian_compare.toml", blocks, SCHEMA,
                         prop.NORMAL_TERMINATION, PROGNAME)
    if missing:
        raise RuntimeError("hessian_compare.toml keys outside the schema: {}".format(missing))
    _write_report(levels_dir / "hessian_compare.out", info, rows, ens)
    return dict(info=info, basins=rows, ensemble=ens, n_computed_now=computed,
                record=levels_dir / "hessian_compare.toml")


def _thermo_rows(path):
    """{basin index: [[Basin]] row} plus the ensemble/result blocks of a thermo_msrrho
    record, or None when the record is absent."""
    path = Path(path)
    if not path.is_file():
        return None
    d = prop.load(path)
    rows = {int(r["INDEX"]): r for r in d.get("Basin", [])}
    ref_idx = int(d["Calculation_Info"]["REFERENCE_BASIN"])
    ref = rows[ref_idx]
    return dict(rows=rows, ref=ref, ensemble=d["Ensemble"], result=d["Result"],
                info=d["Calculation_Info"])


def _thermo_deltas(te, tr, mace_basin, ref_basin):
    """Own-geometry per-basin differences, engine minus reference, when both records
    hold the basin and neither side excluded it."""
    if te is None or tr is None:
        return {}
    e, r = te["rows"].get(mace_basin), tr["rows"].get(ref_basin)
    if e is None or r is None or e.get("EXCLUDED") or r.get("EXCLUDED"):
        return {}
    out = {"D_E_REL": (e["E_EL"] - te["ref"]["E_EL"]) - (r["E_EL"] - tr["ref"]["E_EL"]),
           "D_ZPE": e["ZPE"] - r["ZPE"], "D_H_THERMAL": e["H_THERMAL"] - r["H_THERMAL"],
           "D_S_VIB": e["S_VIB"] - r["S_VIB"],
           "D_G_I_REL": (e["G_I"] - te["ref"]["G_I"]) - (r["G_I"] - tr["ref"]["G_I"]),
           "D_POPULATION": e["POPULATION"] - r["POPULATION"]}
    if "S_ROT" in e and "S_ROT" in r:
        out["D_S_ROT"] = e["S_ROT"] - r["S_ROT"]
    return out


def _ensemble_deltas(te, tr, rows):
    out = {}
    if rows:
        out.update(MEAN_HESSIAN_MAE=float(np.mean([r["HESSIAN_MAE"] for r in rows])),
                   MEAN_FREQ_MAE_CM=float(np.mean([r["FREQ_MAE_CM"] for r in rows])),
                   MEAN_FREQ_MAE_LOW_CM=float(np.mean([r["FREQ_MAE_LOW_CM"] for r in rows])),
                   MEAN_SOFTENING_SLOPE=float(np.mean([r["SOFTENING_SLOPE"] for r in rows])))
    if te is None or tr is None:
        return out
    ee, er = te["ensemble"], tr["ensemble"]
    out.update(D_S_REF=te["ref"]["S_MSRRHO"] - tr["ref"]["S_MSRRHO"],
               D_S_CONF_PRIME=ee["S_CONF_PRIME"] - er["S_CONF_PRIME"],
               D_DS_BAR=ee["DS_BAR"] - er["DS_BAR"], D_H_CONF=ee["H_CONF"] - er["H_CONF"],
               D_CP_CONF=ee["CP_CONF"] - er["CP_CONF"],
               D_S_ABS=te["result"]["S_ABS"] - tr["result"]["S_ABS"],
               D_G_TOTAL_REL=(te["result"]["G_TOTAL"] - te["ref"]["G_I"])
               - (tr["result"]["G_TOTAL"] - tr["ref"]["G_I"]))
    dp = [abs(r["D_POPULATION"]) for r in rows if "D_POPULATION" in r]
    if dp:
        out["MAX_D_POPULATION"] = float(max(dp))
    return out


def _write_report(path, info, rows, ens):
    rep = report.Report("openQHA hessian_compare",
                        "{} Hessian at the {} geometry of every kept basin, four metric families; "
                        "thermochemistry at each level's own geometry".format(info["ENGINE_LEVEL"], info["REFERENCE_LEVEL"]))
    rep.section("conventions")
    for k in ("ENGINE_LEVEL", "REFERENCE_LEVEL", "GEOMETRY", "PRESET", "TEMPERATURE", "LOW_CUTOFF",
              "DEGENERACY_GAP", "N_BASINS_COMPARED", "N_MERGED", "N_SADDLE", "DIPOLE_DERIVATIVES_PRESENT"):
        rep.kv(k, info[k])
    rep.section("family 1-3 per basin: Cartesian, projected, sorted spectrum")
    rep.table(["mace", "ref", "|F| max", "H MAE", "H RMSE", "rel Frob", "eig MAE", "freq MAE", "freq max",
               "low MAE", "low max", "n low", "slope", "slope low", "lowest e", "lowest r", "imag e/r"],
              [[r["MACE_BASIN"], r["REF_BASIN"], "%.4f" % r["MAX_FORCE_ENGINE_AT_REF_EV_A"],
                "%.4f" % r["HESSIAN_MAE"], "%.4f" % r["HESSIAN_RMSE"], "%.4f" % r["HESS_REL_FROBENIUS"],
                "%.5f" % r["EIGVAL_MAE_ECKART"], "%.2f" % r["FREQ_MAE_CM"], "%.2f" % r["FREQ_MAX_CM"],
                "%.2f" % r["FREQ_MAE_LOW_CM"], "%.2f" % r["FREQ_MAX_LOW_CM"], r["N_LOW"],
                "%.4f" % r["SOFTENING_SLOPE"], "%.4f" % r["SOFTENING_SLOPE_LOW"],
                "%.1f" % r["LOWEST_ENGINE_CM"], "%.1f" % r["LOWEST_REF_CM"],
                "%d/%d" % (r["N_IMAGINARY_ENGINE_AT_REF"], r["N_IMAGINARY_REF"])] for r in rows],
              title="H MAE / RMSE in eV/A^2 (HIP hessian_mae convention); eig MAE in eV/A^2/amu; frequencies in cm^-1")
    rep.section("family 4 per basin: the reference eigenbasis")
    rep.table(["mace", "cos v1", "cos v2", "overlap err", "min block", "mixing", "T*S low delta", "S_vib delta", "ZPE delta"],
              [[r["MACE_BASIN"], "%.4f" % r["EIGVEC1_COS_ECKART"], "%.4f" % r["EIGVEC2_COS_ECKART"],
                "%.4f" % r["EIGVEC_OVERLAP_ERROR"], "%.4f" % r["MIN_BLOCK_OVERLAP"], "%.4f" % r["MIXING"],
                "%+.5f" % r["TS_LOW_DELTA"], "%+.4f" % r["S_VIB_CURVATURE_DELTA"], "%+.4f" % r["ZPE_CURVATURE_DELTA"]]
               for r in rows],
              title="D = L_r^T K_e L_r: curvature along each reference mode (PHL's v^T H v with the DFT modes as probes); "
                    "T*S / S_vib / ZPE deltas in kcal/mol, cal/mol/K, kcal/mol")
    for r in rows:
        c = r["_detail"]
        low = [i for i, f in enumerate(c["OMEGA_REF_CM"]) if f < info["LOW_CUTOFF"]]
        if low:
            rep.table(["i", "omega_r", "omega_e (sorted)", "omega along r", "block overlap", "T*S(r)", "T*S(along r)", "delta"],
                      [[i, "%.2f" % c["OMEGA_REF_CM"][i], "%.2f" % c["OMEGA_ENGINE_CM"][i],
                        "%.2f" % c["OMEGA_ALONG_REF_CM"][i], "%.4f" % c["BLOCK_OVERLAP"][i],
                        "%.5f" % c["TS_REF_KCAL"][i], "%.5f" % c["TS_ALONG_REF_KCAL"][i],
                        "%+.5f" % (c["TS_ALONG_REF_KCAL"][i] - c["TS_REF_KCAL"][i])] for i in low],
                      title="MACE basin {}: the low modes (below {:.0f} cm^-1)".format(r["MACE_BASIN"], info["LOW_CUTOFF"]))
    rep.section("thermochemistry at each level's own geometry (engine minus reference)")
    have = [r for r in rows if "D_ZPE" in r]
    if have:
        rep.table(["mace", "ref", "dE_rel", "dZPE", "dH_therm", "dS_vib", "dS_rot", "dG_i,rel", "dp"],
                  [[r["MACE_BASIN"], r["REF_BASIN"], "%+.4f" % r["D_E_REL"], "%+.4f" % r["D_ZPE"],
                    "%+.4f" % r["D_H_THERMAL"], "%+.4f" % r["D_S_VIB"],
                    "%+.4f" % r["D_S_ROT"] if "D_S_ROT" in r else "-",
                    "%+.4f" % r["D_G_I_REL"], "%+.4f" % r["D_POPULATION"]] for r in have],
                  title="kcal/mol and cal/mol/K; read, never recomputed at the reference geometry")
    else:
        rep.text("  one of the two levels has no thermo_msrrho record: no thermochemistry columns")
    for k, v in ens.items():
        rep.kv(k, "%.5f" % v if isinstance(v, float) else v)
    rep.note("Families: (1) element-wise Cartesian MAE/RMSE, HIP eval_horm.py; (2) relative Frobenius error "
             "of the mass-weighted Eckart-projected matrix; (3) sorted-index spectrum (Weyl), softening slope "
             "after Deng et al. 2025; (4) the engine Hessian in the reference eigenbasis, HIP's eigenvector "
             "cosines and overlap error, block overlap for degenerate modes. Dipole derivatives are in the "
             "ORCA .hess but MACE-OFF23 has no charges: not compared. Thermochemistry is compared at each "
             "level's own geometry, never at the reference geometry.")
    rep.write(path, step=STEP)
