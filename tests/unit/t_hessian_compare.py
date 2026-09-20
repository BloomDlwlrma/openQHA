"""Ticket 27: `hessian_compare` -- the engine Hessian at the reference geometry.

UNIT. No engine: tests/data/propanal_molecule carries the ORCA `.hess` of the three
reference basins and the MACE Hessian / forces evaluated at each reference geometry
(`mace/basinNN/{hessian,forces}_at_wb97m-d3bj_def2-tzvppd.npy`, MACE-OFF23_medium,
2026-09-17), so the Calculation reads engine files and computes nothing with a potential.

Asserted:
  * synthetic: K_e = K_r gives every error 0, MIXING 0, slope 1, block overlap 1 and
    NEG_NUM_AGREE; K_e = 0.81 K_r gives slope 0.9 and curvature-along-reference 0.9 omega_r.
  * the reference round-trip against ORCA's own frequencies is below 0.5 cm^-1.
  * every kept basin appears once with all four families; the low-mode statistics count
    only reference modes below 300 cm^-1 (N_LOW written); the residual force is in the row.
  * an engine spectrum with an imaginary mode at x_r is reported, not refused.
  * the own-geometry thermochemistry columns come from the two levels' records and a
    merged / saddle basin is absent from [[Basin]] and counted.
  * DIPOLE_DERIVATIVES_PRESENT = false at the MACE level; no dipole block.
  * D_S_ABS equals level_compare's MODEL_ERROR_S to 1e-9.
"""
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))
from openqha.thermochem import hessian as hessian_mod       # noqa: E402
from openqha.thermochem import hessian_compare as hc         # noqa: E402
from openqha.thermochem import msrrho_ensemble as me         # noqa: E402
from openqha.thermochem import reference_level as rl         # noqa: E402
from openqha.qm_interfaces import orca                       # noqa: E402
from openqha.store import dat, layout, property as prop, report   # noqa: E402

SRC = ROOT / "tests" / "data" / "propanal_molecule"
LEVEL = "wb97m-d3bj_def2-tzvppd"
FAIL = []


def check(label, ok, detail=""):
    print("  {:78s} {}".format(label, "ok" if ok else "FAIL " + str(detail)[:200]))
    if not ok:
        FAIL.append(label)


def main():
    from ase.data import atomic_masses, atomic_numbers
    parsed = orca.parse_hess(layout.orca_level_file(SRC, LEVEL, 0, ".hess"))
    pos = np.asarray(parsed["positions_bohr"]) / orca.BOHR_PER_ANGSTROM
    masses = [float(atomic_masses[atomic_numbers[s]]) for s in parsed["symbols"]]
    h_r = orca.hessian_to_ev_per_angstrom2(parsed["hessian_eh_bohr2"])

    # ---- synthetic seams --------------------------------------------------------------
    same = hc.compare_hessians(h_r, h_r, masses, pos)
    check("K_e = K_r: HESSIAN_MAE, rel Frobenius, eigval MAE, freq MAE all 0",
          same["HESSIAN_MAE"] == 0.0 and same["HESS_REL_FROBENIUS"] == 0.0
          and same["EIGVAL_MAE_ECKART"] == 0.0 and same["FREQ_MAE_CM"] < 1e-9)
    check("K_e = K_r: MIXING 0, slope 1, min block overlap 1, cos v1 = 1, overlap error 0, NEG_NUM_AGREE",
          same["MIXING"] < 1e-12 and abs(same["SOFTENING_SLOPE"] - 1.0) < 1e-12
          and abs(same["MIN_BLOCK_OVERLAP"] - 1.0) < 1e-9 and abs(same["EIGVEC1_COS_ECKART"] - 1.0) < 1e-9
          and same["EIGVEC_OVERLAP_ERROR"] < 1e-9 and same["NEG_NUM_AGREE"])
    check("K_e = K_r: T*S / S_vib / ZPE curvature deltas are 0",
          abs(same["TS_LOW_DELTA"]) < 1e-12 and abs(same["S_VIB_CURVATURE_DELTA"]) < 1e-12
          and abs(same["ZPE_CURVATURE_DELTA"]) < 1e-12)
    check("noise floors (ticket 32): the analytic propanal Hessian's rigid block sits at 5-40 cm^-1 (rotations feel the residual "
          "gradient; translations are ~0), read from the UNPROJECTED matrix, the same on both sides",
          5.0 < same["REF_NOISE_FLOOR_CM"] < 40.0 and same["REF_NOISE_FLOOR_CM"] == same["ENGINE_NOISE_FLOOR_CM"],
          (same["REF_NOISE_FLOOR_CM"], same["ENGINE_NOISE_FLOOR_CM"]))
    # a numerical Hessian's noise lands on the rigid block: eps V V^T in mass-weighted
    # coordinates (eps * sqrt(m_i m_j) (V V^T)_ij in eV/A^2) adds eps to every rigid
    # eigenvalue and nothing to the vibrational ones (P V = 0: the projection removes it)
    eps = (300.0 / hessian_mod.CM_INV_PER_SQRT_EV_A2_AMU) ** 2                     # a 300 cm^-1 floor
    m3 = np.repeat(np.asarray(masses, dtype=float), 3)
    v_rig, _s, _r = hessian_mod.rigid_body_vectors(masses, pos)
    noisy = hc.compare_hessians(h_r, h_r + eps * np.sqrt(np.outer(m3, m3)) * (v_rig @ v_rig.T), masses, pos)
    check("eps on the rigid block: REF_NOISE_FLOOR_CM = 300 cm^-1 (engine side unchanged); the vibrational metrics do not see it "
          "(freq MAE %.2e)" % noisy["FREQ_MAE_CM"],
          abs(noisy["REF_NOISE_FLOOR_CM"] - 300.0) < 3.0 and noisy["ENGINE_NOISE_FLOOR_CM"] == same["ENGINE_NOISE_FLOOR_CM"]
          and abs(noisy["FREQ_MAE_CM"]) < 1.0,
          (noisy["REF_NOISE_FLOOR_CM"], noisy["ENGINE_NOISE_FLOOR_CM"], noisy["FREQ_MAE_CM"]))
    soft = hc.compare_hessians(0.81 * h_r, h_r, masses, pos)
    check("K_e = 0.81 K_r: softening slope 0.9 to 1e-9 (all and low modes)",
          abs(soft["SOFTENING_SLOPE"] - 0.9) < 1e-9 and abs(soft["SOFTENING_SLOPE_LOW"] - 0.9) < 1e-9)
    check("K_e = 0.81 K_r: curvature along every reference mode is 0.9 omega_r, no mixing",
          np.allclose(soft["OMEGA_ALONG_REF_CM"], 0.9 * np.asarray(soft["OMEGA_REF_CM"]), atol=1e-6)
          and soft["MIXING"] < 1e-12 and abs(soft["HESS_REL_FROBENIUS"] - 0.19) < 1e-9)
    check("K_e = 0.81 K_r: a softer surface has more entropy (S_vib curvature delta > 0, ZPE delta < 0)",
          soft["S_VIB_CURVATURE_DELTA"] > 0 and soft["ZPE_CURVATURE_DELTA"] < 0)
    # a spectrum with an imaginary mode at x_r is reported, never refused
    pr = hc.projected(h_r, masses, pos)
    lam = pr["lam"].copy(); lam[0] = -abs(lam[0])
    m3 = np.repeat(np.asarray(masses), 3)
    k_bad = pr["vec"] @ np.diag(lam) @ pr["vec"].T
    h_bad = k_bad * np.sqrt(np.outer(m3, m3))
    bad = hc.compare_hessians(h_bad, h_r, masses, pos)
    check("an imaginary engine mode at x_r is reported: N_IMAGINARY_ENGINE_AT_REF = 1, NEG_NUM_AGREE false",
          bad["N_IMAGINARY_ENGINE_AT_REF"] == 1 and bad["NEG_NUM_AGREE"] is False
          and bad["LOWEST_ENGINE_CM"] < 0 and np.isfinite(bad["TS_LOW_DELTA"]))
    # the engine curvature along a reference mode crossing zero (inside +-1 cm^-1) is
    # dropped as CREST's vibthr drops it, counted, and the comparison still returns
    lam0 = pr["lam"].copy(); lam0[0] = 1e-12
    k0 = pr["vec"] @ np.diag(lam0) @ pr["vec"].T
    zero = hc.compare_hessians(k0 * np.sqrt(np.outer(m3, m3)), h_r, masses, pos)
    check("a near-zero curvature along a reference mode: N_NONPOSITIVE_ALONG_REF = 1, deltas finite",
          zero["N_NONPOSITIVE_ALONG_REF"] == 1 and np.isfinite(zero["S_VIB_CURVATURE_DELTA"])
          and zero["S_VIB_CURVATURE_DELTA"] < 0)
    # a NEGATIVE curvature along a reference mode carries no entropy under every preset,
    # grimme2012 (no ithr) included, and the deltas do not jump at ithr: -30 and -70 give
    # the same contribution (0), and the same as the near-zero case (code-review finding)
    lam_n = pr["lam"].copy(); lam_n[0] = -(30.0 / hessian_mod.CM_INV_PER_SQRT_EV_A2_AMU) ** 2
    lam_m = pr["lam"].copy(); lam_m[0] = -(70.0 / hessian_mod.CM_INV_PER_SQRT_EV_A2_AMU) ** 2
    neg30 = hc.compare_hessians(pr["vec"] @ np.diag(lam_n) @ pr["vec"].T * np.sqrt(np.outer(m3, m3)), h_r, masses, pos)
    neg70 = hc.compare_hessians(pr["vec"] @ np.diag(lam_m) @ pr["vec"].T * np.sqrt(np.outer(m3, m3)), h_r, masses, pos)
    check("a negative curvature along a reference mode contributes no entropy; -30 and -70 cm^-1 give the same deltas (no jump at ithr)",
          neg30["N_NONPOSITIVE_ALONG_REF"] == 1 and neg70["N_NONPOSITIVE_ALONG_REF"] == 1
          and abs(neg30["S_VIB_CURVATURE_DELTA"] - neg70["S_VIB_CURVATURE_DELTA"]) < 1e-12
          and abs(neg30["S_VIB_CURVATURE_DELTA"] - zero["S_VIB_CURVATURE_DELTA"]) < 1e-12)
    g12 = hc.compare_hessians(pr["vec"] @ np.diag(lam_m) @ pr["vec"].T * np.sqrt(np.outer(m3, m3)), h_r, masses, pos,
                              preset="grimme2012")
    check("... and under grimme2012 (no ithr) the comparison still returns", g12["N_NONPOSITIVE_ALONG_REF"] == 1)

    # ---- the Calculation on the fixture ------------------------------------------------
    with tempfile.TemporaryDirectory(prefix="hessian_compare_") as tmp:
        mol = Path(tmp) / "dsgdb9nsd_000035"
        shutil.copytree(SRC, mol)
        try:
            hc.run_calculation(mol)
            check("without a merge map the Calculation refuses and names the file", False)
        except FileNotFoundError as exc:
            check("without a merge map the Calculation refuses and names the file", "merge_map" in str(exc))
        me.run_calculation(mol, level="mace-off23_medium", qm9_index="dsgdb9nsd_000035")
        rl.run_calculation(mol, level=LEVEL, basins=[0, 1, 2], qm9_index="dsgdb9nsd_000035")
        out = hc.run_calculation(mol, qm9_index="dsgdb9nsd_000035")
        check("no engine was run: the stored hessian_at / forces_at files were reused",
              out["n_computed_now"] == 0)
        meta = prop.load(mol / "mace" / "basin00" / "hessian_at_{}.meta.toml".format(LEVEL))["Engine_Hessian"]
        check("the engine files carry their geometry, engine and asymmetry (meta.toml); ASYMMETRY_MAX read from it",
              meta["ENGINE"] and len(meta["POSITIONS"]) == 10 and meta["ASYMMETRY_MAX"] < 1e-10
              and abs(next(r for r in out["basins"] if r["MACE_BASIN"] == 0)["ASYMMETRY_MAX"] - meta["ASYMMETRY_MAX"]) < 1e-30)
        # a stored Hessian taken at another geometry is not reused: the reuse rule compares
        # positions to 1e-8 A (code-review finding). Shifting the stored positions makes the
        # Calculation want the engine, which this unit test does not have -> it must refuse
        # loudly rather than compare Hessians at two different points
        import copy as _copy
        shifted = _copy.deepcopy(meta); shifted["POSITIONS"][0][0] += 1e-3
        prop.write(mol / "mace" / "basin00" / "hessian_at_{}.meta.toml".format(LEVEL),
                   {"Engine_Hessian": shifted}, hc.META_SCHEMA, prop.NORMAL_TERMINATION, hc.PROGNAME)
        try:
            hc.run_calculation(mol, qm9_index="dsgdb9nsd_000035", basins=[0], engine_name="no-such-engine")
            reused_stale = True
        except Exception:                                   # noqa: BLE001 -- any engine failure
            reused_stale = False
        check("a stored engine Hessian at a different geometry is NOT reused (the engine is asked for)",
              reused_stale is False)
        prop.write(mol / "mace" / "basin00" / "hessian_at_{}.meta.toml".format(LEVEL),
                   {"Engine_Hessian": meta}, hc.META_SCHEMA, prop.NORMAL_TERMINATION, hc.PROGNAME)
        out = hc.run_calculation(mol, qm9_index="dsgdb9nsd_000035")
        doc = prop.load(layout.thermo_file(mol, "hessian_compare.toml"))
        rows = {int(r["MACE_BASIN"]): r for r in doc["Basin"]}
        merge = dat.read_table(layout.level_file(mol, LEVEL, "merge_map.dat"))
        kept = sorted(int(r["mace_basin"]) for r in merge if r["status"] == "kept")
        check("every kept MACE basin appears exactly once in [[Basin]] (%s)" % kept, sorted(rows) == kept)
        check("the reference round-trip against ORCA's frequencies is below 0.5 cm^-1",
              all(r["ROUNDTRIP_CM"] < 0.5 for r in rows.values()))
        need = ("HESSIAN_MAE", "HESSIAN_RMSE", "ASYMMETRY_MAX", "HESS_REL_FROBENIUS", "EIGVAL_MAE_ECKART",
                "FREQ_MAE_CM", "FREQ_MAX_CM", "FREQ_MAE_LOW_CM", "FREQ_MAX_LOW_CM", "N_LOW", "SOFTENING_SLOPE",
                "LOWEST_ENGINE_CM", "LOWEST_REF_CM", "N_IMAGINARY_ENGINE_AT_REF", "NEG_NUM_AGREE",
                "EIGVEC1_COS_ECKART", "EIGVEC2_COS_ECKART", "EIGVEC_OVERLAP_ERROR", "MIN_BLOCK_OVERLAP",
                "MIXING", "OMEGA_ALONG_REF_CM", "TS_LOW_DELTA", "S_VIB_CURVATURE_DELTA", "ZPE_CURVATURE_DELTA",
                "MAX_FORCE_ENGINE_AT_REF_EV_A")
        check("all four families are in every row", all(all(k in r for k in need) for r in rows.values()),
              [k for k in need if k not in next(iter(rows.values()))])
        r0 = rows[0]
        n_low = sum(1 for f in r0["OMEGA_REF_CM"] if f < 300.0)
        check("N_LOW counts reference modes below 300 cm^-1 only (basin 0: %d)" % n_low, r0["N_LOW"] == n_low)
        print("      basin 0: H MAE %.4f eV/A^2, freq MAE %.2f (low %.2f) cm^-1, slope %.4f, cos v1 %.4f, |F| %.4f eV/A"
              % (r0["HESSIAN_MAE"], r0["FREQ_MAE_CM"], r0["FREQ_MAE_LOW_CM"], r0["SOFTENING_SLOPE"],
                 r0["EIGVEC1_COS_ECKART"], r0["MAX_FORCE_ENGINE_AT_REF_EV_A"]))
        check("propanal basin 0: MACE reproduces the reference curvature (H MAE < 0.1 eV/A^2, freq MAE < 10 cm^-1, cos v1 > 0.95)",
              r0["HESSIAN_MAE"] < 0.1 and r0["FREQ_MAE_CM"] < 10.0 and r0["EIGVEC1_COS_ECKART"] > 0.95)
        check("0 imaginary modes on both sides at x_r; the residual MACE force is recorded and small",
              all(r["N_IMAGINARY_ENGINE_AT_REF"] == 0 and r["NEG_NUM_AGREE"] for r in rows.values())
              and all(0 < r["MAX_FORCE_ENGINE_AT_REF_EV_A"] < 0.2 for r in rows.values()))
        check("own-geometry thermochemistry columns present (D_ZPE, D_S_VIB, D_S_ROT, D_G_I_REL, D_POPULATION)",
              all(all(k in r for k in ("D_ZPE", "D_S_VIB", "D_S_ROT", "D_G_I_REL", "D_POPULATION")) for r in rows.values()))
        te = prop.load(layout.level_file(mol, "mace-off23_medium", "thermo_msrrho.toml"))
        tr = prop.load(layout.level_file(mol, LEVEL, "thermo_msrrho.toml"))
        e0 = next(r for r in te["Basin"] if r["INDEX"] == 0)
        r0t = next(r for r in tr["Basin"] if r["INDEX"] == rows[0]["REF_BASIN"])
        check("D_ZPE of basin 0 is read from the two records (own geometries), not recomputed",
              abs(rows[0]["D_ZPE"] - (e0["ZPE"] - r0t["ZPE"])) < 1e-12)
        ens = doc["Ensemble"]
        cmp = rl.level_compare(mol, qm9_index="dsgdb9nsd_000035")
        check("D_S_ABS equals level_compare's MODEL_ERROR_S to 1e-9",
              abs(ens["D_S_ABS"] - cmp["tiers"]["MODEL_ERROR_S"]) < 1e-9)
        check("DIPOLE_DERIVATIVES_PRESENT = false at the MACE level; no dipole block",
              doc["Calculation_Info"]["DIPOLE_DERIVATIVES_PRESENT"] is False
              and set(doc) == {"Calculation_Status", "Calculation_Info", "Basin", "Ensemble"})
        check("the Report ends with the terminal line",
              report.terminated_normally(layout.thermo_file(mol, "hessian_compare.out"), "hessian_compare"))

        # a merged basin is absent from [[Basin]] and counted: edit the merge map
        mm = layout.level_file(mol, LEVEL, "merge_map.dat")
        rows_mm = dat.read_table(mm)
        for r in rows_mm:
            if int(r["mace_basin"]) == 2:
                r["status"], r["reference_basin"] = "merged", rows_mm[1]["reference_basin"]
        dat.write_table(mm, rows_mm)
        out2 = hc.run_calculation(mol, qm9_index="dsgdb9nsd_000035")
        doc2 = prop.load(layout.thermo_file(mol, "hessian_compare.toml"))
        check("a MACE basin marked merged is absent from [[Basin]] and counted in N_MERGED",
              sorted(int(r["MACE_BASIN"]) for r in doc2["Basin"]) == [0, 1]
              and doc2["Calculation_Info"]["N_MERGED"] == 1 and doc2["Calculation_Info"]["N_BASINS_COMPARED"] == 2)
    if FAIL:
        print("FAIL: " + ", ".join(FAIL))
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
