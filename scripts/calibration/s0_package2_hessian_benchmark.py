"""包 2 —— Hessian 旁路、频率对标、以及刚转子-谐振子参考自由能.

CALIBRATION. The docstring below settles the classification itself: the free
energies produced here are a reference sample, not stage 0's product (D0-28).

计划 `.mem/plan/plan_stage0-vdos-baseline.md` §4 包 2。做四件事:

1. **7 个物种 x 最低盆 x 1 次有限差分 Hessian**, Eckart 投影在对角化之前;
2. **频率对标**: 与 QM9 原生 B3LYP/6-31G(2df,p) 的 3N-6 个频率逐模式比;
3. **由频率误差换算的自由能误差棒**, 三种算法 (实测直接位移 / 不相关 / 完全相关);
4. **刚转子-谐振子参考自由能**: 四项齐全的 `(G - E_el)`, 以及 11 条边的差值。

> **第 4 件事推翻了计划 §1 前提 4** ("无 L1-min 分支, 不产出竞争性的谐振自由能")。
> 用户 2026-08-27 明确要求"实际产生自由能用于参考"。
> 因此这些数字的身份是**参考量/对比样**, 不是 stage 0 的产品 ——
> stage 0 的产品仍然只有态密度那一条路线。见决定 `D0-28`。

**参考层级尚未裁定** (计划修订 3): `r2SCAN-3c` 还是 `CCSD(T)/cc-pVTZ` 数值频率, 未定。
在裁定之前, 本脚本用 **QM9 自带的 B3LYP/6-31G(2df,p) 频率**做一次**免费的初步对标**。
B3LYP/6-31G(2df,p) 本身不是高层参考 —— 它只是一个已经躺在磁盘上、零成本、
且与 stage 1 完全同源的第二套频率。**它给出的是"频率差多少会让自由能差多少"的实测标度,
不是 MACE-OFF23-SC 的绝对频率误差。** 绝对误差仍然要等高层参考。

用法:
    python scripts/calibration/s0_package2_hessian_benchmark.py [--n-embed 50] [--skip-delta-scan]
产物:
    analysis/package2/<物种>/{basin_census.json, lowest.xyz, hessian.json}
    analysis/frequency_benchmark.json
    analysis/hessian_rrho_free_energy.json
"""
import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

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


sys.path.insert(0, str(_repo_root()))

from openqha import conformers, config, engine, hessian, thermo

T_REF = None   # 由配置给出, 见下
OUT = _repo_root() / "analysis"
PKG2 = OUT / "package2"

# --------------------------------------------------------------------------------------
# 7 个物种. 编号来自 stage 2 生产配置的 11 条边 (C3H6O1N0 {18,35,44,46,48} + C2H5O1N1 {19,36})。
#
# **外对称数与电子基态简并度: 显式声明, 绝不自动推导** (计划 §7 风险 6)。
# 每一条都写出理由; 有争议的那一条 (氧杂环丁烷) 在运行时按实际优化结构判定。
CFG = config.load()
_P2 = config.package(2, CFG)

# 7 个物种、外对称数、电子基态简并度 —— 全部来自本仓自己的配置,
# 不再从 stage 2 的生产配置推导 (决定 D0-41)。缺失即抛, 没有默认值。
SPECIES = {}
for _e in config.edges(CFG):
    for _q in config.edge_species(_e):
        if _q not in SPECIES:
            _s = config.species(_q, CFG)
            SPECIES[_q] = dict(name=_s["name"], zh=_s["zh"], smiles=_s["smiles"],
                               sigma=int(_s["symmetry_number"]), g0=int(_s["electronic_degeneracy"]),
                               sigma_reason=_s["symmetry_reason"],
                               conditional=_s.get("conditional"))

DELTA_SCAN_SPECIES = _P2["delta_scan_species"]
T_REF = config.temperature(CFG)


def load_qm9_row(qid):
    return config.qm9_row(qid, CFG)


def load_edges():
    """边集来自**本仓自己的配置**。原先是从 stage 2 的生产配置逐字解析的;
    2026-08-28 stage 0 独立后抄写一次进 configs/openqha.yaml,
    此后本仓即事实来源, 抄写出处与校验和都在配置里。"""
    return config.edges(CFG)


def edge_species(edge):
    return config.edge_species(edge)


def planarity(positions, ring_idx):
    """环的平面性: 到最佳拟合平面的最大距离 (A). 用于氧杂环丁烷的对称数判定。"""
    p = np.asarray(positions)[list(ring_idx)]
    p = p - p.mean(0)
    _, _, vt = np.linalg.svd(p)
    return float(np.abs(p @ vt[2]).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-embed", type=int, default=conformers.N_EMBED_DEFAULT)
    ap.add_argument("--skip-delta-scan", action="store_true")
    ap.add_argument("--only", default=None, help="只跑一个 QM9 编号, 用于冒烟")
    args = ap.parse_args()

    calc, engine_name, prov = engine.calculator()
    edges = load_edges()
    print("=" * 100)
    print("包 2 —— Hessian 旁路与频率对标   引擎 {}   T = {} K".format(engine_name, T_REF))
    print("=" * 100)
    print("配置 {}".format(CFG["_path"]))
    print("边集 {} 条, 校验和 {}".format(len(edges), config.edge_list_sha256(CFG)[:16]))
    print("物种: {} 个".format(len(SPECIES)))
    print("权重 SHA-256 {}".format(prov["sha256"][:16] + "..."))
    print()

    qids = [args.only] if args.only else list(SPECIES)
    results = {}
    t_all = time.time()

    for qid in qids:
        spec = SPECIES[qid]
        row = load_qm9_row(qid)
        xyz_ref = config.qm9_xyz(qid, CFG)
        d = PKG2 / spec["name"]
        d.mkdir(parents=True, exist_ok=True)

        print("-" * 100)
        print("{}  {}  {}  {}".format(qid, spec["zh"], spec["name"], spec["smiles"]))
        t0 = time.time()

        # ---- 1. 盆普查 (包 1 的模块, 这里只取最低盆) -----------------------------------
        rec, basins, mol = conformers.census(
            spec["smiles"], calc, name=spec["name"], n_embed=args.n_embed,
            fmax=conformers.FMAX_CENSUS_EV_A, reference_xyz=xyz_ref)
        print("   嵌入 {} -> 力场后去重 {} -> 势上优化 {} -> 最终 {} 个盆 (阈值 {} A), "
              "用时 {:.0f} s".format(
                  rec["n_embed_returned"], rec.get("n_after_forcefield_prune", "-"),
                  rec["n_conformers_optimised"], rec["n_basins"],
                  rec["dedup_threshold_A"], time.time() - t0))
        if rec["merge_energy_warnings"]:
            w = rec["merge_energy_warnings"]
            print("   ** 归并一致性报警: {} 对被判为同盆的构象能量差超过 {} kcal/mol "
                  "(最大 {:.3f})".format(len(w), conformers.MERGE_ENERGY_WARN_KCAL,
                                         max(x["energy_difference_kcal"] for x in w)))
        print("   盆的相对能量 (kcal/mol): {}".format(
            " ".join("{:.3f}".format(x) for x in rec["basin_relative_kcal"][:8])))
        print("   1 kT 内 {} 个, 5 kT 内 {} 个, 最低盆玻尔兹曼权重 {:.4f}".format(
            rec["populations"]["n_within_1kT"], rec["populations"]["n_within_5kT"],
            rec["populations"]["weight_of_lowest"]))
        ref = rec.get("reference_geometry", {})
        if ref:
            print("   QM9 原生 B3LYP 几何弛豫后落在第 {} 个盆 (最佳均方根偏差 {:.4f} A)".format(
                ref["nearest_basin_rank_zero_based"],
                min(ref["rms_to_each_basin_A"].values())))
        if rec["n_graph_changed"]:
            print("   ** 警告: {} 个构象在优化中连接矩阵变了".format(rec["n_graph_changed"]))
        conformers.dump_json(rec, d / "basin_census.json")

        # ---- 2. 对称数的条件判定 (氧杂环丁烷) ------------------------------------------
        sigma = spec["sigma"]
        sigma_note = spec["sigma_reason"]
        cond = spec.get("conditional")
        if cond:
            tol = float(cond["planarity_tol_A"])
            ring = [i for i, z in enumerate(basins[0].numbers) if z in (6, 8)]
            flat = planarity(basins[0].positions, ring)
            sigma_note += "  |  实测环平面性 (到最佳拟合平面的最大距离) = {:.4f} A".format(flat)
            if flat > tol:
                sigma = int(cond["symmetry_number_if_puckered"])
                sigma_note += "  -> 折叠 (> {} A), 取 sigma = {}".format(tol, sigma)
            else:
                sigma_note += "  -> 近平面 (<= {} A), 取 sigma = {}".format(tol, sigma)
            print("   {} 对称数条件判定: 平面性 {:.4f} A (阈值 {}) -> sigma = {}".format(
                spec["zh"], flat, tol, sigma))

        # ---- 3. 每个热可及的盆各做一次 Hessian, 各给一份自由能 --------------------------
        # 计划 §4 包 3 验收 3 与 ChemRxiv 2026 的第一误差源都是多构象求和 ——
        # 只做最低盆会漏掉"电子能不是最低、但自由能最低"的构象, 本体系已实测到一例。
        kt = thermo.KB_KCAL * T_REF
        rel = np.asarray(rec["basin_relative_kcal"])
        treat = [i for i in range(len(basins)) if rel[i] <= 5.0 * kt]
        per_basin, saddles = [], []
        for i in treat:
            a = basins[i]
            e_i, fm_i, ok_i, ns_i = conformers.optimise(
                a, calc, fmax=conformers.FMAX_HESSIAN_EV_A, steps=2000)
            if not ok_i:
                raise RuntimeError(
                    "{} 第 {} 个盆没有收敛到 {} eV/A (停在 {:.2e}) —— 残余力会污染曲率, "
                    "拒绝在其上做 Hessian".format(spec["name"], i,
                                                  conformers.FMAX_HESSIAN_EV_A, fm_i))
            t1 = time.time()
            hh, asym_i = hessian.finite_difference_hessian(a, calc, delta=hessian.DELTA_A)
            hr = hessian.project_and_diagonalise(hh, a.get_masses(), a.get_positions())
            hr.update(hessian_asymmetry_eV_A2=asym_i, delta_A=hessian.DELTA_A,
                      seconds=time.time() - t1, energy_eV=e_i, max_force_eV_A=fm_i)
            if hr["n_imaginary"]:
                # 不是极小点 -> 逐出自由能求和, 并原样记录。**不静默丢弃**:
                # 一个被去重当作"盆"的结构其实是鞍点, 这件事本身就是一个发现。
                nu_i = np.asarray(hr["frequencies_cm_inv"])
                saddles.append(dict(
                    basin=i, energy_eV=e_i, max_force_eV_A=fm_i,
                    n_imaginary=hr["n_imaginary"],
                    imaginary_frequencies_cm_inv=[float(x) for x in nu_i[nu_i < 0]],
                    relative_E_kcal=float((e_i - min(rec["basin_energies_eV"]))
                                          * conformers.EV_TO_KCAL),
                    is_reference_basin=bool(
                        ref and ref.get("nearest_basin_rank_zero_based") == i),
                    note="有虚频, 不是极小点 —— 已逐出自由能求和"))
                conformers.write_xyz(a, d / "saddle{}.xyz".format(i),
                                     "{} SADDLE {} on {}  E = {:.8f} eV  "
                                     "imaginary = {}".format(
                                         spec["name"], i, engine_name, e_i,
                                         np.round(nu_i[nu_i < 0], 2).tolist()))
                print("   ** 盆 {} 有 {} 个虚频 ({} cm^-1) —— 是鞍点不是极小点, "
                      "逐出自由能求和{}".format(
                          i, hr["n_imaginary"],
                          " ".join("{:.1f}".format(x) for x in nu_i[nu_i < 0]),
                          "; **它正是参考几何所在的结构**"
                          if ref and ref.get("nearest_basin_rank_zero_based") == i else ""))
                continue
            gi = thermo.g_minus_eel(a.get_masses(), a.get_positions(),
                                    hr["frequencies_cm_inv"], symmetry_number=sigma,
                                    degeneracy=spec["g0"], temperature_K=T_REF, qrrho=True)
            per_basin.append(dict(basin=i, energy_eV=e_i, max_force_eV_A=fm_i,
                                  hessian=hr, thermo=gi))
            conformers.write_xyz(a, d / "basin{}.xyz".format(i),
                                 "{} basin {} on {}  E = {:.8f} eV".format(
                                     spec["name"], i, engine_name, e_i))

        if not per_basin:
            raise RuntimeError("{} 的热可及结构全部是鞍点 —— 没有可用的极小点".format(
                spec["name"]))
        e0 = min(b["energy_eV"] for b in per_basin)
        for b in per_basin:
            b["relative_E_kcal"] = (b["energy_eV"] - e0) * conformers.EV_TO_KCAL
            b["relative_G_kcal"] = (b["relative_E_kcal"]
                                    + b["thermo"]["G_minus_Eel_kcal"]
                                    - per_basin[0]["thermo"]["G_minus_Eel_kcal"])
        print("   热可及盆 {} 个 (5 kT = {:.3f} kcal/mol 内), 逐个做了 Hessian:".format(
            len(per_basin), 5.0 * kt))
        for b in per_basin:
            print("      盆 {}: E_rel {:+7.4f}   (G-E_el) {:+9.4f}   G_rel {:+7.4f} kcal/mol, "
                  "最低频 {:7.2f} cm^-1".format(
                      b["basin"], b["relative_E_kcal"], b["thermo"]["G_minus_Eel_kcal"],
                      b["relative_G_kcal"], min(b["hessian"]["frequencies_cm_inv"])))

        # 多构象求和:  G_tot = -kT ln sum_i exp(-G_i / kT)
        g_rel = np.array([b["relative_G_kcal"] for b in per_basin])
        g_multi = -kt * np.log(np.exp(-g_rel / kt).sum())
        weights = np.exp(-g_rel / kt); weights = weights / weights.sum()
        i_gmin = int(np.argmin(g_rel))
        multi = dict(
            n_basins_summed=len(per_basin),
            relative_G_kcal=[float(x) for x in g_rel],
            weights=[float(x) for x in weights],
            multiconformer_correction_kcal=float(g_multi),
            lowest_by_electronic_energy=int(np.argmin(
                [b["relative_E_kcal"] for b in per_basin])),
            lowest_by_free_energy=i_gmin,
            ranking_disagrees=bool(i_gmin != 0))
        # 近简并盆的告诫: 相隔不到 1 kT 的"两个盆"往往是同一个近自由转子上的两个
        # 浅坑。把它们当作两个独立的谐振子求和会重复计数 —— 正确做法是一维受阻转子
        # (已在 scripts/calibration/s0_lowfreq_and_separable_terms.py 里实现)。
        near = [int(i) for i in range(len(g_rel))
                if i != 0 and abs(g_rel[i] - g_rel[0]) < kt]
        if near:
            multi["near_degenerate_basins"] = near
            multi["caveat"] = (
                "盆 {} 与最低盆的自由能差不到 1 kT ({:.3f} kcal/mol)。这类"
                "近简并极小点通常是同一个近自由转子上的浅坑, 把它们当独立谐振子"
                "求和会**重复计数**; 本条多构象修正因此是一个**诊断量与上界**, "
                "不是可以直接用的结果。正确处理是一维受阻转子。".format(near, kt))
            print("   ** 近简并告诫: 盆 {} 与最低盆相差不到 1 kT —— 多构象修正在这里"
                  "是上界, 不是结果".format(near))
        print("   多构象求和: 相对自由能 {}; 权重 {}; 修正 {:+.4f} kcal/mol{}".format(
            " ".join("{:+.3f}".format(x) for x in g_rel),
            " ".join("{:.3f}".format(x) for x in weights), g_multi,
            "   ** 电子能最低与自由能最低不是同一个盆" if multi["ranking_disagrees"] else ""))

        # ---- 4. 频率对标用**参考几何所在的那个盆** ------------------------------------
        # QM9 的频率是在 QM9 自己的几何上算的; 拿它去比另一个构象的频率, 会把构象差
        # 混进曲率差里。用同一个构象比, 这一项污染就没有了。
        ref_rank = ref.get("nearest_basin_rank_zero_based", 0) if ref else 0
        pick = next((k for k, b in enumerate(per_basin) if b["basin"] == ref_rank), None)
        if pick is None:
            pick = 0
            print("   ** 参考几何所在的结构 (盆 {}) 不可用于对标 (鞍点或不在热可及窗口内), "
                  "退回用盆 {} —— 构象差因此重新混进对标里".format(
                      ref_rank, per_basin[0]["basin"]))
        atoms = basins[per_basin[pick]["basin"]]
        hrec = per_basin[pick]["hessian"]
        g = per_basin[pick]["thermo"]
        e_low = per_basin[pick]["energy_eV"]
        fmax_low = per_basin[pick]["max_force_eV_A"]
        nu = np.asarray(hrec["frequencies_cm_inv"])
        conformers.write_xyz(basins[0], d / "lowest.xyz",
                             "{} lowest-energy basin on {}  E = {:.8f} eV".format(
                                 spec["name"], engine_name, per_basin[0]["energy_eV"]))
        print("   刚体模: 秩 {} -> 扣掉 {} 个; 最大|零本征值| {:.3e}, 最小振动本征值 {:.3e}, "
              "分离比 {:.3e}".format(
                  hrec["rigid_subspace_rank"], hrec["n_rigid_modes_removed"],
                  hrec["max_abs_rigid_eigenvalue"], hrec["min_vibrational_eigenvalue"],
                  hrec["separation_gap_ratio"]))
        print("   对标用第 {} 个盆 (参考几何所在的那个), 频率 {} 个, 最低 {:.2f} cm^-1".format(
            per_basin[pick]["basin"], len(nu), nu.min()))
        print("   A_vib {:+9.4f}   A_rot {:+9.4f}   A_trans {:+9.4f}   A_elec {:+9.4f}"
              "   -> (G - E_el) {:+9.4f} kcal/mol".format(
                  g["vibrational"]["A_vib_kcal"], g["rotational"]["A_rot_kcal"],
                  g["translational"]["value_kcal"], g["electronic"]["A_elec_kcal"],
                  g["G_minus_Eel_kcal"]))

        # ---- 5. 与 QM9 原生 B3LYP 频率的对标 ------------------------------------------
        qm9_nu = np.array(sorted(float(v) for v in row["frequencies"].split()))
        bench = thermo.direct_shift(nu, qm9_nu, T_REF)
        bench.update(
            conformer_used=int(per_basin[pick]["basin"]),
            conformer_is_reference_basin=bool(per_basin[pick]["basin"] == ref_rank),
            pairing="按频率升序逐一配对。两套频率现在取自**同一个构象** (QM9 原生几何"
                    "弛豫后所在的那个盆), 所以构象差已被排除; 但模式指认仍未做重叠检验, "
                    "在模式次序发生交换处逐模式偏差会被高估",
            reference_level="B3LYP/6-31G(2df,p) (QM9 原生)",
            reference_is_high_level=False)
        print("   vs QM9 B3LYP: 平均绝对偏差 {:.2f}, 均方根偏差 {:.2f}, 最大 {:.2f} cm^-1; "
              "有符号均值 {:+.2f}".format(
                  bench["mean_absolute_deviation_cm_inv"],
                  bench["root_mean_square_deviation_cm_inv"],
                  bench["max_absolute_deviation_cm_inv"],
                  bench["signed_mean_deviation_cm_inv"]))
        print("   -> A_vib 之差 (实测, 无相关性假设) = {:+.4f} kcal/mol".format(
            bench["delta_A_vib_kcal"]))

        # ---- 6. 同几何诊断: 在 QM9 几何上再算一次 Hessian ------------------------------
        from ase.io import read as ase_read
        ref_atoms = ase_read(str(xyz_ref))
        ref_atoms.calc = calc
        f_ref = float(np.abs(ref_atoms.get_forces()).max())
        h2, asym2 = hessian.finite_difference_hessian(ref_atoms, calc, delta=hessian.DELTA_A)
        hrec2 = hessian.project_and_diagonalise(h2, ref_atoms.get_masses(),
                                                ref_atoms.get_positions())
        nu2 = np.asarray(hrec2["frequencies_cm_inv"])
        same_geom = None
        if nu2.shape == qm9_nu.shape and hrec2["n_imaginary"] == 0:
            same_geom = thermo.direct_shift(nu2, qm9_nu, T_REF)
        same_geom_rec = dict(
            residual_force_on_qm9_geometry_eV_A=f_ref,
            n_imaginary=hrec2["n_imaginary"],
            frequencies_cm_inv=[float(x) for x in nu2],
            comparison=same_geom,
            caveat="QM9 几何不是 MACE 面上的驻点 (残余力见上), Eckart 投影在非驻点上"
                   "不再严格分离转动 —— 这个诊断量的是**曲率之差**, 不能当作误差棒")
        print("   同几何诊断: QM9 几何上的残余力 {:.4f} eV/A, 虚频 {} 个{}".format(
            f_ref, hrec2["n_imaginary"],
            ", 平均绝对偏差 {:.2f} cm^-1".format(
                same_geom["mean_absolute_deviation_cm_inv"]) if same_geom else ""))

        # ---- 7. 误差棒的两个统计极端 ---------------------------------------------------
        bars = {}
        for tag, sig in (("measured_rmse_vs_qm9_b3lyp",
                          bench["root_mean_square_deviation_cm_inv"]),
                         ("literature_egret1_mae_24.4", 24.4)):
            bars[tag] = thermo.error_bar_from_frequency_error(nu, sig, T_REF)
            bars[tag].pop("per_mode_sensitivity_kcal_per_cm")

        results[qid] = dict(
            qm9_index=qid, **{k: spec[k] for k in ("name", "zh", "smiles", "g0")},
            symmetry_number=sigma, symmetry_number_reason=sigma_note,
            n_atoms=len(atoms), energy_eV=e_low, max_force_eV_A=fmax_low,
            basin_census=dict(n_basins=rec["n_basins"],
                              relative_kcal=rec["basin_relative_kcal"],
                              populations=rec["populations"],
                              merge_energy_warnings=rec["merge_energy_warnings"],
                              reference_geometry=ref),
            per_basin=per_basin, saddles_rejected=saddles, multiconformer=multi,
            energy_lowest_basin=dict(
                energy_eV=per_basin[0]["energy_eV"],
                G_minus_Eel_kcal=per_basin[0]["thermo"]["G_minus_Eel_kcal"],
                G_minus_Eel_qrrho_kcal=per_basin[0]["thermo"]["G_minus_Eel_qrrho_kcal"],
                G_minus_Eel_multiconformer_kcal=(
                    per_basin[0]["thermo"]["G_minus_Eel_kcal"]
                    + multi["multiconformer_correction_kcal"])),
            hessian=hrec, thermo=g, benchmark_vs_qm9=bench,
            same_geometry_diagnostic=same_geom_rec, error_bars=bars,
            seconds=time.time() - t0)
        conformers.dump_json(results[qid], d / "hessian.json")
        print("   小计 {:.0f} s".format(time.time() - t0))

    # ---- 8. 位移收敛性 -----------------------------------------------------------------
    delta_scan = None
    if not args.skip_delta_scan and DELTA_SCAN_SPECIES in results:
        print("-" * 100)
        print("有限差分位移收敛性 (在 {} 上测)".format(SPECIES[DELTA_SCAN_SPECIES]["zh"]))
        from ase.io import read as ase_read
        a = ase_read(str(PKG2 / SPECIES[DELTA_SCAN_SPECIES]["name"] / "lowest.xyz"))
        delta_scan = hessian.delta_convergence(a, calc)
        for r in delta_scan:
            print("   delta = {:.3f} A: 最低频 {:8.2f} cm^-1, 相对 0.01 A 最大偏差 {:.3f}, "
                  "平均 {:.3f} cm^-1, 对称性残差 {:.2e}".format(
                      r["delta_A"], min(r["frequencies_cm_inv"]),
                      r.get("max_deviation_from_0.01A_cm_inv", 0.0),
                      r.get("mean_abs_deviation_from_0.01A_cm_inv", 0.0),
                      r["asymmetry_eV_A2"]))

    # ---- 9. 11 条边的组装 --------------------------------------------------------------
    print("=" * 100)
    print("刚转子-谐振子参考: 11 条边的热力学项  Delta (G - E_el)")
    print("=" * 100)
    print("{:24s} {:>11s} {:>11s} {:>11s} {:>11s} {:>11s} {:>11s}".format(
        "边", "dA_vib", "dA_rot", "dA_trans", "d(G-Eel)", "多构象", "qRRHO"))
    print("-" * 100)
    edge_rows = {}
    for e in edges:
        a, b = edge_species(e)
        if a not in results or b not in results:
            continue
        ta = results[a]["per_basin"][0]["thermo"]
        tb = results[b]["per_basin"][0]["thermo"]
        la, lb = results[a]["energy_lowest_basin"], results[b]["energy_lowest_basin"]
        d_vib = tb["vibrational"]["A_vib_kcal"] - ta["vibrational"]["A_vib_kcal"]
        d_rot = tb["rotational"]["A_rot_kcal"] - ta["rotational"]["A_rot_kcal"]
        d_tr = tb["translational"]["value_kcal"] - ta["translational"]["value_kcal"]
        d_tot = lb["G_minus_Eel_kcal"] - la["G_minus_Eel_kcal"]
        d_multi = (lb["G_minus_Eel_multiconformer_kcal"]
                   - la["G_minus_Eel_multiconformer_kcal"])
        d_q = lb["G_minus_Eel_qrrho_kcal"] - la["G_minus_Eel_qrrho_kcal"]
        d_e = (lb["energy_eV"] - la["energy_eV"]) * conformers.EV_TO_KCAL
        edge_rows[e] = dict(reactant=a, product=b,
                            delta_A_vib_kcal=d_vib, delta_A_rot_kcal=d_rot,
                            delta_A_trans_kcal=d_tr,
                            delta_G_minus_Eel_kcal=d_tot,
                            delta_G_minus_Eel_multiconformer_kcal=d_multi,
                            delta_G_minus_Eel_qrrho_kcal=d_q,
                            delta_E_el_engine_kcal=d_e,
                            n_basins_summed=[results[a]["multiconformer"]["n_basins_summed"],
                                             results[b]["multiconformer"]["n_basins_summed"]],
                            note="**参考量, 不是 stage 0 的产品** —— stage 0 的产品是态密度路线")
        print("{:24s} {:11.4f} {:11.4f} {:11.4f} {:11.4f} {:11.4f} {:11.4f}".format(
            e, d_vib, d_rot, d_tr, d_tot, d_multi, d_q))
    print()
    print("平动项在每条边上都是 {:.2e} kcal/mol —— 同分子式, 逐位相消, 这是 Delta n = 0 "
          "的直接体现".format(max(abs(r["delta_A_trans_kcal"]) for r in edge_rows.values())
                              if edge_rows else 0.0))

    # ---- 10. 误差棒: 边一级 ------------------------------------------------------------
    # 单物种的 A_vib 偏差不是我们要的量 —— 我们要的是它在**差值**里剩下多少。
    print()
    print("=" * 100)
    print("误差棒   频率层级从 MACE-OFF23-SC 换到 B3LYP/6-31G(2df,p) 时, "
          "Delta(G-E_el) 会移动多少")
    print("=" * 100)
    print("{:24s} {:>14s} {:>14s} {:>14s}".format(
        "边", "反应物 dA_vib", "产物 dA_vib", "边上净移动"))
    print("-" * 100)
    shifts = {}
    for e, r in edge_rows.items():
        sa = results[r["reactant"]]["benchmark_vs_qm9"]["delta_A_vib_kcal"]
        sb = results[r["product"]]["benchmark_vs_qm9"]["delta_A_vib_kcal"]
        clean = (results[r["reactant"]]["benchmark_vs_qm9"]["conformer_is_reference_basin"]
                 and results[r["product"]]["benchmark_vs_qm9"]["conformer_is_reference_basin"])
        shifts[e] = dict(reactant_shift_kcal=sa, product_shift_kcal=sb,
                         edge_shift_kcal=sb - sa, same_conformer_both_sides=clean)
        print("{:24s} {:14.4f} {:14.4f} {:14.4f}{}".format(
            e, sa, sb, sb - sa, "" if clean else "   (有一侧构象不匹配, 不可用)"))
    usable = [v["edge_shift_kcal"] for v in shifts.values()
              if v["same_conformer_both_sides"]]
    per_species = [v["benchmark_vs_qm9"] for v in results.values()]
    pooled = dict(
        n_species=len(per_species),
        mean_absolute_deviation_cm_inv=float(np.mean(
            [x["mean_absolute_deviation_cm_inv"] for x in per_species])),
        root_mean_square_deviation_cm_inv=float(np.sqrt(np.mean(
            [x["root_mean_square_deviation_cm_inv"] ** 2 for x in per_species]))),
        signed_mean_deviation_cm_inv=float(np.mean(
            [x["signed_mean_deviation_cm_inv"] for x in per_species])),
        max_absolute_deviation_cm_inv=float(np.max(
            [x["max_absolute_deviation_cm_inv"] for x in per_species])),
        n_edges_usable=len(usable),
        edge_shift_mean_abs_kcal=float(np.mean(np.abs(usable))) if usable else None,
        edge_shift_max_abs_kcal=float(np.max(np.abs(usable))) if usable else None,
        per_edge=shifts)
    print()
    print("七个物种合并: 平均绝对偏差 {:.2f} cm^-1, 均方根偏差 {:.2f} cm^-1, "
          "有符号均值 {:+.2f} cm^-1 (MACE 系统性偏高), 最大 {:.2f} cm^-1".format(
              pooled["mean_absolute_deviation_cm_inv"],
              pooled["root_mean_square_deviation_cm_inv"],
              pooled["signed_mean_deviation_cm_inv"],
              pooled["max_absolute_deviation_cm_inv"]))
    if usable:
        print("可用的 {} 条边上, 边一级净移动 平均 |{:.4f}|, 最大 |{:.4f}| kcal/mol".format(
            len(usable), pooled["edge_shift_mean_abs_kcal"],
            pooled["edge_shift_max_abs_kcal"]))
    print()
    print("**这不是 MACE-OFF23-SC 的频率误差棒。** B3LYP/6-31G(2df,p) 不是高层参考, ")
    print("它自己相对耦合簇也有几十个 cm^-1 的误差。上面的数字量的是")
    print("**「换一个频率层级, 答案会动多少」** —— 一个标度, 不是一个误差。")
    print("真正的误差棒要等参考层级裁定 (r2SCAN-3c 或 CCSD(T)/cc-pVTZ 数值频率) 之后。")

    payload = dict(
        error_bar_scale=pooled,
        generated_by="scripts/calibration/s0_package2_hessian_benchmark.py",
        package="包 2 —— Hessian 旁路与频率对标",
        engine=prov, temperature_K=T_REF,
        config_path=CFG["_path"], edges=edges,
        edges_sha256=config.edge_list_sha256(CFG),
        edges_provenance=CFG.get("edges_provenance"),
        n_species=len(results), species=results,
        delta_convergence=delta_scan,
        edge_thermodynamic_terms=edge_rows,
        reference_level_decision=dict(
            status="未裁定",
            candidates=["r2SCAN-3c", "CCSD(T)/cc-pVTZ 数值频率"],
            interim="QM9 原生 B3LYP/6-31G(2df,p) —— 零成本, 但**不是高层参考**",
            consequence="在高层参考到位之前, 本文件里的误差棒是**标度**而非**误差**"),
        identity_note=("刚转子-谐振子自由能是**参考量/对比样**, 推翻了计划 §1 前提 4, "
                       "依据是用户 2026-08-27 的明确要求; 见决定 D0-28。"
                       "stage 0 的产品仍然只有态密度那一条路线。"),
        total_seconds=time.time() - t_all)

    conformers.dump_json(payload, OUT / "frequency_benchmark.json")
    conformers.dump_json(
        dict(temperature_K=T_REF, engine=prov,
             species={k: dict(name=v["name"], zh=v["zh"],
                              symmetry_number=v["symmetry_number"],
                              symmetry_number_reason=v["symmetry_number_reason"],
                              electronic_degeneracy=v["g0"],
                              n_basins=v["basin_census"]["n_basins"],
                              energy_lowest_basin=v["energy_lowest_basin"],
                              multiconformer=v["multiconformer"],
                              per_basin=[dict(basin=b["basin"],
                                              energy_eV=b["energy_eV"],
                                              relative_E_kcal=b["relative_E_kcal"],
                                              relative_G_kcal=b["relative_G_kcal"],
                                              frequencies_cm_inv=b["hessian"]["frequencies_cm_inv"],
                                              G_minus_Eel_kcal=b["thermo"]["G_minus_Eel_kcal"])
                                         for b in v["per_basin"]]) for k, v in results.items()},
             edges=edge_rows,
             identity_note=payload["identity_note"]),
        OUT / "hessian_rrho_free_energy.json")
    print()
    print("落盘: analysis/frequency_benchmark.json")
    print("      analysis/hessian_rrho_free_energy.json")
    print("      analysis/package2/<物种>/{basin_census.json, lowest.xyz, hessian.json}")
    print("总用时 {:.0f} s".format(time.time() - t_all))


if __name__ == "__main__":
    main()
