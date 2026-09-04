"""包 1 · CREST 支路的汇总 —— **可以在任何时刻跑，包括作业还在跑的时候**.

PRODUCTION. Summary of the CREST branch; safe to run while the jobs are still going.

--------------------------------------------------------------------------------------
为什么这份汇总可以随时跑
--------------------------------------------------------------------------------------
批量驱动的处理顺序是一个**固定种子的随机置换**（配置 `package1.crest_batch.order_seed`）。
因此**已完成的那一批就是 3819 个分子的一个无偏随机样本** ——
不必等全量跑完（实测全量约 20.8 天）才能引用统计量。

本脚本会**明确报出当前样本量与它对应的 95% 区间宽度**，
免得一个 n = 12 的均值被当成 n = 3819 的均值引用。

--------------------------------------------------------------------------------------
它回答的三个问题
--------------------------------------------------------------------------------------
1. **CREST 报的构象数 vs 本仓判据下的盆数** —— 两者差多少（缺陷 54 的批量版）。
2. **CREST 与 ETKDG 谁漏了盆，漏盆值多少 kcal/mol** ——
   这是把两条路并池之后唯一有意义的问题。
3. **成本** —— 每分子实测耗时的分布，以及剩余部分的外推。

用法::

    python scripts/production/s0_package1_crest_summarise.py
产物::
    analysis/package1_crest_summary_<lo>_<hi>.json
    analysis/package1_crest_summary_<lo>_<hi>.png
"""
import argparse
import json
import sys
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
from openqha import S0_ROOT, config, crest_census

CFG = config.load()
P1 = config.package(1, CFG)
T_REF = config.temperature(CFG)
TARGET = float(CFG["thermodynamics"]["target_accuracy_kcal"])


def ci95(x):
    """均值的 95% 区间半宽（正态近似）。**样本量小的时候它就是重点。**"""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float("nan")
    return float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=int, default=1)
    ap.add_argument("--hi", type=int, default=4000)
    args = ap.parse_args()

    root = (S0_ROOT / "analysis" / "package1" / "crest"
            / "{}_{}".format(*tuple(P1["qm9_range"]))
            / "{}_{}".format(args.lo, args.hi))
    mol_dir = root / "mol"
    if not mol_dir.exists():
        raise FileNotFoundError("还没有产物: {}".format(mol_dir))

    ok, failed = [], []
    for p in sorted(mol_dir.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        (failed if p.name.endswith(".FAILED.json") else ok).append(d)

    man = root / "manifest.json"
    n_total = None
    if man.exists():
        n_total = len(json.loads(man.read_text(encoding="utf-8"))
                      .get("processing_order", []))

    print("=" * 104)
    print("包 1 · CREST 支路   汇总   编号 {}-{}".format(args.lo, args.hi))
    print("=" * 104)
    print("已完成 {} 个 / 计划 {} 个 ({:.2%})；失败 {} 个".format(
        len(ok), n_total, len(ok) / max(n_total or 1, 1), len(failed)))
    print("**处理顺序是固定种子 {} 的随机置换，所以这 {} 个是无偏随机样本**".format(
        P1["crest_batch"]["order_seed"], len(ok)))
    if not ok:
        print("没有成功的分子。")
        return

    nc = np.array([r["n_conformers_reported_by_crest"] for r in ok])
    nb = np.array([r["n_basins"] for r in ok])
    co = np.array([r["n_basins_crest_only"] for r in ok])
    eo = np.array([r["n_basins_etkdg_only"] for r in ok])
    bo = np.array([r["n_basins_both"] for r in ok])
    corr = np.array([r["conformational_correction_kcal"] for r in ok])
    w0 = np.array([r["populations"]["weight_of_lowest"] for r in ok])
    sec = np.array([r.get("total_seconds") or np.nan for r in ok], dtype=float)
    sad = np.array([r["n_saddles_rejected"] for r in ok])
    gch = np.array([r["tighten_crest"]["n_graph_changed"] for r in ok])
    steps_c = np.concatenate([r["tighten_crest"]["opt_steps_per_frame"] for r in ok])
    _se = [r["tighten_etkdg"]["opt_steps_per_frame"] for r in ok
           if r.get("tighten_etkdg")]
    steps_e = np.concatenate(_se) if _se else np.array([np.nan])

    # ---- 只用一条路会漏掉多少 ----------------------------------------------------------
    # 从并池后的盆清单里按来源筛出"只用 CREST"与"只用 ETKDG"两个子集, 各自重算修正,
    # 与并池的修正比。**这才是"漏盆值多少钱"的正确算法** —— 不是比盆数。
    only_crest_err, only_etkdg_err = [], []
    for r in ok:
        rel = np.asarray(r["basin_relative_kcal"])
        prov = r["basin_provenance"]
        full = crest_census.conformational_correction_kcal(rel, T_REF)
        for labs, bag in (("crest", only_crest_err), ("etkdg", only_etkdg_err)):
            sel = [i for i, p in enumerate(prov) if labs in p]
            if not sel:
                bag.append(float("nan"))
                continue
            sub = rel[sel] - rel[sel].min()
            bag.append(crest_census.conformational_correction_kcal(sub, T_REF) - full)
    only_crest_err = np.asarray(only_crest_err)
    only_etkdg_err = np.asarray(only_etkdg_err)

    print()
    print("[一] CREST 报的构象数 vs 本仓判据下的盆数")
    print("  CREST 构象数    均值 {:.2f} ± {:.2f}(95%)  中位 {:.0f}  最大 {}".format(
        nc.mean(), ci95(nc), np.median(nc), nc.max()))
    print("  并池后盆数      均值 {:.2f} ± {:.2f}(95%)  中位 {:.0f}  最大 {}".format(
        nb.mean(), ci95(nb), np.median(nb), nb.max()))
    print("  收紧步数        CREST 侧 均值 {:.1f} 最大 {}；ETKDG 侧 均值 {:.1f} 最大 {}".format(
        steps_c.mean(), steps_c.max(), np.nanmean(steps_e), np.nanmax(steps_e)))
    print("                  **步数 = 交来的几何离本势极小点多远**")

    print()
    print("[二] 两条路谁漏了盆，漏盆值多少")
    print("  盆的来源        仅 CREST {:.2f} ± {:.2f}；仅 ETKDG {:.2f} ± {:.2f}；"
          "两者都有 {:.2f}".format(co.mean(), ci95(co), eo.mean(), ci95(eo), bo.mean()))
    print("  有仅-CREST 盆的分子 {:.1%}；有仅-ETKDG 盆的分子 {:.1%}".format(
        (co > 0).mean(), (eo > 0).mean()))
    print("  只用 CREST 的构象修正误差   均值 {:+.4f}  最大 {:+.4f} kcal/mol".format(
        np.nanmean(only_crest_err), np.nanmax(np.abs(only_crest_err))))
    print("  只用 ETKDG 的构象修正误差   均值 {:+.4f}  最大 {:+.4f} kcal/mol".format(
        np.nanmean(only_etkdg_err), np.nanmax(np.abs(only_etkdg_err))))
    print("  超过判据尺度 {} kcal/mol 的分子: 只用 CREST {} 个; 只用 ETKDG {} 个".format(
        TARGET, int((np.abs(only_crest_err) > TARGET).sum()),
        int((np.abs(only_etkdg_err) > TARGET).sum())))

    print()
    print("[三] 盆清单的性质")
    print("  最低盆权重      均值 {:.3f}；权重 < 0.9 的占 {:.1%}".format(
        w0.mean(), (w0 < 0.9).mean()))
    print("  构象修正        均值 {:+.4f} ± {:.4f}  中位 {:+.4f}  最负 {:+.4f}".format(
        corr.mean(), ci95(corr), np.median(corr), corr.min()))
    print("  鞍点踢出        {} 个分子 ({:.1%})，共 {} 个".format(
        int((sad > 0).sum()), (sad > 0).mean(), int(sad.sum())))
    print("  连接矩阵变过    {} 个分子 ({:.1%}) —— 必须逐个查看".format(
        int((gch > 0).sum()), (gch > 0).mean()))

    per_job = float(np.nanmean(sec))
    remain = (n_total or len(ok)) - len(ok)
    par = 4
    print()
    print("[四] 成本")
    print("  单作业实测      均值 {:.0f} s  中位 {:.0f} s  最大 {:.0f} s".format(
        per_job, np.nanmedian(sec), np.nanmax(sec)))
    print("  剩余 {} 个，按 {} 槽并行外推 {:.1f} 小时 = {:.1f} 天".format(
        remain, par, per_job * remain / par / 3600, per_job * remain / par / 86400))

    if failed:
        print()
        print("[五] 失败 {} 个:".format(len(failed)))
        for f in failed[:20]:
            print("   {} {:24s} {}: {}".format(f["qm9_index"], f["smiles"],
                                               f["error_type"], f["error"][:70]))

    payload = dict(
        qm9_range=[args.lo, args.hi], n_done=len(ok), n_planned=n_total,
        n_failed=len(failed), temperature_K=T_REF,
        sample_is_unbiased=True,
        order_seed=int(P1["crest_batch"]["order_seed"]),
        sample_note="处理顺序是固定种子的随机置换，因此已完成的这一批是无偏随机样本；"
                    "均值一律带 95% 区间半宽，样本量小的时候以区间为准",
        crest_conformers=dict(mean=float(nc.mean()), ci95=ci95(nc),
                              median=float(np.median(nc)), max=int(nc.max())),
        pooled_basins=dict(mean=float(nb.mean()), ci95=ci95(nb),
                           median=float(np.median(nb)), max=int(nb.max())),
        tighten_steps=dict(crest_mean=float(steps_c.mean()),
                           crest_max=int(steps_c.max()),
                           etkdg_mean=float(np.nanmean(steps_e)),
                           etkdg_max=float(np.nanmax(steps_e))),
        provenance=dict(crest_only_mean=float(co.mean()), crest_only_ci95=ci95(co),
                        etkdg_only_mean=float(eo.mean()), etkdg_only_ci95=ci95(eo),
                        both_mean=float(bo.mean()),
                        frac_with_crest_only=float((co > 0).mean()),
                        frac_with_etkdg_only=float((eo > 0).mean())),
        correction_error_if_single_route=dict(
            only_crest_mean=float(np.nanmean(only_crest_err)),
            only_crest_max_abs=float(np.nanmax(np.abs(only_crest_err))),
            only_etkdg_mean=float(np.nanmean(only_etkdg_err)),
            only_etkdg_max_abs=float(np.nanmax(np.abs(only_etkdg_err))),
            n_above_target_only_crest=int((np.abs(only_crest_err) > TARGET).sum()),
            n_above_target_only_etkdg=int((np.abs(only_etkdg_err) > TARGET).sum()),
            target_accuracy_kcal=TARGET,
            method="从并池后的盆清单里按来源筛子集重算修正，与并池的修正相减 —— "
                   "**不是**比盆数"),
        populations=dict(weight_of_lowest_mean=float(w0.mean()),
                         frac_below_0p9=float((w0 < 0.9).mean())),
        correction=dict(mean=float(corr.mean()), ci95=ci95(corr),
                        median=float(np.median(corr)), min=float(corr.min())),
        saddles=dict(n_molecules=int((sad > 0).sum()), n_total=int(sad.sum())),
        graph_changed=dict(n_molecules=int((gch > 0).sum())),
        cost=dict(seconds_per_job_mean=per_job,
                  seconds_per_job_median=float(np.nanmedian(sec)),
                  seconds_per_job_max=float(np.nanmax(sec)),
                  n_remaining=int(remain), parallel_slots=par,
                  projected_days=float(per_job * remain / par / 86400)),
        failures=[dict(qm9_index=f["qm9_index"], smiles=f["smiles"],
                       error_type=f["error_type"], error=f["error"])
                  for f in failed])
    out = S0_ROOT / "analysis" / "package1_crest_summary_{}_{}.json".format(
        args.lo, args.hi)
    crest_census.dump_json(payload, out)
    print()
    print("落盘: {}".format(out))

    _plot(nc, nb, co, eo, only_crest_err, only_etkdg_err, args)


def _plot(nc, nb, co, eo, ec, ee, args):
    # **不要 matplotlib.use("Agg")** —— 讲义里两次因此丢图（缺陷 1、缺陷 53）。
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    m = max(nc.max(), nb.max())
    ax[0].scatter(nc, nb, s=28, alpha=0.7)
    ax[0].plot([0, m], [0, m], "k--", lw=1)
    ax[0].set_xlabel("CREST reported conformers")
    ax[0].set_ylabel("basins under repo criteria (pooled)")
    ax[0].set_title("CREST count is not a basin count")
    bins = np.arange(-0.5, max(co.max(), eo.max()) + 1.5)
    ax[1].hist([co, eo], bins=bins, label=["CREST only", "ETKDG only"])
    ax[1].set_xlabel("basins unique to one route")
    ax[1].set_ylabel("molecules")
    ax[1].legend()
    ax[1].set_title("the two routes are complementary")
    ax[2].hist([ec[~np.isnan(ec)], ee[~np.isnan(ee)]], bins=20,
               label=["only CREST", "only ETKDG"])
    ax[2].axvline(TARGET, color="r", ls="--", lw=1)
    ax[2].set_xlabel("error in conformational correction / kcal/mol")
    ax[2].set_ylabel("molecules")
    ax[2].legend()
    ax[2].set_title("what one route alone costs")
    fig.tight_layout()
    png = S0_ROOT / "analysis" / "package1_crest_summary_{}_{}.png".format(
        args.lo, args.hi)
    fig.savefig(png, dpi=140)
    print("落盘: {}".format(png))


if __name__ == "__main__":
    main()
