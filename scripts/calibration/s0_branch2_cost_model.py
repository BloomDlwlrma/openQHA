# -*- coding: utf-8 -*-
"""分路 2 的代价模型 —— **由实测探针外推到全部 436 个构象**.

CALIBRATION. Extrapolates measured probes to all conformers to decide what branch 2
can afford. Its product is a budget, not a result.

**为什么需要模型而不是直接跑**：包 1 给出的 436 个构象分布在 8 到 19 个原子之间，
而 `RI-MP2/cc-pVTZ` 的数值 Hessian 要 `6N` 次位移梯度，每次梯度又随基函数数
按接近四次方增长。**两端的代价能差一到两个数量级**，
不先把这条曲线量出来就开跑，等于不知道要跑多久。

**模型形式**（三个参数，两个由探针定，一个由结构定）：

    墙钟(一个构象) = t_opt + 6 * N_原子 * t_梯度(N_基函数)
    t_梯度(N) = t0 * (N / N0) ** p

`p` 由两个不同尺寸的探针**实测**定出。**两点定一个指数，这很薄** ——
所以本脚本把 `p` 与由它外推出的总量**一并报出**，并给出 `p = 3.0 / 3.5 / 4.0`
三档的敏感性，好让读者自己判断外推有多可信。

**基函数数按 cc-pVTZ 的球谐计数**：H = 14，C/N/O/F = 30。
2026-09-02 用探针分子 `OC(C=O)C=O`（C3H4O3）核对：`6*30 + 4*14 = 236`，
与 ORCA 输出的 236 **完全一致**。

用法::

    python scripts/calibration/s0_branch2_cost_model.py --probes analysis/branch2_probe_10.jsonl,analysis/branch2_probe_15.jsonl
"""
import argparse
import json
import sys
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


sys.path.insert(0, str(_repo_root()))

from openqha import S0_ROOT

# cc-pVTZ 球谐基函数数
CC_PVTZ = {"H": 14, "C": 30, "N": 30, "O": 30, "F": 30}
BASINS = S0_ROOT / "analysis" / "package1_crest_1_4000__basins.parquet"
MOLECULES = S0_ROOT / "analysis" / "package1_crest_1_4000__molecules.parquet"


def n_basis_from_smiles(smiles):
    from rdkit import Chem
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    n = 0
    for a in m.GetAtoms():
        s = a.GetSymbol()
        if s not in CC_PVTZ:
            raise ValueError("没有 {} 的 cc-pVTZ 计数".format(s))
        n += CC_PVTZ[s]
    return n, m.GetNumAtoms()


def load_probes(paths):
    out = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            o = r.get("orca", {})
            if not o.get("terminated_normally"):
                out.append(dict(qm9_index=r.get("qm9_index"),
                                n_atoms=r.get("n_atoms"), failed=True,
                                error=o.get("error")))
                continue
            si = r.get("simple_input", "")
            # **层级必须跟着每个点走。** 缺陷 52（2026-09-02）：第一版不记层级，
            # 于是把 10 原子的 RIJK 与 15 原子的 RIJCOSX 混在一起拟合，
            # 而 RIJCOSX 快约 3 倍 —— 标度指数被压成 1.06，总代价因此被低估两倍多。
            # **混层级拟合出来的指数没有意义，必须分组。**
            method = ("RIJCOSX" if "RIJCOSX" in si else
                      ("RIJK" if "RIJK" in si else "未知"))
            out.append(dict(qm9_index=r["qm9_index"], n_atoms=r["n_atoms"],
                            smiles=r.get("smiles"), method=method,
                            n_basis=o.get("n_basis"),
                            n_basis_lines=o.get("n_basis_lines_reported"),
                            wall=r.get("total_wall_seconds"),
                            failed=False))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes",
                    default="analysis/branch2_probe_10.jsonl,"
                            "analysis/branch2_probe_15_cosx.jsonl,"
                            "analysis/branch2_probe_15_rijk.jsonl,"
                            "analysis/branch2_optfreq_shard0.jsonl,"
                            "analysis/branch2_optfreq_shard1.jsonl,"
                            "analysis/branch2_optfreq_shard2.jsonl")
    ap.add_argument("--jobs", type=int, default=4,
                    help="并发作业数（每个作业 4 核，本机 16 核）")
    ap.add_argument("--method", default=None,
                    help="只用这个层级的点拟合：RIJK / RIJCOSX。缺省取点最多的那个")
    ap.add_argument("--out", default="analysis/branch2_cost_model.json")
    args = ap.parse_args()

    import pandas as pd

    probes = load_probes(args.probes.split(","))
    good = [p for p in probes if not p["failed"] and p.get("wall")]
    print("=" * 96)
    print("分路 2 代价模型")
    print("=" * 96)
    for p in probes:
        if p["failed"]:
            print("探针 {:2d} 原子  **失败**  {}".format(
                p.get("n_atoms") or 0, str(p.get("error"))[:70]))
        else:
            print("{:2d} 原子  {}  {:8s}  基函数 {}  墙钟 {:.0f} s".format(
                p["n_atoms"], p["qm9_index"], p["method"], p["n_basis"],
                p["wall"]))

    # **只在同一个层级内部拟合。** 混层级的指数没有意义（缺陷 52）。
    by_method = {}
    for r in good:
        by_method.setdefault(r["method"], []).append(r)
    print()
    print("按层级分组：{}".format(
        "，".join("{} {} 个点".format(k, len(v)) for k, v in by_method.items())))
    method = args.method or max(by_method, key=lambda k: len(by_method[k]))
    good = by_method.get(method, [])
    print("**本次用 {} 的 {} 个点拟合**（其余层级的点不参与）".format(method, len(good)))
    if len(good) < 2:
        sys.exit("\n**{} 层级下不足两个成功的点，无法定标度指数。**\n"
                 "  —— 不给假数字：等这个层级的第二个尺寸跑完再来。".format(method))

    good.sort(key=lambda r: r["n_basis"])
    # 从总墙钟反推「每个位移的等效代价」：`wall / (6N)`。
    # 优化段也算在里面 —— 它随同一标度走，且占比不大（10 原子探针实测 639/1613 = 40%）。
    import math
    import numpy as np
    for r in good:
        r["t_per_disp"] = r["wall"] / (6.0 * r["n_atoms"])
    a = good[0]
    ta = a["t_per_disp"]

    if len(good) >= 3:
        # **三个点以上就做最小二乘**，别再靠两点定指数。
        x = np.log([r["n_basis"] / a["n_basis"] for r in good])
        y = np.log([r["t_per_disp"] / ta for r in good])
        p_fit, c0 = np.polyfit(x, y, 1)
        resid = y - (p_fit * x + c0)
        rms = float(np.sqrt((resid ** 2).mean()))
        fit_note = "最小二乘，{} 个点，对数残差均方根 {:.3f}".format(len(good), rms)
        ta = ta * math.exp(c0)
    else:
        b = good[-1]
        p_fit = (math.log(b["t_per_disp"] / ta)
                 / math.log(b["n_basis"] / a["n_basis"]))
        fit_note = "**两点定一个指数，很薄** —— 下面给三档敏感性"
    print()
    for r in good:
        print("  {:>4d} 基函数  每位移等效 {:7.2f} s   （{} 原子，总墙钟 {:.0f} s）".format(
            r["n_basis"], r["t_per_disp"], r["n_atoms"], r["wall"]))
    print("**实测标度指数 p = {:.2f}**   （{}）".format(p_fit, fit_note))

    basins = pd.read_parquet(BASINS)
    mols = pd.read_parquet(MOLECULES)
    nb, na = zip(*mols.smiles.map(n_basis_from_smiles))
    mols["n_basis"] = nb
    mols["n_atoms"] = na
    tbl = basins.merge(mols[["qm9_index", "smiles", "n_atoms", "n_basis"]],
                       on="qm9_index")

    rows = []
    for p in (3.0, 3.5, 4.0, p_fit):
        tot = 0.0
        for _, r in tbl.iterrows():
            t_grad = ta * (r.n_basis / a["n_basis"]) ** p
            tot += 6.0 * r.n_atoms * t_grad
        rows.append(dict(exponent=float(p), total_seconds=float(tot),
                         total_days_serial=float(tot / 86400.0),
                         total_days_parallel=float(tot / 86400.0 / args.jobs)))
    print()
    print("全部 {} 个构象的总代价（{} 个并发作业 x 4 核）：".format(len(tbl), args.jobs))
    print("{:>10s} {:>16s} {:>14s} {:>16s}".format(
        "指数 p", "总核时(小时)", "串行(天)", "并发(天)"))
    for r in rows:
        tag = "  <- 实测" if abs(r["exponent"] - p_fit) < 1e-9 else ""
        print("{:>10.2f} {:>16.1f} {:>14.1f} {:>16.1f}{}".format(
            r["exponent"], r["total_seconds"] / 3600.0 * 4,
            r["total_days_serial"], r["total_days_parallel"], tag))

    # 按分子尺寸拆开 —— 重尾在哪里要看得见
    tbl["t_est_s"] = [6.0 * r.n_atoms * ta * (r.n_basis / a["n_basis"]) ** p_fit
                      for _, r in tbl.iterrows()]
    by = tbl.groupby("n_atoms").agg(
        n_basins=("basin", "size"), n_basis=("n_basis", "first"),
        hours_each=("t_est_s", lambda s: s.iloc[0] / 3600.0),
        hours_total=("t_est_s", lambda s: s.sum() / 3600.0))
    by["占比"] = (by.hours_total / by.hours_total.sum() * 100).round(1)
    print()
    print("按分子尺寸（实测指数 p = {:.2f}）：".format(p_fit))
    print(by.to_string())

    out = Path(args.out)
    out.write_text(json.dumps(dict(
        generated_by="scripts/calibration/s0_branch2_cost_model.py",
        probes=probes, method_fitted=method, fitted_exponent=float(p_fit),
        anchor=dict(n_basis=a["n_basis"], seconds_per_gradient=ta),
        n_basins=int(len(tbl)), jobs=args.jobs,
        projections=rows,
        by_size=json.loads(by.reset_index().to_json(orient="records")),
        note=("模型：墙钟 = 6N * t0 * (N基/N0)^p。"
              "**只在同一层级内部拟合** —— 混 RIJK 与 RIJCOSX 会把指数压平（缺陷 52）。"
              "**两点定一个指数很薄**，故同时给出 p=3.0/3.5/4.0 的敏感性。"
              "基函数数按 cc-pVTZ 球谐计数 H=14、C/N/O/F=30，"
              "已用探针分子 OC(C=O)C=O 核对为 236，与 ORCA 输出一致。")),
        indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    print("落盘:", out)


if __name__ == "__main__":
    main()
