"""把逐分子的 parquet 拼成全局表 —— 集群上由 `--dependency` 挂在计算作业之后跑.

PRODUCTION. Concatenates the per-molecule parquet files into the global basin table.

每个分子的 `mol/<编号>.parquet` 都是**同构的"一行一个盆"表**，
所以汇总就是一次 `concat`，**不需要重算任何东西**。

产物::

    <结果根>/<段>/collected/basins.parquet      一行一个盆（全部分子）
    <结果根>/<段>/collected/molecules.parquet   一行一个分子（去重后的分子级列）
    <结果根>/<段>/collected/failures.parquet    一行一个失败
    <结果根>/<段>/collected/summary.log         总览（人读）

用法::

    python scripts/production/s0_package1_collect.py --root <结果根>/result-s0tf_1_16000/1_4000
    python scripts/production/s0_package1_collect.py --root ... --stage stage2-mace
"""
import argparse
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
from openqha import S0_ROOT, record, report

#: 分子级的列（其余都是盆级的）。用于 drop_duplicates 拆出分子表。
BASIN_COLS = ("basin", "energy_eV", "relative_kcal", "provenance",
              "boltzmann_weight", "n_imaginary", "n_rigid_modes_removed",
              "lowest_frequency_cm_inv", "hessian_asymmetry_eV_A2",
              "frequencies_cm_inv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="结果根下的段目录，例如 .../result-s0tf_1_16000/1_4000")
    ap.add_argument("--stage", default="", help="子目录名（分阶段时给）")
    ap.add_argument("--out", default="", help="输出目录，默认 <root>/<stage>/collected")
    args = ap.parse_args()

    import pandas as pd

    root = Path(args.root)
    if not root.is_absolute():
        root = S0_ROOT / root
    if args.stage:
        root = root / args.stage
    mol_dir = root / "mol"
    if not mol_dir.exists():
        raise SystemExit("没有 mol/ 目录: {}".format(mol_dir))
    out = Path(args.out) if args.out else (root / "collected")
    out.mkdir(parents=True, exist_ok=True)

    good = sorted(p for p in mol_dir.glob("*.parquet") if ".FAILED" not in p.name)
    bad = sorted(mol_dir.glob("*.FAILED.parquet"))

    print("=" * 96)
    print("汇总逐分子 parquet -> 全局表")
    print("=" * 96)
    print("源   {}".format(mol_dir))
    print("成功 {} 份, 失败 {} 份".format(len(good), len(bad)))
    if not good:
        raise SystemExit("没有可汇总的产物。")

    # ---- 拼接（分块，避免一次性吃掉太多内存）------------------------------------------
    chunks, step = [], 2000
    for i in range(0, len(good), step):
        chunks.append(pd.concat([pd.read_parquet(p) for p in good[i:i + step]],
                                ignore_index=True))
        print("   已读 {}/{} 份".format(min(i + step, len(good)), len(good)), flush=True)
    basins = pd.concat(chunks, ignore_index=True)
    del chunks

    mols = basins.drop(columns=[c for c in BASIN_COLS if c in basins.columns]) \
                 .drop_duplicates(subset="qm9_index").reset_index(drop=True)

    basins.to_parquet(out / "basins.parquet", index=False)
    mols.to_parquet(out / "molecules.parquet", index=False)
    print("   写出 basins.parquet   {} 行 x {} 列".format(*basins.shape))
    print("   写出 molecules.parquet {} 行 x {} 列".format(*mols.shape))

    fails = None
    if bad:
        fails = pd.concat([pd.read_parquet(p) for p in bad], ignore_index=True)
        fails.to_parquet(out / "failures.parquet", index=False)
        print("   写出 failures.parquet {} 行".format(len(fails)))

    # ---- 总览 .log ----------------------------------------------------------------------
    r = report.Report("包 1 · CREST 支路   全局汇总",
                      subtitle=str(root))
    r.section("规模")
    r.kv("molecules", len(mols))
    r.kv("basins", len(basins))
    r.kv("failures", len(fails) if fails is not None else 0)
    r.section("表")
    rows = [["basins.parquet", len(basins), "一行一个盆（含每个盆的全部频率）"],
            ["molecules.parquet", len(mols), "一行一个分子"]]
    if fails is not None:
        rows.append(["failures.parquet", len(fails), "一行一个失败的分子"])
    r.table(["文件", "行数", "内容"], rows)

    r.section("分布")
    stats = []
    for k, unit in (("n_basins", ""), ("n_conformers_crest", ""),
                    ("n_basins_crest_only", ""), ("n_basins_etkdg_only", ""),
                    ("n_saddles_rejected", ""),
                    ("conformational_correction_kcal", "kcal/mol"),
                    ("weight_of_lowest", ""), ("crest_seconds", "s"),
                    ("analysis_seconds", "s"), ("total_seconds", "s"),
                    ("tighten_steps_max", "")):
        if k not in mols.columns:
            continue
        v = pd.to_numeric(mols[k], errors="coerce").dropna().to_numpy()
        if not len(v):
            continue
        stats.append([k, unit or "-", len(v), "{:.4f}".format(v.mean()),
                      "{:.4f}".format(np.median(v)), "{:.4f}".format(v.min()),
                      "{:.4f}".format(v.max())])
    if stats:
        r.table(["量", "单位", "n", "均值", "中位", "最小", "最大"], stats)

    if "crest_terminated_early" in mols.columns:
        e = pd.to_numeric(mols["crest_terminated_early"], errors="coerce").fillna(0)
        r.section("判据")
        r.verdict("每个分子的 CREST 运行里 terminated EARLY 的次数必须是 0",
                  "非零的分子 {} 个".format(int((e > 0).sum())), bool((e > 0).sum() == 0))
    if "n_imaginary" in basins.columns:
        im = pd.to_numeric(basins["n_imaginary"], errors="coerce").fillna(0)
        r.verdict("进入盆清单的结构必须零虚频",
                  "虚频不为零的盆 {} 个".format(int((im > 0).sum())),
                  bool((im > 0).sum() == 0))
    if "shake_fallback_used" in mols.columns:
        n_fb = int(mols["shake_fallback_used"].notna().sum())
        r.section("SHAKE 回退")
        r.kv("molecules_using_fallback", n_fb,
             note="这些分子在默认 SHAKE 下会 terminated EARLY，用 shake=1 重试后成功")

    r.section("读法")
    r.note("import pandas as pd; df = pd.read_parquet('collected/basins.parquet')")
    r.note("**带精度的是 parquet**（float64 原值）；本 .log 的数字是排版过的，"
           "不要用它做逐位比较。")
    p = r.write(out / "summary.log")
    print()
    print("落盘: {}".format(out))
    print("      {}".format(p.name))


if __name__ == "__main__":
    main()
