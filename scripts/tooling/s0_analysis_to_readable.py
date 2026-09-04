"""把 `analysis/` 里的 JSON 变成**能读的东西** —— `.log` 文本 + parquet 表.

TOOLING. Converts JSON under analysis/ into readable text plus parquet tables.
Produces no science.

用户 2026-08-31：「`analysis/` 里全是 JSON，读不懂」。

**做法（三层，各司其职）**：

| 层 | 产物 | 谁读 |
|---|---|---|
| 存档 | `xxx.json` | 机器；断点标准要求的完整记录。**不删、不改** |
| 报告 | `xxx.log` | **人**。ORCA/CREST 风格：横幅、分节、点线引导、对齐的表、单位在表头 |
| 表 | `xxx__<表名>.parquet` | pandas。**只在有大批同构数字时才生成** |

**`.log` 与 parquet 都是从 JSON 生成的**，所以三者不可能漂移；
JSON 变了重跑一次即可。**JSON 仍是唯一的事实来源。**

逐分子的那两个目录（各 4000 份 JSON）**不逐份转 `.log`** ——
4000 个文本文件同样不可读。它们转成 **parquet 表**：
一行一个分子、一行一个盆、一行一个频率，用 pandas 一句话就能查。

用法::

    python scripts/tooling/s0_analysis_to_readable.py            # 全部
    python scripts/tooling/s0_analysis_to_readable.py --only-logs
    python scripts/tooling/s0_analysis_to_readable.py --only-parquet
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
from openqha import S0_ROOT, report

ANALYSIS = S0_ROOT / "analysis"

#: 逐分子 JSON 的两个集合：**不逐份转 `.log`**，转成 parquet
PER_MOLECULE = (
    ("etkdg", ANALYSIS / "package1" / "1_16000" / "1_4000" / "mol"),
    ("crest", ANALYSIS / "package1" / "crest" / "1_16000" / "1_4000" / "mol"),
)


# ======================================================================================
# 一、通用：每份 JSON 一份 .log
# ======================================================================================
def convert_logs():
    skip_dirs = {p.resolve() for _, p in PER_MOLECULE if p.exists()}
    made = []
    for p in sorted(ANALYSIS.rglob("*.json")):
        if any(d in p.resolve().parents for d in skip_dirs) or p.resolve().parent in skip_dirs:
            continue
        if p.name.endswith(".partial.jsonl"):
            continue
        try:
            out = report.json_to_log(p)
            made.append((out, out.stat().st_size))
        except Exception as exc:                     # 记全, 不吞
            print("   转换失败 {} —— {}: {}".format(p.name, type(exc).__name__, exc))
    return made


# ======================================================================================
# 二、逐分子集合 -> parquet
# ======================================================================================
def _crest_rows(d):
    """一份 CREST 支路记录 -> (分子行, 盆行列表, 频率行列表, 收紧行列表)."""
    q = d["qm9_index"]
    run = d.get("crest_run") or {}
    tc = d.get("tighten_crest") or {}
    te = d.get("tighten_etkdg") or {}
    mol = dict(
        qm9_index=q, smiles=d.get("smiles"),
        n_conformers_crest=d.get("n_conformers_reported_by_crest"),
        n_etkdg_basins_in=d.get("n_etkdg_basins_in"),
        n_pooled_before_hessian=(d.get("pooling") or {}).get("n_global_before_hessian"),
        n_basins=d.get("n_basins"),
        n_basins_crest_only=d.get("n_basins_crest_only"),
        n_basins_etkdg_only=d.get("n_basins_etkdg_only"),
        n_basins_both=d.get("n_basins_both"),
        n_saddles_rejected=d.get("n_saddles_rejected"),
        conformational_correction_kcal=d.get("conformational_correction_kcal"),
        weight_of_lowest=(d.get("populations") or {}).get("weight_of_lowest"),
        n_within_1kT=(d.get("populations") or {}).get("n_within_1kT"),
        crest_seconds=run.get("seconds"),
        total_seconds=d.get("total_seconds"),
        total_engrad_calls=run.get("total_engrad_calls"),
        crest_terminated_normally=run.get("terminated_normally"),
        crest_terminated_early=run.get("n_terminated_early"),
        tighten_steps_crest_max=max(tc.get("opt_steps_per_frame") or [0]),
        tighten_steps_etkdg_max=max(te.get("opt_steps_per_frame") or [0]),
        n_graph_changed=tc.get("n_graph_changed"),
        start_geometry_source=(d.get("start_geometry") or {}).get("source"),
        temperature_K=d.get("temperature_K"))
    basins, freqs = [], []
    prov = d.get("basin_provenance") or []
    hs = d.get("basin_hessian") or []
    for i, (e, rel) in enumerate(zip(d.get("basin_energies_eV", []),
                                     d.get("basin_relative_kcal", []))):
        h = hs[i] if i < len(hs) else {}
        basins.append(dict(qm9_index=q, basin=i, energy_eV=e, relative_kcal=rel,
                           provenance="+".join(prov[i]) if i < len(prov) else None,
                           n_imaginary=h.get("n_imaginary"),
                           lowest_frequency_cm_inv=h.get("lowest_frequency_cm_inv"),
                           n_rigid_modes_removed=h.get("n_rigid_modes_removed"),
                           boltzmann_weight=((d.get("populations") or {})
                                             .get("boltzmann_weights") or [None] * 99)[i]))
        for k, nu in enumerate(h.get("frequencies_cm_inv") or []):
            freqs.append(dict(qm9_index=q, basin=i, mode=k, frequency_cm_inv=nu))
    return mol, basins, freqs


def _etkdg_rows(d):
    q = d.get("qm9_index")
    pop = d.get("populations") or {}
    mol = dict(
        qm9_index=q, smiles=d.get("smiles"),
        n_embed_returned=d.get("n_embed_returned"),
        n_after_forcefield_prune=d.get("n_after_forcefield_prune"),
        n_conformers_optimised=d.get("n_conformers_optimised"),
        n_basins=d.get("n_basins"),
        n_not_converged=d.get("n_not_converged"),
        n_graph_changed=d.get("n_graph_changed"),
        n_merge_warnings=len(d.get("merge_energy_warnings") or []),
        weight_of_lowest=pop.get("weight_of_lowest"),
        n_within_1kT=pop.get("n_within_1kT"), n_within_5kT=pop.get("n_within_5kT"),
        max_residual_force_eV_A=d.get("max_residual_force_eV_A"),
        seconds=d.get("seconds"), n_atoms=d.get("n_atoms"),
        n_heavy_atoms=d.get("n_heavy_atoms"),
        reference_nearest_basin_rank=(d.get("reference_geometry") or {})
        .get("nearest_basin_rank_zero_based"),
        reference_is_own_basin=(d.get("reference_geometry") or {}).get("is_own_basin"))
    basins = []
    w = pop.get("boltzmann_weights") or []
    for i, (e, rel) in enumerate(zip(d.get("basin_energies_eV", []),
                                     d.get("basin_relative_kcal", []))):
        basins.append(dict(qm9_index=q, basin=i, energy_eV=e, relative_kcal=rel,
                           boltzmann_weight=w[i] if i < len(w) else None))
    return mol, basins


def convert_parquet():
    made = []
    for tag, d in PER_MOLECULE:
        if not d.exists():
            continue
        mols, basins, freqs, fails = [], [], [], []
        for p in sorted(d.glob("*.json")):
            obj = json.loads(p.read_text(encoding="utf-8"))
            if p.name.endswith(".FAILED.json"):
                fails.append(dict(qm9_index=obj.get("qm9_index"),
                                  smiles=obj.get("smiles"),
                                  error_type=obj.get("error_type"),
                                  error=obj.get("error")))
                continue
            if tag == "crest":
                m, b, f = _crest_rows(obj)
                freqs += f
            else:
                m, b = _etkdg_rows(obj)
            mols.append(m)
            basins += b
        tables = dict(molecules=mols, basins=basins, failures=fails)
        if freqs:
            tables["frequencies"] = freqs
        stem = ANALYSIS / "package1_{}_1_4000".format(tag)
        made += report.write_parquet(tables, stem)
        _collection_log(tag, stem, mols, basins, freqs, fails)
    return made


def _collection_log(tag, stem, mols, basins, freqs, fails):
    """给每个逐分子集合写一份**总览** `.log` —— 4000 份 JSON 的入口。"""
    import numpy as np
    name = {"etkdg": "包 1 · ETKDG 普查", "crest": "包 1 · CREST 支路"}[tag]
    r = report.Report(name + "   编号 1-4000",
                      subtitle="逐分子记录的总览；明细在同名 parquet 表里")
    r.section("这份文件是什么")
    r.note("逐分子的完整记录是 {} 份 JSON，逐份转成文本同样读不了。"
           "所以明细走 parquet，这里只给总览与入口。".format(len(mols)))
    r.note("读法： import pandas as pd; "
           "df = pd.read_parquet('analysis/{}__molecules.parquet')".format(stem.name))
    r.section("规模")
    r.kv("molecules_done", len(mols))
    r.kv("basins_total", len(basins))
    if freqs:
        r.kv("frequencies_total", len(freqs))
    r.kv("failures", len(fails))
    r.section("表")
    rows = [("{}__molecules.parquet".format(stem.name), len(mols),
             "一行一个分子"),
            ("{}__basins.parquet".format(stem.name), len(basins),
             "一行一个盆（能量、相对能量、玻尔兹曼权重、来源、虚频数）")]
    if freqs:
        rows.append(("{}__frequencies.parquet".format(stem.name), len(freqs),
                     "一行一个简正模"))
    if fails:
        rows.append(("{}__failures.parquet".format(stem.name), len(fails),
                     "一行一个失败的分子（异常类型与消息）"))
    r.table(["文件", "行数", "内容"], rows)

    if mols:
        r.section("分布（当前样本）")
        def col(k):
            v = [m[k] for m in mols if m.get(k) is not None]
            return np.asarray(v, dtype=float) if v else None
        stats = []
        for k in ("n_basins", "conformational_correction_kcal", "weight_of_lowest",
                  "n_saddles_rejected", "total_seconds", "seconds",
                  "n_basins_crest_only", "n_basins_etkdg_only"):
            a = col(k)
            if a is None or not len(a):
                continue
            stats.append([report.strip_unit(k), report.unit_of(k) or "-", len(a),
                          "{:.4f}".format(a.mean()), "{:.4f}".format(np.median(a)),
                          "{:.4f}".format(a.min()), "{:.4f}".format(a.max())])
        if stats:
            r.table(["量", "单位", "n", "均值", "中位", "最小", "最大"], stats)
    if fails:
        r.section("失败清单")
        r.table(["编号", "SMILES", "异常", "消息"],
                [[f["qm9_index"], f["smiles"], f["error_type"],
                  (f["error"] or "")[:48]] for f in fails[:30]])
    p = r.write(stem.with_suffix(".log"))
    print("   总览: {}".format(p.name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-logs", action="store_true")
    ap.add_argument("--only-parquet", action="store_true")
    args = ap.parse_args()

    print("=" * 96)
    print("把 analysis/ 的 JSON 转成可读形态")
    print("=" * 96)
    print("JSON 仍是唯一的事实来源；.log 与 parquet 都是从它生成的，因此不会漂移。")
    print()

    if not args.only_parquet:
        print("[一] 每份 JSON 一份 .log")
        made = convert_logs()
        for p, n in made:
            print("   {:52s} {:8.1f} KB".format(p.name, n / 1024))
        print("   共 {} 份".format(len(made)))

    if not args.only_logs:
        print()
        print("[二] 逐分子集合 -> parquet（4000 份 JSON 不逐份转文本）")
        made = convert_parquet()
        for p, n, cols in made:
            print("   {:52s} {:6d} 行  {} 列".format(p.name, n, len(cols)))

    print()
    print("读法:")
    print("   .log      直接打开；ORCA/CREST 风格")
    print("   .parquet  import pandas as pd; pd.read_parquet(路径)")


if __name__ == "__main__":
    main()
