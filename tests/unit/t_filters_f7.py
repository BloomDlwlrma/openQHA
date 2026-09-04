"""门 F7 —— QM9 官方"未表征"名单的单元测试。

判据（逐条，失败即 AssertionError）：

1. 名单解析出**恰好 3054 条**，且文件的 sha256 与写死的值一致；
2. `dsgdb9nsd_003838` 在名单里，且名单记录的 B3LYP 几何 SMILES 含点号（两个片段）；
3. 该分子过 F7 被判出局，理由里同时出现两个 SMILES；
4. 一个正常分子（`dsgdb9nsd_000010`，CC#N）过 F7 通过；
5. **启用 F7 却不给编号必须抛 ValueError** —— 静默跳过一道启用了的门是不允许的；
6. F7 排在 GATE_ORDER 最后，因此 F0–F6 的既有归因不受影响。

用法::  python tests/unit/t_filters_f7.py
"""
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
from openqha import filters, qm9_uncharacterized

FAIL = []


def check(name, cond, detail=""):
    mark = "通过" if cond else "**失败**"
    print("  [{}] {}{}".format(mark, name, ("   " + detail) if detail else ""))
    if not cond:
        FAIL.append(name)


def main():
    print("门 F7 —— QM9 官方未表征名单")
    print("-" * 88)

    rows, prov = qm9_uncharacterized.load()
    check("名单条数 = 3054", len(rows) == 3054, "实测 {}".format(len(rows)))
    check("sha256 与写死的值一致", prov["sha256_matches_recorded"],
          prov["sha256"][:16] + "...")
    check("其中几何确实与 SMILES 不符的条数已统计",
          prov["n_deposited_geometry_disagrees_with_smiles"] > 0,
          "{} 条不符 / {} 条只是 Corina 不同".format(
              prov["n_deposited_geometry_disagrees_with_smiles"],
              prov["n_only_corina_disagrees"]))

    e = qm9_uncharacterized.entry("dsgdb9nsd_003838")
    check("dsgdb9nsd_003838 在名单里", e is not None)
    if e is not None:
        check("其 B3LYP 几何解析出的 SMILES 含点号（即多个片段）",
              "." in e["smiles_b3lyp_xyz"], repr(e["smiles_b3lyp_xyz"]))
        check("整数编号与字符串编号查到同一行",
              qm9_uncharacterized.entry(3838) == e)

    gates = ("F0", "F1", "F3", "F4", "F5", "F7")
    ok, gate, why = filters.screen("N=C1N=CON=N1", gates=gates,
                                   identifier="dsgdb9nsd_003838")
    check("003838 被 F7 判出局", (not ok) and gate == "F7", why[:100])
    check("理由里同时给出两个 SMILES",
          ("N=C1N=CON=N1" in why) and ("N#N" in why))

    ok2, gate2, _ = filters.screen("CC#N", gates=gates,
                                   identifier="dsgdb9nsd_000010")
    check("正常分子 dsgdb9nsd_000010 通过 F7", ok2 and gate2 is None)

    try:
        filters.screen("CC#N", gates=("F7",), identifier=None)
        raised = False
    except ValueError:
        raised = True
    check("启用 F7 但不给编号 -> 抛 ValueError（不静默跳过）", raised)

    # ---- 作用范围 f7_scope --------------------------------------------------------
    import copy
    from openqha import config as _cfg
    base = _cfg.load()
    c_all = copy.deepcopy(base); c_all["species_filter"]["f7_scope"] = "all"
    c_geo = copy.deepcopy(base)
    c_geo["species_filter"]["f7_scope"] = "identity_from_geometry_only"

    # 003838 的索引身份来源是 gdb17（relaxed SMILES 是两个片段, 索引退回 GDB17），
    # 所以**两档都该把它剔除**。
    r_all = filters.screen("N=C1N=CON=N1", c_all, gates, identifier="dsgdb9nsd_003838")
    r_geo = filters.screen("N=C1N=CON=N1", c_geo, gates, identifier="dsgdb9nsd_003838")
    check("003838 在 f7_scope=all 下被剔除", r_all[1] == "F7")
    check("003838 在 f7_scope=identity_from_geometry_only 下**也**被剔除",
          r_geo[1] == "F7", "它的身份来源是 gdb17，不是 relaxed")

    # 000080：GDB17 是甘氨酸两性离子，B3LYP 几何是中性甘氨酸，
    # 索引采用的是 relaxed（= 从几何解析的）SMILES，所以对我们是自洽的。
    a = filters.screen("NCC(=O)O", c_all, gates, identifier="dsgdb9nsd_000080")
    g = filters.screen("NCC(=O)O", c_geo, gates, identifier="dsgdb9nsd_000080")
    check("000080 在 f7_scope=all 下被剔除", a[1] == "F7")
    check("000080 在 f7_scope=identity_from_geometry_only 下**通过**", g[0] is True,
          "索引用的是 relaxed SMILES，几何与它一致")

    try:
        bad = copy.deepcopy(base); bad["species_filter"]["f7_scope"] = "whatever"
        filters.screen("NCC(=O)O", bad, gates, identifier="dsgdb9nsd_000080")
        raised2 = False
    except ValueError:
        raised2 = True
    check("f7_scope 取了第三种值 -> 抛 ValueError（不静默按默认处理）", raised2)

    check("F7 排在 GATE_ORDER 最后", filters.GATE_ORDER[-1] == "F7",
          str(filters.GATE_ORDER))
    check("F7 有配置键与说明",
          "F7" in filters.GATE_DESCRIPTION and "F7" in filters._CONFIG_KEY)

    print("-" * 88)
    if FAIL:
        print("**{} 项失败**: {}".format(len(FAIL), ", ".join(FAIL)))
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
