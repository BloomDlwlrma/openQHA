"""按 hkuhpc / deimos 的规范写 stage 0 的 DLPNO-CCSD(T)/cc-pVTZ 输入.

PRODUCTION. Writes the ORCA inputs the reference calculations are run from.

**为什么要有这个脚本**: 用户的其他工作全部用同一套设置产出, stage 0 必须与之对标,
否则 DLPNO 截断误差、辅助基近似、程序版本三者的系统偏差都无法与那批数据相消.

规范来源 (逐字):
  * `00_QM9_reaction_eng/hkuhpc/sbatch_tianhe/gen_orca_input.sh`
        - 关键字行 `! DLPNO-CCSD(T) <basis> <aux>`; cc-pVTZ 的 aux 是 `cc-pVTZ/JK RIJK cc-pVTZ/C`
        - `%mdci  TCutPairs 1e-6  printlevel 4 end`
        - `%loc   LocMet AHFB  OCC true end`
        - 坐标用 `*xyzfile 0 1 <path>`
  * `00_QM9_reaction_eng/hkuhpc/sbatch_tianhe/core-bind/orca.md` 的集群表:
        deimos = ORCA **5.0.4**, 64 核, 512 GB   |   intel = ORCA 6.1.0, 32 核, 128 GB
    stage 0 用 **deimos / 5.0.4**.
  * 启动前必须 unset 全部 PMI* 与 SLURM* 变量, 并用绝对路径启动 ORCA (天河要求).

**这个脚本只写输入, 不调用 ORCA** —— 本机没有 ORCA.

用法:
    python scripts/production/s0_write_orca_inputs.py            # 从既有 .inp 读回几何
产物:
    docs/orca_inputs/deimos_504/<edge>_<species>.{xyz,inp}
    docs/orca_inputs/deimos_504/run_hint.sh
"""
import json
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


ROOT = _repo_root()
ORCA = ROOT / "docs" / "orca_inputs"
OUT = ORCA / "deimos_504"

EDGE = "C2H5O1N1_19_36"
SPECIES = ["acetamide", "N-methylformamide"]

# ---- 参数分类 (skills 第 1.4 条) --------------------------------------------------------
# 方法与阈值: 文献值/对标约定 —— 改了就与用户其他工作不可比
KEYWORD_LINE = "! DLPNO-CCSD(T) cc-pVTZ cc-pVTZ/JK RIJK cc-pVTZ/C"
MDCI_BLOCK = "%mdci\n   TCutPairs 1e-6\n   printlevel 4\nend"
LOC_BLOCK = "%loc\nLocMet AHFB\nOCC true\nend"
ORCA_VERSION = "5.0.4"          # deimos
# 资源预算: 节点相关, 天河的 worker 会在运行时改写这两行, 生成时的值一律不信
MAXCORE_MB = 3900
NPROCS = 4

TEMPLATE = """{keyword}
%maxcore {maxcore}
%pal nprocs {nprocs} end
{mdci}
{loc}
*xyzfile 0 1 {xyzname}
"""


def read_geometry(name):
    """从既有的 ORCA 输入里读回 MACE-OFF23-SC 优化的几何 (元素 + 坐标)."""
    p = ORCA / "{}_{}".format(EDGE, name) / "{}_{}.inp".format(EDGE, name)
    if not p.exists():
        raise FileNotFoundError("找不到既有输入, 无法读回几何: {}".format(p))
    lines = p.read_text(encoding="utf-8").split("\n")
    i = next(i for i, l in enumerate(lines) if l.startswith("* xyz"))
    rows = []
    for l in lines[i + 1:]:
        if l.strip() == "*":
            break
        f = l.split()
        rows.append((f[0], float(f[1]), float(f[2]), float(f[3])))
    if not rows:
        raise ValueError("几何为空: {}".format(p))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    record = dict(edge=EDGE, orca_version=ORCA_VERSION, cluster="deimos",
                  keyword_line=KEYWORD_LINE,
                  mdci="TCutPairs 1e-6, printlevel 4",
                  loc="LocMet AHFB, OCC true",
                  maxcore_mb=MAXCORE_MB, nprocs=NPROCS,
                  geometry_source="MACE-OFF23-SC 优化的最低构象 —— 与既有 .inp 逐位相同",
                  composite_notation="DLPNO-CCSD(T)/cc-pVTZ // MACE-OFF23-SC",
                  provenance="00_QM9_reaction_eng/hkuhpc/sbatch_tianhe/gen_orca_input.sh "
                             "与 core-bind/orca.md",
                  files={})
    print("=" * 88)
    print("按 deimos / ORCA {} 的规范写输入   边 {}".format(ORCA_VERSION, EDGE))
    print("=" * 88)
    print(KEYWORD_LINE)
    print(MDCI_BLOCK)
    print(LOC_BLOCK)
    print()
    for name in SPECIES:
        rows = read_geometry(name)
        stem = "{}_{}".format(EDGE, name)
        xyz = OUT / (stem + ".xyz")
        inp = OUT / (stem + ".inp")
        xyz.write_text(
            "{}\n{}  MACE-OFF23-SC optimised lowest conformer\n".format(len(rows), stem)
            + "".join("{:2s} {:18.10f} {:18.10f} {:18.10f}\n".format(*r) for r in rows),
            encoding="utf-8")
        inp.write_text(TEMPLATE.format(keyword=KEYWORD_LINE, maxcore=MAXCORE_MB,
                                       nprocs=NPROCS, mdci=MDCI_BLOCK, loc=LOC_BLOCK,
                                       xyzname=xyz.name), encoding="utf-8")
        record["files"][name] = dict(inp=str(inp.relative_to(ROOT)),
                                     xyz=str(xyz.relative_to(ROOT)),
                                     n_atoms=len(rows))
        print("  {:20s} {} 个原子 -> {} + {}".format(name, len(rows), inp.name, xyz.name))

    hint = OUT / "run_hint.sh"
    hint.write_text("""#!/usr/bin/env bash
# stage 0 电子项 —— deimos / ORCA 5.0.4 的调用约定 (逐条来自 hkuhpc 与 stage 2 的集群脚本)
#
# 1) 天河/deimos 要求: 启动 ORCA 前 unset 全部 PMI* 与 SLURM* 变量
# 2) 用绝对路径启动 ORCA, 不要靠 PATH
# 3) 运行时改写 %maxcore 与 %pal —— 生成时写下的节点相关值一律不信
# 4) 在临时目录里算, 算完再把成品原子性地搬到位
#
# 两个物种必须用**同一个 ORCA 版本、同一条关键字行、同一组阈值**,
# 否则 DLPNO 截断误差在反应两端不相消。
set -u
ORCA_BIN="${ORCA_BIN:?set ORCA_BIN to the absolute orca binary path (deimos: ORCA 5.0.4)}"

for var in $(env | awk -F= '{print $1}' | grep -E '^(PMI|SLURM)'); do unset "$var"; done

for stem in __STEMS__; do
  tmp=$(mktemp -d)
  cp "${stem}.inp" "${stem}.xyz" "$tmp/"
  ( cd "$tmp" && "$ORCA_BIN" "${stem}.inp" > "${stem}.out" 2>&1 )
  mv "$tmp/${stem}.out" ./
  rm -rf "$tmp"
done
""".replace("__STEMS__", " ".join("{}_{}".format(EDGE, n) for n in SPECIES)),
                    encoding="utf-8")
    print()
    print("  调用提示 ->", hint.name)

    (OUT / "provenance.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print("  来源记录 -> provenance.json")
    print()
    print("**既有的两份单点 (ORCA 6.0.1, TightSCF TightPNO, 无 RIJK) 与本规范不同,**")
    print("**因此作废, 不得与本规范下的结果混用。**")


if __name__ == "__main__":
    main()
