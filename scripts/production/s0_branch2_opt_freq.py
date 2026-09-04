# -*- coding: utf-8 -*-
"""分路 2 —— 对包 1 的每个构象做 `OPT FREQ`，由配分函数给出吉布斯自由能.

PRODUCTION. Branch 2: OPT FREQ per conformer and the Gibbs free energy that comes
out of it. Its output is a deliverable.

**做什么**：把单分子当作与热库接触的正则系综，用统计力学由微观性质（电子结构 + 振动频率）
算出配分函数，进而给出 `G(T)`：

    G(总) = E(电子) + G(热力学校正)

`G(热力学校正)` 由理想气体、刚性转子、简谐振动（`RRHO`）模型给出，含平动、转动、
振动与熵的贡献。ORCA 在 `FREQ` 之后的 `THERMOCHEMISTRY` 段直接输出它。

**层级**：`RI-MP2/cc-pVTZ`，库仑与交换走 `RIJK`（`cc-pVTZ/JK`），
相关拟合走 `cc-pVTZ/C`。**这是对「MP2/cc-pVTZ」的恒等分辨率近似，不是正则 MP2** ——
参考论文（J. Chem. Theory Comput. 2020, 16, 196-210）第 3.1 节报告 RI 与正则的梯度差
在 1e-4 au 量级，**但这一句必须写进交付物**，不能让读者以为是正则 MP2。

**三条本仓已实测的陷阱，本脚本逐条处理**（2026-09-01，断点 10）：

1. **绝不直接采用 ORCA 的 `Final Gibbs free energy`。** ORCA 的自动对称数会判错 ——
   实测它把丙酮判成 `C1, sigma = 1`，而正确的外对称数是 2，自由能因此偏低
   `RT ln 2 = 0.411 kcal/mol`。本脚本**把 ORCA 判的对称数原样记下并单列**，
   改正量由 `RT ln(sigma_declared / sigma_orca)` 给出；
   **对称数未独立声明的分子一律标为 provisional，不进最终数字。**
2. **准刚转子-谐振子在 ORCA 6 里是默认开着的**（实测输出 `Quasi RRHO ... True`）。
   实测丙酮两套差 0.466 kcal/mol。沿用 `D0-61`：两套都报。
   **但只跑一次 ORCA** —— 纯谐振的值由本仓 `openqha.thermo` 从同一套频率自算。
   再跑一遍 ORCA 只为关掉一个开关是白花一倍机时；而且自算顺带把
   「我们的配分函数能不能还原 ORCA 的数」变成**逐构象的断言**
   （`crosscheck_vs_orca_kcal`，计划验收 1 要求 < 0.01 kcal/mol）。
3. **虚频必须显式拒绝**，不做「取绝对值」处理 —— 那不是极小点。

**运行地点的身份**：按 `D0-75`，产生生产数字的量子化学只在 deimos 上用 ORCA 6.1.1。
**本机 WSL 的 ORCA 6.0.1 只作测试**，产物的 `provenance.status` 一律写 `test`。
用户 2026-09-02 明确要求「在本机上跑」，因此本脚本可以在本机跑，
**但产物身份如实标注，不冒充生产。**

用法::

    python scripts/production/s0_branch2_opt_freq.py --probe 10          # 代价探针：只跑一个
    python scripts/production/s0_branch2_opt_freq.py --nprocs 4 --resume # 生产
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
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

ORCA = os.environ.get("S0_ORCA_BIN", "/home/ubuntu/packages/orca_6_0_1/orca")
XYZ_DIR = S0_ROOT / "analysis" / "package1" / "1_16000" / "1_4000" / "xyz"
BASINS = S0_ROOT / "analysis" / "package1_crest_1_4000__basins.parquet"
MOLECULES = S0_ROOT / "analysis" / "package1_crest_1_4000__molecules.parquet"

R_KCAL = 1.987204259e-3          # kcal/(mol*K)
EH_KCAL = 627.5094740631         # kcal/mol per Hartree

# 简单输入行。三处都是被 ORCA 自己逼出来的，不是选择：
#
# * **`NumFreq` 而不是 `FREQ`** —— 2026-09-02 实测，`FREQ` 直接退出码 25 并报
#   `ERROR: MP2 analytic Hessian calculations are not implemented - please use NumFreq`。
#   **这决定了代价的量级**：数值 Hessian 要 `6N` 次位移梯度（10 原子 60 次、19 原子 114 次）。
# * **`TightOpt`** —— ORCA 自己警告过，几何收敛不够紧时对称性识别会失败，
#   而且残余梯度会污染最低的那几个频率，而低频正是准刚转子起作用的地方。
# * **冻芯是 ORCA 对 MP2 梯度的默认**（输出里有明确警告），本脚本沿用默认并如实记录。
SIMPLE = "! RI-MP2 cc-pVTZ cc-pVTZ/C cc-pVTZ/JK RIJK TightOpt NumFreq TightSCF"


def read_basin(qm9_index, basin):
    """从多帧 xyz 里取出指定的盆。返回 (原子数, 坐标行列表, 注释行)。"""
    p = XYZ_DIR / "{}_basins.xyz".format(qm9_index)
    lines = p.read_text(encoding="utf-8").splitlines()
    i, k = 0, 0
    while i < len(lines):
        nat = int(lines[i].split()[0])
        comment = lines[i + 1]
        body = lines[i + 2:i + 2 + nat]
        if k == int(basin):
            return nat, body, comment
        i += 2 + nat
        k += 1
    raise KeyError("{} 里没有 basin {}".format(p, basin))


def write_input(path, body, nprocs, maxcore, quasi_rrho=True, charge=0, mult=1,
                simple=None):
    lines = [simple or SIMPLE,
             "%maxcore {}".format(int(maxcore)),
             "%pal nprocs {} end".format(int(nprocs))]
    if not quasi_rrho:
        lines += ["%freq", "  QuasiRRHO false", "end"]
    lines.append("* xyz {} {}".format(int(charge), int(mult)))
    lines += list(body)
    lines.append("*")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


_PATS = dict(
    e_el=r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)",
    zpe=r"Zero point energy\s+\.+\s+(-?\d+\.\d+) Eh",
    thermal_vib=r"Thermal vibrational correction\s+\.+\s+(-?\d+\.\d+) Eh",
    inner_energy=r"Total thermal energy\s+(-?\d+\.\d+)",
    enthalpy=r"Total Enthalpy\s+\.+\s+(-?\d+\.\d+)",
    entropy_term=r"Final entropy term\s+\.+\s+(-?\d+\.\d+) Eh",
    gibbs=r"Final Gibbs free energy\s+\.+\s+(-?\d+\.\d+)",
    g_minus_eel=r"G-E\(el\)\s+\.+\s+(-?\d+\.\d+) Eh",
    quasi_rrho=r"Quasi RRHO\s+\.+\s+(\w+)",
    cutoff=r"Cut-Off Frequency\s+\.+\s+(-?\d+\.\d+)",
    sym_number=r"Point Group:\s+(\S+),\s+Symmetry Number:\s+(\d+)",
    n_basis=r"Number of basis functions\s+\.+\s+(\d+)",
)


def parse_out(text):
    """解析 ORCA 输出。**缺哪一项就报哪一项缺，不填默认值。**"""
    rec, missing = {}, []
    for k, pat in _PATS.items():
        m = re.search(pat, text)
        if not m:
            missing.append(k)
            continue
        if k == "sym_number":
            rec["point_group"] = m.group(1)
            rec["orca_symmetry_number"] = int(m.group(2))
        elif k == "quasi_rrho":
            rec[k] = (m.group(1).lower() == "true")
        elif k == "n_basis":
            rec[k] = int(m.group(1))
        else:
            rec[k] = float(m.group(1))
    # **基函数数要抓 `Dimension of the orbital basis`，不是 `Number of basis functions`。**
    #
    # 缺陷 49，2026-09-02 两次才修对：
    #   第一版取 `Number of basis functions` 的**第一个**匹配 -> 得 272（偏大）；
    #   第二版改取**最后一个** -> **也不对**，因为这一行在 SHARK 与 MP2 两个块里交替出现
    #     （实测序列 272,236,272,236,...），取哪一端都看输出在哪结束。
    #   正解：ORCA 在 `RI-MP2 ENERGY+GRADIENT` 块里明确打印
    #     `Dimension of the orbital basis ... 236` —— **只此一处无歧义**。
    #     272 是通用收缩形式下的计数，236 是 MP2 切到分段收缩后的轨道基维数；
    #     236 = 6x30 + 4x14 与 cc-pVTZ 球谐计数精确吻合。
    m = re.search(r"Dimension of the orbital basis\s+\.+\s+(\d+)", text)
    if m:
        rec["n_basis"] = int(m.group(1))
    m = re.search(r"Dimension of the AuxC basis\s+\.+\s+(\d+)", text)
    if m:
        rec["n_aux_c"] = int(m.group(1))
    rec["n_basis_lines_reported"] = [
        int(x) for x in re.findall(r"Number of basis functions\s+\.+\s+(\d+)", text)][:4]
    freqs = [float(x) for x in re.findall(
        r"^\s+\d+:\s+(-?\d+\.\d+) cm\*\*-1", text, re.M)]
    rec["frequencies_cm_inv"] = freqs
    rec["n_imaginary"] = int(sum(1 for f in freqs if f < -1.0))
    # ORCA 把 6 个平动/转动模也按 0.00 cm**-1 打印出来。
    # **最低频率必须报「最低的振动模」，不是最低的那一行** ——
    # 否则每个分子都报 0.00，这个字段就没有信息（2026-09-02 探针上暴露）。
    rigid = [f for f in freqs if abs(f) <= 1.0]
    vib = [f for f in freqs if abs(f) > 1.0]
    rec["n_zero_modes"] = len(rigid)
    # **断言**：非线型分子必须恰好 6 个零模。不是 6 个就说明几何没收敛好，
    # 或者解析抓错了段落 —— 两种都不能默默放过。
    rec["zero_modes_ok"] = (len(rigid) == 6)
    rec["lowest_vibrational_cm_inv"] = min(vib) if vib else None
    rec["lowest_frequency_cm_inv"] = min(vib) if vib else None
    rec["numerical_hessian"] = bool(re.search(r"NUMERICAL FREQUENCIES", text))
    # 优化后的笛卡尔坐标（最后一次出现的那一段）——自算配分函数要用
    blocks = re.findall(
        r"CARTESIAN COORDINATES \(ANGSTROEM\)\s*\n-+\n"
        r"((?:\s*[A-Za-z]{1,2}(?:\s+-?\d+\.\d+){3}\s*\n)+)", text)
    if blocks:
        sym, xyz = [], []
        for ln in blocks[-1].strip().split("\n"):
            f = ln.split()
            sym.append(f[0])
            xyz.append([float(v) for v in f[1:4]])
        rec["symbols"] = sym
        rec["positions_A"] = xyz
    rec["missing_fields"] = missing
    return rec



def own_thermochemistry(sub, temperature_K=298.15, pressure_Pa=1.0e5):
    """用本仓 `openqha.thermo` 从同一套频率自算纯谐振与准刚转子两套。

    **两个用途**：(a) 省掉第二次 ORCA；(b) 把「我们的配分函数能否还原 ORCA 的数」
    做成**逐构象的断言** —— 计划验收 1 要求两者差 < 0.01 kcal/mol。
    **对称数在这里显式传入，绝不自动推导**（`D0-9`）。
    """
    from openqha import thermo as T
    import numpy as np
    freqs = [f for f in sub.get("frequencies_cm_inv", []) if f > 0.0]
    sym = sub.get("symbols")
    pos = sub.get("positions_A")
    if not freqs or not sym or not pos:
        return dict(error="缺频率或优化后几何，无法自算")
    from ase.data import atomic_masses, atomic_numbers
    masses = np.array([atomic_masses[atomic_numbers[s]] for s in sym])
    pos = np.asarray(pos, dtype=float)
    out = {}
    try:
        mom = T.principal_moments(masses, pos)
        vib_h = T.vibrational(freqs, temperature_K, qrrho=False)
        vib_q = T.vibrational(freqs, temperature_K, qrrho=True,
                              moments_amu_A2=mom)
        # 转动与平动：**对称数用 ORCA 判的那个**，好与 ORCA 逐项对上；
        # 声明值的改正另算，见 sigma_correction_kcal。
        sigma_orca = int(sub.get("orca_symmetry_number") or 1)
        rot = T.rotational(masses, pos, sigma_orca, temperature_K)
        tr = T.translational(float(masses.sum()), temperature_K, pressure_Pa,
                             kind="helmholtz")
        kt = T.KB_KCAL * temperature_K
        base = rot["A_rot_kcal"] + tr["value_kcal"] + kt          # A_elec = 0 (g0=1)
        out["G_minus_Eel_rrho_kcal"] = float(vib_h["A_vib_kcal"] + base)
        out["G_minus_Eel_qrrho_kcal"] = float(vib_q["A_vib_qrrho_kcal"] + base)
        out["qrrho_minus_rrho_kcal"] = (out["G_minus_Eel_qrrho_kcal"]
                                        - out["G_minus_Eel_rrho_kcal"])
        out["ZPE_kcal"] = vib_h["ZPE_kcal"]
        out["lowest_frequency_cm_inv"] = float(min(freqs))
        out["symmetry_number_used"] = sigma_orca
        out["moments_amu_A2"] = [float(x) for x in mom]
        # 与 ORCA 的交叉检验（ORCA 默认是准刚转子）
        if sub.get("g_minus_eel") is not None:
            orca_q = sub["g_minus_eel"] * EH_KCAL
            out["orca_G_minus_Eel_kcal"] = orca_q
            out["crosscheck_vs_orca_kcal"] = out["G_minus_Eel_qrrho_kcal"] - orca_q
    except Exception as e:
        out["error"] = "{}: {}".format(type(e).__name__, str(e)[:300])
    return out


def run_one(qm9_index, basin, outdir, nprocs, maxcore, timeout_s,
            simple=None):
    """一个构象：默认（准刚转子）+ 纯谐振两次。返回记录 dict。"""
    nat, body, comment = read_basin(qm9_index, basin)
    d = Path(outdir) / "{}_b{}".format(qm9_index, basin)
    d.mkdir(parents=True, exist_ok=True)
    rec = dict(qm9_index=qm9_index, basin=int(basin), n_atoms=nat,
               source_comment=comment, simple_input=(simple or SIMPLE),
               nprocs=int(nprocs), maxcore_mb=int(maxcore), workdir=str(d))
    # **只跑一次 ORCA。** 纯谐振的值由本仓 `openqha.thermo` 从同一套频率自算 ——
    # 再跑一遍 ORCA 只为关掉准刚转子是白花一倍机时，而且自算还能顺带把
    # 「我们的配分函数能不能还原 ORCA」变成逐构象的断言（计划验收 1）。
    stem = d / "optfreq"
    write_input(str(stem) + ".inp", body, nprocs, maxcore, quasi_rrho=True,
                simple=simple)
    t0 = time.time()
    with open(str(stem) + ".out", "w") as fh:
        code = subprocess.call([ORCA, str(stem) + ".inp"], stdout=fh,
                               stderr=subprocess.STDOUT, cwd=str(d),
                               timeout=timeout_s)
    wall = time.time() - t0
    text = Path(str(stem) + ".out").read_text(encoding="utf-8", errors="replace")
    ok = "****ORCA TERMINATED NORMALLY****" in text
    if not ok:
        rec["orca"] = dict(error="ORCA 未正常结束，退出码 {}".format(code),
                           tail="\n".join(text.split("\n")[-25:]),
                           wall_seconds=wall, terminated_normally=False,
                           exit_code=code)
        rec["total_wall_seconds"] = wall
        return rec
    sub = parse_out(text)
    sub.update(wall_seconds=wall, terminated_normally=True, exit_code=code)
    rec["orca"] = sub
    rec["total_wall_seconds"] = wall
    rec["n_imaginary"] = sub.get("n_imaginary")
    # **虚频即拒绝**：那不是极小点，在它上面算热力学量没有意义
    rec["is_minimum"] = (rec["n_imaginary"] == 0)
    rec["own"] = own_thermochemistry(sub)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nprocs", type=int, default=4)
    ap.add_argument("--maxcore", type=int, default=3500)
    ap.add_argument("--outdir", default=None, help="缺省 ~/runs/branch2_opt_freq")
    ap.add_argument("--jsonl", default="analysis/branch2_opt_freq.partial.jsonl")
    ap.add_argument("--probe", type=int, default=None,
                    help="代价探针：只跑指定原子数的第一个构象")
    ap.add_argument("--only", default=None, help="逗号分隔的 qm9_index")
    ap.add_argument("--max-atoms", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--timeout-s", type=float, default=None)
    ap.add_argument("--shard", default=None,
                    help="形如 i/n：按顺序轮转分给 n 个并发实例，本实例取第 i 份（i 从 0 起）")
    ap.add_argument("--simple", default=None,
                    help="覆盖简单输入行；用来测 RIJK 与 RIJCOSX 的代价与内存")
    args = ap.parse_args()

    import pandas as pd
    from rdkit import Chem

    basins = pd.read_parquet(BASINS)
    mols = pd.read_parquet(MOLECULES)
    mols["n_atoms"] = mols.smiles.map(
        lambda s: Chem.AddHs(Chem.MolFromSmiles(s)).GetNumAtoms())
    tbl = basins.merge(mols[["qm9_index", "smiles", "n_atoms"]], on="qm9_index")

    if args.probe:
        tbl = tbl[tbl.n_atoms == args.probe].head(1)
    if args.only:
        tbl = tbl[tbl.qm9_index.isin(set(args.only.split(",")))]
    if args.max_atoms:
        tbl = tbl[tbl.n_atoms <= args.max_atoms]
    # **先按尺寸排序再轮转分片**：这样每个分片拿到同样的尺寸混合，
    # 并发的几个实例才会同时结束，而不是一个卡在大分子上、另几个早就闲着。
    tbl = tbl.sort_values(["n_atoms", "qm9_index", "basin"]).reset_index(drop=True)
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        tbl = tbl[tbl.index % n == i]
        print("分片 {}/{}：本实例 {} 个构象".format(i, n, len(tbl)))
    if args.limit:
        tbl = tbl.head(args.limit)

    out = (Path(args.outdir).expanduser() if args.outdir
           else Path.home() / "runs" / "branch2_opt_freq")
    out.mkdir(parents=True, exist_ok=True)
    jsonl = Path(args.jsonl)
    jsonl.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if args.resume and jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["qm9_index"], r["basin"]))
        print("续跑：已完成 {} 个构象".format(len(done)))

    print("=" * 96)
    print("分路 2 —— OPT FREQ 由配分函数给自由能")
    print("层级: {}".format(args.simple or SIMPLE))
    print("构象: {} 个   进程数 {}   每核内存 {} MB".format(
        len(tbl), args.nprocs, args.maxcore))
    print("输出: {}   增量: {}".format(out, jsonl))
    print("**身份**: ORCA 6.0.1 本机 WSL -> provenance.status = test（D0-75）")
    print("=" * 96, flush=True)

    t_all = time.time()
    for _, row in tbl.iterrows():
        key = (row.qm9_index, int(row.basin))
        if key in done:
            continue
        t0 = time.time()
        try:
            rec = run_one(row.qm9_index, int(row.basin), out,
                          args.nprocs, args.maxcore, args.timeout_s,
                          simple=args.simple)
        except Exception as e:
            rec = dict(qm9_index=row.qm9_index, basin=int(row.basin),
                       error="{}: {}".format(type(e).__name__, str(e)[:400]),
                       wall_seconds=time.time() - t0)
        rec.update(smiles=row.smiles, n_atoms=int(row.n_atoms),
                   provenance=dict(status="test", orca_bin=ORCA,
                                   host=os.uname().nodename,
                                   note="D0-75: 生产数字只在 deimos 用 ORCA 6.1.1"))
        with open(str(jsonl), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        q = rec.get("orca", {})
        o = rec.get("own", {})
        low = q.get("lowest_frequency_cm_inv")
        print("{} b{}  {:2d} 原子  基函数 {}  虚频 {}  最低 {}  "
              "对称数(ORCA)={}  准刚转子-谐振 {}  交叉检验 {}  墙钟 {:.0f} s".format(
                  rec["qm9_index"], rec["basin"], rec.get("n_atoms", 0),
                  q.get("n_basis"), rec.get("n_imaginary"),
                  ("{:.2f}".format(low) if low is not None else "?"),
                  q.get("orca_symmetry_number"),
                  ("{:+.4f}".format(o["qrrho_minus_rrho_kcal"])
                   if "qrrho_minus_rrho_kcal" in o else "?"),
                  ("{:+.4f}".format(o["crosscheck_vs_orca_kcal"])
                   if "crosscheck_vs_orca_kcal" in o else "?"),
                  rec.get("total_wall_seconds", 0)),
              flush=True)
    print()
    print("总墙钟 {:.0f} s".format(time.time() - t_all))


if __name__ == "__main__":
    main()
