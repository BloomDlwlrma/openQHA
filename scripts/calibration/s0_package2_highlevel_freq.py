"""包 2 的高层频率 —— `CCSD(T)/cc-pVTZ` 数值频率，误差棒的参考侧.

CALIBRATION. The reference side of the error bar.

参考层级由 `D0-55` 裁定。**为什么数值频率可负担**：本机 ORCA 6.0.1 对 `CCSD(T)`
打印 `CARTESIAN GRADIENT (ANALYTIC)` —— 是**解析梯度**，所以数值频率只要
**6N+1 次梯度**（9 原子 55 次、10 原子 61 次），而不是能量差分所需的 `(3N)²` 量级单点。

**几何取在哪里，是一个必须显式声明的选择。** 两种口径：

* `--geometry mace`（默认）：在**MACE-OFF23-SC 的极小点**上算高层 Hessian。
  两侧几何逐位相同，因此差值**只含曲率之差**，这正是误差棒要的量。
  代价：该几何不是高层的驻点，残余梯度会通过 Eckart 投影泄漏进低频端 ——
  脚本会把高层残余梯度的模长报出来，让这项污染可见。
* `--geometry ccsdt`：先在高层上优化再算频率。口径最正统，
  但要额外几十次梯度，**成本翻倍以上**。

用法:
    python scripts/calibration/s0_package2_highlevel_freq.py --estimate        # 只报成本, 不算
    python scripts/calibration/s0_package2_highlevel_freq.py --only acetamide  # 单个物种
    python scripts/calibration/s0_package2_highlevel_freq.py                   # 七个物种
产物:
    analysis/package2/highlevel/<物种>/{freq.inp,freq.out,freq.hess}
    analysis/package2/highlevel/highlevel_frequencies.json
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
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
from openqha import S0_ROOT, config, thermo

CFG = config.load()
T_REF = config.temperature(CFG)
OUT = S0_ROOT / "analysis" / "package2" / "highlevel"
BENCH = S0_ROOT / "analysis" / "frequency_benchmark.json"

ORCA = os.environ.get("S0_ORCA_BIN", "/home/ubuntu/packages/orca_6_0_1/orca")
# resource_budget：本机 8 个物理核，Open MPI 按物理核算槽位。
NPROCS = int(os.environ.get("S0_ORCA_NPROCS", "8"))
MAXCORE_MB = int(os.environ.get("S0_ORCA_MAXCORE", "4000"))

TEMPLATE = """! CCSD(T) cc-pVTZ {task} TightSCF
# stage 0 包 2 的高层参考频率。层级由 D0-55 裁定。
# 几何口径: {geometry_note}
%maxcore {maxcore}
%pal nprocs {nprocs} end
{extra}* xyz 0 1
{coords}*
"""


def geometry_of(name, basin, cfg=CFG):
    p = S0_ROOT / "analysis" / "package2" / name / "basin{}.xyz".format(basin)
    if not p.exists():
        raise FileNotFoundError("包 2 的盆几何未找到: {} —— 先跑 "
                                "scripts/calibration/s0_package2_hessian_benchmark.py".format(p))
    lines = p.read_text(encoding="utf-8").strip().split("\n")
    return "".join("{:2s} {:18.10f} {:18.10f} {:18.10f}\n".format(
        f[0], float(f[1]), float(f[2]), float(f[3]))
        for f in (l.split() for l in lines[2:]) if len(f) == 4)


def load_bench():
    if not BENCH.exists():
        raise FileNotFoundError("{} 不在 —— 先跑包 2 的 MACE 侧".format(BENCH))
    return json.loads(BENCH.read_text(encoding="utf-8"))


def run_orca(workdir, inp_name, log):
    """在临时目录里算, 算完把成品原子性地搬回来 (沿用集群那套约定)."""
    tmp = Path(tempfile.mkdtemp(prefix="s0_orca_"))
    try:
        for f in workdir.iterdir():
            if f.is_file():
                shutil.copy2(f, tmp)
        t0 = time.time()
        with (tmp / (inp_name.replace(".inp", ".out"))).open("w") as fh:
            proc = subprocess.run([ORCA, inp_name], cwd=tmp, stdout=fh,
                                  stderr=subprocess.STDOUT)
        seconds = time.time() - t0
        for pat in ("*.out", "*.hess", "*.engrad", "*.xyz"):
            for f in tmp.glob(pat):
                shutil.copy2(f, workdir)
        return proc.returncode, seconds
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def parse_hess(path):
    """从 ORCA 的 .hess 里读频率 (cm^-1) 与原子数."""
    txt = Path(path).read_text(encoding="utf-8", errors="replace")
    if "$vibrational_frequencies" not in txt:
        raise RuntimeError("{} 里没有 $vibrational_frequencies".format(path))
    block = txt.split("$vibrational_frequencies", 1)[1].split("$", 1)[0].strip().split("\n")
    n = int(block[0].split()[0])
    freqs = [float(block[1 + i].split()[1]) for i in range(n)]
    return np.asarray(freqs)


def terminated_normally(out_path):
    """**只读输出文件自己**判定成败, 不看退出码 (skills §2.4(c))."""
    txt = Path(out_path).read_text(encoding="utf-8", errors="replace")
    return "****ORCA TERMINATED NORMALLY****" in txt


def residual_gradient(out_path):
    """高层在该几何上的残余梯度模长 (Eh/bohr). 它量的是"这个几何离高层驻点多远"。"""
    txt = Path(out_path).read_text(encoding="utf-8", errors="replace")
    key = "CARTESIAN GRADIENT"
    if key not in txt:
        return None
    tail = txt.rsplit(key, 1)[1].split("\n")
    vals = []
    for line in tail[2:]:
        f = line.split()
        if len(f) >= 6 and f[1] == ":":
            vals.extend(float(x) for x in f[-3:])
        elif vals:
            break
    return float(np.linalg.norm(vals)) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="只跑一个物种 (英文名)")
    ap.add_argument("--geometry", choices=("mace", "ccsdt"), default="mace")
    ap.add_argument("--estimate", action="store_true", help="只报成本, 不提交计算")
    ap.add_argument("--seconds-per-gradient", type=float, default=None,
                    help="实测的单次梯度耗时, 用于成本外推")
    args = ap.parse_args()

    bench = load_bench()
    species = bench["species"]
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("包 2 高层频率   CCSD(T)/cc-pVTZ 数值频率   T = {} K".format(T_REF))
    print("=" * 100)
    print("ORCA        {}".format(ORCA))
    print("并行        {} 进程 x {} MB".format(NPROCS, MAXCORE_MB))
    print("几何口径    {}".format(
        "MACE-OFF23-SC 的极小点（两侧同几何，差值只含曲率之差）"
        if args.geometry == "mace" else "先在 CCSD(T)/cc-pVTZ 上优化再算频率"))
    print()

    rows = []
    for qid, v in species.items():
        if args.only and v["name"] != args.only:
            continue
        n_at = int(v["n_atoms"])
        rows.append((qid, v, n_at, 6 * n_at + 1))

    total_grad = sum(r[3] for r in rows)
    print("{:22s} {:>8s} {:>10s} {:>12s}".format("物种", "原子数", "梯度次数", "对标用盆"))
    print("-" * 100)
    for qid, v, n_at, ng in rows:
        print("{:22s} {:8d} {:10d} {:12d}".format(
            v["name"], n_at, ng, v["benchmark_vs_qm9"]["conformer_used"]))
    print("-" * 100)
    print("{:22s} {:8s} {:10d}".format("合计", "", total_grad))

    spg = args.seconds_per_gradient
    if spg:
        print()
        print("按实测 {:.0f} s/梯度 外推: {:.1f} 小时 = {:.2f} 天（串行，本机 {} 进程）".format(
            spg, total_grad * spg / 3600.0, total_grad * spg / 86400.0, NPROCS))
        if args.geometry == "ccsdt":
            print("  **另加高层几何优化**，按每物种 20–40 次梯度计，再加 {:.1f}–{:.1f} 小时".format(
                20 * len(rows) * spg / 3600.0, 40 * len(rows) * spg / 3600.0))
        print("  **这是资源预算，不是物理参数** —— 缩短只能靠换机器或换层级，")
        print("  不得靠减少位移数（那会直接改变数值频率的精度）。")

    if args.estimate:
        print()
        print("（`--estimate` 模式，未提交任何计算）")
        return

    results = {}
    for qid, v, n_at, ng in rows:
        d = OUT / v["name"]
        d.mkdir(parents=True, exist_ok=True)
        out_file = d / "freq.out"
        hess_file = d / "freq.hess"
        if hess_file.exists() and out_file.exists() and terminated_normally(out_file):
            print("[跳过] {} 已有成品".format(v["name"]))
        else:
            coords = geometry_of(v["name"], v["benchmark_vs_qm9"]["conformer_used"])
            task = "NumFreq" if args.geometry == "mace" else "Opt NumFreq"
            (d / "freq.inp").write_text(TEMPLATE.format(
                task=task, maxcore=MAXCORE_MB, nprocs=NPROCS, extra="",
                geometry_note=args.geometry, coords=coords), encoding="utf-8")
            print("[开算] {}  {} 次梯度 ...".format(v["name"], ng), flush=True)
            code, seconds = run_orca(d, "freq.inp", None)
            if not out_file.exists() or not terminated_normally(out_file):
                print("  ** 未正常结束（退出码 {}，用时 {:.0f} s）—— 记录并继续".format(
                    code, seconds))
                results[qid] = dict(name=v["name"], status="failed",
                                    returncode=code, seconds=seconds)
                continue
            print("  完成，用时 {:.0f} s = {:.2f} 小时（{:.0f} s/梯度）".format(
                seconds, seconds / 3600.0, seconds / ng))

        nu_hi = parse_hess(hess_file)
        nu_hi = np.asarray(sorted(x for x in nu_hi if abs(x) > 1e-6))
        nu_lo = np.asarray(v["hessian"]["frequencies_cm_inv"])
        n_imag_hi = int((nu_hi < 0).sum())
        rec = dict(name=v["name"], status="ok", n_atoms=n_at, n_gradients=ng,
                   geometry_convention=args.geometry,
                   residual_gradient_Eh_bohr=residual_gradient(out_file),
                   n_imaginary_highlevel=n_imag_hi,
                   frequencies_highlevel_cm_inv=[float(x) for x in nu_hi],
                   frequencies_mace_cm_inv=[float(x) for x in nu_lo])
        if nu_hi.shape == nu_lo.shape and n_imag_hi == 0:
            rec["comparison"] = thermo.direct_shift(nu_lo, nu_hi, T_REF)
            c = rec["comparison"]
            print("  vs CCSD(T): 平均绝对偏差 {:.2f}, 均方根偏差 {:.2f}, 最大 {:.2f} cm^-1; "
                  "有符号均值 {:+.2f}".format(
                      c["mean_absolute_deviation_cm_inv"],
                      c["root_mean_square_deviation_cm_inv"],
                      c["max_absolute_deviation_cm_inv"],
                      c["signed_mean_deviation_cm_inv"]))
            print("  -> A_vib 之差（实测，无相关性假设）= {:+.4f} kcal/mol".format(
                c["delta_A_vib_kcal"]))
        else:
            rec["comparison"] = None
            print("  ** 模式数 {} vs {}，高层虚频 {} 个 —— 不可逐模式比".format(
                nu_hi.shape, nu_lo.shape, n_imag_hi))
        results[qid] = rec

    payload = dict(generated_by="scripts/calibration/s0_package2_highlevel_freq.py",
                   reference_level="CCSD(T)/cc-pVTZ 数值频率 (D0-55)",
                   geometry_convention=args.geometry,
                   orca_binary=ORCA, nprocs=NPROCS, temperature_K=T_REF,
                   species=results)
    (OUT / "highlevel_frequencies.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    print("落盘: {}".format(OUT / "highlevel_frequencies.json"))


if __name__ == "__main__":
    main()
