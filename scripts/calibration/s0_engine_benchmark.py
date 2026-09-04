"""测量 Egret-1 与 MACE-OFF23-SC 的单步力计算耗时, 以及线程数的影响.

CALIBRATION. Per-step force timing and thread scaling; its numbers size the
molecular-dynamics budget.

讲义的分子动力学预算: (3 ps 平衡 x 3 轮 + 6 ps 生产) / 0.5 fs = 30000 步/物种,
两个物种共 60000 步. 用下面的实测每步耗时可以直接换算出总时长.
"""
import os
import time

import numpy as np
import torch
from ase import Atoms

print("nproc        =", os.cpu_count())
print("torch threads=", torch.get_num_threads())
print("interop      =", torch.get_num_interop_threads())

ROOT = ("/mnt/c/Users/10704/Documents/01_Free-Energy-alchemical/"
        "lambda-qm9-reaction-deltan_0-mace_v1/stage0-tradition-free-energy-calc")
EGRET = ROOT + "/../source-code/egret-public-master/compiled_models/EGRET_1.model"
EGRET_M = ROOT + "/../source-code/egret-public-master/compiled_models/EGRET_1M.model"
EGRET_S = ROOT + "/../source-code/egret-public-master/compiled_models/EGRET_1S.model"
MACE = ("/mnt/c/Users/10704/Documents/01_Free-Energy-alchemical/related-papers/"
        "workflow-design/mlps/MACE-OFF23-SC-main/MACE-OFF23-SC_swa.model")


def geometry():
    p = ROOT + "/docs/orca_inputs/C2H5O1N1_19_36_acetamide.inp"
    lines = open(p, encoding="utf-8").read().split("\n")
    i = next(i for i, l in enumerate(lines) if l.startswith("* xyz"))
    sym, xyz = [], []
    for l in lines[i + 1:]:
        if l.strip() == "*":
            break
        f = l.split()
        sym.append(f[0]); xyz.append([float(v) for v in f[1:4]])
    return Atoms(symbols=sym, positions=np.array(xyz))


atoms = geometry()
STEPS_TOTAL = 2 * (3 * 3.0 + 6.0) * 1000 / 0.5     # 两个物种的总步数


def bench(calc, label, n=40):
    a = atoms.copy(); a.calc = calc
    a.get_forces()                                   # 预热
    t0 = time.time()
    for _ in range(n):
        a.calc.results.clear()
        a.get_forces()
    dt = (time.time() - t0) / n
    print("   {:22s} {:8.1f} ms/步   -> 讲义全部 {:.0f} 步约 {:6.1f} 分钟".format(
        label, dt * 1e3, STEPS_TOTAL, dt * STEPS_TOTAL / 60))
    return dt


for nthreads in (torch.get_num_threads(), max(1, (os.cpu_count() or 4) - 2)):
    torch.set_num_threads(nthreads)
    print()
    print("=== torch threads = {} ===".format(nthreads))
    from mace.calculators import mace_off, MACECalculator
    bench(MACECalculator(model_paths=MACE, device="cpu", default_dtype="float64"),
          "MACE-OFF23-SC")
    bench(mace_off(model=EGRET, default_dtype="float64", device="cpu"), "Egret-1")
    bench(mace_off(model=EGRET_M, default_dtype="float64", device="cpu"), "Egret-1M")
    bench(mace_off(model=EGRET_S, default_dtype="float64", device="cpu"), "Egret-1S")

print()
print("float32 对照 (只看速度, 不用于生产 —— 优化与频率需要 float64):")
torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
bench(mace_off(model=EGRET, default_dtype="float32", device="cpu"), "Egret-1 float32")
