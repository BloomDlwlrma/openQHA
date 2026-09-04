# -*- coding: utf-8 -*-
"""复现并定位 D0-13 的 Langevin 故障。

假设：ASE 旧接口 `Langevin(atoms, dt, temperature, friction)` 的 `temperature` 单位是
**电子伏特**，不是开尔文。把 298.15 当温度传进去 = 298.15 eV ≈ 3.46e6 K。
"""
import sys, inspect
from pathlib import Path


def _repo_root():
    """Directory holding the openQHA package, found by walking up from this file.

    Replaces `sys.path.insert(0, ".")`, which only worked when the test happened to be
    launched from the repository root and broke silently anywhere else.
    """
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


sys.path.insert(0, str(_repo_root()))
import numpy as np
from ase import units
from ase.io import read
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from openqha import engine

print("ASE Langevin 的签名:")
print("  ", inspect.signature(Langevin.__init__))
print()
print("单位换算: 1 fs = {:.6f} ASE 时间单位".format(units.fs))
print("  tau = 1 ps -> friction = 1/(1000*units.fs) = {:.6e} (ASE^-1)".format(
    1.0 / (1000 * units.fs)))
print("  若误写成 0.001 (以为是 fs^-1) -> 实际 tau = {:.1f} fs = {:.3f} ps".format(
    1.0 / 0.001 * 1.0, 1.0 / 0.001 * 1.0 / 1000))
print("  298.15 若被当作**电子伏特**的温度 -> {:.3e} K".format(298.15 / units.kB))
print()

calc, _, _ = engine.calculator(device="cpu")
mol = read("analysis/package2/acetone/basin0.xyz")

def try_run(label, **kw):
    a = mol.copy(); a.calc = calc
    MaxwellBoltzmannDistribution(a, temperature_K=298.15, rng=np.random.RandomState(1))
    try:
        dyn = Langevin(a, 0.5 * units.fs, **kw)
        temps = []
        for _ in range(20):
            dyn.run(50)
            n_dof = 3 * len(a) - 6
            temps.append(2 * a.get_kinetic_energy() / (n_dof * units.kB))
        span = float(np.abs(a.get_all_distances()).max())
        print("{:52s} 末温 {:10.1f} K  平均 {:9.1f} K  最大间距 {:7.2f} A".format(
            label, temps[-1], float(np.mean(temps)), span))
    except Exception as e:
        print("{:52s} **抛异常**: {}".format(label, str(e)[:70]))

# 正确用法
try_run("正确: temperature_K=298.15, friction=1/(1ps)",
        temperature_K=298.15, friction=1.0 / (1000 * units.fs))
try_run("正确: temperature_K=298.15, friction=1/(0.1ps)",
        temperature_K=298.15, friction=1.0 / (100 * units.fs))
# D0-13 说的 tau = 100 fs
try_run("正确: temperature_K=298.15, friction=1/(100fs)  <- D0-13 的 tau",
        temperature_K=298.15, friction=1.0 / (100 * units.fs))
# 疑似错误用法一：把 K 当 eV 传给旧的 temperature 参数
try_run("错法一: temperature=298.15（旧接口，单位是 eV）",
        temperature=298.15, friction=1.0 / (100 * units.fs))
# 疑似错误用法二：friction 直接写 0.01 / 0.001（以为是 fs^-1）
try_run("错法二: temperature_K=298.15, friction=0.01（当成 fs^-1）",
        temperature_K=298.15, friction=0.01)
try_run("错法三: temperature_K=298.15, friction=0.001（更弱 -> 更坏？）",
        temperature_K=298.15, friction=0.001)
