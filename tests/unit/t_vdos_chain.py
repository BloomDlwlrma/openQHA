# -*- coding: utf-8 -*-
"""态密度链条的确定性检验: 用已知频率成分的合成速度, 看谱峰与求和规则."""
import sys
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
from openqha import vdos

rng = np.random.default_rng(0)
n_at, n_modes = 9, 21                      # 9 原子 -> 3N-6 = 21
dt_fs, n_frames = 4.0, 3000                # 12 ps, 与生产设置同口径
nu_true = np.linspace(120.0, 3100.0, n_modes)
C = 2.99792458e-5                          # cm/fs
KB_EV = 8.617333262e-5
T = 298.15
masses = np.full(n_at, 12.011)

# 每个模给经典能量均分的振幅: <m v^2> = kT  ->  v_amp = sqrt(2 kT / m_tot)
t = np.arange(n_frames) * dt_fs
v = np.zeros((n_frames, n_at, 3))
dirs = rng.normal(0, 1, (n_modes, n_at, 3))
dirs /= np.sqrt((masses[None, :, None] * dirs ** 2).sum(axis=(1, 2)))[:, None, None]
for k in range(n_modes):
    amp = np.sqrt(2.0 * KB_EV * T)          # 使 sum_a m_a v_a^2 的时均 = kT
    ph = rng.uniform(0, 2 * np.pi)
    v += amp * np.cos(2 * np.pi * C * nu_true[k] * t + ph)[:, None, None] * dirs[k]

eq = vdos.equipartition_check(v, masses, T)
print("能量均分检验  C_vib(0)/(3N-6)kT = {:.4f}   (应约 1)".format(eq["ratio"]))
nu, S, Sd, nseg, raw = vdos.vacf_dos(v, masses, dt_fs, seg_ps=3.0, temperature_K=T)
print("归一化前 ∫S dν/(3N-6) = {:.4f}   段数 {}   分辨率 {:.1f} cm^-1".format(
    raw, nseg, nu[1] - nu[0]))
print("归一化后 ∫S dν = {:.6f}  (应为 {})".format(float(np.trapezoid(S, nu)), 3 * n_at - 6))

# 谱峰应落在放进去的频率上
peaks = []
for k in range(1, len(S) - 1):
    if S[k] > S[k - 1] and S[k] > S[k + 1] and S[k] > 0.02 * S.max():
        peaks.append(nu[k])
peaks = np.array(peaks)
err = [float(np.abs(peaks - x).min()) for x in nu_true]
print("放进去 {} 个频率, 找到 {} 个峰; 每个真频到最近峰的距离: 最大 {:.1f} cm^-1 "
      "(谱分辨率 {:.1f})".format(n_modes, len(peaks), max(err), nu[1] - nu[0]))

th = vdos.thermo_from_dos(nu, S, T)
print("A_vib = {:.4f}  E_vib = {:.4f}  T*S_vib = {:.4f} kcal/mol".format(
    th["A_vib_kcal"], th["E_vib_kcal"], T * th["S_vib_kcal_per_K"]))
# 与直接对同一组频率做谐振求和比
from openqha import thermo
a_harm = sum(thermo.a_mode_kcal(x, T) for x in nu_true)
print("同一组频率的谐振求和 A_vib = {:.4f} kcal/mol   差 {:+.4f}".format(
    a_harm, th["A_vib_kcal"] - a_harm))
