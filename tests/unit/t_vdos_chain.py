# -*- coding: utf-8 -*-
"""Deterministic check of the density-of-states chain: synthetic velocities of known frequency content, judged by the spectral peaks and the sum rules."""
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
n_at, n_modes = 9, 21                      # 9 atoms -> 3N-6 = 21
dt_fs, n_frames = 4.0, 3000                # 12 ps, the same setting as production
nu_true = np.linspace(120.0, 3100.0, n_modes)
C = 2.99792458e-5                          # cm/fs
KB_EV = 8.617333262e-5
T = 298.15
masses = np.full(n_at, 12.011)

# Amplitude giving each mode its classical equipartition energy: <m v^2> = kT  ->  v_amp = sqrt(2 kT / m_tot)
t = np.arange(n_frames) * dt_fs
v = np.zeros((n_frames, n_at, 3))
dirs = rng.normal(0, 1, (n_modes, n_at, 3))
dirs /= np.sqrt((masses[None, :, None] * dirs ** 2).sum(axis=(1, 2)))[:, None, None]
for k in range(n_modes):
    amp = np.sqrt(2.0 * KB_EV * T)          # makes the time average of sum_a m_a v_a^2 equal kT
    ph = rng.uniform(0, 2 * np.pi)
    v += amp * np.cos(2 * np.pi * C * nu_true[k] * t + ph)[:, None, None] * dirs[k]

eq = vdos.equipartition_check(v, masses, T)
print("equipartition check  C_vib(0)/(3N-6)kT = {:.4f}   (should be about 1)".format(eq["ratio"]))
nu, S, Sd, nseg, raw = vdos.vacf_dos(v, masses, dt_fs, seg_ps=3.0, temperature_K=T)
print("before normalisation  integral S dnu/(3N-6) = {:.4f}   segments {}   resolution {:.1f} cm^-1".format(
    raw, nseg, nu[1] - nu[0]))
print("after normalisation  integral S dnu = {:.6f}  (should be {})".format(float((getattr(np, "trapezoid", None) or np.trapz)(S, nu)), 3 * n_at - 6))

# the spectral peaks should sit on the frequencies that were put in
peaks = []
for k in range(1, len(S) - 1):
    if S[k] > S[k - 1] and S[k] > S[k + 1] and S[k] > 0.02 * S.max():
        peaks.append(nu[k])
peaks = np.array(peaks)
err = [float(np.abs(peaks - x).min()) for x in nu_true]
print("{} frequencies put in, {} peaks found; distance from each true frequency to the nearest peak: maximum {:.1f} cm^-1 "
      "(spectral resolution {:.1f})".format(n_modes, len(peaks), max(err), nu[1] - nu[0]))

th = vdos.thermo_from_dos(nu, S, T)
print("A_vib = {:.4f}  E_vib = {:.4f}  T*S_vib = {:.4f} kcal/mol".format(
    th["A_vib_kcal"], th["E_vib_kcal"], T * th["S_vib_kcal_per_K"]))
# compare against a direct harmonic sum over the same set of frequencies
from openqha import thermo
a_harm = sum(thermo.a_mode_kcal(x, T) for x in nu_true)
print("harmonic sum over the same frequencies A_vib = {:.4f} kcal/mol   difference {:+.4f}".format(
    a_harm, th["A_vib_kcal"] - a_harm))
