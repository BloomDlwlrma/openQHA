"""Molecular dynamics -> body-fixed velocities -> velocity autocorrelation -> vibrational
density of states -> quantum-harmonic weighted free energy.

RETIRED 2026-09-03, superseded by `openqha/qha.py` (branch B, quasi-harmonic analysis).
Kept rather than deleted, and NOT to be used for a thermodynamic number.

    Why it was retired, in one measurement (D0-P2-13). With the potential held fixed and
    only the molecular-dynamics seed changed, three blank-control trajectories gave A_vib
    spreads of 10.0370 kcal/mol (acetamide, 6 ps) and 16.7406 kcal/mol
    (N-methylformamide), while the perturbed SIGNAL was 8.69 and 6.27. The signal was
    smaller than the noise floor. Pushing that noise below 0.5 kcal/mol needs roughly
    1600x the trajectory length -- about 9.6 ns each, six CPU-days each.

    Why quasi-harmonic analysis might do better: it is a VARIANCE estimator rather than a
    SPECTRAL one. A density of states has to resolve structure along the frequency axis
    and so needs a long time series; quasi-harmonic analysis only needs the second moment
    of the position to converge. "Different" is not "better", which is why plan_B
    acceptance criterion 5 repeats this same blank control on the new route, where it can
    still fail.

    `run_nve` here is dead code as well (AUDIT section 0 item 12).

    The chain below is correct and still tested (tests/unit/t_vdos_chain.py). What changed
    is what it is for: a worked reference for the velocity-side machinery, not a route to
    a number.

**Identity**: this is a **pilot implementation in service of the error bar**, not the
production implementation of package 3 or package 4. It **rewrites locally** the chain of
chapters 3, 5 and 6 of the lecture notes `s0-1`, following this repository's principle of
independence (the lecture notebooks are reference material; they are neither modified nor
imported). When the production implementation lands it should take over from this.

The chain, and its three independent checks:

1. **Pure microcanonical dynamics** (no thermostat) -- a thermostat injects correlations
   into the velocities that do not belong to the potential surface, and the velocity
   autocorrelation is exactly what we are measuring. The temperature is counted over
   `3N-6` degrees of freedom, not `3N`.
2. **Body-fixed velocities**: the centre-of-mass translation and the overall rotation are
   removed frame by frame. Without that, `W_A(nu)` diverges as `nu -> 0` and those 6
   translational and rotational degrees of freedom drag the free energy to minus infinity.
3. **Mass-weighted autocorrelation in Welch segments -> density of states**, with the
   physical prefactor fixed by Parseval's theorem. The ratio of `integral S dnu` to
   `3N-6` is reported **before normalisation** -- normalising first and then "verifying"
   that it equals `3N-6` is circular.

Units: velocities in A/fs (the ASE convention), masses in amu, and `m v^2` is twice the
kinetic energy, in eV.
"""
import numpy as np

#: numpy renamed `trapz` to `trapezoid` in 2.0 and kept the same function. The
#: environment pins 1.26.4 (openmm-torch's pytorch pin settles it), so binding the name
#: once here keeps this module working on both -- rather than on whichever numpy the
#: author happened to have.
#:
#: Found by tests/unit/t_vdos_chain.py the moment the OpenMM stack was merged into the
#: core environment. The stack fingerprint had reported that change as costing zero, and
#: it does cost zero NUMERICALLY -- but a fingerprint compares numbers, and a function
#: that no longer exists produces none to compare. Those are two different questions.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

from .thermo import HC_KCAL, KB_KCAL, T_REF

C_LIGHT_CM_PER_FS = 2.99792458e-5      # cm/fs
KB_EV = 8.617333262e-5                 # eV/K


def molecular_temperature(atoms):
    """Temperature over `3N-6` degrees of freedom.

    ASE's `get_temperature()` divides by `3N`. For a molecule that already has its
    centre-of-mass translation and overall rotation removed, the right count is `3N-6` --
    for a 10-atom molecule that is a factor 30/24 = 1.25, which is not negligible.
    """
    n_dof = 3 * len(atoms) - 6
    from ase import units
    return 2.0 * atoms.get_kinetic_energy() / (n_dof * units.kB)


def rescale_to_target(atoms, target_K):
    """Remove centre-of-mass translation and overall rotation, then scale the kinetic
    energy to the target temperature over `3N-6` degrees of freedom."""
    from ase import units
    from ase.md.velocitydistribution import Stationary, ZeroRotation
    Stationary(atoms)
    ZeroRotation(atoms)
    n_dof = 3 * len(atoms) - 6
    ekin = atoms.get_kinetic_energy()
    atoms.set_velocities(atoms.get_velocities()
                         * np.sqrt(0.5 * n_dof * units.kB * target_K / ekin))
    return molecular_temperature(atoms)


def molecular_integrity(atoms, d_max_ref, e_min, ev_to_kcal=23.060547830618307,
                        temperature_K=T_REF):
    """Is the molecule still one molecule? Returns (passed, report). Failing any one of
    these means it has come apart or passed through itself."""
    d = atoms.get_all_distances()
    np.fill_diagonal(d, np.inf)
    kt = KB_KCAL * temperature_K
    rep = dict(max_distance=float(atoms.get_all_distances().max()),
               min_distance=float(d.min()),
               pe_above_min=float((atoms.get_potential_energy() - e_min) * ev_to_kcal),
               temperature=float(molecular_temperature(atoms)))
    ok = (rep["max_distance"] < 1.5 * d_max_ref
          and rep["min_distance"] > 0.7
          and rep["pe_above_min"] < 3.0 * (3 * len(atoms) - 6) * kt)
    return bool(ok), rep


def equilibrate(atoms, target_K, timestep_fs=0.5, segment_ps=1.0, max_iters=6,
                tolerance_K=None, sample_every=8):
    """Iterative equilibration, scaling by the **segment-averaged** temperature rather
    than the instantaneous one. Returns (history, the size of the single-sample
    fluctuation).

    **Why the segment average is required**: for a molecule with `3N-6 = 24` degrees of
    freedom the single-sample temperature fluctuation is `T*sqrt(2/(3N-6))` which is about
    `86 K`. Scaling by the instantaneous temperature means **treating one sample as the
    mean**; after scaling, kinetic and potential energy re-equipartition over the
    microcanonical segment and the average temperature drifts off by tens of kelvin --
    **which is exactly how this repository's first run of option B went wrong on
    2026-08-29** (the acetamide reference segment ran at 352 K).

    The convergence criterion, written as a sentence: **the average temperature over a
    continuous `segment_ps` differs from the target by less than half the single-sample
    fluctuation**. It is a convention that may be changed, and its consequence (the number
    of iterations) is recorded honestly.
    """
    from ase import units
    from ase.md.velocitydistribution import Stationary, ZeroRotation
    from ase.md.verlet import VelocityVerlet
    n_dof = 3 * len(atoms) - 6
    sigma_1 = float(target_K * np.sqrt(2.0 / n_dof))
    tol = float(tolerance_K if tolerance_K is not None else 0.5 * sigma_1)
    hist, converged = [], False
    for it in range(int(max_iters)):
        temps = []
        dyn = VelocityVerlet(atoms, timestep_fs * units.fs)
        dyn.attach(lambda: temps.append(molecular_temperature(atoms)),
                   interval=sample_every)
        dyn.run(int(segment_ps * 1000 / timestep_fs))
        tm = float(np.mean(temps))
        hist.append(dict(iteration=int(it), segment_mean_K=tm,
                         segment_std_K=float(np.std(temps)),
                         deviation_K=float(tm - target_K)))
        if abs(tm - target_K) <= tol:
            converged = True
            break
        Stationary(atoms)
        ZeroRotation(atoms)
        atoms.set_velocities(atoms.get_velocities() * np.sqrt(target_K / tm))
    return dict(history=hist, converged=bool(converged),
                single_sample_fluctuation_K=sigma_1, tolerance_K=tol,
                n_iterations=len(hist),
                final_segment_mean_K=hist[-1]["segment_mean_K"]), sigma_1


def run_nve(atoms, calc, seed, timestep_fs=0.5, equil_ps=1.0, prod_ps=6.0,
            sample_every=8, n_equil_cycles=6, temperature_K=T_REF,
            initial_velocities=None, progress=None):
    """Iterative equilibration then pure microcanonical production. Returns (velocities,
    positions, the temperature series, the total-energy series, a record).

    **The meaning of `initial_velocities` has changed (a defect fixed on 2026-08-29)**: it
    is now only the **initial sample**, used to make a paired experiment reproducible;
    **every trajectory still equilibrates on its own potential surface**.

    The old meaning was "take the reference trajectory's equilibrated velocities directly
    and skip equilibration", and that was **wrong**: the perturbed potential has non-zero
    forces and a non-zero potential-energy offset at that geometry, the microcanonical
    segment converts that into kinetic energy, and the perturbed trajectories then ran at
    anywhere from 177 K to 1209 K -- so what was being compared was no longer two
    ensembles at the same temperature.
    """
    from ase import units
    from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution,
                                             Stationary, ZeroRotation)
    from ase.md.verlet import VelocityVerlet
    a = atoms.copy()
    a.calc = calc
    e_min = a.get_potential_energy()
    d_max_ref = float(a.get_all_distances().max())

    if initial_velocities is not None:
        a.set_velocities(np.asarray(initial_velocities, dtype=float))
    else:
        rng = np.random.RandomState(seed)
        # start the kinetic energy at 2T: equipartition turns half of it into potential
        # energy, so after equilibration it settles at about T
        MaxwellBoltzmannDistribution(a, temperature_K=2.0 * temperature_K, rng=rng)
    Stationary(a)
    ZeroRotation(a)
    # **every trajectory equilibrates on its own potential surface** -- the perturbed
    # ones included
    equil_rec, _sigma1 = equilibrate(a, temperature_K, timestep_fs=timestep_fs,
                                     segment_ps=equil_ps, max_iters=n_equil_cycles,
                                     sample_every=sample_every)
    ok, rep = molecular_integrity(a, d_max_ref, e_min, temperature_K=temperature_K)
    if not ok:
        raise RuntimeError("molecular-integrity criterion failed (end of "
                           "equilibration): {}".format(rep))
    t_equil = molecular_temperature(a)
    n_cycles_done = equil_rec["n_iterations"]

    v0 = a.get_velocities().copy()
    dyn = VelocityVerlet(a, timestep_fs * units.fs)
    vel, pos, temps, etot = [], [], [], []

    def grab():
        vel.append(a.get_velocities().copy())
        pos.append(a.get_positions().copy())
        temps.append(molecular_temperature(a))
        etot.append(a.get_total_energy())

    dyn.attach(grab, interval=sample_every)
    n_steps = int(prod_ps * 1000 / timestep_fs)
    import time as _t
    t0 = _t.time()
    if progress is None:
        dyn.run(n_steps)
    else:
        chunk = max(1, n_steps // 20)
        done = 0
        while done < n_steps:
            step = min(chunk, n_steps - done)
            dyn.run(step)
            done += step
            progress(done, n_steps)
    wall = _t.time() - t0

    ok, rep = molecular_integrity(a, d_max_ref, e_min, temperature_K=temperature_K)
    if not ok:
        raise RuntimeError("molecular-integrity criterion failed (end of the production "
                           "segment): {}".format(rep))
    v = np.array(vel)
    x = np.array(pos)
    tt = np.array(temps)
    ee = np.array(etot)
    sigma_1 = float(temperature_K * np.sqrt(2.0 / (3 * len(a) - 6)))
    rec = dict(equilibration=equil_rec,
               single_sample_fluctuation_K=sigma_1,
               production_mean_deviation_K=float(tt.mean() - temperature_K),
               production_temperature_ok=bool(
                   abs(float(tt.mean()) - temperature_K) <= sigma_1),
               seed=int(seed), timestep_fs=float(timestep_fs),
               equil_ps=float(equil_ps), prod_ps=float(prod_ps),
               sample_every=int(sample_every), n_equil_cycles=int(n_cycles_done),
               n_frames=int(len(v)), frame_spacing_fs=float(sample_every * timestep_fs),
               temperature_after_rescale_K=float(t_equil),
               temperature_mean_K=float(tt.mean()), temperature_std_K=float(tt.std()),
               total_energy_drift_eV=float(ee[-1] - ee[0]),
               total_energy_drift_eV_per_ps=float((ee[-1] - ee[0]) / prod_ps),
               wall_seconds=float(wall), integrity=rep,
               initial_velocities=v0.tolist())
    return v, x, tt, ee, rec


def body_fixed_velocities(positions, velocities, masses):
    """Remove centre-of-mass translation and overall rotation frame by frame, returning
    the vibrational velocities."""
    m = np.asarray(masses, dtype=float)
    mm = m[None, :, None]
    v_cm = (mm * velocities).sum(1, keepdims=True) / m.sum()
    r_cm = (mm * positions).sum(1, keepdims=True) / m.sum()
    dr = positions - r_cm
    dv = velocities - v_cm
    ang = (mm * np.cross(dr, dv)).sum(1)                                   # (T,3)
    r2 = (dr ** 2).sum(-1)                                                 # (T,N)
    inertia = (m[None, :, None, None]
               * (r2[..., None, None] * np.eye(3)[None, None]
                  - dr[..., :, None] * dr[..., None, :])).sum(1)           # (T,3,3)
    omega = np.linalg.solve(inertia, ang[..., None])[..., 0]               # (T,3)
    return dv - np.cross(omega[:, None, :], dr)


def vacf_dos(vel_vib, masses, dt_fs, seg_ps=3.0, shift_ps=0.5,
             temperature_K=T_REF):
    """Mass-weighted velocity autocorrelation in Welch segments -> density of states.

    Returns (nu in cm^-1, the mean S, the standard deviation of S across segments, the
    number of segments, and the ratio of `integral S dnu` to `3N-6` BEFORE normalisation).

    That last ratio is **the real check** -- normalising first and then "verifying" that
    it equals `3N-6` is circular.
    """
    n_frames, n_at, _ = vel_vib.shape
    seg = int(seg_ps * 1000 / dt_fs)
    shift = max(1, int(shift_ps * 1000 / dt_fs))
    if seg > n_frames:
        seg = n_frames
    w = np.hanning(seg)
    window = w[:, None, None]
    window_power = float((w ** 2).mean())          # Hann window: 3/8
    m = np.asarray(masses, dtype=float)
    spectra = []
    starts = list(range(0, n_frames - seg + 1, shift)) or [0]
    for s in starts:
        v = vel_vib[s:s + seg] * window
        f = np.fft.rfft(v, axis=0)
        # the physical prefactor is fixed by Parseval's theorem:
        #     S(nu_j) = 2 dt / (N kB T) * sum_a m_a |F_{a,j}|^2 / mean(w^2)
        p = (m[None, :, None] * (np.abs(f) ** 2)).sum(axis=(1, 2))
        p *= 2.0 * dt_fs / (seg * KB_EV * temperature_K * window_power)
        spectra.append(p)
    spectra = np.array(spectra)
    nu_cm = np.fft.rfftfreq(seg, d=dt_fs) / C_LIGHT_CM_PER_FS
    # abscissa 1/fs -> cm^-1; the ordinate must be multiplied by C_LIGHT_CM_PER_FS to
    # leave the integral unchanged
    s_mean = spectra.mean(0) * C_LIGHT_CM_PER_FS
    s_std = spectra.std(0) * C_LIGHT_CM_PER_FS
    raw = float(_trapezoid(s_mean, nu_cm))
    target = 3 * n_at - 6
    return (nu_cm, s_mean * (target / raw), s_std * (target / raw),
            len(starts), raw / target)


def thermo_from_dos(nu_cm, dos, temperature_K=T_REF):
    """Quantum-harmonic weighting: `A = kT integral S W_A dnu`, in kcal/mol, with
    `W_A = beta h c nu / 2 + ln(1 - e^{-beta h c nu})`."""
    kt = KB_KCAL * temperature_K
    bhc = HC_KCAL / kt
    nu = np.asarray(nu_cm, dtype=float)
    s = np.asarray(dos, dtype=float)
    ok = nu > 1e-9
    x = bhc * nu[ok]
    ss = s[ok]
    w_a = 0.5 * x + np.log1p(-np.exp(-x))
    w_e = 0.5 * x + x / np.expm1(x)
    w_s = x / np.expm1(x) - np.log1p(-np.exp(-x))
    return dict(A_vib_kcal=float(kt * _trapezoid(ss * w_a, nu[ok])),
                E_vib_kcal=float(kt * _trapezoid(ss * w_e, nu[ok])),
                S_vib_kcal_per_K=float(KB_KCAL * _trapezoid(ss * w_s, nu[ok])),
                temperature_K=float(temperature_K))


def equipartition_check(vel_vib, masses, temperature_K=T_REF):
    """An independent check: `C_vib(0) = (3N-6) kT`. It looks only at the velocities and
    is independent of the normalisation."""
    from ase import units
    m = np.asarray(masses, dtype=float)
    c0 = float((m[None, :, None] * vel_vib ** 2).sum(axis=(1, 2)).mean())   # eV
    n_at = vel_vib.shape[1]
    expect = (3 * n_at - 6) * units.kB * temperature_K
    return dict(c0_eV=c0, expected_eV=float(expect), ratio=c0 / float(expect))
