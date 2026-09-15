"""Branch B production trajectories: unbiased, unconstrained, real-mass canonical MD.

PRODUCTION DRIVER. One basin, one seed, one trajectory. It writes frames and a metadata
record; it computes no thermodynamics. The analysis is `scripts/production/s0_B_qha_analyse.py`.

The three things this script must never do
------------------------------------------
They are prohibitions, not settings, and they are written into the metadata so that
`openqha/qha.py` can refuse a trajectory that violated them (acceptance criterion 11):

  1. NO BIAS. No metadynamics, no restraint, no pulling. A biased distribution is not the
     Boltzmann distribution, the covariance eigenvalues come out too large, the
     frequencies too low and the entropy too HIGH -- in a fixed direction, so it is a bias
     and not noise.
  2. NO CONSTRAINTS. Not "few" -- none. A constrained bond is REMOVED, not softened, and
     its covariance is identically zero, so the count of non-zero eigenvalues cannot reach
     3N-6.
  3. REAL HYDROGEN MASS. lambda_i is the eigenvalue of the MASS-WEIGHTED covariance;
     m_H = 2 amu would report a 3000 cm^-1 C-H stretch near 2200 cm^-1.

Branch A's CREST runs use exactly the opposite of all three (SHAKE on every bond, 5 fs,
m_H = 2 amu), because that is the published iMTD-GC protocol and it is right for finding
conformers. The handover between the branches is a list of GEOMETRIES. Not one CREST frame
is ever reused (S0-B-3, user ruling 2026-09-03).

Equilibration length is measured, not assumed
---------------------------------------------
Rinaldo & Field began with 20 ps of equilibration and found in the end that 520 ps were
needed -- a factor of 26. So `--equil-ps` has a literature default (50 ps, D0-C-30) and
the driver measures the potential-energy and radius-of-gyration relaxation over that
window and reports it, rather than declaring the system equilibrated because the timer
expired.

Restart and wall clock
----------------------
Frames are flushed to disk in chunks. A run that hits `--wall-budget-s` stops at a chunk
boundary and writes its state; re-running the same command continues from there. Nothing
is ever silently truncated: `progress.json` always says how many frames exist and how many
were asked for (branch E acceptance criterion 2).

    python scripts/production/s0_B_qha_trajectory.py --species dsgdb9nsd_000018 --prod-ps 20 --seeds 1
    python scripts/production/s0_B_qha_trajectory.py --basins analysis/branchA/prod/xyz/X_basins.xyz
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

def _repo_root():
    """Directory holding the openQHA package, found by walking up from this file.

    Depth-independent on purpose: this file keeps working wherever under the repository
    it is moved to. An earlier move broke every `parents[1]` in the moved files silently,
    which is what this replaces.
    """
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


sys.path.insert(0, str(_repo_root()))

from openqha import config, engine, qha  # noqa: E402

#: Frames per flush. Small enough that a killed job loses little, large enough that the
#: flush is not the cost. Not a scientific parameter.
CHUNK_FRAMES = 500

#: Protocol constants. Each carries its source; changing one invalidates the numbers that
#: were measured under it, which is why they are named here rather than left as defaults
#: scattered through the argument parser.
TIMESTEP_FS = 1.0          # D0-C-30 (Moore/Cole/Csanyi, JACS 2026, 148, 4928) and
                           # Rinaldo & Field independently. The repo's old 0.5 fs was an
                           # unsourced guess that doubled the cost for nothing.
FRICTION_PER_PS = 1.0      # D0-C-30, the MACE authors' own value.
SAMPLE_EVERY_STEPS = 500   # 0.5 ps, the published interval (configs/branchB_protocol.yaml).
                           # Was 8 fs. The covariance is an EQUAL-TIME average, so a
                           # shorter interval aliases nothing -- it just stores frames
                           # far inside the correlation time, which cost 25 000 frames
                           # for the information in a few hundred.
                           # Positions, not velocities: quasi-harmonic
                           # analysis needs the second moment of the position, so there is
                           # no Nyquist limit to respect -- only near-independence.

#: ASE's Langevin defaults to `fixcm=True`, and on a molecule this small that default does
#: not sample the canonical distribution. MEASURED, 2026-09-03, acetone geometry with a
#: cheap calculator so the statistics could be piled up, 200000 steps, target 298.15 K
#: (scripts/calibration/s0_B_thermostat_fixcm.py). Each variant is read on ITS OWN
#: divisor, because fixing or zeroing the centre of mass removes three degrees of freedom:
#:
#:     fixcm=True (ASE default)   dof 27   T = 429.29 +- 10.11 K   +43.98 percent
#:     fixcm=False (used here)    dof 30   T = 303.65 +-  5.50 K    +1.85 percent
#:     fixcm=False + FixCom       dof 27   T = 298.10 +-  7.00 K    -0.02 percent
#:
#: The default costs +0.6991 kcal/mol on T*S -- 70 per cent of this repo's entire
#: 1.0 kcal/mol accuracy target.
#:
#: Upstream says so itself, in a FutureWarning: "The implementation of fixcm=True in
#: Langevin does not strictly sample the correct NVT distributions. The deviations are
#: typically small for large systems but can be more pronounced for small systems."
#: Ours are 10 to 19 atoms, which is the far end of "small".
#:
#: This matters more than a warm trajectory. Quasi-harmonic analysis reads
#: nu = sqrt(k_B T / lambda) with T taken from the CONFIGURED temperature, so a trajectory
#: that is 44 percent hot inflates every lambda, deflates every nu, and raises the entropy
#: -- silently, in a fixed direction, with the run looking healthy throughout.
#:
#: `fixcm=False` is used WITHOUT `FixCom`, although upstream suggests the pair and the
#: measurement above shows FixCom is if anything the most accurate of the three. The
#: choice is therefore not made on accuracy: FixCom is an ASE constraint, and branch B's
#: first prohibition is that there are no constraints. This particular one would be
#: harmless -- it removes the three translational degrees of freedom the superposition
#: removes anyway -- but "no constraints except the harmless ones" is not a rule anybody
#: can check, and it stays harmless only until somebody adds a second one. Without it the
#: centre of mass random-walks, and the superposition in openqha/qha.py removes that
#: before the covariance is taken.
FIXCM = False

#: Which thermostat produces branch B trajectories.
#:
#: Nose-Hoover on instruction (2026-09-04). It is NOT a drop-in for Langevin here, and the
#: coupling time below is the whole reason: on the harmonic surface built from the
#: production potential's own Hessian, where the answer is known in closed form,
#:
#:     thermostat                       T*S error / kcal    softest mode / cm^-1
#:     Langevin, friction 1/ps               +0.031                81.6
#:     Nose-Hoover, tdamp   20 fs            -0.093                78.6
#:     Nose-Hoover, tdamp  100 fs            -1.150               179.0
#:     Nose-Hoover, tdamp  419 fs            -1.228               229.0
#:     (true value)                              --                79.7
#:
#: The kinetic temperature is right in every row. It is the CONFIGURATIONAL distribution
#: that collapses, and that is the only thing quasi-harmonic analysis reads.
#: `qha.assert_trajectory_identity` refuses any trajectory outside these limits, so this
#: is enforced rather than documented.
THERMOSTAT = "nose-hoover"
NOSE_HOOVER_TDAMP_FS = 20.0
NOSE_HOOVER_CHAIN_LENGTH = 3

#: Pin the centre of mass to the origin every step.
#:
#: This is NOT required for correctness, and the reason has been rewritten twice. What is
#: true as measured on 2026-09-04: the MACE that openQHA imports (`mace_torch 0.3.16`,
#: site-packages) is ALREADY translation invariant -- with the patch OFF, dE = 0.000000
#: kcal/mol for rigid shifts out to 100000 A and the 90-edge neighbour list is identical.
#: `openqha/mace_patch.py` is kept as insurance against being pointed at the develop tree
#: vendored under stage 2, which is NOT invariant, and `engine.calculator()` applies it
#: with `strict=True` so a run in which it failed to bind raises rather than proceeding.
#:
#: It stays on because a rigid translation is an exact symmetry, so it costs nothing and
#: changes nothing, and it keeps coordinates small. Set it False to run a true free
#: Langevin walk; the trajectory is equally valid either way, and the quasi-harmonic
#: analysis superimposes every frame regardless.
PIN_CENTRE_OF_MASS = False
EQUIL_PS = 520.0           # literature_value: Rinaldo & Field 2003 p2 -- 20 ps was
                           # tried and was NOT enough. See configs/branchB_protocol.yaml.
PROD_PS = 1500.0           # literature_value: 1.5 ns, Rinaldo & Field 2003 p2.
                           # STARTING POINT ONLY. The final length is set by the
                           # saturation curve in the analysis (criterion 1).


def load_atoms(args, cfg):
    """The starting geometries, and an honest label for where they came from."""
    from ase.io import read
    if args.basins == "auto":
        from openqha.store import basins as _basins
        btag = args.basin_tag or args.tag
        frames = _basins.read_basins(args.species, tag=btag, cfg=cfg)
        if not frames:
            raise SystemExit(_basins.missing_message(args.species, btag, cfg=cfg))
        return frames, "branch A basins: {}".format(
            _basins.molecule_for(args.species, btag, cfg) / "mace")
    if args.basins:
        frames = read(args.basins, index=":")
        return frames, "branch A basins: {}".format(args.basins)
    if not args.species:
        raise SystemExit("give --species or --basins")
    from ase import Atoms
    path = config.qm9_xyz(args.species, cfg)
    atoms = read(str(path))
    return [Atoms(atoms.get_chemical_symbols(), positions=atoms.get_positions())], (
        "QM9 reference geometry {} -- branch A has NOT been run for this species, so "
        "this is one geometry and not a basin list".format(path.name))


def relax(atoms, calc, fmax=0.005, steps=500):
    """Tighten a geometry on the production potential before any dynamics.

    A trajectory started away from the minimum spends its first picoseconds converting
    potential energy into kinetic energy, and the thermostat then has to take it out
    again. Doing this here rather than letting equilibration absorb it keeps the measured
    relaxation time a property of the molecule instead of a property of the input file.
    """
    from ase.optimize import BFGS
    a = atoms.copy()
    a.calc = calc
    opt = BFGS(a, logfile=None)
    opt.run(fmax=fmax, steps=steps)
    f = np.abs(a.get_forces()).max()
    return a, dict(fmax_reached_eV_per_A=float(f), converged=bool(f <= fmax),
                   n_steps=int(opt.get_number_of_steps()),
                   energy_eV=float(a.get_potential_energy()))


def integrity(atoms, d_max_ref, e_min, temperature_K):
    """Is it still one molecule? Reported every chunk, and fatal if it is not."""
    from openqha.thermochem.thermo import KB_KCAL
    d = atoms.get_all_distances()
    np.fill_diagonal(d, np.inf)
    kt = KB_KCAL * temperature_K
    rep = dict(max_distance_A=float(atoms.get_all_distances().max()),
               min_distance_A=float(d.min()),
               potential_above_minimum_kcal=float(
                   (atoms.get_potential_energy() - e_min) * 23.060547830618307))
    rep["ok"] = bool(rep["max_distance_A"] < 1.5 * d_max_ref
                     and rep["min_distance_A"] > 0.7
                     and rep["potential_above_minimum_kcal"]
                     < 3.0 * (3 * len(atoms) - 6) * kt)
    return rep


def recentre(atoms):
    """Put the centre of mass back on the origin. Called every step during dynamics.

    A rigid translation is an exact symmetry of any interatomic potential, so this cannot
    change physics -- and the quasi-harmonic analysis superimposes every frame anyway, so
    it cannot change a number downstream either.

    It is here as a cheap defence, not as a repair. A neighbour-search box anchored at the
    coordinate ORIGIN loses pairs once a molecule random-walks away from it, and an
    unbiased Langevin centre of mass random-walks at about 3.6 A/ps for a 58 amu molecule
    at 298 K, so it would leave a 14 A box in roughly two picoseconds. That defect is real
    in the MACE develop tree vendored under stage 2 (+978.53 kcal/mol at a 50 A shift,
    reproduced 2026-09-04). It is NOT in the tree openQHA imports, which is invariant to
    0.000000 kcal/mol out to 100000 A with no patch at all.

    So this does not explain the branch B trajectory deaths, and neither did the
    short-range hole before it. Both attributions are withdrawn and the cause is open; see
    the docstring of scripts/calibration/s0_B_md_stability.py. What is kept here is the
    defence itself, which costs nothing: on three different models, at 1.0 and 0.5 fs, the
    same seeds. The potential was never the problem.

    Velocities live on the atoms as momenta and a translation does not touch them, so
    calling this between integrator steps is safe.
    """
    m = atoms.get_masses()
    com = (atoms.get_positions() * m[:, None]).sum(axis=0) / m.sum()
    atoms.set_positions(atoms.get_positions() - com[None, :])
    return float(np.linalg.norm(com))


def instantaneous_temperature(atoms):
    """2 E_kin / (3N k_B).

    3N, not 3N-6. With `fixcm=False` every one of the 3N Cartesian degrees of freedom is
    thermostatted, including the three that carry the centre-of-mass motion, so 3N is the
    count that makes this an estimator of the bath temperature. The same holds under the
    Nose-Hoover chain, which does not remove centre-of-mass momentum either: measured on
    the calibration surface, 2<KE>/(k_B T) came back 29.9 to 30.2 against 3N = 30 for
    every coupling time tried.

    Note what this quantity can and cannot see. It is a KINETIC temperature, and
    Nose-Hoover holds it perfectly correct while the configurational distribution
    collapses -- which is the only thing quasi-harmonic analysis reads. A healthy number
    here is not evidence that the trajectory is admissible; that is what
    `qha.assert_trajectory_identity` is for. The retired
    density-of-states route used 3N-6 for a good reason -- it ran pure microcanonical
    dynamics after removing translation and rotation -- and carrying that divisor over to
    here would report 298 K as 373 K and invite somebody to "fix" a thermostat that was
    working.
    """
    from ase import units
    return 2.0 * atoms.get_kinetic_energy() / (3 * len(atoms) * units.kB)


def pin_centre_of_mass(thermostat, requested):
    """Whether re-centring is applied, and a refusal when it was asked for and cannot be.

    `requested` is None for "whatever suits the thermostat", True or False for an explicit
    choice. Under Nose-Hoover the answer is always False, because ASE's integrator
    overwrites any attempt -- see the message below. An explicit request is REFUSED rather
    than downgraded, because a caller who asked for pinning and silently did not get it is
    the exact failure this branch has been unpicking all day.
    """
    is_langevin = str(thermostat or THERMOSTAT).lower() == "langevin"
    if is_langevin:
        return PIN_CENTRE_OF_MASS if requested is None else bool(requested)
    if requested:
        raise ValueError(
            "--pin-com was requested and the thermostat is Nose-Hoover. ASE's "
            "NoseHooverChainNVT caches positions and momenta at construction "
            "(nose_hoover_chain.py lines 92-93), propagates its own copies and writes "
            "them back over the atoms at the end of every step (lines 118-119). It never "
            "reads the atoms back, so an attached observer that re-centres them -- or "
            "Stationary, or ZeroRotation -- is silently overwritten and does NOTHING. "
            "Measured: a run with the re-centring observer attached came back "
            "bit-identical to one without it, with a 1422 A centre-of-mass drift in the "
            "variant that was supposed to pin it at zero. Refusing rather than writing "
            "centre_of_mass_pinned_to_origin=True into a product where it is false.")
    return False


def make_dynamics(atoms, temperature_K, timestep_fs, friction_per_ps, rng,
                  thermostat=None, tdamp_fs=None, chain_length=None):
    """The integrator, in ONE place so both stages cannot drift apart.

    Equilibration and production used to build their own `Langevin` objects from the same
    literal arguments. That is two places to change and one of them to forget, and the
    thing being changed is the sampled ensemble.
    """
    from ase import units
    name = (thermostat or THERMOSTAT).lower()
    if name == "langevin":
        from ase.md.langevin import Langevin
        return Langevin(atoms, timestep_fs * units.fs, temperature_K=temperature_K,
                        friction=friction_per_ps / (1000.0 * units.fs), rng=rng,
                        fixcm=FIXCM), "ase.md.langevin.Langevin"
    if name in ("nose-hoover", "nose_hoover", "nosehoover"):
        from ase.md.nose_hoover_chain import NoseHooverChainNVT

        return NoseHooverChainNVT(
            atoms, timestep_fs * units.fs, temperature_K=temperature_K,
            tdamp=(tdamp_fs or NOSE_HOOVER_TDAMP_FS) * units.fs,
            tchain=int(chain_length or NOSE_HOOVER_CHAIN_LENGTH)), \
            "ase.md.nose_hoover_chain.NoseHooverChainNVT"
    raise ValueError("unknown thermostat {!r} -- 'langevin' or 'nose-hoover'".format(name))


def equilibrate(atoms, calc, temperature_K, seed, equil_ps, timestep_fs, friction_per_ps,
                thermostat=None, tdamp_fs=None, chain_length=None,
                pin_com=None):
    """Equilibration, with the relaxation MEASURED rather than assumed complete."""
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    rng = np.random.RandomState(int(seed))
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=rng)
    dyn, _name = make_dynamics(atoms, temperature_K, timestep_fs, friction_per_ps, rng,
                               thermostat, tdamp_fs, chain_length)
    trace = []

    def sample():
        trace.append((float(atoms.get_potential_energy()),
                      float(instantaneous_temperature(atoms))))

    dyn.attach(sample, interval=100)
    # Not required on the installed MACE, which is translation invariant on its own; see
    # the comment on PIN_CENTRE_OF_MASS. Every step rather than every hundred because a
    # centre of mass that random-walks at 3.6 A/ps leaves a 14 A box in about 2 ps, which
    # is the margin any origin-anchored neighbour search would give.
    if (PIN_CENTRE_OF_MASS if pin_com is None else pin_com):
        dyn.attach(lambda: recentre(atoms), interval=1)
    n_steps = int(round(equil_ps * 1000.0 / timestep_fs))
    t0 = time.time()
    dyn.run(n_steps)
    wall = time.time() - t0
    arr = np.array(trace) if trace else np.zeros((1, 2))
    half = max(1, len(arr) // 2)
    return dict(
        equil_ps=float(equil_ps), n_steps=int(n_steps), wall_seconds=float(wall),
        # The relaxation test: does the second half of the window differ from the first?
        # If it does, the window was too short and the number below says by how much.
        potential_first_half_eV=float(arr[:half, 0].mean()),
        potential_second_half_eV=float(arr[half:, 0].mean()),
        potential_drift_eV=float(arr[half:, 0].mean() - arr[:half, 0].mean()),
        temperature_second_half_K=float(arr[half:, 1].mean()),
        temperature_std_K=float(arr[half:, 1].std()),
        target_temperature_K=float(temperature_K),
        # The width of one sample, so the deviation above can be read against something.
        # For 10 atoms this is 77 K: a single frame says almost nothing about the
        # temperature, and a thermostat "checked" against one instantaneous value is not
        # checked at all.
        single_sample_fluctuation_K=float(
            temperature_K * np.sqrt(2.0 / (3 * len(atoms)))),
        fixcm=bool(FIXCM),
        degrees_of_freedom_used_for_temperature=int(3 * len(atoms)),
    )


def produce(atoms, calc, outdir, temperature_K, seed, prod_ps, timestep_fs,
            friction_per_ps, sample_every, wall_budget_s, e_min, d_max_ref,
            thermostat=None, tdamp_fs=None, chain_length=None, pin_com=None,
            engine_dir=None, setting="default"):
    """The production segment, flushed in chunks so that it can be resumed.

    `outdir` is the RECORDS folder (frames.npy, progress.json); `engine_dir` the ASE
    engine folder `md_ase/basinNN/` that gets `md.traj` (ASE Trajectory: positions, momenta,
    energy, forces per sampled frame) and `md.log` (ASE MDLogger), the two files ASE
    writes by itself (ADR 0001, ticket 10). A resume continues from the last frame of
    `md.traj`, which carries the momenta; `state.npz` is gone.
    """
    from ase import units
    from ase.io import read as ase_read
    from ase.io.trajectory import Trajectory
    from ase.md import MDLogger
    from openqha.store import layout
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    engine_dir = Path(engine_dir) if engine_dir is not None else outdir
    engine_dir.mkdir(parents=True, exist_ok=True)
    frames_path = outdir / "frames.npy"
    progress_path = outdir / "progress.json"
    traj_path = engine_dir / layout.engine_file_name("md.traj", setting)
    log_path = engine_dir / layout.engine_file_name("md.log", setting)

    n_target = int(round(prod_ps * 1000.0 / timestep_fs / sample_every))
    # THE TRAJECTORY IS md.traj: what is on disk is what ASE wrote there. The float64
    # copy in frames.npy is a record and is rebuilt from md.traj when the two disagree
    # (a kill between the two writes).
    on_disk = ase_read(str(traj_path), index=":") if traj_path.exists() else []
    have = [a.get_positions().copy() for a in on_disk]
    n_have = len(have)
    resumed_from = None
    if n_have >= n_target:
        return np.array(have), dict(resumed=True, complete=True, n_frames=n_have,
                                    n_target=n_target, wall_seconds=0.0,
                                    chunks_this_run=0, n_frames_generated_this_run=0,
                                    resumed_from="md.traj",
                                    engine_files=dict(folder=str(engine_dir), frames_in_traj=n_have))

    # A resumed run CONTINUES the trajectory rather than restarting it: positions and
    # momenta are restored from the last frame ASE wrote. Without this the second segment
    # would begin from a freshly equilibrated state -- statistically defensible, since
    # both segments sample the same canonical distribution, but it would silently discard
    # the correlation across the seam and repeat the equilibration cost every restart.
    #
    # What a restart does NOT reproduce is the Langevin noise sequence, which begins again
    # from the seed, nor a Nose-Hoover chain's own variables (ASE keeps them in the
    # integrator, not in the frame). So a resumed trajectory is not byte-identical to an
    # uninterrupted one, and the record says so.
    if n_have:
        last = on_disk[-1]
        atoms.set_positions(last.get_positions())
        if last.get_momenta() is not None and np.any(last.get_momenta()):
            atoms.set_momenta(last.get_momenta())
        resumed_from = "md.traj"

    n_start = n_have
    rng = np.random.RandomState(int(seed) + 1)
    dyn, _name = make_dynamics(atoms, temperature_K, timestep_fs, friction_per_ps, rng,
                               thermostat, tdamp_fs, chain_length)
    buf, temps = [], []
    traj = Trajectory(str(traj_path), mode="a" if n_have else "w", atoms=atoms)
    logger = MDLogger(dyn, atoms, str(log_path), header=not n_have, stress=False,
                      peratom=False, mode="a" if n_have else "w")

    def grab():
        buf.append(atoms.get_positions().copy())
        temps.append(instantaneous_temperature(atoms))

    # Order matters: the frame (positions, momenta, energy, forces) then the log row, at
    # the same instant, so md.traj and md.log stay row-aligned.
    dyn.attach(traj.write, interval=sample_every)
    dyn.attach(logger, interval=sample_every)
    dyn.attach(grab, interval=sample_every)
    if (PIN_CENTRE_OF_MASS if pin_com is None else pin_com):
        dyn.attach(lambda: recentre(atoms), interval=1)   # see 

    t0 = time.time()
    chunks = 0
    stopped_on_budget = False
    # At least four flushes per run, however short: a single flush at the very end means a
    # killed job loses everything, and a short job is exactly the case where that gets
    # discovered too late to matter.
    chunk_size = max(1, min(CHUNK_FRAMES, (n_target - n_have) // 4 or 1))
    while n_have + len(buf) < n_target:
        want = min(chunk_size, n_target - n_have - len(buf))
        dyn.run(want * sample_every)
        have.extend(buf)
        n_have, buf[:] = len(have), []
        # The engine files first (ASE flushes md.traj per frame; the log per row), the
        # record after them.
        try:
            traj.backend.fd.flush() if hasattr(traj, "backend") else None
        except Exception:                                                 # noqa: BLE001
            pass
        np.save(frames_path, np.array(have, dtype=float))
        chunks += 1
        rep = integrity(atoms, d_max_ref, e_min, temperature_K)
        progress_path.write_text(json.dumps(dict(
            n_frames=n_have, n_target=n_target, integrity=rep,
            wall_seconds=time.time() - t0), indent=2), encoding="utf-8")
        if not rep["ok"]:
            raise RuntimeError("molecular integrity failed after {} frames: {}"
                               .format(n_have, rep))
        if wall_budget_s and (time.time() - t0) > wall_budget_s:
            stopped_on_budget = True
            break

    traj.close()
    logger.logfile.close() if hasattr(logger, "logfile") and hasattr(logger.logfile, "close") else None
    wall = time.time() - t0
    arr = np.array(have, dtype=float)
    generated = int(len(arr) - n_start)
    # Block averaging, because consecutive frames 8 fs apart are not independent and a
    # naive standard error would claim a precision the trajectory does not have. Ten
    # blocks is enough to see whether the mean sits on the target; it is a sanity gate on
    # the thermostat, not one of the branch's acceptance criteria.
    t_arr = np.asarray(temps, dtype=float)
    if len(t_arr) >= 10:
        blocks = np.array([b.mean() for b in np.array_split(t_arr, 10)])
        t_err = float(blocks.std(ddof=1) / np.sqrt(len(blocks)))
    else:
        t_err = None
    return arr, dict(
        resumed=bool(n_start > 0), n_frames_already_on_disk=int(n_start),
        resumed_from=resumed_from,
        engine_files=dict(folder=str(engine_dir), traj=str(traj_path), log=str(log_path),
                          frames_in_traj=int(len(arr))),
        complete=bool(len(arr) >= n_target),
        stopped_on_wall_budget=stopped_on_budget,
        n_frames=int(len(arr)), n_target=int(n_target), chunks_this_run=int(chunks),
        n_frames_generated_this_run=generated,
        wall_seconds=float(wall),
        # Wall clock divided by what THIS run produced. Not divided by anything else:
        # a serial number divided by a process count is not a cost (D0-P1-12, defect 34).
        seconds_per_frame_this_run=(float(wall / generated) if generated else None),
        seconds_per_ps_this_run=(
            float(wall / (generated * sample_every * timestep_fs / 1000.0))
            if generated else None),
        temperature_mean_K=float(t_arr.mean()) if len(t_arr) else None,
        temperature_std_K=float(t_arr.std()) if len(t_arr) else None,
        temperature_block_standard_error_K=t_err,
        temperature_deviation_K=(float(t_arr.mean() - temperature_K)
                                 if len(t_arr) else None),
        temperature_within_three_block_errors=(
            bool(abs(t_arr.mean() - temperature_K) <= 3.0 * t_err)
            if (len(t_arr) and t_err) else None),
        degrees_of_freedom_used_for_temperature=int(3 * len(atoms)),
        fixcm=bool(FIXCM),
        integrity=integrity(atoms, d_max_ref, e_min, temperature_K))


def run_one(atoms, calc, outdir, args, cfg, basin_index, seed, geometry_source,
            engine_dir=None):
    """One basin, one seed: relax, equilibrate, produce, and write the identity metadata.

    `outdir` is the records folder (`_records/md_ase/<setting>/basinNN/`), `engine_dir` the
    engine folder (`md_ase/basinNN/`) that gets `start.extxyz`, `md.traj`, `md.log`.
    """
    from ase.io import write as ase_write
    from openqha.store import layout
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    engine_dir = Path(engine_dir) if engine_dir is not None else outdir
    engine_dir.mkdir(parents=True, exist_ok=True)
    traj_path = engine_dir / layout.engine_file_name("md.traj", args.setting)
    relaxed, relax_rec = relax(atoms, calc, fmax=args.fmax)
    start_path = engine_dir / layout.engine_file_name("start.extxyz", args.setting)
    if not traj_path.exists():
        # The start of THIS trajectory, with the energy and forces at it, as ASE writes
        # them. Not rewritten on a resume: the trajectory did start here.
        relaxed.get_potential_energy(); relaxed.get_forces()
        ase_write(str(start_path), relaxed, format="extxyz")
    e_min = relaxed.get_potential_energy()
    d_max_ref = float(relaxed.get_all_distances().max())

    masses = relaxed.get_masses()
    hydrogen = [m for s, m in zip(relaxed.get_chemical_symbols(), masses) if s == "H"]
    # Equilibration is skipped when there is a state to continue from: the saved momenta
    # are already equilibrated, and re-equilibrating would pay the cost again AND throw
    # away the state it was about to continue.
    is_langevin = args.thermostat.lower() == "langevin"
    thermostat_label = ("ase.md.langevin.Langevin" if is_langevin
                        else "ase.md.nose_hoover_chain.NoseHooverChainNVT")
    # Before either stage runs: this call can REFUSE, and refusing after a trajectory has
    # been produced would be a refusal that arrives too late to mean anything.
    pin_com = pin_centre_of_mass(args.thermostat, args.pin_com)

    resuming = traj_path.exists()
    previous = {}
    if (outdir / "meta.json").exists():
        previous = json.loads((outdir / "meta.json").read_text(encoding="utf-8"))
    if resuming:
        # Carry the original equilibration record forward. It describes how these frames
        # came to be, and a restart that overwrote it with "skipped" would erase the only
        # evidence that the trajectory was ever equilibrated at all.
        equil = previous.get("equilibration", {})
        equil = dict(equil, carried_forward_from_first_run=True,
                     this_run_skipped_equilibration=True)
    else:
        equil = equilibrate(relaxed, calc, args.temperature, seed, args.equil_ps,
                            TIMESTEP_FS, args.friction, args.thermostat,
                            args.tdamp_fs, args.chain_length, pin_com)
    frames, prod = produce(relaxed, calc, outdir, args.temperature, seed, args.prod_ps,
                           TIMESTEP_FS, args.friction, args.sample_every,
                           args.wall_budget_s, e_min, d_max_ref,
                           args.thermostat, args.tdamp_fs, args.chain_length,
                           pin_com, engine_dir=engine_dir, setting=args.setting)

    meta = dict(
        # ---- the identity assertion's inputs. Written here, checked in qha.py. --------
        bias_potential=None,
        constraints=None,
        hydrogen_mass_amu=(float(hydrogen[0]) if hydrogen else qha.HYDROGEN_MASS_AMU),
        timestep_fs=float(TIMESTEP_FS),
        thermostat=thermostat_label,
        # Exactly one of the two blocks below is meaningful, and the other is absent
        # rather than defaulted: `fixcm` is an ASE Langevin option and a Nose-Hoover run
        # that recorded `thermostat_fixcm=False` would carry a field that looks checked
        # and is not. `qha.assert_trajectory_identity` picks which block to enforce from
        # the thermostat name.
        thermostat_fixcm=(bool(FIXCM) if is_langevin else None),
        thermostat_tdamp_fs=(None if is_langevin else float(args.tdamp_fs)),
        thermostat_chain_length=(None if is_langevin else int(args.chain_length)),
        # Under Nose-Hoover this is necessarily False: the integrator overwrites any
        # attempt to re-centre, so the driver refuses the combination outright rather
        # than recording a pinning that did not happen. See make_dynamics.
        centre_of_mass_pinned_to_origin=bool(pin_com),
        neighbour_list_patch=engine.provenance().get("neighbour_list_patch"),
        safe_origin_radius_A=float(engine.SAFE_ORIGIN_RADIUS_A),
        pinning_reason=("insurance: an origin-anchored neighbour search loses pairs once "
                        "the molecule random-walks away, which the MACE develop tree does "
                        "and the installed mace_torch 0.3.16 does not; see openqha.engine."
                        "translation_invariance and the recentre() docstring"),
        source=("openQHA.branchB.langevin" if is_langevin
                else "openQHA.branchB.nose_hoover"),
        # ---- everything else ---------------------------------------------------------
        qm9_index=args.species, basin_index=int(basin_index), seed=int(seed),
        setting=args.setting, route="ase", engine_folder=str(engine_dir),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        geometry_source=geometry_source,
        symbols=list(relaxed.get_chemical_symbols()),
        masses_amu=[float(x) for x in masses],
        temperature_K=float(args.temperature),
        friction_per_ps=float(args.friction),
        sample_every_steps=int(args.sample_every),
        frame_spacing_fs=float(args.sample_every * TIMESTEP_FS),
        ensemble=("canonical (Langevin)" if is_langevin
                  else "canonical (Nose-Hoover chain)"),
        engine=engine.provenance(),
        engine_dtype=engine.DTYPE,
        composite_notation=engine.composite_notation(),
        relaxation=(previous.get("relaxation", relax_rec) if resuming else relax_rec),
        equilibration=equil, production=prod,
        n_runs=int(previous.get("n_runs", 0)) + 1,
        protocol_source=("plan_B section 3; dt and friction from D0-C-30, "
                         "equilibration length measured (Rinaldo & Field needed 26x "
                         "their first guess)"),
        prohibitions_note=("no bias, no constraints, real hydrogen mass -- see the "
                           "module docstring and configs/mdp/s0_qha_production.mdp"),
    )
    # Fail here rather than in the analysis: the metadata is written by this script, so if
    # it cannot pass its own check the trajectory should not reach the disk looking valid.
    qha.assert_trajectory_identity(meta)
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return frames, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # THE PROTOCOL COMES FROM configs/branchB_protocol.yaml. See the OpenMM driver.
    _p = qha.protocol()
    ap.add_argument("--species", default=None, help="QM9 index, e.g. dsgdb9nsd_000018")
    ap.add_argument("--basins", default=None,
                    help="`auto`: branch A's basins under --basin-tag, read from "
                         "<molecule>/mace/basinNN/basin.extxyz (ADR 0001); a path: a "
                         "multi-frame xyz. Either overrides --species for geometry; "
                         "--species still names the record")
    ap.add_argument("--basin-tag", default=None,
                    help="the tag branch A's product is under (default: --tag)")
    ap.add_argument("--tag", default="prod")
    ap.add_argument("--setting", default="default",
                    help="the trajectory setting: md_ase/basinNN/*_<setting>.* in the molecule "
                         "directory of --basin-tag (ADR 0001, ticket 10)")
    ap.add_argument("--molecule-dir", default=None,
                    help="override the molecule directory (default: from S0_RUNS_ROOT, "
                         "--basin-tag or --tag, --species through openqha.store.layout)")
    ap.add_argument("--seeds", type=int, default=3,
                    help="independent trajectories per basin. 3 is the minimum that gives "
                         "a between-seed spread, which is acceptance criterion 5")
    ap.add_argument("--seed0", type=int, default=20260903)
    ap.add_argument("--basin", type=int, default=None, help="run only this basin index")
    ap.add_argument("--seed-index", type=int, default=None,
                    help="run only this seed index. The execution layer fans out one "
                         "task per (basin, seed), and each task must land in its own "
                         "seedNN directory with its own seed -- without this they would "
                         "all be seed 0 and overwrite one another")
    ap.add_argument("--equil-ps", type=float, default=_p["equilibration_ps"])
    ap.add_argument("--prod-ps", type=float, default=_p["production_ps"])
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--friction", type=float, default=FRICTION_PER_PS,
                    help="Langevin friction in 1/ps; ignored under Nose-Hoover")
    ap.add_argument("--thermostat", default=THERMOSTAT,
                    choices=("nose-hoover", "langevin"),
                    help="Nose-Hoover by default. Read the THERMOSTAT constant before "
                         "changing the coupling time: at the conventional 100 fs this "
                         "thermostat returns the softest mode three times too stiff")
    ap.add_argument("--tdamp-fs", type=float, default=NOSE_HOOVER_TDAMP_FS,
                    help="Nose-Hoover coupling time. qha.assert_trajectory_identity "
                         "refuses anything above {} fs".format(NOSE_HOOVER_TDAMP_FS))
    ap.add_argument("--chain-length", type=int, default=NOSE_HOOVER_CHAIN_LENGTH,
                    help="Nose-Hoover chain length; 1 is plain Nose-Hoover and is refused")
    ap.add_argument("--pin-com", dest="pin_com", action="store_true", default=None,
                    help="re-centre the molecule every step. Refused under Nose-Hoover, "
                         "where ASE's integrator overwrites it and it would do nothing")
    ap.add_argument("--no-pin-com", dest="pin_com", action="store_false",
                    help="let the centre of mass random-walk freely")
    ap.add_argument("--sample-every", type=int, default=_p["sample_every_steps"])
    ap.add_argument("--fmax", type=float, default=0.005)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--wall-budget-s", type=float, default=0.0,
                    help="stop at a chunk boundary after this many seconds and write a "
                         "resumable state. 0 means no budget")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # Resolved once, before anything runs, so that a refusal happens here rather than
    # after a trajectory has been produced under a setting the caller did not get.
    pin_com = pin_centre_of_mass(args.thermostat, args.pin_com)

    cfg = config.load()
    if args.temperature is None:
        args.temperature = config.temperature(cfg)

    frames_in, geometry_source = load_atoms(args, cfg)
    from openqha.store import layout
    # The molecule directory is the BASIN TAG's; a run under another --tag is a SETTING
    # inside it (ruling Q8, 2026-09-14). Engine files: md_ase/basinNN/; records:
    # _records/md_ase/<setting>/basinNN/.
    molecule = (Path(args.molecule_dir) if args.molecule_dir
                else layout.molecule_dir(config.runs_root(cfg), args.basin_tag or args.tag,
                                         args.species or "unnamed"))
    root = layout.records_for(molecule, "ase", args.setting)

    print("=" * 92)
    print("Branch B -- quasi-harmonic production trajectories")
    print("=" * 92)
    print("species        {}".format(args.species))
    print("geometries     {}  ({})".format(len(frames_in), geometry_source))
    print("engine         {}".format(engine.engine_name()))
    # Print the coupling that is actually in force. A banner that says "friction" under
    # Nose-Hoover names a parameter the run does not have, and this branch has already
    # spent a session on a product that recorded a request instead of what ran.
    if args.thermostat.lower() == "langevin":
        coupling = "Langevin, friction = {}/ps".format(args.friction)
    else:
        coupling = ("Nose-Hoover chain, tdamp = {} fs, chain = {} "
                    "(refused above {} fs -- see THERMOSTAT)"
                    .format(args.tdamp_fs, args.chain_length,
                            qha.NOSE_HOOVER_MAX_TDAMP_FS))
    print("protocol       dt = {} fs, {}, T = {} K, frame every {} fs"
          .format(TIMESTEP_FS, coupling, args.temperature,
                  args.sample_every * TIMESTEP_FS))
    print("prohibitions   no bias, no constraints, real hydrogen mass")
    print("length         {} ps equilibration + {} ps production x {} seeds"
          .format(args.equil_ps, args.prod_ps, args.seeds))
    print("molecule       {}".format(molecule))
    print("engine         md_ase/basinNN/{}   (ASE's own files; records in _records/md_ase/{}/)".format(
        "" if args.setting == "default" else "  files *_{}.*".format(args.setting), args.setting))
    n_frames = int(round(args.prod_ps * 1000.0 / TIMESTEP_FS / args.sample_every))
    n_dof = 3 * len(frames_in[0]) - 6
    print("frames/traj    {}   against 3N-6 = {}  (rank limit is frames-1)"
          .format(n_frames, n_dof))
    print()
    if args.dry_run:
        return 0

    calc, engine_name, engine_prov = engine.calculator(device=args.device)
    basins = ([(args.basin, frames_in[args.basin])] if args.basin is not None
              else list(enumerate(frames_in)))
    summary = []
    seed_indices = ([args.seed_index] if args.seed_index is not None
                    else list(range(args.seeds)))
    for k, atoms in basins:
        for s in seed_indices:
            # The seed is a pure function of (basin, seed index, seed0), so a task fanned
            # out by the execution layer and the same task run by hand produce the same
            # trajectory. Nothing about placement may enter it.
            seed = args.seed0 + 1000 * k + s
            # No seed level (ruling S0-B-59): one folder per basin. A second seed index,
            # if ever asked for, is a different setting and says so in its file names.
            out = root / "basin{:02d}".format(k)
            eng = layout.ase_dir(molecule, args.setting, k)
            print("-- basin {} seed {} -> {}  (records {})".format(k, s, eng, out))
            t0 = time.time()
            frames, meta = run_one(atoms, calc, out, args, cfg, k, seed, geometry_source,
                                   engine_dir=eng)
            summary.append(dict(
                basin=k, seed=s, path=str(out), n_frames=len(frames),
                complete=meta["production"]["complete"],
                wall_seconds=round(time.time() - t0, 1),
                temperature_mean_K=meta["production"].get("temperature_mean_K"),
                temperature_error_K=meta["production"].get("temperature_deviation_K"),
                equilibration_drift_eV=meta["equilibration"].get("potential_drift_eV")))
            print("   {} frames, complete={}, {:.1f} s, T = {}".format(
                len(frames), meta["production"]["complete"],
                summary[-1]["wall_seconds"], summary[-1]["temperature_mean_K"]))

    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print()
    print("{:<8} {:<6} {:>9} {:>10} {:>12}  {}".format(
        "basin", "seed", "frames", "wall s", "T mean K", "complete"))
    for r in summary:
        print("{:<8} {:<6} {:>9} {:>10.1f} {:>12}  {}".format(
            r["basin"], r["seed"], r["n_frames"], r["wall_seconds"],
            "-" if r["temperature_mean_K"] is None
            else "{:.2f}".format(r["temperature_mean_K"]), r["complete"]))
    print("\nengine files under {}  (setting {})".format(molecule / layout.md_folder("ase"), args.setting))
    print("records under      {}".format(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
