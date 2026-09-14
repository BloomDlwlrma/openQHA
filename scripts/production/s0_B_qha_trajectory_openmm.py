"""Branch B production trajectories, the OpenMM way: MACE + Nose-Hoover chain.

PRODUCTION. It writes the same `frames.npy` + `meta.json` contract as
`s0_B_qha_trajectory.py`, so `s0_B_qha_analyse.py` reads either without knowing which
produced it. That is the point: the two routes are an INDEPENDENT IMPLEMENTATION PAIR of
the same protocol, and their disagreement is a measurement rather than a mystery.

Why a second driver rather than a flag
--------------------------------------
The two run in different environments and neither can import the other's stack:

    openqha   ASE 3.29, MACE 0.3.16 (site-packages), torch 2.13   -- the ASE route
    qm9fe     OpenMM 8.4, openmm-torch, openmmtools, mdtraj,      -- this route
              MACE 0.3.17 (an EDITABLE install pointing at the vendored develop tree)

One (basin, seed) at a time, which is what the execution layer fans out
------------------------------------------------------------------------
    --basins <qid>.basins.xyz   the basin list branch A wrote
    --basin K --seed-index S    ONE task; the velocity seed is DRAWN at run time and
                                recorded in meta.json (seed0=0, the default since the
                                2026-09-13 ruling), or seed0 + 1000*K + S if seed0 != 0
    --wall-budget-s N           stop and flush before the queue kills the job
    --platform CUDA             one trajectory per card (TianheXY-A, ruling 2026-09-07)

Without --basin/--seed-index it still runs every basin and every seed in one process,
which is what you want on a workstation.

Run this one as:

    /home/ubuntu/anaconda3/envs/qm9fe/bin/python \\
        scripts/production/s0_B_qha_trajectory_openmm.py --species dsgdb9nsd_000018 \\
        --tag omm01 --prod-ps 25             # one trajectory per basin (ruling 2026-09-13)

Note the second line of that table. In `qm9fe`, `import mace` resolves to the develop tree
whose neighbour list is NOT translation invariant. This driver does not care, and not by
luck: `openqha/openmm_mace.py` builds the graph itself, as the complete graph, and never
calls MACE's neighbour search at all. `provenance` records that so a reader does not have
to reconstruct it.

The thermostat
--------------
`openmmtools.integrators.NoseHooverChainVelocityVerletIntegrator`, at ITS OWN documented
defaults:

    collision_frequency = 50/ps      (tdamp = 20 fs)
    chain_length        = 5
    num_mts             = 5
    num_yoshidasuzuki   = 5

Those are not this repository's choices. The same 50 ps^-1 appears in the QHA reference
paper this branch already cites (Biophysical Journal; "a friction coefficient of 50 ps-1
was employed for each atom", and "a piston collision frequency of 50 ps-1"), and 20 fs is
independently what the closed-form calibration selected -- see
`scripts/calibration/s0_B_thermostat_choice.py` and `qha.NOSE_HOOVER_MAX_TDAMP_FS`.

What this route cannot do
-------------------------
Pin the centre of mass. OpenMM's `CMMotionRemover` removes translation but not rotation,
and neither is needed: the potential is translation invariant and the quasi-harmonic
analysis superimposes every frame anyway. The centre of mass is left to random-walk and
the drift is recorded, so it can be checked rather than assumed harmless.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))

from openqha import config, engine, openmm_mace, qha  # noqa: E402
from openqha.store import layout  # noqa: E402

# The production protocol, from configs/branchB_protocol.yaml -- the SAME file the ASE
# route reads. These module constants are the fallback and are kept equal to it; `main()`
# overrides them from the config so the two routes cannot drift apart again. Until
# 2026-09-07 this driver ran 2 + 25 ps while the ASE one ran 50 + 200, which made the two
# "independent implementations" implementations of different protocols.
TIMESTEP_FS = 1.0
SAMPLE_EVERY_STEPS = 500       # 0.5 ps at 1 fs -- Rinaldo & Field 2003 p2
#: openmmtools' own defaults, spelled out. See the module docstring.
COLLISION_FREQUENCY_PER_PS = 50.0
CHAIN_LENGTH = 5
NUM_MTS = 5
NUM_YOSHIDA_SUZUKI = 5
EQUIL_PS = 520.0               # Rinaldo & Field 2003 p2; 20 ps was tried and rejected
PROD_PS = 1500.0               # Rinaldo & Field 2003 p2 -- 1.5 ns, giving 3000 frames
CHUNK_FRAMES = 250             # 125 ps per flush at 0.5 ps spacing


def tdamp_fs(collision_frequency_per_ps):
    """The coupling time the identity assertion checks, from the frequency OpenMM takes."""
    return 1000.0 / float(collision_frequency_per_ps)


def build_integrator(temperature_K, timestep_fs, collision_per_ps, chain_length,
                     num_mts, num_ys, system=None):
    """openmmtools' Nose-Hoover chain, with a plain-OpenMM fallback that is RECORDED.

    The two are not the same algorithm to the last digit -- openmmtools defaults to a
    5-term Yoshida-Suzuki decomposition and OpenMM's own integrator to 7 -- so which one
    ran goes into the product rather than being treated as interchangeable.
    """
    from openmm import unit
    dt = timestep_fs * unit.femtosecond
    try:
        from openmmtools.integrators import NoseHooverChainVelocityVerletIntegrator
        integ = NoseHooverChainVelocityVerletIntegrator(
            system=system, temperature=temperature_K * unit.kelvin,
            collision_frequency=collision_per_ps / unit.picosecond,
            timestep=dt, chain_length=int(chain_length), num_mts=int(num_mts),
            num_yoshidasuzuki=int(num_ys))
        return integ, dict(
            implementation="openmmtools.integrators."
                           "NoseHooverChainVelocityVerletIntegrator",
            collision_frequency_per_ps=float(collision_per_ps),
            chain_length=int(chain_length), num_mts=int(num_mts),
            num_yoshidasuzuki=int(num_ys))
    except Exception as exc:                                  # noqa: BLE001
        import openmm
        integ = openmm.NoseHooverIntegrator(
            temperature_K * unit.kelvin, collision_per_ps / unit.picosecond, dt,
            int(chain_length), int(num_mts), 7)
        return integ, dict(
            implementation="openmm.NoseHooverIntegrator",
            fallback_reason="{}: {}".format(type(exc).__name__, exc),
            collision_frequency_per_ps=float(collision_per_ps),
            chain_length=int(chain_length), num_mts=int(num_mts),
            num_yoshidasuzuki=7)


def relax(atoms, model_path):
    """Minimise with the SAME force the dynamics will use, through OpenMM."""
    import openmm
    from openmm import unit
    # The platform here and the platform passed to build_system MUST agree: the traced
    # module carries its device as a constant. This one minimises on the CPU by design.
    system, force_record = openmm_mace.build_system(
        atoms.get_atomic_numbers(), atoms.get_masses(), model_path,
        example_positions_nm=atoms.get_positions() / openmm_mace.NM_TO_A,
        platform="CPU")
    integ = openmm.VerletIntegrator(1.0 * unit.femtosecond)
    context = openmm.Context(system, integ, openmm.Platform.getPlatformByName("CPU"))
    context.setPositions((atoms.get_positions() / openmm_mace.NM_TO_A) * unit.nanometer)
    e0 = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole)
    openmm.LocalEnergyMinimizer.minimize(context, 1e-4, 2000)
    state = context.getState(getPositions=True, getEnergy=True)
    positions = state.getPositions(asNumpy=True).value_in_unit(
        unit.nanometer) * openmm_mace.NM_TO_A
    e1 = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    return positions, dict(
        energy_before_kJ=float(e0), energy_after_kJ=float(e1),
        energy_drop_kcal=float((e0 - e1) / 4.184), force_field=force_record)


def flush_frames(frames_path, frames):
    """Write `frames` to `frames_path` atomically: temp file first, then rename.

    A job killed mid-write must leave the previous complete file, not a truncated array
    that reads as corrupt. So the write goes to a sibling and is renamed into place.

    **The sibling must end in `.npy`.** `np.save(path, arr)` appends `.npy` to any name
    that does not already carry it, so the first version of this --

        tmp = frames_path.with_suffix(frames_path.suffix + ".part")    # frames.npy.part
        np.save(tmp, arr)                                              # -> frames.npy.part.npy
        tmp.replace(frames_path)                                       # FileNotFoundError

    -- failed on the FIRST flush of every trajectory, once any trajectory reached a flush
    at all (an104, 2026-09-13: six of six, `frames.npy.part.npy` left beside each). No GPU
    trajectory had got that far before; the code had never executed. Writing through an
    open file handle sidesteps the suffix rule entirely, and the name is `.part.npy` so a
    leftover from a killed job is still recognisable as a partial.
    """
    frames_path = Path(frames_path)
    tmp = frames_path.with_name(frames_path.stem + ".part.npy")
    with open(tmp, "wb") as fh:
        np.save(fh, np.asarray(frames))
    tmp.replace(frames_path)
    return frames_path


def already_complete(outdir, n_target, engine_dir=None, setting="default"):
    """The trajectory's own record, if this (basin, seed) is finished; else None.

    Finished means: meta.json is there, its production block says complete, and
    frames.npy holds at least n_target frames. Anything less -- a partial run, a killed
    job, an older protocol with fewer frames -- is None and the driver resumes it.

    Why this exists (an113, 2026-09-13). The chain re-ran a task whose six trajectories
    were already complete. run_one() loaded the frames, saw nothing left to generate,
    skipped the production loop -- and so never measured a temperature; the summary
    line then formatted None with `{:7.2f}` and every task exited 1. Before reaching that
    line each one had spent ~70 s re-tracing the model, re-relaxing and re-equilibrating
    a trajectory it was about to do nothing with. The resume path had never executed.
    """
    outdir = Path(outdir)
    meta_p, frames_p = outdir / "meta.json", outdir / "frames.npy"
    if not meta_p.exists():
        return None
    try:
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        if engine_dir is not None:
            # Since 2026-09-14 the trajectory IS traj.dcd (ADR 0001); the float64 copy in
            # the record is not what decides whether a basin is finished.
            from openqha.quasi_harmonic import openmm_files
            n = openmm_files.dcd_frame_count(
                Path(engine_dir) / layout.openmm_file_name("traj.dcd", setting))
        elif frames_p.exists():
            n = int(np.load(frames_p, mmap_mode="r").shape[0])
        else:
            return None
    except Exception:                                                 # noqa: BLE001
        return None
    prod = meta.get("production") or {}
    if prod.get("complete") and n >= int(n_target):
        return meta
    return None


def _fmt(x, spec):
    """Format a number, or '-' for a None the record legitimately carries."""
    return format(x, spec) if x is not None else "-"


def choose_seed(seed0, basin, seed_index):
    """(seed, how it was chosen) for one trajectory.

    seed0 == 0 (the default, ruling 2026-09-13): a fresh 31-bit seed from the OS entropy
    source, the way OpenMM itself treats randomNumberSeed=0 -- but returned, so that the
    caller records it. seed0 != 0: the pure function seed0 + 1000*basin + seed_index, so
    a task fanned out by the execution layer and the same task run by hand coincide.
    """
    if int(seed0) == 0:
        import secrets
        return (secrets.randbits(31) or 1), "drawn at run time (seed0=0; recorded here)"
    return int(seed0) + 1000 * int(basin) + int(seed_index), "seed0 + 1000*basin + seed_index"


def _read_meta(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def segments_record(prev_meta, seed, prod_record):
    """Which frames of frames.npy came from which velocity draw.

    A resume (frames on disk but fewer than the target) re-equilibrates from the relaxed
    geometry with a new draw and appends; the file then holds two independently started
    segments. With a fixed seed the second segment even re-traced the first's opening
    steps. This list is the record of that; a trajectory run in one go has one entry.
    """
    segs = []
    if prev_meta:
        segs = list(prev_meta.get("segments") or [])
        if not segs and "seed" in prev_meta:
            segs = [dict(seed=int(prev_meta["seed"]), frames_from=0,
                         frames_to=int((prev_meta.get("production") or {}).get("n_frames", 0)))]
    segs.append(dict(seed=int(seed),
                     frames_from=int(prod_record.get("n_frames_already_on_disk", 0)),
                     frames_to=int(prod_record.get("n_frames", 0))))
    return segs


def run_one(species, positions_A, numbers, masses, model_path, engine_dir, records_dir,
            temperature_K, seed, equil_ps, prod_ps, args, prev_meta=None):
    """One trajectory. Engine files to `engine_dir`, this driver's record beside them
    in `records_dir` (ADR 0001).

    A partial trajectory -- state files present, fewer frames than the target -- is
    RESUMED from its state: same velocities, same thermostat variables, no second
    equilibration and no second velocity draw. Until 2026-09-14 a resume restarted
    from the relaxed geometry with a new draw and appended, which is what the
    `segments` record was for; it is kept, and now shows one seed.
    """
    import openmm
    from openmm import unit
    from openqha.quasi_harmonic import openmm_files

    records_dir.mkdir(parents=True, exist_ok=True)
    frames_path = records_dir / "frames.npy"
    # Leftovers of the suffix bug described in flush_frames(): never a resumable file.
    stale = records_dir / "frames.npy.part.npy"
    if stale.exists():
        stale.unlink()
    n_target = int(round(prod_ps * 1000.0 / args.timestep_fs)) // args.sample_every
    topology = openmm_files.topology_for(numbers)
    folder = openmm_files.EngineFolder(
        engine_dir, topology, frame_spacing_ps=args.sample_every * args.timestep_fs / 1000.0,
        setting=args.setting)
    n_on_disk = folder.frames_on_disk()
    resume = bool(n_on_disk > 0 and folder.has_state())
    prev_prod = (prev_meta or {}).get("production") or {}
    equilibrated = bool(folder.has_state()
                        and (n_on_disk > 0 or (records_dir / "equilibrated.json").exists()))
    # The float64 frames in the record must agree with the DCD; a record shorter than the
    # DCD (a kill between the two writes) is rebuilt from what the DCD holds.
    have = list(np.load(frames_path)) if (resume and frames_path.exists()) else []
    if resume and len(have) != n_on_disk:
        have = have[:n_on_disk] if len(have) > n_on_disk else have
        if len(have) < n_on_disk:
            import mdtraj as md
            t = md.load(str(folder.paths["traj.dcd"]), top=str(folder.paths["start.pdb"]))
            have = list(t.xyz.astype(np.float64) * openmm_mace.NM_TO_A)[:n_on_disk]
    if not resume:
        n_on_disk = 0

    # **Before** the three minutes of torch import and model tracing: can this driver
    # JIT the PTX nvrtc will emit? OpenMM compiles every kernel at run time, and CUDA
    # minor version compatibility does not cover PTX -- see openqha/gpu_preflight.py.
    # On 2026-09-12 all twelve trajectories of three jobs died at openmm.Context() with
    # CUDA_ERROR_UNSUPPORTED_PTX_VERSION, each after paying that three minutes first.
    from openqha import gpu_preflight
    gpu_preflight.check(args.platform)

    system, force_record = openmm_mace.build_system(
        numbers, masses, model_path,
        example_positions_nm=positions_A / openmm_mace.NM_TO_A,
        platform=args.platform)
    integ, thermo_record = build_integrator(
        temperature_K, args.timestep_fs, args.collision_frequency,
        args.chain_length, args.num_mts, args.num_ys, system=system)
    try:
        # **The properties are not optional on CUDA.** Without them the platform runs
        # at its default `single` while this module is float64, which measured 74.2
        # ms/step against 39.2 for the matched pair on the same A800 -- 1.9x, for
        # nothing. See openmm_mace.platform_properties_for().
        context = openmm.Context(system, integ,
                                 openmm.Platform.getPlatformByName(args.platform),
                                 openmm_mace.platform_properties_for(args.platform))
    except Exception as exc:                                        # noqa: BLE001
        # The preflight above should have caught this; if it did not, say what the
        # numbers were rather than leaving a bare OpenMM error code in a driver.log.
        if "PTX" in str(exc) or "CUDA_ERROR" in str(exc):
            raise RuntimeError(
                "{}\n  openqha/gpu_preflight.py read: {}\n"
                "  (it did not refuse, so either the versions look compatible and the\n"
                "   problem is elsewhere, or it could not read them on this node.)"
                .format(exc, gpu_preflight.describe()))
        raise
    kb_kj = 0.008314462618153241            # kJ/(mol K)
    n_dof = 3 * len(masses)

    resumed_from = None
    if equilibrated:
        # Same System, same integrator class and parameters as the run that wrote the
        # state; positions, velocities and the thermostat's own variables come back
        # from the file, so the trajectory continues rather than restarts.
        resumed_from = folder.load_state(context)
    if resumed_from is None:
        equilibrated = False
        context.setPositions((positions_A / openmm_mace.NM_TO_A) * unit.nanometer)
        context.setVelocitiesToTemperature(temperature_K * unit.kelvin, int(seed))
        # The start of THIS trajectory, in OpenMM's own files, before a single step.
        folder.write_start(positions_A, system, integ)

    # ---- equilibration, with the relaxation measured rather than assumed ------------
    t0 = time.time()
    if equilibrated:
        equil_record = dict((prev_meta or {}).get("equilibration")
                            or _read_meta(records_dir / "equilibrated.json") or {})
        equil_record["skipped_on_resume"] = True
    else:
        n_equil = int(round(equil_ps * 1000.0 / args.timestep_fs))
        trace = []
        step = max(1, n_equil // 20)
        done = 0
        while done < n_equil:
            integ.step(min(step, n_equil - done))
            done += min(step, n_equil - done)
            st = context.getState(getEnergy=True)
            trace.append((float(st.getPotentialEnergy().value_in_unit(
                              unit.kilojoule_per_mole)),
                          float(2.0 * st.getKineticEnergy().value_in_unit(
                              unit.kilojoule_per_mole) / (n_dof * kb_kj))))
        half = len(trace) // 2 or 1
        equil_record = dict(
            ps=float(equil_ps), wall_seconds=float(time.time() - t0),
            potential_first_half_kJ=float(np.mean([t[0] for t in trace[:half]])),
            potential_second_half_kJ=float(np.mean([t[0] for t in trace[half:]])),
            temperature_second_half_K=float(np.mean([t[1] for t in trace[half:]])),
            drift_kJ=float(np.mean([t[0] for t in trace[half:]])
                           - np.mean([t[0] for t in trace[:half]])))
        # The equilibrated state is worth a flush of its own: at 520 ps it is hours of
        # work, and a job killed during production would otherwise redo it.
        folder.flush_state(context)
        (records_dir / "equilibrated.json").write_text(
            json.dumps(equil_record, indent=2), encoding="utf-8")

    # ---- production, flushed in chunks so a killed run is resumable -----------------
    t0 = time.time()
    n_start = len(have)
    temps, com_drift = [], []
    com0 = None
    stopped_on_budget = False
    budget = float(getattr(args, "wall_budget_s", 0.0) or 0.0)
    # At least four flushes per run, however short. A single flush at the very end means
    # a job killed at its walltime loses everything it computed, and a short job is
    # exactly where that gets discovered too late to matter.
    chunk = max(1, min(CHUNK_FRAMES, (n_target - len(have)) // 4 or 1))
    folder.open_frames(append=bool(n_start > 0))
    try:
        while len(have) < n_target:
            want = min(chunk, n_target - len(have))
            for _ in range(want):
                integ.step(args.sample_every)
                st = context.getState(getPositions=True, getEnergy=True)
                p = st.getPositions(asNumpy=True).value_in_unit(
                    unit.nanometer) * openmm_mace.NM_TO_A
                pe = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                ke = st.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
                temp = 2.0 * ke / (n_dof * kb_kj)
                have.append(p)
                temps.append(temp)
                folder.write_frame(p, st.getStepCount(),
                                   st.getTime().value_in_unit(unit.picosecond), pe, ke, temp)
                com = (np.asarray(masses)[:, None] * p).sum(0) / np.sum(masses)
                if com0 is None:
                    com0 = com
                com_drift.append(float(np.linalg.norm(com - com0)))
            # The engine files first (they are the trajectory), the state second (it is
            # what a resume continues from), the record last.
            folder.flush_frames()
            folder.flush_state(context)
            flush_frames(frames_path, have)     # atomic; see the function for the history
            if budget and (time.time() - t0) > budget:
                stopped_on_budget = True
                break
    finally:
        folder.close_frames()
    wall = time.time() - t0
    generated = int(len(have) - n_start)

    prod_record = dict(
        ps=float(prod_ps), n_frames=int(len(have)), n_target=int(n_target),
        wall_seconds=float(wall),
        # `complete` is what a resume and the analysis both ask. Without it a run stopped
        # by its wall budget is indistinguishable from one that finished.
        complete=bool(len(have) >= n_target),
        stopped_on_wall_budget=bool(stopped_on_budget),
        resumed=bool(n_start > 0), n_frames_already_on_disk=int(n_start),
        n_frames_generated_this_run=generated,
        resumed_from=resumed_from,
        engine_files=dict(folder=str(engine_dir), frames_in_dcd=folder.frames_on_disk()),
        # THE SAME KEY NAMES THE ASE ROUTE USES. The two drivers are an implementation
        # pair only if the same reader can read both; `seconds_per_ps` and
        # `temperature_error_K` were names nothing downstream looked for.
        seconds_per_ps_this_run=(
            float(wall / (generated * args.sample_every * args.timestep_fs / 1000.0))
            if generated else None),
        seconds_per_frame_this_run=(float(wall / generated) if generated else None),
        seconds_per_ps=float(wall / max(1e-9, prod_ps)),
        temperature_mean_K=float(np.mean(temps)) if temps else prev_prod.get("temperature_mean_K"),
        temperature_deviation_K=(float(np.mean(temps) - temperature_K)
                                 if temps else prev_prod.get("temperature_deviation_K")),
        temperature_error_K=(float(np.mean(temps) - temperature_K)
                             if temps else prev_prod.get("temperature_error_K")),
        centre_of_mass_drift_A=float(np.max(com_drift)) if com_drift else 0.0)
    prod_record["files"] = {n: str(p) for n, p in folder.paths.items()}
    return np.array(have), equil_record, prod_record, force_record, thermo_record


def main():
    # THE PROTOCOL COMES FROM configs/branchB_protocol.yaml, not from this file. The
    # module constants above are the fallback for a checkout without it; if both exist and
    # disagree, the config wins and `--help` shows the config's value.
    _p = qha.protocol()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species", default="dsgdb9nsd_000018")
    ap.add_argument("--tag", default="omm")
    ap.add_argument("--basins", default=None,
                    help="`auto`: branch A's basins under --basin-tag, read from "
                         "<molecule>/mace/basinNN/basin.extxyz (ADR 0001); a path: a "
                         "multi-frame xyz to use instead. Without it there is ONE "
                         "geometry -- the QM9 reference -- and it is not a basin list; "
                         "the record says so.")
    ap.add_argument("--basin-tag", default=None,
                    help="the tag branch A's product is under (default: --tag)")
    ap.add_argument("--basin", type=int, default=None,
                    help="run only this basin index. This and --seed-index are what "
                         "make one (basin, seed) addressable by the execution layer.")
    ap.add_argument("--seed-index", type=int, default=None,
                    help="run only this seed index")
    # ONE trajectory per (molecule, basin) -- user ruling 2026-09-13, after the campaign
    # arithmetic in docs/branchB_seeds_and_length.md. --seeds 2+ is still accepted (the
    # blank control, criterion 5, needs it) but is no longer the default anywhere.
    ap.add_argument("--seeds", type=int, default=1)
    # 0 = draw a fresh seed per trajectory at run time and RECORD it -- OpenMM's own
    # convention for randomNumberSeed=0 ("a unique seed is chosen when a Context is
    # created"), except that the number chosen is written to meta.json so the run stays
    # re-derivable. Non-zero = the old pure function seed0 + 1000*basin + seed_index.
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--wall-budget-s", type=float, default=0.0,
                    help="stop production and flush after this many seconds. Set it "
                         "below the queue walltime so a task stops itself instead of "
                         "being killed between a write and a rename.")
    ap.add_argument("--equil-ps", type=float, default=_p["equilibration_ps"])
    ap.add_argument("--prod-ps", type=float, default=_p["production_ps"])
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--timestep-fs", type=float, default=_p["timestep_fs"])
    ap.add_argument("--sample-every", type=int, default=_p["sample_every_steps"],
                    help="steps between frames. Default is derived from the config's\n"
                         "sampling_interval_ps, never stored twice.")
    ap.add_argument("--collision-frequency", type=float,
                    default=COLLISION_FREQUENCY_PER_PS,
                    help="1/ps. openmmtools' default is 50; the identity assertion "
                         "refuses a coupling time above {} fs, i.e. a frequency below "
                         "{}/ps".format(qha.NOSE_HOOVER_MAX_TDAMP_FS,
                                        1000.0 / qha.NOSE_HOOVER_MAX_TDAMP_FS))
    ap.add_argument("--chain-length", type=int, default=CHAIN_LENGTH)
    ap.add_argument("--num-mts", type=int, default=NUM_MTS)
    ap.add_argument("--num-ys", type=int, default=NUM_YOSHIDA_SUZUKI)
    ap.add_argument("--platform", default="CPU")
    # WHERE (ADR 0001, 2026-09-14): engine files under
    #   <root>/<tag>/<range>/<chunk>/<qid>/openmm/<setting>/basinNN/
    # and this driver's record under <...>/<qid>/_records/openmm/<setting>/basinNN/.
    # The setting is `default` for the chains and the row name for examples/02d-2.
    ap.add_argument("--setting", default="default",
                    help="the trajectory setting this run belongs to (openmm/<setting>/)")
    ap.add_argument("--molecule-dir", default=None,
                    help="override the molecule directory (default: from S0_RUNS_ROOT, "
                         "--tag and --species through openqha.store.layout)")
    ap.add_argument("--verify-only", action="store_true",
                    help="check the OpenMM force against the ASE calculator and stop")
    args = ap.parse_args()

    cfg = config.load()
    temperature = args.temperature or config.temperature(cfg)
    model_path = engine.model_path()

    from ase.io import read
    if args.basins == "auto":
        from openqha.store import basins as _basins
        _btag = args.basin_tag or args.tag
        frames_in = _basins.read_basins(args.species, tag=_btag, cfg=cfg)
        if not frames_in:
            raise SystemExit(_basins.missing_message(args.species, _btag, cfg=cfg))
        geometry_source = "branch A basins: {}".format(
            _basins.molecule_for(args.species, _btag, cfg) / "mace")
    elif args.basins:
        frames_in = read(args.basins, index=":")
        geometry_source = "branch A basins: {}".format(args.basins)
    else:
        frames_in = [read(str(config.qm9_xyz(args.species, cfg)))]
        geometry_source = (
            "QM9 reference geometry -- branch A has NOT been run for this species, so "
            "this is one geometry and not a basin list")
    atoms = frames_in[0]

    print("=" * 92)
    print("Branch B production -- OpenMM route (MACE + Nose-Hoover chain)")
    print("=" * 92)
    print("species     {}   {} atoms".format(args.species, len(atoms)))
    print("engine      {}".format(engine.engine_name()))

    # ---- what is already done? decided before any trace, relax or equilibration -------
    # config.runs_dir honours S0_RUNS_ROOT, which on a cluster points at node-local
    # scratch. The previous default wrote straight into $HOME -- on Tianhe that is the
    # 100 GB quota'd home on Lustre, which is the one place the site manual asks you not
    # to put job output.
    # The molecule directory is the BASIN TAG's (the branch A product's); a run under
    # another --tag is a different SETTING inside it, not a second molecule directory
    # (ruling Q8, 2026-09-14: one search, one set of basins, many samplings).
    molecule = (Path(args.molecule_dir) if args.molecule_dir
                else layout.molecule_dir(config.runs_root(cfg), args.basin_tag or args.tag,
                                         args.species))
    records_root = layout.openmm_records_dir(molecule, args.setting)
    outroot = records_root                     # where summary.json and the records go
    basins = ([(args.basin, frames_in[args.basin])] if args.basin is not None
              else list(enumerate(frames_in)))
    seed_indices = ([args.seed_index] if args.seed_index is not None
                    else list(range(args.seeds)))
    n_target = int(round(args.prod_ps * 1000.0 / args.timestep_fs)) // args.sample_every

    def _engine_dir(b):
        return layout.openmm_dir(molecule, args.setting, b)

    def _records_dir(b):
        return records_root / "basin{:02d}".format(b)

    def _summary_from(meta, b, k):
        prod = meta.get("production") or {}
        return dict(basin=int(b), seed_index=int(k), seed=int(meta.get("seed", -1)),
                    n_frames=int(prod.get("n_frames") or 0),
                    complete=prod.get("complete"),
                    stopped_on_wall_budget=prod.get("stopped_on_wall_budget"),
                    wall_seconds=prod.get("wall_seconds"),
                    temperature_mean_K=prod.get("temperature_mean_K"),
                    com_drift_A=prod.get("centre_of_mass_drift_A"))

    done = {(b, k): already_complete(_records_dir(b), n_target, engine_dir=_engine_dir(b),
                                     setting=args.setting)
            for b, _g in basins for k in seed_indices}
    done = {bk: m for bk, m in done.items() if m is not None}
    if done:
        print("resume      {} of {} trajectories already complete under {}".format(
            len(done), len(basins) * len(seed_indices), outroot))
    if len(done) == len(basins) * len(seed_indices) and not args.verify_only:
        # Nothing to compute: report what is on disk and stop, without loading a model.
        summary = [_summary_from(m, b, k) for (b, k), m in sorted(done.items())]
        for s in summary:
            print("  basin {:>2} seed {:>2}  {:>5} frames  complete={}  (from meta.json)"
                  .format(s["basin"], s["seed_index"], s["n_frames"], s["complete"]))
        outroot.mkdir(parents=True, exist_ok=True)
        (outroot / "summary.json").write_text(json.dumps(summary, indent=2),
                                              encoding="utf-8")
        print()
        print("nothing to run; {} complete trajectories under {}".format(len(summary), outroot))
        return 0

    check = openmm_mace.verify_against_ase(atoms, model_path)
    print("force check dE = {:.3e} eV   max|dF| = {:.3e} eV/A   traced vs eager "
          "{:.3e}   agrees = {}".format(
              check["delta_energy_eV"], check["max_delta_force_eV_per_A"],
              check["max_delta_force_traced_eV_per_A"], check["agrees"]))
    if not check["agrees"]:
        raise SystemExit(
            "the OpenMM force does not reproduce the ASE calculator. Refusing to produce "
            "a trajectory on a potential that is not the one this branch has calibrated.")
    if args.verify_only:
        return 0

    print("thermostat  Nose-Hoover chain, collision {}/ps (tdamp {:.1f} fs), chain {}, "
          "MTS {}, YS {}".format(args.collision_frequency,
                                 tdamp_fs(args.collision_frequency),
                                 args.chain_length, args.num_mts, args.num_ys))
    print("protocol    dt = {} fs, frame every {} fs, T = {} K".format(
        args.timestep_fs, args.sample_every * args.timestep_fs, temperature))
    print("length      {} ps equilibration + {} ps production x {} seeds".format(
        args.equil_ps, args.prod_ps,
        1 if args.seed_index is not None else args.seeds))
    print("platform    {}{}".format(
        args.platform,
        "   (CUDA: one trajectory per card -- see hpc/resource_configs/tianhe_a.py)"
        if args.platform.upper() == "CUDA" else ""))

    print("geometry    {}".format(geometry_source))
    print("molecule    {}".format(molecule))
    print("engine      openmm/basinNN/{}   (OpenMM's own files; records in _records/openmm/{}/)"
          .format("" if args.setting == "default" else "  files *_{}.*".format(args.setting),
                  args.setting))
    print()

    summary = []
    for b, geom in basins:
        pending = [k for k in seed_indices if (b, k) not in done]
        for k in seed_indices:
            if (b, k) in done:
                summary.append(_summary_from(done[(b, k)], b, k))
                print("  basin {:>2} seed {:>2}  already complete, kept as is".format(b, k))
        if not pending:
            continue                      # every seed of this basin is done: no relax
        relaxed_A, relax_record = relax(geom, model_path)
        print("basin {:>2}    relaxed, dropped {:.4f} kcal/mol".format(
            b, relax_record["energy_drop_kcal"]))
        for k in pending:
            outdir = _records_dir(b)
            engine_dir = _engine_dir(b)
            # A partial trajectory is resumed from its state (same velocities, same
            # thermostat variables), so it keeps the seed it was started with; a fresh
            # one draws. `segments` records every run over the trajectory either way.
            prev_meta = _read_meta(outdir / "meta.json")
            if prev_meta and "seed" in prev_meta and engine_dir.exists():
                seed, seed_formula = int(prev_meta["seed"]), "kept on resume (from the record)"
            else:
                seed, seed_formula = choose_seed(args.seed0, b, k)
            frames, equil, prod, force_record, thermo = run_one(
                args.species, relaxed_A, geom.get_atomic_numbers(), geom.get_masses(),
                model_path, engine_dir, outdir, temperature, seed, args.equil_ps,
                args.prod_ps, args, prev_meta=prev_meta)
            folder_paths = prod.get("files") or {}

            meta = dict(
                # ---- the identity assertion's inputs ----------------------------------
                bias_potential=None,
                constraints=None,
                hydrogen_mass_amu=float(next(
                    (m for m, z in zip(geom.get_masses(), geom.get_atomic_numbers())
                     if int(z) == 1), qha.HYDROGEN_MASS_AMU)),
                timestep_fs=float(args.timestep_fs),
                thermostat="openmmtools Nose-Hoover chain ({})".format(
                    thermo["implementation"]),
                thermostat_tdamp_fs=float(tdamp_fs(args.collision_frequency)),
                thermostat_chain_length=int(args.chain_length),
                source="openQHA.branchB.openmm_nose_hoover",
                # ---- everything else --------------------------------------------------
                qm9_index=args.species, basin_index=int(b), seed=int(seed),
                setting=args.setting, engine_folder=str(engine_dir),
                engine_files={n: str(p) for n, p in folder_paths.items()},
                slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                seed_formula=seed_formula,
                segments=segments_record(prev_meta, seed, prod),
                geometry_source=geometry_source,
                symbols=list(geom.get_chemical_symbols()),
                masses_amu=[float(x) for x in geom.get_masses()],
                temperature_K=float(temperature),
                sample_every_steps=int(args.sample_every),
                frame_spacing_fs=float(args.sample_every * args.timestep_fs),
                ensemble="canonical (Nose-Hoover chain, OpenMM)",
                integrator=thermo,
                route="openmm",
                platform=args.platform,
                protocol_source=_p.get("_source"),
                protocol_status=_p.get("_status"),
                force=force_record,
                force_check_against_ase=check,
                centre_of_mass_pinned_to_origin=False,
                engine=engine.provenance(),
                relaxation=relax_record,
                equilibration=equil, production=prod)
            qha.assert_trajectory_identity(meta)
            (outdir / "meta.json").write_text(
                json.dumps(meta, indent=2, default=str), encoding="utf-8")
            summary.append(dict(basin=int(b), seed_index=int(k), seed=int(seed),
                                n_frames=int(len(frames)),
                                complete=prod["complete"],
                                stopped_on_wall_budget=prod["stopped_on_wall_budget"],
                                wall_seconds=prod["wall_seconds"],
                                temperature_mean_K=prod["temperature_mean_K"],
                                com_drift_A=prod["centre_of_mass_drift_A"]))
            print("  basin {:>2} seed {:>2}  {:>5} frames  complete={}  {:>8} s  "
                  "T = {:>7} K  COM drift {:>8} A".format(
                      b, k, len(frames), prod["complete"],
                      _fmt(prod.get("wall_seconds"), "8.1f"),
                      _fmt(prod.get("temperature_mean_K"), "7.2f"),
                      _fmt(prod.get("centre_of_mass_drift_A"), "8.2f")))

    (outroot / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print()
    print("engine files under {}  (setting {})".format(molecule / "openmm", args.setting))
    print("records under      {}".format(outroot))
    print("analyse with: python scripts/production/s0_B_qha_analyse.py --species {} "
          "--tag {}".format(args.species, args.tag))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
