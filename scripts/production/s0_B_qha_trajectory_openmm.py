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
    --basin K --seed-index S    ONE task; the seed is seed0 + 1000*K + S
    --wall-budget-s N           stop and flush before the queue kills the job
    --platform CUDA             one trajectory per card (TianheXY-A, ruling 2026-09-07)

Without --basin/--seed-index it still runs every basin and every seed in one process,
which is what you want on a workstation.

Run this one as:

    /home/ubuntu/anaconda3/envs/qm9fe/bin/python \\
        scripts/production/s0_B_qha_trajectory_openmm.py --species dsgdb9nsd_000018 \\
        --tag omm01 --seeds 3 --prod-ps 25

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
    system, force_record = openmm_mace.build_system(
        atoms.get_atomic_numbers(), atoms.get_masses(), model_path,
        example_positions_nm=atoms.get_positions() / openmm_mace.NM_TO_A)
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


def run_one(species, positions_A, numbers, masses, model_path, outdir, temperature_K,
            seed, equil_ps, prod_ps, args):
    import openmm
    from openmm import unit

    outdir.mkdir(parents=True, exist_ok=True)
    frames_path = outdir / "frames.npy"
    have = list(np.load(frames_path)) if frames_path.exists() else []
    n_target = int(round(prod_ps * 1000.0 / args.timestep_fs)) // args.sample_every

    system, force_record = openmm_mace.build_system(
        numbers, masses, model_path,
        example_positions_nm=positions_A / openmm_mace.NM_TO_A)
    integ, thermo_record = build_integrator(
        temperature_K, args.timestep_fs, args.collision_frequency,
        args.chain_length, args.num_mts, args.num_ys, system=system)
    context = openmm.Context(system, integ,
                             openmm.Platform.getPlatformByName(args.platform))
    context.setPositions((positions_A / openmm_mace.NM_TO_A) * unit.nanometer)
    context.setVelocitiesToTemperature(temperature_K * unit.kelvin, int(seed))

    kb_kj = 0.008314462618153241            # kJ/(mol K)
    n_dof = 3 * len(masses)

    # ---- equilibration, with the relaxation measured rather than assumed ------------
    t0 = time.time()
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
    while len(have) < n_target:
        want = min(chunk, n_target - len(have))
        for _ in range(want):
            integ.step(args.sample_every)
            st = context.getState(getPositions=True, getEnergy=True)
            p = st.getPositions(asNumpy=True).value_in_unit(
                unit.nanometer) * openmm_mace.NM_TO_A
            have.append(p)
            temps.append(2.0 * st.getKineticEnergy().value_in_unit(
                unit.kilojoule_per_mole) / (n_dof * kb_kj))
            com = (np.asarray(masses)[:, None] * p).sum(0) / np.sum(masses)
            if com0 is None:
                com0 = com
            com_drift.append(float(np.linalg.norm(com - com0)))
        # Written to a temporary name and renamed, so a job killed mid-write leaves the
        # previous complete file rather than a truncated array that reads as corrupt.
        tmp = frames_path.with_suffix(frames_path.suffix + ".part")
        np.save(tmp, np.array(have))
        tmp.replace(frames_path)
        if budget and (time.time() - t0) > budget:
            stopped_on_budget = True
            break
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
        # THE SAME KEY NAMES THE ASE ROUTE USES. The two drivers are an implementation
        # pair only if the same reader can read both; `seconds_per_ps` and
        # `temperature_error_K` were names nothing downstream looked for.
        seconds_per_ps_this_run=(
            float(wall / (generated * args.sample_every * args.timestep_fs / 1000.0))
            if generated else None),
        seconds_per_frame_this_run=(float(wall / generated) if generated else None),
        seconds_per_ps=float(wall / max(1e-9, prod_ps)),
        temperature_mean_K=float(np.mean(temps)) if temps else None,
        temperature_deviation_K=(float(np.mean(temps) - temperature_K)
                                 if temps else None),
        temperature_error_K=(float(np.mean(temps) - temperature_K) if temps else None),
        centre_of_mass_drift_A=float(np.max(com_drift)) if com_drift else 0.0)
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
                    help="branch A basin file (<qid>.basins.xyz). Without it there is "
                         "ONE geometry -- the QM9 reference -- and it is not a basin "
                         "list; the record says so.")
    ap.add_argument("--basin", type=int, default=None,
                    help="run only this basin index. This and --seed-index are what "
                         "make one (basin, seed) addressable by the execution layer.")
    ap.add_argument("--seed-index", type=int, default=None,
                    help="run only this seed index")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=20260903)
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
    ap.add_argument("--outroot", default=None)
    ap.add_argument("--verify-only", action="store_true",
                    help="check the OpenMM force against the ASE calculator and stop")
    args = ap.parse_args()

    cfg = config.load()
    temperature = args.temperature or config.temperature(cfg)
    model_path = engine.model_path()

    from ase.io import read
    if args.basins:
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

    # config.runs_dir honours S0_RUNS_ROOT, which on a cluster points at node-local
    # scratch. The previous default wrote straight into $HOME -- on Tianhe that is the
    # 100 GB quota'd home on Lustre, which is the one place the site manual asks you not
    # to put job output.
    outroot = (Path(args.outroot) if args.outroot
               else Path(config.runs_dir("qha", cfg)) / args.tag / args.species)
    print("geometry    {}".format(geometry_source))
    print("output      {}".format(outroot))
    print()

    basins = ([(args.basin, frames_in[args.basin])] if args.basin is not None
              else list(enumerate(frames_in)))
    seed_indices = ([args.seed_index] if args.seed_index is not None
                    else list(range(args.seeds)))

    summary = []
    for b, geom in basins:
        relaxed_A, relax_record = relax(geom, model_path)
        print("basin {:>2}    relaxed, dropped {:.4f} kcal/mol".format(
            b, relax_record["energy_drop_kcal"]))
        for k in seed_indices:
            # The seed is a pure function of (basin, seed index, seed0), exactly as in
            # the ASE route, so a task fanned out by the execution layer and the same
            # task run by hand produce the same trajectory. Nothing about placement may
            # enter it. For basin 0 this is the same number the old `seed0 + k` gave, so
            # no existing trajectory changes.
            seed = args.seed0 + 1000 * b + k
            outdir = outroot / "basin{:02d}".format(b) / "seed{:02d}".format(k)
            frames, equil, prod, force_record, thermo = run_one(
                args.species, relaxed_A, geom.get_atomic_numbers(), geom.get_masses(),
                model_path, outdir, temperature, seed, args.equil_ps, args.prod_ps,
                args)

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
                seed_formula="seed0 + 1000*basin + seed_index",
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
            print("  basin {:>2} seed {:>2}  {:>5} frames  complete={}  {:8.1f} s  "
                  "T = {:7.2f} K  COM drift {:8.2f} A".format(
                      b, k, len(frames), prod["complete"], prod["wall_seconds"],
                      prod["temperature_mean_K"], prod["centre_of_mass_drift_A"]))

    (outroot / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print()
    print("written to {}".format(outroot))
    print("analyse with: python scripts/production/s0_B_qha_analyse.py --species {} "
          "--tag {}".format(args.species, args.tag))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
