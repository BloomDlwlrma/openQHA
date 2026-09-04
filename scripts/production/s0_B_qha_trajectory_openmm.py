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

TIMESTEP_FS = 1.0
SAMPLE_EVERY_STEPS = 8
#: openmmtools' own defaults, spelled out. See the module docstring.
COLLISION_FREQUENCY_PER_PS = 50.0
CHAIN_LENGTH = 5
NUM_MTS = 5
NUM_YOSHIDA_SUZUKI = 5
EQUIL_PS = 2.0
PROD_PS = 25.0
CHUNK_FRAMES = 250


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
    temps, com_drift = [], []
    com0 = None
    while len(have) < n_target:
        want = min(CHUNK_FRAMES, n_target - len(have))
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
        np.save(frames_path, np.array(have))
    wall = time.time() - t0

    prod_record = dict(
        ps=float(prod_ps), n_frames=int(len(have)), wall_seconds=float(wall),
        seconds_per_ps=float(wall / max(1e-9, prod_ps)),
        temperature_mean_K=float(np.mean(temps)) if temps else None,
        temperature_error_K=(float(np.mean(temps) - temperature_K) if temps else None),
        centre_of_mass_drift_A=float(np.max(com_drift)) if com_drift else 0.0)
    return np.array(have), equil_record, prod_record, force_record, thermo_record


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species", default="dsgdb9nsd_000018")
    ap.add_argument("--tag", default="omm")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=20260903)
    ap.add_argument("--equil-ps", type=float, default=EQUIL_PS)
    ap.add_argument("--prod-ps", type=float, default=PROD_PS)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--timestep-fs", type=float, default=TIMESTEP_FS)
    ap.add_argument("--sample-every", type=int, default=SAMPLE_EVERY_STEPS)
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
    atoms = read(str(config.qm9_xyz(args.species, cfg)))

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
        args.equil_ps, args.prod_ps, args.seeds))

    outroot = Path(args.outroot) if args.outroot else (
        Path.home() / "runs" / "openQHA" / "qha" / args.tag / args.species)
    print("output      {}".format(outroot))
    print()

    relaxed_A, relax_record = relax(atoms, model_path)
    print("relaxed     dropped {:.4f} kcal/mol".format(relax_record["energy_drop_kcal"]))

    summary = []
    for k in range(args.seeds):
        seed = args.seed0 + k
        outdir = outroot / "basin00" / "seed{:02d}".format(k)
        frames, equil, prod, force_record, thermo = run_one(
            args.species, relaxed_A, atoms.get_atomic_numbers(), atoms.get_masses(),
            model_path, outdir, temperature, seed, args.equil_ps, args.prod_ps, args)

        meta = dict(
            # ---- the identity assertion's inputs --------------------------------------
            bias_potential=None,
            constraints=None,
            hydrogen_mass_amu=float(next(
                (m for m, z in zip(atoms.get_masses(), atoms.get_atomic_numbers())
                 if int(z) == 1), qha.HYDROGEN_MASS_AMU)),
            timestep_fs=float(args.timestep_fs),
            thermostat="openmmtools Nose-Hoover chain ({})".format(
                thermo["implementation"]),
            thermostat_tdamp_fs=float(tdamp_fs(args.collision_frequency)),
            thermostat_chain_length=int(args.chain_length),
            source="openQHA.branchB.openmm_nose_hoover",
            # ---- everything else ------------------------------------------------------
            qm9_index=args.species, basin_index=0, seed=int(seed),
            symbols=list(atoms.get_chemical_symbols()),
            masses_amu=[float(x) for x in atoms.get_masses()],
            temperature_K=float(temperature),
            sample_every_steps=int(args.sample_every),
            frame_spacing_fs=float(args.sample_every * args.timestep_fs),
            ensemble="canonical (Nose-Hoover chain, OpenMM)",
            integrator=thermo,
            route="openmm",
            platform=args.platform,
            force=force_record,
            force_check_against_ase=check,
            centre_of_mass_pinned_to_origin=False,
            engine=engine.provenance(),
            relaxation=relax_record,
            equilibration=equil, production=prod)
        qha.assert_trajectory_identity(meta)
        (outdir / "meta.json").write_text(
            json.dumps(meta, indent=2, default=str), encoding="utf-8")
        summary.append(dict(seed=int(seed), n_frames=int(len(frames)),
                            wall_seconds=prod["wall_seconds"],
                            temperature_mean_K=prod["temperature_mean_K"],
                            com_drift_A=prod["centre_of_mass_drift_A"]))
        print("  seed {:>2}  {:>5} frames  {:8.1f} s  T = {:7.2f} K  COM drift {:8.2f} A"
              .format(k, len(frames), prod["wall_seconds"],
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
