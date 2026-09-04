"""Nose-Hoover as OpenMM implements it, on the same closed-form surface as the ASE sweep.

CALIBRATION. It measures a property of the TOOLING and produces no scientific number.

Why a second script rather than another variant
-----------------------------------------------
OpenMM does not live in the environment branch B runs in. `openqha` has MACE 0.3.16 and
ASE; `qm9fe` has OpenMM 8.4, openmm-torch, openmmtools, mace_md and GROMACS. This file is
the half that runs in `qm9fe`:

    /home/ubuntu/anaconda3/envs/qm9fe/bin/python scripts/calibration/s0_B_thermostat_openmm.py

It deliberately does NOT import MACE. The surface is loaded from a file written by
`--dump-surface` under `openqha`, so both halves integrate literally the same Hessian and
compare against literally the same closed form. That also keeps this benchmark clear of a
second problem: in `qm9fe`, `import mace` resolves to the vendored develop tree, whose
neighbour list is NOT translation invariant (see openqha/mace_patch.py). A benchmark of
thermostats has no business depending on that.

Why it exists
-------------
An earlier pass tested one ASE Nose-Hoover setting and reported it far outside budget. That
was too strong a conclusion from one point, and the propagators are not comparable:

    ASE    NoseHooverChainNVT   tchain 3, tloop 1, 3-term Yoshida-Suzuki   ->  3 chain
                                                                              evaluations/step
    OpenMM NoseHooverIntegrator chainLength 3, numMTS 3, numYS 7          -> 21

If a thermostat's measured bias is really its propagator's discretisation error, the fix is
a better propagator, not a different thermostat. This file is how that is told apart.

What is measured
----------------
The same thing as the ASE half: on a harmonic surface the classical canonical distribution
satisfies <q_k^2> = k_B T / omega_k^2 exactly, so a correctly sampled trajectory must return
the Hessian's own frequencies, and the quasi-harmonic entropy must equal the closed form.
The number reported is each integrator's BIAS against that, not a difference between them.

    # under openqha, once:
    python scripts/calibration/s0_B_thermostat_openmm.py --dump-surface
    # then under qm9fe:
    /home/ubuntu/anaconda3/envs/qm9fe/bin/python \\
        scripts/calibration/s0_B_thermostat_openmm.py --steps 500000 --seeds 4
"""
import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))

TARGET_K = 298.15
TIMESTEP_FS = 1.0
SAMPLE_EVERY = 8
EQUILIBRATION_FRACTION = 0.1
BUDGET_KCAL = 0.10
SURFACE_NPZ = ROOT / "analysis" / "qha" / "harmonic_surface.npz"

#: eV -> kJ/mol, and angstrom -> nm. OpenMM is nm/kJ/mol throughout; the Hessian is
#: eV/A^2. Getting either of these wrong shows up as a temperature that is wrong by a
#: constant factor, which is exactly the kind of error a closed-form reference catches.
EV_TO_KJ_PER_MOL = 96.48533212331002
NM_TO_A = 10.0


# ======================================================================================
# Step one, under `openqha`: write the surface out
# ======================================================================================
def dump_surface(species, tether_cm_inv, out_path):
    """Build the harmonic surface with MACE and save it. Runs under `openqha` only."""
    import importlib.util as iu
    spec = iu.spec_from_file_location(
        "tc", str(ROOT / "scripts" / "calibration" / "s0_B_thermostat_choice.py"))
    tc = iu.module_from_spec(spec)
    spec.loader.exec_module(tc)
    from openqha import config, qha

    atoms, h_raw, label = tc.build_surface("mace", species, config.load(), "analytic")
    masses = atoms.get_masses()
    x0 = atoms.get_positions()
    h_tether, rank, check = tc.tethered_hessian(h_raw, masses, x0, tether_cm_inv)
    _frames, ref = qha.synthetic_harmonic_trajectory(h_raw, masses, x0, TARGET_K,
                                                     n_frames=8, seed=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out_path),
             numbers=atoms.get_atomic_numbers(), masses=masses, positions=x0,
             hessian_eV_A2=h_raw, hessian_tethered_eV_A2=h_tether,
             closed_form_TS_kcal=float(ref["closed_form_TS_kcal"]),
             frequencies_cm_inv=np.asarray(ref["frequencies_cm_inv"], dtype=float),
             rigid_rank=int(rank), tether_cm_inv=float(tether_cm_inv),
             engine=str(label), species=str(species),
             n_free_directions=int(check["n_free_directions"]),
             translation_overlap=float(check["translation_overlap"]))
    print("written {}".format(out_path))
    print("  engine            {}".format(label))
    print("  closed form T*S   {:.6f} kcal/mol over {} modes".format(
        ref["closed_form_TS_kcal"], ref["n_modes"]))
    print("  tether check      {} free direction(s), translation overlap {:.6f}".format(
        check["n_free_directions"], check["translation_overlap"]))
    return 0


# ======================================================================================
# Step two, under `qm9fe`: integrate it with OpenMM
# ======================================================================================
class _HarmonicModule:
    """Factory for the TorchScript module OpenMM will call for energy and forces.

    E = 1/2 dx^T H dx with dx measured after matching the centre of mass to the reference,
    which makes a rigid translation an EXACT zero mode -- the same construction as the ASE
    half, so the two are integrating the same surface and not merely a similar one.
    """

    @staticmethod
    def build(hessian_eV_A2, reference_positions_A, masses):
        import torch

        h = torch.tensor(np.asarray(hessian_eV_A2) * EV_TO_KJ_PER_MOL,
                         dtype=torch.float64)
        x0 = torch.tensor(np.asarray(reference_positions_A), dtype=torch.float64)
        m = torch.tensor(np.asarray(masses), dtype=torch.float64)
        w = (m / m.sum()).unsqueeze(1)
        com0 = (x0 * w).sum(dim=0)

        class Harmonic(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("h", h)
                self.register_buffer("x0", x0)
                self.register_buffer("w", w)
                self.register_buffer("com0", com0)
                self.register_buffer("nm_to_a",
                                     torch.tensor(NM_TO_A, dtype=torch.float64))

            def forward(self, positions):
                # OpenMM hands positions in nm; the Hessian is per angstrom. The factor is
                # a buffer rather than a module-level constant because TorchScript will
                # not close over a Python global -- it refuses rather than capturing a
                # stale value, which is the good failure.
                x = positions.to(torch.float64) * self.nm_to_a
                com = (x * self.w).sum(dim=0)
                dx = (x - com + self.com0 - self.x0).reshape(-1)
                return 0.5 * torch.dot(dx, torch.mv(self.h, dx))

        return torch.jit.script(Harmonic())


def _system(numbers, masses, hessian, x0):
    import openmm
    from openmm import unit
    import openmmtorch

    system = openmm.System()
    for mass in masses:
        system.addParticle(float(mass) * unit.amu)
    module = _HarmonicModule.build(hessian, x0, masses)
    # Per process: several workers writing one path would race, and the failure would be
    # a silently wrong surface rather than an error.
    path = "/tmp/openqha_harmonic_module_{}.pt".format(os.getpid())
    module.save(path)
    force = openmmtorch.TorchForce(path)
    force.setUsesPeriodicBoundaryConditions(False)
    system.addForce(force)
    return system


VARIANTS = [
    ("openmm_langevin_middle_1.0", dict(kind="langevin", friction_per_ps=1.0),
     "what mace-md uses by default, and the same friction as branch B"),
    ("openmm_nose_hoover_100fs", dict(kind="nose_hoover", tdamp_fs=100.0),
     "OpenMM defaults: chainLength 3, numMTS 3, numYS 7"),
    ("openmm_nose_hoover_419fs", dict(kind="nose_hoover", tdamp_fs=419.0),
     "coupling time matched to the lowest mode's period"),
    ("openmm_nose_hoover_1000fs", dict(kind="nose_hoover", tdamp_fs=1000.0),
     "loose coupling"),
    ("openmm_nose_hoover_20fs", dict(kind="nose_hoover", tdamp_fs=20.0),
     "tight coupling"),
]
VARIANT_BY_NAME = {v[0]: v for v in VARIANTS}


def _integrator(spec, seed):
    import openmm
    from openmm import unit
    dt = TIMESTEP_FS * unit.femtosecond
    if spec["kind"] == "langevin":
        integ = openmm.LangevinMiddleIntegrator(
            TARGET_K * unit.kelvin, spec["friction_per_ps"] / unit.picosecond, dt)
        integ.setRandomNumberSeed(int(seed))
        return integ
    if spec["kind"] == "nose_hoover":
        # collisionFrequency is 1/tdamp. The remaining three arguments are OpenMM's
        # defaults, spelled out because they are the whole reason this file exists:
        # 3 chain beads, 3 multiple-time-step substeps, 7 Yoshida-Suzuki terms.
        freq = (1000.0 / spec["tdamp_fs"]) / unit.picosecond
        return openmm.NoseHooverIntegrator(TARGET_K * unit.kelvin, freq, dt, 3, 3, 7)
    raise ValueError("unknown integrator kind {!r}".format(spec["kind"]))


def _worker(job):
    """One (variant, seed) in its own process; never raises into the pool."""
    name, spec, seed, steps, surface, platform_name = job
    try:
        return run_one(name, spec, seed, steps, surface, platform_name)
    except Exception as exc:                                  # noqa: BLE001
        return dict(variant=name, seed=int(seed), failed=True,
                    error="{}: {}".format(type(exc).__name__, exc))


def run_one(name, spec, seed, steps, surface, platform_name="CPU"):
    import openmm
    from openmm import unit
    import time

    numbers = surface["numbers"]
    masses = surface["masses"]
    x0 = surface["positions"]
    h = surface["hessian_tethered_eV_A2"]

    system = _system(numbers, masses, h, x0)
    integ = _integrator(spec, seed)
    platform = openmm.Platform.getPlatformByName(platform_name)
    context = openmm.Context(system, integ, platform)
    context.setPositions((x0 / NM_TO_A) * unit.nanometer)
    context.setVelocitiesToTemperature(TARGET_K * unit.kelvin, int(seed))

    n_samples = int(steps) // SAMPLE_EVERY
    frames = np.empty((n_samples, len(masses), 3), dtype=float)
    kinetic = np.empty(n_samples)
    potential = np.empty(n_samples)
    t0 = time.time()
    for i in range(n_samples):
        integ.step(SAMPLE_EVERY)
        state = context.getState(getPositions=True, getEnergy=True)
        frames[i] = state.getPositions(asNumpy=True).value_in_unit(
            unit.nanometer) * NM_TO_A
        kinetic[i] = state.getKineticEnergy().value_in_unit(
            unit.kilojoule_per_mole) / EV_TO_KJ_PER_MOL
        potential[i] = state.getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole) / EV_TO_KJ_PER_MOL
    wall = time.time() - t0

    cut = max(1, int(EQUILIBRATION_FRACTION * n_samples))
    pos, ke, pe = frames[cut:], kinetic[cut:], potential[cut:]

    from openqha import qha, thermo
    kt = (thermo.KB_SI / 1.602176634e-19) * TARGET_K
    dof_kin = float(2.0 * ke.mean() / kt)
    dof_pot = float(2.0 * pe.mean() / kt)
    rec = qha.analyse(pos, masses, TARGET_K)
    return dict(
        variant=name, seed=int(seed), n_frames=int(len(pos)),
        TS_QH_kcal=rec["entropy"]["TS_QH_kcal"],
        lowest_frequency_cm_inv=rec["entropy"]["lowest_frequency_cm_inv"],
        n_nonzero=rec["spectrum"]["n_nonzero_eigenvalues"],
        kinetic_dof_effective=dof_kin,
        configurational_dof_effective=dof_pot,
        kinetic_variance_ratio=float(
            (ke.var(ddof=1) / ke.mean() ** 2) / (2.0 / dof_kin)),
        configurational_variance_ratio=float(
            (pe.var(ddof=1) / pe.mean() ** 2) / (2.0 / dof_pot)),
        wall_seconds=float(wall), platform=platform_name)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump-surface", action="store_true",
                    help="build the surface with MACE and save it; run under `openqha`")
    ap.add_argument("--species", default="dsgdb9nsd_000018")
    ap.add_argument("--tether-cm-inv", type=float, default=500.0)
    ap.add_argument("--surface", default=str(SURFACE_NPZ))
    ap.add_argument("--steps", type=int, default=500000)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=20260904)
    ap.add_argument("--platform", default="CPU")
    ap.add_argument("--processes", type=int, default=1)
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.dump_surface:
        return dump_surface(args.species, args.tether_cm_inv, Path(args.surface))

    npz = np.load(args.surface, allow_pickle=False)
    surface = {k: npz[k] for k in npz.files}
    exact_TS = float(surface["closed_form_TS_kcal"])

    chosen = args.variants or [v[0] for v in VARIANTS]
    unknown = [c for c in chosen if c not in VARIANT_BY_NAME]
    if unknown:
        raise SystemExit("unknown variant(s): {}".format(", ".join(unknown)))

    import openmm
    print("=" * 92)
    print("Branch B calibration -- OpenMM's Nose-Hoover against the closed form")
    print("=" * 92)
    print("surface    {}   loaded from {}".format(surface["engine"], args.surface))
    print("           tether check: {} free direction(s), translation overlap {:.6f}"
          .format(int(surface["n_free_directions"]),
                  float(surface["translation_overlap"])))
    print("openmm     {}   platform {}".format(openmm.version.version, args.platform))
    print("protocol   {} steps of {} fs, sample every {}, first {:.0f}% dropped"
          .format(args.steps, TIMESTEP_FS, SAMPLE_EVERY, 100 * EQUILIBRATION_FRACTION))
    print("truth      T*S = {:.6f} kcal/mol".format(exact_TS))
    print()

    jobs = [(name, VARIANT_BY_NAME[name][1], args.seed0 + i, args.steps, surface,
             args.platform)
            for name in chosen for i in range(args.seeds)]
    if args.processes > 1:
        with mp.Pool(min(args.processes, len(jobs))) as pool:
            raw = pool.map(_worker, jobs)
    else:
        raw = [_worker(j) for j in jobs]

    rows = []
    for name in chosen:
        got = [r for r in raw if r["variant"] == name and not r.get("failed")]
        bad = [r for r in raw if r["variant"] == name and r.get("failed")]
        if not got:
            print("{:<28} FAILED  {}".format(name, bad[0]["error"] if bad else ""))
            continue
        ts = np.array([r["TS_QH_kcal"] for r in got])
        row = dict(variant=name, note=VARIANT_BY_NAME[name][2], n_seeds=len(got),
                   n_failed=len(bad),
                   TS_QH_kcal=float(ts.mean()),
                   TS_error_kcal=float(ts.mean() - exact_TS),
                   TS_seed_sem_kcal=float(ts.std(ddof=1) / np.sqrt(len(ts)))
                   if len(ts) > 1 else 0.0,
                   kinetic_dof_effective=float(
                       np.mean([r["kinetic_dof_effective"] for r in got])),
                   configurational_variance_ratio=float(
                       np.mean([r["configurational_variance_ratio"] for r in got])),
                   wall_seconds=float(np.mean([r["wall_seconds"] for r in got])))
        row["within_budget"] = bool(abs(row["TS_error_kcal"]) <= BUDGET_KCAL)
        rows.append(row)
        print("{:<28} T*S err {:+8.4f} +- {:6.4f}   conf.var ratio {:6.3f}   "
              "kin dof {:6.2f}   {:6.0f} s   {}"
              .format(name, row["TS_error_kcal"], row["TS_seed_sem_kcal"],
                      row["configurational_variance_ratio"],
                      row["kinetic_dof_effective"], row["wall_seconds"],
                      "ok" if row["within_budget"] else "OVER BUDGET"))

    from openqha import report
    out_stem = Path(args.out) if args.out else (
        ROOT / "analysis" / "qha" / "calibration_thermostat_openmm")
    rp = report.Report("Branch B calibration -- OpenMM Nose-Hoover against the closed form",
                       subtitle="{}   openmm {}   {} steps x {} seeds".format(
                           str(surface["species"]), openmm.version.version,
                           args.steps, args.seeds))
    rp.section("Why OpenMM and not only ASE")
    rp.note("The two implement the same algorithm with very different propagation "
            "resolution: ASE's NoseHooverChainNVT defaults to tchain 3, tloop 1 with a "
            "three-term Yoshida-Suzuki decomposition, i.e. three chain evaluations per "
            "step; OpenMM's NoseHooverIntegrator defaults to chainLength 3, numMTS 3, "
            "numYS 7, i.e. twenty-one. A bias that is really the propagator's "
            "discretisation error would look like a property of the thermostat, and this "
            "is how the two are told apart.")
    rp.note("The surface is loaded from a file written under the `openqha` environment, "
            "so both halves integrate the same Hessian and are compared against the same "
            "closed form. No MACE is imported here -- deliberately: in `qm9fe`, "
            "`import mace` resolves to the vendored develop tree, whose neighbour list is "
            "not translation invariant.")
    rp.kv("closed_form_TS_kcal", round(exact_TS, 6), unit="kcal/mol")
    rp.kv("budget_kcal", BUDGET_KCAL)
    rp.kv("timestep_fs", TIMESTEP_FS)
    rp.kv("target_temperature_K", TARGET_K)
    rp.section("Measured")
    rp.table(["variant", "T*S error", "seed s.e.", "conf. var ratio", "kin. dof",
              "wall", "verdict"],
             [[r["variant"], "{:+.4f}".format(r["TS_error_kcal"]),
               round(r["TS_seed_sem_kcal"], 4),
               round(r["configurational_variance_ratio"], 3),
               round(r["kinetic_dof_effective"], 2),
               round(r["wall_seconds"], 0),
               "ok" if r["within_budget"] else "OVER BUDGET"] for r in rows],
             units=[None, "kcal/mol", "kcal/mol", None, None, "s", None])
    raw = [r for r in raw if not r.get("failed")]
    rp.json_dump(dict(surface=str(args.surface), engine=str(surface["engine"]),
                      openmm_version=openmm.version.version, platform=args.platform,
                      steps=args.steps, seeds=args.seeds, seed0=args.seed0,
                      timestep_fs=TIMESTEP_FS, sample_every=SAMPLE_EVERY,
                      target_K=TARGET_K, budget_kcal=BUDGET_KCAL,
                      closed_form_TS_kcal=exact_TS, variants=rows, runs=raw))
    log = rp.write(str(out_stem) + ".log")
    written = report.write_parquet(dict(openmm_variants=rows, openmm_runs=raw), out_stem)
    print("\nwritten:\n  {}".format(log))
    for p, n, _c in written:
        print("  {}  ({} rows)".format(p, n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
