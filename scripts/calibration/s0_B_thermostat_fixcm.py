"""How much entropy ASE's default Langevin thermostat invents on a small molecule.

CALIBRATION. It measures a property of the TOOLING in order to justify one setting in the
branch B driver, and it produces no scientific number.

What it is for
--------------
`ase.md.langevin.Langevin` defaults to `fixcm=True`, and upstream warns:

    The implementation of `fixcm=True` in `Langevin` does not strictly sample the correct
    NVT distributions. The deviations are typically small for large systems but can be
    more pronounced for small systems.

Branch B's molecules are 10 to 19 atoms, which is the far end of "small", and the first
branch B trajectory ever run came back 30 per cent hot. A warning that says "can be more
pronounced" is not a number, and this branch cannot act on prose: quasi-harmonic analysis
reads nu = sqrt(k_B T / lambda) with T taken from the CONFIGURED temperature, so a
trajectory that is hot inflates every lambda, deflates every nu, and raises the entropy in
a fixed direction while everything about the run looks healthy.

So the warning is turned into two numbers -- the temperature error, and the entropy error
it causes in kcal/mol on T*S -- against the 1.0 kcal/mol target accuracy of stage 0.

Why a cheap calculator
----------------------
The question is about the INTEGRATOR, not the potential: `fixcm` decides how the thermostat
treats the centre of mass, and that is the same arithmetic whatever computes the forces. A
cheap calculator buys hundreds of thousands of steps, and this measurement needs long
trajectories far more than it needs an accurate surface -- the quantity being measured is a
sampling bias, and a short run cannot separate one from noise. Pass `--engine mace` to
repeat it on the production potential; expect it to take hours and to say the same thing.

    python scripts/calibration/s0_B_thermostat_fixcm.py
    python scripts/calibration/s0_B_thermostat_fixcm.py --steps 400000
"""
import argparse
import sys
from pathlib import Path

import numpy as np


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


sys.path.insert(0, str(_repo_root()))

from openqha import config, qha, report  # noqa: E402

TARGET_K = 298.15
TIMESTEP_FS = 1.0
FRICTION_PER_PS = 1.0
SAMPLE_EVERY = 8


def build(engine_name, species, cfg):
    """The molecule and its calculator."""
    from ase.io import read
    atoms = read(str(config.qm9_xyz(species, cfg)))
    if engine_name == "emt":
        from ase.calculators.emt import EMT
        calc, label = EMT(), "ase.calculators.emt.EMT (a cheap stand-in, see the header)"
    else:
        from openqha import engine
        calc, name, _prov = engine.calculator(device="cpu")
        label = name
    atoms.calc = calc
    from ase.optimize import BFGS
    BFGS(atoms, logfile=None).run(fmax=0.02, steps=300)
    return atoms, calc, label


def kinetic_degrees_of_freedom(n_atoms, fixcm, fixcom):
    """How many degrees of freedom actually carry kinetic energy in this variant.

    THE DIVISOR IS NOT THE SAME FOR ALL THREE, and getting that wrong produces a
    convincing table of nonsense. This was caught the hard way: the first run of this
    script used 3N throughout and reported `fixcm=False + FixCom` as 10 per cent COLD.
    It is not. Fixing the centre of mass removes three degrees of freedom, so 27 of them
    hold the energy of 30, and 298.15 x 27/30 = 268.3 K -- which is exactly what came
    back. The variant was right and the estimator was wrong.

    `fixcm=True` removes the centre-of-mass momentum every step, so it too has 3N-3.
    Reading it with 3N therefore UNDERSTATES how hot it is, and the honest number for
    ASE's default is worse than the first pass suggested.

    Rotation is not subtracted anywhere here: nothing in this protocol removes angular
    momentum, and the thermostat drives those degrees of freedom like any other.
    """
    return 3 * n_atoms - (3 if (fixcm or fixcom) else 0)


def trajectory(atoms, calc, fixcm, fixcom, steps, seed):
    """One Langevin run; returns (frames, temperatures on this variant's own divisor)."""
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    a = atoms.copy()
    a.calc = calc
    if fixcom:
        from ase.constraints import FixCom
        a.set_constraint(FixCom())
    rng = np.random.RandomState(int(seed))
    MaxwellBoltzmannDistribution(a, temperature_K=TARGET_K, rng=rng)
    dyn = Langevin(a, TIMESTEP_FS * units.fs, temperature_K=TARGET_K,
                   friction=FRICTION_PER_PS / (1000.0 * units.fs), rng=rng, fixcm=fixcm)
    frames, temps = [], []
    n_dof = kinetic_degrees_of_freedom(len(a), fixcm, fixcom)

    def grab():
        frames.append(a.get_positions().copy())
        temps.append(2.0 * a.get_kinetic_energy() / (n_dof * units.kB))

    dyn.attach(grab, interval=SAMPLE_EVERY)
    dyn.run(int(steps))
    # The first tenth is equilibration from the Maxwell-Boltzmann start and is dropped
    # from both the temperature and the covariance.
    cut = max(1, len(frames) // 10)
    return np.array(frames[cut:]), np.array(temps[cut:])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species", default="dsgdb9nsd_000018")
    ap.add_argument("--engine", default="emt", choices=("emt", "mace"))
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = config.load()
    atoms, calc, engine_label = build(args.engine, args.species, cfg)
    masses = atoms.get_masses()
    n_dof = 3 * len(atoms) - 6

    print("=" * 92)
    print("Branch B calibration -- what ASE's fixcm default costs")
    print("=" * 92)
    print("species   {}   atoms {}   3N-6 = {}".format(args.species, len(atoms), n_dof))
    print("engine    {}".format(engine_label))
    print("protocol  {} steps of {} fs, friction {}/ps, target {} K".format(
        args.steps, TIMESTEP_FS, FRICTION_PER_PS, TARGET_K))
    print()

    variants = [
        ("fixcm=True (ASE default)", True, False),
        ("fixcm=False (branch B)", False, False),
        ("fixcm=False + FixCom", False, True),
    ]
    rows = []
    for label, fixcm, fixcom in variants:
        frames, temps = trajectory(atoms, calc, fixcm, fixcom, args.steps, args.seed)
        blocks = np.array([b.mean() for b in np.array_split(temps, 10)])
        t_err = float(blocks.std(ddof=1) / np.sqrt(len(blocks)))
        rec = qha.analyse(frames, masses, TARGET_K)
        rows.append(dict(
            variant=label, fixcm=bool(fixcm), fixcom=bool(fixcom),
            n_frames=int(len(frames)),
            kinetic_degrees_of_freedom=kinetic_degrees_of_freedom(
                len(atoms), fixcm, fixcom),
            temperature_K=float(temps.mean()),
            temperature_block_error_K=t_err,
            temperature_error_K=float(temps.mean() - TARGET_K),
            temperature_error_percent=float(100.0 * (temps.mean() - TARGET_K) / TARGET_K),
            TS_QH_kcal=rec["entropy"]["TS_QH_kcal"],
            TS_Schlitter_kcal=rec["entropy"]["TS_Schlitter_kcal"],
            lowest_frequency_cm_inv=rec["entropy"]["lowest_frequency_cm_inv"],
            n_nonzero=rec["spectrum"]["n_nonzero_eigenvalues"]))
        print("{:<28} dof {:>3}   T = {:7.2f} +- {:5.2f} K ({:+6.2f}%)   "
              "T*S_QH = {:8.4f} kcal/mol"
              .format(label, rows[-1]["kinetic_degrees_of_freedom"],
                      rows[-1]["temperature_K"], t_err,
                      rows[-1]["temperature_error_percent"], rows[-1]["TS_QH_kcal"]))

    reference = next(r for r in rows if not r["fixcm"] and not r["fixcom"])
    for r in rows:
        r["TS_QH_minus_branchB_kcal"] = float(r["TS_QH_kcal"] - reference["TS_QH_kcal"])

    default = rows[0]
    print()
    print("The ASE default costs {:+.4f} kcal/mol on T*S, against a 1.0 kcal/mol target "
          "accuracy.".format(default["TS_QH_minus_branchB_kcal"]))

    out_stem = Path(args.out) if args.out else (
        _repo_root() / "analysis" / "qha" / "calibration_thermostat_fixcm")
    rp = report.Report(
        "Branch B calibration -- ASE Langevin fixcm on a small molecule",
        subtitle="{}   {}   {} steps".format(args.species, engine_label, args.steps))
    rp.section("Why this was measured")
    rp.note("ASE warns that fixcm=True 'does not strictly sample the correct NVT "
            "distributions' and that deviations 'can be more pronounced for small "
            "systems'. Branch B's molecules are 10 to 19 atoms. A warning is not a "
            "number, and quasi-harmonic entropy is read off the covariance at a "
            "CONFIGURED temperature, so a hot trajectory raises the entropy silently "
            "and in a fixed direction.")
    rp.kv("target_temperature_K", TARGET_K)
    rp.kv("timestep_fs", TIMESTEP_FS)
    rp.kv("friction_per_ps", FRICTION_PER_PS)
    rp.kv("engine", engine_label)
    rp.section("Measured")
    rp.table(["variant", "dof", "T", "block error", "error", "T*S_QH", "vs branch B"],
             [[r["variant"], r["kinetic_degrees_of_freedom"],
               round(r["temperature_K"], 2),
               round(r["temperature_block_error_K"], 2),
               "{:+.2f}%".format(r["temperature_error_percent"]),
               round(r["TS_QH_kcal"], 4),
               "{:+.4f}".format(r["TS_QH_minus_branchB_kcal"])] for r in rows],
             units=[None, None, "K", "K", None, "kcal/mol", "kcal/mol"])
    rp.note("Each variant is read on ITS OWN divisor: fixing or zeroing the centre of "
            "mass removes three degrees of freedom, so those two variants divide by 3N-3 "
            "and the third by 3N. A first pass of this script used 3N throughout and "
            "reported FixCom as 10 per cent cold; it was not, 298.15 x 27/30 = 268.3 is "
            "exactly what 3N gives for a correct trajectory. The estimator was wrong, not "
            "the run.")
    rp.section("Ruling")
    rp.note("Branch B uses fixcm=False and adds NO constraint. FixCom is equally correct "
            "on the numbers above and is what upstream suggests, so the choice is not "
            "made on accuracy: FixCom is an ASE constraint, and branch B's first "
            "prohibition is that there are no constraints. 'No constraints except the "
            "harmless ones' is not a rule anyone can check, and the harm here would be "
            "zero only for as long as nobody added a second one. Without FixCom the "
            "centre of mass random-walks, and the superposition in openqha/qha.py removes "
            "that before the covariance is taken.")
    rp.note("qha.assert_trajectory_identity refuses any trajectory whose metadata does "
            "not record thermostat_fixcm = False, so this ruling is enforced rather than "
            "documented.")
    rp.json_dump(dict(species=args.species, engine=engine_label, steps=args.steps,
                      target_K=TARGET_K, variants=rows))
    log = rp.write(str(out_stem) + ".log")
    written = report.write_parquet(dict(fixcm_variants=rows), out_stem)
    print("\nwritten:\n  {}".format(log))
    for p, n, _c in written:
        print("  {}  ({} rows)".format(p, n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
