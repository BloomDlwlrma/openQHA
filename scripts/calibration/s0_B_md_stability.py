"""How long an unbiased 298 K trajectory survives on the production potential.

CALIBRATION. It measures a property of the POTENTIAL, and it decides whether branch B's
protocol is runnable at all. It produces no thermodynamic number.

Why this exists
---------------
The first real branch B trajectory -- acetone, MACE-OFF23_medium, float64, 1 fs, Langevin
at 298.15 K with no bias and no constraints -- dissociated after 1.656 ps, and so did
2 to 3 of every 8 trajectories on three different MACE-OFF models.

**TWO diagnoses have been written here and both were wrong. The cause is open.** They
are both left in place, because a script whose stated reason for existing has been
withdrawn twice should say so rather than quietly acquire a third.

FIRST, retracted: a short-range hole. The molecule died with two geminal hydrogens at
0.315 A, and a frozen scan showed the model turning attractive below 0.45 A (+3799
kcal/mol at 0.40 A, -1855 at 0.25) -- the classic hole a machine-learned potential has
where it has no training data. Refuted by measurement: models whose short-range wall is
intact (MACE-OFF23-SC, MACE-OFF23b_medium) failed at the same rate. See
scripts/calibration/s0_B_short_range_wall.py.

SECOND, retracted 2026-09-04: a neighbour list that is not translation invariant. That
defect is real -- a search box anchored at the coordinate ORIGIN, so a molecule that
random-walks away from it is binned wrongly -- but it is in the MACE **develop** tree
vendored under stage 2, NOT in the `mace_torch 0.3.16` that openQHA imports. Measured with
the patch OFF on the installed tree: dE = 0.000000 kcal/mol for rigid shifts out to
100000 A, with an identical 90-edge neighbour list. The code that runs here cannot have
caused it. See openqha/mace_patch.py.

WHAT IS ACTUALLY KNOWN. On 2026-09-03 this script reported 3 of 8 trajectories reaching a
close contact; on 2026-09-04, same engine, species, seeds and printed protocol, it
reported 0 of 8. Four of the eight seeds were BIT-IDENTICAL across the two runs and four
were completely different, which rules out a changed temperature or timestep and points at
the force evaluation. Neither product recorded which MACE was imported or how the centre
of mass was handled, so the two cannot be told apart after the fact. They are recorded now,
per row -- which is the only lasting thing this episode produced.

What this script is for now
---------------------------
It is the standing check that a trajectory can survive its intended length on whatever
model and settings are in force. That question is now MORE important, not less: no cause
has been established, so the only defence is to keep measuring survival on the settings
actually in use. The short-range holes are still real and still reachable if anything ever
drives the molecule far from equilibrium.

It measures the SURVIVAL TIME over independent seeds and reports the distribution rather
than an average -- a mean survival time is meaningless for a process that either happens
or does not.

    python scripts/calibration/s0_B_md_stability.py --ps 3 --seeds 6
    python scripts/calibration/s0_B_md_stability.py --ps 3 --seeds 8 --free-walk
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


sys.path.insert(0, str(_repo_root()))

from openqha import config, engine, report  # noqa: E402

#: A pair closer than this has entered territory no electronic-structure dataset covers.
#: 0.60 A is chosen ABOVE the hole (which opens near 0.45 A) so the run is stopped while
#: the geometry is still merely improbable rather than already unphysical.
CLOSE_CONTACT_A = 0.60

#: Chemically, nothing in a CHON molecule at 298 K should come within this. Crossing it is
#: reported as a warning level, not a stop: it is the early sign of the same event.
WATCH_CONTACT_A = 0.90


def run_one(species, seed, ps, timestep_fs, temperature_K, engine_name=None,
            pin_com=True, perturb_A=0.0, device="cpu"):
    """One trajectory, stopped the moment a close contact appears. Returns a record."""
    import time
    from ase import units
    from ase.io import read
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.optimize import BFGS
    from scipy.spatial.distance import pdist

    calc, engine_name, prov = engine.calculator(device=device, name=engine_name)
    atoms = read(str(config.qm9_xyz(species)))
    atoms.calc = calc
    BFGS(atoms, logfile=None).run(fmax=0.005, steps=300)
    if perturb_A:
        # A RIGID TRANSLATION, which is an exact symmetry of any interatomic potential and
        # therefore changes no physics -- only the last bits of every force. Measured on
        # MACE-OFF23_medium: a shift leaves the energy identical to 0.000e+00 eV while the
        # forces move by about 1e-14 eV/A. This switch exists to find out how much of this
        # script's own answer is decided by that.
        atoms.set_positions(atoms.get_positions()
                            + np.array([float(perturb_A), 0.0, 0.0])[None, :])
    e_min = atoms.get_potential_energy()
    d_ref = float(pdist(atoms.get_positions()).max())

    rng = np.random.RandomState(int(seed))
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=rng)
    dyn = Langevin(atoms, timestep_fs * units.fs, temperature_K=temperature_K,
                   friction=1.0 / (1000.0 * units.fs), rng=rng, fixcm=False)

    state = dict(min_seen_A=9e9, min_at_fs=None, watch_at_fs=None, stop_at_fs=None)

    def watch():
        d = pdist(atoms.get_positions())
        lo = float(d.min())
        now = dyn.get_number_of_steps() * timestep_fs
        if lo < state["min_seen_A"]:
            state["min_seen_A"], state["min_at_fs"] = lo, now
        if lo < WATCH_CONTACT_A and state["watch_at_fs"] is None:
            state["watch_at_fs"] = now
        if lo < CLOSE_CONTACT_A and state["stop_at_fs"] is None:
            state["stop_at_fs"] = now

    def recentre():
        # Redundant since openqha/mace_patch.py fixed the neighbour list itself, and kept
        # only so that `--free-walk` has something to be compared against: with the patch
        # in place the two give bit-identical trajectories, which is how the patch is
        # tested rather than assumed. Before the patch this was the difference between a
        # trajectory that survived and one that exploded in about two picoseconds.
        m = atoms.get_masses()
        com = (atoms.get_positions() * m[:, None]).sum(axis=0) / m.sum()
        atoms.set_positions(atoms.get_positions() - com[None, :])

    if pin_com:
        dyn.attach(recentre, interval=1)
    dyn.attach(watch, interval=1)      # every step: the event lasts a few femtoseconds

    n_steps = int(round(ps * 1000.0 / timestep_fs))
    t0 = time.time()
    done = 0
    chunk = max(1, n_steps // 50)
    while done < n_steps and state["stop_at_fs"] is None:
        step = min(chunk, n_steps - done)
        dyn.run(step)
        done += step
    wall = time.time() - t0

    d = pdist(atoms.get_positions())
    patch = prov.get("neighbour_list_patch") or {}
    installed = patch.get("installed") or {}
    return dict(
        species=species, seed=int(seed), timestep_fs=float(timestep_fs),
        temperature_K=float(temperature_K), engine=engine_name,
        requested_ps=float(ps), simulated_ps=float(done * timestep_fs / 1000.0),
        survived=bool(state["stop_at_fs"] is None),
        stop_at_ps=(None if state["stop_at_fs"] is None
                    else float(state["stop_at_fs"] / 1000.0)),
        first_watch_at_ps=(None if state["watch_at_fs"] is None
                           else float(state["watch_at_fs"] / 1000.0)),
        min_distance_seen_A=float(state["min_seen_A"]),
        min_distance_at_ps=(None if state["min_at_fs"] is None
                            else float(state["min_at_fs"] / 1000.0)),
        final_max_distance_A=float(d.max()),
        final_potential_above_minimum_kcal=float(
            (atoms.get_potential_energy() - e_min) * 23.060547830618307),
        reference_max_distance_A=d_ref,
        # The conditions that decide the result, carried WITH the result. Two runs of this
        # script on 2026-09-03 and 2026-09-04 disagreed 3-of-8 against 0-of-8 on the same
        # engine, species, seeds and printed protocol, and nothing in either product said
        # what differed. A protocol that is printed in a banner but not stored in the row
        # is not provenance.
        pin_centre_of_mass=bool(pin_com),
        perturb_A=float(perturb_A or 0.0),
        thermostat="ase.md.langevin.Langevin",
        thermostat_fixcm=False,
        friction_per_ps=1.0,
        mace_module_path=str(prov.get("mace_module_path", "")),
        mace_torch_version=str(prov.get("mace_torch_version", "")),
        neighbour_list_patched=bool(patch.get("applied")),
        neighbour_list_defect_present=installed.get("defect_present"),
        wall_seconds=float(wall),
        seconds_per_ps=float(wall / max(1e-9, done * timestep_fs / 1000.0)),
    )


def _worker(args):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    return run_one(*args)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species", default="dsgdb9nsd_000018")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--seed0", type=int, default=20260903)
    ap.add_argument("--ps", type=float, default=3.0)
    ap.add_argument("--timestep-fs", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--engine", default=None,
                    help="engine name from openqha/engine.py; default is the production one")
    ap.add_argument("--free-walk", action="store_true",
                    help="do NOT pin the centre of mass -- a true free Langevin walk. "
                         "Correct only because openqha/mace_patch.py fixed the neighbour "
                         "list; this flag is how that fix is tested rather than assumed")
    ap.add_argument("--perturb-A", type=float, default=0.0,
                    help="rigidly translate the starting structure by this many "
                         "angstrom along x. An exact symmetry of the potential, so "
                         "any change in the outcome is rounding amplified by the "
                         "dynamics, not physics")
    ap.add_argument("--processes", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = config.load()
    temperature = args.temperature or config.temperature(cfg)
    tasks = [(args.species, args.seed0 + k, args.ps, args.timestep_fs, temperature,
              args.engine, not args.free_walk, args.perturb_A)
             for k in range(args.seeds)]
    n_proc = args.processes or min(len(tasks), max(1, (os.cpu_count() or 4) - 2))

    print("=" * 92)
    print("Branch B calibration -- survival of an unbiased 298 K trajectory")
    print("=" * 92)
    print("species     {}   engine {}   centre of mass {}".format(
        args.species, args.engine or engine.engine_name(),
        "free (patch under test)" if args.free_walk else "pinned"))
    print("protocol    {} fs step, Langevin friction 1/ps, {} K, fixcm=False, no bias, "
          "no constraints".format(args.timestep_fs, temperature))
    print("length      {} ps requested x {} seeds, {} at a time".format(
        args.ps, args.seeds, n_proc))
    print("stop when   any interatomic distance < {} A".format(CLOSE_CONTACT_A))
    print()

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_proc) as pool:
        rows = pool.map(_worker, tasks)

    print("{:>6} {:>10} {:>12} {:>12} {:>10} {:>12}".format(
        "seed", "survived", "stopped at", "min seen", "at", "seconds/ps"))
    for r in rows:
        print("{:>6} {:>10} {:>12} {:>12.3f} {:>10} {:>12.1f}".format(
            r["seed"], str(r["survived"]),
            "-" if r["stop_at_ps"] is None else "{:.3f} ps".format(r["stop_at_ps"]),
            r["min_distance_seen_A"],
            "-" if r["min_distance_at_ps"] is None
            else "{:.3f} ps".format(r["min_distance_at_ps"]),
            r["seconds_per_ps"]))

    n_fail = sum(1 for r in rows if not r["survived"])
    simulated = sum(r["simulated_ps"] for r in rows)
    print()
    print("{} of {} trajectories reached a close contact".format(n_fail, len(rows)))
    print("{:.1f} ps simulated in total".format(simulated))
    if n_fail:
        rate = n_fail / simulated
        print("close contacts per ps: {:.4f}   ->  expected survival about {:.1f} ps"
              .format(rate, 1.0 / rate))
        print("A 200 ps production trajectory would therefore be IMPOSSIBLE on this "
              "potential without a short-range fix.")
    else:
        print("No close contact in this sample. That bounds the rate from above at about "
              "{:.4f} per ps (one event would have been enough to see), which still has "
              "to be checked at production length.".format(1.0 / max(simulated, 1e-9)))

    out_stem = Path(args.out) if args.out else (
        _repo_root() / "analysis" / "qha" / "calibration_md_stability_{}".format(
            (args.engine or engine.engine_name()).replace("/", "_")))
    rp = report.Report(
        "Branch B calibration -- trajectory survival on MACE-OFF23_medium",
        subtitle="{}   {}   {} fs   {} K   {} seeds".format(
            args.species, args.engine or engine.engine_name(), args.timestep_fs,
            temperature, args.seeds))
    rp.section("Why this was measured")
    rp.note("The first real branch B trajectory dissociated after 1.656 ps. Two geminal "
            "hydrogens went from 0.900 A to 0.315 A inside one 8 fs sampling interval, "
            "and a frozen scan of that coordinate shows the model is ATTRACTIVE below "
            "about 0.45 A: +3799 kcal/mol at 0.40 A, -609 at 0.315, -1855 at 0.25. That "
            "is a hole in the potential, not an integration error, and no timestep fixes "
            "it.")
    rp.section("Measured")
    rp.table(["seed", "survived", "stopped at", "min distance seen", "at", "seconds/ps"],
             [[r["seed"], r["survived"],
               "-" if r["stop_at_ps"] is None else round(r["stop_at_ps"], 3),
               round(r["min_distance_seen_A"], 3),
               "-" if r["min_distance_at_ps"] is None
               else round(r["min_distance_at_ps"], 3),
               round(r["seconds_per_ps"], 1)] for r in rows],
             units=[None, None, "ps", "angstrom", "ps", "s/ps"])
    rp.kv("trajectories_that_reached_a_close_contact", n_fail)
    rp.kv("total_simulated_ps", round(simulated, 2))
    rp.kv("close_contact_threshold_A", CLOSE_CONTACT_A)
    if args.perturb_A:
        rp.warn("The starting structure was rigidly translated by {} A. That is an EXACT "
                "symmetry of the potential, so every difference from an unperturbed run "
                "is rounding amplified by the dynamics. Read this run against its "
                "unperturbed twin, not on its own.".format(args.perturb_A))
    first = rows[0] if rows else {}
    rp.section("Conditions, stored rather than printed")
    rp.kv("pin_centre_of_mass", first.get("pin_centre_of_mass"))
    rp.kv("thermostat_fixcm", first.get("thermostat_fixcm"))
    rp.kv("mace_torch_version", first.get("mace_torch_version"))
    rp.kv("mace_module_path", first.get("mace_module_path"))
    rp.kv("neighbour_list_patched", first.get("neighbour_list_patched"))
    rp.kv("neighbour_list_defect_present", first.get("neighbour_list_defect_present"))
    rp.note("These lines exist because two runs of this script disagreed 3-of-8 "
            "against 0-of-8 on the same engine, species and seeds, and neither product "
            "recorded enough to say what differed. Four of the eight seeds were "
            "bit-identical across the two runs and four were not, which rules out a "
            "change of temperature or timestep and leaves the force evaluation -- "
            "precisely the thing that was not recorded.")
    rp.json_dump(dict(species=args.species, timestep_fs=args.timestep_fs,
                      temperature_K=temperature, threshold_A=CLOSE_CONTACT_A,
                      trajectories=rows))
    log = rp.write(str(out_stem) + ".log")
    written = report.write_parquet(dict(trajectories=rows), out_stem)
    print("\nwritten:\n  {}".format(log))
    for p, n, _c in written:
        print("  {}  ({} rows)".format(p, n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
