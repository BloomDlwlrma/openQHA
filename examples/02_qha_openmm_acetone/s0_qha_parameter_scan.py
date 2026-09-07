"""Which production length, sampling interval, friction and atom set? Measure, do not adopt.

CALIBRATION. It measures properties of the PROTOCOL in order to choose four settings, and
it produces no scientific number of its own.

WHY THIS EXISTS
---------------
On 2026-09-07 the published protocol of Rinaldo & Field (*Biophys. J.* 2003) was adopted
wholesale for branch B: 520 ps equilibration, 1.5 ns production, one frame per 0.5 ps,
friction 50/ps. **That was wrong, and the reason is the system, not the paper.** Theirs is
a 2586-heavy-atom solvated protein whose basin is barely explored in 1.5 ns. Ours is 10 to
19 atoms in vacuum, where 1.5 ns is long enough to cross torsional barriers many times --
and a trajectory that changes basin is no longer measuring an intra-basin entropy at all.

So the four settings are scanned rather than inherited.

THE TWO STAGES ANSWER DIFFERENT QUESTIONS, AND ONLY ONE OF THEM NEEDS MD
------------------------------------------------------------------------
    --stage estimator   Is the covariance ESTIMATOR accurate at N frames?
                        `qha.synthetic_harmonic_trajectory` draws from the exact
                        classical canonical distribution of this molecule's own Hessian,
                        so the true T*S is known in closed form and the error is purely
                        finite-sample. **Seconds, and decidable.**

    --stage dynamics    Does the real trajectory STAY IN ITS BASIN at that length, and
                        does the thermostat change the answer? No closed form exists, so
                        this can only be measured. **Hours to days -- run it on Tianhe.**

(length, interval) enters the estimator only as `n_frames = length / interval`, so one
scan over frame count answers every combination at once. And in the dynamics stage,
length, interval and atom selection are all POST-HOC on one trajectory: only the
thermostat needs separate runs. That is what makes the scan affordable -- 1 trajectory
per thermostat setting, not 1 per cell of a four-way grid.

    python examples/02_qha_openmm_acetone/s0_qha_parameter_scan.py --stage estimator
    python examples/02_qha_openmm_acetone/s0_qha_parameter_scan.py --stage dynamics \\
        --prod-ps 1000 --thermostats langevin_1 langevin_5 langevin_50 nhc_20 nhc_100
"""
import argparse
import json
import subprocess
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

from openqha import config                                          # noqa: E402
from openqha.quasi_harmonic import basin_residence as br            # noqa: E402
from openqha.quasi_harmonic import qha                              # noqa: E402

#: The grid the user asked for. Lengths in ps, intervals in ps.
LENGTHS_PS = (500.0, 750.0, 1000.0, 1500.0)
INTERVALS_PS = (0.5, 1.0, 2.0)
ATOM_SETS = ("all", "heavy")

#: Thermostat settings. The name collision is the point: the paper's "friction 50/ps" is
#: a LANGEVIN friction, while openQHA's production `collision_frequency = 50/ps` is a
#: Nose-Hoover CHAIN coupling of 20 fs. Different algorithms, different meanings, same
#: number -- so both axes are here.
THERMOSTATS = {
    "langevin_1": dict(thermostat="langevin", friction=1.0),
    "langevin_5": dict(thermostat="langevin", friction=5.0),
    "langevin_50": dict(thermostat="langevin", friction=50.0),
    "nhc_20": dict(thermostat="nose-hoover", tdamp_fs=20.0),     # production
    "nhc_100": dict(thermostat="nose-hoover", tdamp_fs=100.0),
}


# ======================================================================================
def relaxed_and_hessian(species, fmax=1e-3):
    """Relax on the production potential, then take its analytic Hessian.

    The relaxation is not optional: `synthetic_harmonic_trajectory` refuses a Hessian
    with non-positive vibrational eigenvalues, and a QM9 deposited geometry is a minimum
    of B3LYP, not of MACE.
    """
    from ase.io import read
    from ase.optimize import BFGS

    from openqha.potentials import engine
    from openqha.thermochem import hessian as H

    atoms = read(str(config.qm9_xyz(species)))
    calc, name, _prov = engine.calculator(device="cpu")
    atoms.calc = calc
    BFGS(atoms, logfile=None).run(fmax=fmax, steps=500)
    h, asym = H.analytic_hessian(atoms, calc)
    return atoms, np.asarray(h, dtype=float), name, float(asym)


def subset(frames, masses, symbols, which):
    """(frames, masses) restricted to an atom set. `heavy` is the source paper's choice."""
    if which == "all":
        return frames, masses
    m = br.heavy_atom_mask(symbols)
    return np.asarray(frames)[:, m, :], np.asarray(masses)[m]


# ======================================================================================
def stage_estimator(args):
    """Finite-sample error of the covariance estimator, against the closed form."""
    atoms, hess, engine_name, asym = relaxed_and_hessian(args.species)
    masses, syms = atoms.get_masses(), atoms.get_chemical_symbols()
    pos = atoms.get_positions()

    print("=" * 92)
    print("STAGE 1 -- ESTIMATOR.  {}  {} atoms.  {} analytic Hessian "
          "(max|H-H^T| = {:.1e})".format(args.species, len(atoms), engine_name, asym))
    print("The frames are drawn from the EXACT distribution, so every error below is")
    print("finite-sample. {} independent draws per row.".format(args.seeds))
    print("=" * 92)

    out = {}
    for which in ATOM_SETS:
        m = br.heavy_atom_mask(syms) if which == "heavy" else np.ones(len(syms), bool)
        hm = np.repeat(m, 3)
        h_s, mass_s, pos_s = hess[np.ix_(hm, hm)], masses[m], pos[m]
        dof = 3 * int(m.sum()) - 6
        _f, ref = qha.synthetic_harmonic_trajectory(h_s, mass_s, pos_s, n_frames=8, seed=1)
        truth = ref["closed_form_TS_kcal"]
        print("\n{} ATOMS ({} of {}), {} modes, closed-form T*S = {:.4f} kcal/mol".format(
            which.upper(), int(m.sum()), len(syms), ref["n_modes"], truth))
        print("{:>8} {:>9} {:>8} {:>11} {:>10} {:>10} {:>9}".format(
            "len/ps", "step/ps", "frames", "frames/DOF", "mean T*S", "bias", "spread"))
        rows, seen = [], set()
        for length in LENGTHS_PS:
            for interval in INTERVALS_PS:
                n = int(round(length / interval))
                if n in seen:
                    continue
                seen.add(n)
                vals = [qha.analyse(qha.synthetic_harmonic_trajectory(
                            h_s, mass_s, pos_s, n_frames=n,
                            seed=20260907 + 1000 * s)[0], mass_s)["entropy"]["TS_QH_kcal"]
                        for s in range(args.seeds)]
                v = np.asarray(vals)
                rows.append(dict(length_ps=length, interval_ps=interval, n_frames=n,
                                 frames_per_dof=n / dof, mean_TS_kcal=float(v.mean()),
                                 bias_kcal=float(v.mean() - truth),
                                 seed_spread_kcal=float(v.std())))
                print("{:>8.0f} {:>9} {:>8} {:>11.1f} {:>10.4f} {:>+10.4f} {:>9.4f}".format(
                    length, interval, n, n / dof, v.mean(), v.mean() - truth, v.std()))
        out[which] = dict(closed_form_TS_kcal=truth, n_modes=ref["n_modes"],
                          degrees_of_freedom=dof, rows=rows)

    print("\nREAD THIS TABLE AS: bias = is the estimator systematically wrong;")
    print("spread = how much one trajectory differs from the next. They are different")
    print("failures and only the second is fixed by more frames.")
    print("\nIt says NOTHING about whether real dynamics stays in the basin -- stage 2.")
    return out


# ======================================================================================
def run_trajectory(species, tag, setting, prod_ps, equil_ps, interval_ps, seed_index):
    """One production trajectory, through the PRODUCTION driver as a subprocess.

    A subprocess of the real driver rather than MD written here, so that the scan cannot
    accidentally measure a different protocol from the one production runs -- which is
    exactly the failure this scan exists to correct.
    """
    steps = int(round(interval_ps * 1000.0))     # at 1 fs
    cmd = [sys.executable, "-u",
           str(ROOT / "scripts" / "production" / "s0_B_qha_trajectory.py"),
           "--species", species, "--tag", tag,
           "--basin", "0", "--seed-index", str(seed_index),
           "--prod-ps", str(prod_ps), "--equil-ps", str(equil_ps),
           "--sample-every", str(steps),
           "--thermostat", setting["thermostat"]]
    if setting["thermostat"] == "langevin":
        cmd += ["--friction", str(setting["friction"])]
    else:
        cmd += ["--tdamp-fs", str(setting["tdamp_fs"])]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    return proc, time.time() - t0, cmd


def stage_dynamics(args):
    """Does the trajectory stay in its basin, and does the thermostat matter?"""
    from ase.io import read
    atoms = read(str(config.qm9_xyz(args.species)))
    syms = atoms.get_chemical_symbols()
    masses = atoms.get_masses()

    print("=" * 92)
    print("STAGE 2 -- DYNAMICS.  {}  {} atoms".format(args.species, len(atoms)))
    print("One trajectory per thermostat, at the LONGEST length and FINEST interval.")
    print("Length, interval and atom set are then scanned POST HOC by truncating and")
    print("striding that one trajectory -- which is what makes this affordable.")
    print("=" * 92)

    longest = max(LENGTHS_PS if args.prod_ps is None else [args.prod_ps])
    finest = min(INTERVALS_PS)
    results = {}
    for name in args.thermostats:
        setting = THERMOSTATS[name]
        tag = "{}_{}".format(args.tag, name)
        print("\n-- {}  ({})  {} ps at {} ps spacing".format(
            name, setting, longest, finest))
        proc, wall, cmd = run_trajectory(args.species, tag, setting, longest,
                                         args.equil_ps, finest, 0)
        if proc.returncode != 0:
            print("   FAILED: {}".format((proc.stderr or proc.stdout or "")[-400:]))
            results[name] = dict(error=(proc.stderr or "")[-2000:], command=" ".join(cmd))
            continue
        runs = Path(config.runs_dir("qha")) / tag / args.species / "basin00" / "seed00"
        frames = np.load(runs / "frames.npy")
        print("   {} frames in {:.0f} s".format(len(frames), wall))

        rows = []
        for length in (LENGTHS_PS if args.prod_ps is None else [args.prod_ps]):
            for interval in INTERVALS_PS:
                stride = int(round(interval / finest))
                keep = int(round(length / finest))
                sub = frames[:keep:stride]
                if len(sub) < 3 * len(atoms):
                    continue
                res = br.basin_residence(sub, syms)
                for which in ATOM_SETS:
                    f_s, m_s = subset(sub, masses, syms, which)
                    rec = qha.analyse(f_s, m_s)
                    sat = qha.saturation_curve(f_s, m_s)
                    verdict = br.interpret_saturation(sat, res)
                    rows.append(dict(
                        length_ps=length, interval_ps=interval, atoms=which,
                        n_frames=int(len(sub)),
                        TS_QH_kcal=rec["entropy"]["TS_QH_kcal"],
                        n_modes=rec["spectrum"]["n_nonzero_eigenvalues"],
                        rank_is_full=rec["rank_check"]["rank_is_full"],
                        distinct_basin_crossings=res["distinct_basin_crossings"],
                        symmetry_equivalent_crossings=res["symmetry_equivalent_crossings"],
                        saturation_rise_kcal=verdict[
                            "increment_over_last_doubling_kcal"],
                        criterion_1_passed=verdict["criterion_1_passed"],
                        verdict=verdict["verdict"], action=verdict["action"]))
        results[name] = dict(setting=setting, wall_seconds=wall, rows=rows,
                             command=" ".join(cmd))

        print("   {:>7} {:>8} {:>6} {:>7} {:>10} {:>7} {:>7}  {}".format(
            "len/ps", "step/ps", "atoms", "frames", "T*S kcal", "xing", "symxing",
            "criterion 1"))
        for r in rows:
            print("   {:>7.0f} {:>8} {:>6} {:>7} {:>10.4f} {:>7} {:>7}  {}".format(
                r["length_ps"], r["interval_ps"], r["atoms"], r["n_frames"],
                r["TS_QH_kcal"], r["distinct_basin_crossings"],
                r["symmetry_equivalent_crossings"],
                "pass" if r["criterion_1_passed"] else "FAIL"))
    return results


# ======================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species", default="dsgdb9nsd_000018")
    ap.add_argument("--stage", default="estimator",
                    choices=("estimator", "dynamics", "both"))
    ap.add_argument("--seeds", type=int, default=8,
                    help="draws per row in the estimator stage. One draw cannot tell "
                         "bias from scatter.")
    ap.add_argument("--tag", default="scan")
    ap.add_argument("--prod-ps", type=float, default=None,
                    help="dynamics stage: run this length instead of the longest in the "
                         "grid. Use a small value for a smoke test.")
    ap.add_argument("--equil-ps", type=float, default=50.0)
    ap.add_argument("--thermostats", nargs="*", default=["nhc_20", "langevin_1"],
                    choices=sorted(THERMOSTATS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    record = dict(species=args.species, stage=args.stage,
                  protocol_in_config=qha.protocol(),
                  lengths_ps=list(LENGTHS_PS), intervals_ps=list(INTERVALS_PS),
                  atom_sets=list(ATOM_SETS),
                  thermostats={k: THERMOSTATS[k] for k in args.thermostats})

    if args.stage in ("estimator", "both"):
        record["estimator"] = stage_estimator(args)
    if args.stage in ("dynamics", "both"):
        record["dynamics"] = stage_dynamics(args)

    out = Path(args.out) if args.out else (
        ROOT / "analysis" / "qha" / "parameter_scan_{}_{}.json".format(
            args.species, args.stage))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    print("\nwritten {}".format(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
