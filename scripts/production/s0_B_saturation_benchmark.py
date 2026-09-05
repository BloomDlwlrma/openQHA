"""How long must a branch B trajectory be? Answered by a saturation curve, not a guess.

PRODUCTION SUPPORT. It produces the curve that acceptance criterion 1 is written on, and
the number it yields -- the production length -- goes into `configs/openqha.yaml`.

One trajectory, not a ladder
----------------------------
The obvious way to compare 200 ps, 500 ps and 1 ns is to run three trajectories. Do not:
that costs 1.7 ns of force calls to learn what 1 ns already contains. A saturation curve
is `T*S` evaluated on PREFIXES of one trajectory, so a single 1 ns run answers every point
on the ladder at once, and the points share a trajectory rather than differing by both
length and seed.

    run 1 ns per seed  ->  analyse the first 25, 50, 100, 200, 500, 1000 ps
                       ->  plot T*S against prefix length
                       ->  the production length is where the last doubling moves T*S by
                           less than the budget

The seeds still differ, and they must: the spread between seeds at a fixed prefix is
acceptance criterion 5, the blank control, and without it a smooth curve proves nothing.

What the literature says to expect
----------------------------------
This is not a fast-converging quantity, and the published record is blunt about it:

  * Andricioaei & Karplus 2001 (JCP 115, 6289), 256 Lennard-Jones particles: "convergence
    is not complete over 200 ps, and about 4 ns are needed". Their 4 ns run took 276
    minutes in 2001.
  * Rinaldo & Field 2003 (Biophys. J. 85, 3485), the protein this branch's protocol comes
    from: 520 ps of EQUILIBRATION was necessary where 20 ps was planned, then 1.5 ns of
    data collection sampled every 0.5 ps -- and the entropies were still "converging but
    have not yet converged". Their per-batch-of-modes entropies converged quickly, which
    is why `qha.mode_batch_convergence` exists and is acceptance criterion 10.

Branch B's molecules are 10 to 19 atoms rather than a solvated protein, so far fewer soft
modes have to be sampled and convergence should be much faster. SHOULD BE is not a length.
Measured here so far: at 25 ps the last doubling still moved `T*S` by +0.4684 kcal/mol
against a 0.3 budget -- not converged, exactly as the literature would predict.

    python scripts/production/s0_B_saturation_benchmark.py --tag sat01 --analyse-only
    python scripts/production/s0_B_saturation_benchmark.py --tag sat01 --prod-ps 1000 \\
        --seeds 3 --plot
"""
import argparse
import json
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

from openqha import config, qha, report  # noqa: E402

#: The ladder, in ps. Read as prefixes of ONE trajectory, so the longest is the only cost.
DEFAULT_LADDER = (25.0, 50.0, 100.0, 200.0, 500.0, 1000.0)
#: Acceptance criterion 1's budget on the last doubling.
SATURATION_BUDGET_KCAL = 0.3


def prefix_curve(frames, masses, temperature_K, frame_spacing_fs, ladder):
    """`T*S` on the first N ps of a trajectory, for each N in the ladder."""
    rows = []
    total_ps = len(frames) * frame_spacing_fs / 1000.0
    for ps in ladder:
        if ps > total_ps + 1e-9:
            rows.append(dict(ps=float(ps), n_frames=None, TS_QH_kcal=None,
                             note="beyond this trajectory ({:.1f} ps)".format(total_ps)))
            continue
        n = max(2, int(round(ps * 1000.0 / frame_spacing_fs)))
        rec = qha.analyse(frames[:n], masses, temperature_K)
        rows.append(dict(ps=float(ps), n_frames=int(n),
                         TS_QH_kcal=rec["entropy"]["TS_QH_kcal"],
                         lowest_frequency_cm_inv=rec["entropy"]["lowest_frequency_cm_inv"],
                         n_nonzero=rec["spectrum"]["n_nonzero_eigenvalues"]))
    return rows


def last_doubling(rows):
    """The increment over the last doubling of length -- criterion 1's quantity."""
    done = [r for r in rows if r.get("TS_QH_kcal") is not None]
    if len(done) < 2:
        return None
    a, b = done[-2], done[-1]
    if abs(b["ps"] / max(a["ps"], 1e-9) - 2.0) > 0.25:
        # The ladder is meant to double. If it does not, say so rather than reporting an
        # increment over an unstated interval.
        return dict(from_ps=a["ps"], to_ps=b["ps"],
                    increment_kcal=float(b["TS_QH_kcal"] - a["TS_QH_kcal"]),
                    is_a_doubling=False)
    return dict(from_ps=a["ps"], to_ps=b["ps"],
                increment_kcal=float(b["TS_QH_kcal"] - a["TS_QH_kcal"]),
                is_a_doubling=True)


def load(tag, species, outroot=None):
    """Every (basin, seed) under a tag, with its metadata."""
    root = Path(outroot) if outroot else (
        Path.home() / "runs" / "openQHA" / "qha" / tag / species)
    out = []
    for meta_path in sorted(root.rglob("meta.json")):
        d = meta_path.parent
        frames_path = d / "frames.npy"
        if not frames_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        out.append(dict(path=str(d), meta=meta,
                        frames=np.load(frames_path),
                        masses=np.asarray(meta["masses_amu"], dtype=float)))
    return root, out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species", default="dsgdb9nsd_000018")
    ap.add_argument("--tag", default="sat01")
    ap.add_argument("--ladder", nargs="*", type=float, default=list(DEFAULT_LADDER))
    ap.add_argument("--outroot", default=None)
    ap.add_argument("--plot", action="store_true",
                    help="write a PNG of the curve as well as the table")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root, trajs = load(args.tag, args.species, args.outroot)
    if not trajs:
        raise SystemExit(
            "no trajectories under {}.\nProduce them first, e.g.\n"
            "  python scripts/production/s0_B_qha_trajectory.py --species {} --tag {} "
            "--seeds 3 --prod-ps {}\n"
            "or the OpenMM route (different interpreter, see docs/branchB_workflow.md)."
            .format(root, args.species, args.tag, max(args.ladder)))

    ladder = sorted(args.ladder)
    print("=" * 92)
    print("Branch B -- saturation curve")
    print("=" * 92)
    print("species   {}   {} trajectory(ies) under {}".format(
        args.species, len(trajs), root))
    print("ladder    {} ps  (PREFIXES of one trajectory, not separate runs)".format(
        ", ".join("{:g}".format(x) for x in ladder)))
    print()

    per_traj = []
    for t in trajs:
        meta = t["meta"]
        spacing = float(meta.get("frame_spacing_fs", 8.0))
        rows = prefix_curve(t["frames"], t["masses"], float(meta["temperature_K"]),
                            spacing, ladder)
        ld = last_doubling(rows)
        per_traj.append(dict(path=t["path"], seed=meta.get("seed"),
                             route=meta.get("route", "ase"),
                             thermostat=meta.get("thermostat"),
                             total_ps=len(t["frames"]) * spacing / 1000.0,
                             curve=rows, last_doubling=ld))
        print("seed {}  ({:.1f} ps, {})".format(
            meta.get("seed"), per_traj[-1]["total_ps"], per_traj[-1]["route"]))
        for r in rows:
            if r.get("TS_QH_kcal") is None:
                print("    {:>7.0f} ps   {}".format(r["ps"], r["note"]))
            else:
                print("    {:>7.0f} ps   T*S = {:8.4f} kcal/mol   lowest {:7.2f} cm-1   "
                      "modes {}".format(r["ps"], r["TS_QH_kcal"],
                                        r["lowest_frequency_cm_inv"], r["n_nonzero"]))
        if ld:
            print("    last doubling {:g} -> {:g} ps: {:+.4f} kcal/mol   {}".format(
                ld["from_ps"], ld["to_ps"], ld["increment_kcal"],
                "WITHIN budget" if abs(ld["increment_kcal"]) < SATURATION_BUDGET_KCAL
                else "OVER the {} kcal/mol budget".format(SATURATION_BUDGET_KCAL)))
        print()

    # ---- the blank control at each prefix: criterion 5, as a function of length --------
    spread = []
    for i, ps in enumerate(ladder):
        vals = [p["curve"][i]["TS_QH_kcal"] for p in per_traj
                if p["curve"][i].get("TS_QH_kcal") is not None]
        if len(vals) > 1:
            spread.append(dict(ps=float(ps), n_seeds=len(vals),
                               mean_TS_kcal=float(np.mean(vals)),
                               seed_rms_kcal=float(np.std(vals, ddof=1))))
    if spread:
        print("between-seed spread at each prefix (acceptance criterion 5):")
        for s in spread:
            print("    {:>7.0f} ps   mean T*S = {:8.4f}   seed RMS = {:.4f} kcal/mol "
                  "({} seeds)".format(s["ps"], s["mean_TS_kcal"], s["seed_rms_kcal"],
                                      s["n_seeds"]))
        print()
        print("A curve is converged when the last doubling moves T*S by less than the")
        print("budget AND by less than this spread. A smooth curve from one seed proves")
        print("nothing: it can be smoothly wrong.")

    # ---- the recommendation ------------------------------------------------------------
    recommended = None
    for i in range(1, len(ladder)):
        incs = []
        for p in per_traj:
            a, b = p["curve"][i - 1], p["curve"][i]
            if a.get("TS_QH_kcal") is not None and b.get("TS_QH_kcal") is not None:
                incs.append(abs(b["TS_QH_kcal"] - a["TS_QH_kcal"]))
        if incs and max(incs) < SATURATION_BUDGET_KCAL:
            recommended = ladder[i]
            break
    print()
    if recommended:
        print("shortest prefix at which EVERY trajectory's last doubling is inside the "
              "{} kcal/mol budget: {:g} ps".format(SATURATION_BUDGET_KCAL, recommended))
        print("Put that in configs/openqha.yaml as the production length, with this "
              "run's log as its source.")
    else:
        print("NO prefix on this ladder is converged. The production length is longer "
              "than {:g} ps -- extend the trajectories rather than the budget."
              .format(max(ladder)))
        print("For scale: Andricioaei & Karplus needed about 4 ns where 200 ps was not "
              "converged; Rinaldo & Field's 1.5 ns was still 'converging but not yet "
              "converged'.")

    # ---- products ------------------------------------------------------------------------
    out_stem = Path(args.out) if args.out else (
        ROOT / "analysis" / "qha" / "saturation_{}".format(args.tag))
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for p in per_traj:
                xs = [r["ps"] for r in p["curve"] if r.get("TS_QH_kcal") is not None]
                ys = [r["TS_QH_kcal"] for r in p["curve"]
                      if r.get("TS_QH_kcal") is not None]
                ax.plot(xs, ys, marker="o", label="seed {}".format(p["seed"]))
            ax.set_xscale("log")
            ax.set_xlabel("trajectory prefix / ps")
            ax.set_ylabel(r"$T\,S_{\rm QH}$ / kcal mol$^{-1}$")
            ax.set_title("Branch B saturation curve -- {}".format(args.species))
            ax.grid(alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(str(out_stem) + ".png", dpi=150)
            print("\nplot: {}.png".format(out_stem))
        except Exception as exc:                              # noqa: BLE001
            print("\nplot skipped: {}: {}".format(type(exc).__name__, exc))

    rp = report.Report("Branch B -- saturation curve",
                       subtitle="{}   tag {}   {} trajectories".format(
                           args.species, args.tag, len(per_traj)))
    rp.section("Why prefixes and not separate runs")
    rp.note("A saturation curve is T*S on PREFIXES of one trajectory, so one 1 ns run "
            "answers every point on the ladder. Three separate runs would cost 1.7 ns of "
            "force calls to learn what 1 ns already contains, and its points would differ "
            "by both length and seed.")
    rp.section("What the literature expects")
    rp.note("Andricioaei & Karplus 2001 (JCP 115, 6289) on 256 Lennard-Jones particles: "
            "'convergence is not complete over 200 ps, and about 4 ns are needed'. "
            "Rinaldo & Field 2003 (Biophys. J. 85, 3485), whose protocol this branch "
            "follows: 520 ps of equilibration was necessary where 20 ps was planned, and "
            "after 1.5 ns of data collection the entropies were 'converging but have not "
            "yet converged'. Branch B's molecules are 10-19 atoms rather than a solvated "
            "protein, so this should converge far sooner -- which is a reason to measure "
            "it, not a length.")
    rp.kv("saturation_budget_kcal", SATURATION_BUDGET_KCAL)
    rp.kv("recommended_production_ps", recommended or "not converged on this ladder")
    rp.section("Curve")
    rp.table(["seed", "route"] + ["{:g} ps".format(x) for x in ladder],
             [[p["seed"], p["route"]] +
              [("--" if r.get("TS_QH_kcal") is None
                else round(r["TS_QH_kcal"], 4)) for r in p["curve"]]
              for p in per_traj])
    if spread:
        rp.section("Between-seed spread (criterion 5) at each prefix")
        rp.table(["prefix", "seeds", "mean T*S", "seed RMS"],
                 [[s["ps"], s["n_seeds"], round(s["mean_TS_kcal"], 4),
                   round(s["seed_rms_kcal"], 4)] for s in spread],
                 units=["ps", None, "kcal/mol", "kcal/mol"])
    rp.json_dump(dict(species=args.species, tag=args.tag, ladder=ladder,
                      budget_kcal=SATURATION_BUDGET_KCAL,
                      recommended_production_ps=recommended,
                      trajectories=per_traj, between_seed_spread=spread))
    log = rp.write(str(out_stem) + ".log")
    written = report.write_parquet(
        dict(saturation=[dict(seed=p["seed"], route=p["route"], **r)
                         for p in per_traj for r in p["curve"]],
             between_seed=spread), out_stem)
    print("\nwritten:\n  {}".format(log))
    for p, n, _c in written:
        print("  {}  ({} rows)".format(p, n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
