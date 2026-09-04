"""Run branch B over many (basin, seed) trajectories through Parsl. Branch E, for branch B.

PRODUCTION. This is the execution layer: it decides where branch B's trajectories run and
how many run at once. **It computes nothing.** Every scientific decision -- the timestep,
the friction, the temperature, the sampling interval, the three prohibitions -- is made
inside `scripts/production/s0_B_qha_trajectory.py`, and this script must be replaceable by
a for-loop without changing a single number. That is branch E acceptance criterion 1, and
the only way to keep it true is to keep this file free of chemistry.

Why the unit of work is (basin, seed) and not (species)
-------------------------------------------------------
Branch B's protocol is three independent seeds per basin, because the seed-to-seed spread
IS acceptance criterion 5 -- the blank control that decides whether branch B can resolve an
edge at all. The independent work therefore exists by construction, and there is never a
reason to look for parallelism inside one trajectory. It is also the smallest resumable
unit: the driver flushes frames in chunks and reloads them on restart, so a killed task
costs at most one chunk.

Executor
--------
`openqha_qha`, not `openqha_crest`. They differ in cores per worker for a measured reason:
CREST parallelises its own metadynamics runs, so it gets 4 threads; a quasi-harmonic
trajectory is a serial chain of single-structure MACE calls, and MACE's thread scaling on a
10-atom molecule is 111/90/72/101 ms at 1/2/4/8 threads. Four threads buy 1.54x, four
trajectories buy 4x. Sending branch B work to the CREST executor would hold three idle
cores per trajectory.

    python scripts/production/s0_E_branchB_parsl.py --species dsgdb9nsd_000018 --dry-run
    python scripts/production/s0_E_branchB_parsl.py --edges --resource deimos --account X
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))


def _hpc_root(repo):
    """Directory holding the Parsl resource configurations, found by searching.

    NOT `repo / "hpc"`. That was the previous version, and it broke silently when
    the directory was moved to `examples/hpc/` -- and then moved back. Searching
    survives both, which is the point: the directory has now moved twice and a literal
    name would have had to be edited twice.

    This is the same failure as the `scripts/` reorganisation and the `source-code/`
    move (S0-G-30): `_repo_root()` fixed the package import, and every path written as
    a literal string stayed behind. **A name written into a string does not move with
    the thing it names.**
    """
    for cand in sorted(repo.rglob("resource_configs/local.py")):
        if "_superseded" in cand.parts or "_backup" in cand.parts:
            continue
        return cand.parent.parent
    raise RuntimeError(
        "no resource_configs/local.py found under {} -- the Parsl resource "
        "configurations are missing, not merely moved".format(repo))

sys.path.insert(0, str(_hpc_root(ROOT)))

from openqha import config  # noqa: E402


# ======================================================================================
# The task. It runs in a worker process, so it must be self-contained.
# ======================================================================================
def run_one_trajectory(species, basin, seed_index, seed0, repo_root, tag, prod_ps,
                       equil_ps, wall_budget_s, basins_file=None, env=None):
    """One (basin, seed) trajectory, as a SUBPROCESS of the branch B driver.

    A subprocess rather than an import, for the same three reasons branch A uses one:
    a crash or an out-of-memory kill takes down one task and not the worker's whole queue;
    the driver's stdout is captured verbatim per trajectory; and the command line that
    produced a result is recorded literally, which is what defect 57 was missing.
    """
    import json as _json
    import os as _os
    import subprocess as _sp
    import sys as _sys      # imported locally: this function is serialised into a worker
    import time as _time
    from pathlib import Path as _Path

    repo_root = _Path(repo_root)
    started = _time.time()
    e = dict(_os.environ)
    e.update(env or {})
    # One thread per trajectory. Set here as well as in hpc/env/common.sh, because a
    # worker that inherited a different environment would silently oversubscribe and
    # every cost measured in that run would be meaningless.
    e.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")

    cmd = [_sys.executable, str(repo_root / "scripts" / "production"
                                / "s0_B_qha_trajectory.py"),
           "--species", species, "--tag", tag,
           "--basin", str(basin), "--seed-index", str(seed_index),
           "--seed0", str(seed0),
           "--prod-ps", str(prod_ps), "--equil-ps", str(equil_ps)]
    if wall_budget_s:
        cmd += ["--wall-budget-s", str(wall_budget_s)]
    if basins_file:
        cmd += ["--basins", str(basins_file)]

    proc = _sp.run(cmd, cwd=str(repo_root), env=e, text=True,
                   stdout=_sp.PIPE, stderr=_sp.PIPE)
    summary = dict(species=species, basin=basin, seed_index=seed_index,
                   command=" ".join(cmd), returncode=proc.returncode,
                   seconds=_time.time() - started)

    out = (repo_root / "analysis" / "qha" / tag / species
           / "basin{:02d}_seed{}".format(basin, seed_index))
    out.mkdir(parents=True, exist_ok=True)
    (out / "driver.log").write_text(
        proc.stdout + "\n----- stderr -----\n" + proc.stderr, encoding="utf-8")

    if proc.returncode != 0:
        summary["error"] = (proc.stderr or proc.stdout or "")[-2000:]
        return summary

    # The driver's own record is the source of truth; only enough is lifted out of it to
    # build the summary table.
    runs_root = _Path(e.get("S0_RUNS_ROOT", _os.path.expanduser("~/runs/openQHA")))
    meta_path = (runs_root / "qha" / tag / species / "basin{:02d}".format(basin)
                 / "seed{:02d}".format(seed_index) / "meta.json")
    if meta_path.exists():
        meta = _json.loads(meta_path.read_text(encoding="utf-8"))
        prod = meta.get("production", {})
        summary.update(n_frames=prod.get("n_frames"), complete=prod.get("complete"),
                       stopped_on_wall_budget=prod.get("stopped_on_wall_budget"),
                       temperature_mean_K=prod.get("temperature_mean_K"),
                       temperature_deviation_K=prod.get("temperature_deviation_K"),
                       seconds_per_ps=prod.get("seconds_per_ps_this_run"),
                       meta_path=str(meta_path))
    return summary


# ======================================================================================
def species_list(args, cfg):
    if args.species:
        return list(args.species), "given on the command line"
    if args.edges:
        out = []
        for edge in config.edges(cfg):
            for qid in config.edge_species(edge):
                if qid not in out:
                    out.append(qid)
        return out, "both species of every edge in configs/openqha.yaml"
    raise SystemExit("give --species or --edges")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species", nargs="*", default=None)
    ap.add_argument("--edges", action="store_true",
                    help="run every species named by the configured edge set")
    ap.add_argument("--resource", default="local",
                    help="hpc/resource_configs/<name>.py (default: local -- step 0)")
    ap.add_argument("--tag", default="prod")
    ap.add_argument("--basins", type=int, default=1,
                    help="how many basins per species to run (branch A supplies the "
                         "geometries; without it there is one)")
    ap.add_argument("--seeds", type=int, default=3,
                    help="independent trajectories per basin. 3 is the minimum that "
                         "gives a blank control, which is acceptance criterion 5")
    ap.add_argument("--seed0", type=int, default=20260903)
    ap.add_argument("--prod-ps", type=float, default=200.0)
    ap.add_argument("--equil-ps", type=float, default=50.0)
    ap.add_argument("--max-workers", type=int, default=None)
    ap.add_argument("--wall-budget-s", type=float, default=None,
                    help="per-task budget; default comes from the resource config")
    ap.add_argument("--account", default=None, help="scheduler account (cluster only)")
    ap.add_argument("--partition", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and submit nothing")
    args = ap.parse_args()

    cfg = config.load()
    species, source = species_list(args, cfg)

    import resource_configs
    res = resource_configs.load(args.resource)
    described = res.describe()
    budget = args.wall_budget_s
    if budget is None:
        budget = getattr(res, "QHA_WALL_BUDGET_S", 0)

    tasks = [(s, b, k) for s in species
             for b in range(args.basins) for k in range(args.seeds)]

    plan = dict(
        generated_by="scripts/production/s0_E_branchB_parsl.py",
        branch="E (execution layer for branch B)",
        resource=described,
        executor="openqha_qha",
        n_species=len(species), species=species, species_source=source,
        n_tasks=len(tasks), basins_per_species=args.basins, seeds_per_basin=args.seeds,
        tag=args.tag, prod_ps=args.prod_ps, equil_ps=args.equil_ps,
        wall_budget_s=budget,
        science_settings_source=("scripts/production/s0_B_qha_trajectory.py and "
                                 "configs/openqha.yaml -- NOT this script"),
        prohibitions="no bias, no constraints, real hydrogen mass, fixcm=False",
        cost_note=("96.1 s/ps measured on this machine for a 10-atom molecule at 1 "
                   "thread, uncontended. It must NOT be divided by a worker count to "
                   "produce a cluster estimate (D0-P1-12, defects 34 and 56)."),
        estimated_single_task_seconds=round(96.1 * args.prod_ps, 1),
    )

    print("=" * 92)
    print("Branch E -- branch B over Parsl")
    print("=" * 92)
    print("resource     {}  ({} qha workers x {} thread)".format(
        args.resource, described.get("qha_max_workers")
        or described.get("qha_workers_per_node"),
        described.get("qha_threads_per_job", 1)))
    print("tasks        {}  ({} species x {} basins x {} seeds)".format(
        len(tasks), len(species), args.basins, args.seeds))
    print("length       {} ps equilibration + {} ps production".format(
        args.equil_ps, args.prod_ps))
    print("wall budget  {}".format(
        "{} s per task".format(int(budget)) if budget else
        "none -- a task runs to completion (this machine has no queue to be killed by)"))
    print("single task  about {:.0f} s at the measured 96.1 s/ps (10 atoms, "
          "uncontended)".format(96.1 * args.prod_ps))
    print("tag          {}".format(args.tag))
    print()

    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0

    kw = {}
    if args.max_workers:
        kw["qha_max_workers"] = args.max_workers
    if args.account:
        kw["account"] = args.account
    if args.partition:
        kw["partition"] = args.partition
    parsl_config = res.config(**kw)

    import parsl
    from parsl import python_app
    parsl.load(parsl_config)
    app = python_app(run_one_trajectory, executors=["openqha_qha"])

    passthrough = {k: os.environ[k] for k in
                   ("S0_RUNS_ROOT", "S0_ENGINE", "S0_MACE_MODEL", "S0_CONFIG",
                    "S0_GMX_BIN", "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
                   if k in os.environ}

    started = time.time()
    futures = [app(s, b, k, args.seed0, str(ROOT), args.tag, args.prod_ps, args.equil_ps,
                   budget, env=passthrough) for s, b, k in tasks]
    results = []
    for f in futures:
        try:
            results.append(f.result())
        except Exception as exc:                    # record it, do not lose the batch
            import traceback
            results.append(dict(error_type=type(exc).__name__, error=str(exc),
                                traceback=traceback.format_exc()))
    wall = time.time() - started
    parsl.dfk().cleanup()

    print("{:<20} {:>6} {:>8} {:>8} {:>10} {:>10}  {}".format(
        "species", "basin", "seed", "frames", "T mean K", "seconds", "complete"))
    for r in results:
        if r.get("error"):
            print("{:<20} {}".format(r.get("species", "?"), str(r["error"])[:60]))
            continue
        print("{:<20} {:>6} {:>8} {:>8} {:>10} {:>10.1f}  {}".format(
            r["species"], r["basin"], r["seed_index"], r.get("n_frames", "-"),
            "-" if r.get("temperature_mean_K") is None
            else "{:.2f}".format(r["temperature_mean_K"]),
            r["seconds"], r.get("complete")))

    # Branch E acceptance criterion 5: three numbers, kept apart. The third one is
    # deliberately NOT computed -- a slot extrapolation is a claim about contention that
    # nothing here has measured.
    single = [r["seconds"] for r in results if not r.get("error")]
    print()
    print("wall_seconds_for_the_whole_batch  {:.1f}".format(wall))
    print("single_task_seconds_median        {:.1f}".format(
        sorted(single)[len(single) // 2] if single else float("nan")))
    print("slot_extrapolation                NOT COMPUTED -- the per-task cost under "
          "contention has not been measured (D0-P1-12, defects 34 and 56)")

    out = ROOT / "analysis" / "qha" / args.tag / "branchB_parsl_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(plan=plan, results=results,
                                   wall_seconds_for_the_whole_batch=wall), indent=2),
                   encoding="utf-8")
    print("\nwritten {}".format(out))
    return 0 if all(not r.get("error") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
