"""Run branch B over many (basin, seed) trajectories through Parsl. Branch E, for branch B.

PRODUCTION. This is the execution layer: it decides where branch B's trajectories run and
how many run at once. **It computes nothing.** Every scientific decision -- the timestep,
the friction, the temperature, the sampling interval, the three prohibitions -- is made
inside the trajectory drivers, and this script must be replaceable by a for-loop without
changing a single number. That is branch E acceptance criterion 1, and the only way to
keep it true is to keep this file free of chemistry.

WHERE BRANCH B RUNS, SINCE 2026-09-07
--------------------------------------
User ruling: the trajectories are a GPU job.

    trajectories   TianheXY-A, `--resource tianhe_a --route openmm --platform CUDA`
                   ONE TRAJECTORY PER CARD, 8 cards per allocation, 5 allocations =
                   40 concurrent trajectories.
    collection     TianheXY-C, `s0_E_branchB_collect_parsl.py`
                   one molecule per core, 64 at a time, ONE node.

The CPU route has not been deleted and is not deprecated: `--route ase --resource
tianhe_cpu` is the independent implementation pair that makes the OpenMM numbers
checkable. It is simply no longer the production route.

    CAVEAT, stated rather than buried: this repository's only GPU measurement of branch B
    is D0-C-5, where the same trajectory ran 3.5x SLOWER on the GPU than on the CPU. That
    was a T400 -- a 2 GB entry-level card -- against 80 GB HBM2e on TianheXY-A, so the
    number does not transfer. It has also not been replaced. **Take seconds-per-ps off a
    `--debug` run before sizing a campaign**, and do not assume the card is faster
    because it is a card.

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
`openqha_qha_executor`, from hpc/labels.py -- never a literal. A Parsl app that names an
executor the config does not provide builds cleanly, renders cleanly and then schedules
nothing; that is exactly how this file was broken until 2026-09-07, when it still asked
for the retired label `openqha_qha`.

    # smoke test: one species, one basin, one seed, 30 minutes on the short queue
    python -u scripts/production/s0_E_branchB_parsl.py --species dsgdb9nsd_000018 \
        --resource tianhe_a --route openmm --seeds 1 --prod-ps 2 --debug

    # production
    python -u scripts/production/s0_E_branchB_parsl.py --edges \
        --resource tianhe_a --route openmm
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

from openqha import config, qha  # noqa: E402
from openqha.store import basin_store  # noqa: E402

#: Read once, at import, so --help shows the real defaults.
_PROTOCOL = qha.protocol()

#: route -> the production driver that implements it.
ROUTES = {
    "openmm": "s0_B_qha_trajectory_openmm.py",
    "ase": "s0_B_qha_trajectory.py",
}


def _walltime_seconds(text):
    """Slurm walltime -> seconds. Accepts D-HH:MM:SS, HH:MM:SS, MM:SS and MM.

    Written out rather than guessed at, because the two spellings this repository uses
    are `3-00:00:00` and `00:30:00`, and a parser that handles only one of them silently
    returns a budget three orders of magnitude wrong.
    """
    text = str(text).strip()
    days = 0
    if "-" in text:
        d, _, text = text.partition("-")
        days = int(d)
    parts = [int(x) for x in text.split(":")] if text else [0]
    while len(parts) < 3:
        parts.insert(0, 0)                      # MM:SS -> 0:MM:SS, MM -> 0:0:MM
    h, m, s = parts[-3:]
    return days * 86400 + h * 3600 + m * 60 + s


def _accepted_kwargs(fn):
    """Parameter names a resource config's `config()` will accept.

    Asked rather than assumed: the resource configs are separate modules with slightly
    different signatures, and passing a keyword one of them does not take raises
    TypeError from inside Parsl's construction, where it reads as a Parsl problem.
    """
    import inspect
    return set(inspect.signature(fn).parameters)


# ======================================================================================
# The task. It runs in a worker process, so it must be self-contained.
# ======================================================================================
def run_one_trajectory(species, basin, seed_index, seed0, repo_root, tag, prod_ps,
                       equil_ps, wall_budget_s, route="ase", platform=None,
                       basins_file=None, env=None):
    """One (basin, seed) trajectory, as a SUBPROCESS of the branch B driver.

    A subprocess rather than an import, for the same three reasons branch A uses one:
    a crash or an out-of-memory kill takes down one task and not the worker's whole queue;
    the driver's stdout is captured verbatim per trajectory; and the command line that
    produced a result is recorded literally, which is what defect 57 was missing.

    `route` picks which of the two production drivers runs. They write the same
    `frames.npy` + `meta.json` contract, so the analysis reads either without knowing
    which produced it -- that is what makes them an implementation pair rather than a
    fork.
    """
    import json as _json
    import os as _os
    import subprocess as _sp
    import sys as _sys      # imported locally: this function is serialised into a worker
    import time as _time
    from pathlib import Path as _Path

    ROUTE_SCRIPTS = {"openmm": "s0_B_qha_trajectory_openmm.py",
                     "ase": "s0_B_qha_trajectory.py"}
    if route not in ROUTE_SCRIPTS:
        raise ValueError("unknown route {!r}; known: {}".format(
            route, sorted(ROUTE_SCRIPTS)))

    repo_root = _Path(repo_root)
    started = _time.time()
    e = dict(_os.environ)
    e.update(env or {})
    # One CPU thread per trajectory. Set here as well as in hpc/env/common.sh, because a
    # worker that inherited a different environment would silently oversubscribe and
    # every cost measured in that run would be meaningless.
    #
    # This stays 1 on the GPU route too. There the force call happens on the card and the
    # CPU thread only feeds it, so extra threads buy contention with the seven other
    # workers on the node rather than speed.
    e.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")

    # Which card this worker was given. parsl sets CUDA_VISIBLE_DEVICES per worker via
    # `available_accelerators`; recorded here so a product says which card produced it
    # and a mis-pinning (all workers on card 0) is visible in the record rather than only
    # in the wall clock.
    visible = e.get("CUDA_VISIBLE_DEVICES")

    cmd = [_sys.executable, str(repo_root / "scripts" / "production"
                                / ROUTE_SCRIPTS[route]),
           "--species", species, "--tag", tag,
           "--basin", str(basin), "--seed-index", str(seed_index),
           "--seed0", str(seed0),
           "--prod-ps", str(prod_ps), "--equil-ps", str(equil_ps)]
    if platform and route == "openmm":
        cmd += ["--platform", str(platform)]
    if wall_budget_s:
        cmd += ["--wall-budget-s", str(wall_budget_s)]
    if basins_file:
        cmd += ["--basins", str(basins_file)]

    proc = _sp.run(cmd, cwd=str(repo_root), env=e, text=True,
                   stdout=_sp.PIPE, stderr=_sp.PIPE)
    summary = dict(species=species, basin=basin, seed_index=seed_index,
                   route=route, platform=platform,
                   cuda_visible_devices=visible,
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
    ap.add_argument("--route", default=None, choices=sorted(ROUTES),
                    help="openmm (production since 2026-09-07) or ase (the independent "
                         "implementation pair). Default: the resource config's "
                         "QHA_ROUTE, else openmm on a GPU site and ase elsewhere.")
    ap.add_argument("--platform", default=None,
                    help="OpenMM platform for --route openmm. Default: the resource "
                         "config's OPENMM_PLATFORM, else CPU.")
    ap.add_argument("--tag", default="prod")
    ap.add_argument("--basins", default="auto",
                    help="how many basins per species to run. 'auto' (the default) takes "
                         "the count from branch A's own record, so the number and the "
                         "geometries cannot disagree. An integer caps it.")
    ap.add_argument("--seeds", type=int, default=3,
                    help="independent trajectories per basin. 3 is the minimum that "
                         "gives a blank control, which is acceptance criterion 5")
    ap.add_argument("--seed0", type=int, default=20260903)
    # The published protocol, from configs/branchB_protocol.yaml -- the same file both
    # trajectory drivers read. Hard-coding 200/50 here would have quietly overridden it
    # for every Parsl run, which is the whole class of bug this file exists to avoid.
    ap.add_argument("--prod-ps", type=float, default=_PROTOCOL["production_ps"])
    ap.add_argument("--equil-ps", type=float, default=_PROTOCOL["equilibration_ps"])
    ap.add_argument("--max-workers", type=int, default=None)
    ap.add_argument("--wall-budget-s", type=float, default=None,
                    help="per-task budget; default comes from the resource config")
    ap.add_argument("--account", default=None, help="scheduler account (cluster only)")
    ap.add_argument("--partition", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and submit nothing")
    ap.add_argument("--debug", action="store_true",
                    help="submit for real, on the site's short partition, at its debug "
                         "walltime, capped at ONE allocation. This -- not --dry-run -- "
                         "is the gate before production.")
    args = ap.parse_args()

    cfg = config.load()
    species, source = species_list(args, cfg)

    import resource_configs
    res = resource_configs.load(args.resource)
    described = res.describe()

    # The per-task budget must follow the QUEUE, not the site. QHA_WALL_BUDGET_S is
    # 90% of the production walltime; under --debug the walltime is 30 minutes, and
    # handing a debug task a seven-day budget means it never stops itself and gets
    # killed by the queue instead -- the one thing the budget exists to prevent.
    budget = args.wall_budget_s
    if budget is None:
        if args.debug and described.get("debug_walltime"):
            budget = int(0.90 * _walltime_seconds(described["debug_walltime"]))
        else:
            budget = getattr(res, "QHA_WALL_BUDGET_S", 0)

    # Route and platform come from the resource config unless overridden, so that
    # "which machine" and "which implementation" are decided in one place. A GPU site
    # declares OPENMM_PLATFORM = "CUDA"; a CPU site declares nothing and gets CPU.
    route = args.route or getattr(res, "QHA_ROUTE", None)
    if route is None:
        route = "openmm" if getattr(res, "OPENMM_PLATFORM", None) else "ase"
    platform = args.platform or getattr(res, "OPENMM_PLATFORM", None) or "CPU"
    if route == "ase" and args.platform:
        raise SystemExit("--platform applies to --route openmm only; the ASE route has "
                         "no platform concept.")

    # WHICH GEOMETRIES. Until 2026-09-08 this driver passed no basin file at all, so
    # every task started from the QM9 reference geometry and `--basins 5` produced five
    # copies of one basin under five different indices. The count and the file now come
    # from the same place: branch A's product.
    basins_for, missing = {}, []
    for s in species:
        j, x = basin_store.paths_for(s, tag=args.tag)[:2]
        if j.exists() and x.exists():
            rec = basin_store.read(s, tag=args.tag) or {}
            n = len(rec.get("basins", [])) or 1
            basins_for[s] = (str(x), n)
        else:
            basins_for[s] = (None, 1)
            missing.append(s)

    def _n_basins(s):
        _f, n = basins_for[s]
        return n if args.basins == "auto" else min(n, int(args.basins))

    tasks = [(s, b, k) for s in species
             for b in range(_n_basins(s)) for k in range(args.seeds)]

    plan = dict(
        generated_by="scripts/production/s0_E_branchB_parsl.py",
        branch="E (execution layer for branch B)",
        resource=described,
        route=route, route_script=ROUTES[route],
        openmm_platform=platform if route == "openmm" else None,
        # WHICH QUEUE THIS RUN ACTUALLY USED. Without it a --debug run and a production
        # run record the same `resource` block and become indistinguishable afterwards.
        submission=dict(
            debug=bool(args.debug),
            partition=(args.partition
                       or (described.get("debug_partition") if args.debug
                           else described.get("partition"))),
            walltime=(described.get("debug_walltime") if args.debug
                      else described.get("walltime")),
            max_blocks=(1 if args.debug else described.get("max_blocks"))),
        n_species=len(species), species=species, species_source=source,
        n_tasks=len(tasks),
        basins_per_species={s: basins_for[s][1] for s in species},
        basin_files={s: basins_for[s][0] for s in species},
        species_without_branch_a=missing,
        seeds_per_basin=args.seeds,
        tag=args.tag, prod_ps=args.prod_ps, equil_ps=args.equil_ps,
        wall_budget_s=budget,
        science_settings_source=("scripts/production/{} and configs/openqha.yaml -- "
                                 "NOT this script".format(ROUTES[route])),
        prohibitions="no bias, no constraints, real hydrogen mass, fixcm=False",
        cost_note=("96.1 s/ps (ASE) and 100 s/ps (OpenMM) measured on this project's "
                   "workstation for a 10-atom molecule at 1 CPU thread. NEITHER is a "
                   "GPU number, and neither may be divided by a worker count to produce "
                   "a cluster estimate (D0-P1-12, defects 34 and 56)."),
        gpu_cost_status=("UNMEASURED on TianheXY-A. The only GPU figure this repository "
                         "has is D0-C-5, 3.5x SLOWER than CPU on a T400. Measure with "
                         "--debug before sizing a campaign."),
        estimated_single_task_seconds=round(96.1 * args.prod_ps, 1),
    )

    print("=" * 92)
    print("Branch E -- branch B over Parsl")
    print("=" * 92)
    # The role's OWN layout, not the site's default one. Reading `workers_per_node`
    # printed the CREST layout (16 x 4) for a branch B run that uses 64 x 1.
    _lay = (described.get("layouts") or {}).get("qha") or {}
    print("resource     {}  ({} workers x {} cores)".format(
        args.resource,
        _lay.get("workers_per_node") or described.get("workers_per_node")
        or described.get("qha_workers_per_node"),
        _lay.get("cores_per_worker") or described.get("cpus_per_worker")
        or described.get("qha_threads_per_job", 1)))
    print("route        {}  ({}{})".format(
        route, ROUTES[route],
        ", platform " + platform if route == "openmm" else ""))
    print("queue        {} partition={} walltime={} max_blocks={}".format(
        "DEBUG (smoke test)" if args.debug else "production",
        plan["submission"]["partition"], plan["submission"]["walltime"],
        plan["submission"]["max_blocks"]))
    print("tasks        {}  ({} species, basins from branch A, {} seeds)".format(
        len(tasks), len(species), args.seeds))
    for s in species:
        f, n = basins_for[s]
        print("             {:<20} {} basin(s)  {}".format(
            s, n, f if f else "NO BRANCH A PRODUCT -- see the warning below"))
    if missing:
        print()
        print("WARNING: {} species have no branch A basin list under tag {!r}:".format(
            len(missing), args.tag))
        print("         {}".format(", ".join(missing[:8])))
        print("         Their trajectories will start from the QM9 REFERENCE GEOMETRY,")
        print("         which is one geometry and not a basin. The per-trajectory")
        print("         meta.json records that as `geometry_source`, and any F_conf")
        print("         built on them is a sum over ONE basin however many were asked")
        print("         for. Run scripts/production/s0_A_pipeline.py first.")
    print("length       {} ps equilibration + {} ps production".format(
        args.equil_ps, args.prod_ps))
    print("wall budget  {}".format(
        "{} s per task".format(int(budget)) if budget else
        "none -- a task runs to completion (this machine has no queue to be killed by)"))
    print("single task  about {:.0f} s at the measured 96.1 s/ps (10 atoms, CPU, "
          "uncontended). NOT a GPU estimate.".format(96.1 * args.prod_ps))
    print("tag          {}".format(args.tag))
    print()

    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0

    import labels as _labels
    qha_label = _labels.label("qha")

    accepted = _accepted_kwargs(res.config)
    kw = {}
    if "role" in accepted:
        kw["role"] = "qha"
    if args.max_workers:
        kw["max_workers" if "max_workers" in accepted else "qha_max_workers"] = \
            args.max_workers
    if args.account:
        kw["account"] = args.account
    if args.partition:
        kw["partition"] = args.partition
    if args.debug:
        # Refused rather than ignored. Silently running production settings under a flag
        # that says "debug" is how a 30-minute intention becomes a 7-day allocation.
        if "debug" not in accepted:
            raise SystemExit(
                "--debug is not supported by resource config {!r}. Sites with a short "
                "partition accept it (tianhe_a: temp/00:30:00, tianhe_cpu: "
                "debug/00:30:00).".format(args.resource))
        kw["debug"] = True
    parsl_config = res.config(**kw)

    import parsl
    from parsl import python_app
    parsl.load(parsl_config)

    # The config is checked against the label BEFORE anything is submitted. A Parsl app
    # that names an executor the config does not provide fails at neither build nor
    # render time -- it simply never schedules. Until 2026-09-07 this file asked for the
    # retired label `openqha_qha` and would have done exactly that.
    report = _labels.check(parsl_config, expect=qha_label)
    if report["legacy"]:
        print("NOTE: this resource config uses retired executor labels {} -> {}. "
              "They still resolve; update the config.".format(
                  report["legacy"], report["legacy_map"]))
    print("executor     {}".format(qha_label))

    app = python_app(run_one_trajectory, executors=[qha_label])

    passthrough = {k: os.environ[k] for k in
                   ("S0_RUNS_ROOT", "S0_ENGINE", "S0_MACE_MODEL", "S0_CONFIG",
                    "S0_GMX_BIN", "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
                   if k in os.environ}

    started = time.time()
    futures = [app(s, b, k, args.seed0, str(ROOT), args.tag, args.prod_ps, args.equil_ps,
                   budget, route=route, platform=platform,
                   basins_file=basins_for[s][0], env=passthrough)
               for s, b, k in tasks]
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

    print("{:<20} {:>6} {:>6} {:>8} {:>9} {:>10} {:>5}  {}".format(
        "species", "basin", "seed", "frames", "T mean K", "seconds", "card",
        "complete"))
    for r in results:
        if r.get("error"):
            print("{:<20} {}".format(r.get("species", "?"), str(r["error"])[:60]))
            continue
        print("{:<20} {:>6} {:>6} {:>8} {:>9} {:>10.1f} {:>5}  {}".format(
            r["species"], r["basin"], r["seed_index"], r.get("n_frames", "-"),
            "-" if r.get("temperature_mean_K") is None
            else "{:.2f}".format(r["temperature_mean_K"]),
            r["seconds"], r.get("cuda_visible_devices") or "-", r.get("complete")))

    # If every task reports the same card, the per-worker pinning is broken and the run
    # was 8x slower for a reason that would otherwise show up only as "the GPU is slow".
    cards = sorted({r.get("cuda_visible_devices") for r in results
                    if r.get("cuda_visible_devices")})
    if len(cards) == 1 and len(results) > 1:
        print()
        print("WARNING: every task ran on card {} -- if this allocation had more than "
              "one, the workers were not pinned per card and the run used one of "
              "them. Check `available_accelerators` in the resource config.".format(
                  cards[0]))

    # Branch E acceptance criterion 5: three numbers, kept apart. The third one is
    # deliberately NOT computed -- a slot extrapolation is a claim about contention that
    # nothing here has measured.
    single = [r["seconds"] for r in results if not r.get("error")]
    per_ps = [r["seconds_per_ps"] for r in results if r.get("seconds_per_ps")]
    print()
    print("wall_seconds_for_the_whole_batch  {:.1f}".format(wall))
    print("single_task_seconds_median        {:.1f}".format(
        sorted(single)[len(single) // 2] if single else float("nan")))
    if per_ps:
        print("seconds_per_ps_median             {:.1f}   <- the number to plan the "
              "campaign with, measured on THIS machine".format(
                  sorted(per_ps)[len(per_ps) // 2]))
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
