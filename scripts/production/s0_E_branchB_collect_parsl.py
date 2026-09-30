"""Turn finished trajectories into one result per molecule. Branch E, collection pass.

PRODUCTION. This is the execution layer for the second half of branch B: the
trajectories are already on disk and this fans out the quasi-harmonic ANALYSIS over
them. **It computes nothing.** Every number is produced by
`scripts/production/s0_B_qha_analyse.py`, and replacing this file with a for-loop must
not change one of them.

WHY THIS IS A SEPARATE DRIVER FROM THE TRAJECTORIES
---------------------------------------------------
The two halves of branch B run on different machines, for
different reasons:

    trajectories  TianheXY-A, GPU, one trajectory per card, 8 per allocation
                  s0_E_branchB_parsl.py --resource tianhe_a --route openmm
    collection    TianheXY-C, CPU, ONE MOLECULE PER CORE, 64 at a time, ONE NODE
                  this file, --resource tianhe_cpu

The collection pass reads frames and diagonalises a 3N x 3N covariance. That is small,
serial and float64 -- a card buys nothing, and 64 independent molecules on one node buy
64x. **One node, not twelve**: the pass is short and its cost is metadata traffic on
Lustre, so more nodes would buy contention rather than throughput. If collection ever
becomes the bottleneck, the lever is batching more molecules per task, not more nodes.

Executor `openqha_collect_executor`, from hpc/labels.py. A separate label from `qha` on
purpose: the two roles now live on different clusters, and a label that could match
either is a task that can be scheduled onto the wrong machine.

    # smoke test: one species, real job, 30 minutes on the debug queue
    python -u scripts/production/s0_E_branchB_collect_parsl.py \
        --species dsgdb9nsd_000018 --resource tianhe_cpu --debug

    # production
    python -u scripts/production/s0_E_branchB_collect_parsl.py \
        --edges --resource tianhe_cpu --tag prod
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

    NOT `repo / "hpc"` -- see the same function in s0_E_branchA_parsl.py. A name written
    into a string does not move with the thing it names, and this directory has moved
    twice.
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

#: What has to exist for a molecule to count as collected. Stated, printed and stored --
#: if the scanner and the worker disagree about what "done" means, molecules are skipped
#: for ever and never recovered. Same rule as openqha/worklist.py states for branch A.
COMPLETION = "<molecule>/_records/md_<route>/collect[_<setting>].toml carries STATUS NORMAL TERMINATION"


def result_path(repo_root, tag, species, basin_tag=None, setting="default", route="auto"):
    """The collect stem, `<molecule>/_records/md_<route>/collect[_<setting>]`; its `.toml` carries
    the completion marker (STATUS). With route `auto` the route is the one the trajectories
    are found under."""
    from openqha.store import basins
    from openqha.quasi_harmonic import chain_records, trajectory_reader
    mol = basins.molecule_for(species, basin_tag or tag)
    r = route if route in ("openmm", "ase") else (trajectory_reader.route_found(mol, setting) or "openmm")
    return chain_records.stem(mol, r, setting, chain_records.COLLECT)


def _accepted_kwargs(fn):
    """Parameter names a resource config's `config()` will accept."""
    import inspect
    return set(inspect.signature(fn).parameters)


# ======================================================================================
# The task. It runs in a worker process, so it must be self-contained.
# ======================================================================================
def collect_one(species, repo_root, tag, extra_args=(), env=None, basin_tag=None,
                setting="default", route="auto"):
    """One molecule's quasi-harmonic analysis, as a SUBPROCESS of the analysis driver.

    A subprocess rather than an import, for the three reasons branch A uses one: a crash
    takes down one task and not the worker's queue; the driver's stdout is captured
    verbatim per molecule; and the command line that produced a result is recorded
    literally.
    """
    import json as _json
    import os as _os
    import subprocess as _sp
    import sys as _sys
    import time as _time
    from pathlib import Path as _Path

    repo_root = _Path(repo_root)
    started = _time.time()
    e = dict(_os.environ)
    e.update(env or {})
    # One core per molecule. Set here as well as in hpc/env/common.sh: a worker that
    # inherited a different environment would oversubscribe 64 ways on 64 cores.
    e.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")

    cmd = [_sys.executable, "-u",
           str(repo_root / "scripts" / "production" / "s0_B_qha_analyse.py"),
           "--species", species, "--tag", tag,
           "--basin-tag", str(basin_tag or tag), "--setting", str(setting),
           "--route", str(route)] + list(extra_args)
    proc = _sp.run(cmd, cwd=str(repo_root), env=e, text=True,
                   stdout=_sp.PIPE, stderr=_sp.PIPE)

    # This driver's own log and marker beside collect's tables.
    _sys.path.insert(0, str(repo_root))
    from openqha.store import basins as _basins, layout as _layout
    from openqha.quasi_harmonic import trajectory_reader as _tr
    _mol = _basins.molecule_for(species, basin_tag or tag)
    _route = route if route in ("openmm", "ase") else (_tr.route_found(_mol, setting) or "openmm")
    from openqha.quasi_harmonic import chain_records as _cr
    out = _layout.md_records_dir(_mol, _route)
    out.mkdir(parents=True, exist_ok=True)
    _stem = _cr.stem(_mol, _route, setting, _cr.COLLECT)

    from openqha.store import batch_table as _bt
    summary = dict(species=species, command=" ".join(cmd),
                   returncode=proc.returncode, rc=proc.returncode,
                   seconds=_time.time() - started,
                   status=_bt.status_after(proc.returncode, _cr.collect_paths(_stem)["toml"]),
                   record=_bt.absolute(_cr.collect_paths(_stem)["toml"]))

    # Lift only enough out of the driver's own product to build the table. The parquet
    # tables it wrote stay the source of truth.
    n_traj, criteria = None, []
    for line in proc.stdout.splitlines():
        if line.startswith("trajectories "):
            try:
                n_traj = int(line.split()[1])
            except (IndexError, ValueError):
                pass
        if line.startswith("[PASS]") or line.startswith("[FAIL]"):
            criteria.append(dict(passed=line.startswith("[PASS]"),
                                 text=line[7:].strip()))

    # s0_B_qha_analyse.py exits 1 when a criterion FAILS and it printed the verdicts;
    # it exits with a traceback when it could not analyse at all. Until 2026-09-13 both
    # were `error` here: the verdicts were dropped, the marker never written, and the
    # table showed the first 60 characters of a traceback. A verdict is a result.
    if proc.returncode != 0 and not criteria:
        blob = (proc.stderr or proc.stdout or "")
        lines = [l for l in blob.strip().splitlines() if l.strip()]
        summary["error"] = blob[-2000:]
        summary["error_line"] = lines[-1] if lines else "(no output)"
        return summary
    summary.update(n_trajectories=n_traj,
                   n_criteria=len(criteria),
                   n_criteria_passed=sum(1 for c in criteria if c["passed"]),
                   all_criteria_passed=bool(criteria)
                   and all(c["passed"] for c in criteria),
                   criteria=criteria)

    # THE COMPLETION MARKER is STATUS in collect.toml, written by the analysis itself as
    # its last act (records redesign, 2026-09-15). A task killed before it leaves no marker,
    # so the molecule is redone -- the safe direction. This driver writes nothing of its
    # own beside the analysis's files.
    summary["marker"] = str(_cr.collect_paths(_stem)["toml"])
    summary["status"] = _cr.collect_status(_stem)
    summary["marker_present"] = _cr.collect_done(_stem)
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


def remaining(species, tag, cfg, no_resume=False, basin_tag=None, setting="default",
              route="auto"):
    """Which molecules still need collecting, and an account of how that was decided.

    Scan then execute, the pattern taken from this project's own
    `count_remain_fixed.py`: a campaign is a run, a queue limit, a restart. The reasons
    a molecule was skipped are returned, not merely its absence from the list -- a
    resume that cannot say why it dropped a molecule is a resume nobody can audit.
    """
    if no_resume:
        return list(species), dict(completion_criterion="not consulted (--no-resume)",
                                   n_candidates=len(species),
                                   n_remaining=len(species))
    from openqha.store import basins as _basins
    from openqha.quasi_harmonic import chain_records as _cr, trajectory_reader as _tr
    todo, done, no_traj = [], [], []
    for qid in species:
        if _cr.collect_done(result_path(ROOT, tag, qid, basin_tag, setting, route)):
            done.append(qid)
            continue
        # A molecule with no trajectories is not "remaining"; it is upstream work that
        # has not happened. Counting it as remaining makes the collection pass look
        # behind when it is actually waiting.
        if not _tr.trajectory_dirs(_basins.molecule_for(qid, basin_tag or tag, cfg), setting, route):
            no_traj.append(qid)
            continue
        todo.append(qid)
    return todo, dict(
        completion_criterion=COMPLETION,
        trajectories="<molecule>/md_<route>/basinNN/ under tag {!r}, setting {!r}".format(
            basin_tag or tag, setting),
        n_candidates=len(species), n_already_done=len(done),
        n_no_trajectories_yet=len(no_traj), no_trajectories=no_traj[:20],
        n_remaining=len(todo),
        note=("Molecules with no trajectory directory are NOT counted as remaining: "
              "they are waiting on the GPU half of branch B, not on this pass."))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species", nargs="*", default=None)
    ap.add_argument("--edges", action="store_true",
                    help="collect every species named by the configured edge set")
    ap.add_argument("--resource", default="local",
                    help="hpc/resource_configs/<name>.py (default: local -- step 0)")
    ap.add_argument("--tag", default="prod")
    ap.add_argument("--basin-tag", default=None,
                    help="the tag the molecule directory is under (default: --tag)")
    ap.add_argument("--setting", default="default",
                    help="which setting's trajectories to collect")
    ap.add_argument("--route", default="auto", choices=("auto", "openmm", "ase"),
                    help="which engine's trajectories to read: md_openmm/basinNN (traj.dcd) "
                         "or md_ase/basinNN (md.traj); auto takes openmm when present, else ase")

    ap.add_argument("--no-resume", action="store_true",
                    help="do NOT subtract molecules that already have a result")
    ap.add_argument("--no-gmx", action="store_true", default=True,
                    help="skip the GROMACS cross-check (default on a cluster: GROMACS "
                         "is an extension and no number depends on it)")
    ap.add_argument("--with-gmx", dest="no_gmx", action="store_false",
                    help="run the GROMACS cross-check as well")
    ap.add_argument("--limit-frames", type=int, default=None)
    ap.add_argument("--max-workers", type=int, default=None)
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
    candidates, source = species_list(args, cfg)
    species, record = remaining(candidates, args.tag, cfg, args.no_resume,
                                basin_tag=args.basin_tag, setting=args.setting, route=args.route)

    import resource_configs
    res = resource_configs.load(args.resource)
    described = res.describe()

    extra = []
    if args.no_gmx:
        extra.append("--no-gmx")
    if args.limit_frames:
        extra += ["--limit-frames", str(args.limit_frames)]

    plan = dict(
        generated_by="scripts/production/s0_E_branchB_collect_parsl.py",
        branch="E (execution layer for branch B collection)",
        resource=described,
        submission=dict(
            debug=bool(args.debug),
            partition=(args.partition
                       or (described.get("debug_partition") if args.debug
                           else described.get("partition"))),
            walltime=(described.get("debug_walltime") if args.debug
                      else described.get("walltime")),
            max_blocks=1),
        n_candidates=len(candidates), n_species=len(species),
        species=species, species_source=source,
        worklist=record, tag=args.tag,
        analyse_args=extra,
        science_settings_source=("scripts/production/s0_B_qha_analyse.py and "
                                 "configs/openqha.yaml -- NOT this script"),
    )

    print("=" * 92)
    print("Branch E -- branch B collection over Parsl")
    print("=" * 92)
    print("resource     {}  ({} workers x 1 core, {} node)".format(
        args.resource,
        described.get("layouts", {}).get("collect", {}).get("workers_per_node")
        or described.get("qha_workers_per_node"),
        described.get("layouts", {}).get("collect", {}).get("max_blocks", 1)))
    # Same distinction the trajectory driver draws (2026-09-13): inside a job the
    # workers are local and nothing is submitted; say so, or a reader cannot tell this
    # from a driver that is about to queue a block and wait for it.
    mode_now = described.get("mode_now") or (
        "in_allocation" if os.environ.get("SLURM_JOB_ID") and os.environ.get("S0_PARSL_NESTED") != "1"
        else "nested")
    if mode_now == "in_allocation":
        print("workers      IN THIS ALLOCATION (job {}): no block is submitted".format(
            os.environ.get("SLURM_JOB_ID")))
    else:
        print("workers      SUBMITTED as blocks: {} partition={} walltime={}".format(
            "DEBUG (smoke test)" if args.debug else "production",
            plan["submission"]["partition"], plan["submission"]["walltime"]))
    print("molecules    {} of {} candidates ({})".format(
        len(species), len(candidates), source))
    print("resume       {} already collected, {} still waiting on trajectories".format(
        record.get("n_already_done", "-"), record.get("n_no_trajectories_yet", "-")))
    print("             done means: {}".format(record["completion_criterion"]))
    print("tag          {}".format(args.tag))
    print()

    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0
    if not species:
        print("nothing to do.")
        return 0

    import labels as _labels
    collect_label = _labels.label("collect")

    accepted = _accepted_kwargs(res.config)
    kw = {}
    if "role" in accepted:
        kw["role"] = "collect"
    if args.max_workers:
        kw["max_workers"] = args.max_workers
    if args.account:
        kw["account"] = args.account
    if args.partition:
        kw["partition"] = args.partition
    if args.debug:
        if "debug" not in accepted:
            raise SystemExit(
                "--debug is not supported by resource config {!r}.".format(
                    args.resource))
        kw["debug"] = True
    parsl_config = res.config(**kw)

    import parsl
    from parsl import python_app
    parsl.load(parsl_config)

    report = _labels.check(parsl_config, expect=collect_label)
    if report["legacy"]:
        print("NOTE: this resource config uses retired executor labels {} -> {}."
              .format(report["legacy"], report["legacy_map"]))
    print("executor     {}".format(collect_label))

    app = python_app(collect_one, executors=[collect_label])

    passthrough = {k: os.environ[k] for k in
                   ("S0_RUNS_ROOT", "S0_ENGINE", "S0_MACE_MODEL", "S0_CONFIG",
                    "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
                   if k in os.environ}

    started = time.time()
    futures = [app(qid, str(ROOT), args.tag, tuple(extra), env=passthrough,
                   basin_tag=args.basin_tag, setting=args.setting, route=args.route)
               for qid in species]
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

    # The Batch table (records redesign, 2026-09-15): the same first seven columns as the
    # branch A and B drivers, then this driver's own. It is all a Batch leaves.
    from openqha.store import batch_table as _bt
    print()
    for r in results:
        if r.get("error") and not r.get("error_line"):
            tail = [l for l in str(r["error"]).strip().splitlines() if l.strip()]
            r["error_line"] = tail[-1] if tail else "(no error text captured)"
        r["trajs"] = r.get("n_trajectories")
        r["criteria_passed"] = ("{}/{}".format(r.get("n_criteria_passed", 0), r.get("n_criteria", 0))
                                if r.get("n_criteria") else None)
        r["verdict"] = ("PASS" if r.get("all_criteria_passed") else "FAIL") if not r.get("error") else None
    _bt.print_table(results, extra=(("trajs", 5, ">"), ("criteria_passed", 8, ">"), ("verdict", 7, "<")))

    ok = [r for r in results if r.get("all_criteria_passed")]
    single = [r["seconds"] for r in results if not r.get("error")]
    print()
    print("wall_seconds_for_the_whole_batch  {:.1f}".format(wall))
    print("single_task_seconds_median        {:.1f}".format(
        sorted(single)[len(single) // 2] if single else float("nan")))
    print("slot_extrapolation                NOT COMPUTED -- the per-task cost under "
          "contention has not been measured")

    # A Batch leaves no record of its own: the table above,
    # in the Slurm log, is its report. collect_batch.json is gone.
    print()
    for l in _bt.footer(wall, len(ok), len(results)):
        print(l)
    if len(ok) == len(results):
        return 0
    # OPENQHA_SMOKE=1 (set by hpc/tools/test30.sh, never by a production path): a
    # short test exists to prove every step runs and writes where it should, and a
    # 1+5 ps trajectory fails criteria 1, 5 and 9 BY CONSTRUCTION. With the override,
    # a molecule whose analysis RAN (verdicts recorded) does not stop the chain; one
    # that crashed still does. Said loudly, so the report downstream is never mistaken
    # for a number.
    ran = [r for r in results if not r.get("error")]
    if os.environ.get("OPENQHA_SMOKE") == "1" and len(ran) == len(results):
        print()
        print("**SMOKE RUN (OPENQHA_SMOKE=1): {} of {} molecules FAILED criteria, and the "
              "chain continues anyway.**".format(len(results) - len(ok), len(results)))
        print("  Every downstream number under tag {!r} is a test of the plumbing, not a "
              "result.".format(args.tag))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
