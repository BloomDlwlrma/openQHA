"""Run branch A over many molecules through Parsl. Branch E, step 0.

PRODUCTION. This is the execution layer: it decides where branch A runs and how
many molecules run at once. **It computes nothing.** Every scientific decision is
made inside `scripts/production/s0_A_pipeline.py` and `configs/openqha.yaml`, and
this script must be replaceable by a for-loop without changing a single number.
That is branch E acceptance criterion 1, and the only way to keep it true is to
keep this file free of chemistry.

Step 0 before anything else
---------------------------
The rule is "step 0 comes before any action on a cluster", so the default
resource configuration is `local`, and moving to a cluster changes one argument:

    python scripts/production/s0_E_branchA_parsl.py --species dsgdb9nsd_000018
    python scripts/production/s0_E_branchA_parsl.py --edges --resource deimos --account XXX

One MACE server per worker, not one per machine
-----------------------------------------------
CREST's quality layer reaches MACE over a Unix socket held by a resident process.
That process holds one model behind one lock, so N workers sharing one socket are
not parallel -- they queue on the lock, and the fan-out is imaginary while looking
real in every log. Each worker therefore starts its own server on its own socket
and stops it when its molecule is done.

Two ways to look before you leap, and only one of them is the gate
------------------------------------------------------------------
`--dry-run` prints the plan and touches nothing. It is useful for reading what would be
submitted, and it is NOT the gate before production.

**The gate is `--debug`**: the same command, on the site's short
partition, at a 30-minute walltime, capped at one allocation.

    --debug on tianhe_cpu   partition debug, 00:30:00, 1 node   (production: deimos, 3 days)
    --debug on tianhe_a     partition temp,  00:30:00, 1 node   (production: ai, 7 days)

A real short job tests what a rendered plan cannot: that the modules load, that conda
activates on a compute node, that the weights hash matches there, that the scheduler
accepts the directives, and that Parsl can read its own status query. Run one edge under
`--debug`, read the product, then drop the flag.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

def _repo_root():
    """Directory holding the openQHA package, found by walking up from this file.

    Depth-independent on purpose: this file keeps working wherever under the
    repository it is moved to. The earlier move into `scripts/_superseded/` broke
    every `parents[1]` in the moved files silently, which is what this replaces.
    """
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


HERE = _repo_root()
sys.path.insert(0, str(HERE))


def _hpc_root(repo):
    """Directory holding the Parsl resource configurations, found by searching.

    NOT `repo / "hpc"`. That was the previous version, and it broke silently when
    the directory was moved to `examples/hpc/` -- and then moved back. Searching
    survives both, which is the point: the directory has now moved twice and a literal
    name would have had to be edited twice.

    This is the same failure as the `scripts/` reorganisation and the `source-code/`
    move: `_repo_root()` fixed the package import, and every path written as
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

sys.path.insert(0, str(_hpc_root(HERE)))


def _accepted_kwargs(fn):
    """Parameter names a resource config's `config()` will accept.

    Asked rather than assumed: the resource configs are separate modules with slightly
    different signatures, and passing a keyword one of them does not take raises
    TypeError from inside Parsl's construction, where it reads as a Parsl problem.
    """
    import inspect
    return set(inspect.signature(fn).parameters)

from openqha import S0_ROOT, config  # noqa: E402


# ======================================================================================
# The task. It runs in a worker process, so it must be self-contained.
# ======================================================================================
def run_one_molecule(qm9_index, repo_root, tag, threads, timeout_s, hessian_mode,
                     env=None):
    """One molecule, start to finish. Returns a small summary dict.

    Deliberately a THIN wrapper around the branch A driver run as a subprocess,
    rather than an import of it:

      * a CREST crash or an out-of-memory kill takes down one worker's subprocess,
        not the worker and its queue;
      * the driver's own stdout is captured verbatim per molecule, so a failure can
        be read without reconstructing it;
      * the command line that produced a result is recorded literally, which is the
        thing that was missing then.

    The Record is written by the driver itself into the molecule directory (_records/).
    What comes back here is only enough to build the summary table.
    """
    import json as _json
    import os as _os
    import subprocess as _sp
    import sys as _sys
    import time as _time
    from pathlib import Path as _Path
    # EVERY name this function uses is imported HERE, not taken from module scope.
    # Parsl serialises the function body to a worker process where this script's
    # globals do not exist. Measured 2026-09-03: the first version relied on the
    # module-level `Path` and both tasks came back with `name 'Path' is not defined`
    # -- after 5.8 s, with the batch reporting 0/2 and no chemistry attempted. The
    # per-task error capture worked (the batch was not lost), which is why the
    # failure was legible at all.

    repo_root = _Path(repo_root)
    started = _time.time()
    e = dict(_os.environ)
    e.update(env or {})

    # One resident MACE server for this molecule, in THIS JOB's node-local socket
    # directory -- `<TMPDIR or /tmp>/<owner>/<job id>/`, see `openqha.config.socket_dir`.
    #
    # It used to be `runs_root/sockets/`, which is the SHARED filesystem: two jobs on two
    # compute nodes then opened the same path, and whichever bound second unlinked the
    # first one's socket (measured on Tianhe 2026-09-09). A Unix socket is a rendezvous
    # between processes on ONE machine.
    from openqha import config as _config
    sock = str(_config.socket_path("branchA_{}".format(qm9_index), _os.getpid()))
    if _os.path.exists(sock):
        _os.unlink(sock)

    server = _sp.Popen(
        [_sys.executable, "-m", "openqha.potentials.mace_server", "--socket", sock],
        cwd=str(repo_root), env=e,
        stdout=open(sock + ".log", "w"), stderr=_sp.STDOUT)
    try:
        deadline = _time.time() + 180
        while not _os.path.exists(sock):
            if server.poll() is not None:
                raise RuntimeError(
                    "MACE server exited immediately (code {}); log at {}.log".format(
                        server.returncode, sock))
            if _time.time() > deadline:
                raise TimeoutError("MACE server socket {} never appeared".format(sock))
            _time.sleep(0.2)

        e["S0_MACE_SOCKET"] = sock
        cmd = [_sys.executable, "-u", "scripts/production/s0_A_pipeline.py",
               "--species", qm9_index, "--tag", tag,
               "--threads", str(threads), "--timeout-s", str(timeout_s),
               "--hessian-mode", hessian_mode]
        proc = _sp.run(cmd, cwd=str(repo_root), env=e, capture_output=True,
                       text=True, timeout=timeout_s + 600)
    finally:
        try:
            server.terminate()
            server.wait(timeout=30)
        except Exception:
            server.kill()
        if _os.path.exists(sock):
            try:
                _os.unlink(sock)
            except OSError:
                pass

    _sys.path.insert(0, str(repo_root))
    from openqha.store import basins as _basins, layout as _layout
    from openqha.store import branch_a_property as _bap, property as _prop
    # _records/branchA.toml in the molecule directory: branch A's Property file (records
    # redesign, 2026-09-15). STATUS is the completion marker.
    out = _basins.record_path(qm9_index, tag)
    from openqha.store import batch_table as _bt
    summary = dict(qm9_index=qm9_index, species=qm9_index, seconds=_time.time() - started,
                   returncode=proc.returncode, rc=proc.returncode, command=cmd, socket=sock,
                   status=_bt.status_after(proc.returncode, out), record=_bt.absolute(out))
    status = _prop.status_of(out)
    if status == _prop.NORMAL_TERMINATION:
        rec = _basins.read_record(qm9_index, tag)
        crit, cr, cen = rec.get("Criteria", {}), rec.get("CREST_Run", {}), rec.get("Census", {})
        summary.update(
            status=status,
            n_basins=cen.get("N_BASINS"),
            n_conformers_crest_reports=cr.get("N_CONFORMERS"),
            all_criteria_passed=bool(crit.get("ALL_PASSED")),
            failed_criteria=list(crit.get("FAILED") or []),
            sigma=[b.get("SIGMA") for b in _bap.basin_rows(rec)],
            used_shake_fallback=cr.get("USED_SHAKE_FALLBACK"),
            crest_seconds=cr.get("WALL"),
            wall_is_valid_cost=cr.get("WALL_IS_VALID_COST"),
            output=str(out))
    else:
        summary["status"] = status or _prop.FAILED
        summary["error"] = ("branchA.toml has STATUS {}".format(status) if status
                            else "no branchA.toml was written")
        summary["stdout_tail"] = "\n".join(proc.stdout.splitlines()[-25:])
        summary["stderr_tail"] = "\n".join(proc.stderr.splitlines()[-25:])
    # The Calculation's stdout, beside its Record: _records/driver.log next to branchA.out
    # Until then it went into the repository
    # checkout, analysis/branchA/<tag>/<qid>/driver.log, a Batch writing where no run
    # output belongs.
    log = out.parent / "driver.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(proc.stdout + "\n----- stderr -----\n" + proc.stderr,
                   encoding="utf-8")
    summary["driver_log"] = str(log)
    return summary


# ======================================================================================
def molecule_list(args, cfg):
    """Which molecules to run, and where the list came from."""
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
    ap.add_argument("--range", nargs=2, type=int, metavar=("START", "END"),
                    default=None,
                    help="QM9 index range, inclusive. Molecules already complete under "
                         "--tag are subtracted, and molecules with no geometry or that "
                         "fail the F0-F7 gates are dropped. Re-running the same command "
                         "does the remainder.")
    ap.add_argument("--no-resume", action="store_true",
                    help="with --range: do NOT subtract completed molecules")
    ap.add_argument("--resource", default="local",
                    help="hpc/resource_configs/<name>.py (default: local -- step 0)")
    ap.add_argument("--tag", default=None,
                    help="default from the resource config's TAG")
    ap.add_argument("--threads", type=int, default=None,
                    help="CREST threads per molecule; default from the resource config")
    ap.add_argument("--max-workers", type=int, default=None)
    ap.add_argument("--timeout-s", type=int, default=None,
                    help="default from the resource config's TIMEOUT_S")
    ap.add_argument("--hessian-mode", default=None,
                    help="default from the resource config's HESSIAN_MODE")
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
    worklist_record = None
    species, source = ([], "pending --range scan") if args.range else \
        molecule_list(args, cfg)

    import resource_configs
    res = resource_configs.load(args.resource)

    # The resource config carries the site's own defaults, so a production run needs no
    # flags beyond --edges --resource <site>. An explicit flag still wins: a one-off
    # override must not require editing a committed file.
    def _pick(flag_value, attr, fallback):
        if flag_value is not None:
            return flag_value, "command line"
        if hasattr(res, attr):
            return getattr(res, attr), "resource config {}.{}".format(args.resource, attr)
        return fallback, "built-in default"

    threads, threads_src = _pick(args.threads, "THREADS_PER_JOB", 4)
    tag, tag_src = _pick(args.tag, "TAG", "prod")
    timeout_s, timeout_src = _pick(args.timeout_s, "TIMEOUT_S", 14400)
    hessian_mode, hess_src = _pick(args.hessian_mode, "HESSIAN_MODE", "analytic")
    args.tag, args.timeout_s, args.hessian_mode = tag, timeout_s, hessian_mode

    # The worklist needs the resolved tag: "already complete" is a question about a
    # particular campaign's shard, not about the repository as a whole.
    if args.range:
        from openqha import worklist as _worklist
        start, end = args.range
        candidates = _worklist.index_range(start, end)
        if args.no_resume:
            species = candidates
            source = "indices {}..{} (resume disabled)".format(start, end)
            worklist_record = dict(
                completion_criterion="not consulted (--no-resume)",
                n_candidates=len(candidates), n_remaining=len(candidates))
        else:
            species, worklist_record = _worklist.remaining(
                candidates, cfg=cfg, tag=tag,
                progress=lambda m: print("  " + m, end="\r", flush=True))
            print(" " * 48, end="\r")
            source = "indices {}..{} minus what is already complete".format(start, end)

    described = res.describe()
    plan = dict(
        generated_by="scripts/production/s0_E_branchA_parsl.py",
        branch="E (execution layer for branch A)",
        resource=described,
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
        tag=tag, threads_per_task=threads,
        hessian_mode=hessian_mode, timeout_s=timeout_s,
        # Where each of those came from. Without this the product records the value and
        # not the decision, and "why 14400" has no answer six weeks later.
        settings_source=dict(threads=threads_src, tag=tag_src,
                             timeout_s=timeout_src, hessian_mode=hess_src),
        science_settings_source="configs/openqha.yaml -- NOT this script",
        worklist=worklist_record,
        crest_settings={k: cfg["crest"].get(k) for k in
                        ("workhorse", "refine", "shake", "tstep_fs", "optlev",
                         "runtype")},
    )

    print("=" * 92)
    print("Branch E -- branch A over Parsl")
    print("=" * 92)
    print("resource     {}  ({} workers x {} threads)".format(
        args.resource, described.get("max_workers")
        or described.get("workers_per_node"), threads))
    print("queue        {} partition={} walltime={} max_blocks={}".format(
        "DEBUG (smoke test)" if args.debug else "production",
        plan["submission"]["partition"], plan["submission"]["walltime"],
        plan["submission"]["max_blocks"]))
    print("molecules    {}  ({})".format(len(species), source))
    if worklist_record and worklist_record.get("n_already_done") is not None:
        print("resume       {} already complete, {} dropped (no geometry {}, gates {})"
              .format(worklist_record["n_already_done"],
                      worklist_record["n_dropped_no_geometry"]
                      + worklist_record["n_dropped_by_gate"],
                      worklist_record["n_dropped_no_geometry"],
                      worklist_record["n_dropped_by_gate"]))
        print("             done means: {}".format(
            worklist_record["completion_criterion"]))
    print("settings     workhorse={workhorse} refine={refine} shake={shake} "
          "tstep={tstep_fs} fs".format(**plan["crest_settings"]))
    print("tag          {}  ({})".format(tag, tag_src))
    print("timeout      {} s  ({})".format(timeout_s, timeout_src))
    print()

    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0

    kw = dict(role="crest")
    if args.max_workers:
        kw["max_workers"] = args.max_workers
    if args.account:
        kw["account"] = args.account
    if args.partition:
        kw["partition"] = args.partition
    if args.debug:
        # Refused rather than ignored. A resource config with no debug partition has no
        # short queue to fall back to, and silently running the production settings
        # under a flag that says "debug" is how a 30-minute intention becomes a 3-day
        # allocation.
        if "debug" not in _accepted_kwargs(res.config):
            raise SystemExit(
                "--debug is not supported by resource config {!r}. Sites that have a "
                "short partition accept it (tianhe_cpu: debug/00:30:00, tianhe_a: "
                "temp/00:30:00).".format(args.resource))
        kw["debug"] = True
    parsl_config = res.config(**kw)

    import parsl
    from parsl import python_app
    parsl.load(parsl_config)

    # The label comes from hpc/labels.py, and the config is checked against it BEFORE
    # anything is submitted. A Parsl app that names an executor the config does not
    # provide fails at neither build nor render time -- it simply never schedules.
    import labels as _labels
    crest_label = _labels.label("crest")
    report = _labels.check(parsl_config, expect=crest_label)
    if report["legacy"]:
        print("NOTE: this resource config uses retired executor labels {} -> {}. "
              "They still resolve; update the config.".format(
                  report["legacy"], report["legacy_map"]))
    print("executor     {}".format(crest_label))

    app = python_app(run_one_molecule, executors=[crest_label])

    passthrough = {k: os.environ[k] for k in
                   ("S0_CREST_BIN", "S0_RUNS_ROOT", "S0_ENGINE", "S0_MACE_MODEL",
                    "S0_CONFIG", "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
                   if k in os.environ}

    started = time.time()
    futures = [app(qid, str(S0_ROOT), args.tag, threads, args.timeout_s,
                   args.hessian_mode, env=passthrough) for qid in species]
    results = []
    for f in futures:
        try:
            results.append(f.result())
        except Exception as exc:                      # record it, do not lose the batch
            import traceback
            results.append(dict(error_type=type(exc).__name__, error=str(exc),
                                traceback=traceback.format_exc()))
    wall = time.time() - started
    parsl.dfk().cleanup()

    # The Batch table (records redesign, 2026-09-15): the same first seven columns as the
    # branch B and collect drivers, then this driver's own. It is all a Batch leaves.
    from openqha.store import batch_table as _bt
    for r in results:
        if r.get("error") and not r.get("error_line"):
            tail = [l for l in str(r["error"]).strip().splitlines() if l.strip()]
            r["error_line"] = tail[-1] if tail else "(no error text captured)"
        r["CREST"] = r.get("n_conformers_crest_reports")
        r["fallback"] = None if r.get("used_shake_fallback") is None else str(r["used_shake_fallback"])
        r["verdict"] = (("PASS" if r.get("all_criteria_passed") else "FAIL {}".format(r.get("failed_criteria")))
                        if r.get("status") == "NORMAL TERMINATION" else None)
    _bt.print_table(results, extra=(("n_basins", 6, ">"), ("CREST", 5, ">"), ("fallback", 8, ">"), ("verdict", 7, "<")))

    ok = [r for r in results if r.get("all_criteria_passed")]
    # THREE cost numbers, kept apart on purpose: the batch
    # wall (footer), each task's seconds (the table), and a slot extrapolation that is NOT
    # computed: dividing the batch wall by the worker count describes neither one
    # molecule nor the batch.

    # A Batch leaves no record of its own: the table above,
    # in the Slurm log, is its report. analysis/branchE/<tag>/batch.json is gone.
    print()
    for l in _bt.footer(wall, len(ok), len(results)):
        print(l)
    return 0 if len(ok) == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
