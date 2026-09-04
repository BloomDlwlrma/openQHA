"""Run branch A over many molecules through Parsl. Branch E, step 0.

PRODUCTION. This is the execution layer: it decides where branch A runs and how
many molecules run at once. **It computes nothing.** Every scientific decision is
made inside `scripts/production/s0_A_pipeline.py` and `configs/openqha.yaml`, and
this script must be replaceable by a for-loop without changing a single number.
That is branch E acceptance criterion 1, and the only way to keep it true is to
keep this file free of chemistry.

Step 0 before anything else
---------------------------
plan_E section 7: "step 0 comes before any action on a cluster". So the default
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

`--dry-run` renders the plan and touches nothing. Use it to see what would be
submitted, which is the habit branch E acceptance criterion 3 is built around.
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

sys.path.insert(0, str(_hpc_root(HERE)))

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
        thing that was missing in defect 57.

    The full record is written by the driver itself under analysis/branchA/<tag>/.
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

    # One resident MACE server for this molecule, on a socket named after it so two
    # workers can never collide.
    sock_dir = _Path(e.get("S0_RUNS_ROOT",
                          _os.path.expanduser("~/runs/openQHA"))) / "sockets"
    sock_dir.mkdir(parents=True, exist_ok=True)
    sock = str(sock_dir / "branchA_{}_{}.sock".format(qm9_index, _os.getpid()))
    if _os.path.exists(sock):
        _os.unlink(sock)

    server = _sp.Popen(
        [_sys.executable, "-m", "openqha.mace_server", "--socket", sock],
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

    out = repo_root / "analysis" / "branchA" / tag / qm9_index / "basins.json"
    summary = dict(qm9_index=qm9_index, seconds=_time.time() - started,
                   returncode=proc.returncode, command=cmd, socket=sock)
    if out.exists():
        rec = _json.loads(out.read_text(encoding="utf-8"))
        summary.update(
            n_basins=rec["census"]["n_basins"],
            n_conformers_crest_reports=rec["census"]["crest_vs_repo"][
                "n_conformers_reported_by_crest"],
            all_criteria_passed=rec["all_criteria_passed"],
            failed_criteria=[c["number"] for c in rec["criteria"]
                             if not c["passed"]],
            sigma=[b["symmetry"]["sigma"] for b in rec["basins"]],
            used_shake_fallback=rec["crest"].get("used_shake_fallback"),
            crest_seconds=rec["crest"].get("wall_seconds"),
            wall_is_valid_cost=rec["crest"].get("wall_is_valid_cost"),
            output=str(out))
    else:
        summary["error"] = "no basins.json was written"
        summary["stdout_tail"] = "\n".join(proc.stdout.splitlines()[-25:])
        summary["stderr_tail"] = "\n".join(proc.stderr.splitlines()[-25:])
    # The driver's own log, kept per molecule.
    log = repo_root / "analysis" / "branchA" / tag / qm9_index / "driver.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(proc.stdout + "\n----- stderr -----\n" + proc.stderr,
                   encoding="utf-8")
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
    ap.add_argument("--resource", default="local",
                    help="hpc/resource_configs/<name>.py (default: local -- step 0)")
    ap.add_argument("--tag", default="prod")
    ap.add_argument("--threads", type=int, default=None,
                    help="CREST threads per molecule; default from the resource config")
    ap.add_argument("--max-workers", type=int, default=None)
    ap.add_argument("--timeout-s", type=int, default=14400)
    ap.add_argument("--hessian-mode", default="analytic")
    ap.add_argument("--account", default=None, help="scheduler account (cluster only)")
    ap.add_argument("--partition", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and submit nothing")
    args = ap.parse_args()

    cfg = config.load()
    species, source = molecule_list(args, cfg)

    import resource_configs
    res = resource_configs.load(args.resource)
    threads = args.threads or getattr(res, "THREADS_PER_JOB", 4)

    plan = dict(
        generated_by="scripts/production/s0_E_branchA_parsl.py",
        branch="E (execution layer for branch A)",
        resource=res.describe(),
        n_species=len(species), species=species, species_source=source,
        tag=args.tag, threads_per_task=threads,
        hessian_mode=args.hessian_mode, timeout_s=args.timeout_s,
        science_settings_source="configs/openqha.yaml -- NOT this script",
        crest_settings={k: cfg["crest"].get(k) for k in
                        ("workhorse", "refine", "shake", "tstep_fs", "optlev",
                         "runtype")},
    )

    print("=" * 92)
    print("Branch E -- branch A over Parsl")
    print("=" * 92)
    print("resource     {}  ({} workers x {} threads)".format(
        args.resource, res.describe().get("max_workers")
        or res.describe().get("workers_per_node"), threads))
    print("molecules    {}  ({})".format(len(species), source))
    print("settings     workhorse={workhorse} refine={refine} shake={shake} "
          "tstep={tstep_fs} fs".format(**plan["crest_settings"]))
    print("tag          {}".format(args.tag))
    print()

    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0

    kw = {}
    if args.max_workers:
        kw["max_workers"] = args.max_workers
    if args.account:
        kw["account"] = args.account
    if args.partition:
        kw["partition"] = args.partition
    parsl_config = res.config(**kw)

    import parsl
    from parsl import python_app
    parsl.load(parsl_config)

    app = python_app(run_one_molecule, executors=["openqha_crest"])

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

    print("{:<20} {:>8} {:>8} {:>10} {:>8}  {}".format(
        "species", "basins", "CREST", "seconds", "fallback", "criteria"))
    for r in results:
        if r.get("error"):
            print("{:<20} {}".format(r.get("qm9_index", "?"), r["error"]))
            continue
        print("{:<20} {:>8} {:>8} {:>10.1f} {:>8}  {}".format(
            r["qm9_index"], r.get("n_basins", "-"),
            r.get("n_conformers_crest_reports", "-"), r["seconds"],
            str(r.get("used_shake_fallback")),
            "PASS" if r.get("all_criteria_passed") else
            "FAIL {}".format(r.get("failed_criteria"))))

    ok = [r for r in results if r.get("all_criteria_passed")]
    summary = dict(
        plan=plan, results=results,
        cost=dict(
            # THREE numbers, kept apart on purpose (D0-P1-12; defects 34 and 56).
            wall_seconds_for_the_whole_batch=wall,
            single_task_seconds=[r.get("seconds") for r in results
                                 if r.get("seconds")],
            slot_extrapolation_note=(
                "NOT computed here. Dividing the batch wall clock by the worker "
                "count produces a number that describes neither one molecule nor "
                "the batch. If a per-molecule cost under contention is wanted, "
                "measure it."),
        ),
        n_ok=len(ok), n_total=len(results),
        all_passed=bool(len(ok) == len(results)))

    out = S0_ROOT / "analysis" / "branchE" / args.tag / "batch.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print()
    print("batch wall {:.1f} s   {}/{} passed every criterion".format(
        wall, len(ok), len(results)))
    print("written    {}".format(out))
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
