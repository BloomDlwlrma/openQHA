"""Workflow hessian_learning, step 03: reference E-F-H labels for every frame of the selected
molecules (CONTEXT.md "Frame set", "Batch"; ticket 03 of the Hessian-learning set).

PRODUCTION. The Batch driver over `openqha.data.frame_labels`: ONE FRAME PER TASK (an
ORCA single point + EnGrad + analytic Hessian at the frame's fixed geometry, `%pal nprocs
4`), 16 frames per 64-core node on tianhe (role `labels`, user ruling 2026-09-18: 1 node
for the 7-molecule smoke set, 12 nodes for the 200-molecule draw), then `assemble` per
molecule writes `<generator>.<level>.extxyz` and the Record `frames/labels.<level>.{out,
toml}`. Finished frames (terminal line + `.hess`) are skipped, so a resubmission continues
where the last one stopped. The Slurm log is the Batch's report.

The one-shot retry (ticket 02) RIDES EVERY ROUND (ticket 04, ruling 2026-09-26): the
task list carries every FAILED frame whose failure has no archive
(`frame_labels.retryable`) alongside the never-run frames, marks it `retry` (the worker's
5th column), and the failed `.out` is archived as `<stem>.failed.out` before ORCA
overwrites it. `--retry-only` turns the round into the FAILURES-ONLY SWEEP: the list
holds ONLY those failed frames and nothing else, so the recovery is a few hours instead
of a full pass. After that retry the frame is final whatever the outcome -- a failed
retry keeps its archive and is never selected again, a cut retry leaves no `.out` and the
ordinary policy reruns it whole. The round's log carries the summary block only -- no
frame list, no per-molecule walk (ticket 05); in the submitted-job route each task's own
retry slice prints in its slurm log (`hpc/slurm/hl_labels.slurm`).

TWO WAYS TO RUN IT (user ruling 2026-09-19):
  * a submitted job, plain bash + xargs, NO parsl: `hpc/slurm/hl_labels.slurm` calls this
    driver with `--list FILE` (the pending frames, one per line), runs
    `python -m openqha.data.frame_labels` per line through xargs, then `--assemble`;
  * the ALF mode: this driver on the login node, parsl submitting blocks with
    `--resource tianhe_cpu` (SlurmProvider, sbatch), for a campaign nobody wants to babysit.

    # the round's summary only
    python workflows/hessian_learning/03_labels.py --tag rings --all --dry-run
    # this machine, in-process, no Parsl (the same list, the same code path per frame)
    python workflows/hessian_learning/03_labels.py --tag rings --species dsgdb9nsd_000048 --local
    # tianhe: one frame on the debug queue, then the smoke set on one node, then the draw
    python workflows/hessian_learning/03_labels.py --tag rings --all --resource tianhe_cpu --debug --limit-frames 1
    python workflows/hessian_learning/03_labels.py --tag smoke --all --resource tianhe_cpu --max-blocks 1
    python workflows/hessian_learning/03_labels.py --tag draw200 --resource tianhe_cpu

`--name` labels exactly the molecules 01_select chose (select.dat); `--limit N` without
it applies the same rule as 01 (pinned always, then the first N or, with `--stratify`,
N spread evenly over the sorted index list -- QM9 is ordered by heavy-atom count).
"""
import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

signal.signal(signal.SIGPIPE, signal.SIG_DFL)      # `--dry-run | head` must not end in a traceback


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))


def _hpc_root(repo):
    for cand in sorted(repo.rglob("resource_configs/local.py")):
        if "_superseded" in cand.parts or "_backup" in cand.parts:
            continue
        return cand.parent.parent
    raise RuntimeError("no resource_configs/local.py found under {}".format(repo))


sys.path.insert(0, str(_hpc_root(ROOT)))

from openqha import config                                   # noqa: E402
from openqha.data import dataset, frame_labels, frames       # noqa: E402
from openqha.store import basins as basin_reader, layout     # noqa: E402

COMPLETION = "<molecule>/frames/orca.<level>.<generator>_bBB_kK.out carries ****ORCA TERMINATED NORMALLY**** and the .hess (Hessian job) or .engrad (gradient job) exists; a .out without that line is a failed frame -- every round re-attempts it once (the old .out archived first), and an archived failure is final"


# ======================================================================================
# The task. It runs in a worker process, so it must be self-contained.
# ======================================================================================
def label_frame_task(molecule_dir, level, generator, basin, k, nprocs, maxcore, repo_root, env=None, timeout_s=None,
                     retry=False):
    """One frame's ORCA job in a worker: `frame_labels.label_one` with the node-local
    scratch (`S0_SCRATCH`, set by hpc/env/*.sh) when there is one and `timeout_s` (the
    driver's `--timeout`, else the environment's `TIMEOUT_S`). `retry` is the round's
    per-frame retry intent (the task list's 5th column, `retry`/`-`), mapped to the
    frame's existing one retry.
    Returns the Batch row; an
    exception is caught and returned as `error` so one frame never loses the batch. A
    `SystemExit` (the frame was CUT: SIGTERM at a block's time limit) is not an Exception
    and propagates -- parsl then reruns the task on another block (ticket 24)."""
    import os as _os
    import sys as _sys
    import time as _time
    import traceback as _tb
    started = _time.time()
    _os.environ.update(env or {})
    _sys.path.insert(0, str(repo_root))
    from openqha.data import frame_labels as _fl
    row = dict(species=_os.path.basename(str(molecule_dir).rstrip("/")), basin=int(basin), seed=None, generator=generator, k=int(k),
               rc=None, seconds=None, status=None, record=None)
    try:
        scratch = _os.environ.get("S0_SCRATCH") or None
        if timeout_s is None and _os.environ.get("TIMEOUT_S"):
            timeout_s = float(_os.environ["TIMEOUT_S"])
        lab = _fl.label_one(molecule_dir, level, generator, basin, k, nprocs=int(nprocs), maxcore=int(maxcore),
                            scratch=scratch, timeout_s=timeout_s, retry=retry)
        row.update(rc=0, seconds=lab["seconds"] if lab["seconds"] is not None else lab["wall_seconds"],
                   status=lab["status"], record=lab["out"], route=lab["hessian_route"],
                   memory_mb=lab["memory_mb"], floor_cm=lab["noise_floor_cm"], orca_version=lab["orca_version"],
                   frame=_fl.frame_tag(generator, basin, k))
    except Exception as exc:
        row.update(rc=1, seconds=_time.time() - started, status="FAILED", error=_tb.format_exc(),
                   error_line="{}: {}".format(type(exc).__name__, str(exc).strip().splitlines()[-1][:160] if str(exc).strip() else ""),
                   frame=_fl.frame_tag(generator, basin, k))
    return row


# ======================================================================================
def molecules_under(tag, cfg=None):
    """Every molecule directory under `tag` with a Frame set (login-node cheap)."""
    base = Path(config.runs_root(cfg)) / str(tag)
    return sorted(p.parent.parent for p in base.glob("*/frames/{}.toml".format(frames.STEP)))


def choose(mols, limit, stratify):
    """`dataset.apply_limit` on molecule directories: the pinned molecules always, the
    rest first-N or spread -- the same draw 01_select makes, so `--limit` here labels the
    molecules 01 selected. With `--name` the selection itself is read instead."""
    rows = [dict(qm9_index=m.name, pinned=m.name in dataset.PINNED, molecule_dir=m) for m in mols]
    return [r["molecule_dir"] for r in dataset.apply_limit(rows, limit, stratify)]


def pending(mols, level, generators, retry_only=False):
    """(molecule, generator, basin, k, retry) for every frame this round should attempt:
    every kept frame with NO ORCA job on disk, PLUS (ticket 04, ruling 2026-09-26) every
    FAILED frame whose failure is not archived yet -- the one-shot retry that rides every
    round -- `retry` True exactly for the latter. With `retry_only` the list holds ONLY
    those failed frames: the never-run frames are queued by no other sweep. Per molecule
    (frames, finished, failed) -- the counts are the disk's, the same in every mode.

    A finished frame is skipped always; a failed frame is selected when `retryable` (the
    `.out` lacks the terminal line and no `<stem>.failed.out` exists -- one retry per
    frame, ever) and no other process holds it (`running_elsewhere`: its job alive and
    its heartbeat fresh); with `retry_only` the never-run frames are left alone. The
    per-frame manual lever stays `python -m openqha.data.frame_labels ... --retry`."""
    todo, counts = [], {}
    for mol in mols:
        n_all = n_done = n_failed = 0
        folder = layout.frames_dir(mol)
        for g, b, k in frame_labels.frame_list(mol, generators):
            n_all += 1
            stem = layout.orca_frame_stem(level, g, b, k)
            if frame_labels.finished(folder, stem, hessian=frame_labels.wants_hessian(g)):
                n_done += 1
                continue
            if frame_labels.failed(folder, stem):
                n_failed += 1
                if frame_labels.retryable(folder, stem) and not frame_labels.running_elsewhere(folder, stem):
                    todo.append((mol, g, b, k, True))
                continue
            if retry_only:
                continue                                   # the sweep queues no mass work
            if frame_labels.running_elsewhere(folder, stem):
                continue                                   # another process holds it
            todo.append((mol, g, b, k, False))
        counts[mol.name] = (n_all, n_done, n_failed)
    return todo, counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--species", nargs="*", help="QM9 indices")
    g.add_argument("--all", action="store_true", help="every molecule under the tag with a Frame set")
    g.add_argument("--name", help="the Dataset name of 01_select: label exactly its molecules (select.dat under --tag); "
                                  "the default when neither --species nor --all is given: the tag")
    ap.add_argument("--level", default=frame_labels.DEFAULT_LEVEL, help="reference level (orca.LEVELS)")
    ap.add_argument("--generators", nargs="*", default=None, help="subset of {}".format(", ".join(frames.GENERATORS)))
    ap.add_argument("--retry-only", action="store_true",
                    help="list ONLY the FAILED frames whose failure has no archive yet -- the failures-only sweep (ticket 04); "
                         "a plain round already carries them once: the previous failed .out is archived as <stem>.failed.out "
                         "before ORCA runs, and a frame whose retry is spent (archive present) is left alone")
    ap.add_argument("--limit", type=int, default=None, help="at most N molecules")
    ap.add_argument("--stratify", action="store_true", help="spread --limit evenly over the sorted index list")
    ap.add_argument("--limit-frames", type=int, default=None, help="at most N frames in this Batch (the debug job)")
    ap.add_argument("--nprocs", type=int, default=frame_labels.NPROCS, help="ORCA ranks per frame")
    ap.add_argument("--maxcore", type=int, default=None, help="%%maxcore MB per rank (default: the resource's)")
    ap.add_argument("--resource", default="local", help="hpc/resource_configs/<name>.py")
    ap.add_argument("--local", action="store_true", help="run the list here, in this process, without Parsl")
    ap.add_argument("--max-blocks", type=int, default=None, help="allocations at once (1 for the smoke set)")
    ap.add_argument("--max-workers", type=int, default=None)
    ap.add_argument("--walltime", default=None, help="override the resource's walltime (deimos allows 7 days)")
    ap.add_argument("--partition", default=None)
    ap.add_argument("--account", default=None)
    ap.add_argument("--debug", action="store_true", help="the site's short partition, one allocation")
    ap.add_argument("--dry-run", action="store_true", help="print the round's summary and submit nothing")
    ap.add_argument("--list", metavar="FILE", default=None,
                    help="write the pending frames as a task list (molecule_dir generator basin k retry, one per line; "
                         "retry is `retry` or `-`) for the xargs worker of hpc/slurm/hl_labels.slurm, and stop")
    ap.add_argument("--assemble", action="store_true",
                    help="run no ORCA: assemble every chosen molecule's files and Record from the finished jobs on disk")
    ap.add_argument("--timeout", type=float, default=None,
                    help="seconds per ORCA job before it is killed and the frame marked failed (default: $TIMEOUT_S, "
                         "28800 in hl_labels.slurm; round 11 Q2)")
    args = ap.parse_args(argv)

    cfg = config.load()
    if not (args.species or args.all or args.name):
        args.name = args.tag
    if args.species:
        mols = [basin_reader.molecule_for(q, args.tag) for q in args.species]
    elif args.name:
        mols = [Path(r["molecule_dir"]) for r in dataset.read_selection(config.runs_root(cfg), args.tag, args.name)]
    else:
        mols = molecules_under(args.tag, cfg)
    mols = [m for m in mols if (layout.frames_dir(m) / (frames.STEP + ".toml")).is_file()]
    if not args.name:
        mols = choose(mols, args.limit, args.stratify)
    if not mols:
        raise SystemExit("no molecule with a Frame set under tag {!r} (run 02_frames first)".format(args.tag))
    todo, counts = pending(mols, args.level, args.generators, retry_only=args.retry_only)
    if args.limit_frames:
        todo = todo[:args.limit_frames]
    n_retry = sum(1 for e in todo if e[4])

    import resource_configs
    res = resource_configs.load(args.resource)
    described = res.describe()
    maxcore = args.maxcore or described.get("maxcore_mb") or 3000
    keywords, _blocks, route = frame_labels.keyword_line(args.level)

    print("=" * 92)
    print("Hessian-learning 03 -- reference labels per frame over Parsl")
    print("=" * 92)
    print("level        {}   ! {}   ({} Hessian)".format(args.level, keywords, route))
    print("resource     {}   {} ranks per frame, %maxcore {} MB, {} frames per node, {} block(s)".format(
        args.resource, args.nprocs, maxcore,
        described.get("layouts", {}).get("labels", {}).get("workers_per_node") or described.get("max_workers"),
        args.max_blocks or described.get("layouts", {}).get("labels", {}).get("max_blocks", 1)))
    print("molecules    {} ({})".format(len(mols), "given" if args.species else
                                         "selection {!r}".format(args.name) if args.name else
                                         "every Frame set under tag {!r}".format(args.tag)))
    print("frames       {} to label, {} of them the one retry of a failed frame without an archive, {} finished already, {} failed frames on disk   ({})".format(
        len(todo), n_retry, sum(d for _a, d, _f in counts.values()), sum(f for _a, _d, f in counts.values()), COMPLETION))
    print("retry rule   every round carries the failed frames without an archive once: the failed .out is archived as "
          "<stem>.failed.out before ORCA starts; a retry that fails stays failed and is never selected again")
    if args.retry_only:
        print("retry-only   the list holds ONLY the failed frames without an archive; nothing else is queued")
    print("timeout      {} s per ORCA job".format(args.timeout if args.timeout is not None else
                                                 os.environ.get("TIMEOUT_S", "none")))
    print("resume       finished frames are skipped; a failed frame without an archive is re-attempted once; a frame "
          "whose retry is spent (archive present) is final; a cut frame (no .out) is rerun whole; the Record per molecule is rewritten by assemble")
    print()
    # ticket 05: the log carries the summary only -- no full frame list, no per-molecule
    # walk; each task's own retry slice is printed by hpc/slurm/hl_labels.slurm
    if args.dry_run:
        print(json.dumps(dict(level=args.level, keywords=keywords, resource=described, n_molecules=len(mols),
                              n_frames=len(todo), retry_only=args.retry_only, n_retries=n_retry,
                              nprocs=args.nprocs, maxcore=maxcore), indent=2, default=str))
        return 0
    if args.list:
        Path(args.list).parent.mkdir(parents=True, exist_ok=True)
        Path(args.list).write_text("".join("{} {} {} {} {}\n".format(mol, g, b, k, "retry" if retry else "-")
                                          for mol, g, b, k, retry in todo), encoding="utf-8")
        print("task list    {} ({} frames, {} retries; column 5: retry or -)".format(args.list, len(todo), n_retry))
        return 0
    if args.assemble:
        todo = []                                        # nothing runs; the assembly below reads the disk

    passthrough = {k: os.environ[k] for k in ("S0_RUNS_ROOT", "S0_CONFIG", "S0_ORCA_BIN", "S0_ORCA_PATH", "S0_ORCA_LIB")
                   if k in os.environ}
    started = time.time()
    results = []
    if args.assemble:
        pass
    elif todo and args.local:
        for mol, g, b, k, retry in todo:
            r = label_frame_task(str(mol), args.level, g, b, k, args.nprocs, maxcore, str(ROOT), env=passthrough,
                                 timeout_s=args.timeout, retry=retry)
            results.append(r)
            print("  {}  {}  {}  {}".format(r["species"], r["frame"], r["status"], r.get("error_line", "")), flush=True)
    elif todo:
        import labels as _labels
        import parsl
        from parsl import python_app
        label = _labels.label("labels")
        import inspect
        accepted = set(inspect.signature(res.config).parameters)
        kw = {}
        for key, val in (("role", "labels"), ("max_workers", args.max_workers), ("account", args.account),
                         ("partition", args.partition), ("max_blocks", args.max_blocks), ("walltime", args.walltime)):
            if val is not None and key in accepted:
                kw[key] = val
        if args.debug:
            if "debug" not in accepted:
                raise SystemExit("--debug is not supported by resource config {!r}".format(args.resource))
            kw["debug"] = True
        if args.resource != "local" and "role" not in accepted:
            raise SystemExit("resource {!r} has no roles; the labels executor is not there".format(args.resource))
        kw.setdefault("run_dir", os.environ.get("S0_PARSL_RUN_DIR") or os.path.join(
            str(config.runs_root(cfg)), "parsl", "labels_{}".format(os.environ.get("SLURM_JOB_ID") or "pid{}".format(os.getpid()))))
        pcfg = res.config(**kw)
        parsl.load(pcfg)
        _labels.check(pcfg, expect=label)
        print("executor     {}   mode {}".format(label, getattr(res, "LAST_MODE", None) or "local"), flush=True)
        app = python_app(label_frame_task, executors=[label])
        futures = [app(str(mol), args.level, g, b, k, args.nprocs, maxcore, str(ROOT), env=passthrough,
                       timeout_s=args.timeout, retry=retry)
                   for mol, g, b, k, retry in todo]
        for f in futures:
            try:
                results.append(f.result())
            except Exception as exc:
                import traceback
                results.append(dict(species="?", rc=1, status="FAILED", error=traceback.format_exc(),
                                    error_line="{}: {}".format(type(exc).__name__, str(exc)[:160])))
        parsl.dfk().cleanup()
    wall = time.time() - started

    # assemble per molecule: files and Records from whatever is finished on disk
    print()
    print("assemble:")
    computed = {}
    for r in results:
        if r.get("status") == "labelled":
            computed.setdefault(r["species"], []).append(r["frame"])
    summaries = []
    for mol in mols:
        try:
            out = frame_labels.assemble(mol, args.level, args.generators, args.nprocs, maxcore,
                                        computed=computed.get(mol.name, ()))
        except Exception as exc:
            print("  {}  FAILED to assemble: {}: {}".format(mol.name, type(exc).__name__, str(exc)[:160]))
            continue
        i = out["info"]
        summaries.append(i)
        print("  {}  labelled {}/{} (computed {}, reused {}, refused {}, failed {}, unlabelled {})  {:.0f} s/frame  {:.0f} MB  floor max {:.2f} cm^-1  {}".format(
            i["QM9_INDEX"], i["N_LABELLED"], i["N_FRAMES"], i["N_COMPUTED"], i["N_REUSED"], i["N_REFUSED"], i["N_FAILED"], i["N_UNLABELLED"],
            i["SECONDS_PER_FRAME"], i["MAX_MEMORY_MB"], i["NOISE_FLOOR_MAX_CM"], out["record"]), flush=True)

    from openqha.store import batch_table as _bt
    print()
    for r in results:
        r["record"] = r.get("record")
        r["STATUS"] = r.get("status")
    _bt.print_table(results, extra=(("frame", 18, "<"), ("route", 9, "<"), ("memory_mb", 9, ">"), ("floor_cm", 8, ">")))
    ok = [r for r in results if r.get("status") in ("labelled", "reused", "running", "failed")]
    single = sorted(r["seconds"] for r in ok if r.get("seconds") is not None)
    print()
    print("wall_seconds_for_the_whole_batch  {:.1f}".format(wall))
    print("single_task_seconds_median        {:.1f}".format(single[len(single) // 2] if single else float("nan")))
    for l in _bt.footer(wall, len(ok), len(results), what="frames labelled in this Batch"):
        print(l)
    n_fail, n_ref = sum(i["N_FAILED"] for i in summaries), sum(i["N_REFUSED"] for i in summaries)
    if args.assemble:
        # 0 = no frame without an ORCA job. A failed frame does not set the exit code: the
        # next round re-attempts it once unless its archive exists (tickets 02/04); a
        # refused frame is a human's decision (--retry after a rerun of 02).
        print("frames failed {}, refused {} over the chosen molecules (a failure without an archive is re-attempted once by the next round)".format(n_fail, n_ref))
        return 0 if all(i["N_UNLABELLED"] == 0 for i in summaries) else 1
    print("molecules fully labelled          {}/{}  ({} frames still unlabelled; {} failed, {} refused)".format(
        sum(1 for i in summaries if i["N_LABELLED"] == i["N_FRAMES"]), len(mols),
        sum(i["N_UNLABELLED"] for i in summaries), n_fail, n_ref))
    # the exit status is THIS Batch's: every frame it attempted labelled (a --limit-frames
    # debug job that labelled its one frame is a success; the rest of the molecule is
    # information, printed above)
    return 0 if len(ok) == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
