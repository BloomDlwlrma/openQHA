"""Workflow hessian_learning, step 03: reference E-F-H labels for every frame of the selected
molecules (CONTEXT.md "Frame set", "Batch"; ticket 03 of the Hessian-learning set).

PRODUCTION. The Batch driver over `openqha.data.frame_labels`: ONE FRAME PER TASK (an
ORCA single point + EnGrad + analytic Hessian at the frame's fixed geometry, `%pal nprocs
4`), fanned out over Parsl -- 16 frames per 64-core node on tianhe (role `labels`, user
ruling 2026-09-18: 1 node for the 7-molecule smoke set, 12 nodes for the 200-molecule
draw) -- then `assemble` per molecule writes `<generator>.<level>.extxyz` and the Record
`frames/labels.<level>.{out,toml}`. Finished frames (terminal line + `.hess`) are
skipped, so a resubmission continues where the last one stopped. The Slurm log is the
Batch's report (records redesign 2026-09-15): the frame list and the Batch table are
printed, nothing else is written outside the molecule trees.

    # the frame list only
    python workflows/hessian_learning/03_labels.py --tag rings --all --dry-run
    # this machine, in-process, no Parsl (the same list, the same code path per frame)
    python workflows/hessian_learning/03_labels.py --tag rings --species dsgdb9nsd_000048 --local
    # tianhe: one frame on the debug queue, then the smoke set on one node, then the draw
    python workflows/hessian_learning/03_labels.py --tag rings --all --resource tianhe_cpu --debug --limit-frames 1
    python workflows/hessian_learning/03_labels.py --tag smoke --all --resource tianhe_cpu --max-blocks 1
    python workflows/hessian_learning/03_labels.py --tag draw --name draw200 --resource tianhe_cpu

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

COMPLETION = "<molecule>/orca/<level>/frames/<generator>_bBB_kK/job.out carries ****ORCA TERMINATED NORMALLY**** and job.hess exists"


# ======================================================================================
# The task. It runs in a worker process, so it must be self-contained.
# ======================================================================================
def label_frame_task(molecule_dir, level, generator, basin, k, nprocs, maxcore, repo_root, env=None):
    """One frame's ORCA job in a worker: `frame_labels.label_one` with the node-local
    scratch (`S0_SCRATCH`, set by hpc/env/*.sh) when there is one. Returns the Batch
    row; an exception is caught and returned as `error` so one frame never loses the
    batch."""
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
        lab = _fl.label_one(molecule_dir, level, generator, basin, k, nprocs=int(nprocs), maxcore=int(maxcore),
                            scratch=scratch)
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
    return sorted(p.parent.parent for p in base.glob("*/*/*/frames/{}.toml".format(frames.STEP)))


def choose(mols, limit, stratify):
    """`dataset.apply_limit` on molecule directories: the pinned molecules always, the
    rest first-N or spread -- the same draw 01_select makes, so `--limit` here labels the
    molecules 01 selected. With `--name` the selection itself is read instead."""
    rows = [dict(qm9_index=m.name, pinned=m.name in dataset.PINNED, molecule_dir=m) for m in mols]
    return [r["molecule_dir"] for r in dataset.apply_limit(rows, limit, stratify)]


def pending(mols, level, generators):
    """(molecule, generator, basin, k) for every kept frame without a finished job, and
    the counts per molecule."""
    todo, counts = [], {}
    for mol in mols:
        n_all = n_done = 0
        for g, b, k in frame_labels.frame_list(mol, generators):
            n_all += 1
            if frame_labels.finished(layout.orca_frame_dir(mol, level, g, b, k)):
                n_done += 1
                continue
            todo.append((mol, g, b, k))
        counts[mol.name] = (n_all, n_done)
    return todo, counts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--species", nargs="*", help="QM9 indices")
    g.add_argument("--all", action="store_true", help="every molecule under the tag with a Frame set")
    g.add_argument("--name", help="the Dataset name of 01_select: label exactly its molecules (select.dat under --tag)")
    ap.add_argument("--level", default=frame_labels.DEFAULT_LEVEL, help="reference level (orca.LEVELS)")
    ap.add_argument("--generators", nargs="*", default=None, help="subset of {}".format(", ".join(frames.GENERATORS)))
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
    ap.add_argument("--dry-run", action="store_true", help="print the frame list and submit nothing")
    ap.add_argument("--timeout", type=float, default=None, help="seconds per ORCA job")
    args = ap.parse_args()

    cfg = config.load()
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
    todo, counts = pending(mols, args.level, args.generators)
    if args.limit_frames:
        todo = todo[:args.limit_frames]

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
    print("frames       {} to label, {} finished already   ({})".format(
        len(todo), sum(d for _a, d in counts.values()), COMPLETION))
    print("resume       finished frames are skipped; the Record per molecule is rewritten by assemble")
    if os.environ.get("SLURM_JOB_ID") and args.resource != "local" and not args.local:
        print("workers      IN THIS ALLOCATION (job {}, {} node(s)): nothing is submitted".format(
            os.environ["SLURM_JOB_ID"], os.environ.get("SLURM_JOB_NUM_NODES", "?")))
    print()
    print("frame list:")
    for mol, g, b, k in todo:
        print("  {}  {}".format(mol.name, frame_labels.frame_tag(g, b, k)))
    for name, (n_all, n_done) in counts.items():
        if n_all == n_done:
            print("  {}  all {} frames finished".format(name, n_all))
    print()
    if args.dry_run:
        print(json.dumps(dict(level=args.level, keywords=keywords, resource=described, n_molecules=len(mols),
                              n_frames=len(todo), nprocs=args.nprocs, maxcore=maxcore), indent=2, default=str))
        return 0

    passthrough = {k: os.environ[k] for k in ("S0_RUNS_ROOT", "S0_CONFIG", "S0_ORCA_BIN", "S0_ORCA_PATH", "S0_ORCA_LIB")
                   if k in os.environ}
    started = time.time()
    results = []
    if todo and args.local:
        for mol, g, b, k in todo:
            r = label_frame_task(str(mol), args.level, g, b, k, args.nprocs, maxcore, str(ROOT), env=passthrough)
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
        pcfg = res.config(**kw)
        parsl.load(pcfg)
        _labels.check(pcfg, expect=label)
        print("executor     {}   mode {}".format(label, getattr(res, "LAST_MODE", None) or "local"), flush=True)
        app = python_app(label_frame_task, executors=[label])
        futures = [app(str(mol), args.level, g, b, k, args.nprocs, maxcore, str(ROOT), env=passthrough)
                   for mol, g, b, k in todo]
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
        print("  {}  labelled {}/{} (computed {}, reused {}, refused {}, unlabelled {})  {:.0f} s/frame  {:.0f} MB  floor max {:.2f} cm^-1  {}".format(
            i["QM9_INDEX"], i["N_LABELLED"], i["N_FRAMES"], i["N_COMPUTED"], i["N_REUSED"], i["N_REFUSED"], i["N_UNLABELLED"],
            i["SECONDS_PER_FRAME"], i["MAX_MEMORY_MB"], i["NOISE_FLOOR_MAX_CM"], out["record"]), flush=True)

    from openqha.store import batch_table as _bt
    print()
    for r in results:
        r["record"] = r.get("record")
        r["STATUS"] = r.get("status")
    _bt.print_table(results, extra=(("frame", 18, "<"), ("route", 9, "<"), ("memory_mb", 9, ">"), ("floor_cm", 8, ">")))
    ok = [r for r in results if r.get("status") in ("labelled", "reused")]
    single = sorted(r["seconds"] for r in ok if r.get("seconds") is not None)
    print()
    print("wall_seconds_for_the_whole_batch  {:.1f}".format(wall))
    print("single_task_seconds_median        {:.1f}".format(single[len(single) // 2] if single else float("nan")))
    for l in _bt.footer(wall, len(ok), len(results), what="frames labelled in this Batch"):
        print(l)
    print("molecules fully labelled          {}/{}  ({} frames still unlabelled or refused over the chosen molecules)".format(
        sum(1 for i in summaries if i["N_LABELLED"] == i["N_FRAMES"]), len(mols),
        sum(i["N_UNLABELLED"] + i["N_REFUSED"] for i in summaries)))
    # the exit status is THIS Batch's: every frame it attempted labelled (a --limit-frames
    # debug job that labelled its one frame is a success; the rest of the molecule is
    # information, printed above)
    return 0 if len(ok) == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
