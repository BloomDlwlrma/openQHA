"""The molecule list of a campaign stage: which molecules of a draw still need branch A
(`--stage branchA`) or a Frame set (`--stage frames`), this array task's round-robin share,
capped at LIMIT. One line per molecule (qm9_index), read by the xargs loops of
hpc/slurm/hl_branchA.slurm and hl_frames.slurm. Login-node cheap: no MACE, no chemistry.

**The scan is latency-bound, not CPU-bound** (measured on Tianhe 2026-09-24: 6 458 drawn
molecules took **6 min 19 s** single-threaded, ~8 ms per metadata op on the shared pool
-- the silent minutes before an array task's first molecule line, and 12 tasks run it at
once). Each molecule costs a stat (the failure marker), the Property file read, and the
basin listing; the scan therefore runs in threads (`--workers`, `HL_LIST_WORKERS`) and
says when it starts and how long it took.

TOOLING.

    python hpc/slurm/hl_list.py --tag draw300 --stage branchA --out list.txt
    python hpc/slurm/hl_list.py --tag draw300 --stage frames --array-id 3 --array-n 12 --limit 16 --out list.txt
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


sys.path.insert(0, str(_repo_root()))
from openqha import config                                      # noqa: E402
from openqha.data import frames, structure_classes as sc         # noqa: E402
from openqha.store import basins, layout                         # noqa: E402

#: Threads over the per-molecule record reads. Latency, not throughput, is the cost, so
#: a handful of streams per process is enough; lower it (or `--workers 1`) if a cluster
#: objects to the concurrency.
DEFAULT_WORKERS = 8


def _state(stage, tag, root, qid, force):
    """One molecule's state for this stage: "failed", "todo" or "done". Reads only --
    no caches, no writes -- so it is safe to run from threads."""
    if stage == "branchA":
        if basins.failed(qid, tag, root=root):
            return "failed"
        return "todo" if not basins.done(qid, tag, root=root) else "done"
    mol = basins.molecule_for(qid, tag, root=root)
    if basins.done(qid, tag, root=root) and (
            force or not (layout.frames_dir(mol) / (frames.STEP + ".toml")).is_file()):
        return "todo"
    return "done"


def scan(qids, stage, tag, root, force=False, workers=1):
    """(todo, n_failed) over `qids`, in the order given.

    `workers` > 1 threads the per-molecule reads; the answer must not depend on it, and
    that is what `tests/unit/t_hl_list.py` holds.
    """
    if workers <= 1:
        states = [_state(stage, tag, root, q, force) for q in qids]
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            states = list(ex.map(lambda q: _state(stage, tag, root, q, force), qids))
    return ([q for q, s in zip(qids, states) if s == "todo"],
            int(sum(1 for s in states if s == "failed")))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--name", default=None, help="the draw (00_draw.py --name; default: the tag)")
    ap.add_argument("--stage", choices=("branchA", "frames"), required=True)
    ap.add_argument("--array-id", type=int, default=0)
    ap.add_argument("--array-n", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="at most N molecules for this task (the debug gate)")
    ap.add_argument("--force", action="store_true",
                    help="stage frames: list every molecule with branch A, whether or not it already has a Frame "
                         "set -- the REBUILD after a draw change. Ignored for stage branchA.")
    ap.add_argument("--workers", type=int, default=None,
                    help="threads over the per-molecule records (default {}; env HL_LIST_WORKERS; 1 disables)"
                         .format(DEFAULT_WORKERS))
    ap.add_argument("--quiet", action="store_true", help="the summary line only")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    args.name = args.name or args.tag
    workers = max(1, int(args.workers if args.workers is not None
                         else os.environ.get("HL_LIST_WORKERS", DEFAULT_WORKERS)))
    root = config.runs_root(config.load())
    drawn = [r["qm9_index"] for r in sc.read_draw(root, args.tag, args.name)]
    if not args.quiet:
        print("{}: scanning {} drawn molecule record(s) on {} ({} thread(s)) ...".format(
            args.stage, len(drawn), root, workers), flush=True)
    t0 = time.time()
    todo, n_failed = scan(drawn, args.stage, args.tag, root, force=args.force, workers=workers)
    scan_s = time.time() - t0
    mine = [q for i, q in enumerate(todo) if i % max(1, args.array_n) == args.array_id]
    if args.limit:
        mine = mine[:args.limit]
    Path(args.out).write_text("".join(q + "\n" for q in mine), encoding="utf-8")
    print("{}: {} drawn, {} pending, {} for task {}/{} -> {}{}".format(
        args.stage, len(drawn), len(todo), len(mine), args.array_id, args.array_n, args.out,
        "; {} failed earlier (_records/branchA.failed; not rerun)".format(n_failed) if n_failed else ""))
    if not args.quiet:
        print("{}: scan {:.1f} s for {} record(s), {} thread(s)".format(
            args.stage, scan_s, len(drawn), workers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
