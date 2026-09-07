"""Print what is left to compute, one QM9 identifier per line. Scan, then execute.

PRODUCTION. Branch E. It computes nothing and decides nothing about chemistry: it applies
the F0-F7 gates and the completion rule that openqha/worklist.py already defines, and
prints what is left.

This is the small piece both submission routes need and neither should re-implement:

    Parsl route    s0_E_branchA_parsl.py --range calls openqha.worklist itself
    yhbatch route  hpc/slurm/branchA_deimos.slurm calls THIS, and pipes it into xargs

Writing the scan twice is how a scanner and a worker come to disagree about what "done"
means, after which molecules are skipped for ever and never recovered. So the definition
lives in `openqha/worklist.py`, both routes call it, and this file only adds sharding and
a plain-text output an `xargs` can read.

    # what would run, and why it decided that
    python scripts/production/s0_E_worklist.py --range 1 16000 --tag prod --explain

    # node 3 of 12, ids only
    python scripts/production/s0_E_worklist.py --range 1 16000 --tag prod --shard 3/12

SHARDING IS BY POSITION IN THE REMAINING LIST, NOT BY INDEX RANGE
-----------------------------------------------------------------
`--shard k/n` takes every n-th entry of the worklist. QM9 indices are not contiguous and
the F0-F7 gates drop molecules unevenly, so splitting the INDEX range into n blocks gives
n very unequal jobs -- one node finishes in an hour and another runs for three days. A
stride over the already-filtered list gives n balanced ones.

**The scan is done once, by the submitting process, and each shard re-does it.** That is
deliberate: a shard that starts six hours later must not run molecules a neighbour has
since finished. The scan is a directory listing per shard directory, not a stat per
molecule (the lesson from `count_remain_fixed.py`), so re-doing it is cheap.
"""
import argparse
import json
import sys
from pathlib import Path


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))

from openqha import config, worklist  # noqa: E402


def parse_shard(text):
    """'3/12' -> (2, 12), zero-based. Refuses the off-by-one rather than absorbing it."""
    if not text:
        return None
    try:
        k, n = (int(x) for x in str(text).split("/"))
    except ValueError:
        raise SystemExit("--shard wants K/N, for example 3/12; got {!r}".format(text))
    if n < 1 or k < 1 or k > n:
        raise SystemExit(
            "--shard {}/{} is out of range. K counts from 1 and must not exceed N."
            .format(k, n))
    return k - 1, n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--range", nargs=2, type=int, metavar=("START", "END"),
                    default=None, help="QM9 index range, inclusive")
    ap.add_argument("--edges", action="store_true",
                    help="use the configured edge set instead of a range")
    ap.add_argument("--tag", default="prod",
                    help="campaign tag: 'already complete' is a question about one "
                         "campaign's shard, not about the repository")
    ap.add_argument("--shard", default=None, metavar="K/N",
                    help="take every N-th entry, offset K-1. Balanced by construction.")
    ap.add_argument("--no-gates", action="store_true",
                    help="do not apply the F0-F7 species filters")
    ap.add_argument("--explain", action="store_true",
                    help="print the decision record to stderr as well")
    args = ap.parse_args()

    cfg = config.load()
    if args.edges:
        candidates = []
        for edge in config.edges(cfg):
            for qid in config.edge_species(edge):
                if qid not in candidates:
                    candidates.append(qid)
    elif args.range:
        candidates = worklist.index_range(*args.range)
    else:
        raise SystemExit("give --range START END or --edges")

    todo, record = worklist.remaining(candidates, cfg=cfg, tag=args.tag,
                                      apply_gates=not args.no_gates)

    shard = parse_shard(args.shard)
    if shard:
        k, n = shard
        record["shard"] = "{}/{}".format(k + 1, n)
        record["n_remaining_all_shards"] = len(todo)
        todo = todo[k::n]
        record["n_remaining_this_shard"] = len(todo)

    if args.explain:
        # stderr, so that `--explain` can be left on in a job script without polluting
        # the list that xargs is reading from stdout.
        print(json.dumps(record, indent=2, ensure_ascii=False), file=sys.stderr)
        if record.get("warning"):
            print("WARNING: " + record["warning"], file=sys.stderr)

    for qid in todo:
        print(qid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
