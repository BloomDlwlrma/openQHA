"""The molecule list of a campaign stage: which molecules of a draw still need branch A
(`--stage branchA`) or a Frame set (`--stage frames`), this array task's round-robin share,
capped at LIMIT. One line per molecule (qm9_index), read by the xargs loops of
hpc/slurm/hl_branchA.slurm and hl_frames.slurm. Login-node cheap: two globs, no MACE.

TOOLING.

    python hpc/slurm/hl_list.py --tag draw300 --stage branchA --out list.txt
    python hpc/slurm/hl_list.py --tag draw300 --stage frames --array-id 3 --array-n 12 --limit 16 --out list.txt
"""
import argparse
import sys
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--name", default=None, help="the draw (00_draw.py --name; default: the tag)")
    ap.add_argument("--stage", choices=("branchA", "frames"), required=True)
    ap.add_argument("--array-id", type=int, default=0)
    ap.add_argument("--array-n", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="at most N molecules for this task (the debug gate)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    args.name = args.name or args.tag
    root = config.runs_root(config.load())
    drawn = [r["qm9_index"] for r in sc.read_draw(root, args.tag, args.name)]
    todo = []
    for qid in drawn:
        mol = basins.molecule_for(qid, args.tag, root=root)
        if args.stage == "branchA":
            if not basins.done(qid, args.tag, root=root):
                todo.append(qid)
        else:
            if basins.done(qid, args.tag, root=root) and not (layout.frames_dir(mol) / (frames.STEP + ".toml")).is_file():
                todo.append(qid)
    mine = [q for i, q in enumerate(todo) if i % max(1, args.array_n) == args.array_id]
    if args.limit:
        mine = mine[:args.limit]
    Path(args.out).write_text("".join(q + "\n" for q in mine), encoding="utf-8")
    print("{}: {} drawn, {} pending, {} for task {}/{} -> {}".format(
        args.stage, len(drawn), len(todo), len(mine), args.array_id, args.array_n, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
