"""Workflow hessian_learning, step 01: the molecule list of a Dataset (ticket 04).

PRODUCTION. `openqha.data.dataset.select`: every molecule of the draw (`draw.dat` of
step 00, when the Dataset has one -- listed whether or not branch A has run for it, so
this table is the campaign's progress) and every molecule under the given tags whose
branch A finished, with its stratification keys (ring count, heteroatom pattern, heavy
atoms), its structure classes, its MACE-OFF23 training-set membership, whether it is
one of the seven pinned molecules and whether basins / a Frame set exist. Writes
`select.{out,toml,dat}` under `<root>/<first tag>/_datasets/<name>/`. `--limit N` keeps
the pinned molecules plus the first N - pinned others, or, with `--stratify`, others
spread evenly over the sorted index list (QM9 is ordered by heavy-atom count).

    python workflows/hessian_learning/01_select.py --tag rings --tag propanal --name smoke
    python workflows/hessian_learning/01_select.py --tag draw300 --limit 200 --stratify
"""
import argparse
import sys
from pathlib import Path


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))
from openqha import config                                   # noqa: E402
from openqha.data import dataset                             # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", action="append", required=True, help="a tag with branch A products (repeatable; the first holds the Dataset)")
    ap.add_argument("--name", default=None, help="the Dataset name (default: the first tag)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stratify", action="store_true")
    args = ap.parse_args()
    root = config.runs_root(config.load())
    args.name = args.name or args.tag[0]
    rows = dataset.select(root, args.tag, args.name, limit=args.limit, stratify=args.stratify)
    d = dataset.datasets_dir(root, args.tag[0], args.name)
    for r in rows:
        print("{:18s} {:10s} {:24s} {:9s} in_training={:8s} {} basins {:>3} frames {:>3}  {}".format(
            r["qm9_index"], r["tag"], r["smiles"][:24], r["stratum"], r["in_training"],
            "PINNED" if r["pinned"] else "      ", r["n_basins"] if r["has_basins"] else "-",
            r["n_frames"] if r["has_frames"] else "-", r["classes"]))
    print("select: {} molecules ({} drawn, {} with basins, {} with a Frame set, {} pinned) -> {}".format(
        len(rows), sum(1 for r in rows if r["drawn"]), sum(1 for r in rows if r["has_basins"]),
        sum(1 for r in rows if r["has_frames"]), sum(1 for r in rows if r["pinned"]), d / "select.dat"))
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
