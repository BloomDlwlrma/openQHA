"""Workflow hessian_learning, step 00: the structure-class draw.

PRODUCTION. `openqha.data.structure_classes.draw`: from the gated QM9 targets outside
MACE-OFF23's SPICE training file (any match level), PER_CLASS molecules per structure
class of `configs/structure_classes.yaml`, the union over classes, the pinned seven
always in; `draw.{out,toml,dat}` under `<root>/<tag>/_datasets/<name>/`. `draw.dat` is the
molecule list every stage script of the campaign reads. Login node, seconds.

    python workflows/hessian_learning/00_draw.py --tag draw500 --per-class 500 --seed 0
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
from openqha.data import dataset, structure_classes as sc    # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="the tag the campaign's molecule directories go under")
    ap.add_argument("--name", default=None, help="the Dataset name (default: the tag -- one campaign, one tag, one Dataset)")
    ap.add_argument("--per-class", type=int, default=sc.PER_CLASS)
    ap.add_argument("--seed", type=int, default=sc.SEED)
    ap.add_argument("--classes", default=str(sc.CLASSES_FILE))
    ap.add_argument("--membership", default=str(dataset.MEMBERSHIP_FILE))
    ap.add_argument("--include-training", action="store_true", help="do NOT exclude molecules in SPICE (default: exclude)")
    args = ap.parse_args()
    root = config.runs_root(config.load())
    args.name = args.name or args.tag
    rows = sc.draw(root, args.tag, args.name, per_class=args.per_class, seed=args.seed, classes_file=args.classes,
                   membership_file=args.membership, exclude_training=not args.include_training)
    d = dataset.datasets_dir(root, args.tag, args.name)
    print(open(d / (sc.STEP + ".out"), encoding="utf-8").read())
    print("draw: {} molecules -> {}".format(len(rows), d / (sc.STEP + ".dat")))
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
