"""Workflow hessian_learning, step 04: the Dataset -- split, files, index (ticket 04).

PRODUCTION. `openqha.data.dataset.build` over the selection of step 01: test = whole
molecules (the pinned seven + a per-stratum draw), valid = a fraction of the training
molecules' labelled frames by frame, train = the rest, pool = frames without a label at
the level yet. Writes `{train,valid,test,pool}.<level>.extxyz`, `index.dat` and
`dataset.{out,toml}` under `<root>/<first tag>/_datasets/<name>/`; `--export openreact`
adds `molecules-<name>.h5` in OpenREACT's layout (A, Eh, Eh/A, Eh/A^2).

    python workflows/hessian_learning/04_dataset.py --tag rings --tag propanal --name smoke
    python workflows/hessian_learning/04_dataset.py --tag rings --name smoke --export openreact
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
from openqha.data import dataset, frame_labels               # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", action="append", required=True, help="the tags of step 01, in the same order")
    ap.add_argument("--name", default=None, help="the Dataset name (default: the first tag)")
    ap.add_argument("--level", default=frame_labels.DEFAULT_LEVEL, help="the reference level of the labelled splits")
    ap.add_argument("--valid-fraction", type=float, default=dataset.VALID_FRACTION)
    ap.add_argument("--test-fraction", type=float, default=dataset.TEST_FRACTION)
    ap.add_argument("--seed", type=int, default=dataset.SEED)
    ap.add_argument("--export", choices=("openreact",), default=None)
    args = ap.parse_args()
    root = config.runs_root(config.load())
    args.name = args.name or args.tag[0]
    out = dataset.build(root, args.tag, args.name, level=args.level, valid_fraction=args.valid_fraction,
                        test_fraction=args.test_fraction, seed=args.seed)
    i = out["info"]
    for m in out["molecules"]:
        print("{:18s} {:9s} {:5s} {} frames {:3d} labelled {:3d}  train {:3d} valid {:3d} test {:3d} pool {:3d}".format(
            m["QM9_INDEX"], m["STRATUM"], m["MOLECULE_SPLIT"], "PINNED" if m["PINNED"] else "      ",
            m["N_FRAMES"], m["N_LABELLED"], m["N_TRAIN"], m["N_VALID"], m["N_TEST"], m["N_POOL"]))
    print("dataset {!r} at {}: {} molecules ({} test), {} frames: train {} valid {} test {} pool {}  -> {}".format(
        i["NAME"], i["LEVEL"], i["N_MOLECULES"], i["N_TEST_MOLECULES"], i["N_FRAMES"], i["N_TRAIN"], i["N_VALID"],
        i["N_TEST"], i["N_POOL"], out["dir"]))
    if args.export == "openreact":
        p = dataset.export_openreact(out["dir"], args.level, args.name)
        print("exported {} ({} labelled frames)".format(p, i["N_LABELLED"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
