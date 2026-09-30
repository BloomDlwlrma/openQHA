"""Workflow hessian_learning, step 04: the Dataset -- split, files, index.

PRODUCTION. `openqha.data.dataset.build` over the selection of step 01. `--split-by molecule`
(the default; MACE-OFF's granularity): test = whole molecules -- the pinned
seven plus a per-stratum draw of TEST_FRACTION (5 %) -- so conformers of one molecule never
sit on both sides; valid = VALID_FRACTION (5 %) of the TRAINING molecules' labelled frames,
drawn by frame; train = the rest. `--split-by frame` (the smoke / fit mode): the pinned
seven are whole test molecules, every other labelled frame goes to
train / valid / test at 90 / 5 / 5 by its own seeded draw. Changing the split mode of an
existing Dataset is refused by the build -- one index cannot hold two split schemes (the
library's `resplit=True` is the escape; this workflow does not expose it). pool = frames without a label at the level yet. `--train-generators basin`
(the default): only basin frames may train; every labelled frame of
another generator (displaced, merged, saddle) is a held-out frame in test, read by the
judge and never trained on. Writes
`{train,valid,test,pool}.<level>.extxyz` (with MACE-torch's REF_energy / REF_forces /
REF_hessian keys and `split`), the single `mace_<name>.<level>.extxyz`, `index.dat`

THE FIXED VALIDATION PROBES. Every labelled frame with a Hessian is written with
its own `[VALID_PROBE_KMAX, 3N]` standard-normal set (PHL's Algorithm 1;
`valid_probes` / `REF_valid_probes`),
drawn HERE from the frame's identity and this Dataset's seed. The training loss draws
fresh probes every step; the VALIDATION reading takes the first k rows of this set and
draws nothing, so nothing in the loss depends on the Label's bytes and a label recomputed
at the same level leaves the probes untouched. The rows are nested: k may be lowered on
the command line without a rebuild. This is PHL's fixed-vector protocol.
(with `classes`) and `dataset.{out,toml}` (with a `[[Class]]` table) under
`<root>/<first tag>/_datasets/<name>/`; `--export openreact` adds `molecules-<name>.h5`
in OpenREACT's layout (A, Eh, Eh/A, Eh/A^2).

    python workflows/hessian_learning/04_dataset.py --tag draw300 --export openreact
    python workflows/hessian_learning/04_dataset.py --tag rings --tag propanal --name smoke --split-by frame
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
from openqha.data import dataset, frame_labels, frames       # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", action="append", required=True, help="the tags of step 01, in the same order")
    ap.add_argument("--name", default=None, help="the Dataset name (default: the first tag)")
    ap.add_argument("--level", default=frame_labels.DEFAULT_LEVEL, help="the reference level of the labelled splits")
    ap.add_argument("--split-by", choices=dataset.SPLIT_MODES, default=dataset.DEFAULT_SPLIT_MODE,
                    help="molecule: whole test molecules, MACE-OFF's granularity (the production mode); "
                         "frame: 90/5/5 of the non-pinned labelled frames (the smoke / fit set)")
    ap.add_argument("--valid-fraction", type=float, default=None, help="default: 0.05 in both modes")
    ap.add_argument("--test-fraction", type=float, default=None, help="default: 0.05 in both modes")
    ap.add_argument("--seed", type=int, default=dataset.SEED)
    ap.add_argument("--no-pinned", action="store_true",
                    help="drop the pinned rule: the seven split by frame like any other molecule. A FIT Dataset "
                         "for measuring cost, the epoch-0 balance and w_H -- never a claim about "
                         "generalisation, and the Record says PURPOSE = fit")
    ap.add_argument("--train-generators", nargs="+", choices=list(frames.GENERATORS), default=list(dataset.TRAIN_GENERATORS),
                    metavar="GEN", help="the generators whose frames may train (default: basin); the others are held out "
                                        "in test for the judge")
    ap.add_argument("--export", choices=("openreact",), default=None)
    args = ap.parse_args()
    root = config.runs_root(config.load())
    args.name = args.name or args.tag[0]
    out = dataset.build(root, args.tag, args.name, level=args.level, split_by=args.split_by,
                        valid_fraction=args.valid_fraction, test_fraction=args.test_fraction, seed=args.seed,
                        pinned=() if args.no_pinned else dataset.PINNED,
                        purpose="fit" if args.no_pinned else "judge", train_generators=tuple(args.train_generators))
    i = out["info"]
    for m in out["molecules"]:
        print("{:18s} {:9s} {:5s} {} frames {:3d} labelled {:3d}  train {:3d} valid {:3d} test {:3d} pool {:3d}".format(
            m["QM9_INDEX"], m["STRATUM"], m["MOLECULE_SPLIT"], "PINNED" if m["PINNED"] else "      ",
            m["N_FRAMES"], m["N_LABELLED"], m["N_TRAIN"], m["N_VALID"], m["N_TEST"], m["N_POOL"]))
    print("per class:  {:22s} {:>5s} {:>6s} {:>8s} {:>6s} {:>5s} {:>5s} {:>5s}".format(
        "class", "mols", "frames", "labelled", "train", "valid", "test", "pool"))
    for c in out["classes"]:
        print("            {:22s} {:5d} {:6d} {:8d} {:6d} {:5d} {:5d} {:5d}".format(
            c["CLASS"], c["N_MOLECULES"], c["N_FRAMES"], c["N_LABELLED"], c["N_TRAIN"], c["N_VALID"], c["N_TEST"], c["N_POOL"]))
    print("dataset {!r} at {} (split by {}): {} molecules ({} test), {} frames: train {} valid {} test {} pool {}; "
          "{} with a Hessian  -> {}".format(
              i["NAME"], i["LEVEL"], i["SPLIT_BY"], i["N_MOLECULES"], i["N_TEST_MOLECULES"], i["N_FRAMES"], i["N_TRAIN"],
              i["N_VALID"], i["N_TEST"], i["N_POOL"], i["N_HESSIAN_FRAMES"], out["dir"]))
    print("generators: train {} (basin frames in train {}); held out {} ({} labelled frames in test for the judge)".format(
        " ".join(i["TRAIN_GENERATORS"]), i["N_TRAIN_BASIN"], " ".join(i["HELD_OUT_GENERATORS"]) or "-", i["N_TEST_HELD_OUT"]))
    print("Replay (production): {:,} SPICE train frames, one seed ({}), written twice -- config_weight {} -- "
          "so both arms hold the same frame set and differ only in the stored weight:".format(
              dataset.REPLAY_N_FRAMES, dataset.REPLAY_SEED,
              " and ".join("{:g}".format(w) for w in dataset.REPLAY_WEIGHTS)))
    for w in dataset.REPLAY_WEIGHTS:
        print("    python scripts/tooling/s0_spice_pt_draw.py --n {} --seed {} --weight {:g} --out "
              "<root>/spice/spice_pt_replay30k_w{:g}.extxyz".format(
                  dataset.REPLAY_N_FRAMES, dataset.REPLAY_SEED, w, w))
    print("R4 (the S0 replay ladder's scan row, 2026-09-21/22 -- historical; not the production Replay): "
          "{} x {} train frames with a Hessian = {} frames at config_weight {:g}".format(
              dataset.REPLAY_PER_HESSIAN_FRAME_R4, i["N_TRAIN_HESSIAN"], i["REPLAY_R4_FRAMES"],
              dataset.REPLAY_CONFIG_WEIGHT_R4))
    if i["MERGED_FILE"] != "-":
        print("merged  {} ({} labelled frames, keys REF_energy / REF_forces / REF_hessian / split)".format(
            i["MERGED_FILE"], i["N_LABELLED"]))
    if args.export == "openreact":
        p = dataset.export_openreact(out["dir"], args.level, args.name)
        print("exported {} ({} labelled frames)".format(p, i["N_LABELLED"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
