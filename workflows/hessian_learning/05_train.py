"""Workflow hessian_learning, step 05: the fine-tune (ticket 13).

PRODUCTION. openQHA's own entry point into the mace fork's training loop, in mace-md's
shape: this driver builds mace's arguments and calls `mace.cli.run_train.run`, the
Hessian label comes from the Dataset's `mace_<name>.<level>.extxyz` (`REF_hessian`,
fork commit A) and the loss from `openqha.training.phl_loss` (`--loss external
--loss_module`, fork commit B). Nothing here reimplements a training loop.

    python workflows/hessian_learning/05_train.py --tag smoke --run w1 --dry-run
    python workflows/hessian_learning/05_train.py --tag smoke --run w1 --hessian-weight 0.01 --probe modes --max-epochs 2
    python workflows/hessian_learning/05_train.py --tag draw300 --run prod1 --device cuda \
        --hessian-weight 0.01 --n-probes 4 --multiheads --pt-train-file spice_pt_5000.extxyz

One run writes `<root>/<tag>/_datasets/<name>/train/<run>/`: mace's own files plus the
Record `train.{out,toml,dat}` (the settings, the epoch table, the Dataset, the base and
fine-tuned fingerprints, the config SHA, the mace fork's commit). `--register` prints the
`ENGINES` entry for the fine-tuned potential and, with `--register-copy`, puts the model
into `data/potentials/`.

The driver REFUSES a mace that is not the fork, or a dirty checkout: a potential whose
loss cannot be reproduced from a commit is not a product. `--no-strict-fork` is for
trying things out, and says so in the Record.
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
from openqha.training import phl, run as train_run           # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="the campaign tag (the Dataset lives under it)")
    ap.add_argument("--name", default=None, help="the Dataset name (default: the tag)")
    ap.add_argument("--level", default=frame_labels.DEFAULT_LEVEL, help="the reference level of the labels")
    ap.add_argument("--run", required=True, help="the run name; the directory is <dataset>/train/<run>/")
    ap.add_argument("--foundation", default=None, help="the base potential (default: S0_ENGINE)")
    # the loss (design-phl-loss.md eq. 11; T03)
    ap.add_argument("--energy-weight", type=float, default=1.0)
    ap.add_argument("--forces-weight", type=float, default=100.0)
    ap.add_argument("--hessian-weight", type=float, default=1.0, help="w_H; the smoke fit (ticket 15) measures it")
    ap.add_argument("--probe", choices=phl.PROBE_MODES, default="rademacher",
                    help="rademacher/gaussian: k probes per structure (eq. 6); modes/cartesian: the exact loss (eq. 10)")
    ap.add_argument("--n-probes", type=int, default=4, help="k of eq. 6 (ignored by the deterministic probe sets)")
    ap.add_argument("--mode-weighting", choices=("entropy", "none"), default="entropy",
                    help="entropy: w_i = |dS_msRRHO/d omega_i| at 298 K (eq. 3)")
    # the loop
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--valid-batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None, help="default: mace's")
    ap.add_argument("--seed", type=int, default=123, help="mace's seed; also the probe generator's")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    # forgetting (round-2 Q7 a)
    ap.add_argument("--multiheads", action="store_true", help="replay a pretraining head beside the fine-tuning head")
    ap.add_argument("--pt-train-file", default=None, help="the replay frames (a SPICE subsample)")
    ap.add_argument("--num-samples-pt", type=int, default=5000)
    # the rest
    ap.add_argument("--mace-arg", action="append", default=[], metavar="ARG",
                    help="passed to mace as is, repeatable (e.g. --mace-arg --swa)")
    ap.add_argument("--no-strict-fork", action="store_true", help="train on a non-fork or dirty mace anyway")
    ap.add_argument("--register", action="store_true", help="print the ENGINES entry of the fine-tuned potential")
    ap.add_argument("--register-copy", action="store_true", help="also copy the model into data/potentials/")
    ap.add_argument("--dry-run", action="store_true", help="print mace's command and the Record header, then stop")
    args = ap.parse_args()

    name = args.name or args.tag
    root = config.runs_root(config.load())
    dataset_dir = dataset.datasets_dir(root, args.tag, name)
    out = train_run.run_training(
        dataset_dir, args.tag, name, args.level, args.run,
        foundation=args.foundation, dry_run=args.dry_run, strict_fork=not args.no_strict_fork,
        energy_weight=args.energy_weight, forces_weight=args.forces_weight,
        hessian_weight=args.hessian_weight, probe=args.probe, n_probes=args.n_probes,
        mode_weighting=args.mode_weighting, max_epochs=args.max_epochs, batch_size=args.batch_size,
        valid_batch_size=args.valid_batch_size, lr=args.lr, seed=args.seed, device=args.device,
        multiheads=args.multiheads, pt_train_file=args.pt_train_file, num_samples_pt=args.num_samples_pt,
        extra=args.mace_arg)
    info = out["info"]

    print("run {}  {}".format(info["RUN"], out["run_dir"]))
    print("  dataset      {} at {}".format(info["NAME"], info["LEVEL"]))
    print("  train        {} frames, {} with a reference Hessian".format(info["N_TRAIN"], info["N_TRAIN_HESSIAN"]))
    print("  valid        {} frames, {} with a reference Hessian".format(info["N_VALID"], info["N_VALID_HESSIAN"]))
    print("  loss         w_E {} w_F {} w_H {}; probe {} k={}; {} weighting".format(
        info["ENERGY_WEIGHT"], info["FORCES_WEIGHT"], info["HESSIAN_WEIGHT"], info["PROBE"],
        info["N_PROBES"], info["MODE_WEIGHTING"]))
    print("  base         {}  {}...".format(info["FOUNDATION_MODEL"], info["FOUNDATION_PARAMS_SHA256"][:12]))
    print("  mace         {}  fork {}...".format(info["MACE_VERSION"], info["MACE_FORK_COMMIT"][:12]))
    if out["dry_run"]:
        print("\nmace_run_train \\\n  " + " \\\n  ".join(
            " ".join(out["argv"][i:i + 2]) for i in range(0, len(out["argv"]), 2)))
        return 0
    print("  epochs       {} in {:.1f} s ({:.1f} s per epoch)".format(
        info["N_EPOCHS"], info["SECONDS"], info["SECONDS_PER_EPOCH"]))
    print("  model        {}  {}...".format(info["MODEL_FILE"], (info.get("MODEL_PARAMS_SHA256") or "-")[:12]))
    if out["epochs"]:
        print("  {:>5s} {:6s} {:>12s} {:>10s} {:>10s}".format("epoch", "split", "loss", "E meV/at", "F meV/A"))
        for r in out["epochs"][-8:]:
            print("  {:5d} {:6s} {:>12} {:>10} {:>10}".format(
                r["epoch"], r["split"],
                "-" if r["loss"] is None else "{:.6f}".format(r["loss"]),
                "-" if r["rmse_e_per_atom_meV"] is None else "{:.2f}".format(r["rmse_e_per_atom_meV"]),
                "-" if r["rmse_f_meV_A"] is None else "{:.2f}".format(r["rmse_f_meV_A"])))
    if args.register or args.register_copy:
        entry = train_run.registry_entry(info)
        if args.register_copy and Path(info["MODEL_FILE"]).is_file():
            import shutil
            from openqha.potentials import engine
            target = engine.model_root() / entry["filename"]
            shutil.copy2(info["MODEL_FILE"], target)
            print("  copied       {}".format(target))
        print("\nENGINES entry (openqha/potentials/engine.py):")
        print('    "{}": dict('.format(entry["name"]))
        print('        filename="{}",'.format(entry["filename"]))
        print('        source="{}",'.format(entry["source"]))
        print('        note="{}",'.format(entry["note"]))
        print('        params_sha256="{}",'.format(entry["params_sha256"] or ""))
        print("    ),")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
