"""Workflow hessian_learning, step 05: the fine-tune.

PRODUCTION. openQHA's own entry point into the mace fork's training loop, in mace-md's
shape: this driver builds mace's arguments and calls `mace.cli.run_train.run`, the
Hessian label comes from the Dataset's `mace_<name>.<level>.extxyz` (`REF_hessian`,
fork commit A), the loss from `openqha_hessian.phl_loss` (`--loss external
--loss_module`, fork commit B), kept in multihead mode and given the force graph at
evaluation (commit C). Nothing here reimplements a training loop.

    python workflows/hessian_learning/05_train.py --tag smoke --run w1 --dry-run
    python workflows/hessian_learning/05_train.py --tag smoke --run w1 --max-epochs 2             # w_H = balance (default)
    # the production arms: a 30,000-frame Replay, one seed, written twice (weights 1 and 10; the weight
    # lives in the file) -- one job per arm, submitted together (replay30k_w1 / replay30k_w10); `--name`
    # names the Dataset (draw300_r1; the default is the tag, which is not where the Dataset lives)
    python workflows/hessian_learning/05_train.py --tag draw300 --name draw300_r1 --run replay30k_w1 --device cuda --max-epochs 100 \
        --multiheads --pt-train-file $S0_RUNS_ROOT/spice/spice_pt_replay30k_w1.extxyz --pt-valid-file $S0_RUNS_ROOT/spice/spice_pt_replay30k_w1.valid.extxyz

THE TARGET is the Cartesian matrix itself and nothing else (there is no switch);
validation uses four standard-normal probes fixed per frame, drawn by the Dataset and read
from the file, and every control flag
(`--lr`, `--scheduler-patience`, `--patience`, `--eval-interval`, `--ema`, Stage Two with
`--start-swa`, `--swa-lr` and the Stage Two weights) is explicit and in the Record.

THE REPLAY: `--multiheads --pt-train-file FILE` concatenates the file
`scripts/tooling/s0_spice_pt_draw.py` writes -- the only source of a Replay file -- as
mace's pretraining head. The file IS the size knob: there is no `--num-samples-pt` (mace
reads that flag only on its Materials-Project path), and mace's duplication threshold is
passed as 0. The Record counts the file's frames, reads their `config_weight`, parses
both heads' counts from mace's log and prints the ratio as replay frames per Hessian
frame. `--pt-valid-file` names the draw tool's companion validation file; without it
mace takes `--valid-fraction` (10 %) of the Replay for the pretraining head's validation.

One run writes `<root>/<tag>/_datasets/<name>/train/<run>/`: mace's own files plus the
Record `train.{out,toml,dat}` (the settings, the epoch table with the three validation
curves, the Dataset, the base potential's resolved file and the fine-tuned one's, the
config SHA, the mace fork's commit, this package's version and commit). `--register`
prints the `ENGINES` entry for the fine-tuned potential and, with `--register-copy`,
puts the model into `data/potentials/`.

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
from openqha_hessian import phl, phl_loss                    # noqa: E402
from openqha_hessian import run as train_run                 # noqa: E402


def _weight(text):
    return "balance" if str(text).strip().lower() == "balance" else float(text)


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
    ap.add_argument("--hessian-weight", type=_weight, default="balance",
                    help="w_H: a number, or `balance` (default) = w_F L_F / L_H measured on the base model over the "
                         "run's train file with the Cartesian target before the first step")
    ap.add_argument("--probe", choices=phl.PROBE_MODES, default="gaussian",
                    help="gaussian (PHL's draw, the default) / rademacher: k probes per structure (eq. 6'); "
                         "cartesian: the exact loss (3N probes)")
    ap.add_argument("--n-probes", type=int, default=4, help="k of eq. 6' (ignored by the deterministic probe set)")
    # the loop
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--valid-batch-size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=123, help="mace's seed; also the training probe generator's")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    # the control (the base's recipe)
    ap.add_argument("--lr", type=float, default=None, help="default {} (mace's)".format(train_run.DEFAULT_LR))
    ap.add_argument("--scheduler-patience", type=int, default=train_run.DEFAULT_SCHEDULER_PATIENCE,
                    help="ReduceLROnPlateau patience on the total validation loss (the base: 20)")
    ap.add_argument("--patience", type=int, default=train_run.DEFAULT_PATIENCE,
                    help="early-stopping patience on the total validation loss (the base: 50)")
    ap.add_argument("--eval-interval", type=int, default=train_run.DEFAULT_EVAL_INTERVAL)
    ap.add_argument("--no-ema", action="store_true", help="no exponential moving average of the parameters")
    ap.add_argument("--no-swa", action="store_true", help="no Stage Two")
    ap.add_argument("--start-swa", type=int, default=None, help="Stage Two start epoch (default 3/4 of --max-epochs)")
    ap.add_argument("--swa-lr", type=float, default=None, help="Stage Two learning rate (default lr / 40, the base's ratio)")
    ap.add_argument("--swa-energy-weight", type=float, default=train_run.DEFAULT_SWA_ENERGY_WEIGHT)
    ap.add_argument("--swa-forces-weight", type=float, default=train_run.DEFAULT_SWA_FORCES_WEIGHT)
    ap.add_argument("--swa-hessian-weight", type=float, default=None,
                    help="default w_H x swa_forces_weight / forces_weight")
    # the Replay
    ap.add_argument("--multiheads", action="store_true",
                    help="concatenate a Replay (mace's pretraining head) beside the fine-tuning head")
    ap.add_argument("--pt-train-file", default=None,
                    help="the Replay file written by scripts/tooling/s0_spice_pt_draw.py -- its frame count IS the size")
    ap.add_argument("--no-exact-anchors", action="store_true",
                    help="skip the two full-matrix readings of the Hessian term on the validation file (before and "
                         "after training); each costs about a third of one epoch")
    ap.add_argument("--pt-valid-file", default=None,
                    help="the Replay's companion validation file (<stem>.valid.extxyz of the draw tool)")
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
        exact_anchors=not args.no_exact_anchors,
        max_epochs=args.max_epochs, batch_size=args.batch_size,
        valid_batch_size=args.valid_batch_size, lr=args.lr, seed=args.seed, device=args.device,
        scheduler_patience=args.scheduler_patience, patience=args.patience, eval_interval=args.eval_interval,
        ema=not args.no_ema, swa=not args.no_swa, start_swa=args.start_swa, swa_lr=args.swa_lr,
        swa_energy_weight=args.swa_energy_weight, swa_forces_weight=args.swa_forces_weight,
        swa_hessian_weight=args.swa_hessian_weight,
        multiheads=args.multiheads, pt_train_file=args.pt_train_file, pt_valid_file=args.pt_valid_file,
        extra=args.mace_arg)
    info = out["info"]

    print("run {}  {}".format(info["RUN"], out["run_dir"]))
    print("  dataset      {} at {}".format(info["NAME"], info["LEVEL"]))
    print("  train        {} frames, {} with a reference Hessian".format(info["N_TRAIN"], info["N_TRAIN_HESSIAN"]))
    print("  valid        {} frames, {} with a reference Hessian".format(info["N_VALID"], info["N_VALID_HESSIAN"]))
    print("  loss         w_E {} w_F {} w_H {} ({}); probe {} k={}; validation {}".format(
        info["ENERGY_WEIGHT"], info["FORCES_WEIGHT"], info["HESSIAN_WEIGHT"], info["HESSIAN_WEIGHT_RULE"], info["PROBE"],
        info["N_PROBES"], info["VALID_PROBES"]))
    if info["HESSIAN_WEIGHT_RULE"] == "balance":
        print("  balance      base model on the train file: L_E {:.3e} L_F {:.3e} L_H {:.3e} -> w_H = w_F L_F / L_H".format(
            info["BALANCE_L_E"], info["BALANCE_L_F"], info["BALANCE_L_H"]))
    print("  control      lr {} scheduler_patience {} patience {} eval_interval {} ema {} swa {} start_swa {} swa_lr {}".format(
        info["LR"], info["SCHEDULER_PATIENCE"], info["PATIENCE"], info["EVAL_INTERVAL"], info["EMA"], info["SWA"],
        info["START_SWA"], info["SWA_LR"]))
    print("  stage two    w_E {} w_F {} w_H {}".format(
        info["SWA_ENERGY_WEIGHT"], info["SWA_FORCES_WEIGHT"], info["SWA_HESSIAN_WEIGHT"]))
    if info["MULTIHEADS"]:
        print("  replay       {} frames from {} (config_weight {}), {:.3f} per Hessian frame; valid file {}".format(
            info["PT_N_FRAMES"], info["PT_TRAIN_FILE"], info["PT_CONFIG_WEIGHT"], info["REPLAY_PER_HESSIAN_FRAME"],
            info["PT_VALID_FILE"]))
    print("  base         {}  {}".format(info["FOUNDATION_MODEL"], info["FOUNDATION_FILE"]))
    print("  mace         {}  fork {}...".format(info["MACE_VERSION"], info["MACE_FORK_COMMIT"][:12]))
    if out["dry_run"]:
        print("\nmace_run_train \\\n  " + " \\\n  ".join(
            " ".join(out["argv"][i:i + 2]) for i in range(0, len(out["argv"]), 2)))
        return 0
    print("  epochs       {} in {:.1f} s ({:.1f} s per epoch)".format(
        info["N_EPOCHS"], info["SECONDS"], info["SECONDS_PER_EPOCH"]))
    print("  model        {}".format(info["MODEL_FILE"]))
    if info["MULTIHEADS"]:
        print("  heads        pt train {} valid {}; fine-tune train {} valid {} (mace's log)".format(
            info["PT_HEAD_TRAIN"], info["PT_HEAD_VALID"], info["FT_HEAD_TRAIN"], info["FT_HEAD_VALID"]))
    print("  stage two    switched at epoch {}; Hessian curve moved: {}".format(
        info["STAGE_TWO_EPOCH"], info["HESSIAN_CURVE_MOVED"]))
    if not info["HESSIAN_CURVE_MOVED"]:
        print("  WARNING      the validation Hessian term did not move across the epochs")
    if out["epochs"]:
        print("  {:>5s} {:6s} {:>12s} {:>10s} {:>10s} {:>12s} {:>12s} {:>12s}".format(
            "epoch", "split", "loss", "E meV/at", "F meV/A", "valid E", "valid F", "valid H"))
        for r in out["epochs"][-8:]:
            print("  {:5d} {:6s} {:>12} {:>10} {:>10} {:>12} {:>12} {:>12}".format(
                r["epoch"], r["split"],
                "-" if r["loss"] is None else "{:.6f}".format(r["loss"]),
                "-" if r["rmse_e_per_atom_meV"] is None else "{:.2f}".format(r["rmse_e_per_atom_meV"]),
                "-" if r["rmse_f_meV_A"] is None else "{:.2f}".format(r["rmse_f_meV_A"]),
                "-" if r.get("valid_energy") is None else "{:.4e}".format(r["valid_energy"]),
                "-" if r.get("valid_forces") is None else "{:.4e}".format(r["valid_forces"]),
                "-" if r.get("valid_hessian") is None else "{:.4e}".format(r["valid_hessian"])))
    if args.register or args.register_copy:
        entry = train_run.registry_entry(info)
        if args.register_copy and Path(info["MODEL_FILE"]).is_file():
            import shutil
            from openqha.potentials import engine
            target = engine.model_root() / entry["filename"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(info["MODEL_FILE"], target)
            print("  copied       {}".format(target))
        print("\nENGINES entry (openqha/potentials/engine.py):")
        print('    "{}": dict('.format(entry["name"]))
        print('        filename="{}",'.format(entry["filename"]))
        print('        source="{}",'.format(entry["source"]))
        print('        note="{}",'.format(entry["note"]))
        print("    ),")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
