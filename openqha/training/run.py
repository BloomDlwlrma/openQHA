"""The fine-tune: openQHA's own entry point into the mace fork's training loop.

PRODUCTION. Ticket 13 of the Hessian-learning set.

The shape is mace-md's: an entry point of OUR OWN builds mace's argument namespace and
calls `mace.cli.run_train.run(args)`. Nothing here reimplements a training loop; what
mace could not do -- read a Hessian label (commit A) and take a loss from outside
(commit B) -- lives in the fork, and the loss itself is `openqha.training.phl_loss`,
reached by name:

    --loss external --loss_module openqha.training.phl_loss:build

One run is a directory of the Dataset:

    <root>/<tag>/_datasets/<name>/train/<run>/
        config.yaml        every mace argument, as given (the file mace itself can re-run)
        <run>.model        the fine-tuned potential (and <run>_stagetwo.model with --swa)
        checkpoints/, logs/, results/    mace's own
        train.{out,toml}   the Record: the settings, the epoch table, the identities

The Record is what makes a fine-tuned potential traceable: the Dataset and its index, the
loss settings, the mace fork's commit, the base model's parameter fingerprint and the
fine-tuned one's, the SHA-256 of the config file. `05_train.py --register` turns those
into an `ENGINES` entry (ticket 01).

Refused, not warned: training against a mace that is not the fork (`mace_fork_commit`
"unknown"), or against a dirty checkout. A model whose loss cannot be reproduced from a
commit is not a product.
"""
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

from ..potentials import engine
from ..store import dat, property as prop, report

PROGNAME = "openQHA hl_train"
STEP = "train"

#: the loss module the fork's `--loss external` imports
LOSS_MODULE = "openqha.training.phl_loss:build"

#: MACE-torch's keys in the Dataset's merged file (`dataset.REF_*_KEY`)
ENERGY_KEY, FORCES_KEY, HESSIAN_KEY = "REF_energy", "REF_forces", "REF_hessian"

SCHEMA = {
    "Calculation_Info": {
        "PROGNAME": ("String", None, "the step that wrote this file"),
        "VERSION": ("String", None, "openQHA version"),
        "STATUS": ("String", None, "the completion marker"),
        "RUN": ("String", None, "the run name; the directory is <dataset>/train/<run>/"),
        "TAG": ("String", None, "the campaign tag"),
        "NAME": ("String", None, "the Dataset name"),
        "LEVEL": ("String", None, "the reference level of the labels"),
        "DATASET_DIR": ("String", None, "the Dataset directory"),
        "INDEX_FILE": ("String", None, "index.dat of the Dataset: which frame is in which split"),
        "TRAIN_FILE": ("String", None, "the extxyz mace trained on"),
        "VALID_FILE": ("String", None, "the extxyz mace validated on"),
        "N_TRAIN": ("Integer", None, "frames in the training file"),
        "N_VALID": ("Integer", None, "frames in the validation file"),
        "N_TRAIN_HESSIAN": ("Integer", None, "of those, frames carrying a reference Hessian"),
        "N_VALID_HESSIAN": ("Integer", None, "of those, frames carrying a reference Hessian"),
        "FOUNDATION_MODEL": ("String", None, "the base potential fine-tuned"),
        "FOUNDATION_PARAMS_SHA256": ("String", None, "parameter fingerprint of the base potential"),
        "MODEL_FILE": ("String", None, "the fine-tuned potential"),
        "MODEL_PARAMS_SHA256": ("String", None, "parameter fingerprint of the fine-tuned potential"),
        "MODEL_N_TENSORS": ("Integer", None, "tensors in the fine-tuned potential"),
        "CONFIG_FILE": ("String", None, "the mace argument file"),
        "CONFIG_SHA256": ("String", None, "SHA-256 of the config file: the training recipe's identity"),
        "LOSS": ("String", None, "the loss module and factory (mace's --loss external)"),
        "ENERGY_WEIGHT": ("Double", None, "w_E of eq. 11"),
        "FORCES_WEIGHT": ("Double", None, "w_F of eq. 11"),
        "HESSIAN_WEIGHT": ("Double", None, "w_H of eq. 11"),
        "PROBE": ("String", None, "rademacher / gaussian / modes / cartesian (Algorithm 1)"),
        "N_PROBES": ("Integer", None, "probes per structure per step (k of eq. 6)"),
        "MODE_WEIGHTING": ("String", None, "entropy (|dS_msRRHO/d omega|) or none"),
        "MAX_NUM_EPOCHS": ("Integer", None, "epochs asked for"),
        "N_EPOCHS": ("Integer", None, "epochs the log holds"),
        "BATCH_SIZE": ("Integer", None, "structures per step"),
        "SEED": ("Integer", None, "mace's seed; also the probe generator's"),
        "DEVICE": ("String", None, "cpu / cuda"),
        "DTYPE": ("String", None, "float64 throughout, as the Labels are"),
        "MULTIHEADS": ("Boolean", None, "a pretraining head replayed beside the fine-tuning head (round-2 Q7 a)"),
        "PT_TRAIN_FILE": ("String", None, "the replay frames, or - "),
        "NUM_SAMPLES_PT": ("Integer", None, "replay frames drawn per epoch"),
        "MACE_VERSION": ("String", None, "mace.__version__ (the fork: 0.3.16+openqha)"),
        "MACE_FORK": ("String", None, "the fork the training code came from"),
        "MACE_FORK_COMMIT": ("String", None, "commit of the mace checkout (refused when unknown or dirty)"),
        "ENGINE_PARAMS_SHA256": ("String", None, "alias of FOUNDATION_PARAMS_SHA256, for the Record readers"),
        "SECONDS": ("Double", "s", "wall time of the whole run"),
        "SECONDS_PER_EPOCH": ("Double", "s", "wall time / epochs"),
    },
    "Epoch": {
        "EPOCH": ("Integer", None, "epoch index as mace logged it"),
        "SPLIT": ("String", None, "train (the step's loss) or valid (mace's evaluation)"),
        "LOSS": ("Double", None, "the total loss of eq. 11"),
        "RMSE_E_PER_ATOM_MEV": ("Double", "meV", "energy RMSE per atom, when the log gives it"),
        "RMSE_F_MEV_A": ("Double", "meV/A", "force RMSE, when the log gives it"),
    },
}

EPOCH_ROW = {
    "epoch": ("Integer", None, "epoch index"),
    "split": ("String", None, "train / valid"),
    "loss": ("Double", None, "total loss"),
    "rmse_e_per_atom_meV": ("Double", "meV", "energy RMSE per atom"),
    "rmse_f_meV_A": ("Double", "meV/A", "force RMSE"),
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_fork(strict=True):
    """The mace that will train, as an identity. Refuses a non-fork or a dirty checkout
    (`strict`), because a loss that cannot be reproduced from a commit is not a product."""
    import mace
    info = engine.mace_fork_info()
    info["mace_version"] = mace.__version__
    if strict:
        if info["mace_fork_commit"] == "unknown":
            raise RuntimeError(
                "the installed mace at {} is not a git checkout: the Hessian loss needs the fork {} "
                "(pip uninstall -y mace-torch && pip install -e <path>/openQHA-Hessian)".format(
                    getattr(mace, "__file__", "?"), engine.MACE_FORK))
        if info["mace_fork_dirty"]:
            raise RuntimeError(
                "the mace checkout at {} has uncommitted changes to tracked files; commit them so the "
                "run's Record names the code that produced it".format(info["mace_fork_path"]))
    return info


def split_files(dataset_dir, name, level, out_dir):
    """mace takes one file per split; the Dataset writes one file with a `split` key.
    Split it into `out_dir` and count the frames and the Hessians of each."""
    from ase.io import read, write
    from ..data import dataset as dataset_mod
    merged = dataset_mod.merged_file(dataset_dir, name, level)
    if not Path(merged).is_file():
        raise FileNotFoundError("no merged Dataset file at {}; run 04_dataset.py first".format(merged))
    frames = read(str(merged), index=":", format="extxyz")
    out, counts = {}, {}
    for split in ("train", "valid"):
        rows = [a for a in frames if a.info.get("split") == split]
        if not rows:
            raise ValueError("the Dataset has no {} frames in {}".format(split, merged))
        path = Path(out_dir) / "{}.{}.extxyz".format(split, level)
        write(str(path), rows, format="extxyz")
        out[split] = path
        counts[split] = (len(rows), sum(1 for a in rows if HESSIAN_KEY in a.info))
    return out, counts


def mace_argv(train_file, valid_file, run, work_dir, foundation, level, *, energy_weight=1.0,
              forces_weight=100.0, hessian_weight=1.0, probe="rademacher", n_probes=4,
              mode_weighting="entropy", max_epochs=100, batch_size=4, valid_batch_size=None,
              seed=123, device="cpu", lr=None, multiheads=False, pt_train_file=None,
              num_samples_pt=5000, extra=()):
    """mace's command line for one fine-tune, as a list. The keys are the Dataset's
    (`REF_*`), the loss is ours by name, the dtype is float64 because the Labels are."""
    argv = [
        "--name", str(run),
        "--work_dir", str(work_dir),
        "--train_file", str(train_file),
        "--valid_file", str(valid_file),
        "--foundation_model", str(foundation),
        # the isolated-atom energies are the base model's: a fine-tune must not move the
        # energy zero, or the Labels' absolute energies stop meaning what they meant
        "--E0s", "foundation",
        "--energy_key", ENERGY_KEY, "--forces_key", FORCES_KEY, "--hessian_key", HESSIAN_KEY,
        "--loss", "external", "--loss_module", LOSS_MODULE,
        "--energy_weight", repr(float(energy_weight)),
        "--forces_weight", repr(float(forces_weight)),
        "--hessian_weight", repr(float(hessian_weight)),
        "--hessian_probe", str(probe),
        "--n_hessian_probes", str(int(n_probes)),
        "--hessian_mode_weighting", str(mode_weighting),
        "--max_num_epochs", str(int(max_epochs)),
        "--batch_size", str(int(batch_size)),
        "--valid_batch_size", str(int(valid_batch_size or batch_size)),
        "--seed", str(int(seed)),
        "--device", str(device),
        "--default_dtype", "float64",
        "--error_table", "PerAtomRMSE",
        "--save_cpu",
    ]
    if lr is not None:
        argv += ["--lr", repr(float(lr))]
    if multiheads:
        argv += ["--multiheads_finetuning", "True", "--num_samples_pt", str(int(num_samples_pt))]
        if pt_train_file:
            argv += ["--pt_train_file", str(pt_train_file)]
    else:
        argv += ["--multiheads_finetuning", "False"]
    argv += [str(a) for a in extra]
    return argv


def argv_pairs(argv):
    """mace's command line as a mapping. A bare flag (`--save_cpu`) maps to True, so the
    config file and any reader of it see the same settings the parser did."""
    out, i = {}, 0
    argv = [str(a) for a in argv]
    while i < len(argv):
        key = argv[i].lstrip("-")
        if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            out[key] = argv[i + 1]
            i += 2
        else:
            out[key] = True
            i += 1
    return out


_EPOCH = re.compile(
    r"Epoch (?P<epoch>\d+):.*?(?P<valid>head: \w+, )?loss=(?P<loss>[-\d.eE+]+)"
    r"(?:.*?RMSE_E_per_atom=\s*(?P<rmse_e>[-\d.eE+]+) meV)?"
    r"(?:.*?RMSE_F=\s*(?P<rmse_f>[-\d.eE+]+) meV / A)?")


def parse_epochs(log_text):
    """The epoch table out of mace's log. mace prints one line per validation; the
    training loss of the same epoch is in `results/<run>_*.txt` as JSON lines, which is
    where the per-term numbers live too."""
    rows = []
    for m in _EPOCH.finditer(log_text):
        rows.append(dict(epoch=int(m.group("epoch")), split="valid", loss=float(m.group("loss")),
                         rmse_e_per_atom_meV=float(m.group("rmse_e")) if m.group("rmse_e") else None,
                         rmse_f_meV_A=float(m.group("rmse_f")) if m.group("rmse_f") else None))
    return rows


def parse_results(results_dir):
    """mace's `results/*.txt`: one JSON object per evaluation (`mode`, `epoch`, `loss`,
    the error columns). Both splits, in the order they were written."""
    rows = []
    for path in sorted(Path(results_dir).glob("*.txt")):
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("epoch") is None:
                continue
            rows.append(dict(epoch=int(r["epoch"]),
                             split="train" if str(r.get("mode", "")).startswith("opt") else "valid",
                             loss=float(r["loss"]) if r.get("loss") is not None else None,
                             rmse_e_per_atom_meV=(float(r["rmse_e_per_atom"]) * 1000.0
                                                 if r.get("rmse_e_per_atom") is not None else None),
                             rmse_f_meV_A=(float(r["rmse_f"]) * 1000.0 if r.get("rmse_f") is not None else None)))
    return rows


def run_training(dataset_dir, tag, name, level, run, *, foundation=None, dry_run=False,
                 strict_fork=True, **settings):
    """One fine-tune, end to end: the run directory, the split files, mace's arguments,
    `mace.cli.run_train.run`, the Record. Returns the info dict (`--dry-run`: the argv
    and the info without training)."""
    dataset_dir = Path(dataset_dir)
    run_dir = dataset_dir / STEP / run
    run_dir.mkdir(parents=True, exist_ok=True)
    fork = check_fork(strict=strict_fork)

    foundation_name = foundation or engine.engine_name()
    foundation_path = engine.model_path(foundation_name)
    fp_base = engine.parameter_fingerprint(path=foundation_path)

    files, counts = split_files(dataset_dir, name, level, run_dir)
    argv = mace_argv(files["train"], files["valid"], run, run_dir, foundation_path, level, **settings)
    config_file = run_dir / "config.yaml"
    config_file.write_text("\n".join("{}: {}".format(k, v) for k, v in argv_pairs(argv).items()) + "\n",
                           encoding="utf-8")

    info = dict(
        RUN=str(run), TAG=str(tag), NAME=str(name), LEVEL=str(level),
        DATASET_DIR=str(dataset_dir), INDEX_FILE=str(dataset_dir / "index.dat"),
        TRAIN_FILE=str(files["train"]), VALID_FILE=str(files["valid"]),
        N_TRAIN=counts["train"][0], N_TRAIN_HESSIAN=counts["train"][1],
        N_VALID=counts["valid"][0], N_VALID_HESSIAN=counts["valid"][1],
        FOUNDATION_MODEL=foundation_name, FOUNDATION_PARAMS_SHA256=fp_base["params_sha256"],
        ENGINE_PARAMS_SHA256=fp_base["params_sha256"],
        CONFIG_FILE=str(config_file), CONFIG_SHA256=sha256_file(config_file),
        LOSS=LOSS_MODULE,
        ENERGY_WEIGHT=float(settings.get("energy_weight", 1.0)),
        FORCES_WEIGHT=float(settings.get("forces_weight", 100.0)),
        HESSIAN_WEIGHT=float(settings.get("hessian_weight", 1.0)),
        PROBE=str(settings.get("probe", "rademacher")),
        N_PROBES=int(settings.get("n_probes", 4)),
        MODE_WEIGHTING=str(settings.get("mode_weighting", "entropy")),
        MAX_NUM_EPOCHS=int(settings.get("max_epochs", 100)),
        BATCH_SIZE=int(settings.get("batch_size", 4)),
        SEED=int(settings.get("seed", 123)),
        DEVICE=str(settings.get("device", "cpu")), DTYPE="float64",
        MULTIHEADS=bool(settings.get("multiheads", False)),
        PT_TRAIN_FILE=str(settings.get("pt_train_file") or "-"),
        NUM_SAMPLES_PT=int(settings.get("num_samples_pt", 5000)),
        MACE_VERSION=fork["mace_version"], MACE_FORK=engine.MACE_FORK,
        MACE_FORK_COMMIT=fork["mace_fork_commit"],
    )
    if dry_run:
        return dict(info=info, argv=argv, run_dir=run_dir, epochs=[], dry_run=True)

    t0 = time.time()
    _run_mace(argv, run_dir)
    seconds = time.time() - t0

    model_file = run_dir / "{}.model".format(run)
    if not model_file.is_file():
        stage_two = run_dir / "{}_stagetwo.model".format(run)
        model_file = stage_two if stage_two.is_file() else model_file
    epochs = parse_results(run_dir / "results")
    if not epochs:
        epochs = parse_epochs("\n".join(p.read_text(errors="replace") for p in (run_dir / "logs").glob("*.log")))
    info["N_EPOCHS"] = len({r["epoch"] for r in epochs})
    info["SECONDS"] = float(seconds)
    info["SECONDS_PER_EPOCH"] = float(seconds / max(1, info["N_EPOCHS"]))
    info["MODEL_FILE"] = str(model_file) if model_file.is_file() else "-"
    if model_file.is_file():
        fp = engine.parameter_fingerprint(path=model_file)
        info["MODEL_PARAMS_SHA256"] = fp["params_sha256"]
        info["MODEL_N_TENSORS"] = fp["n_tensors"]
    write_record(run_dir, info, epochs)
    return dict(info=info, argv=argv, run_dir=run_dir, epochs=epochs, dry_run=False)


def _run_mace(argv, run_dir):
    """`mace.cli.run_train.run(args)` with our argv, from the run directory. mace parses
    `sys.argv`, so it is swapped for the call and restored after it."""
    from mace.cli.run_train import run as mace_run
    from mace.tools import build_default_arg_parser
    args = build_default_arg_parser().parse_args(argv)
    cwd = os.getcwd()
    old_argv = sys.argv
    try:
        os.chdir(run_dir)
        sys.argv = ["mace_run_train"] + list(argv)
        mace_run(args)
    finally:
        sys.argv = old_argv
        os.chdir(cwd)


def write_record(run_dir, info, epochs):
    rows = [dict(EPOCH=r["epoch"], SPLIT=r["split"], LOSS=r["loss"],
                 RMSE_E_PER_ATOM_MEV=r["rmse_e_per_atom_meV"], RMSE_F_MEV_A=r["rmse_f_meV_A"])
            for r in epochs]
    missing = prop.write(Path(run_dir) / (STEP + ".toml"), {"Calculation_Info": info, "Epoch": rows},
                         SCHEMA, prop.NORMAL_TERMINATION, PROGNAME)
    if missing:
        raise RuntimeError("train.toml keys outside the schema: {}".format(missing))
    dat.write_table(Path(run_dir) / (STEP + ".dat"), epochs, list(EPOCH_ROW), EPOCH_ROW)
    _write_report(Path(run_dir) / (STEP + ".out"), info, epochs)


def _write_report(path, info, epochs):
    rep = report.Report(PROGNAME, "Fine-tune {!r} of {} on {}".format(
        info["RUN"], info["FOUNDATION_MODEL"], info["NAME"]))
    rep.section("the Dataset")
    for k in ("TAG", "NAME", "LEVEL", "DATASET_DIR", "TRAIN_FILE", "N_TRAIN", "N_TRAIN_HESSIAN",
              "VALID_FILE", "N_VALID", "N_VALID_HESSIAN"):
        rep.kv(k, info.get(k))
    rep.section("the loss (design-phl-loss.md eq. 11; T03)")
    for k in ("LOSS", "ENERGY_WEIGHT", "FORCES_WEIGHT", "HESSIAN_WEIGHT", "PROBE", "N_PROBES",
              "MODE_WEIGHTING", "MULTIHEADS", "PT_TRAIN_FILE", "NUM_SAMPLES_PT"):
        rep.kv(k, info.get(k))
    rep.section("identity")
    for k in ("FOUNDATION_MODEL", "FOUNDATION_PARAMS_SHA256", "MODEL_FILE", "MODEL_PARAMS_SHA256",
              "MODEL_N_TENSORS", "CONFIG_FILE", "CONFIG_SHA256", "MACE_VERSION", "MACE_FORK",
              "MACE_FORK_COMMIT", "SEED", "DEVICE", "DTYPE"):
        rep.kv(k, info.get(k))
    rep.section("cost")
    for k in ("MAX_NUM_EPOCHS", "N_EPOCHS", "BATCH_SIZE", "SECONDS", "SECONDS_PER_EPOCH"):
        rep.kv(k, info.get(k))
    if epochs:
        rep.section("epochs")
        rep.table(["epoch", "split", "loss", "RMSE E/atom meV", "RMSE F meV/A"],
                  [[r["epoch"], r["split"],
                    "-" if r["loss"] is None else "{:.6f}".format(r["loss"]),
                    "-" if r["rmse_e_per_atom_meV"] is None else "{:.2f}".format(r["rmse_e_per_atom_meV"]),
                    "-" if r["rmse_f_meV_A"] is None else "{:.2f}".format(r["rmse_f_meV_A"])]
                   for r in epochs])
    rep.write(path)


def registry_entry(info):
    """The `ENGINES` lines for the fine-tuned potential (ticket 01's recipe): the
    Dataset's index and the config SHA are its `source`, so the numbers can be traced."""
    name = "{}-{}".format(info["FOUNDATION_MODEL"], info["RUN"])
    return dict(name=name,
                filename="{}.model".format(name),
                source="{} + config {}".format(info["INDEX_FILE"], info["CONFIG_SHA256"][:16]),
                note="Fine-tuned on {} ({} frames, {} with Hessians) with the projected Hessian loss "
                     "(w_H {}, probe {} k={}, {} weighting); mace fork {}.".format(
                         info["NAME"], info["N_TRAIN"], info["N_TRAIN_HESSIAN"], info["HESSIAN_WEIGHT"],
                         info["PROBE"], info["N_PROBES"], info["MODE_WEIGHTING"],
                         info["MACE_FORK_COMMIT"][:12]),
                params_sha256=info.get("MODEL_PARAMS_SHA256"))
