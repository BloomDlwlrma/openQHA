"""Ticket 13 of the Hessian-learning set: what `openqha.training.run` builds for mace,
what it refuses, and what its Record says -- no engine, no training.

Asserted: `mace_argv` names the external loss and the Dataset's keys, float64 and the
foundation's E0s, and carries the probe settings; `--multiheads` adds the replay flags
and nothing else does; `split_files` splits the merged Dataset file by its `split` key
and counts the Hessians; `check_fork` refuses a non-fork and a dirty checkout with a
message naming the fix; the Record's schema covers every key `run_training` writes and
`parse_results` / `parse_epochs` read mace's two output forms; `registry_entry` names
the index and the config SHA.
"""
import sys
import tempfile
from pathlib import Path


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))
from openqha.store import property as prop                   # noqa: E402
from openqha.training import run as train_run                # noqa: E402

FAIL = []


def check(label, ok, detail=""):
    print("  {:78s} {}".format(label, "ok" if ok else "FAIL " + str(detail)[:200]))
    if not ok:
        FAIL.append(label)


def pairs(argv):
    """the settings as mace's parser sees them, bare flags included (`--save_cpu`)"""
    return {"--" + k: v for k, v in train_run.argv_pairs(argv).items()}


def main():
    import numpy as np
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator
    from openqha.data import dataset

    # --- the command line -----------------------------------------------------------------
    argv = train_run.mace_argv("tr.xyz", "va.xyz", "r1", "/tmp/run", "/w/base.model", "lvl",
                               hessian_weight=0.25, probe="modes", n_probes=7,
                               mode_weighting="none", max_epochs=3, batch_size=2, seed=5, device="cuda")
    p = pairs(argv)
    check("the loss is ours, by name, through the fork's hook",
          p["--loss"] == "external" and p["--loss_module"] == "openqha.training.phl_loss:build"
          and p["--loss_module"].startswith("openqha.training"), p.get("--loss_module"))
    check("the keys are the Dataset's and the dtype is float64",
          (p["--energy_key"], p["--forces_key"], p["--hessian_key"]) == ("REF_energy", "REF_forces", "REF_hessian")
          and p["--default_dtype"] == "float64", p)
    check("the isolated-atom energies come from the foundation (the energy zero does not move)",
          p["--E0s"] == "foundation")
    check("the probe settings reach mace",
          (p["--hessian_weight"], p["--hessian_probe"], p["--n_hessian_probes"], p["--hessian_mode_weighting"])
          == ("0.25", "modes", "7", "none"), p)
    check("the loop settings reach mace",
          (p["--max_num_epochs"], p["--batch_size"], p["--seed"], p["--device"]) == ("3", "2", "5", "cuda"))
    check("without --multiheads the replay is off and no pt file is named",
          p["--multiheads_finetuning"] == "False" and "--pt_train_file" not in p)
    p2 = pairs(train_run.mace_argv("tr.xyz", "va.xyz", "r", "/tmp", "/w/b.model", "l",
                                   multiheads=True, pt_train_file="spice.xyz", num_samples_pt=99))
    check("--multiheads adds the replay head, its file and its sample count",
          p2["--multiheads_finetuning"] == "True" and p2["--pt_train_file"] == "spice.xyz"
          and p2["--num_samples_pt"] == "99")
    extra = train_run.mace_argv("t", "v", "r", "/tmp", "/b", "l", extra=["--swa", "--lr", "0.001"])
    check("--mace-arg passes through as given", extra[-3:] == ["--swa", "--lr", "0.001"])
    check("a bare flag maps to True, not to the next flag's name",
          train_run.argv_pairs(["--save_cpu", "--seed", "3"]) == {"save_cpu": True, "seed": "3"})

    # --- splitting the Dataset's merged file -------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        rng = np.random.default_rng(0)
        rows = []
        for i, split in enumerate(["train", "train", "train", "valid", "test"]):
            at = Atoms("H2O", positions=rng.standard_normal((3, 3)))
            at.calc = SinglePointCalculator(at, energy=float(i), forces=np.zeros((3, 3)))
            at.info = dict(qm9_index="x", generator="basin", basin=0, k=i)
            if i % 2 == 0:                                    # frames 0, 2, 4 carry a Hessian
                h = np.eye(9)
                at.info["hessian"] = h
            rows.append((at, split))
        dataset._write_split(dataset.merged_file(td, "ds", "lvl"), rows, reference=True)
        files, counts = train_run.split_files(td, "ds", "lvl", td)
        check("split_files writes one file per split from the `split` key",
              files["train"].is_file() and files["valid"].is_file()
              and files["train"].name == "train.lvl.extxyz", files)
        check("... and counts the frames and the Hessians (train 3 / 2, valid 1 / 0)",
              counts == {"train": (3, 2), "valid": (1, 0)}, counts)
        try:
            train_run.split_files(td, "no_such", "lvl", td)
            check("a missing Dataset file is refused by name", False)
        except FileNotFoundError as exc:
            check("a missing Dataset file is refused by name", "04_dataset.py" in str(exc))

    # --- the fork guard -------------------------------------------------------------------------
    from openqha.potentials import engine
    real = engine.mace_fork_info
    try:
        engine.mace_fork_info = lambda: dict(mace_fork_commit="unknown", mace_fork_dirty=None, mace_fork_path=None)
        try:
            train_run.check_fork(strict=True)
            check("a non-fork mace is refused, naming the install line", False)
        except RuntimeError as exc:
            check("a non-fork mace is refused, naming the install line",
                  "pip install -e" in str(exc) and engine.MACE_FORK in str(exc), str(exc))
        check("... unless strict is off", train_run.check_fork(strict=False)["mace_fork_commit"] == "unknown")
        engine.mace_fork_info = lambda: dict(mace_fork_commit="a" * 40, mace_fork_dirty=True, mace_fork_path="/w/fork")
        try:
            train_run.check_fork(strict=True)
            check("a dirty checkout is refused", False)
        except RuntimeError as exc:
            check("a dirty checkout is refused", "uncommitted changes" in str(exc), str(exc))
    finally:
        engine.mace_fork_info = real

    # --- mace's two output forms ----------------------------------------------------------------
    results = ('{"mode": "opt", "epoch": 0, "loss": 1.5, "rmse_e_per_atom": 0.02, "rmse_f": 0.5}\n'
               '{"mode": "eval", "epoch": 0, "loss": 1.2, "rmse_e_per_atom": 0.01, "rmse_f": 0.4}\n'
               'not json\n')
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "run_train.txt").write_text(results)
        rows = train_run.parse_results(td)
    check("parse_results reads both splits and converts eV to meV",
          [r["split"] for r in rows] == ["train", "valid"] and rows[0]["rmse_f_meV_A"] == 500.0
          and rows[1]["rmse_e_per_atom_meV"] == 10.0 and len(rows) == 2, rows)
    log = ("2026-09-20 21:09:03.573 INFO: Epoch 1: head: Default, loss=0.05997006, "
           "RMSE_E_per_atom=   20.92 meV, RMSE_F=   77.16 meV / A\n")
    rows = train_run.parse_epochs(log)
    check("parse_epochs reads mace's log line as a fallback",
          len(rows) == 1 and rows[0]["epoch"] == 1 and abs(rows[0]["loss"] - 0.05997006) < 1e-12
          and rows[0]["rmse_f_meV_A"] == 77.16, rows)

    # --- the Record's schema covers what run_training writes ---------------------------------------
    written = {"RUN", "TAG", "NAME", "LEVEL", "DATASET_DIR", "INDEX_FILE", "TRAIN_FILE", "VALID_FILE",
               "N_TRAIN", "N_TRAIN_HESSIAN", "N_VALID", "N_VALID_HESSIAN", "FOUNDATION_MODEL",
               "FOUNDATION_PARAMS_SHA256", "ENGINE_PARAMS_SHA256", "CONFIG_FILE", "CONFIG_SHA256",
               "LOSS", "ENERGY_WEIGHT", "FORCES_WEIGHT", "HESSIAN_WEIGHT", "PROBE", "N_PROBES",
               "MODE_WEIGHTING", "MAX_NUM_EPOCHS", "BATCH_SIZE", "SEED", "DEVICE", "DTYPE",
               "MULTIHEADS", "PT_TRAIN_FILE", "NUM_SAMPLES_PT", "MACE_VERSION", "MACE_FORK",
               "MACE_FORK_COMMIT", "N_EPOCHS", "SECONDS", "SECONDS_PER_EPOCH", "MODEL_FILE",
               "MODEL_PARAMS_SHA256", "MODEL_N_TENSORS"}
    schema = set(train_run.SCHEMA["Calculation_Info"])
    check("every key run_training writes is in the schema", written <= schema, sorted(written - schema))

    info = dict(FOUNDATION_MODEL="MACE-OFF23_medium", RUN="w1", INDEX_FILE="/r/index.dat",
                CONFIG_SHA256="0123456789abcdef" * 4, NAME="draw300", N_TRAIN=90, N_TRAIN_HESSIAN=30,
                HESSIAN_WEIGHT=0.01, PROBE="rademacher", N_PROBES=4, MODE_WEIGHTING="entropy",
                MACE_FORK_COMMIT="b" * 40, MODEL_PARAMS_SHA256="c" * 64)
    e = train_run.registry_entry(info)
    check("registry_entry: name, filename, the index + config SHA as source, the fingerprint",
          e["name"] == "MACE-OFF23_medium-w1" and e["filename"] == "MACE-OFF23_medium-w1.model"
          and "/r/index.dat" in e["source"] and "0123456789abcdef" in e["source"]
          and e["params_sha256"] == "c" * 64 and "w_H 0.01" in e["note"], e)

    # --- the Record writes and reads back ------------------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        full = dict(info)
        for k in schema - set(full) - {"PROGNAME", "VERSION", "STATUS"}:
            full[k] = 0 if "N_" in k or "SECONDS" in k or "WEIGHT" in k else "x"
        full["MULTIHEADS"] = False
        train_run.write_record(td, full, [dict(epoch=0, split="train", loss=1.0,
                                               rmse_e_per_atom_meV=2.0, rmse_f_meV_A=3.0)])
        rec = prop.load(Path(td) / "train.toml")
        check("train.toml round-trips with NORMAL TERMINATION and the epoch block",
              rec["Calculation_Status"]["STATUS"] == prop.NORMAL_TERMINATION
              and rec["Calculation_Info"]["RUN"] == "w1" and len(rec["Epoch"]) == 1, rec.get("Calculation_Status"))
        check("train.out and train.dat are written",
              (Path(td) / "train.out").is_file() and (Path(td) / "train.dat").is_file())

    print("\n{} checks, {} failed".format(21, len(FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
