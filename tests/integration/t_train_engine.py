"""Ticket 13 of the Hessian-learning set: `openqha.training.run` fine-tunes the real
MACE-OFF23_medium through the mace fork, with the projected Hessian loss.

INTEGRATION (engine, CPU, minutes). Builds a two-molecule Dataset directory from the
2-methyloxirane frame fixture (basin frame with a reference Hessian, displaced frames
E-F only, as round-5 Q7 (b) labels them), then:

  * `--probe modes` for 2 epochs: the run directory, the model file, the Record, and
    the epoch table; the training loss at epoch 0 is finite and the fine-tuned model
    loads in `MACECalculator` with a different parameter fingerprint from the base;
  * the exact validation term: mace's `evaluate` gave the loss a FULL Hessian (the
    fork's `wants_hessian_at_eval` line), checked by reading the loss back;
  * `--loss weighted` on the same files runs unchanged (the default path).

SKIPs without the model or without the fork.
"""
import shutil
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
FIX = ROOT / "tests" / "data" / "methyloxirane_frames"
LEVEL = "wb97m-d3bj_def2-tzvppd"
FAIL = []


def check(label, ok, detail=""):
    print("  {:78s} {}".format(label, "ok" if ok else "FAIL " + str(detail)[:300]))
    if not ok:
        FAIL.append(label)


def build_dataset(tmp, name, level):
    """A Dataset directory with the merged MACE-form file: train = basin + 2 displaced,
    valid = a second copy of the basin + 2 displaced (the machinery, not generalisation)."""
    from openqha.data import dataset, frames
    basin = frames.read_frames(FIX / "basin.{}.extxyz".format(level))[0]
    disp = frames.read_frames(FIX / "displaced.{}.extxyz".format(level))
    for a in disp:
        a.info.pop("hessian", None)
        a.info["has_hessian"] = False
    basin_again = frames.read_frames(FIX / "basin.{}.extxyz".format(level))[0]   # re-read: copy() drops the calculator
    rows = [(basin, "train")] + [(a, "train") for a in disp[:2]] + \
           [(basin_again, "valid")] + [(a, "valid") for a in disp[2:4]]
    d = Path(tmp) / name
    d.mkdir(parents=True, exist_ok=True)
    dataset._write_split(dataset.merged_file(d, name, level), rows, reference=True)
    return d


def main():
    from openqha.potentials import engine
    from openqha.training import run as train_run
    try:
        import mace                                              # noqa: F401
        engine.model_path()
    except Exception as exc:                                     # noqa: BLE001
        print("SKIP: {}".format(exc))
        return 0
    fork = engine.mace_fork_info()
    if fork["mace_fork_commit"] == "unknown":
        print("SKIP: the installed mace is not the fork ({})".format(engine.MACE_FORK))
        return 0
    print("  mace {}  fork {}{}".format(__import__("mace").__version__, fork["mace_fork_commit"][:12],
                                        "  DIRTY" if fork["mace_fork_dirty"] else ""))

    with tempfile.TemporaryDirectory() as tmp:
        d = build_dataset(tmp, "smoke_fit", LEVEL)
        out = train_run.run_training(
            d, "test", "smoke_fit", LEVEL, "modes2", dry_run=True, strict_fork=False,
            probe="modes", mode_weighting="none", hessian_weight=1e-3, max_epochs=2, batch_size=2)
        argv = out["argv"]
        check("dry run: the command names the external loss, the keys and float64",
              "--loss" in argv and argv[argv.index("--loss") + 1] == "external"
              and argv[argv.index("--loss_module") + 1] == train_run.LOSS_MODULE
              and argv[argv.index("--hessian_key") + 1] == "REF_hessian"
              and argv[argv.index("--default_dtype") + 1] == "float64", argv)
        check("dry run: the Record counts the frames and their Hessians (3 train / 3 valid, 1 + 1 labelled)",
              (out["info"]["N_TRAIN"], out["info"]["N_TRAIN_HESSIAN"]) == (3, 1)
              and (out["info"]["N_VALID"], out["info"]["N_VALID_HESSIAN"]) == (3, 1), out["info"])
        check("dry run writes no model", not (out["run_dir"] / "modes2.model").is_file())

        out = train_run.run_training(
            d, "test", "smoke_fit", LEVEL, "modes2", strict_fork=False,
            probe="modes", mode_weighting="none", hessian_weight=1e-3, max_epochs=2, batch_size=2,
            seed=7, device="cpu")
        info, run_dir = out["info"], out["run_dir"]
        check("the run produced a model file", Path(info["MODEL_FILE"]).is_file(), info["MODEL_FILE"])
        check("the Record is on disk (train.out / .toml / .dat)",
              all((run_dir / ("train" + e)).is_file() for e in (".out", ".toml", ".dat")))
        check("the epoch table has both splits", {r["split"] for r in out["epochs"]} >= {"train", "valid"},
              sorted({r["split"] for r in out["epochs"]}))
        check("every logged loss is finite", all(r["loss"] is None or r["loss"] == r["loss"] for r in out["epochs"]))
        check("the Record carries the identities (base, model, config SHA, fork commit)",
              len(info["FOUNDATION_PARAMS_SHA256"]) == 64 and len(info.get("MODEL_PARAMS_SHA256", "")) == 64
              and len(info["CONFIG_SHA256"]) == 64 and info["MACE_FORK_COMMIT"] != "unknown"
              and info["SECONDS_PER_EPOCH"] > 0, info)
        check("the fine-tuned potential is NOT the base model (the fingerprint moved)",
              info["MODEL_PARAMS_SHA256"] != info["FOUNDATION_PARAMS_SHA256"])
        print("  {} epochs in {:.1f} s ({:.1f} s per epoch, {} train frames)".format(
            info["N_EPOCHS"], info["SECONDS"], info["SECONDS_PER_EPOCH"], info["N_TRAIN"]))
        for r in out["epochs"][-4:]:
            print("    epoch {:3d} {:6s} loss {}".format(
                r["epoch"], r["split"], "-" if r["loss"] is None else "{:.6f}".format(r["loss"])))

        # the fine-tuned model loads and answers
        from ase.io import read
        from mace.calculators import MACECalculator
        atoms = read(str(info["TRAIN_FILE"]), index="0", format="extxyz")
        calc = MACECalculator(model_paths=info["MODEL_FILE"], device="cpu", default_dtype="float64")
        atoms.calc = calc
        e = float(atoms.get_potential_energy())
        h = calc.get_hessian(atoms)
        check("the fine-tuned model loads in MACECalculator and gives E and a (3N, N, 3) Hessian",
              e == e and h.shape == (3 * len(atoms), len(atoms), 3), (e, h.shape))

        # the validation term is the EXACT one: mace's evaluate gave the loss a full
        # Hessian (the fork's `wants_hessian_at_eval` line), not the estimator
        from mace import data as mdata, tools as mtools
        from mace.tools import torch_geometric
        from mace.tools.train import evaluate
        from openqha.training import phl_loss
        import torch
        torch.set_default_dtype(torch.float64)
        loss_fn = phl_loss.WeightedEnergyForcesHessianLoss(probe="modes", mode_weighting="none")
        ks = mdata.KeySpecification.from_defaults()
        frames = read(str(info["VALID_FILE"]), index=":", format="extxyz")
        table = mtools.AtomicNumberTable(sorted({int(z) for a in frames for z in a.numbers}))
        ads = [mdata.AtomicData.from_config(mdata.config_from_atoms(a, key_specification=ks),
                                            z_table=table, cutoff=float(calc.r_max)) for a in frames]
        loader = torch_geometric.dataloader.DataLoader(dataset=ads, batch_size=3, shuffle=False)
        evaluate(model=calc.models[0], loss_fn=loss_fn, data_loader=loader,
                 output_args={"energy": True, "forces": True, "virials": False, "stress": False,
                              "hessian": loss_fn.wants_hessian_at_eval},
                 device=torch.device("cpu"))
        check("mace's evaluate gave the loss the FULL Hessian: the exact term, not the estimator",
              loss_fn.last_terms["hessian_exact"] is not None and loss_fn.last_terms["n_labelled"] == 1,
              loss_fn.last_terms)
        print("    exact validation Hessian term on the fine-tuned model: {:.6e}".format(
            loss_fn.last_terms["hessian_exact"]))

        entry = train_run.registry_entry(info)
        check("the ENGINES entry names the Dataset index and the config SHA",
              entry["filename"].endswith(".model") and "config" in entry["source"]
              and entry["params_sha256"] == info["MODEL_PARAMS_SHA256"], entry)

        # --- multihead replay (round-2 Q7 a): the pretraining head's frames carry no
        # Hessian, so the Hessian term sees only the fine-tuning head's
        pt = Path(tmp) / "pt.xyz"
        pt_frames = read(str(ROOT / "tests" / "data" / "spice_tiny" / "train_large_neut_no_bad_clean.xyz"),
                         index=":", format="extxyz")
        for a in pt_frames:
            a.info["REF_energy"] = float(a.info.get("energy", -1.0))
            a.arrays["REF_forces"] = a.arrays.get("forces", a.get_forces())   # ASE may park them on the calculator
            a.info.pop("REF_hessian", None)
        from ase.io import write as ase_write
        ase_write(str(pt), pt_frames, format="extxyz")
        mh = train_run.run_training(
            d, "test", "smoke_fit", LEVEL, "mh1", strict_fork=False, probe="modes",
            mode_weighting="none", hessian_weight=1e-3, max_epochs=1, batch_size=2, seed=7,
            multiheads=True, pt_train_file=str(pt), num_samples_pt=len(pt_frames))
        check("a multihead run completes and the Record says so",
              mh["info"]["MULTIHEADS"] is True and mh["info"]["PT_TRAIN_FILE"] == str(pt)
              and Path(mh["info"]["MODEL_FILE"]).is_file(), mh["info"].get("MODEL_FILE"))
        check("... the replay frames carry no Hessian, so only the fine-tuning head's do "
              "({} pt frames, {} labelled train frames)".format(len(pt_frames), mh["info"]["N_TRAIN_HESSIAN"]),
              all("REF_hessian" not in a.info for a in read(str(pt), index=":", format="extxyz"))
              and mh["info"]["N_TRAIN_HESSIAN"] == 1)

        # the default path still works on the same files
        from mace.cli.run_train import run as mace_run
        from mace.tools import build_default_arg_parser
        plain = Path(tmp) / "plain"
        plain.mkdir()
        argv = train_run.mace_argv(info["TRAIN_FILE"], info["VALID_FILE"], "plain", plain,
                                   engine.model_path(), LEVEL, max_epochs=1, batch_size=2, seed=7)
        i = argv.index("--loss")
        argv = argv[:i] + ["--loss", "weighted"] + argv[i + 4:]        # drop --loss external --loss_module
        args = build_default_arg_parser().parse_args(argv)
        import os
        cwd = os.getcwd()
        try:
            os.chdir(plain)
            mace_run(args)
        finally:
            os.chdir(cwd)
        check("the stock loss trains on the same files (the default path is untouched)",
              (plain / "plain.model").is_file() or list(plain.glob("*.model")))

    print("\n{} checks, {} failed".format(14, len(FAIL)))
    print("PASS" if not FAIL else "FAIL")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
