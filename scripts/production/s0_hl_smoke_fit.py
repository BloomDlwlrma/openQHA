"""The smoke fit: the numbers the campaign's settings are read off (ticket 15).

PRODUCTION. Every number it produces is interpolation within the fit molecules (the
pinned rule is off, `PURPOSE = fit`); it measures cost and weights, it does not judge
generalisation.

    # the fit Dataset from an existing one's labelled frames, then the balance
    python scripts/production/s0_hl_smoke_fit.py --tag rings --name smoke --stage balance

    # what a probe setting costs per epoch (1 epoch each)
    python scripts/production/s0_hl_smoke_fit.py --tag rings --name smoke --stage cost

    # the w_H scan and the judge on every run
    python scripts/production/s0_hl_smoke_fit.py --tag rings --name smoke --stage scan \
        --epochs 100 --scan 0.3 1 3

    python scripts/production/s0_hl_smoke_fit.py --tag rings --name smoke --stage all

Writes `<root>/<tag>/_datasets/<fit name>/` (the fit Dataset, its train runs and judge
runs) and `smoke_fit.{out,toml}` beside them: the balance, the cost table, the scan, the
replay arithmetic and the judge's verdict per run.
"""
import argparse
import sys
import time
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
from openqha.potentials import engine                        # noqa: E402
from openqha.store import property as prop, report           # noqa: E402
from openqha.training import judge, run as train_run, smoke_fit   # noqa: E402

PROGNAME = "openQHA hl_smoke_fit"
STEP = "smoke_fit"

SCHEMA = {
    "Calculation_Info": {
        "PROGNAME": ("String", None, "the step that wrote this file"),
        "VERSION": ("String", None, "openQHA version"),
        "STATUS": ("String", None, "the completion marker"),
        "TAG": ("String", None, "the campaign tag"),
        "SOURCE_NAME": ("String", None, "the Dataset the labelled frames came from"),
        "NAME": ("String", None, "the fit Dataset"),
        "PURPOSE": ("String", None, "fit: every number here is interpolation within these molecules"),
        "LEVEL": ("String", None, "the reference level of the Labels"),
        "STAGES": ("ArrayOfStrings", None, "which stages ran"),
        "N_FRAMES": ("Integer", None, "frames in the fit Dataset"),
        "N_HESSIAN_FRAMES": ("Integer", None, "of those, carrying a reference Hessian"),
        "N_TRAIN": ("Integer", None, "frames in train"),
        "N_VALID": ("Integer", None, "frames in valid"),
        "N_MOLECULES": ("Integer", None, "molecules"),
        "ENGINE": ("String", None, "the base potential"),
        "ENGINE_PARAMS_SHA256": ("String", None, "its parameter fingerprint"),
        "DEVICE": ("String", None, "cpu / cuda"),
        "EPOCHS": ("Integer", None, "epochs per scan run"),
        "SECONDS": ("Double", "s", "wall time of the whole fit"),
    },
    "Balance": {
        "N_FRAMES": ("Integer", None, "training frames read"),
        "N_HESSIAN_FRAMES": ("Integer", None, "of those, with a Label"),
        "L_E": ("Double", None, "energy term per atom squared, base model, epoch 0"),
        "L_F": ("Double", None, "force term, base model, epoch 0"),
        "L_H": ("Double", None, "eq. 1 exactly, base model, epoch 0"),
        "WE_LE": ("Double", None, "w_E L_E"),
        "WF_LF": ("Double", None, "w_F L_F"),
        "HESSIAN_WEIGHT_BALANCED": ("Double", None, "w_H at which w_H L_H = w_F L_F"),
    },
    "Cost": {
        "SETTING": ("String", None, "the probe setting"),
        "PROBE": ("String", None, "rademacher / cartesian / - "),
        "N_PROBES": ("Integer", None, "k"),
        "SECONDS": ("Double", "s", "wall time of the measurement run"),
        "SECONDS_PER_EPOCH": ("Double", "s", "measured"),
        "RATIO_TO_EF": ("Double", None, "against the energy-forces loss on the same frames"),
        "RUN": ("String", None, "the train run directory"),
        "N_TRAIN": ("Integer", None, "training frames"),
        "N_TRAIN_HESSIAN": ("Integer", None, "of those, with a Label"),
    },
    "Scan": {
        "RUN": ("String", None, "the train run"),
        "PROBE": ("String", None, "the probe setting"),
        "N_PROBES": ("Integer", None, "k"),
        "HESSIAN_WEIGHT": ("Double", None, "w_H"),
        "W_H_OVER_BALANCED": ("Double", None, "w_H / the epoch-0 balance"),
        "EPOCHS": ("Integer", None, "epochs run"),
        "SECONDS_PER_EPOCH": ("Double", "s", "measured"),
        "FINAL_TRAIN_LOSS": ("Double", None, "the last training loss in the log"),
        "FINAL_VALID_LOSS": ("Double", None, "the last validation loss (its Hessian term is EXACT)"),
        "JUDGE_LOW_MODE_MAE_CM": ("Double", "cm^-1", "the judge's held-out low-mode MAE after the run"),
        "BASE_LOW_MODE_MAE_CM": ("Double", "cm^-1", "the base model's, same frames"),
        "VERDICT": ("String", None, "the judge's verdict on that run"),
    },
    "Replay": {
        "NUM_SAMPLES_PT": ("Integer", None, "the replay flag"),
        "N_TRAIN": ("Integer", None, "fine-tuning frames per epoch"),
        "N_HESSIAN": ("Integer", None, "of those, with a Label"),
        "REPLAY_PER_TRAIN": ("Double", None, "replay frames per fine-tuning frame"),
        "REPLAY_PER_HESSIAN_FRAME": ("Double", None, "replay frames per Hessian-labelled frame -- the number to compare with PFT's 4"),
        "PFT_REFERENCE": ("Double", None, "PFT algorithm 1's K = 4 upstream steps per Hessian step"),
    },
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--name", default=None, help="the Dataset whose labelled frames the fit set is built from")
    ap.add_argument("--fit-name", default=None, help="the fit Dataset (default: <name>_fit)")
    ap.add_argument("--level", default=frame_labels.DEFAULT_LEVEL)
    ap.add_argument("--stage", default="all", choices=("balance", "cost", "scan", "all"))
    ap.add_argument("--epochs", type=int, default=100, help="epochs per scan run")
    ap.add_argument("--cost-epochs", type=int, default=1, help="epochs per cost measurement")
    ap.add_argument("--scan", type=float, nargs="*", default=(0.3, 1.0, 3.0),
                    help="multiples of the epoch-0 balance to scan")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--forces-weight", type=float, default=100.0)
    ap.add_argument("--energy-weight", type=float, default=1.0)
    ap.add_argument("--num-samples-pt", type=int, nargs="*", default=(0, 5000, 20000),
                    help="replay sizes to report the arithmetic for (no run: the ratio is arithmetic)")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--valid-fraction", type=float, default=0.2)
    ap.add_argument("--no-strict-fork", action="store_true", default=True,
                    help="kept on by default: the fit is a measurement, not a product")
    args = ap.parse_args()

    t0 = time.time()
    name = args.name or args.tag
    fit_name = args.fit_name or (name + "_fit")
    root = config.runs_root(config.load())
    src = Path(dataset.datasets_dir(root, args.tag, name))
    fit_dir = Path(dataset.datasets_dir(root, args.tag, fit_name))

    fit_dir, dinfo = smoke_fit.build_fit_dataset(src, args.level, fit_dir, fit_name,
                                                 seed=args.seed, valid_fraction=args.valid_fraction)
    calc, engine_name, prov = engine.calculator(device=args.device)
    print("fit set    {} frames ({} with a Hessian) from {} molecules; train {} valid {}".format(
        dinfo["N_FRAMES"], dinfo["N_HESSIAN_FRAMES"], dinfo["N_MOLECULES"], dinfo["N_TRAIN"], dinfo["N_VALID"]))
    print("           PURPOSE = fit: every number below is interpolation within these molecules")
    print("engine     {}  {}...".format(engine_name, prov["params_sha256"][:12]))

    stages = ("balance", "cost", "scan") if args.stage == "all" else (args.stage,)
    train_file = fit_dir / "train.{}.extxyz".format(args.level)
    balance_rows, cost_rows, scan_rows = [], [], []
    balanced = None

    if "balance" in stages or "scan" in stages:
        print("\n--- the epoch-0 balance (the base model, before a single step) ---")
        b = smoke_fit.epoch_zero_balance(calc, train_file, energy_weight=args.energy_weight,
                                         forces_weight=args.forces_weight, probe="cartesian")
        balance_rows.append(dict(N_FRAMES=b["N_FRAMES"],
                                 N_HESSIAN_FRAMES=b["N_HESSIAN_FRAMES"], L_E=b["L_E"], L_F=b["L_F"],
                                 L_H=b["L_H"], WE_LE=b["WE_LE"], WF_LF=b["WF_LF"],
                                 HESSIAN_WEIGHT_BALANCED=b["HESSIAN_WEIGHT_BALANCED"]))
        print("  L_E {:.4e}  L_F {:.4e}  L_H {:.4e}   w_F L_F {:.4e}   ->  w_H = {:.4g}".format(
            b["L_E"], b["L_F"], b["L_H"], b["WF_LF"], b["HESSIAN_WEIGHT_BALANCED"]))
        balanced = b["HESSIAN_WEIGHT_BALANCED"]
        print("  (the rule is PHL's, the number is not: w_H = w_F L_F / L_H measured HERE, on the full"
              " matrix and on this base model -- S0-C-64)")

    if "cost" in stages:
        print("\n--- the cost of a probe setting ({} epoch(s) each) ---".format(args.cost_epochs))
        cost_rows = smoke_fit.cost_table(fit_dir, args.tag, fit_name, args.level, epochs=args.cost_epochs,
                                         batch_size=args.batch_size, strict_fork=not args.no_strict_fork,
                                         energy_weight=args.energy_weight, forces_weight=args.forces_weight,
                                         hessian_weight=balanced or 1.0,
                                         device=args.device, seed=args.seed)
        print("  {:20s} {:>6s} {:>14s} {:>10s}".format("setting", "k", "s per epoch", "x E/F"))
        for r in cost_rows:
            print("  {:20s} {:6d} {:14.1f} {:>10s}".format(
                r["SETTING"], r["N_PROBES"], r["SECONDS_PER_EPOCH"],
                "-" if r["RATIO_TO_EF"] is None else "{:.2f}".format(r["RATIO_TO_EF"])))

    if "scan" in stages:
        if balanced is None:
            balanced = 1.0
        print("\n--- the scan: {} epochs per run ---".format(args.epochs))
        runs = [dict(label="x{:g}".format(m), probe="gaussian", n_probes=4,
                     hessian_weight=balanced * m) for m in args.scan]
        for spec in runs:
            run_name = "fit_{}".format(spec["label"])
            out = train_run.run_training(fit_dir, args.tag, fit_name, args.level, run_name,
                                         probe=spec["probe"], n_probes=spec["n_probes"],
                                         hessian_weight=spec["hessian_weight"],
                                         energy_weight=args.energy_weight, forces_weight=args.forces_weight,
                                         max_epochs=args.epochs, batch_size=args.batch_size,
                                         device=args.device, seed=args.seed,
                                         strict_fork=not args.no_strict_fork)
            info = out["info"]
            train_losses = [r["loss"] for r in out["epochs"] if r["split"] == "train" and r["loss"] is not None]
            valid_losses = [r["loss"] for r in out["epochs"] if r["split"] == "valid" and r["loss"] is not None]
            row = dict(RUN=run_name, PROBE=spec["probe"], N_PROBES=spec["n_probes"],
                       HESSIAN_WEIGHT=spec["hessian_weight"],
                       W_H_OVER_BALANCED=spec["hessian_weight"] / balanced if balanced else None,
                       EPOCHS=info["N_EPOCHS"], SECONDS_PER_EPOCH=info["SECONDS_PER_EPOCH"],
                       FINAL_TRAIN_LOSS=train_losses[-1] if train_losses else None,
                       FINAL_VALID_LOSS=valid_losses[-1] if valid_losses else None,
                       JUDGE_LOW_MODE_MAE_CM=None, BASE_LOW_MODE_MAE_CM=None, VERDICT="-")
            model = info.get("MODEL_FILE", "-")
            if Path(model).is_file():
                from mace.calculators import MACECalculator
                tuned = MACECalculator(model_paths=model, device=args.device, default_dtype="float64")
                j = judge.run(root, args.tag, fit_name, args.level, tuned, run_name, base_calc=calc,
                              base_engine=engine_name, run_name=run_name, splits=("valid", "train"),
                              write=True)
                held = [d for d in j["distributions"] if d["FREQ_MAE_LOW_CM"] is not None]
                if held:
                    row["JUDGE_LOW_MODE_MAE_CM"] = held[0]["FREQ_MAE_LOW_CM"]
                    row["BASE_LOW_MODE_MAE_CM"] = held[0]["BASE_FREQ_MAE_LOW_CM"]
                row["VERDICT"] = j["info"]["VERDICT"]
            scan_rows.append(row)
            print("  {:10s} w_H {:>10.4g}  {:5.1f} s/epoch  train {:>10}  valid {:>10}  low-mode MAE {:>8}  {}".format(
                run_name, spec["hessian_weight"], row["SECONDS_PER_EPOCH"],
                judge._num(row["FINAL_TRAIN_LOSS"], "{:.4f}"), judge._num(row["FINAL_VALID_LOSS"], "{:.4f}"),
                judge._num(row["JUDGE_LOW_MODE_MAE_CM"], "{:.2f}"), row["VERDICT"]))

    replay_rows = [smoke_fit.replay_ratio(dinfo["N_TRAIN"], dinfo["N_HESSIAN_FRAMES"], n)
                   for n in args.num_samples_pt]
    print("\n--- the replay arithmetic on THIS set (mace shuffles one concatenated set: "
          "every frame once per epoch) ---")
    print("  {:>16s} {:>18s} {:>26s}".format("num_samples_pt", "per train frame", "per Hessian frame (PFT: 4)"))
    for r in replay_rows:
        print("  {:16d} {:>18s} {:>26s}".format(
            r["NUM_SAMPLES_PT"], judge._num(r["REPLAY_PER_TRAIN"], "{:.2f}"),
            judge._num(r["REPLAY_PER_HESSIAN_FRAME"], "{:.2f}")))

    info = dict(TAG=args.tag, SOURCE_NAME=name, NAME=fit_name, PURPOSE="fit", LEVEL=args.level,
                STAGES=list(stages), N_FRAMES=dinfo["N_FRAMES"], N_HESSIAN_FRAMES=dinfo["N_HESSIAN_FRAMES"],
                N_TRAIN=dinfo["N_TRAIN"], N_VALID=dinfo["N_VALID"], N_MOLECULES=dinfo["N_MOLECULES"],
                ENGINE=engine_name, ENGINE_PARAMS_SHA256=prov["params_sha256"], DEVICE=args.device,
                EPOCHS=int(args.epochs), SECONDS=float(time.time() - t0))
    blocks = {"Calculation_Info": info, "Balance": balance_rows, "Cost": cost_rows,
              "Scan": scan_rows, "Replay": replay_rows}
    missing = prop.write(fit_dir / (STEP + ".toml"), blocks, SCHEMA, prop.NORMAL_TERMINATION, PROGNAME)
    if missing:
        raise RuntimeError("smoke_fit.toml keys outside the schema: {}".format(missing))
    rep = report.Report(PROGNAME, "Smoke fit on {} ({})".format(fit_name, info["PURPOSE"]))
    rep.section("the fit set (PURPOSE = fit: interpolation within these molecules)")
    for k in ("TAG", "SOURCE_NAME", "NAME", "LEVEL", "N_FRAMES", "N_HESSIAN_FRAMES", "N_TRAIN", "N_VALID",
              "N_MOLECULES", "ENGINE", "ENGINE_PARAMS_SHA256", "DEVICE", "EPOCHS", "SECONDS"):
        rep.kv(k, info[k])
    if balance_rows:
        rep.section("the epoch-0 balance")
        rep.table(["L_E", "L_F", "L_H", "w_F L_F", "w_H balanced"],
                  [["{:.4e}".format(b["L_E"]), "{:.4e}".format(b["L_F"]),
                    "{:.4e}".format(b["L_H"]), "{:.4e}".format(b["WF_LF"]),
                    "{:.6g}".format(b["HESSIAN_WEIGHT_BALANCED"])] for b in balance_rows])
    if cost_rows:
        rep.section("cost per epoch")
        rep.table(["setting", "k", "s per epoch", "x E/F"],
                  [[c["SETTING"], c["N_PROBES"], "{:.1f}".format(c["SECONDS_PER_EPOCH"]),
                    judge._num(c["RATIO_TO_EF"], "{:.2f}")] for c in cost_rows])
    if scan_rows:
        rep.section("the scan")
        rep.table(["run", "w_H", "w_H / balanced", "epochs", "train loss", "valid loss", "low-mode MAE", "verdict"],
                  [[s["RUN"], "{:.4g}".format(s["HESSIAN_WEIGHT"]), judge._num(s["W_H_OVER_BALANCED"], "{:.2f}"),
                    s["EPOCHS"], judge._num(s["FINAL_TRAIN_LOSS"], "{:.4f}"),
                    judge._num(s["FINAL_VALID_LOSS"], "{:.4f}"),
                    judge._num(s["JUDGE_LOW_MODE_MAE_CM"], "{:.2f}"), s["VERDICT"]] for s in scan_rows])
    rep.section("the replay arithmetic (mace: one concatenated set, every frame once per epoch)")
    rep.table(["num_samples_pt", "per train frame", "per Hessian frame", "PFT's"],
              [[r["NUM_SAMPLES_PT"], judge._num(r["REPLAY_PER_TRAIN"], "{:.2f}"),
                judge._num(r["REPLAY_PER_HESSIAN_FRAME"], "{:.2f}"), r["PFT_REFERENCE"]] for r in replay_rows])
    rep.write(fit_dir / (STEP + ".out"))
    print("\nrecord     {}".format(fit_dir / (STEP + ".out")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
