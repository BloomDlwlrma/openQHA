"""Workflow hessian_learning, step 06: the judge (ticket 14; ticket 22).

PRODUCTION. The ruler of Algorithm 3: the SHIPPED full Hessian (`MACECalculator.get_hessian`)
against the Label through `hessian_compare`, per structure class and per distribution,
with the entropy tier read from the msRRHO Records, the forgetting line on a fixed SPICE
draw, and one line per row. Nothing here calls the estimator or the training loss: the
optimiser reads eq. 6, the judge reads eq. 1 exactly.

THE GATE IS CLOSED (S0-C-60): every row is reported against its number, `VERDICT` reads
`REPORTED`, and nothing is decided by the judge; `--gate` reopens it. With the gate open,
GATE rows decide the verdict (S0-C-58/59): the Hessian MATRIX itself against the Label on
the held-out Hessian frames -- ||H_theta - H_r||_F^2 / (9 N^2), the training target's own
number, engine no worse than base -- the in_distribution no-degradation, the forgetting
line. Everything computed from the matrix afterwards is post-processing and a REFERENCE
row, reported and never gating: the low-mode frequency line, the msRRHO entropy at the
engine's own minima, the held-out generator's frames (displaced / merged / saddle, never
trained on) binned by RMS displacement with H, E and F metrics, and -- only with `--ramp`
-- the MD temperature ramp (Rodriguez 2025: from each model's own minimum, +5 K every 5 ps from
5 K until a pair distance's 50-step mean leaves [0.75, 1.5] x its equilibrium value;
HOURS per molecule per model at 600 K; `--ramp-max-K`, `--ramp-molecules` bound it).

    # the base model on the smoke Dataset (gate closed: every row reported, VERDICT = REPORTED)
    python workflows/hessian_learning/06_judge.py --tag rings --name smoke --engine base

    # the two calibrations with the gate open: base vs base PASSes; Hessian x0.81 fails the Hessian gate and the verdict
    python workflows/hessian_learning/06_judge.py --tag rings --name smoke --engine base --gate
    python workflows/hessian_learning/06_judge.py --tag rings --name smoke --engine base --scale 0.9 --gate

    # a fine-tuned potential (registered by 05_train --register); its train.toml heads the report
    python workflows/hessian_learning/06_judge.py --tag draw300 --engine MACE-OFF23_medium-R1 \
        --spice-file data/training_sets/spice_test_5000.extxyz
    # ... and, as post-processing on the chosen row only, the MD ramp
    python workflows/hessian_learning/06_judge.py --tag draw300 --engine MACE-OFF23_medium-R1 --run R1_ramp --ramp --ramp-max-K 600

Writes `<root>/<tag>/_datasets/<name>/judge/<run>/judge.{out,toml,dat}`.
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
from openqha.potentials import engine                        # noqa: E402
from openqha_hessian import judge                            # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--name", default=None, help="the Dataset name (default: the tag)")
    ap.add_argument("--level", default=frame_labels.DEFAULT_LEVEL, help="the reference level of the Labels")
    ap.add_argument("--engine", default="base",
                    help="a registered engine name, or 'base' for the production default (S0_ENGINE)")
    ap.add_argument("--base-engine", default=None, help="what to compare against (default: the production default)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="judge a deliberately wrong potential: forces x s, Hessian x s^2 (0.9 = the must-fail line)")
    ap.add_argument("--split", action="append", default=None, help="Dataset splits to judge (default: test)")
    ap.add_argument("--run", default=None, help="the judge run name (default: the engine, + the scale)")
    ap.add_argument("--spice-file", default=None, help="the fixed SPICE draw for the forgetting line")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--quiet", action="store_true", help="no per-frame progress")
    # the MD ramp (ticket 22; a reference row; post-processing, off by default -- S0-C-58)
    ap.add_argument("--ramp", action="store_true", help="run the MD temperature ramp (hours per molecule per model at 600 K)")
    ap.add_argument("--ramp-max-K", type=float, default=judge.RAMP["max_K"], help="the ramp's ceiling (default 600)")
    ap.add_argument("--ramp-step-K", type=float, default=judge.RAMP["step_K"])
    ap.add_argument("--ramp-step-ps", type=float, default=judge.RAMP["step_ps"], help="hold per temperature (default 5)")
    ap.add_argument("--ramp-seed", type=int, default=judge.RAMP["seed"])
    ap.add_argument("--ramp-molecules", nargs="+", default=None, metavar="QM9",
                    help="molecules to ramp (default: the pinned molecules present in the judged frames)")
    ap.add_argument("--gate", action="store_true", help="let the gate rows decide the VERDICT (closed by default, S0-C-60)")
    ap.add_argument("--thermo-tag", default=None,
                    help="the tag whose molecule directories hold this engine's msRRHO Records (S0_ENGINE=<engine> branch A under "
                         "its own tag); default: the campaign tag")
    ap.add_argument("--train-record", default=None,
                    help="the fine-tune's train.toml (default: found from the engine's "
                         "<campaign>-<run>+<stamp> name under the Dataset)")
    args = ap.parse_args()

    name = args.name or args.tag
    splits = tuple(args.split or ("test",))
    base_name = args.base_engine or engine.engine_name()
    engine_name = base_name if args.engine == "base" else args.engine

    calc, engine_name, _prov = engine.calculator(device=args.device, name=engine_name)
    base_calc, base_name, _base_prov = engine.calculator(device=args.device, name=base_name)
    run_name = args.run or (engine_name if args.scale == 1.0 else "{}_x{:g}".format(engine_name, args.scale))
    if args.scale != 1.0:
        calc = judge.ScaledCalculator(calc, args.scale)

    root = config.runs_root(config.load())
    train_record = args.train_record
    if train_record is None and engine_name.startswith(args.tag + "-"):
        run_part = engine_name[len(args.tag) + 1:].split("+", 1)[0]
        cand = Path(dataset.datasets_dir(root, args.tag, name)) / "train" / run_part / "train.toml"
        train_record = str(cand) if cand.is_file() else None
    ramp = dict(max_K=args.ramp_max_K, step_K=args.ramp_step_K, step_ps=args.ramp_step_ps, seed=args.ramp_seed) if args.ramp else None
    print("judging    {}{}  against {}".format(engine_name, "" if args.scale == 1.0 else
                                               "  x{:g} (calibration)".format(args.scale), base_name))
    print("dataset    {} / {} at {}, splits {}".format(args.tag, name, args.level, ", ".join(splits)))
    if ramp:
        print("ramp       to {:.0f} K in {:.0f} K steps of {:.1f} ps (seed {}); {}".format(
            ramp["max_K"], ramp["step_K"], ramp["step_ps"], ramp["seed"],
            "molecules " + " ".join(args.ramp_molecules) if args.ramp_molecules else "the pinned molecules present"))
    if train_record:
        print("record     {}".format(train_record))
    out = judge.run(root, args.tag, name, args.level, calc, engine_name,
                    base_calc=base_calc, base_engine=base_name, run_name=run_name, splits=splits,
                    scale=args.scale, spice_file=args.spice_file,
                    progress=None if args.quiet else (lambda s: print("  ...", s, end="\r", flush=True)),
                    ramp=ramp, ramp_molecules=args.ramp_molecules, train_record=train_record, gate=args.gate, thermo_tag=args.thermo_tag)
    if not args.quiet:
        print(" " * 70, end="\r")
    info = out["info"]
    print("frames     {} from {} molecules in {:.1f} s".format(info["N_FRAMES"], info["N_MOLECULES"], info["SECONDS"]))

    print("\n{:18s} {:>6s} {:>5s} {:>9s} {:>9s} {:>13s} {:>9s} {:>9s}".format(
        "distribution", "frames", "mols", "low MAE", "MAE", "||dH||^2/9N^2", "base low", "base MAE"))
    for d in out["distributions"]:
        print("{:18s} {:6d} {:5d} {:>9s} {:>9s} {:>13s} {:>9s} {:>9s}".format(
            d["DISTRIBUTION"], d["N_FRAMES"], d["N_MOLECULES"],
            judge._num(d["FREQ_MAE_LOW_CM"], "{:.2f}"), judge._num(d["FREQ_MAE_CM"], "{:.2f}"),
            judge._num(d["LOSS_CARTESIAN"], "{:.4e}"), judge._num(d["BASE_FREQ_MAE_LOW_CM"], "{:.2f}"),
            judge._num(d["BASE_FREQ_MAE_CM"], "{:.2f}")))
    if out["classes"]:
        print("\n{:24s} {:>6s} {:>5s} {:>9s} {:>9s}".format("class", "frames", "mols", "low MAE", "MAE"))
        for c in out["classes"]:
            print("{:24s} {:6d} {:5d} {:>9s} {:>9s}".format(
                c["CLASS"], c["N_FRAMES"], c["N_MOLECULES"],
                judge._num(c["FREQ_MAE_LOW_CM"], "{:.2f}"), judge._num(c["FREQ_MAE_CM"], "{:.2f}")))
    if out["thermochemistry"]:
        print("\n{:18s} {:>10s} {:>12s} {:>6s}  {}".format(
            "molecule", "S_msRRHO", "model error", "anh", "source"))
        for r in out["thermochemistry"]:
            print("{:18s} {:>10s} {:>12s} {:6d}  {}".format(
                r["QM9_INDEX"], judge._num(r["S_MSRRHO"], "{:.3f}"), judge._num(r["MODEL_ERROR_S_REF"], "{:+.3f}"),
                r["N_ANHARMONIC"], "-" if r["SOURCE"] == "-" else Path(r["SOURCE"]).name))
    if out["forgetting"]:
        f = out["forgetting"]
        print("\nforgetting  {} frames: E {} vs {} meV/atom, F {} vs {} meV/A".format(
            f["N_FRAMES"], judge._num(f["ENGINE_E_RMSE_MEV_PER_ATOM"], "{:.2f}"),
            judge._num(f["BASE_E_RMSE_MEV_PER_ATOM"], "{:.2f}"),
            judge._num(f["ENGINE_F_RMSE_MEV_A"], "{:.2f}"), judge._num(f["BASE_F_RMSE_MEV_A"], "{:.2f}")))

    if out.get("displacement"):
        print("\n{:18s} {:>7s} {:>6s} {:>8s} {:>8s} {:>8s} {:>8s} {:>10s} {:>10s}".format(
            "reference bins", "rms", "frames", "H MAE", "base H", "E MAE", "base E", "F RMSE", "base F"))
        for d in out["displacement"]:
            print("{:18s} {:>7s} {:6d} {:>8s} {:>8s} {:>8s} {:>8s} {:>10s} {:>10s}".format(
                d["DISTRIBUTION"], d["RMS_BIN"], d["N_FRAMES"], judge._num(d["HESSIAN_MAE"], "{:.4f}"),
                judge._num(d["BASE_HESSIAN_MAE"], "{:.4f}"), judge._num(d["E_MAE_MEV_PER_ATOM"], "{:.2f}"),
                judge._num(d["BASE_E_MAE_MEV_PER_ATOM"], "{:.2f}"), judge._num(d["F_RMSE_MEV_A"], "{:.2f}"),
                judge._num(d["BASE_F_RMSE_MEV_A"], "{:.2f}")))
    if out.get("ramp"):
        print("\n{:18s} {:>7s} {:>9s} {:>8s} {:>9s}".format("MD ramp (ref.)", "which", "survived", "fail T", "fail ps"))
        for r in out["ramp"]:
            print("{:18s} {:>7s} {:>9s} {:>8s} {:>9s}".format(
                r["QM9_INDEX"], r["WHICH"], "yes" if r["SURVIVED"] else "no", judge._num(r["FAIL_T_K"], "{:.0f}"),
                judge._num(r["FAIL_PS"], "{:.2f}")))
    print("\n{:34s} {:>4s} {:>10s} {:>10s}  {}".format("verdict line", "gate", "value", "threshold", "result"))
    for line in out["verdict"]:
        print("{:34s} {:>4s} {:>10s} {:>10s}  {}".format(
            line["LINE"], line.get("GATE", "yes"), judge._num(line["VALUE"], "{:.4f}"),
            judge._num(line["THRESHOLD"], "{:.4f}"), line["RESULT"]))
    print("\nVERDICT    {}".format(info["VERDICT"]))
    print("record     {}".format(out["run_dir"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
