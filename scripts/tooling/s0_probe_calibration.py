"""Probe calibration: how far the fixed-probe validation reading is from the exact one (ticket 32; S0-C-65).

TOOLING. Measures the ESTIMATOR, not a model: it needs no fine-tune and no training. The
question it answers is the one S0-C-55 left open -- the validation Hessian term is
`L^(K) = sum_j ||H_theta v_j - H_r v_j||^2 / (9 N^2 K)` with K probes FIXED per frame, so
how far is the mean over frames from the exact `mean_n ||dH_n||_F^2 / (9 N_n^2)`, and is
K = 4 enough?

    # tianhe, the campaign's molecule tree (whatever is labelled so far). The login node has no
    # GPU: read there on the CPU (one full Hessian per frame -- keep --limit small), or submit the
    # tool to a compute node and keep --device cuda.
    python scripts/tooling/s0_probe_calibration.py --tag draw300 --limit 20
    python scripts/tooling/s0_probe_calibration.py --tag draw300 --device cuda
    # a subset, or one molecule
    python scripts/tooling/s0_probe_calibration.py --tag draw300 --limit 50
    python scripts/tooling/s0_probe_calibration.py --tag draw300 --species dsgdb9nsd_000035
    # any directory of labelled frames (no tree needed)
    python scripts/tooling/s0_probe_calibration.py --frames 'tests/data/methyloxirane_frames/basin.wb97m*.extxyz'

WHAT IT DOES. For every labelled basin frame it takes the engine's full Hessian once
(`get_hessian`, 3N HVPs -- the only model call) and the reference Hessian from the label,
then everything else is linear algebra on the two matrices:

  exact      L_n      = ||H_theta - H_r||_F^2 / (9 N^2)                  (the training target)
  check 1    OFFSET   = |mean_n L^(K)_n - mean_n L_n| / mean_n L_n       with the PRODUCTION probes
                       (the frame's own stored set, or the same draw 04_dataset would make from
                        its identity -- the set a run would use; S0-C-67)
  check 2    SPREAD   = sd over `--seed-sets` independent fixed sets of (mean_n L^(K)_n) / mean_n L_n
  predicted  sd       = sqrt(Var) from `phl.estimator_variance` (eq. 2.3), per frame and for the mean

A frame's probes are its own (drawn from its identity by the Dataset, S0-C-67), so the probe errors are
independent ACROSS frames whatever the frames' own correlation, and Var(mean) = sum_n Var_n / n^2
is exact rather than an assumption. What this tool does NOT do is decide whether K is enough:
that depends on how large a change in the validation reading a training decision turns on, which
is a property of a real run and is measured by ticket 34 (the exact value on the validation file
before and after training, beside the last epoch's probe reading). An earlier version of this
tool printed an `ENOUGH` column against an asserted 2 % target and extrapolated the spread to a
hypothetical frame count; both were voided on 2026-09-23 as unearned.

WHAT IT READS. `--tag`: the molecule tree `<root>/<tag>/<qid>/frames/`, preferring the
assembled `basin.<level>.extxyz` (`frame_labels.load_frames`) and falling back to the ORCA
job files `orca.<level>.basin_bBB_kK.hess` (`frame_labels.parse_label`) for molecules whose
labels are not assembled yet. `--frames`: a glob of extxyz files with `hessian` in info.
Only basin frames carry a Hessian in the campaign (S0-C-54); `--generator` overrides. A molecule
whose branch A finished but whose Frame set does not exist yet (step 02 has not reached it) is
skipped, not an error: the campaign's stages run at different speeds and this tool reads whatever
is labelled so far.

WHAT IT WRITES. `<out>` (default `probe_calibration.<level>.toml` beside the Dataset root or
the working directory): `[ProbeCalibration]` (engine, level, counts, the exact mean, the
median stable rank), one `[[K]]` row per K with OFFSET / SPREAD / predicted and measured sd
/ cost ratio, and one `[[Frame]]` row per frame. Nothing else is written; no state changes.
"""
import argparse
import glob as _glob
import sys
import time
from pathlib import Path

import numpy as np


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))
from openqha import config                                       # noqa: E402
from openqha.data import dataset as dataset_mod, frame_labels     # noqa: E402
from openqha.potentials import engine                             # noqa: E402
from openqha.store import layout, property as prop                # noqa: E402
from openqha.training import judge                                # noqa: E402
from openqha_hessian import phl, phl_loss                         # noqa: E402

PROGNAME = "openQHA probe_calibration"
DEFAULT_K = (1, 2, 4, 8, 16)
DEFAULT_SEED_SETS = 10
#: the production value (S0-C-55/65): the K the validation actually uses
PRODUCTION_K = phl_loss.VALID_N_PROBES

K_ROW = {
    "K": ("Integer", None, "probes per frame"),
    "OFFSET": ("Double", None, "check 1: |mean L^(K) - mean L_exact| / mean L_exact with the production probes (each frame's own seed)"),
    "SEED_SPREAD": ("Double", None, "check 2: sd over the independent fixed probe sets of mean L^(K) / mean L_exact"),
    "SD_MEAN_PREDICTED": ("Double", None, "sqrt(sum_n Var_n) / (n_frames * mean L_exact) from eq. 2.3, for the draw the validation uses"),
    "SD_FRAME_PREDICTED": ("Double", None, "the median per-frame sqrt(Var_n) / L_n from eq. 2.3"),
    "SD_FRAME_MEASURED": ("Double", None, "the median per-frame sd over the seed sets, relative to L_n"),
    "COST_RATIO": ("Double", None, "K / 3N (median): the cost of the estimate against the full matrix"),
}

FRAME_ROW = {
    "qm9_index": ("String", None, "the molecule"),
    "generator": ("String", None, "the frame's generator"),
    "basin": ("Integer", None, "basin index"),
    "k": ("Integer", None, "frame index within the basin"),
    "n_atoms": ("Integer", None, "atoms"),
    "L_EXACT": ("Double", "eV^2/A^4", "||H_theta - H_r||_F^2 / (9 N^2), the training target, exact"),
    "L_PRODUCTION_K": ("Double", "eV^2/A^4", "the same frame's reading with its production probes at the production K"),
    "SD_PREDICTED": ("Double", "eV^2/A^4", "sqrt(Var) at the production K, eq. 2.3, for the draw the validation uses"),
    "R_EFF": ("Double", None, "stable rank (tr B)^2 / ||B||_F^2 of B = dH^T dH: over how many directions the error spreads"),
    "SOURCE": ("String", None, "extxyz (assembled label file) or orca (the job files)"),
}

SCHEMA = {
    "ProbeCalibration": {
        "ENGINE": ("String", None, "the engine whose Hessians were taken (the base model by default)"),
        "ENGINE_PARAMS_SHA256": ("String", None, "the engine's parameter fingerprint"),
        "LEVEL": ("String", None, "the reference level of the labels"),
        "TAG": ("String", None, "the molecule tree's tag, or - for --frames"),
        "GENERATOR": ("String", None, "the generator whose frames were read"),
        "N_MOLECULES": ("Integer", None, "molecules with at least one labelled frame"),
        "N_FRAMES": ("Integer", None, "frames read"),
        "N_ATOMS_MEDIAN": ("Integer", None, "median atom count (the cost ratio K/3N is read against it)"),
        "EXACT_MEAN": ("Double", "eV^2/A^4", "mean_n ||dH_n||_F^2 / (9 N_n^2): what the validation term estimates"),
        "R_EFF_MEDIAN": ("Double", None, "median stable rank of the error matrix; the estimator's noise falls as 1/sqrt(K r_eff)"),
        "PRODUCTION_K": ("Integer", None, "the K the validation uses (phl_loss.VALID_N_PROBES)"),
        "PROBE": ("String", None, "the probe distribution the readings and the predicted sd use (phl_loss.VALID_PROBE; gaussian since S0-C-68)"),
        "SEED_SETS": ("Integer", None, "independent fixed probe sets drawn for check 2"),
        "SECONDS": ("Double", "s", "wall time"),
        "FILE": ("String", None, "this Record"),
    },
    "K": K_ROW,
    "Frame": FRAME_ROW,
}


def _frame_key(atoms, default_gen="basin"):
    info = atoms.info
    return (str(info.get("qm9_index", info.get("molecule", "-"))), str(info.get("generator", default_gen)),
            int(info.get("basin", 0)), int(info.get("k", 0)))


def labelled_frames(tag=None, root=None, level=None, species=(), limit=None, generator="basin",
                    frames_glob=None, funnel=None):
    """[(atoms, source)] -- every frame that carries a reference Hessian, from the molecule
    tree of `tag` (assembled `<generator>.<level>.extxyz` first, the ORCA job files after)
    or from a glob of extxyz files. `atoms.info['hessian']` is the flattened Label.

    `funnel`: an optional dict filled with where the frames were lost -- the root and tag
    directory searched, the molecules with a branch A Record, those with an assembled label
    file, those with a finished ORCA Hessian job, and one example path of each kind. An empty
    result is almost always a wrong root or an unsourced environment, and the caller prints
    this instead of "nothing found"."""
    out = []
    f = {} if funnel is None else funnel
    f.setdefault("root", "-"); f.setdefault("tag_dir", "-"); f.setdefault("n_molecules", 0)
    f.setdefault("n_with_frameset", 0); f.setdefault("n_with_assembled", 0); f.setdefault("n_with_orca", 0)
    f.setdefault("example_assembled", "-"); f.setdefault("example_orca", "-"); f.setdefault("example_molecule", "-")
    if frames_glob:
        for path in sorted(_glob.glob(str(frames_glob))):
            for a in frame_labels.frames_mod.read_frames(Path(path)):
                if a.info.get("hessian") is not None:
                    out.append((a, "extxyz"))
        return out
    root = root or config.runs_root(config.load())
    f["root"] = str(root)
    f["tag_dir"] = str(Path(root) / str(tag))
    pairs = dataset_mod.molecules_with_branch_a(root, tag)
    f["n_molecules"] = len(pairs)
    if pairs:
        f["example_molecule"] = str(pairs[0][1])
    if species:
        want = set(species)
        pairs = [(q, d) for q, d in pairs if q in want]
    if limit:
        pairs = pairs[: int(limit)]
    for qid, mol in pairs:
        assembled = layout.frames_file(mol, generator, level)
        if f["example_assembled"] == "-":
            f["example_assembled"] = str(assembled)
        got = [a for a in frame_labels.load_frames(mol, generator, level) if a.info.get("hessian") is not None]
        if got:
            f["n_with_frameset"] += 1
            f["n_with_assembled"] += 1
            out.extend((a, "extxyz") for a in got)
            continue
        # not assembled yet: read the finished ORCA jobs of this molecule's frames
        folder = layout.frames_dir(mol)
        try:
            wanted = frame_labels.frame_list(mol, generators=(generator,))
        except FileNotFoundError:
            continue                    # branch A finished here, step 02 has not run yet: nothing to label
        f["n_with_frameset"] += 1
        had = False
        for gen, basin, k in wanted:
            stem = layout.orca_frame_stem(level, gen, basin, k)
            if f["example_orca"] == "-":
                f["example_orca"] = str(folder / (stem + ".hess"))
            if not frame_labels.finished(folder, stem, hessian=True):
                continue
            had = True
            try:
                atoms = frame_labels.load_frame(mol, gen, basin, k)
            except KeyError:
                continue
            label = frame_labels.parse_label(folder, atoms, stem)
            if label.get("hessian") is None:
                continue
            a = atoms.copy()
            a.info = dict(atoms.info)
            a.info["hessian"] = np.asarray(label["hessian"], dtype=float).reshape(-1)
            a.info.setdefault("qm9_index", qid)
            a.info.setdefault("generator", gen)
            a.info.setdefault("basin", basin)
            a.info.setdefault("k", k)
            out.append((a, "orca"))
        f["n_with_orca"] += int(had)
    return out


def frame_probes(qid, key, n_atoms, k_max, seed_sets, stored=None):
    """The frame's PRODUCTION probe set and `seed_sets` independent ones, all [k_max, 3N].

    Production is what a run would read: the set stored with the frame when it comes from a
    Dataset's valid file, and otherwise the same draw `04_dataset` would make from the
    frame's IDENTITY and the Dataset's seed (S0-C-67) -- never anything derived from the
    Label. The independent sets are the same draw under shifted seeds, for check 2.
    """
    n3 = 3 * int(n_atoms)
    if stored is not None:
        prod = np.asarray(stored, dtype=float).reshape(-1, n3)
    else:
        prod = dataset_mod.valid_probes(dataset_mod.SEED, qid, key, n_atoms, k_max).astype(float)
    sets = [dataset_mod.valid_probes(dataset_mod.SEED + 1_000_003 * (s + 1), qid, key, n_atoms, k_max).astype(float)
            for s in range(int(seed_sets))]
    return prod, sets


def frame_statistics(h_engine, h_ref, ks, production_v, probe_sets):
    """One frame: the exact target, the production reading at every K (the first K rows of
    the frame's own set -- nested, as the validation takes them), the readings of the
    independent sets, the predicted variance, the stable rank."""
    a = np.asarray(h_engine, dtype=float) - np.asarray(h_ref, dtype=float)
    n3 = a.shape[0]
    exact = float(np.sum(a * a)) / (n3 * n3)
    b = a.T @ a
    tr = float(np.trace(b))
    fro2 = float(np.sum(b * b))
    r_eff = (tr * tr / fro2) if fro2 > 0 else float("nan")
    out = dict(exact=exact, n3=n3, r_eff=r_eff, production={}, sets={}, var={})
    for k in ks:
        # the closed form of the draw the validation actually uses (S0-C-68: PHL's normal),
        # not of the one with the smaller variance
        var = phl.estimator_variance(h_engine, h_ref, k=k)[phl_loss.VALID_PROBE]
        out["var"][k] = float(var)
        # the production reading: the FIRST K rows of the frame's own stored set, which is
        # what the validation takes (S0-C-67). The sets are nested, so the K rows of this
        # table are readings of one set, not of K unrelated draws.
        v = np.asarray(production_v, dtype=float)[:k]
        out["production"][k] = float(np.sum((v @ a.T) ** 2)) / (n3 * n3 * k)
        vals = [float(np.sum((np.asarray(vs, dtype=float)[:k] @ a.T) ** 2)) / (n3 * n3 * k)
                for vs in probe_sets]
        out["sets"][k] = np.asarray(vals, dtype=float)
    return out


def summarise(stats, ks):
    """The [[K]] rows from the per-frame statistics: check 1, check 2 and the predictions."""
    n = len(stats)
    exact_mean = float(np.mean([s["exact"] for s in stats])) if n else float("nan")
    rows = []
    for k in ks:
        prod_mean = float(np.mean([s["production"][k] for s in stats]))
        set_means = np.mean(np.asarray([s["sets"][k] for s in stats], dtype=float), axis=0)
        sd_mean_pred = float(np.sqrt(np.sum([s["var"][k] for s in stats])) / n) if n else float("nan")
        sd_frame_pred = float(np.median([np.sqrt(s["var"][k]) / s["exact"] for s in stats if s["exact"] > 0]))
        sd_frame_meas = float(np.median([np.std(s["sets"][k]) / s["exact"] for s in stats if s["exact"] > 0]))
        spread = float(np.std(set_means) / exact_mean) if exact_mean > 0 else float("nan")
        rows.append(dict(K=int(k),
                         OFFSET=float(abs(prod_mean - exact_mean) / exact_mean) if exact_mean > 0 else float("nan"),
                         SEED_SPREAD=spread,
                         SD_MEAN_PREDICTED=float(sd_mean_pred / exact_mean) if exact_mean > 0 else float("nan"),
                         SD_FRAME_PREDICTED=sd_frame_pred, SD_FRAME_MEASURED=sd_frame_meas,
                         COST_RATIO=float(k) / float(np.median([s["n3"] for s in stats]))))
    return exact_mean, rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default=None, help="the molecule tree's tag (the campaign's labels)")
    ap.add_argument("--root", default=None, help="the runs root (default: $S0_RUNS_ROOT)")
    ap.add_argument("--frames", default=None, help="a glob of labelled extxyz files instead of a tree")
    ap.add_argument("--species", action="append", default=[], help="restrict to these molecules")
    ap.add_argument("--limit", type=int, default=None, help="at most this many molecules")
    ap.add_argument("--generator", default="basin", help="the generator whose frames carry a Hessian (S0-C-54)")
    ap.add_argument("--level", default=frame_labels.DEFAULT_LEVEL, help="the reference level")
    ap.add_argument("--engine", default=None, help="the engine (default: the configured base model)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--k", type=int, nargs="+", default=list(DEFAULT_K), help="probe counts to compare")
    ap.add_argument("--seed-sets", type=int, default=DEFAULT_SEED_SETS, help="independent fixed sets for check 2")
    ap.add_argument("--out", default=None, help="the Record (default: probe_calibration.<level>.toml here)")
    args = ap.parse_args(argv)
    if not args.tag and not args.frames:
        ap.error("--tag (the molecule tree) or --frames (a glob) is required")
    if str(args.device).startswith("cuda"):
        import torch                                             # noqa: E402  (kept out of import time)
        if not torch.cuda.is_available():
            ap.error("--device cuda, but torch.cuda.is_available() is False -- this is a login node,"
                     " or a torch build without CUDA. Either drop the flag and read on the CPU here"
                     " (one full Hessian per frame: keep --limit small), or submit the tool to a"
                     " compute node with a GPU and keep --device cuda.")
    t0 = time.time()

    funnel = {}
    frames = labelled_frames(tag=args.tag, root=args.root, level=args.level, species=args.species,
                             limit=args.limit, generator=args.generator, frames_glob=args.frames,
                             funnel=funnel)
    if not frames:
        print("no labelled frame with a Hessian found. Where it looked:", file=sys.stderr)
        if args.frames:
            print("  glob                  {}".format(args.frames), file=sys.stderr)
            print("  matched files         {}".format(len(_glob.glob(str(args.frames)))), file=sys.stderr)
            print("  (a file matches only if a frame carries `hessian` in info: the reference-level"
                  " file, not the engine-level one)", file=sys.stderr)
            return 2
        print("  runs root             {}{}".format(funnel["root"], "" if args.root else "   (from the config / $S0_RUNS_ROOT)"),
              file=sys.stderr)
        print("  tag directory         {}   {}".format(funnel["tag_dir"],
              "exists" if Path(funnel["tag_dir"]).is_dir() else "DOES NOT EXIST"), file=sys.stderr)
        print("  molecules with branch A (_records/branchA.toml + basins.done): {}".format(funnel["n_molecules"]),
              file=sys.stderr)
        print("  with a Frame set (frames/frames.toml, step 02): {}".format(funnel["n_with_frameset"]),
              file=sys.stderr)
        print("  with an assembled label file: {}   e.g. {}".format(funnel["n_with_assembled"], funnel["example_assembled"]),
              file=sys.stderr)
        print("  with a finished ORCA Hessian job: {}   e.g. {}".format(funnel["n_with_orca"], funnel["example_orca"]),
              file=sys.stderr)
        print("  level                 {}".format(args.level), file=sys.stderr)
        if not funnel["n_molecules"]:
            print("\n  -> the root or the tag is wrong, or the environment was not sourced:\n"
                  "     source hpc/env/common.sh && source hpc/env/tianhe.sh   (sets S0_RUNS_ROOT)\n"
                  "     or pass it: --root <the directory that holds {}/>".format(args.tag), file=sys.stderr)
        elif not funnel["n_with_frameset"]:
            print("\n  -> branch A has finished but step 02 has not: there is no Frame set to label yet."
                  "\n     python workflows/hessian_learning/02_frames.py --tag {}   (then 03_labels)".format(args.tag),
                  file=sys.stderr)
        else:
            print("\n  -> the Frame sets are there but no frame carries a Hessian at this level:"
                  "\n     check the level spelling, or that 03_labels has finished a basin job"
                  "\n     (ls {}/frames/ | head)".format(funnel["example_molecule"]), file=sys.stderr)
        return 2
    calc, ename, prov = engine.calculator(device=args.device, name=args.engine)
    ks = sorted(set(int(k) for k in args.k))
    print("engine     {} ({})".format(ename, prov.get("params_sha256", "-")[:12]))
    print("frames     {} labelled frames of {} molecules at {}".format(
        len(frames), len({_frame_key(a)[0] for a, _s in frames}), args.level))
    if args.tag:
        print("           of {} molecules with branch A: {} have a Frame set, {} an assembled label file, "
              "{} a finished ORCA Hessian job".format(funnel["n_molecules"], funnel["n_with_frameset"],
                                                      funnel["n_with_assembled"], funnel["n_with_orca"]))
    print("probing    K = {}, {} seed sets, production K = {} ({})".format(
        ks, args.seed_sets, PRODUCTION_K, phl_loss.VALID_PROBE))

    stats, rows = [], []
    for i, (atoms, source) in enumerate(frames, 1):
        n3 = 3 * len(atoms)
        h_ref = np.asarray(atoms.info["hessian"], dtype=float).reshape(n3, n3)
        h_eng = judge.hessian_at(calc, atoms)
        qid, gen, basin, k = _frame_key(atoms, args.generator)
        prod_v, sets_v = frame_probes(qid, (gen, basin, k), len(atoms), max(ks), args.seed_sets,
                                      stored=atoms.info.get("valid_probes"))
        st = frame_statistics(h_eng, h_ref, ks, prod_v, sets_v)
        stats.append(st)
        rows.append(dict(qm9_index=qid, generator=gen, basin=int(basin), k=int(k), n_atoms=len(atoms),
                         L_EXACT=st["exact"],
                         L_PRODUCTION_K=st["production"].get(PRODUCTION_K, float("nan")),
                         SD_PREDICTED=float(np.sqrt(st["var"].get(PRODUCTION_K, float("nan")))),
                         R_EFF=st["r_eff"], SOURCE=source))
        if i % 50 == 0 or i == len(frames):
            print("   {:5d} / {} frames".format(i, len(frames)))

    exact_mean, k_rows = summarise(stats, ks)
    info = dict(ENGINE=ename, ENGINE_PARAMS_SHA256=str(prov.get("params_sha256", "-")), LEVEL=args.level,
                TAG=str(args.tag or "-"), GENERATOR=args.generator,
                N_MOLECULES=len({r["qm9_index"] for r in rows}), N_FRAMES=len(rows),
                N_ATOMS_MEDIAN=int(np.median([r["n_atoms"] for r in rows])), EXACT_MEAN=exact_mean,
                R_EFF_MEDIAN=float(np.median([s["r_eff"] for s in stats])), PRODUCTION_K=int(PRODUCTION_K),
                PROBE=str(phl_loss.VALID_PROBE), SEED_SETS=int(args.seed_sets), SECONDS=time.time() - t0)
    out = Path(args.out) if args.out else Path("probe_calibration.{}.toml".format(args.level))
    info["FILE"] = str(out)
    missing = prop.write(out, {"ProbeCalibration": info, "K": k_rows, "Frame": rows}, SCHEMA,
                         prop.NORMAL_TERMINATION, PROGNAME)
    if missing:
        raise RuntimeError("keys outside the schema: {}".format(missing))

    print("\nexact      mean_n ||dH||^2/(9N^2) = {:.4e} eV^2/A^4 over {} frames; median stable rank {:.1f}".format(
        exact_mean, len(rows), info["R_EFF_MEDIAN"]))
    print("\n  K   check 1 OFFSET   check 2 SPREAD   sd(mean) predicted   per-frame sd pred / meas   cost K/3N")
    for r in k_rows:
        print("  {:2d}     {:7.2%}         {:7.2%}          {:7.2%}              {:.2f} / {:.2f}            {:5.1%}".format(
            r["K"], r["OFFSET"], r["SEED_SPREAD"], r["SD_MEAN_PREDICTED"], r["SD_FRAME_PREDICTED"],
            r["SD_FRAME_MEASURED"], r["COST_RATIO"]))
    prod = next((r for r in k_rows if r["K"] == PRODUCTION_K), None)
    if prod:
        print("\nproduction K = {}: offset {:.2%}, spread {:.2%} over {} seed sets and {} frames".format(
            PRODUCTION_K, prod["OFFSET"], prod["SEED_SPREAD"], args.seed_sets, len(rows)))
        print("no verdict is drawn from this table: whether K = {} changes a training decision is measured on a real\n"
              "run (ticket 34: the exact value on the validation file before and after training, beside the last\n"
              "epoch's probe reading), not extrapolated from here.".format(PRODUCTION_K))
    print("written    {}".format(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
