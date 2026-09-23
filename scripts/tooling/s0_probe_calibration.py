"""Probe calibration: how far the fixed-probe validation reading is from the exact one (ticket 32; S0-C-65).

TOOLING. Measures the ESTIMATOR, not a model: it needs no fine-tune and no training. The
question it answers is the one S0-C-55 left open -- the validation Hessian term is
`L^(K) = sum_j ||H_theta v_j - H_r v_j||^2 / (9 N^2 K)` with K probes FIXED per frame, so
how far is the mean over frames from the exact `mean_n ||dH_n||_F^2 / (9 N_n^2)`, and is
K = 4 enough?

    # tianhe, the campaign's molecule tree (whatever is labelled so far)
    python scripts/tooling/s0_probe_calibration.py --tag draw300 --device cuda --project-n 900
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
                       (the frame's own seed, `phl_loss.frame_seed` -- the set a run would use)
  check 2    SPREAD   = sd over `--seed-sets` independent fixed sets of (mean_n L^(K)_n) / mean_n L_n
  predicted  sd       = sqrt(Var) from `phl.estimator_variance` (eq. 2.3), per frame and for the mean

A frame's probes are its own (seeded from its Label's bytes), so the per-frame errors are
independent and the mean's error falls as 1/sqrt(n_frames): OFFSET and SPREAD should agree,
and SPREAD is what decides K. The report prints the smallest K whose SPREAD is under
`--target` (2 % by default, the scale of a late-training improvement) -- at `--project-n` frames when that is given (the
validation split's size, about 900 on draw300), because the per-frame spread is the
estimator's property while the mean's falls as 1/sqrt(n).

WHAT IT READS. `--tag`: the molecule tree `<root>/<tag>/<qid>/frames/`, preferring the
assembled `basin.<level>.extxyz` (`frame_labels.load_frames`) and falling back to the ORCA
job files `orca.<level>.basin_bBB_kK.hess` (`frame_labels.parse_label`) for molecules whose
labels are not assembled yet. `--frames`: a glob of extxyz files with `hessian` in info.
Only basin frames carry a Hessian in the campaign (S0-C-54); `--generator` overrides.

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
from openqha.training import judge, phl, phl_loss                 # noqa: E402

PROGNAME = "openQHA probe_calibration"
DEFAULT_K = (1, 2, 4, 8, 16)
DEFAULT_SEED_SETS = 10
#: the production value (S0-C-55/65): the K the validation actually uses
PRODUCTION_K = phl_loss.VALID_N_PROBES
#: "K is enough" when the spread of the mean is under this fraction of the mean
DEFAULT_TARGET = 0.02

K_ROW = {
    "K": ("Integer", None, "probes per frame"),
    "OFFSET": ("Double", None, "check 1: |mean L^(K) - mean L_exact| / mean L_exact with the production probes (each frame's own seed)"),
    "SEED_SPREAD": ("Double", None, "check 2: sd over the independent fixed probe sets of mean L^(K) / mean L_exact"),
    "SD_MEAN_PREDICTED": ("Double", None, "sqrt(sum_n Var_n) / (n_frames * mean L_exact) from eq. 2.3 (Rademacher)"),
    "SD_FRAME_PREDICTED": ("Double", None, "the median per-frame sqrt(Var_n) / L_n from eq. 2.3"),
    "SD_FRAME_MEASURED": ("Double", None, "the median per-frame sd over the seed sets, relative to L_n"),
    "COST_RATIO": ("Double", None, "K / 3N (median): the cost of the estimate against the full matrix"),
    "SPREAD_AT_N": ("Double", None, "the spread the mean would have over --project-n frames: the estimator's per-frame sd / sqrt(--project-n)"),
    "ENOUGH": ("Boolean", None, "the deciding spread (SPREAD_AT_N when --project-n is given, else SEED_SPREAD) <= --target"),
}

FRAME_ROW = {
    "qm9_index": ("String", None, "the molecule"),
    "generator": ("String", None, "the frame's generator"),
    "basin": ("Integer", None, "basin index"),
    "k": ("Integer", None, "frame index within the basin"),
    "n_atoms": ("Integer", None, "atoms"),
    "L_EXACT": ("Double", "eV^2/A^4", "||H_theta - H_r||_F^2 / (9 N^2), the training target, exact"),
    "L_PRODUCTION_K": ("Double", "eV^2/A^4", "the same frame's reading with its production probes at the production K"),
    "SD_PREDICTED": ("Double", "eV^2/A^4", "sqrt(Var) at the production K, eq. 2.3 (Rademacher)"),
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
        "PROBE": ("String", None, "the probe distribution (rademacher: phl_loss.VALID_PROBE)"),
        "SEED_SETS": ("Integer", None, "independent fixed probe sets drawn for check 2"),
        "TARGET": ("Double", None, "the spread under which a K counts as enough"),
        "N_PROJECT": ("Integer", None, "the frame count ENOUGH is decided at (the validation split size), or -1 for the frames read"),
        "K_ENOUGH": ("Integer", None, "the smallest K tried whose SEED_SPREAD is under TARGET, or -1"),
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


def labelled_frames(tag=None, root=None, level=None, species=(), limit=None, generator="basin", frames_glob=None):
    """[(atoms, source)] -- every frame that carries a reference Hessian, from the molecule
    tree of `tag` (assembled `<generator>.<level>.extxyz` first, the ORCA job files after)
    or from a glob of extxyz files. `atoms.info['hessian']` is the flattened Label."""
    out = []
    if frames_glob:
        for path in sorted(_glob.glob(str(frames_glob))):
            for a in frame_labels.frames_mod.read_frames(Path(path)):
                if a.info.get("hessian") is not None:
                    out.append((a, "extxyz"))
        return out
    root = root or config.runs_root(config.load())
    pairs = dataset_mod.molecules_with_branch_a(root, tag)
    if species:
        want = set(species)
        pairs = [(q, d) for q, d in pairs if q in want]
    if limit:
        pairs = pairs[: int(limit)]
    for qid, mol in pairs:
        got = [a for a in frame_labels.load_frames(mol, generator, level) if a.info.get("hessian") is not None]
        if got:
            out.extend((a, "extxyz") for a in got)
            continue
        # not assembled yet: read the finished ORCA jobs of this molecule's frames
        folder = layout.frames_dir(mol)
        for gen, basin, k in frame_labels.frame_list(mol, generators=(generator,)):
            stem = layout.orca_frame_stem(level, gen, basin, k)
            if not frame_labels.finished(folder, stem, hessian=True):
                continue
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
    return out


def frame_statistics(h_engine, h_ref, ks, seed, seed_sets):
    """One frame: the exact target, the production reading at every K, the readings of
    `seed_sets` independent fixed sets, the predicted variance, the stable rank."""
    a = np.asarray(h_engine, dtype=float) - np.asarray(h_ref, dtype=float)
    n3 = a.shape[0]
    exact = float(np.sum(a * a)) / (n3 * n3)
    b = a.T @ a
    tr = float(np.trace(b))
    fro2 = float(np.sum(b * b))
    r_eff = (tr * tr / fro2) if fro2 > 0 else float("nan")
    out = dict(exact=exact, n3=n3, r_eff=r_eff, production={}, sets={}, var={})
    for k in ks:
        var = phl.estimator_variance(h_engine, h_ref, None, None, k=k, metric="cartesian")["rademacher"]
        out["var"][k] = float(var)
        # the production probes: the frame's own seed, exactly as the validation draws them
        rng = np.random.default_rng(seed)
        v = rng.choice([-1.0, 1.0], size=(k, n3))
        out["production"][k] = float(np.sum((v @ a.T) ** 2)) / (n3 * n3 * k)
        vals = []
        for s in range(seed_sets):
            rs = np.random.default_rng((seed + 1_000_003 * (s + 1)) % (2 ** 63))
            vs = rs.choice([-1.0, 1.0], size=(k, n3))
            vals.append(float(np.sum((vs @ a.T) ** 2)) / (n3 * n3 * k))
        out["sets"][k] = np.asarray(vals, dtype=float)
    return out


def summarise(stats, ks, target=DEFAULT_TARGET, project_n=None):
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
        # the same estimator read over `project_n` frames instead of these n: the per-frame spread is a
        # property of the estimator, the mean's falls as 1/sqrt(n)
        at_n = float(sd_mean_pred / exact_mean * np.sqrt(n / float(project_n))) if (project_n and exact_mean > 0) \
            else float("nan")
        deciding = at_n if project_n else spread
        rows.append(dict(K=int(k),
                         OFFSET=float(abs(prod_mean - exact_mean) / exact_mean) if exact_mean > 0 else float("nan"),
                         SEED_SPREAD=spread,
                         SD_MEAN_PREDICTED=float(sd_mean_pred / exact_mean) if exact_mean > 0 else float("nan"),
                         SD_FRAME_PREDICTED=sd_frame_pred, SD_FRAME_MEASURED=sd_frame_meas,
                         COST_RATIO=float(k) / float(np.median([s["n3"] for s in stats])),
                         SPREAD_AT_N=at_n, ENOUGH=bool(deciding <= target)))
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
    ap.add_argument("--target", type=float, default=DEFAULT_TARGET, help="the spread under which a K is enough")
    ap.add_argument("--project-n", type=int, default=None,
                    help="decide ENOUGH at this frame count (the validation split size, e.g. 900 on draw300) instead of "
                         "at the frames read: the per-frame spread is the estimator's, the mean's falls as 1/sqrt(n)")
    ap.add_argument("--out", default=None, help="the Record (default: probe_calibration.<level>.toml here)")
    args = ap.parse_args(argv)
    if not args.tag and not args.frames:
        ap.error("--tag (the molecule tree) or --frames (a glob) is required")
    t0 = time.time()

    frames = labelled_frames(tag=args.tag, root=args.root, level=args.level, species=args.species,
                             limit=args.limit, generator=args.generator, frames_glob=args.frames)
    if not frames:
        print("no labelled frame with a Hessian found", file=sys.stderr)
        return 2
    calc, ename, prov = engine.calculator(device=args.device, name=args.engine)
    ks = sorted(set(int(k) for k in args.k))
    print("engine     {} ({})".format(ename, prov.get("params_sha256", "-")[:12]))
    print("frames     {} labelled frames of {} molecules at {}".format(
        len(frames), len({_frame_key(a)[0] for a, _s in frames}), args.level))
    print("probing    K = {}, {} seed sets, production K = {} ({})".format(
        ks, args.seed_sets, PRODUCTION_K, phl_loss.VALID_PROBE))

    stats, rows = [], []
    for i, (atoms, source) in enumerate(frames, 1):
        n3 = 3 * len(atoms)
        h_ref = np.asarray(atoms.info["hessian"], dtype=float).reshape(n3, n3)
        h_eng = judge.hessian_at(calc, atoms)
        seed = phl_loss.frame_seed(h_ref)
        st = frame_statistics(h_eng, h_ref, ks, seed, args.seed_sets)
        stats.append(st)
        qid, gen, basin, k = _frame_key(atoms, args.generator)
        rows.append(dict(qm9_index=qid, generator=gen, basin=int(basin), k=int(k), n_atoms=len(atoms),
                         L_EXACT=st["exact"],
                         L_PRODUCTION_K=st["production"].get(PRODUCTION_K, float("nan")),
                         SD_PREDICTED=float(np.sqrt(st["var"].get(PRODUCTION_K, float("nan")))),
                         R_EFF=st["r_eff"], SOURCE=source))
        if i % 50 == 0 or i == len(frames):
            print("   {:5d} / {} frames".format(i, len(frames)))

    exact_mean, k_rows = summarise(stats, ks, target=args.target, project_n=args.project_n)
    enough = [r["K"] for r in k_rows if r["ENOUGH"]]
    info = dict(ENGINE=ename, ENGINE_PARAMS_SHA256=str(prov.get("params_sha256", "-")), LEVEL=args.level,
                TAG=str(args.tag or "-"), GENERATOR=args.generator,
                N_MOLECULES=len({r["qm9_index"] for r in rows}), N_FRAMES=len(rows),
                N_ATOMS_MEDIAN=int(np.median([r["n_atoms"] for r in rows])), EXACT_MEAN=exact_mean,
                R_EFF_MEDIAN=float(np.median([s["r_eff"] for s in stats])), PRODUCTION_K=int(PRODUCTION_K),
                PROBE=str(phl_loss.VALID_PROBE), SEED_SETS=int(args.seed_sets), TARGET=float(args.target),
                N_PROJECT=int(args.project_n) if args.project_n else -1,
                K_ENOUGH=int(min(enough)) if enough else -1, SECONDS=time.time() - t0)
    out = Path(args.out) if args.out else Path("probe_calibration.{}.toml".format(args.level))
    info["FILE"] = str(out)
    missing = prop.write(out, {"ProbeCalibration": info, "K": k_rows, "Frame": rows}, SCHEMA,
                         prop.NORMAL_TERMINATION, PROGNAME)
    if missing:
        raise RuntimeError("keys outside the schema: {}".format(missing))

    print("\nexact      mean_n ||dH||^2/(9N^2) = {:.4e} eV^2/A^4 over {} frames; median stable rank {:.1f}".format(
        exact_mean, len(rows), info["R_EFF_MEDIAN"]))
    at = "   spread at n={}".format(args.project_n) if args.project_n else ""
    print("\n  K   check 1 OFFSET   check 2 SPREAD   sd(mean) predicted   per-frame sd pred / meas   cost K/3N{}   enough".format(at))
    for r in k_rows:
        tail = "   {:8.2%}".format(r["SPREAD_AT_N"]) if args.project_n else ""
        print("  {:2d}     {:7.2%}         {:7.2%}          {:7.2%}              {:.2f} / {:.2f}            {:5.1%}{}      {}".format(
            r["K"], r["OFFSET"], r["SEED_SPREAD"], r["SD_MEAN_PREDICTED"], r["SD_FRAME_PREDICTED"],
            r["SD_FRAME_MEASURED"], r["COST_RATIO"], tail, "yes" if r["ENOUGH"] else "no"))
    prod = next((r for r in k_rows if r["K"] == PRODUCTION_K), None)
    if prod:
        verdict = "the production seeds are a typical draw" if prod["OFFSET"] <= 2.0 * prod["SEED_SPREAD"] \
            else "the production seeds sit outside the typical spread -- read the Frame rows"
        print("\nproduction K = {}: offset {:.2%} against a spread of {:.2%} -> {}".format(
            PRODUCTION_K, prod["OFFSET"], prod["SEED_SPREAD"], verdict))
    print("smallest K with a spread under {:.0%}{}: {}".format(
        args.target, " at n = {} frames".format(args.project_n) if args.project_n else " over the frames read",
        info["K_ENOUGH"] if info["K_ENOUGH"] > 0 else "none of those tried"))
    print("written    {}".format(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
