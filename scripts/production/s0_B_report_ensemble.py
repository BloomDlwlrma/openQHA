"""The molecule's answer: F_conf over every basin, from branch A + branch B products.

PRODUCTION. It assembles, it does not compute: every number it uses was produced by
branch A (`s0_A_pipeline.py`) or branch B (`s0_B_qha_trajectory*.py`), and the assembly
itself lives in `openqha.quasi_harmonic.ensemble` so that this file and the parameter
scans cannot drift apart.

WHAT IT PUTS TOGETHER
---------------------
    dG_i = dE_el,i  -  T*S_i           per basin, relative to the lowest
    F_conf = -kT ln sum_i exp(-dG_i / kT)

    dE_el,i   branch A, from the basin store
    T*S_i     branch B, from each basin's trajectories (mean over seeds)

WHY THIS IS A SEPARATE STEP AND NOT PART OF THE ANALYSIS
--------------------------------------------------------
`s0_B_qha_analyse.py` works on ONE species' trajectories and answers "is this basin's
entropy converged". It is per-trajectory and per-basin by design. The ensemble is the
level above: it needs branch A's relative energies, which that script never reads, and it
is where the two branches finally meet.

Keeping it separate also keeps the failure separable. "Basin 3's trajectory did not
converge" and "the ensemble is dominated by one basin" are different problems with
different fixes, and a single script reporting one number would hide whichever it was not
looking for.

    python scripts/production/s0_B_report_ensemble.py --species dsgdb9nsd_000035 --tag prod
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))

from openqha import config                                          # noqa: E402
from openqha.quasi_harmonic import basin_residence as br            # noqa: E402
from openqha.quasi_harmonic import ensemble, qha                    # noqa: E402
from openqha.store import basin_store                               # noqa: E402


def basin_electronic(rec):
    """Relative electronic energy per basin, kcal/mol, from branch A's record.

    The key is `relative_kcal`. It is looked up by name and a basin without it RAISES --
    it does not fall back to 0.0. An earlier version guessed three other names and, when
    all three missed, gave every basin zero: propanal's three basins (0.0000 / 0.8360 /
    0.8360) came out degenerate and F_conf read -0.6509 instead of -0.2354. There was no
    error, because a default of 0.0 is a perfectly valid relative energy.
    """
    out = []
    for i, b in enumerate(rec.get("basins", [])):
        if "relative_kcal" not in b:
            raise KeyError(
                "basin {} of this branch A record has no `relative_kcal`. Its keys are "
                "{}. Refusing to assume zero: a missing relative energy would make the "
                "basins look degenerate and F_conf would be wrong without saying so."
                .format(i, sorted(b)))
        out.append(float(b["relative_kcal"]))
    return out


def entropy_per_basin(species, tag, cfg, atoms_set="all"):
    """Mean T*S over the seeds of each basin, plus what the trajectories did.

    A basin with no trajectory returns None rather than 0. `ensemble` refuses to treat
    the two the same, because "we did not run it" and "its entropy is zero" are different
    statements and only one of them is ever true.
    """
    root = Path(config.runs_dir("qha", cfg)) / tag / species
    per_basin = {}
    for basin_dir in sorted(root.glob("basin*")):
        try:
            b = int(basin_dir.name.replace("basin", ""))
        except ValueError:
            continue
        vals, res_all, frames_seen = [], [], 0
        for seed_dir in sorted(basin_dir.glob("seed*")):
            f = seed_dir / "frames.npy"
            m = seed_dir / "meta.json"
            if not (f.exists() and m.exists()):
                continue
            meta = json.loads(m.read_text(encoding="utf-8"))
            frames = np.load(f)
            syms = meta["symbols"]
            masses = np.asarray(meta["masses_amu"], dtype=float)
            mask = (br.heavy_atom_mask(syms) if atoms_set == "heavy"
                    else np.ones(len(syms), bool))
            if len(frames) < 3 * len(syms):
                continue                       # below 3N the covariance cannot be formed
            rec = qha.analyse(frames[:, mask, :], masses[mask])
            vals.append(rec["entropy"]["TS_QH_kcal"])
            res_all.append(br.basin_residence(frames, syms))
            frames_seen = len(frames)
        if vals:
            per_basin[b] = dict(
                TS_kcal=float(np.mean(vals)), spread_kcal=float(np.std(vals)),
                n_seeds=len(vals), n_frames=frames_seen,
                distinct_crossings=int(sum(r["distinct_basin_crossings"] for r in res_all)),
                symmetry_crossings=int(sum(r["symmetry_equivalent_crossings"]
                                           for r in res_all)))
        else:
            per_basin[b] = None
    return per_basin, root


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species", required=True)
    ap.add_argument("--tag", default="prod")
    ap.add_argument("--atoms", default="all", choices=("all", "heavy", "both"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = config.load()
    temperature = config.temperature(cfg)
    rec_a = basin_store.read(args.species, tag=args.tag)
    if rec_a is None:
        raise SystemExit(
            "no branch A product for {} under tag {!r}. The ensemble is a sum over the "
            "basins branch A found; without it there is nothing to sum.\n"
            "  python scripts/production/s0_A_pipeline.py --species {} --tag {}"
            .format(args.species, args.tag, args.species, args.tag))

    e_rel = basin_electronic(rec_a)
    sets = ("all", "heavy") if args.atoms == "both" else (args.atoms,)

    print("=" * 88)
    print("ENSEMBLE FREE ENERGY -- {}  tag {}  T = {} K".format(
        args.species, args.tag, temperature))
    print("{} basin(s) from branch A".format(len(e_rel)))
    print("=" * 88)

    report = dict(species=args.species, tag=args.tag, temperature_K=temperature,
                  n_basins_branch_a=len(e_rel), results={})
    for atoms_set in sets:
        per_basin, root = entropy_per_basin(args.species, args.tag, cfg, atoms_set)
        basins = []
        for i, de in enumerate(e_rel):
            got = per_basin.get(i)
            basins.append(dict(rel_electronic_kcal=de,
                               TS_kcal=(got or {}).get("TS_kcal")))
        text, rec = ensemble.describe(basins, temperature)
        print()
        print("-- atoms: {} ".format(atoms_set) + "-" * (72 - len(atoms_set)))
        print(text)

        xing = sum((per_basin.get(i) or {}).get("distinct_crossings", 0)
                   for i in range(len(e_rel)))
        sym = sum((per_basin.get(i) or {}).get("symmetry_crossings", 0)
                  for i in range(len(e_rel)))
        print()
        print("basin crossings across all trajectories: {} distinct, {} "
              "symmetry-equivalent".format(xing, sym))
        if xing:
            print("  ** {} DISTINCT crossings: those trajectories left their basin and "
                  "their T*S is inflated. F_conf above is not an intra-basin "
                  "answer.".format(xing))
        elif sym:
            print("  symmetry-equivalent only: no conformer changed, but the atoms moved,")
            print("  so the all-atom covariance carries them. Compare --atoms heavy.")
        rec["per_basin_detail"] = {str(k): v for k, v in per_basin.items()}
        rec["distinct_crossings"] = xing
        rec["symmetry_crossings"] = sym
        report["results"][atoms_set] = rec
        report["trajectory_root"] = str(root)

    out = Path(args.out) if args.out else (
        ROOT / "analysis" / "qha" / args.tag / "{}_ensemble.json".format(args.species))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print()
    print("written {}".format(out))

    # A missing basin means the sum is over fewer terms than the molecule has, and the
    # answer is biased toward whatever was run. That is a failure, not a note.
    for atoms_set, rec in report["results"].items():
        if rec.get("n_electronic_only"):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
