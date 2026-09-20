"""Workflow hessian_learning, step 02: the Frame set of every selected molecule at the
engine level (CONTEXT.md "Frame set"; ticket 02 of the Hessian-learning set).

PRODUCTION. One Calculation per molecule: `openqha.data.frames.generate` -> one extxyz
per generator under `<molecule>/frames/` and the Record `frames/frames.{out,toml}`.
Nothing is labelled at the reference level here (that is step 03); nothing is moved
by MACE (the basins are branch A's; MACE only evaluates and filters).

    python workflows/hessian_learning/02_frames.py --tag rings --species dsgdb9nsd_000048
    python workflows/hessian_learning/02_frames.py --tag rings --all
    python workflows/hessian_learning/02_frames.py --tag rings --all --n-displaced 4 --force
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
from openqha.data import frames                              # noqa: E402
from openqha.store import basins as basin_reader, layout     # noqa: E402


def molecules_under(tag, cfg=None):
    """Every molecule directory under `tag` with a branch-A basin list (login-node cheap)."""
    base = Path(config.runs_root(cfg)) / str(tag)
    return sorted(p.parents[2] for p in base.glob("*/mace/basin00/basin.extxyz"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--species", help="one QM9 index")
    g.add_argument("--all", action="store_true", help="every molecule under the tag with basins")
    ap.add_argument("--n-displaced", type=int, default=frames.N_DISPLACED)
    ap.add_argument("--temperature", type=float, default=frames.TEMPERATURE_K)
    ap.add_argument("--max-rms", type=float, default=frames.MAX_RMS_A)
    ap.add_argument("--distribution", default=frames.DISTRIBUTION, choices=("quantum", "classical"),
                    help="harmonic draw of the displaced frames (round-2 Q14, ruled classical 2026-09-18)")
    ap.add_argument("--engine", default=None, help="registered engine name (default: the production default)")
    ap.add_argument("--force", action="store_true", help="rebuild a Frame set whose Record exists")
    args = ap.parse_args()

    mols = [basin_reader.molecule_for(args.species, args.tag)] if args.species else molecules_under(args.tag)
    if not mols:
        raise SystemExit("no molecule with basins under tag {!r}".format(args.tag))
    done = skipped = failed = 0
    for mol in mols:
        rec = layout.frames_dir(mol) / (frames.STEP + ".toml")
        if rec.is_file() and not args.force:
            skipped += 1
            continue
        try:
            out = frames.generate(mol, n_displaced=args.n_displaced, temperature_K=args.temperature,
                                  max_rms_A=args.max_rms, engine_name=args.engine, distribution=args.distribution)
        except Exception as exc:                                 # one molecule must not stop the Batch
            failed += 1
            print("FAILED  {}  {}: {}".format(mol.name, type(exc).__name__, str(exc)[:200]), flush=True)
            continue
        done += 1
        i = out["info"]
        print("{}  basins {}  frames {} (dropped {})  {}  {:.0f} s".format(
            i["QM9_INDEX"], i["N_BASINS"], i["N_FRAMES"], i["N_DROPPED"],
            "  ".join("{}={}".format(r["GENERATOR"], r["N_FRAMES"]) for r in out["generators"]), i["SECONDS"]), flush=True)
    print("frames: {} built, {} skipped (Record present), {} failed".format(done, skipped, failed))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
