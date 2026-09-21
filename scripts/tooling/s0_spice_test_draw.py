"""A fixed draw of SPICE test frames: the forgetting line's yardstick (ticket 14).

TOOLING. Produces no scientific number of its own; it selects frames, once, by seed, and
writes the list of ids beside them so that the same draw can be made again anywhere.

    python scripts/tooling/s0_spice_test_draw.py --n 5000
    python scripts/tooling/s0_spice_test_draw.py --n 50 --source tests/data/spice_tiny/test_large_neut_all.xyz --out /tmp/x.xyz

Round-2 Q7 judges forgetting as the E/F error on MACE-OFF23's own test split, engine
against base. That comparison only means something if BOTH models see the same frames,
so the draw is made once and committed as a file of ids (`<out>.ids.dat`): the frames
themselves are large and belong to the SPICE release, the ids are ours.

The source is the test file of `data.training_set.settings()` (`S0_TRAINING_SETS_ROOT`
moves it). When it is absent this script says so and stops -- the judge then reports the
forgetting line as `-`, which is the honest answer, rather than a number from whatever
frames happened to be lying about.
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
from openqha.data import training_set                       # noqa: E402
from openqha.store import dat                               # noqa: E402

DEFAULT_OUT = ROOT / "data" / "training_sets" / "spice_test_{n}.extxyz"

ID_ROW = {
    "index": ("Integer", None, "the frame's position in the source file, 0-based"),
    "smiles": ("String", None, "the frame's SMILES, as the source gives it"),
    "config_type": ("String", None, "SPICE's config_type, or - "),
    "n_atoms": ("Integer", None, "atoms in the frame"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=5000, help="frames to draw")
    ap.add_argument("--seed", type=int, default=0, help="the draw's seed; it is written into the id table")
    ap.add_argument("--source", default=None, help="the SPICE test file (default: data.training_set's)")
    ap.add_argument("--out", default=None, help="the extxyz to write (default: data/training_sets/spice_test_<n>.extxyz)")
    ap.add_argument("--energy-key", default="REF_energy")
    ap.add_argument("--forces-key", default="REF_forces")
    args = ap.parse_args()

    import numpy as np
    from ase.io import read, write

    src = Path(args.source) if args.source else Path(training_set.settings()["test"])
    if not src.is_file():
        print("openQHA: no SPICE test file at {}.\n"
              "  The frames are not in this repository (the index is). Point S0_TRAINING_SETS_ROOT at the\n"
              "  MACE-OFF23 SPICE release ({}), or pass --source.\n"
              "  Until then the judge reports the forgetting line as '-' rather than a number from other frames."
              .format(src, training_set.settings()["doi"]), file=sys.stderr)
        return 2

    frames = read(str(src), index=":", format="extxyz")
    n = min(int(args.n), len(frames))
    idx = sorted(np.random.default_rng(args.seed).choice(len(frames), size=n, replace=False).tolist())
    out = Path(args.out) if args.out else Path(str(DEFAULT_OUT).format(n=n))
    out.parent.mkdir(parents=True, exist_ok=True)

    drawn, rows = [], []
    for i in idx:
        a = frames[i]
        if args.energy_key not in a.info:
            try:
                a.info[args.energy_key] = float(a.get_potential_energy())
            except Exception:                                # noqa: BLE001
                pass
        if args.forces_key not in a.arrays:
            try:
                a.arrays[args.forces_key] = a.get_forces()
            except Exception:                                # noqa: BLE001
                pass
        a.info["spice_index"] = int(i)
        drawn.append(a)
        rows.append(dict(index=int(i), smiles=str(a.info.get("smiles", "-")),
                         config_type=str(a.info.get("config_type", "-")), n_atoms=len(a)))
    write(str(out), drawn, format="extxyz")
    # the draw's provenance goes in the table's first rows' file header: `dat` writes one
    # unnamed section, so the source and the seed ride in a comment line above it
    ids = out.with_suffix(out.suffix + ".ids.dat")
    dat.write_table(ids, rows, list(ID_ROW), ID_ROW)
    header = "# source {}\n# seed {}\n# n_source {}\n# n_drawn {}\n".format(src, args.seed, len(frames), n)
    ids.write_text(header + ids.read_text(encoding="utf-8"), encoding="utf-8")
    print("source     {} ({} frames)".format(src, len(frames)))
    print("drawn      {} frames with seed {}".format(n, args.seed))
    print("written    {}\n           {}".format(out, ids))
    print("\nthe judge's forgetting line: 06_judge.py --spice-file {}".format(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
