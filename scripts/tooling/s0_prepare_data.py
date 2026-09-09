"""Put the QM9 data into stage 0 own `data/` directory.

TOOLING. Puts QM9 in place under data/. It moves data; it computes nothing.

**Why this script exists**: since 2026-08-28 stage 0 is an independent open-source
framework and reads nothing from the stage 1 / stage 2 directories. But the raw QM9 data
is too large (index 119 MB, geometries 224 MB) and **does not go into git**, so this script puts it in place once.

Only the **reference geometries of the 7 target species** are distributed with the
repository (`data/reference-geometries/`, 11 KB), which is what makes **stage 0's core
deliverable (package 2) reproducible with no external data**; only the large package 1 census needs this script.

Usage:
    python scripts/tooling/s0_prepare_data.py --from <a QM9 directory>     # copy
    python scripts/tooling/s0_prepare_data.py --from <dir> --link          # symlink instead, to save disk
    python scripts/tooling/s0_prepare_data.py --check                      # report the current state only

The directory `--from` points at should contain:
    index_Chem_composition.csv
    xyz_files/dsgdb9nsd_XXXXXX.xyz

You can also skip this script entirely and set the environment variables:
    export S0_QM9_ROOT=/path/to/qm9
"""
import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

def _repo_root():
    """Directory holding the openQHA package, found by walking up from this file.

    Depth-independent on purpose: this file keeps working wherever under the
    repository it is moved to. The earlier move into `scripts/_superseded/` broke
    every `parents[1]` in the moved files silently, which is what this replaces.
    """
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


sys.path.insert(0, str(_repo_root()))

from openqha import S0_ROOT, config


def sha256_head(path, n_bytes=1 << 20):
    """Digest of the first 1 MB -- digesting all 119 MB is too slow, and 1 MB is enough to notice that the file was swapped."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read(n_bytes))
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", default=None, help="directory holding the QM9 data")
    ap.add_argument("--link", action="store_true", help="create symlinks instead of copying")
    ap.add_argument("--check", action="store_true", help="report the current state only; touch nothing")
    args = ap.parse_args()

    cfg = config.load()
    root = config.qm9_root(cfg)
    index_name = cfg["data"]["qm9_index_csv"]
    xyz_name = cfg["data"]["qm9_xyz_dir"]
    vend = S0_ROOT / cfg["data"]["vendored_reference_geometries"]

    print("=" * 92)
    print("stage 0 data preparation")
    print("=" * 92)
    print("configuration  {}".format(cfg["_path"]))
    print("target dir     {}{}".format(root, "   (from S0_QM9_ROOT)"
                                      if os.environ.get("S0_QM9_ROOT") else ""))
    print("with the repo  {}  ({} file(s))".format(
        vend.relative_to(S0_ROOT), len(list(vend.glob("*.xyz"))) if vend.exists() else 0))
    print()

    if args.check or not args.src:
        idx = root / index_name
        xyz = root / xyz_name
        print("index table    {}  {}".format(idx, "present" if idx.exists() else "**MISSING**"))
        if idx.exists():
            print("               {:.1f} MB, digest of the first 1 MB {}".format(
                idx.stat().st_size / 1048576, sha256_head(idx)[:16]))
        print("geometry dir   {}  {}".format(xyz, "present" if xyz.exists() else "**MISSING**"))
        if xyz.exists():
            n = sum(1 for _ in xyz.glob("dsgdb9nsd_*.xyz"))
            print("               {} xyz file(s)".format(n))
        print()
        if not args.src:
            if idx.exists() and xyz.exists():
                print("The full data is in place; package 1 can run.")
            else:
                print("The full data is NOT in place -- **package 2 can still run** (it uses only the 7 geometries shipped with the repository),")
                print("but the large package 1 census needs this first:")
                print("    python scripts/tooling/s0_prepare_data.py --from <a QM9 directory>")
            return

    src = Path(args.src)
    if not src.exists():
        raise FileNotFoundError("source directory does not exist: {}".format(src))
    s_idx, s_xyz = src / index_name, src / xyz_name
    for p in (s_idx, s_xyz):
        if not p.exists():
            raise FileNotFoundError("{} is missing from the source directory: {}".format(p.name, p))

    root.mkdir(parents=True, exist_ok=True)
    d_idx, d_xyz = root / index_name, root / xyz_name

    if args.link:
        for s, d in ((s_idx, d_idx), (s_xyz, d_xyz)):
            if d.exists() or d.is_symlink():
                print("already present, skipping: {}".format(d))
                continue
            os.symlink(s.resolve(), d)
            print("link {} -> {}".format(d, s.resolve()))
    else:
        if d_idx.exists():
            print("already present, skipping: {}".format(d_idx))
        else:
            print("copying the index table, {:.1f} MB ...".format(s_idx.stat().st_size / 1048576))
            shutil.copy2(s_idx, d_idx)
        if d_xyz.exists():
            print("already present, skipping: {}".format(d_xyz))
        else:
            n = sum(1 for _ in s_xyz.glob("dsgdb9nsd_*.xyz"))
            print("copying {} xyz file(s) ...".format(n))
            shutil.copytree(s_xyz, d_xyz)

    print()
    print("provenance:")
    print("   source dir     {}".format(src.resolve()))
    print("   index digest   {} (first 1 MB)".format(sha256_head(d_idx)[:32]))
    print("   dataset        {}".format(cfg["data"]["upstream"]["dataset"]))
    print()
    print("Done. `data/qm9/` is in `data/.gitignore` and will not enter version control.")


if __name__ == "__main__":
    main()
