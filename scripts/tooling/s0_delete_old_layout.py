#!/usr/bin/env python
"""Delete the pre-2026-09-14 result trees, after a plan that lists them.

TOOLING. Standard library only; runs on a login node.

    python scripts/tooling/s0_delete_old_layout.py            # --plan: list, remove nothing
    python scripts/tooling/s0_delete_old_layout.py --delete   # remove exactly what --plan listed

The ruling (user, 2026-09-14): the old layout -- three trees per run plus a per-job
scratch and its copy-back -- is not migrated into the molecule tree (ADR 0001); it is
deleted and the examples are re-run. This script is the only thing that deletes it, and
it deletes only what it has listed.

What counts as the old layout, by period:

    <repo>/analysis  <repo>/data/basins  <repo>/logs/node_local  <repo>/runs
    <home>/runs/openQHA                    off-cluster default; Tianhe before 2026-09-12
    <home>/runs/<jobid>[_cardK_rowJ]       Tianhe per-job scratch, 2026-09-12 .. 2026-09-14
    <home>/HDD_POOL/runs/openQHA           the tianhe_cpu default

Anything under the NEW root (S0_RUNS_ROOT, or --new-root) is never listed, whatever it is
called: a directory named `analysis` inside the molecule tree is not the old layout.
Slurm output beside `logs/node_local` stays -- it keeps its place until step 2 of the
layout change decides otherwise.

Run it on each machine that ran the old layout (the XYFS02 checkout, the XYAIFS00
checkout, this workstation); nothing here reaches across a filesystem it cannot see.
"""
import argparse
import os
import re
import shutil
import sys
from pathlib import Path


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


JOB_DIR = re.compile(r"^\d+(_card\d+(_row\d+)?)?$")


def old_trees(repo, home):
    """Every directory of the old layout that exists on this machine, in a fixed order."""
    out = []
    for rel in ("analysis", "data/basins", "logs/node_local", "runs"):
        p = repo / rel
        if p.is_dir():
            out.append(p)
    runs = home / "runs"
    if (runs / "openQHA").is_dir():
        out.append(runs / "openQHA")
    if runs.is_dir():
        for p in sorted(runs.iterdir()):
            if p.is_dir() and JOB_DIR.match(p.name):
                out.append(p)
    p = home / "HDD_POOL" / "runs" / "openQHA"
    if p.is_dir():
        out.append(p)
    return out


def under(p, root):
    if root is None:
        return False
    try:
        Path(p).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def measure(p):
    n, size = 0, 0
    for dirpath, _dirs, files in os.walk(p):
        for f in files:
            fp = os.path.join(dirpath, f)
            try:
                size += os.lstat(fp).st_size
            except OSError:
                pass
            n += 1
    return n, size


def human(b):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if b < 1024 or unit == "TB":
            return "{:.0f} {}".format(b, unit) if unit == "B" else "{:.1f} {}".format(b, unit)
        b /= 1024.0
    return "{} B".format(b)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default=None, help="the checkout (default: this one)")
    ap.add_argument("--home", default=None, help="the home directory (default: HOME)")
    ap.add_argument("--new-root", default=None,
                    help="the root of the molecule tree; never listed (default: S0_RUNS_ROOT)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--plan", action="store_true", help="list what would be removed (default)")
    g.add_argument("--delete", action="store_true", help="remove what --plan lists")
    args = ap.parse_args(argv)

    repo = Path(args.repo) if args.repo else _repo_root()
    home = Path(args.home) if args.home else Path(os.path.expanduser("~"))
    new_root = args.new_root or os.environ.get("S0_RUNS_ROOT")

    trees = [p for p in old_trees(repo, home) if not under(p, new_root)]
    mode = "delete" if args.delete else "plan"
    print("old layout on this machine   ({})".format(mode))
    print("  repo      {}".format(repo))
    print("  home      {}".format(home))
    print("  new root  {}   (never listed)".format(new_root or "unset"))
    print()
    if not trees:
        print("nothing to do: no tree of the old layout exists here.")
        return 0

    total_n, total_b = 0, 0
    for p in trees:
        n, b = measure(p)
        total_n += n
        total_b += b
        print("  {:>9}  {:>7} files   {}".format(human(b), n, p))
    print("  {:>9}  {:>7} files   total".format(human(total_b), total_n))
    print()

    if not args.delete:
        print("nothing removed (--plan). Run again with --delete to remove exactly these.")
        return 0

    removed = 0
    for p in trees:
        try:
            shutil.rmtree(p)
            removed += 1
            print("  removed  {}".format(p))
        except OSError as exc:
            print("  FAILED   {}: {}".format(p, exc), file=sys.stderr)
    print("removed {} of {} trees.".format(removed, len(trees)))
    return 0 if removed == len(trees) else 1


if __name__ == "__main__":
    raise SystemExit(main())
