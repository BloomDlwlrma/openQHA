"""One trajectory per (molecule, basin); the velocity seed is drawn at run time and recorded.

UNIT. Imports the trajectory driver (torch, openmm); a few seconds.

The seeding rule
----------------
After the campaign arithmetic in docs/branchB_seeds_and_length.md: production runs ONE
trajectory per (molecule, basin), and the seed follows OpenMM's own convention for
randomNumberSeed=0 -- a fresh one per run -- with one addition: the number drawn is
written to meta.json, so a run stays re-derivable.

What is asserted
----------------
  A. choose_seed(0, ...) draws a non-zero seed, says so, and two draws differ;
     choose_seed(seed0 != 0, ...) is the old pure function seed0 + 1000*basin + index.
  B. segments_record: a trajectory run in one go has one segment; a resume appends a
     second with its own seed and frame range; an old record without `segments` is
     folded in as the first segment. (A resume restarts from the relaxed geometry with
     new velocities and appends, so the file holds two independently started segments;
     the record has to say so.)
  C. The defaults: --seeds 1 and --seed0 0 in both drivers, SEEDS=1 in
     chain_body, trajectories_per_basin 1 in the protocol file.
"""
import importlib.util
import re
import sys
from pathlib import Path


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))
FAIL = []


def _load(rel):
    path = ROOT / rel
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_choose_seed(drv):
    print("A. choose_seed")
    s1, how1 = drv.choose_seed(0, 0, 0)
    s2, _ = drv.choose_seed(0, 0, 0)
    ok = s1 > 0 and s2 > 0 and s1 != s2 and "drawn" in how1
    print("   {}  seed0=0 -> {} and {} ({})".format("ok  " if ok else "FAIL", s1, s2, how1))
    if not ok:
        FAIL.append("seed0=0 did not draw two distinct non-zero seeds")
    s3, how3 = drv.choose_seed(20260903, 1, 2)
    ok = s3 == 20260903 + 1000 + 2 and "seed0 + 1000*basin" in how3
    print("   {}  seed0=20260903, basin 1, index 2 -> {} ({})".format(
        "ok  " if ok else "FAIL", s3, how3))
    if not ok:
        FAIL.append("fixed seed formula changed: {}".format(s3))


def check_segments(drv):
    print("\nB. segments_record")
    one = drv.segments_record(None, 7, dict(n_frames_already_on_disk=0, n_frames=500))
    ok = one == [dict(seed=7, frames_from=0, frames_to=500)]
    print("   {}  one go -> {}".format("ok  " if ok else "FAIL", one))
    if not ok:
        FAIL.append("single-run segments: {}".format(one))
    prev = dict(seed=7, production=dict(n_frames=200), segments=one[:0] or None)
    prev["segments"] = None
    two = drv.segments_record(prev, 9, dict(n_frames_already_on_disk=200, n_frames=500))
    ok = two == [dict(seed=7, frames_from=0, frames_to=200),
                 dict(seed=9, frames_from=200, frames_to=500)]
    print("   {}  resume of an old record -> {}".format("ok  " if ok else "FAIL", two))
    if not ok:
        FAIL.append("resume segments: {}".format(two))
    three = drv.segments_record(dict(segments=two), 11,
                                dict(n_frames_already_on_disk=500, n_frames=600))
    ok = len(three) == 3 and three[-1] == dict(seed=11, frames_from=500, frames_to=600)
    print("   {}  second resume -> {} segments, last {}".format(
        "ok  " if ok else "FAIL", len(three), three[-1]))
    if not ok:
        FAIL.append("second resume segments: {}".format(three))


def check_defaults():
    print("\nC. the defaults")
    checks = [
        ("scripts/production/s0_B_qha_trajectory_openmm.py",
         r'add_argument\("--seeds", type=int, default=1\)', "--seeds default 1"),
        ("scripts/production/s0_B_qha_trajectory_openmm.py",
         r'add_argument\("--seed0", type=int, default=0\)', "--seed0 default 0"),
        ("scripts/production/s0_E_branchB_parsl.py",
         r'add_argument\("--seeds", type=int, default=1', "--seeds default 1 (parsl)"),
        ("scripts/production/s0_E_branchB_parsl.py",
         r'add_argument\("--seed0", type=int, default=0', "--seed0 default 0 (parsl)"),
        ("examples/chain_body.sh", r'^SEEDS="\$\{SEEDS:-1\}"', "SEEDS default 1 (chain_body)"),
        ("configs/branchB_protocol.yaml", r'^\s*trajectories_per_basin: 1', "protocol: 1 per basin"),
    ]
    for rel, pat, label in checks:
        text = (ROOT / rel).read_text(encoding="utf-8")
        ok = re.search(pat, text, re.M) is not None
        print("   {}  {:<34} {}".format("ok  " if ok else "FAIL", label, rel))
        if not ok:
            FAIL.append("{}: {}".format(rel, label))


def main():
    drv = _load("scripts/production/s0_B_qha_trajectory_openmm.py")
    check_choose_seed(drv)
    check_segments(drv)
    check_defaults()
    print()
    if FAIL:
        print("{} problem(s):".format(len(FAIL)))
        for f in FAIL:
            print("  - " + f)
        return 1
    print("one trajectory per basin, its seed drawn and recorded, its segments on record")
    return 0


if __name__ == "__main__":
    sys.exit(main())
