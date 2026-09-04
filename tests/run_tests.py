"""Run the test suite. Each test is a program that exits non-zero when it fails.

TOOLING. Branch D, plan_D section 2.

Why subprocesses rather than a framework
----------------------------------------
Every test here is already a standalone program with a `main()` returning an exit
code, and several must be run that way to mean anything -- `t_defect46_shebang`
exists precisely because a script behaved differently when executed than when
imported (D0-79). Importing tests into one process would erase the distinction this
suite is partly about. So each is launched exactly as a person would launch it.

Layout (plan_D section 2):

    unit/         pure functions and file-level checks; seconds, no engine
    integration/  needs an engine, CREST or ORCA; slow
    regression/   reproduces one numbered defect; file name carries the number
    data/         inputs and expected outputs for the regression tests, paired in the
                  style of MSTor's testrun/ and testo/

Only `unit/` runs by default, because it is the part that is fast enough to run on
every change. Pass `--all` for everything, or `--group regression` for one group.

Run::  python tests/run_tests.py
       python tests/run_tests.py --all
       python tests/run_tests.py --group regression
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _testlib import repo_root  # noqa: E402

ROOT = repo_root(__file__)
TESTS = ROOT / "tests"
GROUPS = ("unit", "integration", "regression")


def discover(groups):
    for g in groups:
        d = TESTS / g
        if not d.is_dir():
            continue
        for p in sorted(d.glob("t_*.py")):
            yield g, p


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="run every group, not just unit")
    ap.add_argument("--group", choices=GROUPS, help="run one group")
    ap.add_argument("--verbose", action="store_true", help="show each test's output")
    args = ap.parse_args()

    groups = (args.group,) if args.group else (GROUPS if args.all else ("unit",))
    tests = list(discover(groups))
    if not tests:
        print("no tests found in {}".format(", ".join(groups)))
        return 1

    print("=" * 92)
    print("running {} test(s) from {}".format(len(tests), ", ".join(groups)))
    print("=" * 92)
    failed = []
    for g, p in tests:
        t0 = time.time()
        r = subprocess.run([sys.executable, str(p)], cwd=str(ROOT),
                           capture_output=not args.verbose, text=True)
        dt = time.time() - t0
        mark = "pass" if r.returncode == 0 else "FAIL"
        print("  {:<4s} {:<10s} {:<44s} {:6.2f}s".format(mark, g, p.name, dt))
        if r.returncode != 0:
            failed.append((p, r))

    print()
    if failed:
        for p, r in failed:
            print("-" * 92)
            print("FAILED: {}".format(p.relative_to(ROOT)))
            print("-" * 92)
            if not args.verbose:
                tail = (r.stdout or "").splitlines()[-24:]
                print("\n".join(tail))
                if r.stderr:
                    print(r.stderr.strip()[-1500:])
        print("\n{} of {} test(s) failed".format(len(failed), len(tests)))
        return 1
    print("all {} test(s) passed".format(len(tests)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
