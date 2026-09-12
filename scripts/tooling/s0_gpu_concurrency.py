"""How many MACE trajectories does one card actually hold?

TOOLING. Runs `s0_probe_openmm_cuda.py --quiet` N times at once and reports what each
worker got, for several N. Produces no science: it sizes a campaign.

The question it exists for
--------------------------
One trajectory does not fill an A800. Measured on an45 2026-09-12, MACE-OFF23_medium,
10 atoms, float64 with Precision=double: 39.2 ms/step alone on the card, and 45.2/46.7
with two workers -- so the second worker cost ~17% and bought ~1.7x aggregate. Whether
that holds at 12 (the CPUs that come with one card under the fine-grained policy) or
16 is the number that decides how `hpc/resource_configs/tianhe_a.py` should lay out
branch B, and it cannot be inferred from the single-worker figure.

**Throughput is the answer, not latency.** A worker slowing from 39 to 60 ms/step is a
good trade if twelve of them run at once: that is 12/60 against 1/39, five times the
work. The per-worker column is here to show where it stops scaling, and the aggregate
column is what a campaign is sized from.

Why subprocesses and not threads
--------------------------------
Because that is how the work actually arrives: parsl gives each trajectory its own
process, and CUDA contexts in one process would share a scheduler in a way separate
processes do not. Measuring threads would measure something nobody runs.

Run::
    python scripts/tooling/s0_gpu_concurrency.py 1 2 4 8 12 16
    python scripts/tooling/s0_gpu_concurrency.py --dtype float32 --precision single 1 12
    python scripts/tooling/s0_gpu_concurrency.py --platform CPU 1 12    # the comparison

Add --mps to check whether the CUDA Multi-Process Service changes the picture; it has
to be started separately (`nvidia-cuda-mps-control -d`) and this flag only records that
you did, so the number carries its own provenance.
"""
import argparse
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


ROOT = _repo_root()
PROBE = ROOT / "scripts" / "tooling" / "s0_probe_openmm_cuda.py"

#: The quiet line starts with the tag and carries "<number> ms/step".
_MS = re.compile(r"([-+]?\d+\.\d+)\s+ms/step")


def worker_env(args):
    """One worker's share of the node, not all of it.

    **Every OpenMM CPU worker and every torch process takes all cores by default.**
    Measured while writing this, 16-thread laptop, float32, 50 steps: 29.3 ms/step
    alone, 41.9 with two workers, and **606** with four -- a 20x collapse that is pure
    thread thrashing, not the cost of sharing anything. On an45 the CPU probe reported
    Threads=112, so every published CPU figure here is a whole-node number and NOT what
    a worker in a 12-per-card layout would ever see.

    So a sweep has to pin. --threads 1 is the honest per-worker setting for the layout
    in hpc/resource_configs/tianhe_a.py (12 workers x 1 core per card); leaving it unset
    keeps the old behaviour and says so.
    """
    env = dict(os.environ)
    if args.threads:
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "OPENMM_CPU_THREADS"):
            env[var] = str(args.threads)
    return env


def run_batch(n, args):
    """Launch n probes at once; return (per-worker ms/step, batch wall seconds, failures)."""
    cmd = [sys.executable, str(PROBE), "--quiet", "--no-accuracy",
           "--platform", args.platform, "--dtype", args.dtype,
           "--steps", str(args.steps)]
    if args.precision:
        cmd += ["--precision", args.precision]
    env = worker_env(args)
    procs = []
    t0 = time.time()
    for i in range(n):
        procs.append(subprocess.Popen(
            cmd + ["--tag", "n{:02d}w{:02d}".format(n, i)],
            cwd=str(ROOT), env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, universal_newlines=True))
    ms, failures = [], []
    for proc in procs:
        out, _ = proc.communicate()
        line = (out or "").strip().splitlines()
        line = line[-1] if line else ""
        hit = _MS.search(line)
        if proc.returncode == 0 and hit:
            ms.append(float(hit.group(1)))
            print("    " + line)
        else:
            failures.append(out or "(no output)")
            print("    FAILED (exit {})".format(proc.returncode))
    return ms, time.time() - t0, failures


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("workers", nargs="+", type=int, help="worker counts to sweep")
    ap.add_argument("--platform", default="CUDA")
    ap.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    ap.add_argument("--precision", default=None,
                    choices=["single", "mixed", "double"],
                    help="default matches --dtype, as the probe does")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--threads", type=int, default=None,
                    help="cores per worker (sets OMP/MKL/OPENMM_CPU_THREADS). Use 1 for "
                         "the 12-workers-per-card layout; unset means every worker takes "
                         "the whole node, which measures thrashing, not sharing")
    ap.add_argument("--mps", action="store_true",
                    help="record that CUDA MPS was running; does not start it")
    args = ap.parse_args()

    print("probe      {}".format(PROBE.relative_to(ROOT)))
    print("platform   {}   dtype {}   precision {}   steps {}   MPS {}".format(
        args.platform, args.dtype, args.precision or "(matched to dtype)",
        args.steps, "yes" if args.mps else "no"))
    print("threads    {} per worker".format(
        args.threads if args.threads else "UNPINNED -- each worker takes the whole node"))
    if not args.threads:
        print("           **that is usually not what you want.** Measured on a 16-thread")
        print("           box: 29.3 ms/step at N=1, 41.9 at N=2, 606 at N=4 -- the last")
        print("           is thread thrashing, not the cost of sharing a card. Pass")
        print("           --threads 1 for a per-worker layout.")
    print()

    rows = []
    for n in args.workers:
        print("  N = {}".format(n))
        ms, wall, failures = run_batch(n, args)
        if not ms:
            print("    all {} worker(s) failed; first log follows".format(n))
            print(failures[0] if failures else "(nothing captured)")
            return 1
        rows.append(dict(n=n, ok=len(ms), ms_med=statistics.median(ms),
                         ms_min=min(ms), ms_max=max(ms), wall=wall,
                         failed=len(failures)))
        print()

    # Aggregate throughput is steps per second across all workers that finished. It is
    # the only column a campaign can be sized from; per-worker ms/step alone would say
    # concurrency is bad news when it is usually the opposite.
    print("=" * 92)
    print("  {:>3s}  {:>4s}  {:>9s}  {:>9s}  {:>9s}  {:>9s}  {:>12s}  {:>8s}".format(
        "N", "ok", "med ms/st", "min", "max", "batch s", "steps/s all", "speedup"))
    base = None
    for r in rows:
        thru = r["ok"] * args.steps / r["wall"] if r["wall"] > 0 else float("nan")
        base = base if base is not None else thru
        print("  {:>3d}  {:>4d}  {:>9.2f}  {:>9.2f}  {:>9.2f}  {:>9.2f}  {:>12.1f}  "
              "{:>7.2f}x".format(r["n"], r["ok"], r["ms_med"], r["ms_min"], r["ms_max"],
                                 r["wall"], thru, thru / base if base else float("nan")))
    print()
    print("  med ms/step is per worker and WILL rise with N; that is the cost of")
    print("  sharing. steps/s all is the throughput a campaign is sized from, and")
    print("  speedup is against the first row of this sweep -- so put 1 first.")
    worst = max(rows, key=lambda r: r["failed"])
    if worst["failed"]:
        print("  {} worker(s) failed at N={}; the table above counts only those that "
              "finished.".format(worst["failed"], worst["n"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
