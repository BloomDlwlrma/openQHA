"""Parsl resource configuration: deimos, 64-core CPU nodes. Branch A production.

Why branch A goes here and not on a GPU partition
-------------------------------------------------
Branch A's workload is CREST with a GFN2-xTB workhorse plus MACE refinement calls.
GFN2-xTB does not use a GPU at all, and the MACE calls are single 10-20 atom
structures, which do not fill one (D0-56). Tianhe's `h100x` partition bills by the
whole card and provides only 14 CPUs with it, so running branch A there would burn
an H100 to do CPU work and starve the CPU work while doing it.

**Branch A is a CPU workload parallel over MOLECULES.** That is what this file
describes. The GPU configuration (`tianhe_h100x.py`) is for branch C's training,
and is not part of the branch-A slice.

The arithmetic, from measured numbers only
------------------------------------------
    285 s per species          measured, 4 threads, uncontended (S0-A-8)
    4 threads per molecule     the condition that number was measured under
    64 cores per node          -> 16 molecules per node at a time

**16 x is a node capacity, not a speed-up.** The measured 285 s is a single
uncontended run; 16 concurrent jobs share memory bandwidth and the cost per
molecule will be higher. Branch E acceptance criterion 5 requires the wall clock,
the single-job time and the slot extrapolation to be reported as three separate
numbers, precisely so that nobody divides one by the other later (defect 34, 56).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import labels as _labels  # noqa: E402

#: Physical cores per deimos node (D0-C-26).
CORES_PER_NODE = 64

#: CREST threads per molecule -- the condition the 285 s/species cost was measured
#: under. Changing it invalidates that number. resource_budget.
THREADS_PER_JOB = 4

#: Concurrent molecules per node. cores / threads, no oversubscription.
WORKERS_PER_NODE = CORES_PER_NODE // THREADS_PER_JOB

#: Branch B: ONE core per trajectory. Same measured reason as in `local.py` -- a
#: quasi-harmonic trajectory is a serial chain of MACE force calls, and MACE's thread
#: scaling on a 10-atom molecule is 111/90/72/101 ms at 1/2/4/8 threads (D0-P1-27,
#: D0-P1-32). Four threads buy 1.54x; four independent trajectories buy 4x. Branch B has
#: (basins x 3 seeds) of independent work per species, so it never runs short of it.
QHA_THREADS_PER_JOB = 1
QHA_WORKERS_PER_NODE = CORES_PER_NODE // QHA_THREADS_PER_JOB

#: Job walltime. Set to 95% of the queue limit so the job finishes its own work and
#: writes its results out, instead of being killed with them still in memory
#: (D0-C-25). The pipeline is per-molecule and every molecule writes on completion,
#: so a kill costs at most one molecule -- but only if nothing buffers.
WALLTIME = "11:24:00"          # 95% of a 12 h queue limit

#: Branch B's own wall-clock budget inside the job, in seconds: 90% of WALLTIME. The
#: trajectory driver stops at a chunk boundary when it passes this and leaves a resumable
#: state, so a trajectory longer than one queue slot costs restarts and never results
#: (branch E acceptance criterion 2). It is deliberately tighter than the job walltime:
#: the remaining 10% is for the final flush and the analysis handoff.
QHA_WALL_BUDGET_S = int(0.90 * (11 * 3600 + 24 * 60))


def config(partition="cpu", account=None, nodes_per_block=1, max_blocks=8,
           walltime=WALLTIME, run_dir=None, worker_init=None):
    """Parsl Config for branches A and B on deimos.

    `worker_init` defaults to sourcing hpc/env/common.sh and hpc/env/deimos.sh, so
    the three measured environment settings (single-threaded workers, the OpenBLAS
    defect fix, the site's ulimit/GLEX lines) apply to every worker. They are the
    reason those files exist; a worker that skips them is a different machine.
    """
    from parsl.config import Config
    from parsl.executors import HighThroughputExecutor
    from parsl.providers import SlurmProvider
    from parsl.launchers import SimpleLauncher

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if worker_init is None:
        worker_init = ("source {0}/env/common.sh; source {0}/env/deimos.sh; "
                       "openqha_report_env").format(here)

    return Config(
        executors=[
            HighThroughputExecutor(
                label=_labels.label("crest"),
                max_workers_per_node=WORKERS_PER_NODE,
                cores_per_worker=float(THREADS_PER_JOB),
                provider=SlurmProvider(
                    partition,
                    account=account,
                    nodes_per_block=nodes_per_block,
                    # init_blocks = 0 so nothing is asked for until there is work.
                    # min_blocks = 0 so an idle allocation is given back rather than
                    # held: this repo is charged for held nodes whether or not they
                    # compute.
                    init_blocks=0, min_blocks=0, max_blocks=max_blocks,
                    scheduler_options="#SBATCH --ntasks-per-node=1",
                    # SimpleLauncher, not SrunLauncher: one worker pool per node
                    # that then forks its own workers. srun would place one task per
                    # core and each of those would try to be a full worker pool.
                    launcher=SimpleLauncher(),
                    worker_init=worker_init,
                    walltime=walltime,
                    cmd_timeout=60,
                ),
            ),
        ],
        run_dir=run_dir or os.path.join(
            os.environ.get("S0_RUNS_ROOT", os.path.expanduser("~/runs/openQHA")),
            "parsl"),
        # Unlike local.py: a cluster job can die for reasons that are not the
        # molecule's fault (pre-emption, a node going away). One retry covers that
        # without hiding a reproducible failure, which would need more than one.
        retries=1,
        strategy="simple",
    )


def describe():
    return dict(
        site="deimos",
        cores_per_node=CORES_PER_NODE,
        workers_per_node=WORKERS_PER_NODE,
        threads_per_job=THREADS_PER_JOB,
        walltime=WALLTIME,
        provider="parsl.providers.SlurmProvider",
        retries=1,
        cost_basis=("285 s/species measured uncontended at 4 threads (S0-A-8). "
                    "Node capacity is 16 concurrent molecules; that is a capacity, "
                    "NOT a speed-up, and the per-molecule cost under contention has "
                    "not been measured."),
    )
