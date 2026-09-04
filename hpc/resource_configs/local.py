"""Parsl resource configuration: this machine. Branch E step 0.

**This file is not filler.** D0-C-20 rules that the first thing done on a cluster
is a benchmark, not a production run, and debugging a Parsl workflow on a cluster
costs far more than debugging it here. So the whole branch-A chain is proved
locally first, and going to a cluster then changes exactly one file.

What "the same chain" means, precisely
--------------------------------------
The tasks, their inputs, their outputs and their acceptance criteria are identical
between this file and `deimos.py`. The only differences are how many workers there
are and who starts them. That is what makes branch E acceptance criterion 7
("local can run the whole chain, and it differs from the cluster by one file")
checkable rather than aspirational.

One worker per CREST job, and each CREST job takes its own threads
------------------------------------------------------------------
There are two nested levels of parallelism and they are not interchangeable:

  * BETWEEN molecules -- one Parsl worker per molecule. Each worker holds its own
    resident MACE server on its own socket, because a server holds one model behind
    one lock and sharing it would serialise the fan-out into a queue.
  * INSIDE one CREST job -- `threads` in hpc/configs/crest.json. This parallelises
    independent metadynamics runs, which is a real speed-up and not the same axis
    as BLAS threading (which is pinned to 1 in hpc/env/common.sh, measured).

So the machine's core count is divided, not multiplied: `max_workers * threads`
must not exceed the physical cores, or the measured cost of a molecule stops
meaning anything (D0-P1-12, defect 34).
"""
import os

#: Cores this machine has. Measured, not assumed.
N_CORES = os.cpu_count() or 4

#: CREST threads per molecule. Matches hpc/configs/crest.json; measured at 4 for
#: the cost numbers this repo quotes (285 s/species). resource_budget.
THREADS_PER_JOB = 4

#: Concurrent molecules. Kept so that workers * threads <= cores.
MAX_WORKERS = max(1, N_CORES // THREADS_PER_JOB)

#: Branch B: ONE thread per trajectory, not four.
#:
#: The two branches divide the cores differently, and the reason is measured. CREST
#: parallelises its own metadynamics runs across `threads`, so 4 threads per molecule buys
#: real speed. A branch B trajectory is a single serial chain of MACE force calls, and
#: MACE's thread scaling on a 10-atom molecule is 111/90/72/101 ms at 1/2/4/8 threads
#: (D0-P1-27, D0-P1-32): 4 threads gives 1.54x, and 8 is slower than 4. Four independent
#: trajectories on four cores give 4x. Branch B's parallelism is (basin x seed), of which
#: there are 3 seeds per basin by protocol, so there is always enough of it.
QHA_THREADS_PER_JOB = 1
QHA_MAX_WORKERS = max(1, N_CORES // QHA_THREADS_PER_JOB)


def config(max_workers=None, threads_per_job=THREADS_PER_JOB, run_dir=None,
           qha_max_workers=None):
    """Parsl Config for running branches A and B on this machine.

    Returns a `parsl.config.Config` with one executor per task class, which is the
    shape copied from ALF's `parsl_resource_configs/chicoma.py`. Adding branch B's
    executor did not disturb branch A's, which is the property that shape was chosen for.

    The two executors are NOT interchangeable: they differ in cores per worker, for the
    measured reason recorded at `QHA_THREADS_PER_JOB`. Submitting a branch B task to
    `openqha_crest` would run it with three idle cores held.

    Each executor is sized to the whole machine, because branch A and branch B run as
    separate stages and never at once. Running both drivers simultaneously would
    oversubscribe the cores, and every cost measured while that was happening would be
    meaningless -- the same trap as dividing a serial time by a process count
    (D0-P1-12, defect 34). `describe()` reports both sizes so the sum is visible.
    """
    from parsl.config import Config
    from parsl.executors import HighThroughputExecutor
    from parsl.providers import LocalProvider

    workers = max_workers or MAX_WORKERS
    qha_workers = qha_max_workers or QHA_MAX_WORKERS
    return Config(
        executors=[
            HighThroughputExecutor(
                label="openqha_crest",
                max_workers_per_node=workers,
                cores_per_worker=float(threads_per_job),
                cpu_affinity="none",
                provider=LocalProvider(init_blocks=1, min_blocks=1, max_blocks=1),
            ),
            HighThroughputExecutor(
                label="openqha_qha",
                max_workers_per_node=qha_workers,
                cores_per_worker=float(QHA_THREADS_PER_JOB),
                cpu_affinity="none",
                provider=LocalProvider(init_blocks=1, min_blocks=1, max_blocks=1),
            ),
        ],
        run_dir=run_dir or os.path.join(
            os.environ.get("S0_RUNS_ROOT",
                           os.path.expanduser("~/runs/openQHA")), "parsl"),
        # A retry silently hides a reproducible failure. Branch A failures are
        # reproducible (a molecule that aborts CREST aborts it again), so a retry
        # would only cost machine time and make the record ambiguous about how many
        # times a thing ran. Cluster configs raise this for pre-emption, which is
        # not reproducible; here it stays 0.
        retries=0,
        strategy="none",
    )


def describe():
    """The record of what this configuration is, for the product."""
    return dict(
        site="local",
        n_cores=N_CORES,
        max_workers=MAX_WORKERS,
        threads_per_job=THREADS_PER_JOB,
        qha_max_workers=QHA_MAX_WORKERS,
        qha_threads_per_job=QHA_THREADS_PER_JOB,
        executors=["openqha_crest", "openqha_qha"],
        oversubscribed=bool(MAX_WORKERS * THREADS_PER_JOB > N_CORES
                            or QHA_MAX_WORKERS * QHA_THREADS_PER_JOB > N_CORES),
        provider="parsl.providers.LocalProvider",
        retries=0,
        note=("Branch E step 0. Costs measured here are single-machine, "
              "uncontended numbers; they must NOT be divided by a process count "
              "to produce a cluster estimate (D0-P1-12, defect 34)."),
    )
