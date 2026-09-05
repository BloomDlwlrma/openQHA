"""Parsl resource configuration: Tianhe (TianheXY-AI), CPU partition. Branch A.

WRITTEN, NOT VERIFIED
---------------------
Nothing in this repository has ever been submitted to Tianhe. This file is built from the
site facts recorded in `configs/cluster_tianhe.yaml` (supplied by the user, 2026-08-31)
and from the deimos configuration it mirrors; **every number that is a guess is marked as
one**, and `describe()` reports which is which so a product can never present an assumed
value as a measured one.

Read `docs/tianhe_runbook.md` before using it. In particular: run `providers.preflight()`
on the login node first, because three of the five scheduler command names here are still
assumptions.

WHY BRANCH A GOES ON A CPU PARTITION
-----------------------------------
Branch A is CREST with a GFN2-xTB workhorse plus MACE refinement. GFN2-xTB never touches
a GPU, and the MACE calls are single 10-20 atom structures, which cannot fill one (D0-56).
Tianhe's `h100x` bills by the whole card and gives 14 CPUs with it, so branch A there
would burn an H100 to do CPU work and starve the CPU work while doing it.

**Branch A is a CPU workload parallel over MOLECULES.** Branch C's training is the part
that wants a GPU, and that is a different config.

THE ARITHMETIC, FROM MEASURED NUMBERS ONLY
------------------------------------------
    285 s per species        measured, 4 threads, uncontended, on this project's
                             workstation (S0-A-8) -- NOT on Tianhe
    4 threads per molecule   the condition that number was measured under

`CORES_PER_NODE` below is an ASSUMPTION. The site announcement quoted in
`configs/cluster_tianhe.yaml` gives CPU counts for the GPU partitions (14 CPUs per H100)
and says nothing about the CPU partition's node size. Until someone runs `lscpu` on a
compute node, the worker count derived from it is a guess and is reported as one.

**A per-node capacity is not a speed-up.** 285 s is a single uncontended run on different
hardware; concurrent jobs share memory bandwidth and the cost per molecule will be higher.
Branch E acceptance criterion 5 requires wall clock, single-job time and slot
extrapolation to be reported as three separate numbers so that nobody divides one by
another later (defects 34, 56).
"""
import os

#: Physical cores per CPU-partition node. **ASSUMED, NOT MEASURED.** Nothing in the site
#: information records it. Override with `--max-workers`, or fix this line once `lscpu`
#: on a compute node has been read.
CORES_PER_NODE = 64
CORES_PER_NODE_IS_ASSUMED = True

#: CREST threads per molecule -- the condition the 285 s/species cost was measured under.
#: Changing it invalidates that number. resource_budget.
THREADS_PER_JOB = 4

#: Concurrent molecules per node: cores / threads, with no oversubscription.
WORKERS_PER_NODE = CORES_PER_NODE // THREADS_PER_JOB

#: Job walltime. The site allows up to 7 days per job (user, 2026-08-31). This is
#: deliberately far below that: a long job that is killed loses everything still in
#: flight, and branch A writes one product per molecule, so short jobs lose at most one
#: molecule each. Raise it only after a first campaign has shown the queue is not the
#: bottleneck.
WALLTIME = "11:24:00"          # 95% of a 12 h slot, the deimos convention

#: The partition name. **ASSUMED.** `configs/cluster_tianhe.yaml` records five GPU
#: partitions (hx, h100x, a100x, a800x, v100x) and NO CPU partition name, because the
#: site information the user supplied does not contain one. Pass `--partition` explicitly;
#: this default exists so the failure is a scheduler error naming the partition, not a
#: silent submission to a GPU queue.
PARTITION = "cpu"
PARTITION_IS_ASSUMED = True


def config(partition=None, account=None, nodes_per_block=1, max_blocks=4,
           walltime=WALLTIME, run_dir=None, worker_init=None, max_workers=None):
    """Parsl Config for branch A on Tianhe's CPU partition.

    `worker_init` sources `hpc/env/common.sh` so every worker gets the measured
    environment settings -- single-threaded workers, the OpenBLAS fix, the site's
    `ulimit -l unlimited` and `GLEX_USE_ZC_RNDV=0`. A worker that skips them is a
    different machine.

    NOTE the provider is `TianheSlurmProvider`, not `SlurmProvider`: the directives are
    `#SBATCH` (the site's own scripts use them) but the front-end commands are renamed.
    Three of those five names are unverified, and since 2026-09-05 a failing STATUS query
    raises instead of leaving every job to look "pending" for ever.
    """
    from parsl.config import Config
    from parsl.executors import HighThroughputExecutor
    from parsl.launchers import SimpleLauncher

    from providers import TianheSlurmProvider

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if worker_init is None:
        worker_init = ("source {0}/env/common.sh; "
                       "source {0}/env/tianhe.sh; "
                       "openqha_report_env").format(here)

    return Config(
        executors=[
            HighThroughputExecutor(
                label="openqha_crest_tianhe",
                max_workers_per_node=int(max_workers or WORKERS_PER_NODE),
                cores_per_worker=float(THREADS_PER_JOB),
                provider=TianheSlurmProvider(
                    partition or PARTITION,
                    account=account,
                    nodes_per_block=nodes_per_block,
                    # init_blocks = 0: ask for nothing until there is work.
                    # min_blocks = 0: give an idle allocation back rather than hold it --
                    # a held node is charged whether or not it computes.
                    init_blocks=0, min_blocks=0, max_blocks=max_blocks,
                    # No scheduler_options. parsl already emits
                    # "#SBATCH --ntasks-per-node" from tasks_per_node, and repeating
                    # it here renders the directive TWICE -- seen in the first render,
                    # 2026-09-05. Slurm takes the last one, so it is harmless in
                    # effect; a duplicated directive is still exactly what the render
                    # check is for.
                    scheduler_options="",
                    # SimpleLauncher, not the site's `yhrun`: one worker pool per node
                    # which forks its own workers. A launcher that places one task per
                    # core would start one full worker pool per core.
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
        # A cluster job can die for reasons that are not the molecule's fault
        # (pre-emption, a node going away). One retry covers that without hiding a
        # reproducible failure, which would need more than one.
        retries=1,
        strategy="simple",
    )


def describe():
    """What this configuration claims, and which parts of it are assumptions.

    Carried into the product record. A value that was guessed must never reach a report
    looking like one that was measured.
    """
    try:
        import providers
        commands = dict(providers.COMMANDS["tianhe"])
        confirmed = list(providers.TIANHE_CONFIRMED)
    except Exception:                                    # pragma: no cover
        commands, confirmed = {}, []
    return dict(
        site="tianhe",
        partition=PARTITION,
        cores_per_node=CORES_PER_NODE,
        threads_per_job=THREADS_PER_JOB,
        workers_per_node=WORKERS_PER_NODE,
        walltime=WALLTIME,
        scheduler_commands=commands,
        scheduler_commands_confirmed=confirmed,
        scheduler_commands_assumed=[k for k in commands if k not in confirmed],
        assumptions=[
            "CORES_PER_NODE={} is ASSUMED: the site information records CPU counts for "
            "the GPU partitions only. Read `lscpu` on a compute node and fix it."
            .format(CORES_PER_NODE),
            "PARTITION={!r} is ASSUMED: no CPU partition name appears in the site "
            "information. Pass --partition explicitly.".format(PARTITION),
            "285 s/species was measured on this project's workstation, NOT on Tianhe. "
            "It is a planning figure here, not a prediction.",
            "Nothing in this repository has ever been submitted to Tianhe.",
        ],
        verified=False,
    )
