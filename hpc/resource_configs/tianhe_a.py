"""Parsl resource configuration: TianheXY-A (login ln206). GPU session cluster.

THIS IS THE THIRD TIANHE CLUSTER. There are three, not two.
-----------------------------------------------------------
    TianheXY-C   tianhe_cpu.py   CPU   xyfree/mars/deimos/e9   64 cores, whole-node
    TianheXY-AI  tianhe_ai.py    GPU   hx/h100x/a100x/...      per GPU card, 80 GB
    TianheXY-A   this file       GPU   ai/temp                 56 cores, 1006 GB/node

Each has its own partitions, its own allocation policy and its own filesystem. Do not
copy settings between them; the `--exclusive` answer alone is already opposite between
the first two.

MEASURED 2026-09-05 (`sinfo` on ln206)
--------------------------------------
    PARTITION  AVAIL  TIMELIMIT  NODES(A/I/O/T)  CPUS   MEMORY
    ai         up     infinite   15/1/9/25       56     1030000 MB
    temp       up     2:00:00    1/0/0/1         56     1030000 MB

`ai` is the production partition; `temp` is a 2-hour queue on a single node and is what
a smoke test should use -- a short queue schedules sooner and cannot silently eat an
allocation.

**9 of the 25 `ai` nodes were in state O (other/down) at that reading.** Effective
capacity is smaller than the node count suggests; check `yhi` before planning a campaign
around 25.

CUDA: 12.3 IS THE CEILING HERE
------------------------------
`module avail` on this machine tops out at **CUDA/12.3** -- there is no CUDA/12.4, which
TianheXY-AI does have. The matched MPI is
`mpi/openmpi/5.0.0-gcc-11.4.0-cuda12.2`, and the newest matched NCCL is
`nccl/2.19.3-cuda-12.3`.

So `environment-tianhe-cuda.yml`, which pins `cuda-version=12.4`, **does not belong on
this cluster**. Use `CUDA_VERSION` below, which pins 12.3 to match. Loading a CUDA the
site does not have fails at `module load`; pinning a conda CUDA newer than the driver
fails at the first kernel launch, which is far worse.

WHAT IS NOT KNOWN ABOUT THIS MACHINE
------------------------------------
The manual supplied covers TianheXY-AI and TianheXY-C. For TianheXY-A there is no manual
here, so two questions are open and both change the job script:

  * is `-G/--gpus` mandatory, as on TianheXY-AI?
  * is `--exclusive` banned, as on TianheXY-AI, or expected, as on TianheXY-C?

`GPUS_PER_JOB` and `EXCLUSIVE` below carry the TianheXY-AI answer as the provisional one,
because this is a GPU cluster and that is the nearer neighbour -- but they are marked
assumed, `describe()` reports them as such, and **step 2 of docs/tianhe_runbook.md is
where to settle them.** A one-line `yhbatch` of `hostname` answers both.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import labels as _labels  # noqa: E402

# =========================================================================================
# SETTINGS
# =========================================================================================
ACCOUNT = None
PARTITION = "ai"               #: ai (production, no time limit) | temp (2 h, 1 node)

#: The node, from the site manual section 1.1 (and matching `sinfo`):
#:     2 x 28 = 56 cores, 8 GPU cards, 1024 GB RAM
#:     80 GB HBM2e per card, driver 535.104.12
CORES_PER_NODE = 56
GPUS_PER_NODE = 8
MEM_MB_PER_NODE = 1030000

#: Quota (Starlight, 2026-09-05): 10 running jobs, **5 nodes**. The node quota binds
#: first, so 5 allocations -- and at 8 cards each that is 40 GPUs, which is the whole
#: usable share of this cluster.
NODE_QUOTA = 5
JOB_QUOTA = 10

#: CUDA, pinned to what this cluster actually has. NOT 12.4 -- that exists on
#: TianheXY-AI and not here.
CUDA_VERSION = "12.3"
MPI_MODULE = "mpi/openmpi/5.0.0-gcc-11.4.0-cuda12.2"
NCCL_MODULE = "nccl/2.19.3-cuda-12.3"

#: USER RULING 2026-09-05: TianheXY-A follows TianheXY-AI's rules --
#: `--gpus` is passed, `--exclusive` is not.
#:
#: One tension recorded rather than hidden: this cluster's own manual says its nodes are
#: used exclusively, and `yhbatch --help` here DOES list `--exclusive`. So the flag is
#: probably accepted, unlike on TianheXY-AI where it is disabled. The ruling is followed;
#: if a submission is ever refused for want of exclusivity, EXCLUSIVE is the line to change.
GPUS_PER_JOB = GPUS_PER_NODE   #: 8 -- take the node's cards, since the node is exclusive
EXCLUSIVE = False

#: **Eight independent single-GPU workers per allocation, not one eight-way job.**
#:
#: Branch C fine-tunes MACE with the PHL loss. That model is small; a committee of eight
#: independent fine-tunes (different seeds, or a hyper-parameter sweep) is embarrassingly
#: parallel, needs no collective communication, and loses nothing to synchronisation. A
#: data-parallel eight-GPU job would spend the difference on gradient all-reduce and give
#: ONE model instead of eight.
#:
#: Switch to DDP only when one model genuinely does not fit on one 80 GB card, or when a
#: single fine-tune is too slow to be useful -- neither has been measured here, and until
#: it is, throughput beats latency because a committee is what branch C wants anyway.
WORKERS_PER_NODE = GPUS_PER_NODE           # 8
CPUS_PER_WORKER = CORES_PER_NODE // GPUS_PER_NODE   # 7 cores per card

NODES_PER_BLOCK = 1
#: 5 nodes x 8 cards = 40 concurrent single-GPU trainings, the whole quota.
MAX_BLOCKS = NODE_QUOTA
WALLTIME = "16:00:00"          #: `ai` has no limit; a bounded request still schedules sooner
# =========================================================================================


def _worker_init(here):
    """Modules first -- compute nodes are a minimal environment.

    CUDA and the MPI built against it are loaded together. An MPI built for a different
    CUDA is the same class of mistake as two BLAS builds in one environment, except that
    it fails at the first collective rather than at import.

    The SLURM_/PMI_ unset and the `mkdir -p` are rules 7 and 6 of this project's HPC
    skill: inherited task-layout variables make a child process try to relaunch itself
    through the scheduler, and Slurm opens `--output` before the script runs.
    """
    return "; ".join([
        "mkdir -p ${S0_RUNS_ROOT:-$HOME/HDD_POOL/runs/openQHA}/logs",
        "module purge 2>/dev/null || true",
        "module load anaconda3/2023.09 2>/dev/null || "
        "module load miniforge/24.7.1 2>/dev/null || true",
        "module load CUDA/{} 2>/dev/null || true".format(CUDA_VERSION),
        "module load {} 2>/dev/null || true".format(MPI_MODULE),
        "module load {} 2>/dev/null || true".format(NCCL_MODULE),
        "for v in $(env | awk -F= '{print $1}' | grep -E '^(PMI|SLURM)_'); do "
        "unset $v; done",
        "source {0}/env/common.sh".format(here),
        "source {0}/env/tianhe.sh".format(here),
        "openqha_report_env",
    ])


def config(partition=None, account=None, nodes_per_block=None, max_blocks=None,
           walltime=None, run_dir=None, worker_init=None, max_workers=None,
           gpus=None, exclusive=None, role="train"):
    """Parsl Config for TianheXY-A. Every argument defaults to SETTINGS."""
    from parsl.config import Config
    from parsl.executors import HighThroughputExecutor
    from parsl.launchers import SimpleLauncher

    from providers import TianheSlurmProvider

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gpus = GPUS_PER_JOB if gpus is None else gpus
    excl = EXCLUSIVE if exclusive is None else exclusive

    cfg = Config(
        executors=[
            HighThroughputExecutor(
                label=_labels.label(role),
                max_workers_per_node=int(max_workers or WORKERS_PER_NODE),
                cores_per_worker=float(CPUS_PER_WORKER),
                # ONE CARD PER WORKER. parsl assigns an accelerator per worker and sets
                # that worker's CUDA_VISIBLE_DEVICES itself.
                #
                # Do NOT try to do this from `worker_init`: that runs once per BLOCK,
                # before the pool starts, so every worker inherits the same value and all
                # 8 land on card 0. The job still runs and still gives correct numbers --
                # it is simply 8x slower, and reads as "the GPU is slow" rather than as a
                # placement bug. (Written that way first, on 2026-09-05.)
                available_accelerators=int(max_workers or WORKERS_PER_NODE),
                provider=TianheSlurmProvider(
                    partition or PARTITION,
                    account=account if account is not None else ACCOUNT,
                    nodes_per_block=nodes_per_block or NODES_PER_BLOCK,
                    init_blocks=0, min_blocks=0,
                    max_blocks=min(int(max_blocks or MAX_BLOCKS),
                                   NODE_QUOTA // int(nodes_per_block or NODES_PER_BLOCK),
                                   JOB_QUOTA),
                    # `--gpus=N` if this cluster wants it. Set GPUS_PER_JOB to None once
                    # it is known that it does not, rather than leaving a directive that
                    # might be refused.
                    scheduler_options=("#SBATCH --gpus={}".format(gpus)
                                       if gpus else ""),
                    exclusive=excl,
                    launcher=SimpleLauncher(),
                    worker_init=worker_init or _worker_init(here),
                    walltime=walltime or WALLTIME,
                    cmd_timeout=60,
                ),
            ),
        ],
        run_dir=run_dir or os.path.join(
            os.environ.get("S0_RUNS_ROOT",
                           os.path.expanduser("~/HDD_POOL/runs/openQHA")),
            "parsl"),
        retries=1,
        strategy="simple",
    )
    _labels.check(cfg, expect=_labels.label(role))
    return cfg


def describe():
    try:
        import providers
        commands = dict(providers.COMMANDS["tianhe"])
        confirmed = list(providers.TIANHE_CONFIRMED)
    except Exception:                                    # pragma: no cover
        commands, confirmed = {}, []
    return dict(
        site="tianhe_a", cluster="TianheXY-A", partition=PARTITION,
        partitions_available=["ai", "temp"],
        cores_per_node=CORES_PER_NODE, gpus_per_node=GPUS_PER_NODE,
        mem_mb_per_node=MEM_MB_PER_NODE,
        gpus_per_job=GPUS_PER_JOB, exclusive=EXCLUSIVE,
        workers_per_node=WORKERS_PER_NODE, cpus_per_worker=CPUS_PER_WORKER,
        max_blocks=MAX_BLOCKS, node_quota=NODE_QUOTA, job_quota=JOB_QUOTA,
        max_trainings_in_flight=MAX_BLOCKS * WORKERS_PER_NODE,
        strategy="8 independent single-GPU workers per node, not one 8-way DDP job",
        gpu_pinning="parsl available_accelerators (one card per worker)",
        walltime=WALLTIME,
        cuda=CUDA_VERSION, mpi_module=MPI_MODULE, nccl_module=NCCL_MODULE,
        labels=[_labels.label("train")],
        scheduler_commands=commands, scheduler_commands_confirmed=confirmed,
        notes=[
            "CUDA ceiling here is 12.3 -- there is no CUDA/12.4 on this cluster, "
            "so environment-tianhe-cuda.yml (which pins 12.4) does NOT belong here.",
            "`temp` is a 2-hour single-node queue: use it for smoke tests.",
            "9 of 25 `ai` nodes were in state O at the 2026-09-05 reading; effective "
            "capacity is below the node count.",
        ],
        assumptions=[
            "EXCLUSIVE=False follows the user ruling that this cluster uses TianheXY-AI's "
            "rules. Its own manual says nodes are exclusive and `yhbatch --help` lists "
            "--exclusive, so the flag is probably accepted here; revisit if a submission "
            "is refused for want of exclusivity.",
            "8 independent workers rather than DDP is a throughput choice, not a "
            "measurement: neither layout has been timed on this machine.",
            "Nothing has been submitted to any Tianhe cluster yet.",
        ],
        verified=False,
    )
