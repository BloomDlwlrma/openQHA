"""Parsl resource configuration: TianheXY-A (login ln206). The GPU cluster we prefer.

THIS IS THE THIRD TIANHE CLUSTER. There are three, not two.
-----------------------------------------------------------
    TianheXY-C   tianhe_cpu.py   CPU   debug / deimos          64 cores, whole-node
    TianheXY-AI  tianhe_ai.py    GPU   hx/h100x/a100x/...      per GPU card, 80 GB
    TianheXY-A   this file       GPU   temp / ai               56 cores, 8 cards/node

Each has its own partitions, its own allocation policy and its own filesystem. Do not
copy settings between them; the `--exclusive` answer alone is already opposite between
the first and the other two.

**Prefer this cluster for GPU work** (user ruling 2026-09-07). Eight cards per allocation
against TianheXY-AI's one is the whole reason.

TWO PARTITIONS, TWO WALLTIMES (user ruling 2026-09-07)
------------------------------------------------------
    temp    30 minutes    smoke test. `sinfo` gives the queue a 2 h ceiling and one node.
    ai      7 days        production.

`--debug` on the drivers selects the first pair. **There is no `--dry-run` gate before
production**: a 30-minute real job on `temp` tests what a rendered plan cannot -- that
CUDA/12.3 loads, that conda activates on a compute node, that the weights hash matches
there, and that a card is actually visible to the worker that was given it.

MEASURED 2026-09-05 (`sinfo` on ln206)
--------------------------------------
    PARTITION  AVAIL  TIMELIMIT  NODES(A/I/O/T)  CPUS   MEMORY
    ai         up     infinite   15/1/9/25       56     1030000 MB
    temp       up     2:00:00    1/0/0/1         56     1030000 MB

**9 of the 25 `ai` nodes were in state O (other/down) at that reading.** Effective
capacity is smaller than the node count suggests; check `yhi` before planning a campaign
around 25.

WHAT RUNS HERE
--------------
    role   what it is
    qha    **Branch B production trajectories** (user ruling 2026-09-07): OpenMM with
           openqha/openmm_mace.py on the CUDA platform, ONE TRAJECTORY PER CARD.
    train  Branch C MACE fine-tuning with the PHL loss.

Both are the same shape -- eight independent single-card tasks per allocation -- which is
why they share a layout and differ only in their label.

    CAVEAT ON BRANCH B, recorded rather than buried: this repository's only GPU
    measurement of branch B is D0-C-5, where the same trajectory ran 3.5x SLOWER on the
    GPU than on the CPU. That was a T400, a 2 GB entry-level card, against 80 GB HBM2e
    here, so the number does not transfer -- but nothing has replaced it either. Take
    seconds-per-ps off the `temp` smoke test before sizing a campaign.

CUDA: 12.3, THE VERSION BOTH GPU CLUSTERS HAVE
----------------------------------------------
`module avail` here tops out at **CUDA/12.3**; TianheXY-AI has 12.3 as well. That is why
there is now ONE GPU environment file (`environment-tianhe-gpu.yml`) instead of two.
Loading a CUDA the site does not have fails at `module load`; pinning a conda CUDA newer
than the runtime imports cleanly and dies at the first kernel launch, which is worse.

**No MPI is loaded.** Eight independent single-card workers need no collectives, and the
openmpi module names differ between the two GPU clusters -- so not needing MPI is exactly
what let one environment file serve both. `NCCL_MODULE` below is recorded for the day a
genuine DDP job is added; nothing loads it today.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import labels as _labels  # noqa: E402

# =========================================================================================
# SETTINGS
# =========================================================================================
ACCOUNT = None

#: Production partition and walltime (user ruling 2026-09-07).
PARTITION = "ai"
WALLTIME = "7-00:00:00"

#: Smoke-test partition and walltime. 30 minutes, inside `temp`'s own 2 h ceiling.
DEBUG_PARTITION = "temp"
DEBUG_WALLTIME = "00:30:00"

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

#: CUDA, pinned to what this cluster has -- and, since 2026-09-07, to what BOTH GPU
#: clusters have. environment-tianhe-gpu.yml pins the same number on the conda side.
CUDA_VERSION = "12.3"
#: Recorded, NOT loaded. See the module docstring: nothing here needs collectives.
NCCL_MODULE = "nccl/2.19.3-cuda-12.3"
MPI_MODULE = None

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
#: Branch B runs one trajectory per card and branch C fine-tunes one model per card. Both
#: are embarrassingly parallel, need no collective communication, and lose nothing to
#: synchronisation. A data-parallel eight-GPU job would spend the difference on gradient
#: all-reduce and give ONE result instead of eight.
#:
#: Switch to DDP only when one model genuinely does not fit on one 80 GB card, or when a
#: single fine-tune is too slow to be useful -- neither has been measured here, and until
#: it is, throughput beats latency because a committee is what branch C wants anyway.
WORKERS_PER_NODE = GPUS_PER_NODE           # 8
CPUS_PER_WORKER = CORES_PER_NODE // GPUS_PER_NODE   # 7 cores per card

NODES_PER_BLOCK = 1
#: 5 nodes x 8 cards = 40 concurrent single-GPU tasks, the whole quota.
MAX_BLOCKS = NODE_QUOTA

#: Per-task budget at 90% of the walltime, so a branch B task stops and flushes its last
#: chunk rather than being killed between a write and a rename.
QHA_WALL_BUDGET_S = int(0.90 * 7 * 24 * 3600)

#: OpenMM platform for branch B tasks here. The whole point of this cluster.
OPENMM_PLATFORM = "CUDA"
# =========================================================================================

#: Roles this cluster serves. Both are one task per card; they differ only in label.
_ROLES = ("qha", "train")


def _worker_init(here):
    """Modules first -- compute nodes are a minimal environment.

    Only CUDA. No MPI: eight independent single-card workers exchange nothing, and the
    openmpi module names differ between the two GPU clusters, so not loading one is what
    lets a single environment file serve both.

    The SLURM_/PMI_ unset and the `mkdir -p` are rules 7 and 6 of this project's HPC
    skill: inherited task-layout variables make a child process try to relaunch itself
    through the scheduler, and Slurm opens `--output` before the script runs.

    Module failures are NOT silenced. A missing CUDA here means every task in the block
    falls back to the CPU and takes 8x longer for reasons that appear nowhere.
    """
    return "; ".join([
        "mkdir -p ${S0_RUNS_ROOT:-$HOME/HDD_POOL/runs/openQHA}/logs",
        "module purge 2>/dev/null || true",
        "module load anaconda3/2023.09 2>/dev/null || "
        "module load miniforge/24.7.1 2>/dev/null || true",
        "module load CUDA/{0} || echo 'openQHA: module load CUDA/{0} FAILED -- tasks in "
        "this block will not see a card' >&2".format(CUDA_VERSION),
        "for v in $(env | awk -F= '{print $1}' | grep -E '^(PMI|SLURM)_'); do "
        "unset $v; done",
        "source {0}/env/common.sh".format(here),
        "source {0}/env/tianhe.sh".format(here),
        "openqha_report_env",
    ])


def config(partition=None, account=None, nodes_per_block=None, max_blocks=None,
           walltime=None, run_dir=None, worker_init=None, max_workers=None,
           gpus=None, exclusive=None, role="qha", debug=False):
    """Parsl Config for TianheXY-A. Every argument defaults to SETTINGS.

    `role` is `qha` (branch B trajectories) or `train` (branch C). They share the layout
    -- one task per card, eight per allocation -- and differ only in the executor label,
    so that a branch B app and a branch C app can never be scheduled onto each other's
    pool by accident.

    `debug=True` swaps in DEBUG_PARTITION and DEBUG_WALLTIME and caps the run at one
    allocation: 8 cards, 30 minutes, on the `temp` queue.
    """
    from parsl.config import Config
    from parsl.executors import HighThroughputExecutor
    from parsl.launchers import SimpleLauncher

    from providers import TianheSlurmProvider

    if role not in _ROLES:
        raise KeyError(
            "role {!r} does not run on TianheXY-A. This cluster serves {}.\n"
            "Branch A (CREST) is CPU work -- see hpc/resource_configs/tianhe_cpu.py."
            .format(role, " and ".join(_ROLES)))

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gpus = GPUS_PER_JOB if gpus is None else gpus
    excl = EXCLUSIVE if exclusive is None else exclusive
    blocks = MAX_BLOCKS
    if debug:
        partition = partition or DEBUG_PARTITION
        walltime = walltime or DEBUG_WALLTIME
        blocks = 1

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
                    max_blocks=min(int(max_blocks or blocks),
                                   NODE_QUOTA // int(nodes_per_block or NODES_PER_BLOCK),
                                   JOB_QUOTA),
                    # `--gpus=N`: the spelling TianheXY-AI's manual requires, applied here
                    # by the 2026-09-05 ruling. NOT parsl's `gpus_per_node`, which renders
                    # `--gpus-per-node=N`, a different Slurm option.
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
        site="tianhe_a", cluster="TianheXY-A", partition=PARTITION, walltime=WALLTIME,
        debug_partition=DEBUG_PARTITION, debug_walltime=DEBUG_WALLTIME,
        partitions_available=["ai", "temp"],
        cores_per_node=CORES_PER_NODE, gpus_per_node=GPUS_PER_NODE,
        mem_mb_per_node=MEM_MB_PER_NODE,
        gpus_per_job=GPUS_PER_JOB, exclusive=EXCLUSIVE,
        workers_per_node=WORKERS_PER_NODE, cpus_per_worker=CPUS_PER_WORKER,
        max_blocks=MAX_BLOCKS, node_quota=NODE_QUOTA, job_quota=JOB_QUOTA,
        max_tasks_in_flight=MAX_BLOCKS * WORKERS_PER_NODE,
        roles=list(_ROLES),
        strategy="8 independent single-GPU workers per node, not one 8-way DDP job",
        gpu_pinning="parsl available_accelerators (one card per worker)",
        openmm_platform=OPENMM_PLATFORM,
        qha_wall_budget_s=QHA_WALL_BUDGET_S,
        cuda=CUDA_VERSION, mpi_module=MPI_MODULE, nccl_module_recorded=NCCL_MODULE,
        environment_file="environment-tianhe-gpu.yml",
        labels=[_labels.label(r) for r in _ROLES],
        scheduler_commands=commands, scheduler_commands_confirmed=confirmed,
        notes=[
            "Preferred cluster for GPU work: 8 cards per allocation against "
            "TianheXY-AI's one.",
            "Branch B production trajectories run here, one per card (ruling "
            "2026-09-07). What stays on the CPU cluster is the collection pass.",
            "CUDA 12.3 is the ceiling here and is also present on TianheXY-AI, which is "
            "why one environment file now serves both.",
            "No MPI module is loaded; NCCL is recorded for a future DDP job only.",
            "`temp` is a 2-hour single-node queue: 30-minute smoke tests go there.",
            "9 of 25 `ai` nodes were in state O at the 2026-09-05 reading; effective "
            "capacity is below the node count.",
        ],
        assumptions=[
            "EXCLUSIVE=False follows the user ruling that this cluster uses TianheXY-AI's "
            "rules. Its own manual says nodes are exclusive and `yhbatch --help` lists "
            "--exclusive, so the flag is probably accepted here; revisit if a submission "
            "is refused for want of exclusivity.",
            "Branch B on a card is UNMEASURED here. The only GPU number this repository "
            "has for it (D0-C-5) is 3.5x SLOWER than CPU, on a T400 -- not transferable "
            "to an 80 GB card, and not a substitute for measuring it.",
            "8 independent workers rather than DDP is a throughput choice, not a "
            "measurement: neither layout has been timed on this machine.",
            "Nothing has been submitted to any Tianhe cluster yet.",
        ],
        verified=False,
    )
