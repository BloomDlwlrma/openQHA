"""Parsl resource configuration: Tianhe GPU cluster (TianheXY-AI, Xingyi). Branch C training.

    python -u <a branch C driver> --resource tianhe_ai

THIS IS THE GPU CLUSTER. Branch A does NOT belong here -- see tianhe_cpu.py.
--------------------------------------------------------------------------
The one part of this repository that is genuinely faster on a GPU is branch C's MACE
fine-tuning with the PHL loss. Branch A is ~1e5 GFN2-xTB gradient calls, and xtb has no
GPU path at all.

ALLOCATION IS PER GPU CARD (manual section 1.1)
-----------------------------------------------
    partition   policy
    hx          1 GPU / 14 CPUs / 240 GB
    h100x       1 GPU / 14 CPUs / 240 GB
    a100x       1 GPU / 12 CPUs / 120 GB
    a800x, v100x   present in `sinfo`; their policies are not in the manual

Cards carry 80 GB; the driver is 550.54.15. **A job may not request more CPUs or more
memory than its package**, or the submission is rejected.

THREE SITE RULES, AND TWO OF THEM WOULD REJECT EVERY JOB
--------------------------------------------------------
1. **`-G` / `--gpus` is MANDATORY** (manual 6.2.1.1, 6.2.2): resources are allocated at
   GPU granularity and a job without it does not run.
2. **`--exclusive` is BANNED** (manual 6.2.2): allocation here is shared, and the flag is
   disabled by the scheduler. **parsl's SlurmProvider sets `exclusive=True` by default**,
   so a stock provider renders a script this cluster refuses.
3. **Memory must NOT be specified** (manual 6.2.2): the system sizes it from the GPU
   count. `mem_per_node` is therefore never passed.

Note rule 2 is the exact opposite of the CPU cluster, whose normal partitions allocate a
whole node. Two clusters, two answers -- which is why they are two files.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import labels as _labels  # noqa: E402

# =========================================================================================
# SETTINGS
# =========================================================================================
ACCOUNT = None
PARTITION = "h100x"            #: hx | h100x | a100x | a800x | v100x
GPUS_PER_JOB = 1

#: CUDA. 12.3 because that is the version BOTH GPU clusters have -- TianheXY-A has no
#: 12.4 -- so one `environment-tianhe-gpu.yml` serves both. The conda side pins the same
#: number. Change them together or neither.
#:
#: **Check this exists before the first GPU job here**: the recorded module listing for
#: this cluster was elided (`CUDA/11.8 ... CUDA/13.2`), so 12.3 is inside the recorded
#: range but was never read off the screen.
#:     module avail CUDA 2>&1 | grep -o "CUDA/12[.][0-9]*"
CUDA_VERSION = "12.3"

#: Recorded, NOT loaded. These are the modules built against 12.4; they are what a DDP
#: job would need, and openQHA runs none.
MPI_MODULE_RECORDED = "mpi/openmpi/5.0.10-gcc-11.4.0-cuda12.4"
NCCL_MODULE_RECORDED = "nccl/2.23.4-cuda-12.4"

#: TianheXY-A is preferred for GPU work (user ruling 2026-09-07): 8 cards per allocation
#: against this cluster's one. Use this cluster when A is full, or for a job that wants a
#: specific card type from the PACKAGE table below.
PREFER_INSTEAD = "tianhe_a"

#: CPUs and memory that come with one card, from the manual's table. Used to keep the
#: job inside its package -- exceeding either is a rejected submission, not a slow job.
PACKAGE = {
    "hx":    dict(cpus=14, mem_gb=240),
    "h100x": dict(cpus=14, mem_gb=240),
    "a100x": dict(cpus=12, mem_gb=120),
    # a800x and v100x appear in `sinfo` but not in the manual's policy table.
    "a800x": dict(cpus=None, mem_gb=None),
    "v100x": dict(cpus=None, mem_gb=None),
}

#: Training is one task per allocation: the task uses the whole card and all the CPUs in
#: the package. This is ALF's "one QM job per allocation" pattern applied to training.
WORKERS_PER_NODE = 1

NODES_PER_BLOCK = 1

#: Quota (Starlight, 2026-09-05): 6 running jobs, 6 nodes.
NODE_QUOTA = 6
JOB_QUOTA = 6

#: One allocation is ONE CARD here (allocation is per GPU, not per node), so 6 blocks is
#: 6 concurrent single-GPU trainings -- a committee, or a hyper-parameter sweep.
MAX_BLOCKS = 6
WALLTIME = "16:00:00"
# =========================================================================================


def cpus_for(partition):
    """CPUs available in one card's package, or None where the manual does not say."""
    return (PACKAGE.get(partition) or {}).get("cpus")


def _worker_init(here, partition):
    """Modules, then the environment. CUDA 12.3 -- and nothing else.

    **12.3, not 12.4** (user ruling 2026-09-07). 12.4 exists here and does not exist on
    TianheXY-A; 12.3 exists on both. Pinning the version the two clusters share is what
    let `environment-tianhe-cuda.yml` and `environment-tianhe-a-cuda.yml` collapse into
    one `environment-tianhe-gpu.yml`.

    **No MPI, and no NCCL.** They used to be loaded here because they were the modules
    built against 12.4, and they were the other thing that differed between the two GPU
    clusters. Nothing openQHA runs on a card needs collectives: training is one model per
    allocation and branch B is one trajectory per card. An openmpi built against a
    different CUDA than the one loaded fails at the first collective rather than at
    import -- so the safest version of that dependency is not having it. The matched
    module names stay recorded in `describe()` for the day a real DDP job appears.

    The module load is NOT silenced: a missing CUDA here means the task falls back to the
    CPU and takes far longer for a reason that appears nowhere in the product.
    """
    cpus = cpus_for(partition)
    threads = "export OMP_NUM_THREADS=1" if cpus is None else (
        "export OMP_NUM_THREADS=1   # never more than the package's %d CPUs" % cpus)
    return "; ".join([
        "mkdir -p ${S0_RUNS_ROOT:-$HOME/HDD_POOL/runs/openQHA}/logs",
        "module purge 2>/dev/null || true",
        "module load anaconda3/2023.09 2>/dev/null || true",
        "module load CUDA/{0} || echo 'openQHA: module load CUDA/{0} FAILED -- this "
        "block will not see a card' >&2".format(CUDA_VERSION),
        "for v in $(env | awk -F= '{print $1}' | grep -E '^(PMI|SLURM)_'); do "
        "unset $v; done",
        threads,
        "source {0}/env/common.sh".format(here),
        "source {0}/env/tianhe.sh".format(here),
        "openqha_report_env",
    ])


def config(partition=None, account=None, nodes_per_block=None, max_blocks=None,
           walltime=None, run_dir=None, worker_init=None, max_workers=None,
           gpus=None, role="train"):
    """Parsl Config for branch C training on the Tianhe GPU cluster."""
    from parsl.config import Config
    from parsl.executors import HighThroughputExecutor
    from parsl.launchers import SimpleLauncher

    from providers import TianheSlurmProvider

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    part = partition or PARTITION
    gpus = GPUS_PER_JOB if gpus is None else gpus

    cfg = Config(
        executors=[
            HighThroughputExecutor(
                label=_labels.label(role),
                max_workers_per_node=int(max_workers or WORKERS_PER_NODE),
                provider=TianheSlurmProvider(
                    part,
                    account=account if account is not None else ACCOUNT,
                    nodes_per_block=nodes_per_block or NODES_PER_BLOCK,
                    init_blocks=0, min_blocks=0,
                    max_blocks=min(int(max_blocks or MAX_BLOCKS),
                                   NODE_QUOTA, JOB_QUOTA),
                    # `--gpus=N`: the spelling the manual requires. NOT parsl's
                    # `gpus_per_node`, which renders `--gpus-per-node=N`, a different
                    # Slurm option the manual never mentions.
                    scheduler_options="#SBATCH --gpus={}".format(gpus),
                    # BANNED here (manual 6.2.2) and parsl defaults it to True.
                    exclusive=False,
                    # mem_per_node deliberately never passed: specifying memory is also
                    # banned; the system sizes it from the GPU count.
                    launcher=SimpleLauncher(),
                    worker_init=worker_init or _worker_init(here, part),
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
        site="tianhe_ai", cluster="TianheXY-AI", partition=PARTITION,
        partitions_available=sorted(PACKAGE),
        package=PACKAGE.get(PARTITION), gpus_per_job=GPUS_PER_JOB,
        workers_per_node=WORKERS_PER_NODE, max_blocks=MAX_BLOCKS,
        node_quota=NODE_QUOTA, job_quota=JOB_QUOTA,
        max_trainings_in_flight=MAX_BLOCKS * WORKERS_PER_NODE,
        walltime=WALLTIME,
        exclusive=False, filesystem="XYAIFS00 (lustre, 1 TB, NO BACKUP)",
        modules=["anaconda3/2023.09", "CUDA/" + CUDA_VERSION],
        modules_recorded_not_loaded=[MPI_MODULE_RECORDED, NCCL_MODULE_RECORDED],
        environment_file="environment-tianhe-gpu.yml",
        prefer_instead=PREFER_INSTEAD, cuda=CUDA_VERSION,
        labels=[_labels.label("train")],
        scheduler_commands=commands, scheduler_commands_confirmed=confirmed,
        site_rules=[
            "-G/--gpus is MANDATORY (manual 6.2.1.1, 6.2.2)",
            "--exclusive is BANNED (manual 6.2.2); parsl defaults it to True",
            "memory must NOT be specified (manual 6.2.2)",
            "CPUs requested must not exceed the card's package",
        ],
        assumptions=[
            "a800x and v100x packages are not in the manual's table.",
            "CUDA/12.3 is ASSUMED present here. The recorded module listing was elided "
            "(CUDA/11.8 ... CUDA/13.2); confirm with `module avail CUDA` before the "
            "first GPU job, because the conda side is pinned to it.",
            "Nothing has been submitted to any Tianhe cluster yet.",
        ],
        verified=False,
    )
