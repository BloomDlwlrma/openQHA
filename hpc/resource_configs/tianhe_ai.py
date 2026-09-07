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
    """Modules, then the environment. CUDA 12.4 and the matching openmpi, per the user.

    The module set is the site's own: `CUDA/12.4` with
    `mpi/openmpi/5.0.10-gcc-11.4.0-cuda12.4`, which is the openmpi built against that
    CUDA. Mixing an openmpi built for one CUDA with another CUDA is the same class of
    mistake as two BLAS builds in one environment, and it fails later and less clearly.

    `nccl/2.23.4-cuda-12.4` and `gdrcopy/2.4-cuda-12.4` are the matching collectives and
    RDMA-copy modules; loaded only because they are the ones built for 12.4, and harmless
    for a single-card job.
    """
    cpus = cpus_for(partition)
    threads = "export OMP_NUM_THREADS=1" if cpus is None else (
        "export OMP_NUM_THREADS=1   # never more than the package's %d CPUs" % cpus)
    return "; ".join([
        "mkdir -p ${S0_RUNS_ROOT:-$HOME/HDD_POOL/runs/openQHA}/logs",
        "module purge 2>/dev/null || true",
        "module load anaconda3/2023.09 2>/dev/null || true",
        "module load CUDA/12.4 2>/dev/null || true",
        "module load mpi/openmpi/5.0.10-gcc-11.4.0-cuda12.4 2>/dev/null || true",
        "module load nccl/2.23.4-cuda-12.4 2>/dev/null || true",
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
        modules=["anaconda3/2023.09", "CUDA/12.4",
                 "mpi/openmpi/5.0.10-gcc-11.4.0-cuda12.4", "nccl/2.23.4-cuda-12.4"],
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
            "Nothing has been submitted to either Tianhe cluster yet.",
        ],
        verified=False,
    )
