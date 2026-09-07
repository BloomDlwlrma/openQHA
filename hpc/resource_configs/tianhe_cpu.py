"""Parsl resource configuration: Tianhe CPU cluster (TianheXY-C, Xingyi). Branches A/B/C-QM.

Edit SETTINGS, not the command line. Production is then

    python -u scripts/production/s0_E_branchA_parsl.py --edges --resource tianhe_cpu

THIS IS THE CPU CLUSTER, AND IT IS A DIFFERENT MACHINE FROM TianheXY-AI
----------------------------------------------------------------------
There are two clusters, with separate partitions, separate allocation policies and
separate filesystems. An earlier version of this repository conflated them and put branch
A on the GPU cluster; that was wrong.

    TianheXY-C   (this file)      CPU. Partitions xyfree / mars / deimos / e9.
                                  Node = 2 x 32 cores, 512 GB. Filesystem XYFS01.
                                  Normal partitions allocate a WHOLE NODE (exclusive).
    TianheXY-AI  (tianhe_ai.py)   GPU. Partitions hx / h100x / a100x (and a800x, v100x).
                                  Allocated per GPU card. Filesystem XYAIFS00.
                                  `-G` mandatory, `--exclusive` BANNED.

**Branch A belongs here.** Its cost is ~1e5 GFN2-xTB gradient calls and xtb has no GPU
path; MACE refines 10-20 atom structures that cannot fill a card. Sending it to the GPU
cluster would take a card to do CPU work.

PARTITIONS (manual section 1.1)
-------------------------------
    xyfree   64 cores  512 GB   trial-user partition
    mars     64 cores  512 GB   long queue -- no runtime limit
    deimos   64 cores  512 GB   short queue -- 7 days maximum
    e9       64 cores  512 GB   fine-grained: allocated per CPU, not per node

`deimos` is the default: 7 days is far more than one openQHA block needs, and a queue
with a limit schedules sooner than one without. Use `mars` only for something that
genuinely cannot be cut into blocks. **`e9` is the one to reach for if whole-node
exclusive allocation is wasteful** -- branch A fills 64 cores with 16 molecules, so it
is not, but branch B's single-core trajectories might be.

THE ARITHMETIC
--------------
    64 cores per node       manual section 1.1, 2 x 32
    4 threads per molecule  the condition the 285 s/species cost was measured under
    -> 16 molecules per node, no oversubscription

    285 s per species       measured, 4 threads, UNCONTENDED, on this project's
                            workstation (S0-A-8). NOT measured here, and not under
                            contention: 16 concurrent jobs share memory bandwidth.

A per-node capacity is not a speed-up. Branch E acceptance criterion 5 wants wall clock,
single-job time and slot extrapolation reported as three separate numbers.

FILESYSTEM (manual section 1.2)
-------------------------------
Home is quota'd at 100 GB and is for configuration. **Code, builds, data and job output
go in `HDD_POOL`** (a symlink in home onto HDD storage), which the manual asks for
explicitly. XYFS01 is lustre with NO BACKUP -- a deleted file is gone.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import labels as _labels  # noqa: E402

# =========================================================================================
# SETTINGS -- the only block most people should edit
# =========================================================================================
ACCOUNT = None                 #: scheduler account; None lets the site choose
PARTITION = "deimos"           #: xyfree | mars | deimos | e9  (see the docstring)
CORES_PER_NODE = 64            #: manual section 1.1: 2 x 32
THREADS_PER_JOB = 4            #: CREST threads per molecule; changing it invalidates 285 s
WORKERS_PER_NODE = CORES_PER_NODE // THREADS_PER_JOB      # 16

NODES_PER_BLOCK = 1

#: Node quota for this cluster (Starlight, 2026-09-05): 32 running jobs, 32 nodes.
#: `max_blocks * nodes_per_block` must not exceed it -- Parsl does not know about the
#: quota, so beyond it the scheduler simply refuses and the run looks stalled, not capped.
NODE_QUOTA = 32
JOB_QUOTA = 32

#: Allocations held at once. Set to the quota: branch A at 16 workers/node is then 512
#: molecules in flight, branch B at 64 workers/node is 2048. `init_blocks=0` and
#: `min_blocks=0` still mean nothing is requested until there is work.
MAX_BLOCKS = 32

#: Walltime. `deimos` allows 7 days; this is deliberately far below it. Branch A writes
#: one product per molecule, so a killed job loses only what was in flight, and a short
#: request schedules sooner. Raise it once a campaign shows the queue is not the problem.
WALLTIME = "11:24:00"

#: Branch B: ONE core per trajectory, so one node runs 64 at a time.
#:
#: Not a preference -- measured. A quasi-harmonic trajectory is a serial chain of MACE
#: force calls, and MACE on a 10-atom molecule runs 111/90/72/101 ms at 1/2/4/8 threads
#: (D0-P1-27, D0-P1-32). Four threads buy 1.54x; four independent trajectories buy 4x.
#: Branch B has (basins x seeds) of independent work per species, so it never runs short.
QHA_THREADS_PER_JOB = 1
QHA_WORKERS_PER_NODE = CORES_PER_NODE // QHA_THREADS_PER_JOB      # 64

#: Driver defaults, so the command line stays `--edges --resource tianhe_cpu`.
TAG = "prod"
TIMEOUT_S = 14400
HESSIAN_MODE = "analytic"
# =========================================================================================


def _worker_init(here):
    """What every worker runs first. Three of these are not optional.

    * `module load` -- compute nodes are a minimal environment (the manual says only sh
      and bash are supported there), so nothing is on PATH unless it is loaded.
    * **Unset inherited SLURM_/PMI_ variables.** They make a child process misread the
      task layout and try to relaunch itself through the scheduler. Rule 7 of this
      project's own HPC skill, learnt on ORCA under xargs; CREST forks its own parallel
      workers and is exposed to the same thing.
    * `mkdir -p` the log directory -- Slurm opens `--output` BEFORE the script runs, so
      creating it inside the script is too late (rule 6 of the same skill).
    """
    return "; ".join([
        "mkdir -p ${S0_RUNS_ROOT:-$HOME/HDD_POOL/runs/openQHA}/logs",
        "module purge 2>/dev/null || true",
        "module load anaconda3/2023.09 2>/dev/null || true",
        "for v in $(env | awk -F= '{print $1}' | grep -E '^(PMI|SLURM)_'); do "
        "unset $v; done",
        "source {0}/env/common.sh".format(here),
        "source {0}/env/tianhe.sh".format(here),
        "openqha_report_env",
    ])


def config(partition=None, account=None, nodes_per_block=None, max_blocks=None,
           walltime=None, run_dir=None, worker_init=None, max_workers=None,
           role="crest"):
    """Parsl Config for the Tianhe CPU cluster. Every argument defaults to SETTINGS.

    `role` selects the layout, because the two branches want opposite things from the
    same node:

        crest  16 workers x 4 threads   branch A: CREST is internally parallel
        qha    64 workers x 1 thread    branch B: a trajectory is a serial chain of
                                        force calls, so parallelism goes between them

    Whole-node allocation suits `crest` -- 16 x 4 fills 64 cores exactly. It suits `qha`
    too at 64 x 1. **What it does not suit is a partial job**, and that is what the `e9`
    fine-grained partition is for; pass `partition="e9"` if a run cannot fill a node.
    """
    from parsl.config import Config
    from parsl.executors import HighThroughputExecutor
    from parsl.launchers import SimpleLauncher

    from providers import TianheSlurmProvider

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    cfg = Config(
        executors=[
            HighThroughputExecutor(
                # From hpc/labels.py, never a literal: a label that does not match the
                # one the app asks for builds and renders cleanly, then schedules
                # nothing. That is how it failed on 2026-09-05.
                label=_labels.label(role),
                max_workers_per_node=int(
                    max_workers or (QHA_WORKERS_PER_NODE if role == "qha"
                                    else WORKERS_PER_NODE)),
                cores_per_worker=float(QHA_THREADS_PER_JOB if role == "qha"
                                       else THREADS_PER_JOB),
                provider=TianheSlurmProvider(
                    partition or PARTITION,
                    account=account if account is not None else ACCOUNT,
                    nodes_per_block=nodes_per_block or NODES_PER_BLOCK,
                    init_blocks=0, min_blocks=0,
                    # Capped at the site quota, whatever the caller asks for.
                    max_blocks=min(int(max_blocks or MAX_BLOCKS),
                                   NODE_QUOTA // int(nodes_per_block or NODES_PER_BLOCK),
                                   JOB_QUOTA),
                    scheduler_options="",
                    # THIS CLUSTER IS THE OPPOSITE OF THE AI ONE. Its normal partitions
                    # allocate a whole node -- the manual calls this exclusive
                    # mode -- so `exclusive=True` is correct here,
                    # while on TianheXY-AI the flag is banned outright. Two clusters, two
                    # answers; that is why they are two files.
                    exclusive=True,
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
        site="tianhe_cpu", cluster="TianheXY-C", partition=PARTITION,
        partitions_available=["xyfree", "mars", "deimos", "e9"],
        cores_per_node=CORES_PER_NODE, threads_per_job=THREADS_PER_JOB,
        workers_per_node=WORKERS_PER_NODE,
        qha_workers_per_node=QHA_WORKERS_PER_NODE,
        qha_threads_per_job=QHA_THREADS_PER_JOB,
        max_blocks=MAX_BLOCKS, node_quota=NODE_QUOTA, job_quota=JOB_QUOTA,
        max_molecules_in_flight=dict(crest=MAX_BLOCKS * WORKERS_PER_NODE,
                                     qha=MAX_BLOCKS * QHA_WORKERS_PER_NODE),
        walltime=WALLTIME,
        exclusive=True, filesystem="XYFS01 (lustre, 1 TB, NO BACKUP)",
        labels=[_labels.label("crest"), _labels.label("qha")],
        tag=TAG, timeout_s=TIMEOUT_S, hessian_mode=HESSIAN_MODE,
        scheduler_commands=commands, scheduler_commands_confirmed=confirmed,
        notes=[
            "Normal partitions allocate a whole node (manual 1.1); `e9` is the "
            "fine-grained per-CPU partition if that is wasteful.",
            "Home is 100 GB and for configuration only -- run out of HDD_POOL.",
            "XYFS01 has no backup: a deleted file cannot be recovered.",
        ],
        assumptions=[
            "285 s/species was measured on this project's workstation, not here, and "
            "not under contention.",
            "Nothing has been submitted to either Tianhe cluster yet.",
        ],
        verified=False,
    )
