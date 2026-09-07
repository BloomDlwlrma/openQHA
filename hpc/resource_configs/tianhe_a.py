"""Parsl resource configuration: TianheXY-A (login ln206). The GPU cluster we prefer.

THIS IS THE THIRD TIANHE CLUSTER. There are three, not two.
-----------------------------------------------------------
    TianheXY-C   tianhe_cpu.py   CPU   debug / deimos          64 cores, whole-node
    TianheXY-AI  tianhe_ai.py    GPU   hx/h100x/a100x/...      per GPU card, 80 GB
    TianheXY-A   this file       GPU   temp / ai               56 cores, 8 cards/node

**Prefer this cluster for GPU work** (user ruling 2026-09-07). Eight cards per allocation
against TianheXY-AI's one is the whole reason.

USING THE WHOLE NODE: 56 WORKERS, 8 CARDS, 7 WORKERS PER CARD
--------------------------------------------------------------
The node is **2 x 28 = 56 cores + 8 cards + 1024 GB**. An earlier version of this file ran
**8 workers, one per card** -- which used every card and **48 of the 56 cores sat idle**.

That is the wrong shape for this workload. A branch B trajectory is a serial chain of
single-structure MACE calls on a 10-19 atom molecule. On an 80 GB H100 one such call
occupies a rounding error of the card: the cost is kernel-launch latency and Python, not
arithmetic. **One trajectory cannot fill a card, so seven of them share one.**

    role     workers/node   cores/worker   cards   workers per card
    qha      56             1              8       7
    train     8             7              8       1

`train` keeps one task per card because a fine-tune genuinely does use the card, and a
committee wants one model per member. `qha` is the one that was leaving the node empty.

At the 5-node quota: **5 x 56 = 280 concurrent trajectories**, against 40 before.

    MPS. Without it, kernels from different processes time-slice on the card rather than
    running concurrently, so seven workers per card is a real gain but a sub-linear one.
    `nvidia-cuda-mps-control -d` in the job's prologue makes them concurrent. It is NOT
    enabled here: it needs the site to permit it, and an unmeasured change that also
    changes the failure modes is not something to switch on by default. Measure both.

HOW THE CARD IS CHOSEN, AND THE TRAP IN IT
------------------------------------------
`available_accelerators` is passed as an **int**, and that is load-bearing. Parsl's worker
does its own partitioning (`process_worker_pool.py`):

    procs_per_cuda_device = pool_size // num_cuda_devices      # 56 // 8 = 7
    CUDA_VISIBLE_DEVICES  = int(accelerator) // procs_per_cuda_device

With the int, parsl expands it to `['0' ... '55']` and the arithmetic spreads the workers
7 per card, all 8 used -- verified by replaying that arithmetic. **Passing a hand-built
list of device ids instead (`['0']*7 + ['1']*7 + ...`) puts 49 of the 56 workers on card
0**, because parsl divides again by 7 on a number that is already a device id.

That is the THIRD time this project has met the same failure: correct results, badly
placed workers, and a wall clock that reads as "the GPU is slow" rather than as a bug.
The other two were `CUDA_VISIBLE_DEVICES` in `worker_init` (runs once per block, not per
worker) and the same variable set once in a job script.

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
    qha    **Branch B production trajectories** (user ruling 2026-09-07): OpenMM with
           openqha/openmm_mace.py on the CUDA platform, 7 trajectories per card.
    train  Branch C MACE fine-tuning with the PHL loss, one model per card.

    CAVEAT ON BRANCH B, recorded rather than buried: this repository's only GPU
    measurement of branch B is D0-C-5, where the same trajectory ran 3.5x SLOWER on the
    GPU than on the CPU. That was a T400, a 2 GB entry-level card, against 80 GB HBM2e
    here, so the number does not transfer -- but nothing has replaced it either. Take
    seconds-per-ps off the `temp` smoke test before sizing a campaign.

MODULES: MEASURED 2026-09-07 from `/APP/u22/ai_x86/modulepath/`
----------------------------------------------------------------
    GPU_compiler    CUDA/11.3 11.7 11.8 12.0 12.1 12.2 12.3
                    mpi/openmpi/4.1.5-gcc-11.4.0
                    mpi/openmpi/5.0.0-gcc-11.4.0-cuda12.2
                    mpi/openmpi/5.0.0-{icc,icx}-oneapi2023.2-cuda12.2
                    intel/oneapi2023.2, nvhpc/{22.11,24.1}-openmpi4
    GPU_lib         cudnn/8.9.6.50-cuda12, cudnn/8.9.7.29-cuda11
                    nccl/2.19.3-cuda-{12.0,12.2,12.3}, gdrcopy/2.4-cuda-{11.8,12.2}
    GPU_application anaconda3/2023.09, miniforge/24.7.1, python/3.10.10, 3.12.10
                    Pytorch/{1.11.0-cuda11.3, 2.1.2-cuda11.8, 2.7.0-cuda11.8}
                    gromacs/2022.06, fftw, hdf5, OpenBLAS/0.3.30, ...

**CUDA/12.3 is the ceiling and there is no CUDA/12.4** -- which is what
`environment-tianhe-gpu.yml` pins, and this listing is the evidence for it.

**No MPI is loaded.** 56 independent single-card-sharing workers need no collectives, and
the openmpi module names differ between the two GPU clusters -- so not needing MPI is
exactly what let one environment file serve both. `NCCL_MODULE` below is recorded for the
day a genuine DDP job is added; nothing loads it today.

The site's own `Pytorch/*` modules are deliberately unused: they are CUDA 11.3/11.8 and
would drag their own Python. conda-forge supplies the whole stack above CUDA.
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
#: first, so 5 allocations -- and at 56 workers each that is 280 concurrent tasks.
NODE_QUOTA = 5
JOB_QUOTA = 10

#: CUDA, measured 2026-09-07 from this site's own module tree: 12.3 is the ceiling and
#: there is no 12.4. `environment-tianhe-gpu.yml` pins the same number on the conda side.
#: Change them together or neither.
CUDA_VERSION = "12.3"
#: Recorded, NOT loaded. See the module docstring: nothing here needs collectives.
NCCL_MODULE = "nccl/2.19.3-cuda-12.3"
MPI_MODULE = None
CUDNN_MODULE = "cudnn/8.9.6.50-cuda12"

#: USER RULING 2026-09-05: TianheXY-A follows TianheXY-AI's rules --
#: `--gpus` is passed, `--exclusive` is not.
#:
#: One tension recorded rather than hidden: this cluster's own manual says its nodes are
#: used exclusively, and `yhbatch --help` here DOES list `--exclusive`. So the flag is
#: probably accepted, unlike on TianheXY-AI where it is disabled. The ruling is followed;
#: if a submission is ever refused for want of exclusivity, EXCLUSIVE is the line to change.
GPUS_PER_JOB = GPUS_PER_NODE   #: 8 -- take the node's cards, since the node is exclusive
EXCLUSIVE = False

#: **Branch B: one worker per CORE, seven workers per CARD.** See the module docstring.
#: This is the setting that took the node from 8 busy cores to 56.
QHA_WORKERS_PER_NODE = CORES_PER_NODE                     # 56
QHA_CORES_PER_WORKER = 1
#: Derived, and reported by describe() so the layout is legible without arithmetic.
WORKERS_PER_CARD = QHA_WORKERS_PER_NODE // GPUS_PER_NODE  # 7

#: **Branch C: one worker per CARD.** A fine-tune does use the card, and a committee wants
#: one model per member, so here the card is the scarce thing and the cores follow it.
TRAIN_WORKERS_PER_NODE = GPUS_PER_NODE                    # 8
TRAIN_CORES_PER_WORKER = CORES_PER_NODE // GPUS_PER_NODE  # 7

#: Kept for anything that still reads the old names.
WORKERS_PER_NODE = QHA_WORKERS_PER_NODE
CPUS_PER_WORKER = QHA_CORES_PER_WORKER

NODES_PER_BLOCK = 1
#: 5 nodes x 56 workers = 280 concurrent trajectories, the whole quota.
MAX_BLOCKS = NODE_QUOTA

#: Per-task budget at 90% of the walltime, so a branch B task stops and flushes its last
#: chunk rather than being killed between a write and a rename.
QHA_WALL_BUDGET_S = int(0.90 * 7 * 24 * 3600)

#: OpenMM platform for branch B tasks here. The whole point of this cluster.
OPENMM_PLATFORM = "CUDA"

#: NOT enabled. `nvidia-cuda-mps-control -d` would let the 7 workers on a card run
#: concurrently instead of time-slicing. It needs the site to permit it and it changes the
#: failure modes, so it is a measurement to make, not a default to assume.
USE_MPS = False
# =========================================================================================

#: role -> (workers per node, cores per worker)
_LAYOUT = {
    "qha": (QHA_WORKERS_PER_NODE, QHA_CORES_PER_WORKER),
    "train": (TRAIN_WORKERS_PER_NODE, TRAIN_CORES_PER_WORKER),
}


def layout(role):
    """(workers_per_node, cores_per_worker) for a role. Unknown roles raise."""
    if role not in _LAYOUT:
        raise KeyError(
            "role {!r} does not run on TianheXY-A. This cluster serves {}.\n"
            "Branch A (CREST) is CPU work -- see hpc/resource_configs/tianhe_cpu.py."
            .format(role, " and ".join(sorted(_LAYOUT))))
    return _LAYOUT[role]


def cards_for(workers):
    """Which card each of `workers` workers lands on, by parsl's own arithmetic.

    Reproduced here rather than trusted, because getting it wrong is invisible: the job
    runs, the numbers are right, and only the wall clock is wrong. `describe()` reports
    the histogram so a misconfiguration is readable before submission rather than after.
    """
    per_card = max(1, workers // GPUS_PER_NODE)
    return [i // per_card for i in range(workers)]


def _worker_init(here):
    """Modules first -- compute nodes are a minimal environment.

    Only CUDA. No MPI: independent single-card workers exchange nothing, and the openmpi
    module names differ between the two GPU clusters, so not loading one is what lets a
    single environment file serve both.

    The SLURM_/PMI_ unset and the `mkdir -p` are rules 7 and 6 of this project's HPC
    skill: inherited task-layout variables make a child process try to relaunch itself
    through the scheduler, and Slurm opens `--output` before the script runs.

    Module failures are NOT silenced. A missing CUDA here means every task in the block
    falls back to the CPU and takes far longer for reasons that appear nowhere.

    **`CUDA_VISIBLE_DEVICES` must not be set here.** worker_init runs once per BLOCK,
    before the worker pool starts, so anything set here is inherited identically by all 56
    workers. Parsl assigns the card per worker; see the module docstring.
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

    `role` is `qha` (branch B trajectories, 56 workers over 8 cards) or `train`
    (branch C, 8 workers one per card). They differ in layout and in the executor label,
    so that a branch B app and a branch C app can never be scheduled onto each other's
    pool by accident.

    `debug=True` swaps in DEBUG_PARTITION and DEBUG_WALLTIME and caps the run at one
    allocation: one node, 30 minutes, on the `temp` queue.
    """
    from parsl.config import Config
    from parsl.executors import HighThroughputExecutor
    from parsl.launchers import SimpleLauncher

    from providers import TianheSlurmProvider

    workers, cores = layout(role)
    workers = int(max_workers or workers)

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
                max_workers_per_node=workers,
                cores_per_worker=float(cores),
                # AN INT, NOT A LIST. parsl expands it to ['0'..'N-1'] and its worker then
                # computes  CUDA_VISIBLE_DEVICES = int(acc) // (pool_size // n_devices),
                # which spreads N workers evenly over the node's cards. Handing it a list
                # of device ids instead makes it divide a device id by 7 and puts 49 of 56
                # workers on card 0 -- the job runs, the results are right, and it is
                # seven times slower for a reason nothing reports.
                available_accelerators=workers,
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
    hist = {}
    for card in cards_for(QHA_WORKERS_PER_NODE):
        hist[card] = hist.get(card, 0) + 1
    return dict(
        site="tianhe_a", cluster="TianheXY-A", partition=PARTITION, walltime=WALLTIME,
        debug_partition=DEBUG_PARTITION, debug_walltime=DEBUG_WALLTIME,
        partitions_available=["ai", "temp"],
        cores_per_node=CORES_PER_NODE, gpus_per_node=GPUS_PER_NODE,
        mem_mb_per_node=MEM_MB_PER_NODE,
        gpus_per_job=GPUS_PER_JOB, exclusive=EXCLUSIVE,
        layouts={r: dict(workers_per_node=w, cores_per_worker=c,
                         workers_per_card=max(1, w // GPUS_PER_NODE))
                 for r, (w, c) in _LAYOUT.items()},
        workers_per_node=QHA_WORKERS_PER_NODE, cpus_per_worker=QHA_CORES_PER_WORKER,
        workers_per_card=WORKERS_PER_CARD,
        card_assignment_histogram=hist,
        node_utilisation=dict(
            cores_used=QHA_WORKERS_PER_NODE * QHA_CORES_PER_WORKER,
            cores_available=CORES_PER_NODE,
            cards_used=len(hist), cards_available=GPUS_PER_NODE),
        max_blocks=MAX_BLOCKS, node_quota=NODE_QUOTA, job_quota=JOB_QUOTA,
        max_tasks_in_flight={r: MAX_BLOCKS * w for r, (w, _c) in _LAYOUT.items()},
        roles=sorted(_LAYOUT),
        strategy=("branch B: 56 workers over 8 cards, 7 per card, one core each -- the "
                  "whole node. branch C: 8 workers, one card each."),
        gpu_pinning="parsl available_accelerators passed as an INT (see the docstring)",
        mps_enabled=USE_MPS,
        openmm_platform=OPENMM_PLATFORM,
        qha_wall_budget_s=QHA_WALL_BUDGET_S,
        cuda=CUDA_VERSION, mpi_module=MPI_MODULE, nccl_module_recorded=NCCL_MODULE,
        cudnn_module_recorded=CUDNN_MODULE,
        environment_file="environment-tianhe-gpu.yml",
        labels=[_labels.label(r) for r in sorted(_LAYOUT)],
        scheduler_commands=commands, scheduler_commands_confirmed=confirmed,
        notes=[
            "Preferred cluster for GPU work: 8 cards per allocation against "
            "TianheXY-AI's one.",
            "Branch B uses the WHOLE node: 56 workers, 1 core each, 7 sharing each card. "
            "The previous 8-worker layout left 48 of 56 cores idle.",
            "CUDA 12.3 is the ceiling here, measured from the site module tree "
            "2026-09-07; there is no CUDA/12.4.",
            "No MPI module is loaded; NCCL and cuDNN are recorded for a future DDP job.",
            "`temp` is a 2-hour single-node queue: 30-minute smoke tests go there.",
            "9 of 25 `ai` nodes were in state O at the 2026-09-05 reading; effective "
            "capacity is below the node count.",
        ],
        assumptions=[
            "EXCLUSIVE=False follows the user ruling that this cluster uses TianheXY-AI's "
            "rules. Its own manual says nodes are exclusive and `yhbatch --help` lists "
            "--exclusive, so the flag is probably accepted here; revisit if a submission "
            "is refused for want of exclusivity.",
            "7 workers per card is an ARGUMENT (a 10-19 atom molecule cannot fill an "
            "80 GB card), not a measurement. Without MPS the seven time-slice rather "
            "than run concurrently, so the gain is real but sub-linear and unmeasured.",
            "Branch B on a card is UNMEASURED here. The only GPU number this repository "
            "has for it (D0-C-5) is 3.5x SLOWER than CPU, on a T400 -- not transferable "
            "to an 80 GB card, and not a substitute for measuring it.",
            "Nothing has been submitted to any Tianhe cluster yet.",
        ],
        verified=False,
    )
