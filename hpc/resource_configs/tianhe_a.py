"""Parsl resource configuration: TianheXY-A (login ln206 / ln207), GPU, **fine-grained**.

THIS IS THE THIRD TIANHE CLUSTER. There are three, not two.
-----------------------------------------------------------
    TianheXY-C   tianhe_cpu.py   CPU   debug / deimos          64 cores, whole-node
    TianheXY-AI  tianhe_ai.py    GPU   hx/h100x/a100x/...      per GPU card, 80 GB
    TianheXY-A   this file       GPU   ai (fine-grained)       per GPU card: 12 cpus, 120 GB

ONE LOGIN NODE, TWO SLURM ENVIRONMENTS  (measured 2026-09-11 + site PDF)
-------------------------------------------------------------------------
This is the fact that reconciles every contradictory measurement this project has made
on ln207, and it comes from the site's own document, "星逸 AI 集群 GPU 细粒度调用使用说明":

    DEFAULT environment (what a fresh login gives you)
        $ sinfo -p ai -o "%P %N %G %c %m"
        ai   an[9,11-17,20-23,25-26,28-29,31-32,34-35,38-40,42-43]   (null)   56+   983040+
      Gres=(null) on all 25 nodes. Allocated BY CPU. The 8 cards per node exist and
      `nvidia-smi` sees all of them, but Slurm does not track them, so **any** card
      request is refused: "Invalid generic resource (gres) specification". Two jobs on a
      node both see all 8 cards.

    FINE-GRAINED environment ("tianhexy-i")
        $ source /APP/u22/ai_x86/toolshs/set-XY-I.sh      # = export PATH=/usr/local/slurm.24051/bin:$PATH
        $ sinfo -h -p ai -N -o "%N %G %c %m"
        an45 gpu:a800:8 96 983040      (an[45-47,49-51,53,65]: 8 nodes, all identical)
        $ scontrol show node an45
        CoresPerSocket=24 Sockets=2 ThreadsPerCore=2 CPUTot=96 RealMemory=983040
        Gres=gpu:a800:8 Version=24.05.1
        CfgTRES=cpu=96,mem=960G,billing=128,gres/gpu:a800=8
        $ scontrol show partition ai
        OverSubscribe=NO ExclusiveUser=NO MaxNodes=UNLIMITED MaxTime=UNLIMITED
        AllowGroups=root,tianhexy_i_ai   TRES=cpu=768,mem=7.50T,node=8,gres/gpu:a800=64
        TRESBillingWeights=CPU=1.0,Mem=0.0,GRES/gpu:a800=4.0
      The environment script is nothing but a PATH change to a Slurm 24.05 client that
      talks to a second controller. Allocated BY CARD. The PDF's policy **1 GPU = 12 CPUs
      = 120 GB** is simply the node divided by eight: 96 logical CPUs (48 cores x 2
      threads) / 8 = 12, 960 GB / 8 = 120. `-G N` or `--gres=gpu:N` is **mandatory**
      (`srun -p ai hostname` -> "Unable to allocate resources"). `--mem` is **forbidden**
      (same error). `nvidia-smi` inside the job shows only the allocated cards.
      Billing, two layers: Slurm's weights make one card + its 12 CPUs = 4 + 12 = 16 units
      (a node = 128); the site's invoice rule on top is max(GPUs, ceil(CPUs/12)) card-
      equivalents x hours x weight.

**This file uses the fine-grained environment.** It is what the user's site document
describes, it is what a small job actually needs, and it is the one where the scheduler
does the card isolation instead of hoping nobody else lands on the node.

The two environments are the same physical machine and the same login prompt. Which one
a command talks to depends only on whether `set-XY-I.sh` has been sourced in that shell.
`examples/run_chain.sh` checks (`sinfo -h -p ai -o %G` must mention `gpu`) before it
submits, and `examples/slurm/ai.slurm` sources the script inside the job so the driver's
own submissions -- if any -- go to the same controller.

WHAT THIS RESOLVES
------------------
  * `D0-C-25` "Tianhe requires an explicit -G" -- TRUE, in the fine-grained environment.
  * 2026-09-11 "TianheXY-A refuses any -G" -- TRUE, in the default environment.
  * The 2026-09-05 ruling "no --exclusive here" -- now MOOT. In a per-card environment
    `--exclusive` would take the node and be billed as 8 cards. EXCLUSIVE stays False,
    and there is nothing left to rule on (memory.md open item 39 is closed by this).
  * "Two jobs sharing a node both pin from device 0" -- cannot happen here: the job sees
    only its own cards.

SIZING: PAY FOR CARDS, USE EVERY CPU THEY BRING
-----------------------------------------------
The bill counts cards, and 12 CPUs come with each one whether they are used or not. So
the pool is sized in cards, and the CPUs follow:

    role    workers per card    cores per worker      what a card runs
    qha     12                  1                     12 branch B trajectories
                                                          **measured optimal**, see below
    train   1                   12                    one fine-tune

A branch B trajectory is a serial chain of single-structure MACE calls on a 10-19 atom
molecule; one cannot fill an 80 GB card, so twelve share it (the earlier layout was 7 per
card from 56 cores / 8 cards; the policy now hands out 12 per card and it is the policy
that is billed).

**12 is now measured, not assumed.** an45, 2026-09-12, MACE-OFF23_medium on 10 atoms,
float64 with Precision=double, one core per worker, 200 steps each
(`scripts/tooling/s0_gpu_concurrency.py`):

    N workers   median ms/step   steps/s, all workers
     1           45.8              13.8
     2           45.3              26.3
     4           41.9              44.1
     8           54.3              74.1
    12           60.0              94.1     <- the layout above
    16           98.7              90.5     <- past the knee: slower AND less throughput

Throughput peaks at 12 and FALLS at 16 while per-worker latency nearly doubles, so the
policy's 12 CPUs per card and the best point on this curve are the same number. That is
luck, not design, but it does mean there is nothing to trade off.

The same sweep on the CPUs alone, one core per worker, is the honest comparison -- both
sides get 12 cores, one side also gets the card:

    N workers   median ms/step   steps/s, all workers
     1           98.9               8.1
    12          186.3              50.4

So **the card is worth 1.87x on top of the twelve CPUs it comes with** (94.1 vs 50.4).
Real, and nothing like the order of magnitude a GPU suggests -- a ten-atom molecule
evaluated one structure at a time is latency-bound, and the card is being fed by twelve
serial streams.

Note what this corrects: the single-worker CPU figure quoted elsewhere as 61.7 ms/step
was the probe running with Threads=112, i.e. THE WHOLE NODE for one trajectory. At one
core -- what a worker in this layout actually gets -- it is 98.9. Comparing a card
against a whole node's CPU was never the question being asked.

Acetone (`examples/02a`) is 1 basin x 3 seeds = 3 trajectories. That is **one card**,
billed as one card, with 9 of its 12 CPUs idle -- and no cheaper request exists.

TWO WAYS TO RUN, AND WHICH IS THE DEFAULT
-----------------------------------------
    IN-ALLOCATION (default when the driver is already inside a Slurm job)
        The driver and the workers share ONE job: `yhbatch -G 1 -c 12 ai.slurm conf`.
        Parsl uses a LocalProvider on the node it is already on. No second allocation,
        no driver card sitting idle, no submission from a compute node. This is what a
        one-molecule example wants and what `examples/run_chain.sh` produces.

    NESTED (the driver submits its own blocks; set S0_PARSL_NESTED=1 to force it)
        The original design: the driver holds a small job and parsl submits up to
        MAX_BLOCKS further jobs of GPUS_PER_BLOCK cards each. For a campaign of hundreds
        of molecules this is how the quota gets filled. The driver's own card is the
        price of it, so it is not the default for one molecule.

HOW THE CARD IS CHOSEN, AND WHY A SMALL POOL IS SAFE
----------------------------------------------------
`available_accelerators` is passed as an **int** equal to the worker count. Parsl's
worker (`process_worker_pool.py:730-745`, read 2026-09-11 rather than assumed):

    num_cuda_devices  = `nvidia-smi -L | wc -l`      -> the cards THIS JOB can see
    procs_per_device  = pool_size // num_cuda_devices
    CUDA_VISIBLE_DEVICES = int(accelerator) // procs_per_device
    except ZeroDivisionError: CUDA_VISIBLE_DEVICES = accelerator

In the fine-grained environment `nvidia-smi` reports only the allocated cards, so with
G cards and 12G workers every card gets 12; with 1 card and 3 workers, 3 // 1 = 3 and
0 // 3 = 1 // 3 = 2 // 3 = 0 -- all three on the one card, which is right. The one shape
that misplaces workers is a pool that is not a multiple of the card count (5 workers on
2 cards puts one on a card index that does not exist), so `config()` rounds the pool to
a multiple. **Never pass a hand-built list of device ids**: parsl divides a list entry by
`procs_per_device` again, and `['0']*6 + ['1']*6` puts all twelve on card 0. That
mistake has been made three times in this project; it shows up only as wall clock.

    MPS. Without it, kernels from different processes time-slice on the card. It is NOT
    enabled: it needs the site to permit it and it changes the failure modes.

SLURM SEMANTICS THIS FILE LEANS ON  (slurm.schedmd.com gres.html + sbatch.html, read 2026-09-11)
-----------------------------------------------------------------------------------------
    --gpus / -G N          "total number of GPUs required for the job". Per JOB.
    --gpus-per-node N      "equivalent to the --gres option for GPUs". Per node.
    --gpus-per-task N      per task; "requires the job to specify a task count" (-n).
    --cpus-per-gpu N       CPUs per allocated GPU; steps inheriting it imply --exact.
    --cpus-per-task / -c   "job steps will require ncpus processors per task". Since
                           22.05 it propagates to srun as SRUN_CPUS_PER_TASK.
    --mem-per-gpu          memory per GPU -- here FORBIDDEN by site policy, like --mem.
    CUDA_VISIBLE_DEVICES   "set for each job step ... restricted to allocated GPUs", and
                           with cgroup device fencing it is RENUMBERED from 0 inside the
                           job (the Prolog sees 1, the job sees 0). That is why parsl's
                           `nvidia-smi -L` count equals the allocation and why worker ids
                           are card indices 0..G-1 and nothing else.
    SLURM_GPUS             "total number of GPUs allocated to the job"
    SLURM_GPUS_ON_NODE     "number of GPUs allocated to the job on each node"
    SLURM_JOB_GPUS         "global GPU IDs allocated to the job" (comma list)
    SLURM_CPUS_PER_TASK    what -c asked for;  SLURM_CPUS_ON_NODE  CPUs on this node
    #SBATCH parsing        stops at "the first non-comment, non-whitespace line" -- the
                           `if` in ai.slurm comes AFTER every directive for that reason.
    MPS                    "GRES types of GPU and MPS can not be requested within a
                           single job" and only one user's MPS server per node -- so MPS
                           is not something a job here can switch on for itself.

    THE SHAPE THIS PROJECT USES: one task, many worker PROCESSES on it:
        --nodes=1 --ntasks=1 --gpus=G --cpus-per-task=12G
    not --gpus-per-task (that is for one process per card, MPI-style). Parsl's worker
    pool is the fan-out, and it pins each worker to a card itself.

STORAGE: TWO FILESYSTEMS, AND WHICH CLUSTERS SHARE ONE  (storage tags read 2026-09-11)
--------------------------------------------------------------------------------------
    XYFS02     tianhexy-cn, tianhexy-a, k8s_xingyi, k8s_xingyiAI
    XYAIFS00   tianhexy-ai, k8s_xingyiAI_2

**TianheXY-CN and TianheXY-A share XYFS02.** A basin list branch A writes on deimos is
already there for branch B on `ai`: no transfer, same relative path, same repository.
TianheXY-AI (h100x) is on XYAIFS00 and everything it needs must be copied there and
back -- `hpc/tools/xfer_tianhe_ai.sh` is the manual's 3.2.2 scp recipe with the paths
filled in. Prefer this cluster for branch B for that reason too.

MEASUREMENT HISTORY (kept; each was true of what it measured)
-------------------------------------------------------------
    2026-09-05  `sinfo` on ln206 (default env): ai 15/1/9/25 nodes, 56 cpus, 1030000 MB;
                temp 1 node, 2 h ceiling. 9 of 25 `ai` nodes in state O.
    2026-09-07  modules from /APP/u22/ai_x86/modulepath/: CUDA/11.3 ... 12.3 (ceiling; no
                12.4), cudnn/8.9.6.50-cuda12, nccl/2.19.3-cuda-12.3, no MPI loaded.
                Node: 2 x 28 = 56 cores, 8 cards, 1024 GB, 80 GB HBM2e/card, driver
                535.104.12. Layout then: 56 workers, 7 per card.
    2026-09-11  default env: Gres=(null) on all 25 `ai` nodes; `--gpus=8` refused.
    2026-09-11  parsl source read: the ZeroDivisionError fallback above.
    2026-09-11  site PDF: fine-grained env, 1gpu/12cpu/120GB, -G mandatory, --mem
                forbidden, an[44-53] (2024 screenshot), billing rule. Driver 535.104.12,
                CUDA 12.2 in the screenshots (module CUDA/12.3 still loads).
    2026-09-11  fine-grained env read directly (evening): 8 nodes an[45-47,49-51,53,65],
                each 96 CPUs (2x24 cores, HT) / 960 GB / gpu:a800:8, Slurm 24.05.1,
                OverSubscribe=NO, TRESBillingWeights CPU=1 GPU=4, MaxSubmit=10, no
                GrpTRES. At that reading ALL 64 CARDS WERE ALLOCATED while 248 of 672
                CPUs on the mixed nodes were idle -- cards are the scarce resource, and
                a new job queues on cards, not on CPUs. Default env at the same time:
                `temp` (2 h) exists there; 22 of 25 `ai` nodes drain/drng/inval.

WHAT RUNS HERE
--------------
    qha    branch B production trajectories: OpenMM + openqha/openmm_mace.py, CUDA.
    train  branch C MACE fine-tuning, one model per card.

    CAVEAT: this repository's only GPU measurement of branch B is D0-C-5, 3.5x SLOWER on
    a T400 than on CPU. Not transferable to an 80 GB card, and not yet replaced. Take
    seconds-per-ps off the first real job before sizing a campaign.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import labels as _labels  # noqa: E402

# =========================================================================================
# SETTINGS
# =========================================================================================
ACCOUNT = None

#: The script that switches a shell from the default Slurm environment to the fine-grained
#: one. From the site PDF, section "登录方法" (2). Sourced in `examples/slurm/ai.slurm`;
#: checked for (by its effect, not by an env var) in `examples/run_chain.sh`.
ENV_SCRIPT = "/APP/u22/ai_x86/toolshs/set-XY-I.sh"

#: Production partition and walltime. `ai` = an[45-47,49-51,53,65], 8 nodes (measured
#: 2026-09-11 evening; the PDF's 2024 screenshot showed an[44-53]). MaxTime=UNLIMITED.
PARTITION = "ai"
#: **24 h, not 7 days** (user ruling 2026-09-12: use the machine, do not sit on it).
#: The campaign fits: 02d is EQUIL 50 + PROD 500 = 550 ps = 550k steps, and at the
#: measured ~60 ms/step for 12 workers sharing a card that is ~9.2 h per trajectory with
#: all 12 running at once -- so ~9.5 h wall, inside the 21.6 h task budget below. A
#: shorter walltime also backfills: the scheduler can fit a 24 h job into gaps a 7-day
#: job would never be offered.
#: If a campaign ever needs longer, raise this rather than the budget -- they are tied
#: together by _walltime_seconds() so they cannot drift apart again.
WALLTIME = "24:00:00"

#: Smoke test. **The fine-grained controller has ONE partition** (measured 2026-09-11:
#: `source set-XY-I.sh && sinfo` lists only `ai`). `temp` (2 h, 1 node) exists on the
#: DEFAULT controller only. So a smoke test here is a short job on `ai`.
DEBUG_PARTITION = "ai"
DEBUG_WALLTIME = "00:30:00"

#: **The fine-grained policy** (site PDF), and where the numbers come from (measured):
#: a node is 96 logical CPUs = 2 sockets x 24 cores x 2 threads, 960 GB, 8 x A800; divide
#: by eight. So the 12 CPUs a card brings are **6 physical cores hyper-threaded**, which is
#: why a worker here gets ONE thread (cores_per_worker=1, OMP_NUM_THREADS=1) and not more.
#: Asking for fewer CPUs saves nothing on the invoice; asking for more is another card.
CPUS_PER_GPU = 12
MEM_GB_PER_GPU = 120          #: NEVER requested -- `--mem` is refused. Recorded for sizing.
THREADS_PER_CORE = 2          #: measured; 12 CPUs = 6 cores
GPU_MODEL = "a800"            #: `gpu:a800:8` on every node; 80 GB
SLURM_VERSION = "24.05.1"     #: so --cpus-per-task propagates to srun (>= 22.05)

#: Cards per node, measured (`Gres=gpu:a800:8`, CfgTRES gres/gpu:a800=8, all 8 nodes).
GPUS_PER_NODE = 8
NODES_IN_PARTITION = 8        #: TRES node=8; 64 cards in the partition

#: Slurm's own billing weights on this partition (scontrol show partition ai):
#: CPU=1.0, Mem=0.0, GRES/gpu:a800=4.0. One card + 12 CPUs = 16; a node = 128.
BILLING_WEIGHTS = dict(cpu=1.0, mem=0.0, gpu=4.0)

#: Limits, measured 2026-09-11 (`sacctmgr show assoc user=$USER`): MaxSubmit=10, no
#: GrpTRES, no MaxTRES, no MaxJobs. QoS available: emergency, normal, st(andard?) --
#: truncated by sacctmgr; none is passed. There is NO node quota in this environment; the
#: partition's 8 nodes are the bound.
JOB_QUOTA = 10
NODE_QUOTA = NODES_IN_PARTITION

#: CUDA module (2026-09-07 module tree): 12.3 is the ceiling. `environment-tianhe-gpu.yml`
#: pins the same. Change them together or neither.
#: **12.2, not 12.3, and the reason is PTX.** The driver here is 535.104.12,
#: which supports CUDA 12.2. Minor version compatibility lets a 12.3-built
#: CUBIN run on it -- which is why torch and MACE were fine -- but it does NOT
#: cover PTX JIT, and OpenMM compiles every kernel at run time. 2026-09-12: all
#: twelve branch B trajectories died with CUDA_ERROR_UNSUPPORTED_PTX_VERSION.
#: `openqha/gpu_preflight.py` now measures both numbers and refuses first.
CUDA_VERSION = "12.2"
NCCL_MODULE = "nccl/2.19.3-cuda-12.3"     #: recorded, not loaded
MPI_MODULE = None
CUDNN_MODULE = "cudnn/8.9.6.50-cuda12"    #: recorded, not loaded

#: Never. In a per-card environment this would take the whole node and be billed as 8
#: cards. (The 2026-09-05 ruling reached the same value for a different reason.)
EXCLUSIVE = False

#: Layouts, per CARD.
QHA_WORKERS_PER_GPU = CPUS_PER_GPU        # 12 trajectories share a card
QHA_CORES_PER_WORKER = 1
TRAIN_WORKERS_PER_GPU = 1                 # a fine-tune uses the card
TRAIN_CORES_PER_WORKER = CPUS_PER_GPU

#: Nested mode only: cards per parsl block, and how many blocks.
#:
#: **A WHOLE NODE PER BLOCK** (user ruling 2026-09-12: make full use of the machine).
#: 8 cards x 12 workers = 96 workers = 96 CPUs = exactly one node, because the
#: fine-grained policy hands out 12 CPUs with every card. Nothing is left idle and
#: nothing is over-requested; the block IS the node.
#:
#: **The cost of this choice is queue time, and it is not small.** Measured 2026-09-11:
#: all 64 cards in the partition were allocated while 248 of 672 CPUs sat idle -- cards
#: are the scarce resource and a job queues on cards. Asking for 8 free cards ON ONE NODE
#: is a much rarer event than asking for 1, so a full-node block can wait where eight
#: single-card blocks would already be running. If a campaign is queueing rather than
#: computing, drop this to 1 and let parsl scale out instead; the arithmetic below
#: follows either way.
#:
#: **And 96 workers on one node is NOT the configuration that was measured.** The curve
#: in SIZING (12 workers, peak throughput) had ONE card busy and 84 of the node's 96 CPUs
#: idle. At 96 workers the node's 48 physical cores are two-way oversubscribed and every
#: worker competes for memory bandwidth -- and there is already evidence that bites: on
#: the CPU-only sweep, 12 workers (of 96 CPUs) slowed each other from 98.9 to 186.3
#: ms/step. Before a long campaign runs this way, measure it:
#:     python scripts/tooling/s0_gpu_concurrency.py --dtype float64 --precision double \
#:         --threads 1 --steps 200 12 24 48 96      # inside an 8-card allocation
#: **ONE CARD PER BLOCK for the first campaign** (user ruling 2026-09-12, after the
#: full-node setting above was written and before anything ran with it). Both reasons
#: from that comment stand and neither is hypothetical:
#:   * cards are the scarce resource -- 2026-09-11, all 64 allocated while 248 of 672
#:     CPUs idled -- so 8 free cards ON ONE NODE is a far rarer event than 1, and a
#:     full-node block can queue while single-card blocks compute;
#:   * 96 workers on a node has never been measured (open item 46).
#: Setting this to GPUS_PER_NODE turns the blocks into whole nodes again; the arithmetic
#: below follows either value, and nothing else needs changing.
GPUS_PER_BLOCK = 1
NODES_PER_BLOCK = 1
#: MaxSubmit is 10 and in nested mode the driver's own job is one of them. Capped at the
#: partition's node count too, which only binds when a block is a whole node.
MAX_BLOCKS = min(JOB_QUOTA - 1, NODES_IN_PARTITION * GPUS_PER_NODE // GPUS_PER_BLOCK)

def _walltime_seconds(spec):
    """Slurm walltime -> seconds. Accepts D-HH:MM:SS, HH:MM:SS, MM:SS.

    Written because the budget below used to hard-code `7 * 24 * 3600` beside a WALLTIME
    that could be edited independently: changing one silently left the other behind, and
    the failure mode is a task that thinks it has six more days than the job does.
    """
    days, _, rest = str(spec).partition("-")
    if not rest:
        days, rest = 0, str(spec)
    parts = [int(x) for x in rest.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts[-3:]
    return int(days) * 86400 + h * 3600 + m * 60 + s


#: Per-task budget at 90% of the walltime, so a branch B task stops and flushes its last
#: chunk rather than being killed between a write and a rename. DERIVED from WALLTIME.
QHA_WALL_BUDGET_S = int(0.90 * _walltime_seconds(WALLTIME))


def parsl_walltime(spec):
    """HH:MM:SS with unbounded hours -- the only form parsl's providers parse. Slurm's
    `D-HH:MM:SS` dies there as `invalid literal for int(): '3-00'` (tianhe_ai, 2026-09-13)."""
    total = _walltime_seconds(spec)
    return "{:02d}:{:02d}:{:02d}".format(total // 3600, (total % 3600) // 60, total % 60)

OPENMM_PLATFORM = "CUDA"
USE_MPS = False

#: Kept for anything that still reads the node-shaped names.
CORES_PER_NODE = GPUS_PER_NODE * CPUS_PER_GPU             # 96 if the whole node were taken
QHA_WORKERS_PER_NODE = GPUS_PER_NODE * QHA_WORKERS_PER_GPU
WORKERS_PER_NODE = QHA_WORKERS_PER_NODE
CPUS_PER_WORKER = QHA_CORES_PER_WORKER
WORKERS_PER_CARD = QHA_WORKERS_PER_GPU
# =========================================================================================

#: Filled by config(): the placement the last built pool will get.
LAST_PLACEMENT = {}

#: role -> (workers per CARD, cores per worker)
_LAYOUT = {
    "qha": (QHA_WORKERS_PER_GPU, QHA_CORES_PER_WORKER),
    "train": (TRAIN_WORKERS_PER_GPU, TRAIN_CORES_PER_WORKER),
    # The qha chain's THIRD step (s0_E_branchB_collect_parsl.py -> s0_B_qha_analyse.py,
    # one molecule per core, no card) runs in the same allocation right after the
    # trajectories. Until 2026-09-13 neither GPU config had this role, so the first chain
    # that ever got past branch B on a card died here with
    # "role 'collect' does not run on ...". The CPUs are the card's own 12; the card idles.
    "collect": (CPUS_PER_GPU, 1),
}


def layout(role):
    """(workers_per_card, cores_per_worker) for a role. Unknown roles raise."""
    if role not in _LAYOUT:
        raise KeyError(
            "role {!r} does not run on TianheXY-A. This cluster serves {}.\n"
            "Branch A (CREST) is CPU work -- see hpc/resource_configs/tianhe_cpu.py."
            .format(role, " and ".join(sorted(_LAYOUT))))
    return _LAYOUT[role]


def cards_for(workers, gpus):
    """Which card each of `workers` workers lands on, by parsl's own arithmetic.

    Reproduced from `process_worker_pool.py:738-745` so a misplacement is readable before
    submission rather than after. `gpus` is what `nvidia-smi -L` will report inside the
    job -- in the fine-grained environment, exactly the cards allocated.
    """
    per_card = workers // max(1, gpus)
    if per_card == 0:                       # ZeroDivisionError branch: id used as-is
        return list(range(workers))
    return [i // per_card for i in range(workers)]


def size_pool(workers, gpus, cores):
    """Round a pool so parsl places it correctly, and say what it costs.

    Returns (workers, gpus, cpus_to_request). `workers` is rounded UP to a multiple of
    `gpus` (a few idle workers cost nothing -- the CPUs are paid for with the card; a
    worker on a non-existent card costs a whole trajectory's wall clock). `cpus_to_request`
    is the full CPUS_PER_GPU x gpus, because that is what is billed regardless.
    """
    gpus = max(1, int(gpus))
    workers = max(gpus, int(workers))
    if workers % gpus:
        workers = gpus * math.ceil(workers / gpus)
    cpus = gpus * CPUS_PER_GPU
    if workers * cores > cpus:
        raise ValueError(
            "{} workers x {} cores = {} CPUs, but {} card(s) bring {} x {} = {}. "
            "Either fewer workers or more cards; the policy is fixed at {} CPUs per card."
            .format(workers, cores, workers * cores, gpus, gpus, CPUS_PER_GPU, cpus,
                    CPUS_PER_GPU))
    return workers, gpus, cpus


def _allocated_gpus():
    """How many cards THIS job holds, from Slurm's own variables, else nvidia-smi, else 1.

    Inside a fine-grained allocation Slurm restricts the visible cards, so every one of
    these agrees; the order is cheapest first.
    """
    for var in ("SLURM_GPUS_ON_NODE", "SLURM_GPUS", "SLURM_JOB_GPUS"):
        v = os.environ.get(var, "")
        if v:
            try:
                return int(v) if v.isdigit() else len([x for x in v.split(",") if x])
            except ValueError:
                pass
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cvd:
        return len([x for x in cvd.split(",") if x])
    try:
        import subprocess
        out = subprocess.run("nvidia-smi -L", shell=True, capture_output=True, text=True,
                             timeout=20)
        n = len([l for l in out.stdout.splitlines() if l.startswith("GPU ")])
        if out.returncode == 0 and n:
            return n
    except Exception:                                   # noqa: BLE001
        pass
    return 1


def _allocated_cpus():
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "SLURM_JOB_CPUS_PER_NODE"):
        v = os.environ.get(var, "")
        if v and v.split("(")[0].isdigit():
            return int(v.split("(")[0])
    return os.cpu_count() or 1


def in_allocation_now():
    """True when the caller is already inside a Slurm job and has not asked to nest.

    `S0_PARSL_NESTED=1` forces the nested mode from inside a job -- for a campaign that
    wants parsl to submit further blocks.
    """
    return bool(os.environ.get("SLURM_JOB_ID")) and os.environ.get("S0_PARSL_NESTED") != "1"


def _worker_init(here):
    """Modules first -- compute nodes are a minimal environment.

    Only CUDA. No MPI: independent single-card workers exchange nothing.

    The SLURM_/PMI_ unset and the `mkdir -p` are rules 7 and 6 of this project's HPC
    skill: inherited task-layout variables make a child process try to relaunch itself
    through the scheduler, and Slurm opens `--output` before the script runs.

    Module failures are NOT silenced. A missing CUDA here means every task in the block
    falls back to the CPU and takes far longer for reasons that appear nowhere.

    **`CUDA_VISIBLE_DEVICES` must not be set here.** worker_init runs once per BLOCK,
    before the worker pool starts, so anything set here is inherited identically by every
    worker. Parsl assigns the card per worker; see the module docstring.
    """
    return "; ".join([
        "mkdir -p ${S0_RUNS_ROOT:-$HOME/runs/openQHA}/logs",
        "module purge 2>/dev/null || true",
        "module load anaconda3/2023.09 2>/dev/null || "
        "module load miniforge/24.7.1 2>/dev/null || true",
        # No unconditional CUDA module here any more (2026-09-13). Whether a site module
        # is needed at all is a MEASUREMENT -- the environment's nvrtc against the driver
        # -- and examples/chain_body.sh makes it (openqha_cuda_fit) before the driver
        # starts, exporting OPENQHA_CUDA_MODULE_CHOSEN when, and only when, a module was
        # required and fit. Workers load exactly that. On an104 the old line ran in a
        # shell with no `module` command and printed "FAILED -- tasks will not see a
        # card" while the tasks saw the card fine; on the same cluster the CUDA/12.4
        # module puts lib64/stubs on LD_LIBRARY_PATH, which is CUDA error 34 for every
        # task. CUDA_VERSION stays as the recorded ceiling for the environment file.
        'if [ -n "${OPENQHA_CUDA_MODULE_CHOSEN:-}" ]; then module load "$OPENQHA_CUDA_MODULE_CHOSEN" 2>/dev/null || echo "openQHA: module load $OPENQHA_CUDA_MODULE_CHOSEN failed in this block; using the environment\'s own toolkit" >&2; fi',
        'case ":${LD_LIBRARY_PATH:-}:" in *stubs*) export LD_LIBRARY_PATH="$(printf \'%s\' "$LD_LIBRARY_PATH" | tr : \'\\n\' | grep -v stubs | paste -sd: -)";; esac',
        "for v in $(env | awk -F= '{print $1}' | grep -E '^(PMI|SLURM)_'); do "
        "unset $v; done",
        "source {0}/env/common.sh".format(here),
        "source {0}/env/tianhe.sh".format(here),
        "openqha_report_env",
    ])


def config(partition=None, account=None, nodes_per_block=None, max_blocks=None,
           walltime=None, run_dir=None, worker_init=None, max_workers=None,
           gpus=None, exclusive=None, role="qha", debug=False, in_allocation=None):
    """Parsl Config for TianheXY-A's fine-grained environment.

    `role` is `qha` (12 workers per card) or `train` (1 per card). `gpus` is cards per
    block in nested mode, and is ignored in-allocation (the job already has its cards).
    `max_workers` caps the pool; in-allocation it defaults to what the cards bring.

    `in_allocation=None` means: look. Inside a Slurm job (and without S0_PARSL_NESTED=1)
    the workers run in THIS job under a LocalProvider. Otherwise parsl submits blocks
    with `--gpus=N` through `TianheSlurmProvider`.

    `debug=True` swaps in DEBUG_PARTITION and DEBUG_WALLTIME and caps at one block.
    """
    from parsl.config import Config
    from parsl.executors import HighThroughputExecutor
    from parsl.launchers import SimpleLauncher

    per_card, cores = layout(role)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    nested = not (in_allocation_now() if in_allocation is None else in_allocation)

    if exclusive:
        raise ValueError(
            "--exclusive would allocate the whole node in a per-card environment and be "
            "billed as {} cards. Not offered here.".format(GPUS_PER_NODE))

    pin = True
    if nested:
        # -------- the driver will submit blocks of `gpus` cards each --------------------
        from providers import TianheSlurmProvider
        gpus = int(gpus or GPUS_PER_BLOCK)
        workers = int(max_workers or per_card * gpus)
        workers, gpus, cpus = size_pool(workers, gpus, cores)
        blocks = MAX_BLOCKS
        if debug:
            partition = partition or DEBUG_PARTITION
            walltime = walltime or DEBUG_WALLTIME
            blocks = 1
        provider = TianheSlurmProvider(
            partition or PARTITION,
            account=account if account is not None else ACCOUNT,
            nodes_per_block=nodes_per_block or NODES_PER_BLOCK,
            # Rendered as `#SBATCH --cpus-per-task=N`. It is the CPUs the cards bring,
            # not the worker count: the bill is the same and idle CPUs are free.
            cores_per_node=cpus,
            # `--mem` is FORBIDDEN in this environment (site PDF, 注意事项 3). parsl only
            # renders it when mem_per_node is given, so it is never given.
            mem_per_node=None,
            init_blocks=0, min_blocks=0,
            max_blocks=min(int(max_blocks or blocks), JOB_QUOTA),
            # `-G N` / `--gpus=N`: MANDATORY here (site PDF, 注意事项 1). The `--gpus=`
            # spelling is the one that has been accepted on both GPU clusters.
            scheduler_options="#SBATCH --gpus={}".format(gpus),
            exclusive=False,
            launcher=SimpleLauncher(),
            worker_init=worker_init or _worker_init(here),
            walltime=parsl_walltime(walltime or WALLTIME),
            cmd_timeout=60,
        )
    else:
        # -------- run inside the job the driver already holds ---------------------------
        from parsl.providers import LocalProvider
        card = os.environ.get("S0_CARD")
        if card is not None:
            # **ONE CARD OF A MULTI-CARD JOB.** examples/02d-2 packs G drivers into a
            # `--gpus=G` job, one setting per card, each launched with
            # `S0_CARD=k CUDA_VISIBLE_DEVICES=k`. Parsl must then NOT pin: its own
            # arithmetic uses `nvidia-smi -L`, which ignores CUDA_VISIBLE_DEVICES and
            # reports all G cards, and its fallback writes the worker's index into
            # CUDA_VISIBLE_DEVICES -- so every driver's workers would land on cards
            # 0, 1, 2 and collide. With no `available_accelerators` parsl leaves the
            # variable alone and the workers inherit the one card they were given.
            gpus, pin = 1, False
        else:
            gpus, pin = _allocated_gpus(), True
        cpus_here = _allocated_cpus() if card is None else CPUS_PER_GPU
        workers = int(max_workers or per_card * gpus)
        workers = min(workers, max(gpus, cpus_here // max(1, cores)))
        workers, gpus, _cpus_billed = size_pool(workers, gpus, cores)
        provider = LocalProvider(
            init_blocks=1, min_blocks=1, max_blocks=1,
            launcher=SimpleLauncher(),
            worker_init=worker_init or _worker_init(here),
        )

    cfg = Config(
        executors=[
            HighThroughputExecutor(
                label=_labels.label(role),
                max_workers_per_node=workers,
                cores_per_worker=float(cores),
                # AN INT, NOT A LIST -- see the module docstring for parsl's arithmetic
                # and for what a list does. Passed whenever parsl is the one placing the
                # workers on the job's cards; withheld under S0_CARD (see above).
                **(dict(available_accelerators=workers) if pin else {}),
                provider=provider,
            ),
        ],
        # Per card under S0_CARD: G drivers starting at once would otherwise race for
        # parsl's numbered runinfo/NNN directories in one run_dir.
        run_dir=run_dir or os.path.join(
            os.environ.get("S0_RUNS_ROOT",
                           os.path.expanduser("~/runs/openQHA")),
            "parsl" + ("_card{}".format(os.environ["S0_CARD"])
                       if os.environ.get("S0_CARD") is not None else "")),
        retries=1,
        strategy="simple",
    )
    _labels.check(cfg, expect=_labels.label(role))
    # Legible before submission: the placement this pool will get. Module-level rather
    # than an attribute on the Config, which is a typed object.
    global LAST_PLACEMENT
    LAST_PLACEMENT = dict(
        mode="in-allocation" if not nested else "nested",
        workers=workers, gpus=gpus, cores_per_worker=cores,
        pinned_by_parsl=pin, card=os.environ.get("S0_CARD"),
        cards=cards_for(workers, gpus) if pin else [int(os.environ["S0_CARD"])] * workers)
    return cfg


def describe():
    try:
        import providers
        commands = dict(providers.COMMANDS["tianhe"])
        confirmed = list(providers.TIANHE_CONFIRMED)
    except Exception:                                    # pragma: no cover
        commands, confirmed = {}, []
    hist = {}
    for card in cards_for(QHA_WORKERS_PER_GPU * GPUS_PER_BLOCK, GPUS_PER_BLOCK):
        hist[card] = hist.get(card, 0) + 1
    return dict(
        site="tianhe_a", cluster="TianheXY-A", environment="fine-grained (tianhexy-i)",
        env_script=ENV_SCRIPT,
        partition=PARTITION, walltime=WALLTIME,
        debug_partition=DEBUG_PARTITION, debug_walltime=DEBUG_WALLTIME,
        partitions_available=["ai"],           # the only one; measured 2026-09-11
        policy=dict(cpus_per_gpu=CPUS_PER_GPU, mem_gb_per_gpu=MEM_GB_PER_GPU,
                    threads_per_core=THREADS_PER_CORE, gpu_model=GPU_MODEL,
                    gpu_request="mandatory (-G / --gpus / --gres)",
                    mem_request="forbidden",
                    billing_site="max(gpus, ceil(cpus/12)) card-equivalents x hours x weight",
                    billing_slurm_weights=BILLING_WEIGHTS),
        nodes=["an45", "an46", "an47", "an49", "an50", "an51", "an53", "an65"],
        slurm_version=SLURM_VERSION,
        gpus_per_node=GPUS_PER_NODE, cores_per_node=CORES_PER_NODE,
        gpus_per_block=GPUS_PER_BLOCK, exclusive=EXCLUSIVE,
        layouts={r: dict(workers_per_card=w, cores_per_worker=c,
                         workers_per_node=w * GPUS_PER_NODE)
                 for r, (w, c) in _LAYOUT.items()},
        workers_per_node=QHA_WORKERS_PER_NODE, qha_workers_per_node=QHA_WORKERS_PER_NODE,
        cpus_per_worker=QHA_CORES_PER_WORKER, workers_per_card=QHA_WORKERS_PER_GPU,
        card_assignment_histogram=hist,
        modes=dict(
            in_allocation="default inside a Slurm job: LocalProvider, the job's own cards",
            nested="S0_PARSL_NESTED=1 or a login-node driver: blocks of GPUS_PER_BLOCK "
                   "cards, up to MAX_BLOCKS"),
        max_blocks=MAX_BLOCKS, node_quota=NODE_QUOTA, job_quota=JOB_QUOTA,
        max_tasks_in_flight={r: MAX_BLOCKS * w * GPUS_PER_BLOCK
                             for r, (w, _c) in _LAYOUT.items()},
        roles=sorted(_LAYOUT),
        strategy=("pay in cards, use every CPU a card brings: qha 12 workers per card, "
                  "train 1 per card. One-molecule examples run in-allocation."),
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
            "ONE login node, TWO Slurm environments. This file targets the fine-grained "
            "one, entered by sourcing " + ENV_SCRIPT + ".",
            "Fine-grained policy (site PDF): 1 GPU = 12 CPUs = 120 GB; -G mandatory; "
            "--mem forbidden; billed by max(gpus, ceil(cpus/12)).",
            "The default environment (no script sourced) has Gres=(null) and refuses any "
            "card request -- measured 2026-09-11. Same prompt, different controller.",
            "In-allocation is the default inside a job: no second allocation, no idle "
            "driver card. S0_PARSL_NESTED=1 restores block submission for campaigns.",
            "CUDA 12.3 is the module ceiling (2026-09-07); the PDF's nodes show driver "
            "535.104.12 / CUDA 12.2.",
            "Shape: --nodes=1 --ntasks=1 --gpus=G --cpus-per-task=12G. One task, parsl "
            "fans out the processes and pins each to a card (slurm.schedmd.com, read "
            "2026-09-11: CUDA_VISIBLE_DEVICES is per-step, restricted, renumbered from 0).",
            "Storage: XYFS02 is shared by tianhexy-cn and tianhexy-a -- branch A products "
            "need no transfer to reach branch B here. tianhexy-ai is on XYAIFS00; use "
            "hpc/tools/xfer_tianhe_ai.sh.",
            "Measured 2026-09-11: 8 nodes x (96 CPU, 960 GB, 8 x A800), Slurm 24.05.1, "
            "billing CPU=1 GPU=4, MaxSubmit=10, no node quota. All 64 cards were allocated "
            "at the reading: expect to queue on cards.",
        ],
        assumptions=[
            "12 trajectories per card is an ARGUMENT (a 10-19 atom molecule cannot fill "
            "an 80 GB card), not a measurement; without MPS they time-slice.",
            "12 trajectories per card time-slice without MPS; the per-ps cost on an A800 "
            "is unmeasured until the first real job.",
            "12 CPUs per card are 6 physical cores hyper-threaded; 12 single-thread "
            "workers per card is therefore 2 per core. Fine for a launch-latency-bound "
            "MACE call chain, unmeasured.",
            "Branch B on a card is UNMEASURED here (D0-C-5 was a T400, 3.5x slower "
            "than CPU).",
        ],
        verified=False,
    )
