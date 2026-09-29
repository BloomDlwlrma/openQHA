"""Parsl resource configuration: Tianhe CPU cluster (TianheXY-C). Branch A + collection.

Edit SETTINGS, not the command line. Production is then

    python -u scripts/production/s0_E_branchA_parsl.py --edges --resource tianhe_cpu

TWO PARTITIONS, AND ONLY TWO
-----------------------------------------------------
    debug    30 minutes    the smoke test. One edge, real work, a real queue.
    deimos   3 days        production. **The only partition production uses.**

`xyfree`, `mars` and `e9` exist on this machine and are deliberately not used: one
production queue means one set of costs to compare against, and a run that lands in a
different queue is not comparable with the one before it. They are recorded in
`PARTITIONS_NOT_USED` so that "why not mars" has an answer.

**There is no `--dry-run` step in this workflow any more.** A 30-minute `debug` job is
cheaper than the argument about whether a rendered plan would have worked, and it tests
the things a plan cannot: that the modules load, that conda activates on a compute node,
that the weights resolve there, and that Parsl's status query is understood. Render
the script if you want to read it (`providers.TianheSlurmProvider.render_only`), but the
gate before production is a debug job that finished.

WHAT RUNS HERE
-------------------------------------------
    role       layout            what it is
    crest      16 x 4 threads    branch A. CREST iMTD-GC, GFN2-xTB workhorse.
    collect    64 x 1 core       branch B results: one molecule per core, 64 at a time,
                                 ON ONE NODE. This is the analysis pass that turns
                                 frames into a conformational correction -- not the
                                 trajectories, which now run on GPU (tianhe_a.py).
    qha        64 x 1 core       the ASE-route CPU fallback for branch B trajectories.
                                 KEPT, not deleted: it is the implementation pair that
                                 makes the OpenMM route checkable. Not the production
                                 route.
    labels     16 x 4 ranks      Hessian learning: one ORCA single point + analytic
                                 Hessian per frame (workflows/hessian_learning/03_labels.py),
                                 `%pal nprocs 4`, `%maxcore 6000`; 1 node for the 7-molecule
                                 smoke set, 12 nodes for the 200-molecule draw.
                                 ORCA 6.1.1 from the conda env `orca611`
                                 (`~/env_orca611.sh`), located by hpc/env/orca.sh without
                                 activating it in the worker. deimos: 7 days, 512 GB.

Branch A belongs here and cannot move: its cost is ~1e5 GFN2-xTB gradient calls and xtb
has no GPU path at all.

THE ARITHMETIC PRODUCTION IS SIZED TO
-------------------------------------
    64 cores per node        manual section 1.1, 2 x 32
    4 threads per molecule   the condition the 285 s/species cost was measured under
    -> 16 molecules per node, no oversubscription
    x 12 nodes               MAX_BLOCKS
    = 192 molecules in flight

    285 s per species        measured, 4 threads, UNCONTENDED, on this project's
                             workstation. NOT measured here, and not under
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

#: Production partition and its walltime: production is deimos,
#: 3 days, and nothing else. (deimos allows 7 days; 3 days is kept as
#: the default, a labels Batch that needs more passes `--walltime`.)
PARTITION = "deimos"
WALLTIME = "3-00:00:00"

#: Smoke-test partition and its walltime. `--debug` on the drivers selects this pair.
#: 30 minutes is enough for one edge (two molecules at ~285 s each, uncontended) and
#: short enough that the queue schedules it almost at once.
DEBUG_PARTITION = "debug"
DEBUG_WALLTIME = "00:30:00"

#: Present on the machine, deliberately unused. Recorded so the choice is auditable.
PARTITIONS_NOT_USED = {
    "xyfree": "trial-user partition",
    "mars": "no runtime limit; only worth it for something that cannot be cut into blocks",
    "e9": "fine-grained, allocated per CPU rather than per node. Would matter for a run "
          "that cannot fill 64 cores -- neither of ours is: crest fills it with 16 "
          "molecules and collect with 64.",
}

CORES_PER_NODE = 64            #: manual section 1.1: 2 x 32
THREADS_PER_JOB = 4            #: CREST threads per molecule; changing it invalidates 285 s
WORKERS_PER_NODE = CORES_PER_NODE // THREADS_PER_JOB      # 16

NODES_PER_BLOCK = 1

#: Node quota for this cluster (Starlight, 2026-09-05): 32 running jobs, 32 nodes.
#: `max_blocks * nodes_per_block` must not exceed it -- Parsl does not know about the
#: quota, so beyond it the scheduler simply refuses and the run looks stalled, not capped.
NODE_QUOTA = 32
JOB_QUOTA = 32

#: Allocations held at once for branch A. **12, not the 32 the quota allows**:
#: 12 x 16 = 192 molecules in flight. Raising it to the quota is one
#: number, but do it after a campaign has shown the queue rather than the node count is
#: the limit. `init_blocks=0` and `min_blocks=0` still mean nothing is requested until
#: there is work, and an idle allocation is given back -- a held node is charged whether
#: or not it computes.
MAX_BLOCKS = 12

#: Branch B: ONE core per molecule, 64 on ONE node.
#:
#: Not a preference -- measured. A quasi-harmonic trajectory is a serial chain of MACE
#: force calls, and MACE on a 10-atom molecule runs 111/90/72/101 ms at 1/2/4/8 threads.
#: Four threads buy 1.54x; four independent tasks buy 4x.
QHA_THREADS_PER_JOB = 1
QHA_WORKERS_PER_NODE = CORES_PER_NODE // QHA_THREADS_PER_JOB      # 64

#: The collection pass is deliberately ONE node. It reads frames and writes a number per
#: molecule; it is short, it is I/O-bound on Lustre, and 64 concurrent readers on one
#: node is already more metadata traffic than the filesystem enjoys. Widening this is the
#: wrong lever -- if collection is the bottleneck, batch more molecules per task.
COLLECT_MAX_BLOCKS = 1

#: Driver defaults, so the command line stays `--edges --resource tianhe_cpu`.
TAG = "prod"
TIMEOUT_S = 14400              #: per MOLECULE, not per job. 4 h; the job gets 3 days.
HESSIAN_MODE = "analytic"

#: Per-task budget for branch B tasks, at 90% of the walltime so a task stops and flushes
#: rather than being killed mid-chunk.
QHA_WALL_BUDGET_S = int(0.90 * 3 * 24 * 3600)

#: Hessian-learning labels: 4 ORCA ranks per frame, 16 frames per node, the
#: same division of the node as CREST; `%maxcore` per rank at the 75 % rule on deimos's
#: 512 GB: 512 x 1024 x 0.75 / 64 = 6144 -> 6000 MB. 16 x 4 x 6 GB = 384 GB per node.
LABELS_RANKS_PER_JOB = 4
LABELS_WORKERS_PER_NODE = CORES_PER_NODE // LABELS_RANKS_PER_JOB      # 16
LABELS_MAXCORE_MB = 6000
# =========================================================================================

#: role -> (workers per node, cores per worker, max blocks)
_LAYOUT = {
    "crest":   (WORKERS_PER_NODE, THREADS_PER_JOB, MAX_BLOCKS),
    "qha":     (QHA_WORKERS_PER_NODE, QHA_THREADS_PER_JOB, MAX_BLOCKS),
    "collect": (QHA_WORKERS_PER_NODE, QHA_THREADS_PER_JOB, COLLECT_MAX_BLOCKS),
    "labels":  (LABELS_WORKERS_PER_NODE, LABELS_RANKS_PER_JOB, MAX_BLOCKS),
}


def layout(role):
    """(workers_per_node, cores_per_worker, max_blocks) for a role. Unknown roles raise."""
    if role not in _LAYOUT:
        raise KeyError(
            "role {!r} has no layout on this cluster. Known: {}.\n"
            "Branch C training and branch B trajectories run on the GPU clusters -- "
            "see hpc/resource_configs/tianhe_a.py.".format(role, sorted(_LAYOUT)))
    return _LAYOUT[role]


def _worker_init(here, role="crest"):
    """What every worker runs first. Three of these are not optional.

    * `module load` -- compute nodes are a minimal environment (the manual says only sh
      and bash are supported there), so nothing is on PATH unless it is loaded.
    * **Unset inherited SLURM_/PMI_ variables.** They make a child process misread the
      task layout and try to relaunch itself through the scheduler. Rule 7 of this
      project's own HPC skill, learnt on ORCA under xargs; CREST forks its own parallel
      workers and is exposed to the same thing.
    * `mkdir -p` the log directory -- Slurm opens `--output` BEFORE the script runs, so
      creating it inside the script is too late (rule 6 of the same skill).

    The module load is NOT silenced. A missing module here means the whole allocation
    computes nothing; `2>/dev/null || true` would turn that into an unexplained
    ModuleNotFoundError several minutes later, in a worker log nobody is reading.
    """
    lines = [
        "mkdir -p ${S0_RUNS_ROOT:-$HOME/HDD_POOL/runs/openQHA}/logs",
        "module purge 2>/dev/null || true",
        # the site's module carries no dot: anaconda3/202309 (measured 2026-09-09,
        # docs/tianhe_runbook.md 5c); the dotted name is tried second, and a failure is
        # survivable because hpc/env/tianhe.sh sources ~/init_conda.sh afterwards
        "module load anaconda3/202309 2>/dev/null || module load anaconda3/2023.09 2>/dev/null || "
        "echo 'openQHA: no anaconda3 module loaded -- relying on ~/init_conda.sh from "
        "hpc/env/tianhe.sh' >&2",
        "for v in $(env | awk -F= '{print $1}' | grep -E '^(PMI|SLURM)_'); do "
        "unset $v; done",
        "source {0}/env/common.sh".format(here),
        "source {0}/env/tianhe.sh".format(here),
        "openqha_report_env",
    ]
    if role == "labels":
        # ORCA from its own conda env, located without activating it here (hpc/env/orca.sh).
        # Loud on failure, for the same reason the module load is: a block whose workers
        # have no ORCA computes nothing.
        lines += [
            "source {0}/env/orca.sh".format(here),
            "openqha_find_orca || echo 'openQHA: ORCA NOT FOUND -- every labels task in "
            "this block will fail (hpc/env/orca.sh)' >&2",
        ]
    return "; ".join(lines)


def config(partition=None, account=None, nodes_per_block=None, max_blocks=None,
           walltime=None, run_dir=None, worker_init=None, max_workers=None,
           role="crest", debug=False):
    """Parsl Config for the Tianhe CPU cluster. Every argument defaults to SETTINGS.

    `role` selects the layout, because the jobs want opposite things from the same node:

        crest    16 workers x 4 threads   CREST is internally parallel
        collect  64 workers x 1 core      one molecule per core, one node
        qha      64 workers x 1 core      the CPU fallback route for trajectories
        labels   16 workers x 4 ranks     one ORCA frame label per worker

    THE ONE MODE PARSL IS FOR HERE (after ALF's
    `parsl_resource_configs`): a driver on the login node whose blocks parsl submits
    with `SlurmProvider(partition, init_blocks=0, min_blocks=0, max_blocks, nodes_per_block=1,
    scheduler_options, SimpleLauncher(), walltime='HH:MM:SS', cmd_timeout)` -- the ALF
    shape, with the site's command names and worker init on top. **A job submitted with
    `sbatch` does not run parsl inside it**: the Slurm scripts (hpc/slurm/hl_*.slurm) are
    plain bash + `xargs` over a task list, the hkuhpc shape (an in-allocation
    LocalProvider mode was tried and removed).

    `debug=True` swaps in DEBUG_PARTITION and DEBUG_WALLTIME and caps the run at one
    allocation. An explicit `partition`/`walltime` still wins.
    """
    from parsl.config import Config
    from parsl.executors import HighThroughputExecutor
    from parsl.launchers import SimpleLauncher

    from providers import TianheCNSlurmProvider

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    workers, cores, role_blocks = layout(role)
    if debug:
        partition = partition or DEBUG_PARTITION
        walltime = walltime or DEBUG_WALLTIME
        role_blocks = 1
    # TianheXY-CN is stock Slurm: `sbatch`, not the GPU clusters' `yhbatch`.
    # The CN provider maps nothing; it normalises the walltime (parsl reads HH:MM:SS).
    provider = TianheCNSlurmProvider(
        partition or PARTITION,
        account=account if account is not None else ACCOUNT,
        nodes_per_block=nodes_per_block or NODES_PER_BLOCK,
        init_blocks=0, min_blocks=0,
        # Capped at the site quota, whatever the caller asks for.
        max_blocks=min(int(max_blocks or role_blocks),
                       NODE_QUOTA // int(nodes_per_block or NODES_PER_BLOCK),
                       JOB_QUOTA),
        scheduler_options="",
        # THIS CLUSTER IS THE OPPOSITE OF THE GPU ONES. Its normal partitions allocate a
        # whole node -- the manual calls this exclusive mode -- so `exclusive=True` is
        # correct here, while on TianheXY-AI the flag is banned outright.
        exclusive=True,
        launcher=SimpleLauncher(),
        worker_init=worker_init or _worker_init(here, role),
        walltime=walltime or WALLTIME,
        cmd_timeout=60,
    )
    mode = "nested"

    cfg = Config(
        executors=[
            HighThroughputExecutor(
                # From hpc/labels.py, never a literal: a label that does not match the
                # one the app asks for builds and renders cleanly, then schedules
                # nothing. That is how it failed on 2026-09-05.
                label=_labels.label(role),
                max_workers_per_node=int(max_workers or workers),
                cores_per_worker=float(cores),
                provider=provider,
            ),
        ],
        run_dir=run_dir or os.environ.get("S0_PARSL_RUN_DIR") or os.path.join(
            os.environ.get("S0_RUNS_ROOT",
                           os.path.expanduser("~/HDD_POOL/runs/openQHA")),
            "parsl"),
        retries=1,
        strategy="simple",
    )
    _labels.check(cfg, expect=_labels.label(role))
    global LAST_MODE
    LAST_MODE = mode
    return cfg


LAST_MODE = None            #: "nested": the only mode (see config)


def describe():
    try:
        import providers
        commands = dict(providers.COMMANDS["tianhe_cn"])
        confirmed = ["submit"]           # sbatch on the CPU cluster (user, 2026-09-09)
    except Exception:                                    # pragma: no cover
        commands, confirmed = {}, []
    return dict(
        site="tianhe_cpu", cluster="TianheXY-C", partition=PARTITION,
        walltime=WALLTIME,
        debug_partition=DEBUG_PARTITION, debug_walltime=DEBUG_WALLTIME,
        partitions_used=[DEBUG_PARTITION, PARTITION],
        partitions_not_used=dict(PARTITIONS_NOT_USED),
        cores_per_node=CORES_PER_NODE, threads_per_job=THREADS_PER_JOB,
        workers_per_node=WORKERS_PER_NODE,
        qha_workers_per_node=QHA_WORKERS_PER_NODE,
        qha_threads_per_job=QHA_THREADS_PER_JOB,
        layouts={r: dict(workers_per_node=w, cores_per_worker=c, max_blocks=b)
                 for r, (w, c, b) in _LAYOUT.items()},
        max_blocks=MAX_BLOCKS, node_quota=NODE_QUOTA, job_quota=JOB_QUOTA,
        max_in_flight={r: w * min(b, NODE_QUOTA) for r, (w, _c, b) in _LAYOUT.items()},
        exclusive=True, filesystem="XYFS01 (lustre, 1 TB, NO BACKUP)",
        labels=[_labels.label(r) for r in sorted(_LAYOUT)],
        tag=TAG, timeout_s=TIMEOUT_S, hessian_mode=HESSIAN_MODE,
        qha_wall_budget_s=QHA_WALL_BUDGET_S,
        labels_ranks_per_job=LABELS_RANKS_PER_JOB, labels_workers_per_node=LABELS_WORKERS_PER_NODE,
        maxcore_mb=LABELS_MAXCORE_MB,
        orca_env="conda env orca611 via ~/env_orca611.sh (hpc/env/orca.sh); ORCA 6.1.1 verified on the login node 2026-09-18",
        scheduler_commands=commands, scheduler_commands_confirmed=confirmed,
        notes=[
            "Production uses `deimos` only, at 3 days; `debug` at 30 minutes is the "
            "smoke test. The other three partitions are recorded and unused.",
            "Branch A production is 12 nodes x 16 molecules = 192 in flight, which is "
            "below the 32-node quota on purpose.",
            "Branch B TRAJECTORIES are no longer a CPU job -- they run on TianheXY-A, "
            "one card per trajectory. What stays here is the collection pass: one "
            "molecule per core, 64 at a time, one node.",
            "Normal partitions allocate a whole node (manual 1.1); `e9` is the "
            "fine-grained per-CPU partition if that were ever wasteful.",
            "Home is 100 GB and for configuration only -- run out of HDD_POOL.",
            "Hessian-learning labels: 16 x 4-rank ORCA jobs per node, %maxcore 6000, "
            "node-local scratch (S0_SCRATCH); 1 node for the 7-molecule smoke set, 12 for "
            "the 200-molecule draw.",
            "XYFS01 has no backup: a deleted file cannot be recovered.",
        ],
        assumptions=[
            "The `debug` partition and both walltimes are the user's; "
            "they have not been read off `sinfo` here.",
            "285 s/species was measured on this project's workstation, not here, and "
            "not under contention.",
            "Nothing has been submitted to any Tianhe cluster yet.",
        ],
        verified=False,
    )
