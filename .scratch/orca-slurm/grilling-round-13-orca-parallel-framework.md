# Hessian learning -- grilling, round 13: the ORCA child environment and the parallel framework (2026-09-25)

Scope: the draw300 labels array on TianheXY-CN (`TAG=draw300 sbatch --array=0-11
--time=3-00:00:00 hpc/slurm/hl_labels.slurm`), whose every frame died in ORCA "Startup"
with `[file orca_tools/qcmsg.cpp, line 394]: .... aborting the run` -- the OpenMPI
`ras/slurm` force-terminate, because the job script's narrow unset loop kept
`SLURM_JOBID`/`SLURM_NODELIST` while deleting `SLURM_TASKS_PER_NODE`. Evidence:
[`research-orca-slurm-primary-sources.md`](research-orca-slurm-primary-sources.md) and the
frames' own `.out` files (34 of 34 for `dsgdb9nsd_006885`; the workstation reproduction is
deterministic). The ask: **design the fix and the parallel-framework strategy before
anything is implemented** -- the code change, the regression test, the recovery of the
frames already marked failed, and the docs/ADR. No code is touched until the rulings below.

## 1. Facts

**The failure, from the artefacts [ours].** Every attempted frame's
`frames/orca.<level>.<frame>.out` ends, just after the basis-set groups:

```
--------------------------------------------------------------------------
While trying to determine what resources are available, the SLURM
resource allocator expects to find the following environment variables:

    SLURM_NODELIST
    SLURM_TASKS_PER_NODE

However, it was unable to find the following environment variable:

    SLURM_TASKS_PER_NODE
--------------------------------------------------------------------------
[cnode4898:3900743] [[37107,0],0] ORTE_ERROR_LOG: Not found in file /openmpi-4.1.8/orte/mca/ras/base/ras_base_allocate.c at line 193
--------------------------------------------------------------------------
An internal error has occurred in ORTE: FORCE-TERMINATE AT (null):1

ORCA finished by error termination in Startup
Calling Command: mpirun -np 4  .../orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg/orca_startup_mpi job.int.tmp job
[file orca_tools/qcmsg.cpp, line 394]:
  .... aborting the run
```

Seconds per frame, no chemistry. The mechanism, from the OpenMPI v4.1.8 source: `ras/slurm`
is selected by `SLURM_JOBID` **alone** (`ras_slurm_component.c`, priority 50) and then
requires `SLURM_NODELIST` **and** `SLURM_TASKS_PER_NODE` (`ras_slurm_module.c`;
`help-ras-slurm.txt` is the exact text above); `ORTE_ERR_NOT_FOUND` is not a tolerated
return, so `ras_base_allocate.c` force-terminates. `plm/slurm` (priority 75, daemons via
`srun`) and `ess/slurm` are armed by the same variable. Deterministic reproduction on the
workstation (OpenMPI 4.1.6):

```
RED    env SLURM_JOBID=1 SLURM_NODELIST=fake1 mpirun -np 2 hostname     # the ORTE block above
GREEN  env -u SLURM_JOBID SLURM_NODELIST=fake1 mpirun -np 2 hostname    # runs
```

**The env the job hands the workers.** `hl_labels.slurm:63`:
`grep -E '^(PMI|SLURM_(CPU|TASK|NTASKS|NPROCS|STEP))'` -- **deletes** `SLURM_TASKS_PER_NODE`,
`SLURM_CPUS_PER_TASK`, `SLURM_STEP_*`, `SLURM_NTASKS`, `PMI_*`; **keeps** `SLURM_JOBID`,
`SLURM_NODELIST`, `SLURM_JOB_CPUS_PER_NODE`, `SLURM_NNODES`, `SLURM_SUBMIT_DIR`,
`SLURM_JOB_PARTITION`. `SLURM_JOBID` + `SLURM_NODELIST` are exactly the two the launcher
must not see, and the one it then demands is exactly the one the loop removed.

**The seams in the code** (file:line) [ours]:

| site | what runs | `env=` today |
|---|---|---|
| `openqha/qm_interfaces/orca.py:142` (`_run_job`) | every `optimise_and_hessian` / level job (msRRHO) | `subprocess_env()` |
| `openqha/data/frame_labels.py:422` (`_run_orca`) | the labels worker's ORCA | `orca.subprocess_env()` |
| `openqha/qm_interfaces/orca.py:249` (`single_point`) | composite / term-pool points | **none at all** (so on Tianhe it would not even get `S0_ORCA_PATH`/`S0_ORCA_LIB`) |
| `scripts/production/s0_branch2_opt_freq.py:269` | branch B opt+freq | none |
| `scripts/calibration/s0_package2_highlevel_freq.py:103` | package2 high-level | none |

`subprocess_env()` (`orca.py:184`) builds `dict(os.environ)`, prepends `S0_ORCA_PATH` to
PATH and `S0_ORCA_LIB` to LD_LIBRARY_PATH -- the one function whose whole job is "the
environment an ORCA subprocess runs in".

**The convention already in the repo.** The parsl worker init
(`hpc/resource_configs/tianhe_cpu.py:195`) and the hkuhpc bundle worker
(`openqha/qm_interfaces/orca_jobs.py`) unset **everything** `^(PMI|SLURM)_`;
`hpc/slurm/README.md` rule 2 states the reason ("They make a child process misread the task
layout and try to relaunch itself through the scheduler ... learnt on ORCA under xargs");
`orca_jobs.py` also sets `ORCA_SKIP_CPU_BIND=1` and `OMPI_MCA_rmaps_base_oversubscribe=1`.
Nine of the ten `hpc/slurm/*.slurm` scripts carry the narrow filter (`install_env_tianhe.slurm`
is the exception); only `hl_labels` and `hl_pipeline_debug` run a step-03/ORCA payload --
the others run CREST/MACE/torch, which is why only the labels step blew up.

**What the research confirms about the strategy** (details in the research note):
ORCA's driver must call its own `mpirun` ("Do not start the ORCA driver with mpirun!");
binding is OpenMPI's and ORCA documents its off-switches (`--bind-to none`,
`OMPI_MCA_hwloc_base_binding_policy none`); no ORCA-documented `ORCA_*` variables exist
(so `ORCA_SKIP_CPU_BIND` is a hkuhpc convention, not an ORCA one); 4 ranks/job × 16 jobs is
consistent with ORCA's efficiency ceilings; 16 raw `mpiruns` is Slurm MPI-guide mode 2/3
(Slurm neither binds nor accounts them -- we own placement); `--exclusive` + `--mem=0`
does give the node's memory, and `%maxcore 6000 × 64 = 384 GB = 0.75 × 512 GB` matches
ORCA's own `MaxCore × nprocs <= 0.75 × memory` rule.

**On disk.** Every frame the array attempted is in the **failed** state (`.out` without the
terminal line; ticket 24: one attempt, `--retry` is the human's lever). Two operational
consequences to design against [ours, from `03_labels.py`]:

- `pending()` skips failed frames and the assemble exit code counts only *unlabelled*
  ones -- so a resubmitted round would print "nothing pending" and task 0 could **exit 0
  while the whole attempted set is failed**. Without a retry lever the failure mass is
  invisible to the round signal.
- Count to be read before anything: `python scripts/tooling/s0_hl_progress.py --tag draw300`
  (its `failed` column).

## 2. The design on the table (proposal only)

The invariants the fix buys, in one place:

1. **The ORCA child is Slurm-blind.** Every ORCA subprocess gets an environment with no
   `SLURM*`/`PMI*` variable at all, built in one place (`subprocess_env()`), so no ORCA
   launch route (xargs worker, parsl worker, msRRHO, future) can depend on the caller's
   shell. With `SLURM_JOBID` gone no Slurm ORTE component is even eligible; ras falls back
   to the local node (source-quoted in the research note §2.4).
2. **Placement has exactly one owner: the worker's `taskset` range.** OMPI's own binding
   is switched off with the ORCA-documented `OMPI_MCA_hwloc_base_binding_policy=none`, so
   nothing binds against the node topology while a process is pinned to, say, `48-51`.
3. **One mpirun per frame, `%pal nprocs 4`, 16 frames per 64-core node** (unchanged);
   `--exclusive --mem=0`, `%maxcore 6000` (the 0.75 rule) unchanged.
4. **The job script keeps its Slurm variables** for its own bookkeeping (runs root,
   submit dir, task list name, frame-lock owner); the narrow loop is not load-bearing for
   ORCA anymore once (1) holds.
5. **Recovery is an explicit act**: the failed mass is re-dispatched by the deliberate
   retry path (Q5), after one hand-verified frame (below), not by deleting evidence or by
   silently widening the round.

Change list (to be ticketed per Q7): `subprocess_env()` strip + MCA var; `single_point`
routed through it; the two `scripts/` launchers audited; a unit test on the child env;
the retry path; ADR + the two README/comment updates.

Verification plan (after the rulings, before any array resubmission): (i) the local red/green
loop above; (ii) one frame on Tianhe --
`python -m openqha.data.frame_labels $S0_RUNS_ROOT/draw300/dsgdb9nsd_006885 displaced 1 3 --retry`
 -- expect `labelled` and `****ORCA TERMINATED NORMALLY****` in the fresh `.out`; (iii) only
then the retry round. `OMPI_MCA_rmaps_base_oversubscribe=1` stays a documented fallback,
set only if a "not enough slots" line ever appears.

Explicitly **not** in this design: the CREST launches (not MPI-launched; the narrow loop is
for their CPU-count vars and they have run fine), the parsl worker init and the hkuhpc
bundle (already full-unset), the `srun`-steps rewrite (ORCA's driver must own the mpirun),
and the `%maxcore`/layout arithmetic (verified above).

## 3. Questions (round 13) -- answers recorded here when they arrive

❓ **Q1** -- the seam and the strip rule. Where does invariant (1) live, and what exactly is
removed? Options: (a) `subprocess_env()` removes every variable whose **name starts with
`SLURM` or `PMI`** (covers `SLURM_JOBID`, `SLURM_NODELIST`, `SLURM_TASKS_PER_NODE`,
`SLURMD_NODENAME`, `PMI_FD`, `PMIX_*`), then adds the Q3 knob; keeps `S0_ORCA_PATH` /
`S0_ORCA_LIB` and everything else; (b) the same strip in `hl_label_worker.sh` -- fixes only
the xargs route; (c) the full unset in the `.slurm` job scripts -- but `_claim` reads
`SLURM_JOB_ID` for the frame lock, so the job-level version needs captures (and changes
lock owners to `pid<n>`); (d) MCA exclusions -- must cover `ras`+`plm`+`ess`, three knobs
to keep consistent. Note (a) is a *prefix rule*, not another pattern list: the narrow
regex is exactly how this bug happened.
➡️ (a). One seam, every ORCA route, nothing for the caller to remember.

---

❓ **Q2** -- make the invariant total. `single_point` (`orca.py:249`) passes no `env=` at
all (a second bug in the opposite direction -- it misses the Tianhe paths), and the two
`scripts/` launchers do the same. In this change: (a) route all of them through
`subprocess_env()`; (b) only `single_point` now, a separate audit ticket for `scripts/`;
(c) leave them.
➡️ (a) -- one line each, same class; the test then asserts "every ORCA spawn goes through
`subprocess_env()`".

---

❓ **Q3** -- binding. With `taskset -c s-e` the worker owns placement; OMPI's default
binding binds against the node topology, which for the slots whose mask is not `0-3`
(`4-7` … `60-63`) is untested and can fail or misplace ranks. Options: (a) set
`OMPI_MCA_hwloc_base_binding_policy=none` in the ORCA child env for **every** launch
(ORCA-documented off-switch); (b) only the labels route; (c) rely on `taskset` + defaults
until measured; (d) also set `OMPI_MCA_rmaps_base_oversubscribe=1` now.
➡️ (a); (d) stays a fallback. While touching this: the hkuhpc bundle's
`ORCA_SKIP_CPU_BIND=1` is not an ORCA-documented variable -- replace it with the documented
knob the next time that file is touched (separate cleanup, not this ticket).

---

❓ **Q4** -- the nine `.slurm` scripts that carry the narrow loop (only `hl_labels` and
`hl_pipeline_debug` run step 03). (a) Keep their loops as they are and add one comment at
`hl_labels.slurm:63` pointing to the ORCA-child strip (so the next reader does not "repair"
the payload issue at the job level); or (b) align all nine to the full `^(PMI|SLURM)_`
unset -- which needs each script's after-loop Slurm reads captured first (`hl_labels`:
`SLURM_SUBMIT_DIR` :66, `SLURM_JOB_PARTITION` :67, `SLURM_JOB_ID` :81) and costs the frame
lock its Slurm owner (`_claim` writes `pid<n>`; `_job_alive` then answers nothing) unless
only the modern `SLURM_JOB_ID` name is re-exported.
➡️ (a): once invariant (1) holds, the job-level loop is deliberately about scheduling vars
for non-ORCA payloads, and (b) would change lock owners and touch scripts that are fine.

---

❓ **Q5** -- recovery of the failed mass. Options: (a) teach the round to retry --
`03_labels.py --retry-failed` (its list includes failed frames; the worker passes
`--retry`, which the frame CLI already has; list gains a retry column / the worker a
flag), invoked deliberately per round; (b) a standalone tooling script that lists and
re-runs failed frames; (c) delete the file groups whose `.out` carries the ORTE signature
so they read as "never run" (no code, but loses evidence and bends the ticket-24 policy).
Common prerequisites either way: count the failures first (`s0_hl_progress`), verify one
frame by hand, then dispatch at array scale.
➡️ (a): one dispatch path (array + worker), the human's decision becomes an explicit flag,
evidence stays on disk, and it is reusable for genuine failures.

#### Q5 refinement (2026-09-25, the user's two constraints): preservation + the once-only rule

(1) The previously failed ORCA results must be saved -- a retry must not destroy them.
(2) A retry runs at most once -- a retry that fails must never be re-selected by any later
round (no retry loop across rounds).

❓ **Q5a** -- where a failed output goes before a retry overwrites it. Today
`label_one(retry=True)` runs ORCA and `copy2`s the KEEP files over the existing ones, so the
failed `.out` is destroyed by the retry. Options: (a) rename `<stem>.out` to
`<stem>.failed1.out`, then `.failed2.out`, ... -- append-only, never overwritten (+24 KB per
attempt; the number is the failed attempt's index); (b) one `<stem>.failed.out`, replaced by
each new archived failure; (c) no per-frame archiving -- freeze one bulk incident snapshot
and leave the frame dirs as they are; (d) refuse to retry while a failed `.out` exists
(archiving by hand) -- too manual for a mass retry.
➡️ (a).

❓ **Q5b** -- what makes a frame ineligible for a second automated retry. Options: (a) the
existence of any `<stem>.failed*.out` archive -- the archive doubles as the retry marker
(the retry archives before running; a retried-and-failed frame then carries an archive, and
`--retry-failed` skips it); (b) a `<stem>.attempts` log (round 11's proposed format: one
line per failed run with job id, rc, seconds, last line) with the predicate "the file
exists"; (c) both; (d) operator discipline only.
➡️ (a) -- one predicate, no new file type; (b) can layer on later if per-attempt metadata is
wanted. Boundary accepted in the same breath: a retry CUT by walltime/SIGKILL leaves no
`.out`; the frame then reads "never run" and the ordinary policy reruns it whole -- the cap
counts failures, not cuts (cuts were always rerun).

❓ **Q5c** -- the manual lever vs the cap. `--retry-failed` is capped at one retry per frame
(by Q5b's predicate); the per-frame manual `python -m openqha.data.frame_labels ... --retry`
is the human's deliberate act. Options: (a) manual stays allowed; archives append
(`.failed2.out` ...) and the output says which attempt it is; (b) manual refuses when an
archive exists unless `--force`.
➡️ (a): the cap protects the automation from looping; a human working a special case is not
locked out (ticket 24's "the reader judges" stays true).

❓ **Q5d** -- does the save need a separate incident snapshot as well? Options: (a) in-place
preservation + archive-before-overwrite (Q5a) IS the save -- nothing extra to run; (b) also
freeze the whole failed set before the retry round (e.g. a tar under `_records/`) as an
incident record.
➡️ (a); (b) only if the incident must be frozen outside the molecule trees (the signature is
documented in the research note; one molecule's full file set is in the user's `dbg/`).

**Sequencing consequence (hard rule).** With the once-only cap, running `--retry-failed`
before the fix is deployed would burn every frame's single shot. Order: deploy the Q1 fix ->
count (`s0_hl_progress`) -> verify ONE frame by hand (the manual `--retry`; it counts as that
frame's first retry -- acceptable) -> only then the `--retry-failed` round.

---

❓ **Q6** -- docs and the domain model. (a) ADR **0007** "ORCA children are Slurm-blind;
the worker owns placement" -- next free number; it passes the three tests (surprising: "why
is the scheduler unset inside a scheduler job?"; a real trade-off: no RM binding/accounting
for the launchers, alternatives (Slurm-shaped allocation, srun steps, MCA exclusions) were
on the table; hard to reverse: the policy is encoded in the child-env seam and the docs);
(b) update `hpc/slurm/README.md` rule 2, the `hpc/env/orca.sh` header and the
`hpc/README.md` labels row to state the **two layers** (job-level narrow unset for
scheduling vars; ORCA-child full strip in `subprocess_env()`); (c) `CONTEXT.md` -- no new
glossary term (this is implementation policy, not domain vocabulary).
➡️ (a)+(b); not (c).

---

❓ **Q7** -- tickets and numbering. (a) Two: **40 -- the ORCA child environment** (strip +
MCA knob + `single_point`/`scripts` routing + unit test + ADR/README) and **41 -- the
retry round** (`--retry-failed`); (b) one ticket for everything; (c) other numbering (this
tracker's highest used is 39).
➡️ (a), subject to the numbering.

---

## 4. Rulings

### Q4 ruled: leave the nine narrow loops (option (a)) -- 2026-09-25
The job scripts' filter is unchanged: the payload-side strip lives in `subprocess_env()`
(Q1), the only layer that can make the ORCA child Slurm-blind while the worker keeps its
Slurm view (frame-lock owner, submit dir, reporting). Accepted: two unset conventions
coexist (full in the parsl init and the hkuhpc bundle; narrow in the sbatch scripts), to be
documented by the pointing comment at `hl_labels.slurm:63` (+ the twin at
`hl_pipeline_debug.slurm:50`), ADR 0007 and the README rule-2 wording (Q6). §1/Q4's count
was corrected to nine scripts (`install_env_tianhe.slurm` carries no loop).

### Q1 ruled: the strip lives in `subprocess_env()`, prefix rule (a) -- 2026-09-25
Every ORCA child is Slurm-blind: delete every variable whose name starts `SLURM` or `PMI`;
keep `S0_ORCA_PATH`/`S0_ORCA_LIB` and everything else.

### Q2 ruled: make it total (a) -- 2026-09-25
`single_point()` and the two direct `scripts/` launchers are routed through
`subprocess_env()` in the same change.

### Q3 ruled: `OMPI_MCA_hwloc_base_binding_policy=none` for every ORCA child (a) -- 2026-09-25
Placement's only owner is the worker's `taskset`; `OMPI_MCA_rmaps_base_oversubscribe=1` stays
a documented fallback. The hkuhpc bundle's undocumented `ORCA_SKIP_CPU_BIND` is replaced with
the documented knob the next time that file is touched (separate cleanup).

### Q5 ruled: `--retry-failed`, one retry per frame, archive-before-overwrite -- 2026-09-25
Base shape (a); the refinement questions ruled as follows.
- **Q5a = (b)**: one archive slot, `<stem>.failed.out`, replaced each time it is written.
- **Q5b**: each retry run archives the previous failure before ORCA starts; successes stay
  skipped; only failed frames are ever retried; after the single retry the frame is final
  whatever the outcome. Predicate: a frame is retryable iff its `.out` lacks the terminal
  line AND no `<stem>.failed.out` exists.
- **Q5c**: `--retry-failed` is the only retry lever; a `--force`-style switch exists only for
  the deliberate "re-run all ORCA labels" operation and is off by default.
- **Q5d**: no separate incident snapshot -- in-place preservation plus the archive is the
  save.

### Q6 ruled: ADR 0007 + README/`orca.sh` wording (a)+(b); no CONTEXT.md change -- 2026-09-25

### Q7 ruled: two tickets (a) -- 2026-09-25
The ORCA child environment and the retry round, cut in the new `.scratch/orca-slurm/`
tracker; this round and the research note move there with them.

## Amendment 2026-09-26 (the retry's trigger reverses; tickets 03-06 of the set)

Ruled 2026-09-26, after this round: the one-shot retry **rides every round by default**.
A plain round's task list carries the failed frames without an archive (`retry` in the
5th column -- Q5's mechanism unchanged), and `RETRY_ONLY=1` / `--retry-only` is the
FAILURES-ONLY SWEEP; `RETRY_FAILED=1` / `--retry-failed` are deleted from code, script,
comments and docs. Superseded here: Q5's `--retry-failed` as the deliberate recovery
lever, Q5c's "`--retry-failed` is the only retry lever", and the "Sequencing
consequence" above's retry round. Unchanged: the once-only cap, the single archive slot
(Q5a = (b)), the Q5b predicate, Q5d's save and `--force`. The sequencing consequence
reads: the fix deployed and verified -> count the failures -> verify ONE frame by hand
-> **any round** (any round burns the retries it carries). No ADR: the change is
reversible and page-recorded (tickets 02/04, `spec.md`'s amendment, this note).
