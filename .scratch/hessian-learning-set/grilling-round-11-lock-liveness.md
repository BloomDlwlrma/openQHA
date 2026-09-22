# Hessian learning -- grilling, round 11: the `.running` lock after a walltime kill, (c) ask Slurm + (a) SIGTERM (2026-09-21)

Scope: the production run of step 03 (labels) on TianheXY-CN, either route -- `hl_labels.slurm`
arrays resubmitted per round, or the ALF mode (a parsl driver in tmux on the login node,
`tianhe_cpu` role `labels`, blocks of one node, `--walltime 3-00:00:00`). Both leave the same
hole when a node hits its time limit; the user asked for (c)+(a) of the three fixes discussed
on 2026-09-21 and for this round before the ticket.

## 1. Facts (read from the code and the Slurm manual; the two site values to measure are marked)

**The lock.** `frame_labels._claim` (`:238-246`) writes `<frames>/<stem>.running` with one line,
`"<SLURM_JOB_ID or pid<pid>> <ISO time>"`, and `running_elsewhere` (`:229-235`) calls a lock
fresh when its **mtime is younger than `LOCK_MAX_AGE_S = 2 h`** -- the mtime is written once,
so "fresh" means "claimed less than 2 h ago", nothing about the process. Three readers:
`label_one` through `_claim` (skips with `status="running"`, `:384-388`), `03_labels.pending`
(`:131`, the task list), `s0_hl_progress` (`:75`, the "running" column). `_release` runs in
`finally` (`:412-413`).

**What kills a job at its time limit.** Slurm sends SIGTERM to every process of the job, waits
`KillWait` (default 30 s; **site value unmeasured**: `scontrol show config | grep KillWait`),
then SIGKILL. Python's default for SIGTERM is the kernel default -- terminate at once, no
exception, no `finally` -- so the lock stays with its claim-time mtime. (Ctrl+C is different:
Python turns SIGINT into `KeyboardInterrupt`, which is why a manual interrupt never leaks a
lock.) `subprocess.call` kills its child on any exception leaving `wait()` (CPython's bare
`except: p.kill(); raise`), so an exception raised by a handler also ends ORCA.

**The hole, in the parsl route.** A lost block makes HTEX fail its tasks (`ManagerLost`);
`retries=1` (`tianhe_cpu.py:288`) re-runs each within about a minute on another block;
`_claim` finds the lock < 2 h old and returns `status="running"`, which `label_frame_task`
returns as a row, not an exception -- no further retry, the frame is dropped for this driver
run. 12 blocks x one time limit each per 3 days x 16 frames in flight = ~200 frames per
round, ~500 per campaign, recoverable only by a second driver run started >= 2 h later.
In the sbatch route the next round (`afterany`) lists the same fresh locks and skips them
until the round after.

**The hole's other face.** A legitimate frame running longer than 2 h (the 19-atom Hessian is
estimated at 40-80 min; the tail is unmeasured) has its lock called stale by a second Batch,
which then runs the same frame; both copy into the same `<stem>.{out,hess,engrad}`.

**No per-frame ORCA timeout in step 03.** `label_one(timeout_s=None)`; the CLI's `--timeout`
defaults to None and `hl_label_worker.sh` does not pass it; `label_frame_task` neither. A
hung ORCA holds its 4 cores until the node's time limit. (Branch A has `TIMEOUT_S`, 3600 in
the script, 14400 in `s0_A_pipeline.py`.) And a frame whose ORCA did not terminate normally
has no terminal line, so `finished` is False and it is pending again in every later round --
there is no attempt count anywhere.

**Nothing resumes; every unit is rerun whole (user's premise checked 2026-09-22).** ORCA
restarts only the SCF guess from a same-named `.gbw` (`AutoStart`); the analytic Hessian has no
restart. Our jobs run in the node-local `S0_SCRATCH`, `KEEP` has no `.gbw`, the run directory
is removed -- a dead node leaves nothing; keeping the `.gbw` would be 5-20 MB x 100k frames.
CREST has no mid-run checkpoint. So the unit of rerun is the frame (<= 4 h under Q2), the
molecule's branch A (<= 25 min), the Frame set (~1 min). The cost of the policy: a node
hitting its limit loses at most 16 units in flight, ~half a unit each, ~0.03 % of a 3-day
round on 12 nodes. The lock therefore has two duties only -- no two jobs on one unit, no
dead lock blocking a unit -- and none about resuming.

**`squeue` from a compute node.** `squeue -h -j <id> -o %T` prints the job's state (`RUNNING`,
`COMPLETING`, `TIMEOUT`, ..., or nothing once the record is purged) and exits non-zero for an
unknown id. Whether the CN compute nodes can reach slurmctld is **unmeasured** (stock Slurm
sites can; one line in a gate job tells). Job ids are monotonic and wrap only at MaxJobId, so
a stale lock's id is not reused by a live job within a campaign.

**Tests today.** `t_frame_labels` (`:219-230`) writes a lock `"999 now"`, checks the skip, sets
the mtime to 1970, checks the take-over. Under (c) that test must control `squeue` (a fake
on PATH) or the workstation path (no `squeue`) is what runs.

## 2. The two fixes as proposed

**(c) ask Slurm.** `running_elsewhere`: no lock -> False; lock -> read the owner; owner a Slurm
id and `squeue` answers -> Slurm's word (`RUNNING PENDING SUSPENDED CONFIGURING` = alive;
`COMPLETING`, any terminal state, no record, non-zero exit = dead, the lock is overwritten);
`squeue` absent / failing / timed out (20 s) or a `pid...` owner -> today's age rule. One
`squeue` per lock met, cached per job id for 60 s.

**(a) SIGTERM handler.** In `label_one`, around the ORCA run only: `signal.signal(SIGTERM,
raise SystemExit(143))`, previous handler restored in `finally`. The exception leaves
`subprocess.call` (which kills ORCA), reaches `finally: _release`, the lock is gone before the
retry arrives. Covers time limit and `scancel`; not SIGKILL, OOM of the python, node death --
those are (c)'s.

## 3. Questions (round 11) -- answers recorded here when they arrive

❓ **Q0** -- record the rerun policy as a decision: no unit checkpoints or resumes; an unfinished
unit is rerun whole; a unit is therefore bounded in time (Q2) and in attempts (Q3); `.gbw`
never leaves the node. A sentence in CONTEXT's Frame set entry and an S0-G decision.
➡️ Yes. It closes the "copy the .gbw back" idea before anyone has it.

---

❓ **Q1** -- the liveness rule: Slurm's word alone, or Slurm's word AND the age?
Slurm alone: a live job's lock is never stale (no double labelling of a long Hessian), but a
python worker that died alone inside a living job leaves its frame locked until that job
ends (rare: OOM takes the ORCA ranks, not the python, and then `finally` releases). Slurm
AND age (2 h): covers that case, brings back the long-Hessian double labelling.
➡️ Slurm's word alone, and close the "job alive, frame stuck" case with a per-frame ORCA
timeout instead (Q2) -- a timeout is a fact about the frame; an age is a guess about the
process.

---

❓ **Q2** -- a per-frame ORCA timeout for step 03: none (today), or `TIMEOUT_S` like branch A?
Without one a hung or pathological ORCA holds 4 cores for up to 3 days. With one,
`subprocess.call` kills ORCA at the limit, `finally` releases the lock, the worker prints
`FAILED TimeoutExpired`.
➡️ `TIMEOUT_S=14400` (4 h; 3x the 19-atom Hessian estimate's upper end), passed by
`hl_labels.slurm` -> `hl_label_worker.sh --timeout` and by `label_frame_task`; the same
variable name as branch A. Measured tails from the campaign's first round can move it.

---

❓ **Q3** -- the rerun cap: under Q0 a frame whose ORCA did not terminate normally (timeout,
crash) is rerun from scratch in every later round, for the same 4 h each time, and a third
run differs from the first two only by hardware noise. Options: (a) leave it
(the progress table shows it as unlabelled forever; a human notices); (b) count attempts in
`<stem>.attempts` (one line per failed run: job id, rc, seconds, last line) and let `pending`
give up after `MAX_ATTEMPTS = 2`, the progress table gaining a "given up" column.
➡️ (b). Two attempts of 4 h on 4 cores is the price of one bad frame; a third is waste on
12 nodes. The Dataset simply lacks that frame (the split is per frame).

---

❓ **Q4** -- which `squeue` states count as alive?
➡️ `RUNNING PENDING SUSPENDED CONFIGURING` alive; `COMPLETING` dead (the processes are
already killed, the node is in epilog); every terminal state, an empty answer and a non-zero
exit dead. `squeue` failing to run at all (not found, timeout, controller down) is "no
answer" -> age rule, never "dead".

---

❓ **Q5** -- where the SIGTERM handler lives: (a) inside `label_one`, installed after the claim
and restored in `finally` (the only place with a lock to release); (b) process-wide at the
start of `python -m openqha.data.frame_labels` and of `label_frame_task`.
➡️ (a). A handler that raises inside code that does not expect it is a bug waiting to happen;
the lock is held in exactly one place. Note for the parsl route: `label_frame_task` catches
`Exception`, and `SystemExit` is not one, so it propagates and the task fails -> retry, which
is the wanted behaviour.

---

❓ **Q6** -- parsl `retries`: keep 1, or 2?
A frame cut by a time limit is retried once on another block; cut twice (two time limits
within one frame's life, ~impossible unless the frame runs > 3 days) it is a `FAILED` row
and pending for the next driver run.
➡️ Keep 1. Under Q3 an attempt count on disk is the memory, not parsl's retry counter.

---

❓ **Q7** -- verifying `squeue` from a compute node: add one report line to `hl_labels.slurm`
(and `hl_pipeline_debug.slurm`) -- `squeue from the node: $(squeue -h -j $SLURM_JOB_ID -o %T
|| echo unreachable)` -- so the next gate's log states which branch of (c) this site takes?
➡️ Yes; one line, and `t_hl_campaign` learns it. If the answer is `unreachable`, (c)
degrades to today's rule on tianhe and only (a) works there; the ticket then adds a second
route (the lock records the node's hostname and `scontrol show job` from the login-side
lister) -- not built until measured.

---

❓ **Q8** -- ticket scope: one ticket for (c)+(a)+Q2+Q3 (the lock's whole life), or (c)+(a) now
and Q2/Q3 as a second ticket?
➡️ One ticket, **24 -- the lock's life: liveness, SIGTERM, timeout, attempts**. They touch the
same forty lines of `frame_labels.py` and the same test; splitting them means two rounds of
the same `t_frame_labels` fixtures. The production-sequence rewrite of the campaign page
(sbatch A -> sbatch 02 -> tmux driver 03, the tmux gate) is a separate ticket 25, after the
gate has run.

## 4. Rulings

### Q1 -- recommendation revised (2026-09-22, after the "how is this solved elsewhere" question)
The objection to "Slurm AND age" was that age meant time since the claim. With a heartbeat
(the holder touches its lock every `HEARTBEAT_S = 60` from a daemon thread) age means time
since the holder last proved itself alive, and the objection is gone. Revised ➡️: **Slurm's
word AND heartbeat age (`LOCK_MAX_AGE_S = 30 min`)** -- (c)+(b)+(a): no lock -> free; job
dead (squeue) -> free; job alive but no heartbeat for 30 min (the python died alone) -> free;
job alive and heartbeat fresh -> held; squeue unanswerable -> heartbeat alone. This is the
FireWorks shape (ping + detect_lostruns) and the Nextflow shape (squeue + .exitcode) put
together. Cost: ~10 lines (`_Heartbeat` thread + Event), one more test. Awaiting the ruling.

### Q2 ruled: TIMEOUT_S = 28800 (8 h)
The unit's time bound for step 03 is eight hours, passed as `TIMEOUT_S` by `hl_labels.slurm`
-> `hl_label_worker.sh --timeout` and by `label_frame_task`; the same variable name as branch
A. One pathological frame costs at most 8 h x 4 cores = 32 core-h.

### Q3 ruled: no automatic rerun -- one attempt per frame; the reader judges
A frame whose ORCA ran and did not terminate normally (crash, rc != 0, the 8 h timeout) is
**not** rerun by any later round or driver run. The evidence stays on disk and in the logs
and a human reads it: the worker's one line in the Slurm `.out` (`<molecule> <frame> FAILED
...`), anything on the Slurm `.err`, and the returned ORCA files (`<stem>.out` without the
terminal line; `.hess`/`.engrad` absent or partial). Consequences for the code:

- three states per frame instead of two: **finished** (`.out` with `****ORCA TERMINATED
  NORMALLY****`, plus `.hess` where wanted), **failed** (`.out` present, no terminal line),
  **never run** (no `.out`). `pending` lists only the third; `label_one` returns
  `status="failed"` on the second without claiming or running; a `--retry` flag on the CLI
  (and `retry=True` on `label_one`) is the human's way to rerun one frame deliberately.
- the 8 h timeout must leave evidence: `TimeoutExpired` is caught, the partial `job.out` is
  copied back as `<stem>.out` with one appended line `openQHA: ORCA killed after TIMEOUT_S=28800
  s` -- otherwise a timed-out frame looks never run and would be rerun for another 8 h.
- a frame CUT before anything came back (walltime SIGTERM raising out of `subprocess.call`,
  SIGKILL, node death: the node-local run directory is lost, nothing is copied) has no `.out`
  and is therefore "never run" -> rerun whole. Cut is not failed; Q0's policy is unchanged.
- `s0_hl_progress` and `assemble` gain a **failed** count (frames with a `.out` and no terminal
  line) beside labelled / unlabelled / running, so the campaign's failures are one number in
  the progress table and the Batch, not a grep.
- no `<stem>.attempts` file, no `MAX_ATTEMPTS`: exactly one attempt unless a human asks.

### Q0 ruled: yes, without the `.gbw` clause
The rerun-whole policy is recorded (CONTEXT, Frame set entry; decision S0-G-96): no unit
checkpoints or resumes, an unfinished unit is rerun whole, a unit is bounded in time (Q2) and
attempted once (Q3). Whether a `.gbw` ever comes back to the shared disk is NOT ruled out --
left open, not part of the policy.

### Q1 ruled: Slurm's word AND heartbeat age (30 min) -- (c)+(b)+(a)
As revised in §4 above.

### Q4 ruled with the handling spelled out (the user asked what the states are)
Slurm's job states (`squeue -o %T`, `%t` for the code): PENDING (PD), RUNNING (R), SUSPENDED
(S), CONFIGURING (CF), RESIZING (RS), SIGNALING (SI), STAGE_OUT (SO), STOPPED (ST), REQUEUED (RQ),
REQUEUE_HOLD (RH), REQUEUE_FED (RF), RESV_DEL_HOLD (RD) -- the job exists and may run again;
COMPLETING (CG) -- **transient for every ending, normal or not**: the processes have been sent
their signals and the node runs its epilog; then one terminal state: COMPLETED (CD), FAILED (F),
TIMEOUT (TO), CANCELLED (CA), NODE_FAIL (NF), OUT_OF_MEMORY (OOM), PREEMPTED (PR), BOOT_FAIL
(BF), DEADLINE (DL), REVOKED (RV), SPECIAL_EXIT (SE). A finished job stays visible for
`MinJobAge` (default 300 s), then `squeue -j` prints nothing and exits 1 ("Invalid job id").
So: a normal end and an abnormal one both pass through COMPLETING and differ only in the
terminal state after it; for the lock the distinction is irrelevant -- in COMPLETING the ORCA
and the python are already gone.
Handling: a **dead list**, not an alive list -- `state.split()[0]` in {COMPLETING, COMPLETED,
FAILED, TIMEOUT, CANCELLED, NODE_FAIL, OUT_OF_MEMORY, PREEMPTED, BOOT_FAIL, DEADLINE, REVOKED,
SPECIAL_EXIT}, or empty output, or exit != 0 -> dead; any other state (including one this list
has never seen) -> alive, and the heartbeat decides within 30 min. `squeue` not on PATH,
timing out (20 s) or failing to reach the controller -> no answer -> **heartbeat age alone**
(Q1's rule, replacing the old 2 h claim-age rule). `%T` may print "CANCELLED by <uid>", hence
`split()[0]`. A requeued job keeps its id, so its own stale lock reads "alive" until the
heartbeat rule frees it -- 30 min, acceptable.

### Q5 ruled: (a) -- inside `label_one`, after the claim, restored in `finally`
### Q6 ruled: parsl `retries` stays 1
### Q7 ruled: no verification line; (c) is built with its fallback and needs no gate of its own
### Q8 ruled: one ticket -- 24

Round 11 closed 2026-09-22.
