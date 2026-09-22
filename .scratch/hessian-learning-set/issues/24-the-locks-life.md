# 24: The lock's life -- liveness by Slurm AND heartbeat, SIGTERM release, `TIMEOUT_S` = 8 h, one attempt per frame (`data/frame_labels.py`, `03_labels.py`, `hl_labels.slurm`, `hl_label_worker.sh`, `s0_hl_progress.py`)

**Why (round 11, 2026-09-21/22):** a node hitting its time limit kills the python worker with
SIGTERM, which Python does not turn into an exception, so `finally: _release` never runs and
the frame's `.running` lock stays with its claim-time mtime; `running_elsewhere` calls it fresh
for 2 h. In the parsl route `retries=1` reruns the frame within a minute on another block,
`_claim` refuses, `label_one` returns `status="running"` and the frame is dropped for the
whole driver run (~200 frames per 3-day round). The same rule calls a legitimate >2 h Hessian
stale and lets a second Batch double-run it. Step 03 also had no per-frame ORCA timeout and
no memory of failed attempts. Rulings: Q0 rerun whole (S0-G-96), Q1 Slurm AND heartbeat,
Q2 8 h, Q3 one attempt, Q4 dead list, Q5 handler in `label_one`, Q6 retries 1, Q7 no gate line,
Q8 one ticket.

**What to build (`openqha/data/frame_labels.py`):**

- `HEARTBEAT_S = 60`, `LOCK_MAX_AGE_S = 30 * 60`, `DEAD_STATES` (COMPLETING and the eleven
  terminal states), `_owner(lock)`, `_job_alive(job_id)` -> True / False / None (squeue's word,
  cached 60 s per id; None when squeue is absent, times out at 20 s or the owner is a `pid`),
  `running_elsewhere` = lock exists AND job not dead AND mtime younger than 30 min (no answer
  from Slurm -> the heartbeat alone). `_claim` unchanged.
- `_Heartbeat(lock)`: a daemon thread touching the lock every `HEARTBEAT_S` (`os.utime`),
  stopped by an `Event`, used as a context manager around the ORCA run.
- `label_one`: three states -- finished (terminal line, `.hess` where wanted) -> `reused`;
  **failed** (`<stem>.out` present, no terminal line) -> `status="failed"`, no claim, no run,
  unless `retry=True`; never run (no `.out`) -> claim, SIGTERM handler installed
  (`SystemExit(128 + signum)`, main thread only, previous handler restored in `finally`),
  heartbeat running, ORCA with `timeout_s`; `TimeoutExpired` caught: the partial `job.out` is
  copied back as `<stem>.out` with the appended line `openQHA: ORCA killed after TIMEOUT_S=<s> s`
  and the frame raises as today (the worker prints `FAILED`). A cut before anything came back
  copies nothing (no `.out` -> never run -> rerun whole, Q0).
- `failed(folder, stem)` beside `finished`; `main()` gains `--retry` and `--timeout` default
  from `TIMEOUT_S` in the environment.
- `03_labels.pending`: never-run frames only, counts gain `n_failed`; `label_frame_task` passes
  `timeout_s` from `TIMEOUT_S`; the Batch report prints labelled / failed / running.
- `assemble`: the Record gains `N_FAILED`; failed frames are named in `labels.<level>.out`.
- `hl_labels.slurm` + `hl_pipeline_debug.slurm`: `TIMEOUT_S="${TIMEOUT_S:-28800}"`, exported;
  `hl_label_worker.sh` passes `--timeout "$TIMEOUT_S"`.
- `s0_hl_progress.py`: a **failed** column (a `.out` without the terminal line) beside labelled /
  unlabelled / running.
- Docs: CONTEXT.md Frame set entry (one sentence: units are rerun whole, never resumed; a frame
  is attempted once; the lock = job alive AND heartbeat fresh); `docs/hessian_learning_campaign.md`
  (the lock paragraph, `TIMEOUT_S`, the failed column, "a failed frame is a human's decision:
  `python -m openqha.data.frame_labels <mol> <gen> <b> <k> --retry`");
  `workflows/hessian_learning/README.md` step 03 paragraph.

**Blocked by:** nothing. **Unblocks:** the labels campaign in either route (sbatch rounds or
the tmux driver); ticket 25 (the production sequence page).

**Status:** implemented 2026-09-22; tianhe item open.

- [x] unit (`t_frame_labels.py`, +7): a lock owned by `999` with a fake `squeue` on PATH answering
      `RUNNING` and mtime 40 min ago -> free; mtime now -> held; fake `squeue` answering
      `COMPLETING` / `CANCELLED by 1000` / exit 1 -> free; PATH without `squeue` -> heartbeat age
      alone; a `runner` that sleeps 2.5 s with `HEARTBEAT_S` = 1 -> the lock's mtime advanced;
      SIGTERM sent to the process 1 s into a 3 s `runner` -> `SystemExit(143)`, no lock left;
      a `runner` raising `TimeoutExpired` -> `<stem>.out` present with the trailer line, status
      failed on the next call, `retry=True` reruns; `pending` lists never-run frames only and
      counts the failed one
- [x] `t_hl_campaign`: the progress table's failed column on the fake tree (15 / 4 / 1 / 10 / 1); the page names
      `TIMEOUT_S=28800`, the failed state, `--retry`, the 30-min heartbeat rule
- [x] unit group green (see the report); integration `t_hessian_compare_engine` untouched
- [x] one detail found while building: a cut (SIGTERM, or ORCA returning rc < 0 = killed by a signal) must also
      remove the run directory and anything half-copied under the stem, or a workstation run leaves
      `frames/.<stem>/` behind and a `.out` copied before the handler fired would read as failed; `label_one`
      does both in its cut branch. `--assemble` now exits 0 when no frame is WITHOUT a job (failed and refused
      frames are counted, not pending), so a round's task 0 stops signalling "resubmit" over frames no
      resubmission would touch.
- [ ] tianhe (user): the next labels gate's log shows `TIMEOUT_S 28800` in the report block
