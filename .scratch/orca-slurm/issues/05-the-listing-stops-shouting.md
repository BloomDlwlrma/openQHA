# 05: The listing stops shouting

**What to build:** A round's task log no longer opens by dumping the campaign into it.
The labels driver keeps its summary block (the counts line -- including the retry count --
the retry rule, the task-list line) and stops printing the full frame list and the
per-molecule walk lines. Each task's log then prints **its own slice's retry frames**: a
count line and one line per frame (molecule + frame tag) -- the frames that task will
re-attempt; a slice with none prints the count line alone. A full-round task log drops
from ~47,000 lines to at most a few thousand.

**Blocked by:** 04 (The retry rides every round; `RETRY_ONLY=1` sweeps the failures) --
sequenced because both tickets edit the same files and test block

**Status:** implemented 2026-09-26. Evidence: the full unit suite passes (65/65,
`python tests/run_tests.py`), including the extended seam C in `tests/unit/t_frame_labels.py`
(the driver's printed round holds no full frame list and no per-molecule walk lines --
a second fake molecule, every frame finished, pins the absent walk line -- and no frame
tag or `(retry)` marker appears; the `hl_labels.slurm` pin holds the per-task retry
slice: the `retries    R in this task` count line, `n = split($1, p, "/")` and the
`generator_bBB_kK` printf); `bash -n` clean on `hl_labels.slurm`. **Not verified offline:**
the live Tianhe round after the pull (a task log carrying only its own retry slice) --
the user's run.

- [x] The driver prints no full frame list and no per-molecule walk lines in any mode;
      the summary block and the `task list ...` line are unchanged.
- [x] Each task's log prints its own retry slice (count + per-frame lines) after its
      echoes; a slice with no retry frames still prints the count line (the awk prints
      nothing for an empty slice, so the count line stands alone).
- [x] The approved echo lines (`frames` / `timeout` / `retry` / `generators`, plus the
      `retry-only` variant when armed) are what the log shows (unchanged; already pinned).
- [x] Tests: negative assertions on the driver's printed output (no full list, no walk
      lines) and a script pin for the per-task retry-slice print; the unit suite passes.
