# 02: The one-shot retry round (`--retry-failed`)

**What to build:** a failed frame can be re-attempted exactly once, deliberately, through
the ordinary round. `--retry-failed` makes the round's task list include the frames whose
ORCA output lacks the terminal line and for which no archive exists; before ORCA runs, the
previous failed `.out` is saved beside the file group as `<stem>.failed.out` (one slot,
replaced each time it is written), so the evidence of the failure survives the retry.
Finished frames are skipped exactly as today. After that single retry the frame is final
whatever the outcome: a retry that fails stays failed and is never selected again, and a
retry that is cut leaves no output and follows the ordinary cut policy (rerun whole; the
cap counts failures, not cuts). `--force` bypasses the skip only for the deliberate
"re-run all ORCA labels" operation and is off by default. This is the recovery path for
the draw300 mass failure and for any future failed frame.

**Blocked by:** 01 -- the retry is once-only, so firing `--retry-failed` before 01 is
deployed and one frame verified would burn every frame's single shot.

**Status:** ready-for-agent

- [ ] `--retry-failed` selects exactly the failed frames without an archive;
      failed-with-archive, finished and never-run frames stay out; the task list carries
      the retry intent to the worker, which maps it to the existing per-frame retry.
- [ ] Archive before overwrite: on a retry run with a failed `.out` present,
      `<stem>.failed.out` exists before ORCA starts and holds the previous output; writing
      again replaces the one slot; a finished frame is never archived or touched.
- [ ] Once-only: after one retry the frame is final -- success ends as finished; failure
      stays failed and no later `--retry-failed` selects it; a cut leaves no `.out` and the
      ordinary policy reruns it whole.
- [ ] Offline tests at the frame seam (fake runner): the archive exists while the runner
      runs; a retry with an archive present refuses without running; a failed retry keeps
      the archive; a finished frame is skipped with no archive; the listing predicate and
      its counts on a fake tree match.
- [ ] `--force` is off by default, is not wired into any round or the driver, and exists
      only for the deliberate full-relabel operation.
- [ ] The campaign page states the retry rule and the sequencing (01 -> count the failures
      -> verify one frame -> retry round); the archive is inert to parsers, the Dataset and
      the progress walk (no schema change).
- [ ] On Tianhe, after 01: one `--retry-failed` round recovers the frames it re-attempts;
      retried-and-failed frames remain failed and are never selected again.
