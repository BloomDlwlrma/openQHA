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
deployed and one frame verified would burn every frame's single shot. **Cleared:**
ticket 01 is done and its site gate passed (job 7673439, 2026-09-25).

**Status:** implemented 2026-09-25. Evidence: the full unit suite passes (65/65,
`python tests/run_tests.py`), including the extended `tests/unit/t_frame_labels.py`
(seams B(i)-(iv), the `--force` behaviour and its wiring pin, seam C's `pending()` and
`--list` on a fake tree) and `tests/unit/t_hl_campaign.py` (the archive is inert to the
progress walk; the campaign page carries the retry rule and its sequencing); `py_compile`
clean on the three edited Python files; `bash -n` clean on the three edited shell files.
**Not verified offline:** the live Tianhe round itself (the last box -- it runs after the
sequencing rule: count the failures with `s0_hl_progress --tag draw300`, verify ONE frame
by hand, then `RETRY_FAILED=1 TAG=draw300 sbatch --array=0-11 --time=3-00:00:00
hpc/slurm/hl_labels.slurm`); the parsl route's per-frame `retry` argument (the same
`label_frame_task` carries it, but no offline test drives parsl); the worker's runtime
mapping of the 5th column (pinned textually in the tests, not executed -- it needs ORCA).

- [x] `--retry-failed` selects exactly the failed frames without an archive;
      failed-with-archive, finished and never-run frames stay out; the task list carries
      the retry intent to the worker, which maps it to the existing per-frame retry
      (`pending()` returns `(mol, g, b, k, retry)`; the task list's 5th column `retry`/`-`;
      `hl_label_worker.sh` appends `--retry`; `t_frame_labels.py` seam C).
- [x] Archive before overwrite: on a retry run with a failed `.out` present,
      `<stem>.failed.out` exists before ORCA starts and holds the previous output; writing
      again replaces the one slot; a finished frame is never archived or touched
      (`label_one` archives after the claim and before the runner; seam B(i) probes the
      archive WHILE the runner runs, and the `--force` check shows the slot replaced).
- [x] Once-only: after one retry the frame is final -- success ends as finished; failure
      stays failed and no later `--retry-failed` selects it; a cut leaves no `.out` and the
      ordinary policy reruns it whole (seam B(ii)/(iii); `retryable()` = failed `.out`,
      no archive; the cut path already removed only the exact `<stem>.*` KEEP names).
- [x] Offline tests at the frame seam (fake runner): the archive exists while the runner
      runs; a retry with an archive present refuses without running; a failed retry keeps
      the archive; a finished frame is skipped with no archive; the listing predicate and
      its counts on a fake tree match (`tests/unit/t_frame_labels.py`, seams B and C).
- [x] `--force` is off by default, is not wired into any round or the driver, and exists
      only for the deliberate full-relabel operation (the frame CLI's `--force`; a test
      check pins that `03_labels.py` and `hl_label_worker.sh` never pass it).
- [x] The campaign page states the retry rule and the sequencing (01 -> count the failures
      -> verify one frame -> retry round); the archive is inert to parsers, the Dataset and
      the progress walk (no schema change) -- the page's §3 table + retry paragraph and §4
      log lines; `t_hl_campaign.py` pins the strings and holds a progress walk unchanged
      by a `<stem>.failed.out` on disk.
- [ ] On Tianhe, after 01: one `--retry-failed` round recovers the frames it re-attempts;
      retried-and-failed frames remain failed and are never selected again. **Pending: this
      is the live round the user runs.**

## Notes (small extras beyond the ticket's text, for the record)

- `03_labels.py::main(argv=None)` -- one parameter added so the seam-C listing test can
  call the driver with arguments (the same shape `hpc/slurm/hl_list.py::main` already has).
- `workflows/hessian_learning/README.md` -- the step-03 mechanism paragraph now states the
  one-shot retry, the archive and `--force` (it documented `--retry` as the only lever).
- The on-site sequencing: count first (`s0_hl_progress --tag draw300`), then the retry
  round; the one frame the 01 site gate retried already spent its own shot
  (`dsgdb9nsd_006885 displaced_b01_k3`), so it will not be re-selected.
