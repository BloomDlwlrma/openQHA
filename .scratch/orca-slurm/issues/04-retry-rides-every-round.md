# 04: The retry rides every round; `RETRY_ONLY=1` sweeps the failures

**What to build:** The labels round carries the one-shot retry by default and gains a
failures-only mode. A plain round lists the never-run/cut frames **and** the failed frames
without an archive -- marked `retry` in the task list's 5th column, the failed `.out`
archived before ORCA starts, exactly as ticket 02 defined. `RETRY_ONLY=1` on the round
script -- `--retry-only` on the labels driver -- lists **only** the failed frames without
an archive; nothing else is queued. The old switch and every reference to it disappear
from the driver, the round script, the worker's and frame CLI's comments, the command
examples, and the campaign page's command lines (the header/page equality check keeps the
two in step). All once-only mechanics -- the archive slot, the cut policy, the finality of
a retried-and-failed frame, the worker's 5th-column protocol -- are unchanged.

**Blocked by:** None (can start immediately)

**Status:** implemented 2026-09-26. Evidence: the full unit suite passes (65/65,
`python tests/run_tests.py`), including the extended seam C in `tests/unit/t_frame_labels.py`
(the default round carries the retryable failure; `--retry-only` lists exactly it; the
`--list` 5th column; the script-protocol pins: `RETRY_ONLY` in / `RETRY_FAILED` out,
`--retry-only` used exactly once on the LIST call, the old "not rerun" claims gone from
driver and script) and the updated page pins in `tests/unit/t_hl_campaign.py` (the header
examples and the page carry `GENERATORS=basin RETRY_ONLY=1 ...`); `bash -n` clean on
`hl_labels.slurm` and `hl_label_worker.sh`. **Not verified offline:** the live Tianhe
round after the pull (a flagless round carrying the retries; `RETRY_ONLY=1` sweeping the
failures) -- the user's run.

- [x] A flagless round's task list contains the retryable failed frames with `retry` in
      the 5th column, alongside the never-run/cut frames; `RETRY_ONLY=1` (`--retry-only`)
      lists exactly the retryable failed frames and nothing else (`pending()`'s new
      default and `retry_only=True`; seam C).
- [x] `RETRY_FAILED` / `--retry-failed` do not appear in the driver, the round script,
      the worker and frame-CLI comments, the command examples, or the campaign page's
      command lines; the driver's dry-run report names `retry_only` (pinned in the source
      check; the page's narrative paragraphs are ticket 06's).
- [x] Tests: the driver seam (seam C on the fake tree) pins default-carries-retries and
      retry-only-lists-only-those; the script-protocol pins check `RETRY_ONLY` in /
      `RETRY_FAILED` out and the single `--retry-only` pass-through; the page pins carry
      the new command strings; the unit suite passes (65/65).
- [ ] Not verified offline: a live round after the pull (flagless -> the retry rides;
      `RETRY_ONLY=1` -> failures only) -- the user's run. **Pending.**

## Notes (small extras/deviation beyond the ticket's text, for the record)

- The old ordinary-round claims were rewritten, not just the flag names: the driver's
  assemble summary and `hl_labels.slurm`'s closing echo say the next round carries the
  unarchived failures; a pin asserts "not rerun" is gone from both files.
- The script's `timeout` echo was reworded: ticket 03's keep-list said "unchanged", but
  its old tail ("not rerun by an ordinary round") became false with this ticket -- the
  line stays, its statement now matches the default.
- Runtime Record text in `frame_labels.py` that still describes the old policy (`REASON`
  "not rerun -- read it, then `--retry`", the `N_FAILED` schema description, the report
  note "a round does not rerun it") was deliberately left for ticket 06 ("the record
  catches up"); one of them is pinned by an existing check.
- Kept as ruled: `RETRY_ONLY` arms on any non-empty value; the archive slot, the cut
  policy, `--force`, the frame CLI's `--retry` and the worker's 5-column protocol are
  untouched; the listing trim is ticket 05; the page narrative/table, `README.md` and
  `CONTEXT.md` are ticket 06.
