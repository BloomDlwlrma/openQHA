# 06: The record catches up

**What to build:** The documentation and the record match the new default. The campaign
page's sequencing rule becomes "the fix deployed and verified -> count the failures ->
verify ONE frame by hand -> **any round**", with the reason (every round now burns the
shots it touches); its retry narrative, its commands, and its log-line table describe the
default-carried retry, the `RETRY_ONLY=1` sweep and the quiet listing. The workflow
README's "One attempt per frame" paragraph and the step-03 mechanism paragraph, and
CONTEXT.md's Frame-set entry, say the same. Ticket 02, the feature spec and the round-13
grilling record each carry a dated 2026-09-26 amendment recording the reversal of
"deliberate, never by an ordinary round" and the new guard. No ADR (the change is
reversible and page-recorded).

**Blocked by:** 04 (The retry rides every round; `RETRY_ONLY=1` sweeps the failures),
05 (The listing stops shouting)

**Status:** implemented 2026-09-26. Evidence: the full unit suite passes (65/65,
`python tests/run_tests.py`); `tests/unit/t_hl_campaign.py` pins the page's new
sequencing ("**any round**", "rides every round by default", "burns the shots of the
failures it touches"), the §3 table's "carried by every round by default", the §4
log-line row ("re-attempted ONCE in this round", "retries    R in this task"), the §3
fallback list ("plus the failed frames without an archive (their one retry)") and the
README/CONTEXT contract, and asserts the page holds no `RETRY_FAILED` / `--retry-failed`
/ `(retry)` / "never by an ordinary round"; `tests/unit/t_frame_labels.py`'s REASON pin
now reads "carried once by the next round" and its source pin holds "not rerun" out of
the driver, the round script, the frame CLI and the debug script; `py_compile` clean on
the edited Python, `bash -n` clean on `hl_pipeline_debug.slurm`. **Not verified
offline:** none new -- the set's live Tianhe round remains the user's run.

- [x] The page's sequencing rule, retry narrative, commands and log-line table match the
      shipped behaviour; its pins pass.
- [x] README and CONTEXT.md use the new wording (no "until a human asks (`--retry`)",
      no `RETRY_FAILED`).
- [x] Ticket 02, the feature spec and the round-13 record carry the dated amendment; no
      ADR is added.
- [x] A repo-wide search for `RETRY_FAILED` / `--retry-failed` finds it only inside the
      dated amendment notes quoting the old name -- nowhere else.
      *Judged by the criterion's intent (reader-facing docs): the surviving hits are the
      tracker's own change records -- this set's tickets 02-06, `spec.md`'s and the
      round-13 record's amended text, and the other set's ceding note (its README line
      is now fixed) -- plus the tests' negative pins, which assert the absence on
      purpose. No page, README, CONTEXT entry, runtime string or code comment presents
      the flag as live.*

## Notes

- **Small extras beyond the ticket's text, for the record.** `hpc/slurm/hl_pipeline_debug.slurm`'s
  `TIMEOUT_S` comment ("a killed frame is failed, not rerun (ticket 24)") now says the
  next round re-attempts it once; the ticket-05 session flagged it for this ticket, and
  the new text is pinned (`t_frame_labels.py`). The frame CLI's runtime Record text
  (`N_FAILED`'s schema description, the failed frames' `REASON`, the report note, the
  `run()`/`main()` docstrings) was updated here too -- ticket 04's notes deferred it to
  this ticket, and the `REASON` change moves its test pin.
- The test pins also extend the handoff's named seams inside their existing blocks:
  `t_hl_campaign.py` gains the README/CONTEXT wording check, and `t_frame_labels.py`'s
  source pin reaches the frame CLI and the debug script (the "not rerun" negatives).
  The spec review found one more stale spot the handoff had missed: §3's fallback
  paragraph described the task list as "no ORCA job on disk ... in the round's
  generators" only; it now adds "plus the failed frames without an archive (their one
  retry)", pinned. The review also moved the ruling's attribution to ticket 03 (it
  ruled the reversal; ticket 04 implemented it) in the page and the README.
- **Reported, not changed** (outside this ticket's text; the user decides):
  `scripts/tooling/s0_hl_progress.py` still describes failed frames as "not rerun" --
  its module docstring and its summary print (`{} failed (not rerun: read the .out,
  then --retry)`). Ticket 03's story 15 / the spec's out-of-scope list kept the progress
  tool's reading untouched; the wording now trails the default. Options: (a) fix the
  strings here; (b) a follow-up ticket; (c) leave. Recommendation: (a) -- wording only,
  no behaviour change.
