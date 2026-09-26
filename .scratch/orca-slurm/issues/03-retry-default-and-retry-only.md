# 03: The retry rides every round by default; `--retry-only` sweeps the failures; the listing stops shouting

**Status:** implemented 2026-09-26, split over three slices: ticket 04 (commit 256f1ef --
the retry rides every round; `RETRY_ONLY=1` sweeps the failures), ticket 05 (commit
3263349 -- the quiet listing; each task's retry slice) and ticket 06 (commit 69040ff --
the page/README/CONTEXT wording, the runtime Record text, the dated amendments). The
slices' Status lines carry the test evidence (unit suite 65/65; seam C and the page pins
extended); no ADR (the change is reversible and page-recorded). One honest note on this
ticket's own estimate: the "full-round task log <= ~2,000 lines" below did not count the
remaining per-frame work lines (~3.6k for a full draw300 task at 12 tasks -- so Further
Notes' "a few hundred" is off too); the trim meets ticket 05's restated target ("at most
a few thousand"). **Not verified offline:** the live round after the pull -- the user's
run (the one unchecked box below).

**Why (grilling rounds 1-2, ruling 2026-09-26).** The one-shot retry (ticket 02) required
remembering `RETRY_FAILED=1` on the command line, so the recovery behavior was invisible
and easy to forget -- and the flag's name read as "retry only" while its behavior was
additive, a distinction that consumed operator attention across the live draw300 rounds
(jobs 7675131 / 7675382 / 7675707). In the same rounds, every task's Slurm `.out` opened
by dumping the whole campaign into it: the frame list (~43,637 lines) plus a per-molecule
walk (~thousands of lines) -- ~47,000 lines of listing per task per round, pure I/O and
unreadable logs. The ruling: delete the flag; make the additive behavior the default; add
`--retry-only` for "recompute only the failures"; trim the listing to the failed frames
the round will touch plus the round's summary.

**Blocked by:** none.

## Problem Statement

The campaign operator runs recovery and production rounds by hand on TianheXY-CN.

1. The retry of failed frames happens only if the operator remembers `RETRY_FAILED=1`.
   Forgetting it silently orphans every failed frame -- the exact failure mode the retry
   exists to repair -- and the flag's name suggests the opposite of what it does.
2. There is no way to ask for *only* the failed frames. A "retry" run necessarily carries
   the whole remaining backlog (43,637 frames on draw300), so a 3-hour recovery sweep is
   not expressible without a multi-day commitment.
3. Every slurm `.out` begins with ~47,000 lines of listing (the full frame list + a
   per-molecule walk). It costs I/O on the shared filesystem, fills the log quota, and
   buries the one thing the log is read for -- which frames this task is retrying.

## Solution

- **The default round carries the retries.** A plain round (`GENERATORS=basin TAG=draw300
  sbatch ...`) lists the never-run/cut frames **and** the failed frames without an archive,
  marking the latter `retry` (the worker's existing 5th column). No flag.
- **`--retry-only` / `RETRY_ONLY=1`** lists **only** the failed frames without an archive;
  nothing else is queued; the round is over when they are.
- **The listing quiets down.** The driver keeps its summary block (counts, the retry rule,
  the task-list line); it prints no full frame list and no per-molecule walk lines. Each
  task's slurm log prints **its own slice's retry frames** (count + one line each) -- the
  frames that task will re-attempt.
- **The sequencing guard is rewritten** to match the new default: because any round now
  burns the shots of the failures it touches, the rule becomes "the fix deployed and
  verified -> count the failures -> verify ONE frame by hand -> **any round**".
- `RETRY_FAILED=1` / `--retry-failed` are deleted from code, script, comments and docs.

## User Stories

1. As the campaign operator, I want a round to carry the one-shot retries by default, so that a forgotten flag can never silently orphan failed frames again.
2. As the campaign operator, I want the mass round's command line to be flagless, so that the recovery behavior is the path of least resistance.
3. As the operator recovering from an incident, I want `RETRY_ONLY=1` to sweep only the failed frames, so that clearing the retry backlog is a few hours, not a multi-day commitment.
4. As the operator, I want a retry-only round to leave the never-run frames alone, so that "what did that round touch" is answerable from its command alone.
5. As the on-call debugger reading a task log, I want the listing phase to print only the retry frames this task will touch plus the round's summary, so that the log is readable in one screenful.
6. As the infrastructure owner, I want the per-task log to stop scaling with the whole campaign, so that I/O and log storage shrink by ~two orders of magnitude.
7. As the campaign planner, I want each task's own retry slice in its log, so that progress is visible from any single task without cross-referencing.
8. As a reviewer, I want the superseded flag gone from code, script, comments and docs, so that no reader believes it still exists.
9. As a new joiner reading the campaign page, I want the sequencing rule to say "no round before the fix is verified", so that the new default cannot burn shots unnoticed.
10. As the human doing ONE frame by hand, I want the frame CLI's `--retry` and `--force` levers untouched, so that manual recovery still works exactly as before.
11. As the trainer and analyst, I want the one-shot semantics -- the archive, the cut policy, the finality of a retried-and-failed frame -- unchanged, so that existing Records and their conclusions stay valid.
12. As the operator, I want an old command string containing `RETRY_FAILED=1` to be harmless after the pull (the variable simply has no reader), so that muscle memory cannot break a round.
13. As the tests' maintainer, I want the change locked at the existing seams, so that no new scaffolding is introduced.
14. As the auditor, I want ticket 02, the feature spec and the round-13 record amended with this ruling, so that the reversal of "never by an ordinary round" is on the record.
15. As the progress watcher, I want `s0_hl_progress` untouched, so that the census keeps reading the same disk state.
16. As the parsl-route operator, I want the driver's default to carry the retries there too, so that both routes stay semantically identical.
17. As the reader of the campaign page's log-line table, I want it to describe the new lines, so that the doc does not lie about what a round prints.

## Implementation Decisions

- **Driver interface.** `--retry-failed` is deleted; `--retry-only` is added (a plain
  switch; no argument). The listing predicate gains it as a parameter: by default it
  includes the retryable failed frames (`retry=True` in the round's entry tuples, i.e. the
  worker's 5th column `retry`); with `--retry-only` it returns **only** those, excluding
  never-run/cut, finished and archived-failure frames. The finished/archive/cut rules are
  unchanged.
- **Script interface.** `RETRY_FAILED` is deleted; `RETRY_ONLY` (any non-empty value)
  passes `--retry-only` to the task-list call only. The assemble call is untouched; the
  exit signal keeps its meaning ("1 = frames without an ORCA job remain"; for a retry-only
  round this stays 1 while the mass work is unfinished -- expected, not an error).
- **The quiet listing.** The driver drops (a) the full "frame list:" dump and (b) the
  per-molecule walk lines ("all N frames finished" / "N frames finished, M failed ...").
  It keeps: the level/resource/molecules block, the `frames ...` counts line (which
  already carries the retry count), the retry-rule line, the resume line, and -- in
  `--list` mode -- the `task list ... (N frames, R retries; column 5: retry or -)` line.
  **2026-09-26 addendum (ticket 04 implementation; review Q1(a)/Q2(b)).** The driver's
  summary block also prints, when `--retry-only` is passed, the notice `retry-only   the
  list holds ONLY the failed frames without an archive; nothing else is queued`. Kept
  deliberately -- a retry-only round's log must say nothing else is queued -- and it
  rides the summary block that ticket 05 leaves unchanged.
- **Each task prints its own retry slice.** After the slice file is built, the script
  prints a count line (`retries    R in this task`) and one line per retry frame of the
  slice, formatted like the old list (`molecule name`, `generator_bBB_kK`). A slice with
  none prints the count line alone. Target size: a full-round task log <= ~2,000 lines
  (from ~47,000).
- **Echo lines (the approved keep-list).**
  - always: `retry      the failed frames without an archive are re-attempted ONCE in this round; the failed .out is archived as <stem>.failed.out before ORCA starts; a failed retry stays final`
  - with `RETRY_ONLY`: `retry-only the list holds ONLY those failed frames; nothing else is queued`
  - unchanged: `frames ...`, `timeout ...`, `generators ...`.
- **Header examples** in the script: one flagless campaign example (the retry rides it)
  and one `RETRY_ONLY=1` example; the `RETRY_FAILED=1` example is deleted.
- **Dry-run report**: `retry_failed` key -> `retry_only`; `n_retries` stays.
- **Doc rewrites** (worded in the grilling round, applied here):
  - Campaign page §3: the state table's "one-shot `--retry-failed` round (below)" cell
    becomes "the next round (the one-shot retry, carried by every round by default)"; the
    retry paragraph's sequencing rule becomes "fix deployed and verified -> count the
    failures -> verify ONE frame by hand -> **any round**", with one sentence on why
    (every round burns the shots it touches); the fallback/retry commands become
    `GENERATORS=basin TAG=draw300 sbatch ...` (mass) and `GENERATORS=basin RETRY_ONLY=1
    TAG=draw300 sbatch ...` (sweep; the flagless mass round needs no retry flag at all).
  - §4 log-line table: the retry row describes the new lines.
  - `workflows/hessian_learning/README.md`: the "One attempt per frame" paragraph says
    every round carries the failures by default and `RETRY_ONLY=1` makes a failures-only
    round.
  - `CONTEXT.md` (Frame set entry): "read, not rerun, until a human asks (`--retry`)" is
    replaced by the new contract (the next round re-attempts it once by default, ruling
    2026-09-26; the archive is the durable finality marker; `--retry-only` sweeps; the
    frame CLI's `--retry` remains the human's lever).
  - Tracker amendments: ticket 02, `.scratch/orca-slurm/spec.md` and the round-13 grilling
    record each get a dated amendment stating the reversal and the new default. **No ADR**
    (the change is reversible and page-recorded).
- **Comments**: `hl_label_worker.sh` and `openqha/data/frame_labels.py` mention
  `--retry-failed` in prose only; the prose is updated (no behavior change in either).
- Everything about the archive slot, the cut policy, the locks, the worker's 5th-column
  protocol, `--force` and the frame CLI's `--retry` stays exactly as ticket 02 left it.

## Testing Decisions

- **What makes a good test here**: external behavior only -- the CLI's observable
  listing (the task-list file's contents and the printed summary/retry lines) and the
  script's protocol text. The mechanism (archive, locks) is already locked by earlier
  tests and is untouched.
- **Seams (all existing; no new seam):**
  1. **Driver seam C** (`tests/unit/t_frame_labels.py`): on the existing fake tree
     (finished + failed-no-archive + archived failure + never-run frames), (i) the
     **default** `pending()` lists the never-run frames **and** the failed-no-archive one
     with `retry=True`; (ii) `--retry-only` lists **exactly** the failed-no-archive frame;
     (iii) the `--list` file's 5th column is `retry` exactly for it; (iv) the printed
     output contains no full frame list and no walk lines (asserted as absence, e.g. the
     never-run molecule's non-retry line does not appear in retry-only output).
  2. **Script-protocol pins** (`t_frame_labels.py`): `RETRY_ONLY` present and
     `RETRY_FAILED` absent in `hl_labels.slurm`; `--retry-only` passed exactly once (the
     list call); the per-task retry-slice print present; the worker protocol pins (`-n 6`,
     `$5`, `[ "$retry" = "retry" ]`) unchanged.
  3. **Page pins** (`tests/unit/t_hl_campaign.py`): the new command strings
     (`GENERATORS=basin RETRY_ONLY=1 ...`, flagless mass round) and the rewritten
     sequencing phrase; the header-vs-page equality check keeps passing with the new
     header examples.
- **Prior art**: ticket 02's seam C (fake tree + `_load_driver`), the script pins in the
  same block, and `t_hl_campaign.py`'s page pins -- all reused, none replaced by new
  scaffolding.

## Out of Scope

- The archive/once-only mechanics, the cut policy, the locks, the worker's 5th-column
  protocol, the assemble/`04_dataset` step, and the `--force` / per-frame `--retry`
  levers (all unchanged).
- Running or queued site jobs: they keep the checkout they were submitted with; the
  pull/deploy itself is the user's step.
- `s0_hl_progress` and every other reader of the disk state.
- The q5 parked-rerun work and any other campaign.
- Log rotation or archival tooling -- the listing trim is the fix.
- The list-mode banner cosmetics (`over Parsl`, `resource local`, `%maxcore 3000`),
  previously classified cosmetics-not-bugs.

## Further Notes

- Live numbers (2026-09-25/26 reads): 43,637 basin frames to label on draw300 =
  41,989 never-run/cut + 1,472-1,664 retries (the count falls as rounds consume them);
  each task's log was ~47,000 lines; after this change expect a few hundred.
- The reversal, precisely: ticket 02 said the retry is "deliberate, never by an ordinary
  round"; this ruling makes it the ordinary round's default and moves the guard to
  "no round before the fix is verified". Recorded in the ticket/spec amendments (no ADR).
- After the pull, an old command string with `RETRY_FAILED=1` sets a variable no script
  reads -- behavior identical to the new default; nothing breaks.
- One known oddity kept on purpose: in a retry-only round, task 0's assemble still exits
  1 while the mass backlog is unlabelled -- the exit signal is about "frames without an
  ORCA job", not about this round.

## Acceptance Criteria

- [x] `--retry-failed` and `RETRY_FAILED` are gone from `03_labels.py`,
      `hl_labels.slurm`, comments and docs; the default round carries the retryable
      failures (`retry` in the 5th column) and `--retry-only` / `RETRY_ONLY=1` lists
      exactly those frames.
- [x] The driver prints no full frame list and no per-molecule walk lines; each task's
      log prints its own retry slice; the approved echo lines are the ones printed (the
      `timeout` line's old "not rerun by an ordinary round" tail was reworded, the line
      kept -- ticket 04's Notes).
- [x] The page's sequencing rule and the README/CONTEXT wording match the new default;
      ticket 02, spec.md and the round-13 record carry the dated amendment.
- [x] Tests: seam C extended (default, retry-only, quiet listing), script pins and page
      pins updated; the unit suite passes (65/65).
- [ ] Not verified offline: the live round after the pull (submit flagless -> retry rides;
      `RETRY_ONLY=1` -> failures only) -- the user's run. **Pending: this is the user's
      live round.**
