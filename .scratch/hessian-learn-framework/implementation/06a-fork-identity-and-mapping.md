# 06a: The fork identity and the mapping

Type: task
Status: resolved
Serves: 06
Blocked by: None (can start immediately)
Part of: [hessian-learn-framework](../map.md)

> Execution slice for [Identity after the split](../decisions/06-identity-after-the-split.md);
> the spec is [spec-identity-after-the-split.md](../spec-identity-after-the-split.md). It
> carries the spec's two small independent pieces — the fork constant flip and the
> old→new mapping — as one slice, per the breakdown. The spec's other half (the
> package-identity fields and their reader) rides the move, [01d](01d-the-switchover.md);
> the sweep items (the training test's pinned old commit B, the fork reader test's
> cosmetic strings, T05's wording, the hessian-learning-set pointers, the gate's
> install-path text) are [References sweep](../decisions/12-references-sweep.md)'s.
> Agent-run, no user action required. Wayfinder conventions apply: the `Status` protocol,
> no triage labels (`docs/agents/issue-tracker.md`, ADR 0009).

## What to build

The engine's identity strings stop naming the retired repository: `MACE_FORK` reads
`BloomDlwlrma/mace@openqha-hessian` — the rebuilt true fork, same branch — so every print,
Record field and message that interpolates it names the repository that exists;
`MACE_FORK_BASE` keeps `base-v0.3.16`, its comment refreshed to say the tag now lives on
the fork's rebuilt history at the binary-drop commit
(`5c2d7612eed88dc1463b5588a79c2d5f5718d322`). The engine comments that still name
`BloomDlwlrma/openQHA-Hessian` — whose GitHub address now belongs to the package — are
refreshed too. Nothing else changes: the strings are report-only (messages, prints, Record
fields) and no test pins the old value.

And the old world becomes translatable: [ADR 0011](../../../docs/adr/0011-mace-fork-rebuild.md)'s
consequence line "the old→new mapping note is ticket 06's obligation" is replaced by the
table itself — the six replayed commits and the moved base tag, old→new full 40-hex, as
verified by [02b](02b-the-rebuild.md) (old-tip ↔ new-tip diff empty) — plus one note that
the old `MACE_FORK` string names the address the package now occupies, and the archive
bundle's path
(`../../../../_backup/openQHA-Hessian-old-history-2026-09-26.bundle`). No separate docs page.

## Acceptance

- [x] `MACE_FORK` reads `BloomDlwlrma/mace@openqha-hessian`; `MACE_FORK_BASE` still reads `base-v0.3.16` with the refreshed comment; no comment in the engine names the retired repository; openQHA's unit group stays green (no occurrence pins the old value)
- [x] `s0_check_weights` prints the new fork identity (the strings are report-only — no behaviour changes)
- [x] ADR 0011 carries the old→new table (the six commits + the moved base tag, full 40-hex), the old-string note, and the bundle path — and no longer calls the mapping ticket 06's obligation

## Answer (2026-09-29, implemented in this commit)

The fork identity and the mapping landed as the slice specified. What changed:

- **Engine** (`openqha/potentials/engine.py`): `MACE_FORK` reads
  `BloomDlwlrma/mace@openqha-hessian`; `MACE_FORK_BASE` keeps `base-v0.3.16`, its
  comment refreshed to say the tag now lives on the fork's rebuilt history at the
  binary-drop commit `5c2d7612eed88dc1463b5588a79c2d5f5718d322`; the provenance comment
  at the `mace_fork_info()` call was refreshed too. Nothing else moved: the strings are
  report-only, no behaviour change, and no test pinned the old value (the two references
  to `engine.MACE_FORK` -- `t_train_run`'s refusal-message check and `t_engine_fork`'s
  print -- interpolate it).
- **ADR 0011** (`docs/adr/0011-mace-fork-rebuild.md`): the Consequences bullet now
  carries the old→new mapping -- the six commits and the moved base tag, full 40-hex
  (old ids from the archived checkout and the bundle, new ids from the fork; cross-checked
  against [02b](02b-the-rebuild.md)) -- the note that the old `MACE_FORK` string names
  the address the package now occupies, and the bundle's path; the "ticket 06's
  obligation" pointer is gone, and the Decision-section sentence it left behind now
  points at Consequences.

Checks run (WSL, `openqha` env, repo root):

- `python tests/run_tests.py` -> **all 63 tests passed** (unit group).
- Integration, item by item (the runner's `--all` run was externally terminated mid-way;
  the remaining tests were then run individually in sequence) -> **12 of 13 passed**; the
  one failure is `t_train_engine`, unchanged from 04a: it dies at
  `openqha/training/run.py:600` calling the deleted `engine.parameter_fingerprint` -- the
  training-side half of decision 04, owned by [The move](../decisions/07-the-move.md).
  `t_frames_engine` rewrites `tests/data/propanal_molecule` in place; the fixture was
  restored to HEAD after the runs.
- `python scripts/tooling/s0_check_weights.py` -> prints the new identity:
  `mace fork 1110ffbafd651a74d1d4678deb4748056d1dff0a (intended: BloomDlwlrma/mace@openqha-hessian on base-v0.3.16)`.

Flagged, not done (outside this slice's scope): `openqha/training/__init__.py`'s module
docstring still names `BloomDlwlrma/openQHA-Hessian` -- it rides the training side's move
([01d](01d-the-switchover.md)) or the sweep ([12](../decisions/12-references-sweep.md);
its list does not mention this file yet); `openqha/training/run.py`'s gate message is
already on 12's list.

## Review record (2026-09-29, annotations)

Two-axis review of the slice's diff (against its parent `c2205be`), per the
`code-review` skill, with the dispositions below:

- unit group: **all 63 tests passed** (`python tests/run_tests.py`);
- integration group: **12 of 13 passed** -- the one failure is `t_train_engine`, unchanged.

**Standards (repo standards + the smell baseline).**

- No hard violations: the changed text uses no `CONTEXT.md` `_Avoid_` word; the edits sit
  inside the slice's declared scope; the `#:` comment block matches the file's style.
- The ADR amended in place, undated (precedent dates amendments: ADRs 0001/0003/0004)
  -> kept: ticket 06a rules the consequence line "replaced by the table itself"; the
  mapping is the existing consequence being filled in, not a new decision, and the
  file's history dates it.
- "each old commit vs its replay" used the word `CONTEXT.md` reserves for training
  Replay -> fixed to "its rebased counterpart" before the commit.
- The mapping now lives in two formats (ADR full 40-hex; 02b short-old/full-new), the
  tag fact a third time in the constant's comment -> kept: the ticket ruled the ADR the
  mapping's home ("No separate docs page") and 02b is an immutable verification record;
  the comment states the base-tag fact only.

**Spec (this ticket).**

- All three acceptance items verified: `MACE_FORK`/`MACE_FORK_BASE`/no retired name in
  the engine; `s0_check_weights` prints the new identity; ADR 0011 carries the table, the
  old-string note and the bundle path (all 14 SHAs re-verified against the archived
  checkout, the bundle heads and the fork; the relative bundle path resolves from the
  ADR's directory).
- No missing requirements, no scope creep, no implemented-but-wrong findings; the
  expected absences (package fields -> 01d; sweep items -> 12) confirmed absent.
