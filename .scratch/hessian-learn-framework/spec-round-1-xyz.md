# Spec: Round-1 xyz — the `draw300_r1` dataset and its one canonical version

Label: `ready-for-agent`. Tracker: `.scratch/hessian-learn-framework/`. Spec for
[08-round-1-xyz](decisions/08-round-1-xyz.md), rulings of 2026-09-28/29; execution:
[08a-the-tianhe-merge-runbook](implementation/08a-tianhe-merge-runbook.md).

**Rulings taken with this spec.** Q1: the dataset is `draw300_r1`, over the `draw300` tag's
selection (6,048 molecules with a Frame set). Q2: snapshot semantics — take the
labeled-as-of-now set, and once the set suffices for training CANCEL the labeling sweep
rather than wait for full coverage (2026-09-29). Q3: the fetch comes home: merged xyz +
OpenREACT h5 + `index.dat` + `dataset.{out,toml}` — no `dataset.dat` (never a build
product). Q4: the POST-CANCEL rebuild is r1's single canonical version; the 2026-09-28
build and its recorded sha256 set are retired. Q5: the Tianhe cleanup of superseded
extxyz artifacts rides the fetch.

## Problem Statement

Round 1 of Hessian training needs one cross-molecule xyz of every labeled frame that exists
on Tianhe. The campaign's 6,048-molecule tree had never been merged, nothing of draw300
existed locally, the labeling coverage was partial, and two stale artifacts (the 2026-09-28
merge at 16,824 Hessian frames, and the sha256 set recorded from an aborted first fetch)
risked being mistaken for the deliverable. The account's remaining budget is reserved for
training (same account, user ruling), so coverage had to be bought only up to "enough for
training", not to completion.

## Solution

Produce `draw300_r1` on Tianhe through the existing pipeline — `01_select` (the r1
selection), `03_labels --assemble` (write the per-molecule label files from the finished
ORCA file groups), `04_dataset --split-by molecule --export openreact` (the merged xyz, the
four split files, `index.dat`, the Records, the OpenREACT h5) — cancel the labeling sweep at
the training-sufficient set, rebuild on the now-static tree, and fetch the artifact set home
as a tar with a fresh sha256 set. Post-cancellation numbers: 5,022 of 6,048 molecules carry
labels (83.0%), 29,979 Hessian-bearing frames (train 27,740 / valid 868 / test 1,371), 0 failures or refusals; the unfinished ≈19k
frames were declined (2.07 core-hours per frame, ≈39k core-hours). The build prints the R4
recipe — Replay = 4 × train-with-Hessian frames at `config_weight` 10 (S0-C-60) — which
round 1's replay row consumes.

## User Stories

1. As the operator, I want every labeled frame of the campaign in one xyz with `REF_energy`
   / `REF_forces` / `REF_hessian` / `split` keys, so that round 1 trains without touching
   per-molecule files.
2. As the operator, I want `--split-by molecule` passed explicitly, so that the smoke/fit
   argparse default (`frame`) is never inherited by the production dataset (S0-C-65).
3. As the operator, I want the build's counts cross-checked against `s0_hl_progress` and the
   labels Records before the artifact is trusted.
4. As the operator, I want exactly one canonical version of r1 — the post-cancel rebuild —
   so that no reader can pick a stale file.
5. As the operator, I want the 2026-09-28 build and its sha256 set explicitly retired, so
   that old numbers are never cited again.
6. As the operator, I want the fetch recorded with sha256s of the FINAL files, so any
   transfer can be re-verified.
7. As the operator, I want `index.dat` (the split authority) in the fetch with the Records,
   so a future rebuild keeps the splits.
8. As the operator, I want the h5 in OpenREACT layout, so the PHL training route can read
   the dataset directly.
9. As the operator, I want the R4 replay line printed by the build, so the replay draw is
   parameterized from the actual train count.
10. As the operator, I want only Hessian-bearing (basin) frames in train (S0-C-54) and the
    other generators' frames held out (ADR 0005), so the loss's targets are unambiguous.
11. As the operator, I want snapshot semantics "as-of-now" — and, once sufficient, the sweep
    cancelled — so budget is spent up to "enough for training", not to coverage.
12. As the operator, I want the budget facts recorded (≈49.3k core-hours for the cancelled
    round; 2.07 core-hours/frame; training-first ruling), so the campaign history explains
    the stop.
13. As the operator, I want the superseded artifacts cleaned on Tianhe with the fetch (stale
    tarball; pre-rebuild extxyz copies), so only canonical files remain.
14. As the next session, I want ticket 08's Answer to carry the final counters + sha256s, so
    [Round 1: the first training + debug run](decisions/09-round-1-run.md) can claim against
    a frozen artifact.

## Implementation Decisions

- **The pipeline and its flags.** `03_labels.py --tag draw300 --name draw300_r1 --level
  wb97m-d3bj_def2-tzvppd --generators basin --assemble` refreshes the per-molecule label
  files (quiet form, ticket 07 of the labels set). `04_dataset.py --tag draw300 --name
  draw300_r1 --split-by molecule --export openreact` writes the merged
  `mace_draw300_r1.wb97m-d3bj_def2-tzvppd.extxyz`, the four split files, `index.dat`,
  `dataset.{out,toml}` and `molecules-draw300_r1.h5`. Basin-only throughout (S0-C-54).
- **The level has one spelling.** `wb97m-d3bj_def2-tzvppd` — `orca.LEVELS` holds exactly
  this; the literature-style hyphen form is a trap the driver refuses at startup.
- **Supersession (user ruling, 2026-09-29).** r1 has ONE canonical version: the post-cancel
  rebuild. The earlier build (2026-09-28; 16,824 Hessian frames) and the sha256 list from
  the aborted first fetch are retired — not used, cited, or transferred; the final fetch's
  sha256 set becomes the only reference.
- **The fetch manifest.** Merged xyz, `molecules-draw300_r1.h5`, `index.dat`, `dataset.out`,
  `dataset.toml` — tarred from the dataset directory. `dataset.dat` does not exist (the
  `select.dat` confusion); the four split files and `pool` are derivable and not fetched.
- **Cleanup with the fetch.** On Tianhe: delete the stale fetch tarball; the dataset
  directory is replaced in place by the rebuild (verify by timestamps); any other
  pre-rebuild "old extxyz" copies under the runs root are located and deleted in the same
  pass — the canonical files are never touched. (The exact extra set is confirmed with the
  user at execution.)
- **Numbers on the record.** Coverage 5,022/6,048 molecules (83.0%; every class 79–100%
  except `primary_alcohol` at 32.3% — the lone low class, a candidate for a later top-up at
  2.07 core-hours/frame). The cancelled round: 13 blocks ≈ 59.3 h each ≈ 49.3k core-hours;
  ≈10.6k core-hours saved versus the wall; the remaining budget is the training reserve.
- **The R4 line is part of the artifact.** The build prints `Replay = 4 × N_TRAIN_HESSIAN`
  frames at `config_weight` 10 and the `s0_spice_pt_draw.py` command that draws them
  (S0-C-60); round 1's replay row is parameterized from it.

## Testing Decisions

- The reconciliation is the test: on the static tree, `s0_hl_progress`'s labelled count and
  the build's labelled / with-Hessian counts agree (the step-5 delta — finished-but-
  unassembled frames — is zero once the refresh has run).
- The pass conditions on the refresh log: `0 with failures or refusals`; `frames failed 0,
  refused 0`.
- The fetch is verified by the fresh sha256 set (five files) against the transfer; the tar
  is the transport, the sha256s are the record.
- Rebuild idempotence is prior art (`keep_previous`, same name): a second run after more
  labels rebuilt byte-comparable files on the small set; a future top-up rides the same path.

## Out of Scope

- Completing coverage: the remaining ≈1,026 molecules / ≈19k frames and the
  `primary_alcohol` gap — declined under the training-first ruling; re-buyable at 2.07
  core-hours per frame.
- The training run itself and its stack/knobs/venue (settled at
  [09](decisions/09-round-1-run.md) claim time).
- Labeling displaced / merged / saddle frames (only basin is labeled in this campaign).

## Further Notes

- Execution detail and the case history (state block, budget stop plan, amendments):
  [08a](implementation/08a-tianhe-merge-runbook.md).
- The budget model: mean 1,862 s/frame (p50 1,687, p90 2,621; 4 ranks) → 2.07 core-hours per
  frame; the 13-block round burned ≈20k core-hours/day while running.
- The h5's route: the PHL reference implementation reads OpenREACT-CHON-EFH-layout h5
  files; `--export openreact` makes our dataset speak that layout.
- If more coverage is ever bought, refresh (03) + rebuild (04) under the same name keeps the
  splits (`keep_previous`).
