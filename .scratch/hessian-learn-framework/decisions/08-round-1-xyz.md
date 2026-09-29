# Round-1 xyz: assemble the labeled frames

Type: task
Status: resolved
Part of: [hessian-learn-framework](../map.md)

> In flight: the runbook [08a](../implementation/08a-tianhe-merge-runbook.md); spec: [spec-round-1-xyz.md](../spec-round-1-xyz.md).
> Update 2026-09-29 (post-cancel state): the labeling sweep was CANCELLED at the training-sufficient set — 5,022 of 6,048 molecules carry labels (83.0%), ~29.9k Hessian frames, 0 failures/refusals; the remaining ~1,026 molecules / ~19k frames were declined (2.07 core-hours per frame; the budget goes to training). The final assemble refresh ran clean (5,631 s: 5,022 with labels / 1,026 untouched / 0 failed) and the post-cancel REBUILD is the canonical r1 dataset; the 2026-09-28 16.8k build and its recorded sha256 set are RETIRED (never cite or transfer them), and the Tianhe cleanup (stale fetch tarball; any pre-rebuild extxyz copies) rides the fetch. Pending: the final build's three summary lines + the fetch receipts (sha256 x5, tar path) — they land in this ticket's Answer, then Status: resolved.

## Question / work

The user's second work item (2026-09-26): the Tianhe `hl_labels` run ("orca hessian label", `--array=0-11`) has labeled frames; assemble every already-labeled frame into one xyz to feed round 1 of Hessian training and the debug run.

Work:

1. On Tianhe, run step 04 (`workflows/hessian_learning/04_dataset.py`) over the run's tagged tree to produce `<root>/<tag>/_datasets/<name>/mace_<name>.<level>.extxyz` — the only cross-molecule merge that exists (`openqha.data.dataset.build`; `REF_energy`/`REF_forces`/`REF_hessian` and `split` keys are added there).
2. **Pass `--split-by molecule` explicitly**: `04_dataset.py`'s argparse default is `frame` (the smoke/fit mode) while the library default and S0-C-65 say `molecule`, and `hl_labels.slurm`'s tail call omits the flag. Do not inherit the trap.
3. Settle with the user: the dataset name; snapshot semantics (take the labeled-as-of-now set while the sweep continues — do not wait for it); whether the merged xyz (and which Records) are copied back to the workstation (nothing of draw300 exists locally today; the local mirror holds only `dsgdb9nsd_000044`'s 5 frames and the two smoke datasets).
4. Report: frame counts (Hessian vs gradient, by generator), the `mace_<name>.<level>.extxyz` path, cross-checked against `s0_hl_progress` / the labels Records.

## Facts to stand on

- Labels live per molecule as `frames/<generator>.<level>.extxyz` (info key `hessian`, not yet `REF_hessian`) plus `frames/labels.<level>.{out,toml}`; `REF_*` keys only appear at the dataset step.
- Standing rulings: basin frames are the training frames (S0-C-54); other generators' labeled frames go to test (ADR 0005).
- `05_train` reads the merged file through `training/run.py split_files()` and passes `--energy_key REF_energy --forces_key REF_forces --hessian_key REF_hessian`.
- Runs where the tree is (Tianhe); the session hands the user the exact commands or runs them, whichever the user prefers — record which.

## Answer

Resolved 2026-09-29. Steps 0–7 of [08a](../implementation/08a-tianhe-merge-runbook.md) were
executed by the user on Tianhe; the labeling sweep was cancelled at the training-sufficient
set (user ruling: the budget is the training reserve) and the post-cancel rebuild is r1's
canonical artifact.

**The final dataset** — `$S0_RUNS_ROOT/draw300/_datasets/draw300_r1/`, build log
`dataset_draw300_r1_2026-09-29_1346.log`:

    dataset 'draw300_r1' at wb97m-d3bj_def2-tzvppd (split by molecule): 6048 molecules (304 test),
      324978 frames: train 27740 valid 868 test 1371 pool 294999; 29979 with a Hessian
    merged  .../mace_draw300_r1.wb97m-d3bj_def2-tzvppd.extxyz (29979 labelled frames, keys
      REF_energy / REF_forces / REF_hessian / split)
    R4 (S0-C-60): 27740 train frames with a Hessian -> Replay = 110960 frames at config_weight 10

The 13:46 run was cut (no traceback) between the `merged` print and the h5 write, leaving
the previous build's h5 in place; the export was re-run single-step at 20:15 (same inputs:
`index.dat` + the splits) and the fresh h5 is 781M. Everything else is the 13:46–14:58 build.

**Fetch** — `~/draw300_r1_fetch.tgz` (from the 17G directory: merged 4.4G, h5 781M,
`index.dat` 123M, `dataset.out`, `dataset.toml`). The five sha256s, r1's only fingerprints
(the 2026-09-28 set is retired, spec Q4):

    72e9d364d232f3cb7031b3f3a64e06c1cc78415e2adffa5486dd65dacf2e3622  mace_draw300_r1.wb97m-d3bj_def2-tzvppd.extxyz
    74d20775cd297ebb69167358b9db8bc56b50f8c93aeafbe567dd38f8634aaf82  molecules-draw300_r1.h5
    4e62a0a316f66f876631472cae82cd35d3146f09aae551806d4c04d8fd306088  index.dat
    ce62a5ca59162ac6a412023c0094de4676e4997d09f0985a1547307d2c49972d  dataset.out
    ce119e9ecc1b445b685388060580071fc67940c2bd579af5df5ebe456525d045  dataset.toml

**Sweep and budget** (detail in 08a's State and Budget-stop sections): cancelled 2026-09-29
at 5,022/6,048 molecules (83.0%) / 29,979 Hessian frames, 0 failures or refusals — 13 blocks
× ≈59.3 h ≈ 49.3k core-hours this round (~10.6k saved vs the wall); the remaining budget is
the training reserve (same account).

**Remaining**: place the dataset locally (its home is settled at [09](09-round-1-run.md)
claim) and run the Tianhe old-extxyz cleanup sweep (spec Q5; the exact extra set is confirmed
with the user at execution). The transferred copy is VERIFIED (2026-09-29): it sits at
`C:\Users\10704\Downloads\draw300_r1_fetch\` (extracted, five loose files, 5.3G) and all
five SHA256s above re-compute natively (`Get-FileHash`) and match. `primary_alcohol` remains
the lone low-coverage class (32.3%) if a top-up is ever bought.

Spec: [spec-round-1-xyz.md](../spec-round-1-xyz.md).
