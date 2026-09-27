# Round-1 xyz: assemble the labeled frames

Type: task
Status: open
Part of: [hessian-learn-framework](../map.md)

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

<!-- resolver: append what was done + evidence; set Status: resolved; add a line to the map's Decisions so far -->
