# 33 — the split's granularity: whole molecules on the test side (MACE-OFF's rule)

**Set:** hessian-learning-set · **Spec:** `spec-phl-verbatim.md` (Step 0) · **Ruling:** S0-C-65 (supersedes S0-C-49 / round 5 Q4)

**Status:** done 2026-09-23

**Why.** The production split drew every frame on its own (round 5 Q4): the pinned seven were
whole test molecules, everything else went 90/5/5 by frame. That put OTHER CONFORMERS OF A
TRAINING MOLECULE in the test split, so the judge's largest distribution — `interpolation` —
was not a generalisation reading, and the gate row mixed it with the seven pinned molecules.
MACE-OFF's own rule is the opposite (Kovács et al., *JACS* 147, 17598 (2025), §2.2): 95 % of
SPICE for training and validation, 5 % for testing, "splitting was performed at the molecule
level, ensuring that conformers of the same molecule do not appear in both train/validation
and test sets"; the validation set is then drawn by configuration inside the 95 %, which is
what mace's `--valid_fraction` does. We adopt that granularity.

- [x] `dataset.DEFAULT_SPLIT_MODE = "molecule"`, `TEST_FRACTION = VALID_FRACTION = 0.05`
- [x] `04_dataset.py --split-by` defaults to `molecule`; the by-frame mode stays for the smoke
      and fit Datasets, where whole-molecule test frames would eat a small label budget
- [x] a rebuild that CHANGES the mode is refused unless `--resplit` is given (one index cannot
      hold two split schemes); `RESPLIT` in the Record, and a note in the report
- [x] the judge: a non-pinned molecule whose `molecule_split` is `test` reads `out_of_molecule`,
      so `interpolation` is empty by construction in production and the gate row is a
      generalisation reading
- [x] tests

**Closing (2026-09-23):** `dataset.build(split_by="molecule")` was already implemented (it was
the smoke set's mode of rounds 3–4) and its semantics are exactly MACE-OFF's — test = whole
molecules, the pinned seven plus a per-stratum draw of `TEST_FRACTION`; valid = `VALID_FRACTION`
of the TRAINING molecules' labelled frames, drawn by frame; train = the rest — so the ticket is
a change of default, of fractions and of what the judge makes of it, not a new algorithm. Both
draws stay per-identity (`_rng(seed, "test", stratum, …)` for a molecule, `_rng(seed, "valid",
qid)` for its frames), so the prefix property across label rounds holds at both levels: round
t's splits are a prefix of round t+1's. `_previous_mode` reads `SPLIT_BY` from the Dataset's
Record and `build` refuses a mode change without `resplit=True`, because `keep_previous` would
otherwise leave one index with the old molecules split by frame and the new frames by molecule.
`judge.distribution_of` gained `molecule_split`: `in_distribution` still wins, then a pinned
molecule OR a whole test molecule is `out_of_molecule`, else `interpolation` — one function for
both modes, and the docstring says which mode produces which. Unit: `t_dataset` (the refusal
names both modes and `--resplit`; the resplit is recorded and a same-mode rebuild is not
refused; the production default is molecule at 5 %/5 %; `distribution_of` with and without
`molecule_split`), `t_dataset_generators` 15/15, `t_dataset_mace_form` 10/10, `t_judge` 39/39.

Arithmetic for draw300 (6,458 molecules, ~19,000 basin Hessian frames): 323 (+7 pinned) test
molecules ≈ 970 held-out Hessian frames for the judge, ~900 valid frames, `N_TRAIN_HESSIAN` ≈
17,130 and `REPLAY_R4_FRAMES` ≈ 68,520 — R4's recipe is unchanged. One consequence to carry
into the threshold work: a test frame's neighbours are now its own molecule's other basins, so
the bootstrap that would give the gate row a measured standard error must resample MOLECULES,
not frames.
