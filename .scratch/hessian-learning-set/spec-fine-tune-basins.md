# Spec: the fine-tune as ruled -- basin Hessians, the Cartesian target, replay that is what the Record says (2026-09-21; amended 2026-09-22 to S0-C-58/59/60)

> **Amended 2026-09-22.** Three rulings after this spec was approved change its judge and its
> scan, and the code follows them (tickets 22 and 16 as built): **S0-C-58** -- the judge judges
> the Hessian, the training target; msRRHO is the thermodynamic property computed from it,
> NMS / MD / VIB are post-processing; the MD ramp is off by default; the tianhe smoke is one
> day of CREST + one day of ORCA, not the full labels. **S0-C-59** -- the gate row is the
> matrix itself: `held_out_hessian_cartesian` = the frame-weighted held-out
> `||H_theta - H_r||_F^2 / (9 N^2)`, engine / base - 1 <= 0; the low-mode line and the entropy
> row are reference rows. **S0-C-60** -- production trains ONE row, R4 (Replay = 4 x
> `N_TRAIN_HESSIAN` at `config_weight` 10, `w_H` = the epoch-0 balance measured by the driver);
> no scan, no R0-R3; the gate is CLOSED (`VERDICT = REPORTED`; `06_judge --gate` reopens it).
> The paragraphs below are kept as approved with the amended sentence marked **[S0-C-5x]**;
> where they disagree with the box, the box rules. The 15 % / 8.5 cm^-1 / 0.2 cal/mol/K numbers
> are round-2 placeholders that were never ruled; with the gate closed they only label rows.

Label: `done` (tickets 18-23 as built; 16 runs on tianhe). Tracker: `.scratch/hessian-learning-set/`. This spec supersedes the
training side of `spec.md` (its "The loss, in one place", the training rows of its
Implementation decisions, and ticket 16 as written); `spec.md`'s steps 00-04, the judge and
tickets 01-15 stand. Rulings it rests on: S0-C-53 (the Cartesian matrix is the training
target; frequencies are evaluation quantities), S0-C-54 (basin frames only; the msRRHO
result is the deliverable; extrapolation is reference), S0-C-55 (validation by 4 fixed
Rademacher probes, in the control signal), S0-C-56 (the replay draw and its assertions;
the Replay noun), S0-C-57 (the five replay rows; measure g first), and since 2026-09-22
S0-C-58, S0-C-59, S0-C-60 (the box above). Grilling rounds 7-12 hold the facts (round 12,
active learning, is parked); T04/T05 hold the derivations and were re-executed to this state.

## Problem statement

The msRRHO entropy of a QM9 molecule computed with MACE-OFF23_medium is wrong in the modes
below 300 cm^-1, because the base model's Hessians were never supervised (951,813 E/F frames,
no Hessian). The project needs a fine-tuned MACE-OFF23_medium whose msRRHO thermochemistry
at the reference level (wB97M-D3(BJ)/def2-TZVPPD) passes the judge, produced at a label
cost the allocation can pay, without forgetting what the base model knew, and with a
training run whose Record says what actually happened -- which today it does not: with
replay on, the fork drops the external loss silently, two replay flags are recorded but
never read, and a hidden duplication rule changes the frame ratio on small sets.

## Solution

Fine-tune MACE-OFF23_medium on **basin frames only** -- every frame of the training set is an
engine-surface minimum labelled with E, F and the analytic Hessian at the reference level
(~19,000 frames on draw300, ~4 days of 12 nodes) -- with PHL's loss on the **Cartesian
Hessian matrix** sampled by four Rademacher probes, against forgetting by a **Replay** of
SPICE train-split frames whose size, weight and disjointness are what the Record says,
validated every epoch by the same estimator with probes fixed per frame, in two stages as
the base was trained. **[S0-C-59]** The judge's gate rows are the Hessian matrix itself on the
held-out Hessian frames (engine no worse than base), the in_distribution row and the forgetting
line; everything computed from the matrix afterwards -- the low-mode frequency line, the msRRHO
entropy at the engine's own minima, Rodriguez's extrapolation views (off-minimum by RMS bin, the
MD temperature ramp on request, vibrational rows) -- is a reference row without a veto.
**[S0-C-60]** The gate is closed: every row is reported against its number, nothing is decided.
One production row runs, R4; the smoke run on the tianhe test set (one day CREST + one day
ORCA; the same recipe at 20 epochs) gives the seconds per epoch before it.

## User stories

1. As the project, I want the training set to hold basin frames only, so that every labelled Hessian is one the msRRHO result reads and the ORCA bill is ~4 days, not 22.
2. As the project, I want displaced frames to exist in the Dataset as a held-out generator, so that the reference rows can read the base and the fine-tuned model off the minimum without those frames ever entering training.
3. As the trainer, I want the loss to be PHL's Cartesian Hessian term sampled by k = 4 Rademacher probes, so that the training target is the literature's and the estimator is unbiased for it.
4. As the trainer, I want `w_H` measured on the base model at epoch 0 (w_H L_H = w_F L_F) ~~and scanned below it~~ **[S0-C-60: by the driver, `05_train --hessian-weight balance` as the default, no scan; `HESSIAN_WEIGHT_RULE` and the three terms in the Record]**, so that no literature weight is copied across units and weightings.
5. As the trainer, I want the validation Hessian term to be the same estimator with four probes fixed per valid frame, entering the total validation loss, so that the scheduler, the checkpoint and the Stage Two switch see the Hessian converge at ~5 % of the full-matrix cost.
6. As the trainer, I want Stage Two as the base was trained (`--swa`, `swa_hessian_weight` scaled with `swa_forces_weight`), so that the Hessian share of the loss is the same in both stages and both weight sets are in the Record.
7. As the trainer, I want `scheduler_patience` and `patience` set explicitly (the base used 20 / 50; mace's defaults 50 / 2048 never fire in 100 epochs), so that the learning rate and the stop are governed.
8. As the trainer, I want `--loss external` to survive multihead mode in the fork, so that a replay run trains the Hessian term at all.
9. As the trainer, I want the replay's size to be the file's frame count and its weight each frame's `config_weight`, recorded as `PT_N_FRAMES` and `PT_CONFIG_WEIGHT`, so that the Record never carries a flag mace does not read.
10. As the trainer, I want `--real_pt_data_ratio_threshold 0` passed explicitly, so that the frame ratio in the Record is the one that ran, on the smoke set as on the campaign.
11. As the trainer, I want a tool that draws the Replay from SPICE's train split by frame with one seed, writes `config_weight` and an ids file, and refuses when the draw shares a molecule with the forgetting set or with the in_distribution molecules, so that the forgetting line measures forgetting.
12. As the trainer, I want the replay rows R0-R4 (0 / 5,000 / 17,132 / 68,528 / 68,528 x weight 10) as nested prefixes of one permutation, every row printing frames per Hessian frame and the forgetting line, so that coverage and pull are told apart. **[S0-C-60: R4 alone is trained -- Replay = 4 x `N_TRAIN_HESSIAN`, the number the Dataset Record prints as `REPLAY_R4_FRAMES`, at `config_weight` 10; R0-R3 stay defined and unrun.]**
13. As the operator, I want the first job of the campaign to be a measurement on one A800 (~~basin-only smoke set + 5,000 replay~~ **[S0-C-58/60: the tianhe smoke set of one day CREST + one day ORCA, with R4's own recipe -- Replay = 4 x its Hessian frames at weight 10, `w_H` = balance -- at 20 epochs]**) whose Record carries seconds per epoch, u, g and the validation share, so that every later row's wall time is measured, not extrapolated.
14. ~~As the judge, I want the thermochemistry rows (msRRHO S at the engine's own minima vs the reference, per pinned molecule) to be the gate, so that the deliverable is what is judged.~~ **[S0-C-58/59: the gate row is the Hessian matrix itself -- what is trained is what is judged; the entropy row is computed from it and is a reference row, read from the engine's own msRRHO Records under its own tag (`--thermo-tag`). S0-C-60: the gate is closed.]**
15. As the judge, I want the RMS-displacement bins (0 / < 0.08 / < 0.15 / >= 0.15 A) and an MD temperature-ramp line reported as reference rows, so that Rodriguez's extrapolation views are on the table without gating a decision they were not asked to make. **[S0-C-58: the ramp only on request, `06_judge --ramp`.]**
16. As the judge, I want the three validation curves (E, F, Hessian) in every run's Record, so that a flat Hessian curve (the term not acting) or a rising force curve (w_H too large) is seen before the judge table.
17. As a reader of CONTEXT.md, I want **Replay** and **Held-out generator** defined, so that tickets and tables use one word each.
18. As a future reader, I want ADR 0005 to say why Hessians are labelled at basins only and why extrapolation is reference, so that the PHL every-frame rule is not re-adopted by accident.
19. As the msRRHO pipeline, I want the chosen model registered in `ENGINES` with its parameter fingerprint, Dataset index and config SHA, so that a thermochemistry Record names the potential it ran with.

## Implementation decisions

- **Training set = basin frames.** The Dataset gains a `train_generators` argument, default `("basin",)`: only those generators' frames enter train / valid; frames of every other generator (displaced, merged, saddle) are written to test -- the **Held-out generator** -- and the Dataset Record names the argument. `merged` stays out until ruled otherwise; `saddle` is never located, only counted.
- **Loss = PHL's Cartesian term.** The probe module gains the `B = I` path: raw Cartesian probes, reference side `H_r v`, denominator 9 N^2 k (9 N^2 for the deterministic Cartesian set); the loss's `mode_weighting` gains `cartesian` as the default; `flat` / `entropy` / `relative` stay as diagnostics and scan rows. Validation: the same estimator, k = 4 Rademacher probes fixed per frame (seed derived from the frame's Label bytes), entering the total validation loss; `wants_hessian_at_eval = False`; the fork's full-matrix evaluate hook stays available.
- **`w_H`** from the epoch-0 balance on the training split of the base model ~~, scanned 0.1x / 0.3x / 1x~~ **[S0-C-60: `run.hessian_weight_balance` -> `smoke_fit.epoch_zero_balance(probe=cartesian, mode_weighting=cartesian)` on the run's own train file, the driver's default; a number on the command line is `HESSIAN_WEIGHT_RULE = given`]**; Stage Two's `swa_hessian_weight = w_H x swa_forces_weight / forces_weight`; both sets in the Record.
- **Training control** passed explicitly: `scheduler_patience`, `patience`, `eval_interval 1`, `--swa` with `start_swa` at 3/4 of the epochs, EMA as the base; every value in the Record.
- **Fork commit C**: in multihead mode `args.loss` is overwritten only when it is not `external`; a test with a two-frame replay file asserts the external loss is built.
- **Replay corrections** in the driver: `--num_samples_pt` not passed; `PT_N_FRAMES` counted from the file; `PT_CONFIG_WEIGHT` read from the file; `--real_pt_data_ratio_threshold 0` always; the mace log's head sizes parsed into the Record.
- **The replay draw tool**: source = SPICE train split (`training_set.settings()`), uniform by frame, one seed, `--n`, `--weight`, ids file (file, index, SMILES, frames per molecule), Record with DOI / split / seed / n / weight; two assertions that fail the tool: no molecule shared with the forgetting draw's ids, none of the in_distribution molecules present. **[As built (ticket 19): SPICE's test split is by frame, so 15,542 of the 17,132 train molecules are also in the test file and a refusal would fire on every draw; the tool EXCLUDES such frames (any fragment of a frame's molecule in the forgetting ids or in the in_distribution four is skipped) and asserts disjointness afterwards; it refuses on a missing ids file or too few eligible frames. A deviation from the approved letter, flagged in the ticket's Closing, not yet ruled.]** The Dataset Record prints `N_TRAIN_HESSIAN` and `REPLAY_R4_FRAMES = 4 x N_TRAIN_HESSIAN` with the draw command of that size.
- ~~**The scan** (ticket 16 rewritten): the measurement job first; then R0-R4 on the campaign Dataset with the same seed, probes, w_H, epochs and judge; sizes' wall times from g; if g is small, R0, R1 and a weight row on R1's frames.~~ **[S0-C-60: ONE row, R4, after the smoke run; the five-step chain is `workflows/hessian_learning/README.md` "The production row R4, end to end": 04_dataset -> the two SPICE draws at `REPLAY_R4_FRAMES` -> `hl_train.slurm` R4 (balance default) -> `--register-copy`, then branch A and msRRHO for the pinned seven under `--tag r4` with `S0_ENGINE=<base>-R4` -> `06_judge --thermo-tag r4`, gate closed. Exercised end to end on the methyloxirane fixture on 2026-09-22 (`REPLAY_PER_HESSIAN_FRAME 4.0`, `PT_CONFIG_WEIGHT 10`, `HESSIAN_WEIGHT_RULE balance`, fork 904dd3b). What follows R4 -- a `w_H` row, R0-R3, reopening the gate -- is decided on its numbers.]**
- **The judge**: ~~gate rows = thermochemistry (msRRHO at own minima, per pinned molecule), in_distribution, forgetting~~ **[S0-C-59: `GATE_ROWS = (held_out_hessian_cartesian, in_distribution_degradation, forgetting)`; `REFERENCE_ROWS = (held_out_low_mode_mae_cm, model_error_s_ref_cal_per_mol_K, held_out_generator_hessian_vs_base, md_ramp_K)`; calibration with the gate open: base vs base reads exactly 0 and passes, the 0.9x potential (Hessian x 0.81) fails the gate and the verdict. S0-C-60: `verdict_of(lines, gate=False)` -> `REPORTED`, the default; `--gate` reopens]**; reference rows = RMS-displacement bins of the held-out generator's labelled frames, the MD temperature ramp (branch B's stability calibration with a 5 K / 5 ps ramp and its failure criterion; **off by default, `--ramp`**), the frequency rows, the entropy row (from the engine's own msRRHO Records, `--thermo-tag`); every run's Record keeps the three validation curves.
- **Glossary and ADR**: CONTEXT.md carries Replay (written) and Held-out generator; ADR 0005 records S0-C-54.
- **Kept as built**: hvp.py, phl.py, phl_loss.py, fork commits A and B, run.py, judge.py, smoke_fit.py, the Dataset's split, the frame generators and the classical draws already labelled (the smoke set's 52 displaced frames are the reference rows' data).

## Seams (where the behaviour is tested)

The highest existing seam is **the driver's command line and Record**: `05_train.py`
builds mace's argv from the Dataset and writes a Record. Every decision above is visible
there -- the argv (which flags are and are not passed), the Record's `PT_*`, weight and
control fields -- without running mace. Second seam, existing: **the loss module's
`forward` on a synthetic batch** (the probe path, the validation probes' fixedness, the
denominator). Third, existing: **the Dataset's index** (`train_generators` routing). Fourth,
new and small: **the draw tool's refusal** on a constructed overlap. The fork's commit C is
tested at the fork's own seam (its test suite, a two-frame replay file). No new seam is
proposed inside mace's training loop.

## Testing decisions

- A good test reads the argv, the Record, the index or the tool's exit -- never mace's
  internals. Prior art: `tests/unit/t_train_run.py` (argv pairs and Record), `t_phl_loss.py`
  (forward on synthetic batches), `t_dataset_mace_form.py` (index and files),
  `t_smoke_fit.py` (replay arithmetic), the fork's `tests/test_external_loss.py`.
- Unit: `train_generators` routes displaced / merged / saddle to test and the Record says so;
  the `B = I` probe path is unbiased on a stored Hessian pair and exact with 3N unit probes;
  validation probes are identical across two calls on the same frame and differ between
  frames; the driver never emits `--num_samples_pt`, always emits the threshold 0, records
  `PT_N_FRAMES` / `PT_CONFIG_WEIGHT` from a two-frame file; the draw tool refuses an overlap
  built on `tests/data/spice_tiny`; the Stage Two weight rule.
- Fork: multihead + `--loss external` builds the external loss (fails today).
- Integration (engine): one short basin-only fine-tune with a two-frame replay file on CPU:
  the Record's three validation curves exist, the Hessian curve moves, `MACE_FORK_COMMIT`
  is C's.
- The measurement job on tianhe is a run, not a test; its Record fields are asserted by the
  parser's unit test.

## Out of scope

Displaced-frame Hessian labels for the campaign (S0-C-54); the NMS generator rewrite and
ANI's S x K counts (design-nms-frames.md, shelved); transition-state location; from-scratch
training, RI-MP2 labels, a surface potential (round 8); a second-stage trigger from the
reference rows (round 9 Q5, closed); relative / entropy weighting as defaults (diagnostics
only); upstreaming the fork.

## Further notes

The cost table of round 10 is in u and CPU seconds; the first A800 run replaces it. The
replay draw is by frame (S0-C-56): molecules with many conformers weigh more, and the ids
file lists frames per molecule so the coverage of each row is a number, not a guess.

## Tickets, in order (`issues/`)

| # | ticket | status | blocked by |
|---|---|---|---|
| 18 | fork commit C and the replay corrections (driver argv + Record) | done 09-21 (fork 904dd3b) | -- |
| 19 | the Replay draw tool `s0_spice_pt_draw.py` (one seed, `config_weight`, exclusion + disjointness assertion) | done 09-21 | -- |
| 20 | basin frames only: `dataset.build(train_generators=)`, the Held-out generator, ADR 0005 | done 09-21 | -- |
| 21 | the Cartesian target as default, four fixed Rademacher probes at validation, explicit training control and the Stage Two weight rule | done 09-21 | -- |
| 22 | the judge's gate rows and reference rows (RMS bins, MD temperature ramp) | done 09-22; re-ordered under S0-C-58/59, gate closed under S0-C-60 | 20 |
| 16 | the campaign fine-tune: the smoke run, ONE row R4 (`w_H` = balance), registration, msRRHO under tag r4, the judge gate closed (rewritten for S0-C-60) | code done 09-22 (the R4 chain exercised on the fixture); the tianhe run is the user's | 08, 18, 19, 20, 21, 22 |
| 23 | T04 / T05 rewritten to S0-C-54..60 and re-executed | done 09-22; refreshed to the R4 chain the same day | 21, 22 |

Frontier: nothing on the code side; 16's steps run on tianhe as the labels land (`README.md`, the R4 block). Parked: grilling round 12 (active learning); the exclusion-vs-refusal letter of ticket 19; whether the placeholder thresholds get a measured basis (bootstrap s.e. of the held-out row) before the gate is reopened.
