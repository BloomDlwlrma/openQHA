# 34 — the exact anchors: the full matrix on the validation file, before and after training

**Set:** hessian-learning-set · **Spec:** `spec-phl-verbatim.md` (Step 4) · **Ruling:** S0-C-65 (path A)

**Status:** done 2026-09-23

**Why.** The in-loop validation estimates the Hessian term with 4 probes fixed per frame
(S0-C-55). Whether that estimator is good enough cannot be settled by extrapolating a
fixture's variance — it is a property of a real run, of how large a change in the reading a
training decision turns on. The cheapest measurement that answers it is the exact quantity on
the same frames at two real model states: the base model before training and the fine-tuned
model after it. mace decides once, before epoch 1, whether its evaluation computes the full
matrix (`run_train.py` reads the loss's `wants_hessian_at_eval` into `output_args`), so a
"sometimes exact" epoch needs a fork change; path A avoids it by measuring outside the loop.

- [x] `run.exact_valid_hessian(model, valid_file)`: the frame-weighted mean of
      `||H_theta − H_r||_F^2 / (9 N^2)` over the validation file's labelled frames, from the
      full matrix (`smoke_fit.epoch_zero_balance` pointed at that file)
- [x] `run.run_training` takes both readings unless `exact_anchors=False`
- [x] Record: `VALID_HESSIAN_EXACT_BEFORE` / `_AFTER`, `VALID_HESSIAN_PROBE_LAST`,
      `VALID_PROBE_OFFSET_RUN`, `EXACT_ANCHORS`; a note in the report
- [x] `05_train.py --no-exact-anchors`
- [x] tests (unit + a real 3-epoch fine-tune)

**Closing (2026-09-23):** no fork change. `exact_valid_hessian` takes the model by name (a
registered engine) or by path (`_calculator_for`: the model a run has just written is loaded
directly, with the translation patch applied first, because no registry knows it yet) and
returns `None` when the file holds no Label. `run_training` measures the base model before
`_run_mace` and the written model after it, then puts the last epoch's probe reading beside
them and their relative distance in `VALID_PROBE_OFFSET_RUN`; `exact_anchors` is popped from
the settings before `mace_argv`, so mace never sees a flag of ours. The report prints the pair,
the change, the probe reading and the distance, and states what the pair is NOT: the validation
frames belong to TRAINING molecules (S0-C-65), so neither number is a generalisation reading —
that is the judge's test split. Cost: one full-matrix pass per reading, 3N force-graph passes
per frame; on draw300's ~900 validation frames of 19 atoms that is about a third of one epoch,
twice per run, against 34 % of EVERY epoch if the loop did it.

Unit `t_train_run` 40/40 (the four keys in the schema and the Record's round trip; `_record_num`
keeps the report from raising on a Record that holds a non-number). Integration
`t_train_engine` 28/28 on a real 3-epoch multihead fine-tune: both anchors positive, the BEFORE
one equal to `exact_valid_hessian("MACE-OFF23_medium", …)` to 1e-10, the probe reading recorded
beside them with the stated relative distance, the report carrying the note, and
`exact_anchors=False` leaving all four keys at −1.

This is what answers the question ticket 32 could not: the probe calibration measures the
estimator on stored matrices, while this measures what the estimator costs on the run whose
decisions it drove. The verdict columns of ticket 32 were voided on the same day for claiming
more than that.
