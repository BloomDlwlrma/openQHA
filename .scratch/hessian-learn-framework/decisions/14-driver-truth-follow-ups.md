# Driver truth follow-ups: the multihead control mirror and the `--key=value` extras form

Type: task
Status: open
Blocked by: None.
Part of: [hessian-learn-framework](../map.md)

## Question / work

Two small driver/package truth items, both identified before round 1's production run and
deliberately kept out of it (the run-level flags carry round 1; spec Out of Scope):

1. **The multihead control mirror** (spec Q7 / Further Notes). In multihead mode the fork
   silently forces `lr = 1e-4` and EMA (with its own decay `0.99999`), while a run's
   nominal `control_settings` / Record keep whatever was asked (or the class defaults).
   Mirror mace's rule in `control_settings` -- and answer the `EMA_DECAY` schema question
   (the fork's decay has no field today) -- so the driver cannot describe a schedule it
   did not train, even when a caller forgets the explicit run-level flags. The fork's
   forced values with file:line live in
   [research/mace-finetuning-parameters.md](../research/mace-finetuning-parameters.md).
2. **`argv_pairs` and the single-token `--key=value` form** (the 2026-09-30 local gate's
   finding, [09f](../implementation/09f-the-local-gate.md)). `--mace-arg=--clip_grad=1.0`
   reaches mace fine (argparse reads `--opt=value`), but `argv_pairs` -- and therefore
   `config.yaml`, whose promise is "every mace argument, as given (the file mace itself
   can re-run)" -- reads the single token as a bare flag: the file gets
   `clip_grad=1.0: True` instead of `clip_grad: 1.0`. Round 1 works around it at the run
   level (two tokens per mace flag; the pinned EXTRA form; spec story 14 amended on
   2026-09-30), but the driver should either split `--name=value` tokens in `argv_pairs`
   or have `05_train.py` validate/document the two-token form, so the natural single-token
   spelling cannot produce a config that misstates the run. Extend
   `openQHA-Hessian/tests/unit/t_train_run.py`'s extras check with the `=` form.

Both are `openQHA-Hessian` code changes; neither blocks round 1.

Spec: [spec-driver-truth-follow-ups.md](../spec-driver-truth-follow-ups.md) -- the
2026-10-01 rulings (R1–R5) and the fix's decisions.

## Status note (2026-10-01)

Re-checked after the [15](../decisions/15-the-balance-on-the-probe-estimator.md) series
landed (package tip `547c394`, openQHA tip `f2aa2ee`, `mace` unchanged at `1110ffb`).
Both items remain open: `argv_pairs` still reads a single-token `--key=value` as a bare
flag (`run.py:434`), and no multihead mirror or `EMA_DECAY` field exists yet.

- The shared surfaces moved: `control_settings` is now `run.py:297` (no multihead
  parameter), `mace_argv` :367, `argv_pairs` :434, `run_training` :594. 15b added the
  `BALANCE_PROBE` / `BALANCE_N_PROBES` fields and amended descriptions -- the
  provenance-recording pattern this ticket's mirror should follow (state the rule that
  produced a recorded value, not only the value).
- Findings beyond the originals: (a) the fork silently *disables* multihead mode when a
  non-MP foundation path comes without `--pt_train_file` (`run_train.py:191-199`), so a
  `MULTIHEADS = True` run can be single-head in fact; (b) extras (`--mace-arg`, appended
  last) can override any emitted flag -- argparse last-wins -- including
  `--force_mh_ft_lr`, which voids the mirror's premise.
- The design choices were settled in the 2026-10-01 session (R1–R5; the spec is
  published), and implementation follows the spec.
- Sequencing (revised after [15f](../implementation/15f-the-yhbatch-name-defect.md)):
  the timing job already runs on the post-15f checkout, so there is no refresh window
  to catch -- the change lands on both repos and deploys at the next convenient sync;
  the arms are unaffected either way (their launch lines already carry the same
  effective values). The mace row stays "(unchanged, frozen)" while the rule is
  mirrored in our package rather than exposed by the fork.

## Answer

<!-- resolver: fix, test, record -->
