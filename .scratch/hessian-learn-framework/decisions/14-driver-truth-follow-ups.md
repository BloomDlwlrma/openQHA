# Driver truth follow-ups: the multihead control mirror and the `--key=value` extras form

Type: task
Status: open
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

## Answer

<!-- resolver: fix, test, record -->
