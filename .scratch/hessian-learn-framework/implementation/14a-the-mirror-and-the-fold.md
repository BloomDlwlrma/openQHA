# 14a: The mirror and the fold -- the driver resolves one control

Type: task
Status: resolved
Part of: [hessian-learn-framework](../map.md)
Serves: [14](../decisions/14-driver-truth-follow-ups.md) · spec: [spec-driver-truth-follow-ups.md](../spec-driver-truth-follow-ups.md).

**What happened.** The package side of ticket 14 landed as `f8e6dea` (review annotations
`d4640d6`): `control_settings` is now the single resolution -- the nominal defaults, the
extras' controlled keys folded in (the native channel, argparse last-wins), then the
fork's multihead rule mirrored on top (`mace/mace/cli/run_train.py:204-210`: LR 0.0001,
EMA True, EMA_DECAY 0.99999, under its `if not args.force_mh_ft_lr` gate) -- and the
mirror warns what it replaced. `EMA_DECAY` became a first-class control (schema field,
`--ema_decay` emission, mace's 0.99 default), `argv_pairs` learned the single-token
`--key=value` form (split at the first `=`), `--multiheads` without a Replay file is
refused, and the extras' own force verdict wins over the driver's flag. The pinned
launch lines' trained values do not change: their explicit values coincide with the
mirror.

## The doors, before and after

| door | before | after |
|---|---|---|
| the fork's silent forcing | Record / argv / config kept the request (or the defaults) while the fork trained 1e-4 / EMA 0.99999 | the mirror states the fork's values, and says what it replaced |
| the extras channel | argv won at mace's parser; the Record did not follow | the controlled keys are folded under the rule; the Record states the final values |
| a single `--key=value` extra | argv honored it; `config.yaml` read it as a bare flag | `argv_pairs` splits at the first `=`; the file re-runs truthfully |
| multiheads without a Replay file | the fork silently trained single-head | refused, naming `--pt-train-file` |
| EMA_DECAY | not recorded at all | a Double field; pre-change Records carry a comparability note |

`SWA_LR` stays derived from the requested LR (the fork does not recompute it either);
the forced path emits `--force_mh_ft_lr True` on multihead runs only; the extras tokens
stay appended verbatim (the argv stays "as given").

## Evidence

- Unit (`t_train_run.py`): 43 checks, 0 failed standalone (log `%TEMP%\oqt14-trun.log`)
  -- the mirror's three modes + the warning + SWA_LR, the fold under the rule, the
  `=`-form five cases, the emissions (the single-head negative included), the refusal,
  the description pins.
- Integration (`t_train_engine.py`): the fixture multihead Record carries
  (0.0001, True, 0.99999), cross-checked against the fork's own log line; the dry-run
  cases eq1 (the `=` form), fs1 (the driver force), ov1 (a single-head extras override
  recorded), fx1 (an extras force stepping the mirror aside), fz1 (an extras False
  beating the driver flag, the mirror applied).
- Suites: package `--all` 9/9, rc 0 (`%TEMP%\oqt14-pkg-suite.log`).

## Notes

- The fork is untouched (`1110ffb`); the rule lives in our resolution beside its pinned
  file:line reference, and the tests pin the constants as literals beside the same.
- The two-axis review's catches are folded in as `d4640d6`: the extras force verdict now
  beats the driver flag (before, a conflicting pair could let the fork force while the
  Record stated the request -- the exact lie class this ticket exists to kill), the
  single-head emission was dropped, the EMA/EMA_DECAY descriptions gained the pre-change
  comparability note, and three process tokens left the public comments.
