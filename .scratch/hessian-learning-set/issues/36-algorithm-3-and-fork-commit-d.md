# 36 — Algorithm 3 and fork commit D: the flags that no longer mean anything, and the probe field

**Set:** hessian-learning-set · **Spec:** `spec-phl-verbatim.md` (Step 3, Step 4) · **Rulings:** S0-C-64, **S0-C-67**

**Status:** done 2026-09-23 · **Blocked by:** 35 (done)

**Why.** Two things belong to the fork and therefore to one commit. First, S0-C-64 left
`--hessian_mode_weighting` and `--hessian_probe modes` in the parser: ticket 35 made
`build(args)` refuse them rather than ignore them, but a flag that only ever raises is a
flag that should not be offered, and mace writes every parsed flag into the run's
`config.yaml`, where it reads as a setting of the run. Second, S0-C-67 replaces the
derived probe seed with PHL's fixed-vector protocol, and the vectors reach the loss only
if the fork's data loading carries them into the batch. Both are parser/data changes in
`BloomDlwlrma/openQHA-Hessian` (branch `openqha-hessian`); one commit, one pin.

## Fork commit D (`..\openQHA-Hessian`)

- [x] `--hessian_mode_weighting` removed from `mace/tools/arg_parser.py`
- [x] `--hessian_probe` loses the `modes` choice (`rademacher`, `gaussian`, `cartesian`
      remain)
- [x] a per-graph `valid_probes` field in the data pipeline, mirroring commit A's
      `hessian` / `has_hessian`: read from the config's `valid_probes` key (a flat
      `k_max * 3N` array), reshaped to `[k_max, 3n]`, `has_valid_probes` false when the
      key is absent — no default draw, no silent zero
- [x] the fork's own tests: a config without the key loads; one with it round-trips
      through `AtomicData` and the collater with the rows in order
- [x] the fork's commits are documented where they already are -- the fork note of
      `docs/tianhe_install.md`, which now lists A-D. There is no `docs/fork.md` and no
      pinned sha to update: `engine.provenance()` reads the checkout's commit and
      `05_train` refuses it when unknown or dirty, so committing the fork IS the pin

## openQHA side (Algorithm 3, the driver and the Record)

- [x] `phl_loss`: `MODE_WEIGHTINGS`, `DEFAULT_MODE_WEIGHTING` and the refusal branch in
      `build(args)` go — with the flag gone there is nothing left to refuse
- [x] `05_train.py --mode-weighting`, the Slurm `MODE_WEIGHTING` variable and the Record
      key go; `hpc/slurm/hl_train.slurm` header examples updated (`PROBE=modes` appears
      in the gate example)
- [x] the smoke fit's `mode_weighting` ladder and its projected diagnostic go; what it
      keeps is the Cartesian balance `w_H = w_F L_F / L_H` (S0-C-64) on the full matrix
- [x] `phl.py`'s vibrational-analysis block loses the callers that ticket 35 left it
      (the smoke fit's diagnostic); the block itself goes with the judge in 38

## Acceptance

- [x] a run submitted with `--hessian_mode_weighting entropy` fails in mace's parser, not
      in our loss, and the string appears nowhere in `config.yaml`
- [x] `t_smoke_fit`, `t_train_run`, `t_phl_loss` green; `t_train_engine` green on the real
      3-epoch multihead fine-tune with the new fork pin
- [x] the fork checkout is clean and pinned; the driver still refuses a mace that is not
      the fork (unchanged)

**Follows:** 37 (the stored probes on the openQHA side: the dataset writes them, the loss
reads them, the two `hashlib` calls die).

## What it came to

**Fork commit D** removes `--hessian_mode_weighting` entirely and the `modes` choice of
`--hessian_probe`, and adds `--valid_probes_key` (`REF_valid_probes`) with the field
behind it: `Configuration.properties["valid_probes"]` validated as `(k, 3N)` Rademacher
(a length that is not a multiple of `3N`, or an entry other than +-1, is refused), and
`AtomicData.valid_probes / has_valid_probes` concatenating like `hessian` / `has_hessian`
so a loss slices graph g's block by `cumsum(k * 3 n_g)`. Nothing draws a probe in the
fork: a structure without the key gets `None` with weight 0, and the loss that wants them
must say so itself (ticket 37). `tests/test_valid_probes.py` (5 tests) covers the key
specification, the validation, the masking, the batch order and that a dataset without
the key loads exactly as before; `tests/test_external_loss.py` asserts the flag is gone
and that both `hutchinson` and `modes` now exit in argument parsing.

**openQHA side.** `MODE_WEIGHTINGS` / `DEFAULT_MODE_WEIGHTING` are gone from `phl_loss`;
`build(args)` has no refusal branch left (there is no flag to refuse -- a stale
`--hessian_mode_weighting entropy` now fails in mace's own parser, before a model is
built). `mace_argv` emits no `--hessian_mode_weighting`, `run_training` takes no
`mode_weighting`, the Record loses `MODE_WEIGHTING` (schema, report, registry note, the
judge's echo) and `05_train.py` loses `--mode-weighting`. `epoch_zero_balance` lost the
argument and the two projected branches: it computes `phl.loss_full`, the full matrix,
and nothing else. The Slurm script loses the `MODE_WEIGHTING` variable and its gate
example stops saying `PROBE=modes`; `COST_SETTINGS` loses its `modes (exact)` row; the
production smoke fit's balance is one row instead of a three-weighting ladder and its
scan is the `w_H` multiples alone (the `modes` and `none` runs are gone).

**What this ticket did NOT touch.** The vibrational-analysis block of `phl.py` is still
there: its last caller is now the judge's `loss_exact` row alone, so it goes with ticket
38. `t_phl` still asserts it, because it is still live code.

## Result

Unit suite 59/59 files, 0 failures (`t_phl` 26, `t_phl_loss` 29, `t_smoke_fit` 13,
`t_train_run` 40). Fork `pytest tests/test_valid_probes.py tests/test_hessian_label.py
tests/test_external_loss.py` 20 passed. Integration `t_hvp_engine` 8/8 and
`t_train_engine` on a real 3-epoch multihead fine-tune against the rebuilt fork.

**Follows:** 37 -- the dataset writes the probes into the valid file, the loss reads them
from the batch, and both `hashlib` calls leave `phl_loss`.
