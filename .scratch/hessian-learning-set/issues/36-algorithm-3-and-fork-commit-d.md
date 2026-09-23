# 36 — Algorithm 3 and fork commit D: the flags that no longer mean anything, and the probe field

**Set:** hessian-learning-set · **Spec:** `spec-phl-verbatim.md` (Step 3, Step 4) · **Rulings:** S0-C-64, **S0-C-67**

**Status:** ready-for-agent · **Blocked by:** 35 (done)

**Why.** Two things belong to the fork and therefore to one commit. First, S0-C-64 left
`--hessian_mode_weighting` and `--hessian_probe modes` in the parser: ticket 35 made
`build(args)` refuse them rather than ignore them, but a flag that only ever raises is a
flag that should not be offered, and mace writes every parsed flag into the run's
`config.yaml`, where it reads as a setting of the run. Second, S0-C-67 replaces the
derived probe seed with PHL's fixed-vector protocol, and the vectors reach the loss only
if the fork's data loading carries them into the batch. Both are parser/data changes in
`BloomDlwlrma/openQHA-Hessian` (branch `openqha-hessian`); one commit, one pin.

## Fork commit D (`..\openQHA-Hessian`)

- [ ] `--hessian_mode_weighting` removed from `mace/tools/arg_parser.py`
- [ ] `--hessian_probe` loses the `modes` choice (`rademacher`, `gaussian`, `cartesian`
      remain)
- [ ] a per-graph `valid_probes` field in the data pipeline, mirroring commit A's
      `hessian` / `has_hessian`: read from the config's `valid_probes` key (a flat
      `k_max * 3N` array), reshaped to `[k_max, 3n]`, `has_valid_probes` false when the
      key is absent — no default draw, no silent zero
- [ ] the fork's own tests: a config without the key loads; one with it round-trips
      through `AtomicData` and the collater with the rows in order
- [ ] `docs/fork.md` (openQHA side) records commit D's sha and what it changes;
      `mace_patch.apply(strict=True)` pin updated

## openQHA side (Algorithm 3, the driver and the Record)

- [ ] `phl_loss`: `MODE_WEIGHTINGS`, `DEFAULT_MODE_WEIGHTING` and the refusal branch in
      `build(args)` go — with the flag gone there is nothing left to refuse
- [ ] `05_train.py --mode-weighting`, the Slurm `MODE_WEIGHTING` variable and the Record
      key go; `hpc/slurm/hl_train.slurm` header examples updated (`PROBE=modes` appears
      in the gate example)
- [ ] the smoke fit's `mode_weighting` ladder and its projected diagnostic go; what it
      keeps is the Cartesian balance `w_H = w_F L_F / L_H` (S0-C-64) on the full matrix
- [ ] `phl.py`'s vibrational-analysis block loses the callers that ticket 35 left it
      (the smoke fit's diagnostic); the block itself goes with the judge in 38

## Acceptance

- [ ] a run submitted with `--hessian_mode_weighting entropy` fails in mace's parser, not
      in our loss, and the string appears nowhere in `config.yaml`
- [ ] `t_smoke_fit`, `t_train_run`, `t_phl_loss` green; `t_train_engine` green on the real
      3-epoch multihead fine-tune with the new fork pin
- [ ] the fork checkout is clean and pinned; the driver still refuses a mace that is not
      the fork (unchanged)

**Follows:** 37 (the stored probes on the openQHA side: the dataset writes them, the loss
reads them, the two `hashlib` calls die).
