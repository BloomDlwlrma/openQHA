# The framework, restated (2026-09-21): PHL on MACE-OFF23_medium, so that E, F and H all reach wB97M-D3(BJ)/def2-TZVPPD

The user's intent, in one sentence: take MACE-OFF23_medium and, with PHL's method (energies,
forces and Hessian-vector products in one loss), bring its energies, forces AND Hessians to
the accuracy of the level it was trained at -- wB97M-D3(BJ)/def2-TZVPPD -- on the molecules
we care about. Not RI-MP2, not from scratch, not a thermochemistry-only potential.
Round 8 (from-scratch / RI-MP2 / surface-first) is superseded by this file.

> **Superseded in part by S0-C-54 (2026-09-21, grilling round 9).** The deliverable is the
> msRRHO thermochemistry at the reference level; the training set holds **basin frames only**
> (engine-surface minima, E/F/H on every one); displaced frames are never trained on;
> extrapolation (NMS by RMS bin, MD temperature ramp, vibrational rows) is post-hoc
> validation and reference, not a gate. Section 2's "every frame carries H" and the
> 22-day bill are withdrawn (basins only: ~19,000 Hessians, ~4 days of 12 nodes). The loss
> (section 3, Cartesian target, S0-C-53), the training mechanics (section 4) and the fixes
> (section 5) stand.

## 1. Goal, as three numbers that can fail

On held-out frames labelled at wB97M-D3(BJ)/def2-TZVPPD, per distribution
(interpolation / out_of_molecule / in_distribution), the fine-tuned model against the base:

| quantity | ruler (shipped paths only) | target |
|---|---|---|
| E | per-atom energy error, mace's PerAtomRMSE | no worse than the base on in_distribution; better on the rest |
| F | force RMSE | same |
| H | full Cartesian Hessian from `get_hessian` vs the Label through `hessian_compare`: element MAE, projected Frobenius, frequency MAE (all modes), low-mode MAE, mode overlap | better than the base on every row where the base is measurably wrong; no worse anywhere |
| forgetting | E/F on a fixed 5,000-frame draw of SPICE's test split | within 15 % of the base |

"DFT level" is not a single number; the judge's table with thresholds per row is the claim.

## 2. Data: the same Level as the base, every frame with E, F, H

- Level: wB97M-D3(BJ)/def2-TZVPPD (PSI4 for the base, ORCA 6.0.1 here; S0-C-38: analytic
  Hessian runs, 226 s for propanal on 8 cores). Same level -> no energy-zero shift,
  `--E0s foundation` stays, the base's own E/F errors are the yardstick.
- Frames per molecule: basin, displaced (298 K classical, 4 per basin), merged, saddle --
  the existing generators. PHL's rule: EVERY frame carries E, F, H. That replaces the
  round-5 Q7 (b) rule (Hessians at stationary frames only); the E/F-only displaced frames
  are what PFT showed to be harmful without a Hessian term, and PHL never has them.
- Cost (round 5's own numbers): Hessians on all ~100,000 campaign frames ~22 days of 12
  nodes, against ~8.4 for the stationary-only rule. The smoke set (65 frames, 7 molecules)
  and the fit set already have H on every frame.

## 3. Loss: PHL verbatim -- the Cartesian matrix is the target (S0-C-53, 2026-09-21)

    L = w_E L_E + w_F L_F + w_H L_H,   L_H = (1/(9 N^2 k)) sum_j || (H_theta - H_r) v_j ||^2,
    v_j Rademacher (or Gaussian) in Cartesian coordinates, k = 4 per structure per step.

This IS PHL's Hutchinson loss (eq. 2.1 of T05): one HVP per probe per batch, third-order
graph, unbiased for ||Delta H||_F^2 / (9 N^2). **The training target is the Cartesian Hessian
matrix as stored, as in Rodriguez, PHL, HIP and PFT; frequencies, modes and the msRRHO
entropy are evaluation quantities computed from it afterwards, never terms of the loss**
(user ruling 2026-09-21). The Eckart-projected, mass-weighted norms of T03 (flat, entropy,
relative) stay in the code as diagnostics and scan rows, not defaults. Their price and the
Cartesian target's price are both measured in T05 section 2-3: the rigid block (force error
rotated) is learned twice and is 0.1 % of the error on the fixture, ~100 cm^-1 at foreign
geometries; the Cartesian norm puts 65 % of its gradient in the 1000-2000 cm^-1 heavy-atom
band and < 1 % below 300 cm^-1 -- the judge's frequency rows are where that shows.
Validation: the full matrix (`probe = cartesian`, exact), so the logged number is the loss
itself.

`w_H`: measured on the base at epoch 0 so that `w_H L_H = w_F L_F` (T05 section 5 on the
fixture: cartesian 2.95, flat 1.71, entropy 987 at `w_F = 100`), then scanned 0.1x / 0.3x / 1x;
PHL's 0.09 / 0.30 is ANI's number and would put the Hessian term at 10x the force term here.

Code: `phl.make_probes` needs a `B = I` option (no `P`, no `M^-1/2`, `r = H_r v`,
denominator 9 N^2 k, or 9 N^2 for the deterministic Cartesian set) and `phl_loss` a
`mode_weighting = cartesian` that selects it as the default (ticket 17); the judge's primary
H row becomes HESSIAN_MAE / HESSIAN_RMSE with the frequency rows after it.

## 4. Training: fine-tune, replay, two stages

- `--foundation_model MACE-OFF23_medium`, the openQHA-Hessian fork (`--loss external
  --loss_module openqha.training.phl_loss:build`), float64, batch by frames.
- Replay (forgetting): `--multiheads_finetuning True --pt_train_file <SPICE train draw>`;
  the replay's size is the file's frame count, its weight the frames' `config_weight`,
  `--real_pt_data_ratio_threshold 0`. Fork commit C (keep `--loss external` in multihead
  mode) is REQUIRED before any replay run.
- Stage Two as the base did it: `--swa`, `swa_hessian_weight` scaled with
  `swa_forces_weight` so the Hessian share is unchanged; Record carries both weight sets.
- Epochs: the smoke fit decides; the base used 190 from scratch, fine-tunes converge in
  tens.

## 5. What stays, what changes in the tickets

| kept as built | tickets 10-15: hvp.py, phl.py, phl_loss.py (flat weighting is `mode_weighting = none`), fork A + B, run.py, judge.py, smoke_fit.py |
| fix first | fork commit C (round 7 Q1); replay knobs renamed and `PT_N_FRAMES` / `config_weight` recorded (Q2); threshold 0 (Q3); a SPICE train-split draw tool (Q5) |
| change | `HESSIAN_GENERATORS` -> all four generators (a per-run flag, recorded in `labels.<level>.toml`); `05_train` default `mode_weighting = cartesian` (B = I, ticket 17); ticket 16's scan rows: w_H x {0.1, 0.3, 1}, k in {2, 4}, replay {0, n1, n2}, freeze {0, 5}, all judged on the three distributions with E, F and H rows |
| dropped | RI-MP2 anywhere; from-scratch; the surface / thermochemistry split (entropy weighting stays available as a later re-fine-tune from this model, judged by msRRHO, if wanted) |

## 6. Order

1. Fork commit C + the replay corrections (hours). 2. Smoke fit on the 65-frame set with
the Cartesian target, all frames with H, `probe = modes` validation: the balance, the cost, the
first E/F/H judge table (tianhe, one A800). 3. `hl_labels` resubmitted with Hessians on all
generators (the irreversible bill; ride-along with the array that has not gone in).
4. Ticket 16 on draw300. 5. Only then any second objective.
