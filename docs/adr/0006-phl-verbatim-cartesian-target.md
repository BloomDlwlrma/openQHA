---
status: accepted
date: 2026-09-23
---

# The loss is PHL verbatim: the Cartesian Hessian trained by random-probe HVPs; the Eckart projection is evaluation-only

## Context

`design-phl-loss.md` (2026-09-18, ticket 11) designed a projected, mass-weighted,
entropy-weighted Hessian loss whose target was the msRRHO entropy itself: the error
operator was `A = P_W M^-1/2 (H_theta - H_r) M^-1/2 P_W`, with `P_W` built from the
reference vibrational modes and weights from the entropy's mode-by-mode sensitivity.
S0-C-53 had already made the Cartesian matrix the training target, and S0-C-54 basin
frames only; the projected variants survived as "diagnostics and scan rows" — a `metric`
argument threaded through `phl.py` (seven functions), a branch in every `FrameConstants`
with seven extra slots, a three-way `mode_weighting` in the loss, the driver, the Slurm
script and every Record, a `modes` probe set that needed the reference eigenvectors, a
three-row balance ladder in the smoke fit, a `LOSS_EXACT` column in the judge, and two
flags in the fork's argument parser.

Nobody trains on the projected path. Rodriguez 2025 (element-wise RMSE), PHL (eq. 6,
MSE with Hutchinson probes), PFT (MAE on a randomly sampled column) and HIP (element-wise
MAE/MSE plus a subspace term in the reference Cartesian eigenbasis) all train on the
Cartesian matrix. HIP's mass-weighting and Eckart projection appear only where
frequencies are read — the standard vibrational analysis (Louck & Galbraith 1976;
gaussian.com/vib; HIP §5), applied to both matrices by `hessian_compare`, never to a
training loss.

Keeping the projected path cost tests, reading time, and carried a permanent risk: a
flag mace parses but our loss no longer reads sits in every `config.yaml` claiming a
target that did not run (the failure S0-C-56 was written about). It bought nothing the
judge does not already read from the trained matrix by the standard analysis.

## Decision

The training loss is PHL's as published (S0-C-64): the Cartesian Hessian's mean squared
error per structure over `(3N)^2`, estimated by Hutchinson random probes through
Hessian-vector products. "Projected" means the random-vector projection and nothing
else. The projected, mass-weighted, entropy-weighted path is deleted outright, not kept
as an option:

* `phl.py` loses `projector`, `reference_modes`, `entropy_weights`,
  `weighted_projector`, `error_operator`, `projected_loss_full`, `mode_basis_terms`,
  the `metric` argument and the `modes` probe set (tickets 35, 38).
* `phl_loss.py` loses `mode_weighting`, `temperature_K`, `preset` and the projected
  `FrameConstants` slots (ticket 35).
* The fork's commit D removes `--hessian_mode_weighting` and the `modes` choice of
  `--hessian_probe`, so an unread flag never enters a Record (ticket 36).
* The judge's `LOSS_EXACT` column (the projected norm) goes; `LOSS_CARTESIAN` — the
  training target itself — is the gate's quantity, and the frequency rows are stated as
  the standard vibrational analysis of the trained matrix (ticket 38).
* T03 (the projected design) is archived, not deleted; `design-phl-loss.md` is marked
  superseded (ticket 38).

Mass weighting and the Eckart projection exist only in evaluation: the judge's frequency
rows and the msRRHO result are computed by `hessian_compare` by the standard vibrational
analysis, exactly where a frequency is read. PHL's published weights are not adopted with
its form — the ratio `lambda_H / lambda_F` is dimensional (length squared) and was tuned
for ANI from scratch; the rule `w_H = w_F L_F / L_H` is measured on the base model with
the full matrix instead (S0-C-60).

## Consequences

* The probe and loss modules are about half their former size, with one target and no
  `metric` switch; a reader can no longer configure a run onto a target that is not the
  ruling's.
* The vibrational analysis survives in `thermochem/hessian.py` and `hessian_compare.py`,
  which keep their mass-weighting and Eckart projection — that is what a frequency is.
  Nothing in the training path projects or mass-weights.
* T04/T05 carry the derivation as the code runs it; T03 is the record of the superseded
  design. If the projected loss is ever wanted again, it must be re-argued from the
  entropy target, not re-enabled by a flag.
