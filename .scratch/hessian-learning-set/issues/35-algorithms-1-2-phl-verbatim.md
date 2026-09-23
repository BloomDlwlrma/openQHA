# 35 — Algorithms 1-2 PHL verbatim: the frame's constants and the probes

**Set:** hessian-learning-set · **Spec:** `spec-phl-verbatim.md` (Steps 1-2) · **Ruling:** S0-C-64

**Status:** done 2026-09-23

**Why.** S0-C-64: train the full Cartesian Hessian with PHL's random probes, and let
"projected" mean the random vector and nothing else. Until this ticket the probe module
carried two metrics and four probe sets, and the loss module built a projector, the
reference modes and the msRRHO entropy weights for every labelled frame before it could
ask the model for anything. None of that is on the target any more, and a switch that is
never switched is a place for a run to go quietly wrong.

- [x] `phl.make_probes(H_r, mode, k, rng)`: no `metric`, no masses, no positions, no
      weights; `PROBE_MODES` loses `modes`; `METRICS` gone
- [x] `phl.loss_full` (eq. 1'), `cartesian_loss_full` kept as an alias for one release;
      `estimator_from_products` without the projector; `estimator_variance(H_theta, H_r, k)`
- [x] `phl_loss.FrameConstants(H_r)`: the Label, `nu = 9 N^2`, the frame's seed and its
      fixed probes — five slots instead of fourteen, nothing diagonalised; the cache key
      is the Label's bytes alone
- [x] the loss constructor loses `mode_weighting` / `temperature_K` / `preset`;
      `hvp_error` and `hessian_error_full` lose the metric branch; `eval_summary` loses
      `valid_target`; the `projected_*` aliases go
- [x] `build(args)` REFUSES a target that no longer exists instead of ignoring the flag
- [x] the derivations of spec step 2 as tests; `s0_probe_calibration.py` follows the
      new signature

**Scope boundary (decided while implementing).** The spec's file-by-file inventory is the
END state of tickets 35-38, not a list for this one. Two things therefore stayed:

1. **The vibrational-analysis block of `phl.py`** (`mass_weighted`, `projector`,
   `reference_modes`, `entropy_weights`, `weighted_projector`, `error_operator`,
   `projected_loss_full`, `mode_basis_terms`). Its callers are the smoke fit's diagnostic
   (ticket 36) and the judge's `loss_exact` row (ticket 38); deleting it here would have
   pulled both tickets forward. It now sits under a banner saying it is not a training
   path, that mass weighting and the Eckart projection are what a FREQUENCY is, and that
   it goes with its last caller. Nothing in `phl_loss` reaches it — asserted.
2. **`MODE_WEIGHTINGS` / `DEFAULT_MODE_WEIGHTING`**, reduced to `("cartesian",)`. They are
   no longer the loss's: what is left is the driver's vocabulary
   (`05_train.py --mode-weighting`, the Slurm variable, the Record key) and it goes with
   the fork's `--hessian_mode_weighting` in ticket 36. Until then the flag is still
   emitted, and `build(args)` raises on any value but `cartesian` rather than accepting a
   flag it ignores — a run that asks for the entropy target stops instead of silently
   training something else.

**Closing (2026-09-23).** `make_probes` now returns the raw draw, `r_j = H_r v_j` and a
denominator of `9 N^2 K` (stochastic) or `9 N^2` (the 3N unit set): the deterministic
limit of the estimator IS `get_hessian`. `FrameConstants` went from fourteen slots to
five and from an eigendecomposition per frame to a SHA-1. `estimator_variance` lost the
two arguments it did not use and gives both kurtosis cases.

Unit `t_phl` 26/26 and `t_phl_loss` 29/29, rewritten around the derivations rather than
around the old equation numbers:

- eq. 1' is the mean square per matrix element; the 3N unit probes reproduce it to 1e-12
  with zero variance (Derivation 2.3)
- 4000 Rademacher and 4000 Gaussian single-probe draws have their mean within 3 standard
  errors of it (Derivation 2.1)
- each sample variance sits within 3 sd(s²) of the closed form, where the band is read
  off the draws' own kurtosis, `sd(s²) = s² sqrt((kurt-1)/n)` — 7.8 % for Rademacher,
  13.7 % for Gaussian, because X = vᵀBv is a weighted sum of χ²₁ there. The fixed 5 %
  band of the old test was an invented number and the Gaussian draw failed it by 5.2 %
  for a reason that has nothing to do with the code
- `Var_Gaussian − Var_Rademacher = 2 Σ_i B_ii² / (9N²)²` to 1e-18, and it is positive;
  both fall as 1/K
- refused: `metric=`, masses, `mode="modes"`, a non-square Label, `mode_weighting=` on the
  constructor, `constants(H_r, masses, positions)`, and `build(args)` with
  `--hessian_mode_weighting entropy`
- on the loop: the unit probes through the HVP path equal the full-matrix path to 1e-12,
  the gradient matches a central difference, `H_r := H_theta` gives 0 with a zero
  gradient and `H_r := 0.81 H_theta` gives `0.19² ||H_θ||_F²/(9N²)`, and the fixed-probe
  validation of S0-C-55 still holds (same value twice, same under another mace seed,
  different between frames, different from a training draw)

The whole unit suite (59 files) is green; the projected diagnostics are still asserted by
`t_phl`, `t_judge` and `t_smoke_fit` because they are still live code.

**Follows:** 36 (Algorithm 3 and fork commit D: the flag itself, the Record key, the
driver, the Slurm variable, the smoke fit's ladder) — nothing in it is blocked now.
