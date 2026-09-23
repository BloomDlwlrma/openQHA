# 39 — the probes are PHL's standard normal, everywhere a default is chosen

**Set:** hessian-learning-set · **Spec:** `spec-phl-verbatim.md` (Step 2, Derivation 2.2) · **Ruling:** **S0-C-68** (revises S0-C-55)

**Status:** done 2026-09-23 · **Blocked by:** 37 (done)

**Why.** The stored probes were `±1` (Rademacher). That is not PHL's draw: PHL's Algorithm 1
draws the standard normal, which this repository's own documents have said all along
(`spec-phl-verbatim.md` step 2, `phl.make_probes`'s docstring, T04 §2). Rademacher came from
S0-C-55 on a variance argument, and when S0-C-64 said "PHL verbatim" I applied it to the loss's
FORM — the Cartesian full matrix, the `(3N)²` normalisation, the Hutchinson estimator — and
left the distribution alone without asking. The user found the `±1` in a printed file and said
so. Following the published method is the point of the ruling, so the draw follows it too.

- [x] `dataset.VALID_PROBE_MODE = "gaussian"`; `valid_probes()` draws `rng.standard_normal`
      and rounds to 8 digits before writing (the numbers in the file ARE the probes, so the
      only thing that matters is that every reader gets the same ones)
- [x] `REF_valid_probes` is written as float, not int
- [x] `phl_loss.VALID_PROBE = "gaussian"`; the loss constructor's and `build(args)`'s defaults
- [x] `phl.make_probes(..., mode="gaussian")` — the signature default too, so no code path
      silently picks the other draw
- [x] the driver (`05_train.py --probe`), the Slurm `PROBE`, the Record's description, the
      smoke fit's cost ladder and scan rows
- [x] the fork: `--hessian_probe` default `gaussian`; `_validated_probes` stops requiring
      `±1` and checks the shape and that the numbers are finite — the DISTRIBUTION is the
      dataset's business, not the data loader's
- [x] `rademacher` stays in `PROBE_MODES` and keeps Derivation 2.2; it is simply nobody's
      default

## The cost, stated and accepted

Both draws give the same expectation (`E[vᵀBv] = tr B`), so the estimator stays unbiased. The
variance does not match:

```
Var_Gau − Var_Rad = 2 Σ_i B_ii² / (9N²)² K   ≥ 0        (measured 1.76x on the propanal pair)
```

so at the same `K = 4` the validation reading's standard deviation is about `√1.76 ≈ 1.33`
times what it was. The user chose to keep `K = 4` rather than raise it to 8 to compensate.

Two things follow from the heavier tail, and both are in the code now:

1. the predicted spread must come from `estimator_variance(...)["gaussian"]` — the calibration
   tool reads `phl_loss.VALID_PROBE` instead of naming a draw;
2. **a check on a measured spread cannot use a fixed percentage.** With a normal draw
   `X = vᵀBv` is a weighted sum of `χ²₁`, so a sample variance carries its own error
   `sd(s²) = s² √((kurt−1)/n)`. `t_probe_calibration`'s flat 15 % band failed for exactly that
   reason once the draw changed; it now reads the band off the draws' own kurtosis, as
   `t_phl` already did. (Same class of mistake as the voided probe table of 2026-09-23.)

## On disk

One real frame (`dsgdb9nsd_000044`, 10 atoms, 3N = 30, k_max = 16):

```
REF_valid_probes="-0.39203034 0.3261445 -0.30452687 0.79621481 -0.46411058 ..."   480 numbers
comment line      47,542 characters   (39,004 with the ±1 set, +22 %)
```

The frame already carried `(3N)² = 900` Hessian numbers, so the probe table is not what makes
these files big.

## Result

Unit 59/59 files; fork `pytest test_valid_probes + test_hessian_label + test_external_loss`
20 passed; integration `t_train_engine` on the real 3-epoch multihead fine-tune.

**Left for 38.** T04 §2 and T05 still call Rademacher "the default" in prose; they are
re-executed by ticket 38 and the wording goes with that pass.
