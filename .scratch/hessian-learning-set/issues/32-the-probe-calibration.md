# 32 — the probe calibration: is K = 4 fixed probes enough?

**Set:** hessian-learning-set · **Spec:** `spec-phl-verbatim.md` (Step 4) · **Rulings:** S0-C-55, S0-C-65

**Status:** done 2026-09-23 (the tool and its test; the measurement on the campaign's labels is the user's run on tianhe)

**Why.** S0-C-55 made the in-loop validation a Hutchinson estimator with K = 4 Rademacher
probes fixed per frame, and S0-C-65 kept K = 4. Nothing measured that choice: the per-frame
spread of a K = 4 draw is about half the value it estimates, and what matters is the spread of
the MEAN over the validation frames. The question "how do you know those 4 probes represent
anything" has a computable answer — sd(mean) = sigma / sqrt(n_frames) — and a measurable one.
This ticket measures it on real campaign molecules (19 atoms), not on the 10-atom fixture.

- [x] `scripts/tooling/s0_probe_calibration.py`: reads labelled basin frames, takes the
      engine's full Hessian once per frame, and reports check 1 (the production probes'
      offset from the exact value) and check 2 (the spread over independent fixed probe sets)
      for a list of K, with the variance eq. 2.3 predicts beside each
- [x] reads the campaign's molecule tree (`--tag`, assembled `basin.<level>.extxyz` first,
      the ORCA job files after) or any glob of labelled extxyz (`--frames`)
- [x] `--project-n`: decide "is K enough" at the validation split's size rather than at the
      frames read, since the per-frame spread is the estimator's property and the mean's
      falls as 1/sqrt(n)
- [x] Record `probe_calibration.<level>.toml`: `[ProbeCalibration]`, one `[[K]]` row with
      `VALID_PROBE_OFFSET` / `VALID_PROBE_SEED_SPREAD` (named `OFFSET` / `SEED_SPREAD` in the
      row), one `[[Frame]]` row per frame with the exact value, the production reading, the
      predicted sd and the stable rank
- [x] unit test on the stored fixture pair (no model call)
- [ ] the measurement on tianhe's 197 labelled molecules (the user's run)

**Closing (2026-09-23):** the tool measures the ESTIMATOR, so it needs no fine-tune and no
training: for each labelled frame it takes `judge.hessian_at` once (3N HVPs — the only model
call) and does the rest as linear algebra on the two matrices. Check 1 is
`|mean_n L^(K)_n − mean_n L_n| / mean_n L_n` with each frame's own production seed
(`phl_loss.frame_seed`, the set a run would draw); check 2 is the sd of `mean_n L^(K)_n` over
`--seed-sets` independent fixed sets; both are printed against the prediction of
`phl.estimator_variance` (eq. 2.3, Rademacher). `--project-n` rescales the deciding spread to
the validation split's size. Unit `t_probe_calibration.py` 18/18: the exact target to 1e-12,
unbiasedness over 200 sets within 4 s.e. at every K, the 1/sqrt(K) fall (K = 1 vs 16: ratio
4.0), eq. 2.3 within 15 % of the measured spread at every K, the 3N unit probes exact,
`summarise`'s three columns, the 1/sqrt(n_frames) fall, the reader on a glob (and skipping a
frame without a Label), and the tool end to end with a stub engine writing a Record whose
`[[K]]` rows carry every schema key. Measured with the real base model on the five
methyloxirane frames (10 atoms, 20 seed sets): exact mean 2.4503e-02 eV²/Å⁴, median stable
rank 2.4, and

| K | check 1 offset | check 2 spread (5 frames) | per-frame sd pred / meas | cost K/3N | spread at n = 900 |
|---|---|---|---|---|---|
| 1 | 63.8 % | 38.2 % | 0.79 / 0.86 | 3.3 % | 2.88 % |
| 2 | 28.8 % | 26.4 % | 0.56 / 0.53 | 6.7 % | 2.03 % |
| **4** | **12.1 %** | **20.9 %** | **0.39 / 0.39** | **13.3 %** | **1.44 %** |
| 8 | 5.5 % | 15.9 % | 0.28 / 0.28 | 26.7 % | 1.02 % |
| 16 | 2.3 % | 10.0 % | 0.20 / 0.19 | 53.3 % | 0.72 % |

so on this molecule the production seeds are a typical draw (offset inside the spread), the
predicted and measured per-frame spreads agree to two digits, and K = 4 clears the 2 % target
at the campaign's validation size. The campaign's own number — 19-atom molecules, where 3N =
57 and the stable rank may differ — is the tianhe run:

```bash
# on tianhe, whatever is labelled so far (197 molecules on 2026-09-23)
python scripts/tooling/s0_probe_calibration.py --tag draw300 --device cuda --project-n 900
```

If `SEED_SPREAD` at n = 900 comes out above 2 %, the answer is a larger K (the cost is
K/3N of a full matrix, still a fraction of it), not fresh probes per epoch: redrawing every
epoch would put an independent error of that size on EVERY reading, and the validation is
read only as a difference between epochs.
