# The smoke fit: what the campaign's settings should be read off (ticket 15)

Run 2026-09-21 on this machine (WSL, CPU, MACE-OFF23_medium `e986a6cf...`, the mace fork
`0.3.16+openqha`). Everything below comes from
`scripts/production/s0_hl_smoke_fit.py --tag rings --name smoke --stage all --epochs 30`
and one extra run (below); the Records are under
`$S0_RUNS_ROOT/rings/_datasets/smoke_fit/` (`smoke_fit.{out,toml}`, `train/<run>/`,
`judge/<run>/`).

## Read this first: how small the fit set is

| | |
|---|---|
| frames | **5** (all with a reference Hessian), from **one** molecule -- dsgdb9nsd_000044, 2-methyloxirane |
| split | 4 train, 1 valid, by frame (`PURPOSE = fit`: the pinned rule is off) |
| epochs | 30, CPU |

The ticket asks for the seven molecules' 65 frames. **They are not on this machine**: the
local ORCA run of ticket 03 labelled one molecule, and the 65-frame smoke set lives on
tianhe. So this is the machinery proved end to end and the numbers measured on what is
here -- not the campaign's numbers. Specifically:

- the judge's rows here are `train` and `valid` of the SAME molecule, so every accuracy
  number is interpolation, and with four training frames much of it is memorisation;
- an ordering between settings measured on four frames may not survive on 65, let alone
  on 6,458 molecules;
- the cost table is the one thing that transfers directly, because it is a per-step cost,
  not an accuracy.

Everything below is stated so that the tianhe run can contradict it.

## 1. The epoch-0 balance: `w_H` is not transferable, in any direction

The base model, before a single step, on the four training frames:

| mode weighting | L_E | L_F | L_H (eq. 1, exact) | w_F L_F | **w_H at which w_H L_H = w_F L_F** |
|---|---|---|---|---|---|
| entropy (eq. 3) | 1.937e-7 | 7.366e-4 | 6.463e-5 | 7.366e-2 | **1139.6** |
| flat | 1.937e-7 | 7.366e-4 | 4.213e-2 | 7.366e-2 | **1.748** |

**652x apart.** The entropy weights suppress exactly the part of `||A||_F^2` that is
largest (the stiff modes), so the same balance condition gives a completely different
number. Two consequences:

1. PHL's and Rodriguez's `w_H/w_F ~ 0.25-0.30` is a CARTESIAN, unweighted band. Our flat
   number (1.75 with `w_F = 100`, i.e. `w_H/w_F = 0.017`) is already an order below it,
   and the entropy number is four orders above. The design said the band would not
   transfer (section 2.6); it does not, and now there is a number for it.
2. A `w_H` quoted without its weighting is meaningless. The Record carries both.

## 2. Cost: the estimator is nearly free, the exact loss is 2.6x

Seconds per epoch on the same four frames (2 epochs each, CPU):

| setting | probes | s per epoch | x energy-forces |
|---|---|---|---|
| energy + forces only | 0 | 10.7 | 1.00 |
| rademacher k=2 | 2 | 11.6 | **1.08** |
| rademacher k=4 | 4 | 13.1 | **1.23** |
| rademacher k=8 | 8 | 17.1 | 1.60 |
| modes (exact, k = n_vib = 24) | 24 | 27.4 | 2.57 |
| cartesian (exact, k = 3N = 30) | 30 | 27.8 | 2.60 |

The design's arithmetic is `(2 + 2k)` forward-equivalents against `(2 + 6N)` for the full
matrix, i.e. 24 probes should cost ~10x the 4-probe run. Measured: 2.1x. Two reasons, both
worth knowing before sizing the campaign:

- only the Hessian-labelled frames pay for probes, and the E/F work of every frame is
  paid anyway;
- one backward pass per probe serves the WHOLE batch (the batch is block-diagonal, T03
  section 8), so the cost per probe is amortised over the batch, not per structure.

So on a set where most frames are displaced (E-F only, round-5 Q7 (b)), **the exact loss
is affordable** and `k = 4` costs almost nothing. The campaign's 100,000-frame epochs will
have a different constant, but the ratios are structural.

## 3. The scan: what the weighting and the weight did

30 epochs each, judged by `06_judge` on the fit set's own frames (interpolation!). The
base model on these frames: **low-mode MAE 4.14 cm^-1**.

| run | weighting | w_H | w_H / its own balance | final train loss | final valid loss | low-mode MAE | |
|---|---|---|---|---|---|---|---|
| `fit_x0.3` | entropy | 341.9 | 0.3 | 0.078 | 0.108 | **1.18** | best |
| `fit_x1` | entropy | 1139.6 | 1.0 | 0.100 | 0.128 | **1.78** | |
| `fit_x3` | entropy | 3418.8 | 3.0 | 0.813 | 0.967 | 2.58 | |
| `fit_modes` | entropy | 1139.6 | 1.0 | 0.084 | 0.125 | 4.02 | exact probe |
| `fit_none_fair` | flat | 1.748 | 1.0 | 0.691 | 0.340 | **5.39** | worse than the base |
| `fit_none` | flat | 1139.6 | 652 | 134.7 | 54.7 | 8.27 | mis-weighted by construction |

Readings, in decreasing order of how much I would bet on them:

1. **Entropy weighting is what moves the low modes.** At each weighting's OWN balance,
   entropy goes 4.14 -> 1.78 and flat goes 4.14 -> **5.39, worse than doing nothing**.
   That is eq. 3 doing its job: with flat weights the loss spends its capacity on the
   stiff modes, which carry the norm and none of the entropy.
   (`fit_none` at `w_H = 1139.6` is not evidence of anything -- it is the entropy balance
   applied to a flat loss, 652x too large, and it is listed only so the mistake is on the
   record rather than hidden.)
2. **More `w_H` is not better**: 0.3x beat 1x beat 3x. The epoch-0 balance is a scale, not
   an optimum; the campaign should scan BELOW it, not above.
3. **The exact probe did not win** (4.02 against 1.78 for `k = 4` at the same weight). On
   four frames I would not call this a property of the method -- the stochastic probe may
   be acting as a regulariser, or 30 epochs may be too few for either. It is a reason to
   keep `probe = modes` for the validation number (where it is exact and free of this
   question) and to let the campaign's own scan decide the training probe.

## 4. The replay arithmetic (measured as arithmetic, not as a run)

mace's `--multiheads_finetuning` concatenates the two heads' training sets and shuffles,
so every frame of either head is seen once per epoch: the ratio is set by dataset size,
not by PFT's step loop (K = 4 upstream steps per Hessian step).

| `--num_samples_pt` | on this fit set (4 train / 5 Hessian) | on the campaign (~99,000 train / ~29,000 Hessian) |
|---|---|---|
| 0 | 0 | 0 |
| 5,000 | 1000 per Hessian frame | **0.17** per Hessian frame |
| 20,000 | 4000 | 0.69 |
| PFT's reference | — | 4 |

**The same flag means 1000 here and 0.17 on the campaign.** No forgetting number could be
measured on this machine (the SPICE release is not here; `s0_spice_test_draw.py` refuses
and the judge's forgetting line reads `-`), so the campaign's replay size is still an open
question, and ticket 16 must scan it with the ratio printed beside each run.

## 5. What I propose for the campaign, and what I do not

Proposed (the user rules):

| setting | proposal | why |
|---|---|---|
| `--hessian_mode_weighting` | **entropy** | the only setting here that improved the low modes; flat made them worse than the base model |
| `--hessian_probe` / `--n-probes` | **rademacher, k = 4** | 1.23x the E/F epoch; k = 2 (1.08x) is the fallback if the campaign's epochs are the bottleneck |
| validation | **`probe = modes` / the full matrix** (already the default) | the logged validation number is then the ruler's quantity, exactly, not a sample |
| `w_H` | **measure the epoch-0 balance on the campaign's own training set, then scan 0.1x / 0.3x / 1x of it** | 1139.6 is THIS set's number and must not be carried over; the direction (scan below the balance) is the transferable part |
| replay | scan 0 / 5,000 / 20,000 with the per-Hessian-frame ratio printed, plus one `--weight_pt_head 4` run at fixed frames | coverage and weight are different knobs (ticket 16) |

Not proposed, because nothing here can support it: an epoch count, any accuracy target for
the campaign, and any statement about generalisation. Those need the 65-frame smoke set
(tianhe) and then the draw.

## 6. What the tianhe run should repeat

```bash
# the fit set from the 65-frame smoke Dataset, then all three stages on one A800
python workflows/hessian_learning/04_dataset.py --tag smoke --name smoke_fit --split-by frame --no-pinned
python scripts/production/s0_hl_smoke_fit.py --tag smoke --name smoke --stage all \
    --epochs 100 --scan 0.1 0.3 1 --device cuda
```

and, where the SPICE release lives:

```bash
python scripts/tooling/s0_spice_test_draw.py --n 5000
python workflows/hessian_learning/06_judge.py --tag smoke --name smoke_fit --engine <run> \
    --spice-file data/training_sets/spice_test_5000.extxyz
```
