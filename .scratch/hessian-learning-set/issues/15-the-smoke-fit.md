# 15: The smoke fit: cost, ceiling, `w_H`, must-pass / must-fail (`04_dataset.py --no-pinned`, `05_train`, `06_judge` on the seven)

**What to build:** the measurement that decides the campaign's settings before the A800 job is submitted, on the seven molecules' 65 labelled frames (33 Hessian frames). (1) `04_dataset.py --no-pinned --split-by frame --name smoke_fit`: the pinned rule off, so the seven's frames split 90 / 5 / 5 by frame -- a FIT set, its Record marked `PURPOSE = "fit"` and `06_judge` printing "interpolation only" on every line from it. (2) Cost: one epoch each on the local CPU (and on one A800 if available) with `probe = modes` (exact, `k = n_vib`), `cartesian` (`k = 3N`), `rademacher k = 2 / 4 / 8`, and `--loss energy_forces` -- wall per epoch, peak memory, the ratio to the E/F epoch (the design's `(2 + 2k)` vs `(2 + 6N)` claim measured). (3) The epoch-0 balance: `w_F L_F` and `L^_H` on the base model -> `w_H` such that `w_H L^_H = w_F L_F` (the value the campaign starts from), then three runs at `w_H x0.3 / x1 / x3`, 100 epochs, `rademacher k = 4`, `entropy` weighting, plus one run `probe = modes` at `x1` (the ceiling: what the exact loss reaches on the seven) and one `mode_weighting none` at `x1` (what the weights buy on the low modes). (4) The judge on each: `06_judge.py --engine <run>` -- the seven's low-mode MAE at x_r and at own minima, `MODEL_ERROR_S_REF`, the interpolation-only caveat; `--engine base` must-pass and `--scale 0.9` must-fail as the calibration of the table; the forgetting line on `spice_test_5000` for every run (no replay in the smoke fit: this measures how fast the seven alone forget SPICE, the number the campaign's `--num-samples-pt` answers). Written as `.scratch/hessian-learning-set/smoke-fit-report.md` with the tables, and every run's Record on disk under `_datasets/smoke_fit/train/<run>/` and `judge/<run>/`. Decision: `w_H`, `k`, whether `entropy` weighting stays on, and the epoch count for ticket 16 -- proposed in the report, ruled by the user.

**Blocked by:** 13, 14; the local ORCA labels of the seven (done: 65 frames, ticket 03) or the tianhe smoke labels.

**Status:** code and machinery done 2026-09-21; the measurement ran on 5 frames of one molecule here, and the 65-frame programme is the tianhe run.

**Delivers:** the numbers the campaign is configured from; the first fine-tuned MACE-OFF23 the judge has seen.

- [~] the fit Dataset: `04_dataset.py --no-pinned` (+ `dataset.build(purpose=)`, `PURPOSE` in the Record) and `smoke_fit.build_fit_dataset` (re-splits an existing Dataset's labelled frames by frame, for a machine whose molecule tree predates ticket 09). **Built here with 5 frames of ONE molecule, not 65 of seven**: the local ORCA run of ticket 03 labelled one molecule and the 65-frame set is on tianhe. `06_judge` prints the caveat via `PURPOSE = fit`
- [x] the cost table (2 epochs each, CPU, 4 train frames): E/F 10.7 s/epoch = 1.00; rademacher k=2 1.08x, k=4 1.23x, k=8 1.60x; modes (24 probes) 2.57x; cartesian (30) 2.60x. **The exact loss is 2.6x, not the ~10x the (2+2k) arithmetic suggests** -- only Hessian-labelled frames pay for probes, and one backward pass per probe serves the whole batch. The `probe = modes` train loss equals `projected_loss_full` on the loop's own code path (`t_phl_loss.py`, 1e-12); mace's log has no per-term number, so it is not re-checked from a run
- [x] the balance and the scan: **entropy w_H = 1139.6, flat w_H = 1.748 -- 652x apart**, so a weight is never carried between weightings and PHL's Cartesian 0.25-0.30 band transfers to neither. Six 30-epoch runs judged: base 4.14 cm^-1 low-mode MAE; entropy x0.3 -> 1.18, x1 -> 1.78, x3 -> 2.58, modes -> 4.02; **flat at its own balance -> 5.39, worse than the base model**. Two readings: entropy weighting is what moves the low modes, and more w_H is not better (scan BELOW the balance). 30 epochs, not 100, and 4 training frames: the ordering is not claimed to survive on 65
- [x] the replay arithmetic: `smoke_fit.replay_ratio` and a `[Replay]` block in every fit Record -- 5,000 frames is 1000 per Hessian frame HERE and 0.17 on the campaign, against PFT's 4. **The two-knob pair (coverage against `--weight_pt_head`) was NOT run**: no SPICE release on this machine, so no forgetting line could judge either; it moves to ticket 16 with the ratio printed per run
- [~] the judge ran on every scan run (`judge/<run>/`, low-mode MAE column above) and its two calibrations are ticket 14's (`t_judge_engine.py`: base must-pass, 0.9x must-fail). **The seven's own-minimum table and the -13 / -57 / +66 rings are not here**: they need the seven molecules and their msRRHO Records, i.e. the tianhe run. No forgetting ratio per run (no SPICE frames)
- [x] `smoke-fit-report.md`: the numbers above, what they do and do not support, the proposed campaign settings (entropy weighting; rademacher k = 4; exact validation; measure the balance on the campaign's own set and scan 0.1 / 0.3 / 1 x BELOW it) and the tianhe commands; note in `.mem/notes/`


**Closing (2026-09-21):** `openqha/training/smoke_fit.py` (`build_fit_dataset`, `epoch_zero_balance`, `replay_ratio`, `cost_table`), `dataset.build(purpose=)` + `PURPOSE` in the Record + `04_dataset.py --no-pinned`, `scripts/production/s0_hl_smoke_fit.py` (`--stage balance|cost|scan|all`, the Record `smoke_fit.{out,toml}` with `[Balance] [Cost] [Scan] [Replay]`), `tests/unit/t_smoke_fit.py` (13 checks); unit group 54/54. The report is `smoke-fit-report.md`; the runs are under `$S0_RUNS_ROOT/rings/_datasets/smoke_fit/`.

**The first fine-tuned MACE-OFF23 in this project**: six 30-epoch runs, all of which lowered the low-mode error of the molecule they were trained on (best 4.14 -> 1.18 cm^-1), and one (flat weighting at its own balance) which RAISED it to 5.39 -- the first direct evidence that eq. 3's weighting is load-bearing and not decoration.

**A mistake of mine, on the record:** the first `mode_weighting = none` run used the entropy-balanced `w_H = 1139.6`, i.e. 652x its own balance, and produced a loss of 134 and a low-mode MAE of 8.27. That number compares nothing; the fair point (`w_H = 1.748`) was run afterwards and is the 5.39 above. Both are in the report, because a scan that silently drops its bad point is not a scan.

**What the tianhe run must still do** (the ticket's own words, unmet here): the seven molecules' 65 frames, 100 epochs, the own-minimum thermochemistry table, the forgetting ratio per run, and the two-knob replay pair.

## The replay arithmetic, and the three knobs (added 2026-09-21)

mace 0.3.16's `--multiheads_finetuning` is **not** PFT's algorithm 1. PFT alternates by
STEP (K = 4 upstream E/F steps per Hessian step); mace concatenates the two heads'
training sets into one `ConcatDataset` and shuffles it (`cli/run_train.py`), so **every
replay frame and every fine-tuning frame appears exactly once per epoch** and the ratio
is set by DATASET SIZE, not by a step loop. A batch may hold both kinds; the replay
frames carry no `REF_hessian`, so ticket 12's masking already keeps them out of the
Hessian term.

What the default setting actually buys:

| setting | train frames | with Hessian | `--num_samples_pt` | replay : fine-tune | SPICE frames per Hessian frame |
|---|---|---|---|---|---|
| smoke set | 65 | 33 | 5,000 (all of them) | 77 : 1 | 152 |
| draw300 (estimated: 6,458 molecules x ~17 frames, ~5 with Hessians, 90 % train) | ~99,000 | ~29,000 | 5,000 | 1 : 20 | **0.17** |
| draw300 | ~99,000 | ~29,000 | 20,000 | 1 : 5 | 0.69 |
| PFT's 4 : 1, for comparison | ~99,000 | ~29,000 | ~400,000 | 4 : 1 | 4 |

So on the campaign the default replay is ~23x LESS than PFT's, per Hessian frame; on the
smoke set it is 150x MORE (the fit set is tiny). A number quoted without this ratio says
nothing, so every run's Record and every table row carries it.

Three knobs, and they are not interchangeable:

1. `--num_samples_pt` -- **coverage**: more of SPICE's chemical space is seen per epoch.
   Costs epoch time linearly.
2. `--weight_pt_head` (mace default 1.0) -- **weight**: the same frames' `config_weight`,
   so the replay's share of the gradient rises with no extra frames and no extra time.
   It does NOT widen coverage: it shouts the same frames louder.
3. epochs -- every frame of both heads is seen once per epoch, so 100 epochs means each
   replay frame is learned 100 times. The mixing ratio decides the relative gradient
   share within an epoch; the epoch count decides how often either is revisited.

Measure 1 and 2 separately. Scanning only `--num_samples_pt` cannot tell coverage from
weight, and the forgetting line alone cannot say which one bought the result.
