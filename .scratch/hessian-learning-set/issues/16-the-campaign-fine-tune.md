# 16: The campaign fine-tune on `draw300` -- the measurement job, the replay rows R0-R4, the judge table, the registered engine (`hl_train.slurm`, `05_train`, `06_judge`)

**Rewritten 2026-09-21** under S0-C-53..57 (the earlier text -- entropy weighting, every frame with H, `--num_samples_pt`, the 0/5,000/20,000 rows by `--weight_pt_head` -- is superseded; its replay arithmetic lives in grilling round 7 and `spec-fine-tune-basins.md`).

**What to build:** the production fine-tune of MACE-OFF23_medium on the campaign Dataset's **basin frames** (ticket 08's labels assembled; `04_dataset --train-generators basin`, the displaced / merged / saddle frames a held-out generator in test), judged, registered. In this order:

1. **The measurement job** (Q7, S0-C-57): one A800, the 65-frame smoke Dataset rebuilt basin-only, `cartesian` target, k = 4, `w_H` from the epoch-0 balance on that split, a 5,000-frame Replay (ticket 19), a fixed epoch count (20). Its Record carries seconds per epoch, the derived `u` (seconds per E/F frame-pass), `g` (this CPU's u over the A800's), the validation share, and the peak memory. The cost table of round 10 is then rewritten in measured hours per row.
2. **The epoch-0 balance on draw300's train split** (`smoke_fit.epoch_zero_balance`, Cartesian), `w_H` = that value; the scan 0.1x / 0.3x / 1x runs with R1's replay.
3. **The replay rows** at the chosen `w_H`, everything else identical, nested prefixes of one seeded draw: R0 = 0; R1 = 5,000; R2 = 17,132 (~1 per Hessian frame); R3 = 68,528 (~4, PFT's K = 4); R4 = R3's frames with `config_weight = 10`. Each row's Record prints `REPLAY_PER_HESSIAN_FRAME` and the forgetting line; if `g` makes R3/R4 exceed the allocation, R0, R1 and a `config_weight = 10` row on R1's frames run first and the rest are queued.
4. **The judge** (ticket 22) on every row: gate rows (thermochemistry at own minima for the seven, in_distribution, forgetting) and reference rows (RMS bins over the held-out generator's labelled frames, the MD ramp on the seven, the frequency rows); the three validation curves per row in the report.
5. **The choice and the registration**: the row that passes every gate row with the smallest replay, registered in `ENGINES` (`--register`, ticket 01) with the Dataset index path, the config SHA and `MACE_FORK_COMMIT` as `source`, `params_sha256` pinned; `docs/` gains the campaign page (what was trained on what, the settings, the measured cost from the array logs and the A800 job, the judge table, the paper's sentence per class).

**Blocked by:** 08 (the labelled campaign Dataset), 18, 19, 20, 21, 22.

**Status:** ready-for-agent (the measurement job can start as soon as 18-21 land and the smoke labels are on tianhe)

- [ ] the campaign Dataset: `[[Class]]` table with basin-frame counts per class in train / valid / test and the held-out generator counts; `index.dat` complete; `s0_hl_progress.py` all labelled
- [ ] the measurement job's Record: `SECONDS_PER_EPOCH`, `U_SECONDS`, `G_FACTOR`, `VALID_SHARE`, `PEAK_MEMORY_GB`; the cost table of round 10 rewritten in measured hours and committed to `grilling-round-10-replay-cost.md`
- [ ] the balance Record on draw300 (Cartesian `L_H`, `w_H`), the three `w_H` rows judged, `w_H` chosen and stated with its reason
- [ ] R0-R4 Records with `PT_N_FRAMES`, `PT_CONFIG_WEIGHT`, `REPLAY_PER_HESSIAN_FRAME`, the forgetting ratio, `MACE_FORK_COMMIT` = C, both weight sets (Stage One / Two), the three validation curves; the Hessian curve moves in every row
- [ ] the judge table per row: gate rows PASS / FAIL with thresholds; reference rows reported; the R4-vs-R3 reading written down (coverage or pull)
- [ ] the chosen engine registered; `MACE_FORK_COMMIT`, config SHA and Dataset index in `source`; the msRRHO pipeline runs one pinned molecule with it and its Record names the engine
- [ ] `docs/hessian_learning_campaign.md` (the other session's page) extended with the fine-tune section; `.mem/notes` round note; ticket 08's measured-cost column closed with the array logs
