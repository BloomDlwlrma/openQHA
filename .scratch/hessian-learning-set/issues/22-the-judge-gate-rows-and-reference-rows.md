# 22: The judge's gate rows and reference rows -- thermochemistry gates; RMS-displacement bins and the MD temperature ramp report (`training/judge.py`, `06_judge.py`)

**What to build:** the judge table re-ordered for S0-C-54. **Gate rows** (PASS / FAIL with a threshold each): the thermochemistry rows (msRRHO entropy at the engine's own minima against the reference, per pinned molecule, read from the msRRHO Records), the in_distribution row, the forgetting line. **Reference rows** (reported, never gated, marked `reference` in the table): the H, E and F metrics of the held-out generator's labelled frames binned by `rms_displacement_A` (0 = basin / < 0.08 / < 0.15 / >= 0.15 A), per distribution; the frequency rows as today; and an **MD temperature-ramp** line -- Rodriguez's protocol on branch B's stability calibration: from each pinned molecule's optimised geometry, Langevin, 5 K start, +5 K every 5 ps, until the failure criterion (any atom pair's 50-step mean distance > 1.5x or < 0.75x its equilibrium value), reporting the failure temperature and time for the fine-tuned model and the base. The verdict reads gate rows only; the report prints both kinds with the three validation curves of the run beside them.

**Blocked by:** 20 (held-out generator frames in test). **Unblocks:** 16, 23.

**Status:** ready-for-agent

- [ ] `judge.aggregate` groups rows by (distribution, rms_bin) with bins `0 / <0.08 / <0.15 / >=0.15`; every H/E/F metric per bin; the smoke Dataset's 52 displaced frames populate the bins on the base model
- [ ] `judge.verdict`: gate rows = thermochemistry (`MODEL_ERROR_S_REF` per pinned molecule vs threshold), in_distribution no-degradation, forgetting <= 1.15 x base; reference rows carry `gate = no`; `VERDICT` unchanged by any reference row (unit: worsen a reference row, the verdict does not change)
- [ ] `judge.md_ramp(calc, base_calc, molecules, ...)` on `s0_B_md_stability.run_one`'s machinery: the ramp schedule, the failure criterion, `FAIL_T_K` and `FAIL_PS` per molecule for engine and base; the report row "MD ramp (reference)"; the cost is bounded by `--ramp-max-K` (default 600) and `--ramp-molecules` (default the seven)
- [ ] the run's three validation curves (ticket 21's Record) printed in the judge report's header when the engine is a fine-tune with a Record
- [ ] `06_judge.py --no-ramp` to skip the MD line; the Record's `[Reference]` section names the bins, the ramp settings and the seeds
- [ ] calibrations kept: base vs base passes every gate row; the 0.9x-scaled calculator fails the low-mode reference row AND is reported as such without failing the verdict -- the ticket states this change of meaning explicitly in CONTEXT's Judge entry (low-mode line is reference now)
- [ ] unit (`t_judge.py`): binning, gate/reference split, ramp schedule on a fake calculator that "fails" at a chosen step; integration (`t_judge_engine.py`, CPU): the smoke Dataset on the base model with `--ramp-max-K 20` for one molecule
- [ ] `.mem/notes` round note
