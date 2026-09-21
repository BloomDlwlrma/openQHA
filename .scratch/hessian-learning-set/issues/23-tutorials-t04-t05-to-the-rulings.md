# 23: T04 and T05 rewritten to the rulings S0-C-54..57 (`docs/tutorials/T04_*.ipynb`, `T05_*.ipynb`)

**What to build:** the two theory notebooks brought to the state the code will be in after tickets 20-22, executed end to end with zero errors. Both were rewritten on 2026-09-21 for S0-C-53 (the Cartesian target) but still say "every frame carries H" (PHL's rule), name `--num_samples_pt` / `--weight_pt_head` as knobs, describe validation as the full matrix, and present the RMS bins as gate-like. **T05**: section 4 becomes "basin frames only, the displaced frames a held-out generator" with Rodriguez's stationary-only evidence (tables 2-4) and the measured rho_k as the reference reading; section 6 describes the Replay by its file, `config_weight`, threshold 0 and commit C, with the replay arithmetic R0-R4 and the coverage-by-frame estimate; section 7 splits gate rows from reference rows; section 8's recipe uses the flags of tickets 18-21 (`--train-generators basin`, `--mode-weighting cartesian`, the draw tool, `--swa`, the control flags) and the measurement job first; section 9's comparison table and A1-A8 follow. **T04**: section 5's framing of rho_k as "what the every-frame rule buys" becomes "what the held-out generator reads"; section 7's objective states basin frames + unlabelled Replay frames; Algorithm 4 (evaluate) becomes the four-fixed-probe estimator entering the control signal, with the full matrix as the judge's tool only; Algorithm 5 lists gate and reference rows; section 9's S-table and code map name the functions as they exist after ticket 21.

**Blocked by:** 21 (function names and flags), 22 (the judge rows). **Unblocks:** nothing; documentation.

**Status:** ready-for-agent

- [ ] T05 sections 4, 6, 7, 8, 9 rewritten as above; the cell of section 6 asserts against the fork that commit C is present (the check that read "NO" flips to "YES") and prints the R0-R4 table with coverage (distinct molecules) per row
- [ ] T04 sections 5, 7, 8 (Algorithms 4 and 5), 9 rewritten as above; the cell of section 7 runs the toy with the `B = I` path from `phl.make_probes` instead of the notebook's hand-written `cartesian_probes`
- [ ] both notebooks executed in the `openqha` env with zero errors; every commentary line matches the numbers its cell prints (the rule of 2026-09-21: no inherited numbers)
- [ ] the header of each states the rulings it rests on and which earlier version it replaces; T05's section 9 table has a "since 2026-09-21 (S0-C-54..57)" column
- [ ] `docs/tutorials/README` (or the index that lists T01-T05) updated; `.mem/notes` round note
