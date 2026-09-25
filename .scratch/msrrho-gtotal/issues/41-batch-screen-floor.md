# 41: The package-1 batch screen uses the frequency floor

**What to build:** the batch screen admits what the frequency floor allows — the same rule the branch-A census screen applies since ticket 37. A pooled candidate whose lowest projected mode lies in the inversion window [ithr, 0) becomes a Basin and carries its lowest frequency and window count; only a candidate whose lowest mode lies below ithr is rejected, and it is listed with its below-floor count and lowest frequency. The floor is the configured one (`package2.ithr_cm`, the value the msRRHO layer uses), read from the configuration and not hard-coded. The package-1 texts stop describing a rule production no longer has: the Record fields, the collect step and the batch census report/summary that say "any non-zero imaginary frequency is thrown out" / "must have no imaginary frequency" name the floor and the window instead. The change is dated; package-1 Records written before it stay readable as data, and this ticket records whether old and new records stay comparable for the counts they feed.

The decision shape is ticket 37's pure `census_verdict`; this ticket wires that helper into the batch screen — it does not define a second rule.

**Blocked by:** None (can start immediately) — tickets 35 and 37 are done; this ticket carries their floor into the batch path.

**Status:** done 2026-09-25 — all four boxes paid; ruling Q2's close-out bundle rides in this commit; the verification runs and the two-axis review record are below. Unit suite 65/65; `tests/integration/t_batch_screen_floor.py` and `tests/integration/t_census_second_pass.py` (its new `converged_orca_default` section included) pass on the real MACE engine; `python tests/run_tests.py --all` all 81 tests passed; py_compile clean. **The batch driver itself could not be imported before this ticket** — its three config reads named the pre-split key locations (found and fixed as the smallest change that unblocks box 4; see Notes). **Not verified here:** ruling Q5's production rerun of the five parked Tianhe molecules — it runs after this change is on the Tianhe checkout, submitted by hand from there; nothing was submitted from this workstation.

- [x] `hessian_screen` applies the floor: a window candidate is kept and carries its lowest frequency and window count; a below-floor candidate is rejected and listed with the same counters. The verdict stays engine-free (reuse the pure helper); no second rule is defined. — `hessian_screen` calls `census_verdict` once per candidate and records `verdict` / `lowest_frequency_cm_inv` / `n_below_ithr` / `n_inversion_window` (rejected candidates carry the same counters in the `saddles` list); sections A-C of `t_hessian_screen_floor.py` pin the wired decision and assert the record equals `census_verdict`'s output on the same spectrum; `reject_imaginary` is deleted from both screens (section D; ruling Q1 — ticket 37's review row 7's deferred option landed here).
- [x] The floor comes from the configuration (`package2.ithr_cm`); no literal floor in the batch call path. — `s0_package1_crest_census.py` reads `ITHR_CM = float(P2["ithr_cm"])` and passes it to the screen; the record carries `ithr_cm` (the parquet's molecule level too); `t_msrrho_presets` already holds `package2.ithr_cm == MSRRHO_PRESETS["crest"].ithr_cm`; section B2 pins the screen's own default to the preset.
- [x] The package-1 Record, the collect step and the batch census report/summary use the floor vocabulary; every rejected candidate is listed with its below-floor count and lowest frequency. — `record.py`: the `n_saddles_rejected` note, the saddle warnings (old records keep their imaginary wording and the new fields read as `-`; section E), the log's basin table gains a `window` column, `ithr_cm` is printed; `s0_package1_collect.py`: the verdict names the floor, `BASIN_COLS` carries `n_below_ithr` / `n_inversion_window`, a wholly pre-floor table falls back to `n_imaginary`; `s0_package1_crest_census.py`: the refusal names the floor and lists every condemned candidate with its below-floor count and lowest frequency; `s0_package1_crest_summarise.py`: the saddle line names the floor, admitted window basins are counted; `openqha/store/report.py`'s docstring example.
- [x] A fixture run through the batch screen path is green (no CREST), and the ticket records the date, old-record readability and the comparability statement. — `tests/integration/t_batch_screen_floor.py` (2026-09-25): `analyse_one` on `tests/data/propanal_crest/crest_conformers.xyz` with the real MACE engine, three CREST conformers -> two pooled candidates, both clean basins; the record's floor equals the configured one equals the preset's. Date, readability and comparability: see Notes.

## Rulings of 2026-09-25 (recorded at the ticket-40 close-out; nothing executed yet)

The 2026-09-25 grilling round settled how the imaginary-mode regime (tickets 34-41) closes
out. Recorded here because this ticket is the next work item and carries the close-out
commit:

- **Q1 — approved to proceed; start deferred.** Scope stays the four boxes above.
  `hessian_screen`'s `reject_imaginary` parameter loses its production meaning and is
  deleted with the wiring (ticket 37's review-record row 7's "or delete in 38/40" lands
  here); say so in the review record.
- **Q2 — the close-out bundle (approved).** This ticket's commit also carries:
  - `issues/34-one-imaginary-mode-policy-and-floor.md`, committed for the first time, with
    a done line ("implemented by 35-41");
  - the OPEN disposition rows of tickets 37 (9 rows), 38 (7 rows) and 40 (11 rows) marked
    **accepted, default keep**, dated 2026-09-25 — each row's keep/revert was already the
    implementer's recommendation in its own table, and each table names how to revert a
    row mechanically if the user later rejects one.
- **Q3 — the primary sources (approved).** `research-soft-modes-primary-sources.md` is
  committed to `main` (ADR 0007 cites it; untracked would leave the citation dangling);
  the prototype HTML rides onto a throwaway branch (`scratch/imaginary-mode-prototype`),
  per ticket 34's Further Notes.
- **Q4 — the admitted-and-flagged class gets a real-engine sample (approved).**
  `tests/integration/t_census_second_pass.py` gains a `converged_orca_default` section (a
  `max_opt_steps` cap chosen so the FIRST-pass residual lands in (1e-4, 1.543e-2] eV/A):
  it is the only certification class with no real-calculator end-to-end run. Measured
  cap-to-residual points on the propanal fixture (seed-7 +-0.05 A displacement, real
  MACE): cap 1 -> 8.438, 2 -> 3.577, 4 -> 0.1356 eV/A; probe caps 6-12.
- **Q5 — the five parked Tianhe molecules (approved, after this ticket lands).** Their
  rerun is this ticket's production verification. Submission follows the by-hand pattern
  of `.scratch/orca-slurm/verify-orca-one.sh`: execute from the Tianhe checkout, `sbatch`
  by hand, no ssh from the workstation. Nothing is submitted before this ticket's change
  is on the checkout; the five ids come from the Tianhe branchA logs (the draw300 runs
  that died with "no basin survives the tightening and the imaginary-frequency filter").

## Notes

- **The date.** The rule changed and this wiring landed on 2026-09-25 (the tickets 34-41
  round); the batch screen was the last surface still applying "any non-zero imaginary
  frequency is thrown out".
- **Old-record readability.** Records written before 2026-09-25 carry no `verdict` /
  `n_below_ithr` / `n_inversion_window` / `ithr_cm` fields. Every reader added or touched
  here tolerates that: the log re-render keeps the old saddle wording ("N imaginary
  mode(s)") and shows the new fields as `-` (`t_hessian_screen_floor.py` section E drives
  `record.write_log` on an old-shaped record -- the superseded migration path re-renders
  old records); the batch summary skips absent window counts; the collect step falls back
  to `n_imaginary` for a wholly pre-floor table (a mixed concat reads old rows'
  `n_below_ithr` as 0, which is their true reading -- the old rule admitted only
  zero-imaginary basins). Nothing historical is rewritten.
- **Comparability of the counts they feed.** The new admitted set is a superset of the
  old one: a candidate with a mode below ithr was rejected before and is rejected now
  (with better counters), and a candidate whose lowest mode lies in [ithr, 0) -- thrown
  out by the old screen -- is now a Basin carrying an inversion window. So for molecules
  with no window candidate (every shipped record) the counts are identical under old and
  new (`n_basins`, `n_saddles_rejected`, the Boltzmann weights, the conformational
  correction); across the change the direction is new >= old, and a molecule that gained
  basins is exactly one whose CREST/ETKDG ensemble carried a window mode.
- **The batch driver could not be imported before this ticket** (found while wiring it;
  fixed as the smallest change that unblocks the acceptance criterion, reported per
  AGENTS.md's scope rule). `s0_package1_crest_census.py` read `P1["fmax_census_eV_A"]`,
  `CB["tighten_fmax_eV_A"]` and `CB["order_seed"]` -- the pre-config-split locations.
  The keys have lived at `package1.etkdg_retired.fmax_census_eV_A` = 0.001,
  `package1.tighten_fmax_eV_A` = 1e-4 and `package1.order_seed` = 20260830 (same values)
  since the configuration split, so the module raised `KeyError` at import and no
  fixture could run through it. The three reads now name the keys' actual homes; the
  module computes `-50.0 0.001 0.0001 20260830`.
- **Found, not fixed (beyond this ticket's scope; options for the user).**
  `scripts/production/s0_package1_crest_summarise.py` reads
  `P1["crest_batch"]["order_seed"]` (lines 97 and 199) -- the same pre-split location --
  and takes its products from `mol/*.json`, a layout retired on 2026-08-31 (the products
  are `.log` + `.parquet` today). The script therefore cannot run on current products at
  all. Options: (a) repair the key here and port the script to the parquet tables in its
  own ticket -- my recommendation; the port is the real work and is independent of this
  ticket's vocabulary edit; (b) port it in this commit. Not done: the one-line key
  repair alone leaves the script dead, and the port is a different job.
- **The re-measured cap table** (propanal fixture, seed-7 +-0.05 A displacement, real
  MACE, 2026-09-25): cap 1 -> 8.438, 2 -> 3.577, 4 -> 0.335, 12 -> 0.048, 24 -> 0.030,
  40 -> 0.0144, **48 -> 0.00143**, 50 -> 5.65e-4, 60 -> 6.56e-4, 70 -> 8.85e-5
  (converged; the optimiser stops at 69). Caps 1 and 2 reproduce the ruling session's
  numbers exactly; cap 4 was re-measured at 0.335 (twice; the ruling recorded 0.1356);
  cap 48 was chosen as the geometric middle of (1e-4, 1.543e-2] (~14x below the line,
  ~11x above the target). The test asserts the window, the class, the single pass and
  the merge, so a drift shows as a failure, not as a silent pass.

## The verification runs

### 1. `python tests/run_tests.py` -- the unit group, from the repository root in the `openqha` env

```
all 65 test(s) passed
```

(65 = 64 at handoff + `t_hessian_screen_floor.py`; the new test runs standalone too: 9
checks, sections A-E, `PASS`.)

### 2. `python tests/integration/t_batch_screen_floor.py` -- the fixture run (no CREST, real MACE)

```
A. the floor reached the screen from the configuration
  the record names the floor the screen applied                    ok
  that floor is the crest preset's own value -- one number, two seams ok
B. every pooled candidate carries the verdict and the counters
  2 pooled candidate(s), every record with verdict / n_below_ithr / n_inversion_window ok
  the verdict is exactly the floor rule: saddle iff a mode is below ithr (n_below_ithr > 0) ok
  every saddle is listed with its below-floor count and lowest frequency ok
C. the basin list is exactly the kept subset of the verdicts
  basin_hessian holds one record per kept candidate, in energy order ok
  every kept basin carries its verdict, its lowest mode and its window count; the fixture's spectra are clean minima ok

PASS
```

(The fixture: `tests/data/propanal_crest/crest_conformers.xyz`, three CREST conformers ->
two pooled candidates, both clean -- the wired floor path, end to end, without CREST.)

### 3. `python tests/integration/t_census_second_pass.py` -- the forcing fixture (real MACE)

Sections A-D as in ticket 40, plus the new section for Q4:

```
E. a frame inside ORCA's line is admitted as converged_orca_default, no pass
  one converged frame, one converged_orca_default, no pass, no rejection ok
  the final residual sits inside (fmax, ORCA's line] -- the class's window ok
  the displaced frame consumed the whole cap: its first pass stopped inside the line, not at the target ok
  exactly one optimisation ran: one header, LBFGS 0..48 once       ok
  the trajectory holds the one pass (initial state + 48 steps)     ok
E2. the certified pair deduplicates; the survivor keeps its own class
  two certified frames merge into one basin                        ok
  the surviving basin's class and tighten_converged answer to the two lines ok

PASS
```

### 4. `python tests/run_tests.py --all`

```
all 81 test(s) passed
```

## Review record (2026-09-25): the two-axis review -- findings and dispositions

The two-axis review (Standards / Spec; two independent sub-agents over the working-tree
change set, the same shape as tickets 37/38/40's records) raised the findings below.
Each is dispositioned, none left open:

| # | finding | where | axis | disposition |
|---|---------|-------|------|-------------|
| 1 | the summarise script reads `P1["crest_batch"]["order_seed"]` (a live `KeyError`) and `mol/*.json` (a layout retired 2026-08-31) | `s0_package1_crest_summarise.py` | Standards | **reported, not fixed** -- beyond the four boxes; the key repair alone leaves the script dead (Notes, options given) |
| 2 | a mixed old/new concat fills old rows' `n_below_ithr` with 0 | `s0_package1_collect.py` | Standards | **kept, comment added** -- 0 is the old rows' true reading (the old rule admitted only zero-imaginary basins); the `elif` still covers a wholly pre-floor table |
| 3 | the legacy-record shim appears in two files | `record.py`, `s0_package1_collect.py` | Standards | **kept** -- the two sites read different shapes (a dict vs a pandas column); the shared rule is `census_verdict`'s and does not fit either shim |
| 4 | the four floor fields travel as a clump in two producers | `crest_census.py` | Standards | **kept** -- both producers already share the rule through `census_verdict`; each builds its own record dict (one adds the certification fields, the other an index) |
| 5 | `ITH_CM` drops the `r` every sibling name carries | `s0_package1_crest_census.py` | Standards | **fixed** -- renamed `ITHR_CM` |
| 6 | the config-key repair (`FMAX_COARSE` / `FMAX_TIGHT` / `ORDER_SEED`) is outside the four boxes | `s0_package1_crest_census.py` | Standards + Spec | **kept, recorded** -- the module could not be imported, so box 4 could not be paid at all; values unchanged, the keys' actual homes named (Notes) |
| 7 | the test docstring's cap-4 residual (0.335) disagrees with the ruling's (0.1356) | `t_census_second_pass.py` | Standards + Spec | **kept** -- re-measured twice today; caps 1/2 reproduce the ruling exactly; the docstring states the measured table and the test asserts the window, not one number (Notes) |
| 8 | section E repeats the A-D `mkdtemp`/`try`/`finally` scaffold | `t_census_second_pass.py` | Standards | **kept** -- the file's own style; a loop would rewrite A-D for one new section |
| 9 | section E2's dedup/survivor assertions go past Q4's stated purpose | `t_census_second_pass.py` | Spec | **kept** -- the class's end-to-end meaning is that the census proceeds (admit -> dedup -> survivor); counts alone would not show the frame is neither re-run nor re-ejected |
| 10 | the machine table carried the counters without the floor value | `record.py` (`record_to_rows`) | Spec | **fixed** -- `ithr_cm` at the parquet molecule level, with a comment; box 3 asks the Record to name the floor |
| 11 | the old-record fallbacks had no check | `record.py`, `s0_package1_collect.py` | Spec | **fixed for the reachable path** -- section E of `t_hessian_screen_floor.py` drives `write_log` + `record_to_rows` on an old-shaped record; collect's fallback is a guard inside a script's `main` and stays untested (documented in its comment) |
| 12 | the summary's new window statistic is more than "name the floor and the window" | `s0_package1_crest_summarise.py` | Spec | **kept** -- the box asks the summary to name the window; a window that is admitted but never counted is not named, and the count is the naming |
| 13 | the old-record fallback re-rendered old saddles as "N mode(s) below ithr" | `record.py` (`write_log`) | Spec | **fixed** -- old records keep the imaginary wording (their lowest mode need not lie below ithr); new records use the floor wording; section E pins both |

Totals: Standards 8 findings (3 fixed, 1 reported, 4 kept as judgement calls); Spec 7
findings (3 fixed, 1 kept-and-recorded, 3 kept as judgement calls). Worst Standards-axis
finding: #1, a pre-existing dead script just outside this change set (reported, not
fixed). Worst Spec-axis finding: #10/#11, the floor's provenance and the old-record path
(both now fixed).

**Ruling Q1, stated here as the ruling asks.** `reject_imaginary` lost its production
meaning and was **deleted from both screens** with this wiring: `hessian_screen` (whose
parameter the ruling names) and `census_from_frames` (whose parameter ticket 37's
review-record row 7 had deferred, "keep -- or delete in 38/40").
`t_hessian_screen_floor.py` section D pins both deletions; ticket 37's row 7 now records
that its deferred option was taken here.
