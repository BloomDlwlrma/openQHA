# 38: The census certifies the tighten; a double failure is rejected

**What to build:** every candidate's tighten is certified against two lines: the repository target (`fmax = 1e-4 eV/Å`) and ORCA's default max-gradient (`TolMaxG = 3e-4 Eh/bohr = 1.543e-2 eV/Å`). Outcomes: `converged` (at or below the target); `converged_orca_default` (above the target, at or below the ORCA line — admitted, and marked `tighten_converged = false` because it missed our own target); above the ORCA line — one bounded second optimisation pass runs, and if the residual still exceeds the line the candidate is **rejected** as `not_certified`, listed with its residual in a class of its own, separate from the frequency-floor saddles. Per the ruling of 2026-09-25 the old "admitted with a flag" behaviour is superseded: spec 34's line on `admitted_flagged` is overridden by this ticket. The refusal message lists the two rejection reasons in separate sections; the census summary counts every class. The classification is a pure helper with unit tests; the second pass is orchestrated inside the census tighten step and runs at most once.

The decision shape came from the prototype (`prototype_imaginary_mode_state_machine.html`, `convergenceClass`), amended by the ruling: the last class ejects after the single second pass.

**Blocked by:** 37 (The census screen uses the frequency floor and records the inversion window) — same function, same record

**Status:** done 2026-09-25 (unit suite 63/63 with the new `tests/unit/t_census_convergence.py`; `tests/integration/t_mace_engine_folder.py` pass with the new section E; py_compile + `s0_A_pipeline.py --help` clean; the two-axis review's findings are in the review record below). The second pass's engine-file path is wired (`Trajectory(..., mode="a")` on the conformer's own `opt.traj`, the same `opt.log`) and pinned by unit section F; no shipped fixture triggers it — their residuals sit at or below `fmax` — so that path has no end-to-end run until a real frame exceeds ORCA's line. **Closed 2026-09-25 by ticket 40:** `tests/integration/t_census_second_pass.py` forces the pass on the propanal fixture (a displaced frame with `max_opt_steps = 1`, first residual ~8 eV/A) and reads what the two passes left — both runs in `opt.log`, the first pass's frames still in `opt.traj`, `conf.extxyz` at the final geometry, `mace/confNN/` holding exactly the three engine files (13 checks, all pass; the run is recorded in ticket 40).

- [x] All four outcomes are unit-tested (`t_census_convergence.py` A/A2 on the pure helper: `converged` / `converged_orca_default` / `converged_second_pass` / `not_certified`; B/B3/B4 on the orchestrated path); `converged_orca_default` basins are admitted and carry `tighten_converged = false` (B3, on the census hessian record; the wired path in integration E).
- [x] A candidate above the ORCA line receives exactly one second pass (B: four optimiser calls for three frames — a third call would overflow the script); a candidate still above it is rejected as `not_certified` with its residual (C, the final post-pass residual); a molecule all of whose candidates are rejected refuses (`t_census_convergence.py` D), and the message lists both rejection sections (`t_branch_a_crash` C).
- [x] The census record carries `convergence_class` (every census hessian record, B3), the not-certified count and the rejected residuals (`n_not_certified` + the `not_certified` list of `{conformer_id, residual_eV_A}`, C); the four counts are printed in `branchA.out`'s census summary.
- [x] The measured baseline, recorded here: every observed record and message shows residuals at or below 1e-4 eV/Å (local records 5.5-9.4e-5; Tianhe `dsgdb9nsd_052993` 8.13e-5; the MACE integration fixture at the target). The new classes were therefore empty in all evidence — the new fields measure them going forward.

Notes:

- The second pass reuses `conformers.optimise` with the census's own bounds (target `fmax`, `max_opt_steps`), appends to the same engine files, and folds its steps into `opt_steps_per_frame` (both passes together; the docstring says so).
- `not_certified` candidates are rejected **before** the deduplication, so each rejected candidate carries exactly one reason (certification vs frequency floor), and a frame not at a stationary point cannot absorb a certified duplicate of the same basin. A reader tracing those frames finds them in `not_certified`, not in `duplicate_map`.
- The record shape follows the user's ruling of 2026-09-25: the census hessian record carries `convergence_class` + `tighten_converged` (final residual at or below `fmax`); the top level carries the four class counts and the `not_certified` list; no per-frame arrays, no ORCA-line value at the record top, `branchA.toml` untouched.
- The refusal prints each rejection section only when its list is non-empty (message test C holds the two-section form; D holds a refusal where nothing was floor-screened).

## Review record (2026-09-25): additions beyond this ticket's text — dispositions OPEN

The two-axis review (Standards / Spec) of this change set raised the following judgement
calls. None blocks the acceptance criteria; the keep/revert decision is the user's and is
**open** (ticket 37's record is the pattern). This table is the implementer's
recommendation, not a ruling.

| # | addition / finding | where | basis | recommendation |
|---|--------------------|-------|-------|----------------|
| 1 | `convergence_class` + `tighten_converged` per basin | census hessian record | user ruling 2026-09-25; ticket 34's census bullet names them ("each basin's ... when needed") | keep |
| 2 | per-class counts `n_converged` / `n_converged_orca_default` / `n_converged_second_pass` / `n_not_certified` + the `not_certified` list | census record top level | "the census summary counts every class"; "the rejected residuals" | keep |
| 3 | counts printed in `branchA.out`; integration section E; message test C extended | `s0_A_pipeline.py`, tests | reporting a counted fact; coverage of the new sections | keep |
| 4 | `opt_steps_per_frame` folds the second pass's steps | `crest_census.py` (docstring says so) | reviewer finding; the alternative (first pass only) hides cost | keep |
| 5 | no separate record field for the second pass's steps or its residual beyond the class/bool | `crest_census.py` | lean record per the user's ruling | keep |
| 6 | `_locate_reference`'s note also names the certification as a reason the reference geometry is absent | `s0_A_pipeline.py` | the new rejection made the old note incomplete | keep |
| 7 | `branchA.toml` (`[[Basin]]` / `Census`) carries none of the new fields | `branch_a_property.py` (untouched) | the user's ruling (only the certification judgment is recorded; nothing else); no later step reads them yet; criterion 3 is met by the census record in `branchA.out` | keep — add schema rows only when a reader exists |
