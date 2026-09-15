# 17: Branch A Calculation: branchA.toml replaces basins.toml

**What to build:** the branch A pipeline leaves `_records/branchA.out` and `_records/branchA.toml`. The Property file holds `[Calculation_Status]`, `[Calculation_Info]` (molecule, tag, engine, the CREST and census settings), `[CREST_Run]`, `[Census]`, `[[Basin]]` (INDEX, ENERGY, RELATIVE, SIGMA, G0, N_IMAGINARY, LOWEST_FREQ, G_MINUS_EEL, A_MINUS_EEL) and `[Criteria]` (N_PASSED, N_TOTAL, ALL_PASSED). Provenance (engine, weights, versions, machine, patch probe), the criterion detail lines, the sigma tolerance sweep, the thermochemistry breakdown and the file lists are printed in the Report only. Every reader of the branch A record (ensemble report, 02a debug, 02c benchmark, 02d, the branch A parsl driver's "done") reads the blocks; "done" means STATUS is NORMAL TERMINATION.

**Blocked by:** 16.

**Status:** done 2026-09-15

- [x] after the branch A integration tests, `_records/` holds exactly `branchA.out` and `branchA.toml`; the `.toml` starts with `[Calculation_Status]`, STATUS is NORMAL TERMINATION, and the `.out` ends with the terminal line
- [x] `branchA.toml` has no key outside the schema; the sigma sweep, criterion details and thermochemistry breakdown are found in `branchA.out`
- [x] the basins reader hands basins, sigma, g0 and G - E_el to the ensemble report from `[[Basin]]` (unit test)
- [x] the branch A parsl driver's completion check reads STATUS (unit test)
- [x] no `basins.toml` is written or read anywhere in the repository
