# 19: Collect, ensemble and 02d Calculations

**What to build:** the analysis leaves `_records/md_<route>/collect.out`, a small `collect.toml` (`[Calculation_Status]`, `[Calculation_Info]`, `[Criteria]` with N_PASSED, N_TOTAL, ALL_PASSED) and the four `.dat` tables, all with the setting in the stem when it is not the default (`collect_s2.out`, `collect_s2.trajectories.dat`). The ensemble report leaves `ensemble.out` and `ensemble.toml` (`[Result]`: F_conf, populations, delta G, effective basins, crossings; `[[Basin]]`: INDEX, TS, N_FRAMES, crossings) beside them, and 02d leaves `02d_frequency_identity.out` and `.toml` with its four stages as blocks. The ensemble reads `[[Basin]]` from `branchA.toml` and TS from `collect.trajectories.dat`; the collect Batch's "done" reads STATUS from `collect.toml`.

**Blocked by:** 17, 18.

**Status:** ready-for-agent

- [ ] after the end-to-end integration tests, `_records/md_<route>/` holds exactly `collect.out`, `collect.toml`, the four `.dat`, `ensemble.out`, `ensemble.toml` (plus the 02d pair when 02d ran) and the `basinNN/` folders; a setting run adds the `_s2` stems in the same folder
- [ ] every `.toml` starts with `[Calculation_Status]` and every `.out` ends with the terminal line
- [ ] the ensemble report's numbers come from `branchA.toml` `[[Basin]]` and the `.dat` (unit test extended)
- [ ] no `_records/md_<route>/<setting>/` path is written or read anywhere
