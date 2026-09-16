# 03: The ensemble reads the Table and the Property file

**What to build:** the ensemble takes each basin's T*S from the `trajectories` section of `collect.dat` and collect's verdict (passed, total) from `[Criteria]` in `collect.toml`; it opens no `.criteria.dat`. A molecule with no `collect.dat` is refused with the path and the collect command to run, as the missing trajectories table is refused today.

**Blocked by:** 02 (Collect leaves `collect.dat`).

**Status:** done 2026-09-16

- [x] the ensemble's per-basin T*S equals the mean of the `trajectories` rows of that basin (unit test with a fake product: one `collect.dat` + one `collect.toml` under a temporary root)
- [x] `COLLECT_PASSED` / `COLLECT_TOTAL` in `ensemble.toml` equal `N_PASSED` / `N_TOTAL` of `collect.toml`
- [x] no `collect.dat` is refused by name; a `collect.toml` without `[Criteria]` is reported as "no verdict", not as zero
- [x] `t_report_reads_collect` rewritten to the new product and green
