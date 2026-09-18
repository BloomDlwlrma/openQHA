# 13: Collect as collect.out + .dat tables

**What to build:** collect writes `collect.trajectories.dat`, `collect.criteria.dat`, `collect.assembly.dat`, `collect.blank.dat` (whitespace tables, one `#` header line) and then `collect.out` (the report; last line `openQHA collect terminated normally`), in that order, so the terminal line is the completion marker. The collect driver writes nothing of its own beside them (`collect.json`, `collect.driver.log` gone) and resumes on the terminal line.

**Blocked by:** 12.

**Status:** done 2026-09-15

- [x] `openqha.store.dat` round-trips ints, floats (nan/inf), booleans, None and quoted strings (t_dat_table_roundtrip)
- [x] collect writes the four `.dat` tables and `collect.out` last
- [x] the collect driver's completion test is `terminated_normally(collect.out, "collect")`
- [x] the ensemble report reads `.dat` (t_report_reads_collect)
