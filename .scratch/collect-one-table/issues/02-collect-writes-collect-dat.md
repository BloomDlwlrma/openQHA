# 02: Collect leaves `collect.dat`; the Report's dump stops repeating the rows

**What to build:** running collect on a molecule leaves exactly `collect.out`, `collect.toml` and `collect.dat` in the route's records folder (stem rule for a setting: `collect_s2.dat`). The Table holds `trajectories`, `blank`, `assembly` in that order, always all three, every column commented from a column schema in the collect record module with the wording settled in the spec; column names unchanged. The path helper returns `out`, `toml`, `dat`. The expanded dump at the end of `collect.out` no longer carries the trajectories, blank control, assembly and criteria row lists; the `Criteria` section and the per-trajectory analysis dicts stay. Write order stays Table, Report, Property file.

**Blocked by:** 01 (Table module: sections and column comments).

**Status:** ready-for-agent

- [ ] after collect, the records folder holds `collect.out`, `collect.toml`, `collect.dat` and no other `.dat`
- [ ] `collect.dat` has the three sections in order; each header column has a comment; the writer reports no unknown column (unit test over the schema against the columns collect builds)
- [ ] `collect.out` contains each criterion sentence exactly once
- [ ] the collect Batch driver still reads STATUS from `collect.toml` and is otherwise untouched
- [ ] the `N_TOTAL` comment in the collect Property file points at `collect.out` only
