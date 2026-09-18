# 11: Branch A record as branchA.out + basins.toml

**What to build:** branch A leaves its record in `_records/` as `basins.toml` (the record a program reads back: TOML, the same keys as before minus None) and `branchA.out` (a sectioned report: settings, CREST, census, the basins table, the criteria, the complete record; its last line is `openQHA branch A terminated normally`). `basins.json` and `basins.xyz` are gone.

**Blocked by:** step 2 ruling (2026-09-15).

**Status:** done 2026-09-15

- [x] `openqha.store.toml_out` writes TOML the standard library reads back equal to the record minus Nones (t_toml_record_roundtrip)
- [x] `Report.write(path, step=...)` ends the file with the terminal line; `report.terminated_normally()` finds it and refuses a cut file
- [x] `_records/` of a branch A run holds exactly `branchA.out` and `basins.toml` (t_crest_engine_folder)
- [x] Every reader of the record (`basins.read_record`, the ensemble report, 02c, 02d, 02a debug, the parsl branch A driver) reads `basins.toml`
