# 16: Property file writer and the no-setting-level path rule

**What to build:** a writer that renders a Calculation's Property file in ORCA property style carried into TOML: `[Calculation_Status]` first (PROGNAME, VERSION, STATUS), upper-case keys each followed by a `# Type, unit: doc` comment taken from a schema table, arrays of tables for repeated things with an `INDEX` key; the standard-library reader gets every value back and ignores the comments. The records path rule becomes the engine-file rule: `_records/md_<route>/` with the setting in the file stem when it is not the default, and the literal `default` never appears in any path a record is written to. Prefactor: nothing in the chain uses it yet.

**Blocked by:** None (can start immediately).

**Status:** done 2026-09-15

- [x] a property file written for a mock branch A record starts with `[Calculation_Status]` and reads back to the same values (unit test)
- [x] every key carries its schema comment; a key missing from the schema is written with no comment and the writer reports it (unit test)
- [x] `[[Basin]]` and `[[Segment]]` render as arrays of tables with `INDEX` (unit test)
- [x] the records folder for (molecule, route) is `_records/md_<route>/` and the record file name for (name, setting) is `md.toml` / `md_s2.toml`; no function returns a path containing `default` (unit test)
- [x] expand-contract (to-tickets rule for a wide refactor: ten callers across 17-20): the new `md_records_dir`, `basin_records_dir`, `record_file_name` sit beside the old `records_for`, which is marked for deletion in ticket 19 once every caller has moved
