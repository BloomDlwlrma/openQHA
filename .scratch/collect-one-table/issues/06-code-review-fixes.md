# 06: Code-review fix set (two-axis review of tickets 01-05, 2026-09-16)

**What to build:** the Table and its readers behave exactly as the spec and the glossary say, with nothing silent. A `collect.dat` without a `[trajectories]` section is refused by name, not read as "every basin missing"; a section key the Table does not know, or a `[name]` repeated in one file, is refused rather than dropped or overwritten; column drift is judged over every row, not the first; the Report footer stops naming "the `.dat` tables"; the column explanations and the test comments use the glossary's words (no "run", no "product" for a Record); the `Type, unit: doc` comment is written by one helper shared with the Property file; the spec text states the header rule the reader applies and the runtime refusal on drift; the integration test checks the column comments per section; the test inventory lists the new tests.

**Blocked by:** 05 (Re-run collect and the ensemble on the propanal sample).

**Status:** done 2026-09-16

Hard fixes (Standards 1-4, correctness 1-4 and 6; Spec 1-3, 5, 8):
- [x] `collect_trajectories` refuses a Table with no `[trajectories]` section, naming the file and its sections
- [x] `write_collect_table` refuses a section key outside `SECTIONS`; drift is computed over every row of a section
- [x] `read_tables` refuses a repeated `[name]`; the unreachable flush guard is gone
- [x] `dat._comment_lines` and `property` share one `Type, unit: doc` helper (an empty doc renders the same way in both)
- [x] the Report footer (`openqha.store.report`) names "the Table (.dat) of the same stem", not "the .dat tables"
- [x] glossary words: "product" -> Record in the integration-test comment; "run" -> the driver invocation / segment in `COLUMNS` wall and cost docs; the records-inventory note says "four `.dat` files"
- [x] the analyse driver's two adjacent "Property file LAST" comments are one; its `table_sections` docstring matches the rule (the driver refuses drift, the Table is still written first)
- [x] `t_ase_engine_folder` checks the column comments per section; `t_property_file`'s two-dot example is not a retired file name
- [x] `tests/README.md` lists `t_dat_table_sections` and `t_collect_table_columns`
- [x] spec: the Table reader's header rule reads "the last `#` line before the rows"; the drift decision says collect refuses at run time (the same rule `md_record` applies to `md.toml`) after writing the Table and before the Report, so the molecule is redone
- [x] unit tests cover the three new refusals; all suites green

Judgement calls left as they are (user's call later): `table_sections` staying in the analyse driver; `write_tables` accepting a mapping or pairs and rows or (rows, columns); `collect_paths` composing the suffix beside `layout`; the trailing blank line after the last section.
