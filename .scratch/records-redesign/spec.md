# Records redesign: one Record per Calculation, ORCA property style, no Batch records

Status: ready-for-agent
Date: 2026-09-15
Vocabulary: CONTEXT.md (Calculation, Batch, Record, Report, Property file). Decisions:
docs/adr/0003 (amended today). Facts this spec was written from:
`toml_keys_inventory.md` and `mock_basins_property_style.toml` beside it.
Grilling rounds 1 and 2 (Q1-Q12) were all answered with the recommended option except
Q4, where the user chose (b): no Batch record at all.

## Problem Statement

The records written after step 2 are the right file kinds (`.out`, `.toml`, `.dat`) but
the wrong content and the wrong home. `basins.toml` carries about 330 keys of which a
later step reads about 20; the rest is prose, provenance and diagnostics that belong in
the `.out`, and about 55 keys are written twice. `md.toml` is the same at half the size.
CREST's `input.toml` and ORCA's `.property.txt` are what the user asked for: a small,
typed, blocked file that a program reads, next to a report a human reads. The records
also sit under a `default/` level the user had ruled out, and three JSON files describe a
Batch (a parsl driver's run over many Calculations) in three different shapes that
nothing reads, while the Slurm log, which everybody does read, does not say which
Calculations ran, how they ended, or where their records are.

## Solution

Two words fix the units. A **Calculation** is one step on one molecule or basin and owns
a **Record**: a **Report** (`.out`, for a human, last line "terminated normally") and a
**Property file** (`.toml`, for a program). A **Batch** is one driver invocation over many
Calculations and owns nothing: the Slurm log is its report, and it lists every
Calculation on one aligned line with the return code, the STATUS found in the Property
file, and the Property file's absolute path.

The Property file takes ORCA's `.property.txt` shape carried into TOML: a
`[Calculation_Status]` block first (`PROGNAME`, `VERSION`, `STATUS = "NORMAL
TERMINATION"`), a `[Calculation_Info]` block with the inputs, then the result blocks a
later step reads, and nothing else. Keys are upper-case with a `# Type, unit: doc`
comment generated from one schema table, so every file carries the same comments;
repeated things (basins, resume segments) are arrays of tables with an `INDEX` key.
Provenance (engine, weights, versions, machine), diagnostics, the criterion detail lines,
the sigma tolerance sweep and every file list are printed in the Report only.

The records leave the `default/` level: the setting goes into the file stem exactly as
it does for engine files.

    <molecule>/_records/
      branchA.out   branchA.toml                            (branchA.toml replaces basins.toml)
      md_openmm/basinNN/md.out  md.toml  driver.log         (default setting)
      md_openmm/basinNN/md_s2.out  md_s2.toml  driver_s2.log   (setting s2)
      md_openmm/collect.out  collect.toml  collect.trajectories.dat  .criteria.dat  .assembly.dat  .blank.dat
      md_openmm/collect_s2.out  collect_s2.toml  collect_s2.trajectories.dat  ...
      md_openmm/ensemble.out  ensemble.toml   /  ensemble_s2.out  ensemble_s2.toml
      md_openmm/02d_frequency_identity.out  .toml           (02d only)
      md_ase/...                                            (the same for the ASE route)
    <root>/<tag>/_records/parsl/<job>.<pid>/                 parsl's own logs, unchanged

Gone: `_records/md_<route>/<setting>/` as a level, `basins.toml`,
`<root>/<tag>/_records/branchB_parsl_summary.json`, `collect_batch.json`,
`analysis/branchE/<tag>/batch.json`.

## User Stories

1. As a user opening a finished molecule directory, I want `_records/branchA.toml` to read like ORCA's property file (status, inputs, results, one block each), so that I see what was asked and what came out without scrolling through provenance.
2. As a user, I want every Property file to start with `[Calculation_Status]` and `STATUS = "NORMAL TERMINATION"`, so that one `grep` over a tag tells me which Calculations finished.
3. As a user, I want the inputs of a Calculation in `[Calculation_Info]` with units in the comment, so that I can compare them with `crest/input.toml` by eye.
4. As a user, I want a Property file that a later step reads to hold only what that step reads, so that a change in the report never changes the interface.
5. As a user, I want `md.toml`, `collect.toml`, `ensemble.toml` and `branchA.toml` to share one shape, so that I learn the form once.
6. As a user, I want the Report to carry the provenance (engine, weights, versions, machine), the criterion detail lines and the diagnostics, so that nothing measured is lost.
7. As a user, I want no `default/` folder under `_records`, so that records follow the same rule as engine files: the setting is in the stem or absent.
8. As a user, I want a Report and its Property file to share a stem (`md_s2.out`, `md_s2.toml`), so that they sort together.
9. As a user reading a Slurm log, I want one aligned line per Calculation with its return code, STATUS and the absolute path of its Property file, so that a failed trajectory is located without opening anything else.
10. As a user, I want the three parsl drivers to print that table in the same shape, so that a branch A log and a branch B log read alike.
11. As a user, I want no JSON summary of a Batch anywhere, so that the Slurm log is the one place to look.
12. As a user, I want `analysis/branchE/` no longer written into the repository checkout, so that a run leaves the checkout clean.
13. As a user, I want a killed trajectory to leave `STATUS = "RUNNING"` in its Property file, so that the next run resumes from the DCD without re-equilibrating and the Slurm table shows which ones were cut.
14. As a user, I want `driver.log` to stay beside `md.out` (with the setting in its stem), so that a trajectory's stdout is with its Record.
15. As a user, I want collect's four `.dat` tables kept beside `collect.out`, so that per-trajectory numbers stay in table form.
16. As a user, I want the ensemble step and 02d to read basins, sigma and G - E_el from `branchA.toml`'s `[[Basin]]` blocks, so that the answer is computed from the property file and nothing else.
17. As a user, I want the analysis to assert the identity of a trajectory from `md.toml`'s `[Calculation_Info]` and `[Masses]`, so that the assertion reads the same block a human reads.
18. As a user, I want the branch A parsl driver to decide "done" from `STATUS` in `branchA.toml`, so that the completion marker is the same for programs and people.
19. As a user, I want a rerun of every example locally into `$HOME/runs/openQHA`, so that the new records exist for real and not only in tests.
20. As a user, I want sample Records for 02b (both routes) and 02d (OpenMM) in the repository, so that a reader of the README opens a real `branchA.toml` before running anything.
21. As a user, I want the key inventory and the mock kept beside this spec, so that the decision's evidence is not lost.
22. As a user, I want the docs (output inventory, READMEs, runbook) to show the new tree and the Slurm table, so that no document describes the `default/` level or the JSON summaries.
23. As a user, I want the test suite to assert there is no `default` directory anywhere under `_records` after each integration test, so that the level cannot creep back.

## Implementation Decisions

- **Two units.** Calculation and Batch as defined in CONTEXT.md. Every writer in the chain is classified: the branch A pipeline, the two trajectory drivers, the analysis, the ensemble report and 02d write Calculation Records; the three parsl drivers are Batches and write nothing but their stdout.
- **Property file shape.** `[Calculation_Status]` (PROGNAME, VERSION, STATUS), `[Calculation_Info]` (inputs), result blocks; arrays of tables for repeated things with an `INDEX` key. Keys upper-case. A schema table per step in the writer gives each key its type, unit and doc, rendered as a trailing comment; the reader is the standard library's TOML parser and ignores comments. STATUS is the completion marker for programs; the Report's terminal line stays for humans.
- **Per step blocks.** Branch A: `[CREST_Run]`, `[Census]`, `[[Basin]]` (INDEX, ENERGY, RELATIVE, SIGMA, G0, N_IMAGINARY, LOWEST_FREQ, G_MINUS_EEL, A_MINUS_EEL), `[Criteria]` (N_PASSED, N_TOTAL, ALL_PASSED). MD: `[Masses]`, `[Force_Check]`, `[Equilibration]`, `[Production]`, `[[Segment]]`. Collect: `[Criteria]` and the four `.dat`. Ensemble: `[Result]` with F_conf, populations, delta G, effective basins, crossings, and `[[Basin]]` with TS and frames. 02d: its four stages as blocks.
- **STATUS values.** `NORMAL TERMINATION` at the end of every step; `RUNNING` written by the MD driver after equilibration (resume marker; found at start means resume from the DCD frames on disk); a Property file missing or without `STATUS` means the Calculation never got that far.
- **No setting level.** The records path rule becomes the engine-file rule: `_records/md_<route>/` plus a stem carrying the setting when it is not the default. One function gives the record file name from (name, setting); one gives the records folder from (molecule, route). The `default` literal never appears in a path.
- **Batch table.** All three parsl drivers print, after the run, one header and one aligned line per Calculation: `species basin seed rc seconds STATUS record` first, driver-specific columns after; a footer with the batch wall and the pass count. `record` is the Property file's absolute path; `STATUS` is read from that file after the subprocess returns (`FAILED` if absent and rc != 0). The JSON writers and the `analysis/branchE/` writer are removed.
- **Readers move to the blocks.** basins reader (ensemble, 02a/02c/02d, branch A parsl), md_record reader (resume, analysis, 02d, branch B parsl line), collect tables (ensemble) read the new blocks; no reader keeps a JSON or `basins.toml` fallback: the old layout was deleted by ticket 08's tool and a rerun is the migration.
- **Report content.** Each Report prints once at the top its provenance (engine, weights, versions, machine, patch probe), then what it prints today. The criterion detail lines, sigma tolerance sweep, thermochemistry breakdown, force-field internals, relaxation and file lists move from the Property file to the Report.
- **driver.log** stays beside `md.out`, stem carrying the setting.
- **Local rerun and samples.** After the tickets land: 02b on OpenMM and ASE and 02d on OpenMM, into `$HOME/runs/openQHA` in WSL; each `_records/` tree copied into `examples/<example>/sample_records/<route>/` (text only, engine files excluded).
- **ADR 0003 amended**, not replaced: the file kinds stand; the content rule (status + inputs + results a later step reads) and "no Batch record" are the amendment.

## Testing Decisions

- A good test opens the files a Calculation leaves and asserts on their content and names, or runs a driver and asserts on its stdout; it never asserts on the dict a writer built.
- Unit seams (existing, extended): the layout tests (`records_for` and the new record file name; assert no `default` in any path), the TOML round trip (upper-case keys with trailing comments read back unchanged), the md record resume test (STATUS RUNNING then NORMAL TERMINATION), the report-reads-collect test (ensemble reads `[[Basin]]` from `branchA.toml` and the `.dat`).
- Integration seams (existing, extended): the four engine-folder tests run the real drivers with stand-in CREST and short MD and assert the exact set of files under `_records/` (no `default`, stems by setting), the first block of each `.toml` is `[Calculation_Status]` with STATUS, and the Report ends with the terminal line. One new integration test runs the branch B parsl driver on the local resource config with one basin and asserts the aligned table in stdout (columns, absolute path, STATUS) and that no JSON is written under `<root>/<tag>/_records/`.
- Prior art: `tests/unit/t_layout_molecule_directory.py`, `t_toml_record_roundtrip.py`, `t_md_record_resume.py`, `t_report_reads_collect.py`; `tests/integration/t_*_engine_folder.py`.
- The user intends to change the `tests/unit/t_*.py` style later; this spec adds tests in the present style.

## Out of Scope

- The engine folders and the molecule tree (ADR 0001, 0002): unchanged.
- The physics, the criteria, the protocols: unchanged; only where their verdicts are written moves.
- Cluster runs: the user's.
- The test style change.
- xtb/orca engine folders (02c) and its level benchmark's own output form.

## Further Notes

- `driver.log` is the one file written by a Batch into a Calculation's Record; it is the Calculation's stdout, so it counts as the Calculation's.
- The Slurm job id is allocated by Slurm and only ever read; the Batch table prints it in its header for the reader.
