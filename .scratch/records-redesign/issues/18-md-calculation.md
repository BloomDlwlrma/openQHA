# 18: MD Calculation: md.toml in property style, RUNNING then NORMAL TERMINATION

**What to build:** both trajectory drivers leave `_records/md_<route>/basinNN/md.out`, `md.toml` and (when run by a Batch) `driver.log`, with the setting in every stem when it is not the default (`md_s2.out`, `md_s2.toml`, `driver_s2.log`) and no `default/` directory. The Property file holds `[Calculation_Status]`, `[Calculation_Info]` (molecule, basin, route, setting, engine, thermostat, timestep, hydrogen mass, temperature, frame spacing, sample every, platform), `[Masses]`, `[Force_Check]`, `[Equilibration]`, `[Production]` and `[[Segment]]`. The OpenMM driver writes it with STATUS RUNNING the moment equilibration ends and rewrites it with NORMAL TERMINATION at the end; a RUNNING file found at start means resume from the frames in the DCD. Engine internals, integrator parameters, relaxation, provenance and the file lists go to `md.out`. The analysis identity assertion, 02d and the resume read the blocks.

**Blocked by:** 16, and 17 by the user's sequencing (17 first, then 18).

**Status:** ready-for-agent

- [ ] after the OpenMM and ASE integration tests, `_records/md_<route>/basinNN/` holds exactly `md.out` and `md.toml`; a run with `--setting s2` adds `md_s2.out` and `md_s2.toml` in the same folder; no directory named `default` exists under `_records`
- [ ] the `.toml` starts with `[Calculation_Status]`; killed after equilibration it reads RUNNING and the next run resumes without re-equilibrating (unit test on the record module, integration on OpenMM)
- [ ] the analysis asserts identity from `[Calculation_Info]` and `[Masses]` (unit test)
- [ ] the md record module reads and writes both routes' files; no reader looks for the old flat keys
