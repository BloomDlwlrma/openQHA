# 12: Trajectory record as md.out + md.toml

**What to build:** both trajectory drivers leave, beside the engine folder, `_records/md_<route>/<setting>/basinNN/md.toml` (seed, segments, thermostat, timestep, masses, force check, equilibration, production, engine file names, job id) and `md.out` (the readable report, last line `openQHA md terminated normally`). `frames.npy`, `meta.json`, `summary.json`, `progress.json`, `equilibrated.json` are gone; a resume reads `md.toml` and counts the frames in the engine file.

**Blocked by:** 11.

**Status:** done 2026-09-15

- [x] `openqha.quasi_harmonic.md_record` reads and writes the record for both routes
- [x] the OpenMM driver writes a partial `md.toml` the moment equilibration ends (resume after a kill during production)
- [x] `already_complete` decides from `md.toml` + the DCD header (t_md_record_resume, replacing t_frames_flush)
- [x] the records folder holds exactly `md.out` and `md.toml` on both routes (t_openmm_engine_folder, t_ase_engine_folder)
- [x] the reader hands the record to collect and 02d from `md.toml`; the parsl driver's summary reads it
