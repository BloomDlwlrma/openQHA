# 10: ASE engine folder

**What to build:** the ASE-route trajectory driver (`s0_B_qha_trajectory.py`, the CPU cross-check of the OpenMM route) leaves, under `md_ase/basinNN/` of the molecule directory, exactly ASE's own files: `start.extxyz` (the post-relax start, energy and forces in the header), `md.traj` (ASE `Trajectory`: positions, momenta, energy, forces per sampled frame) and `md.log` (ASE `MDLogger`: time, Etot, Epot, Ekin, T per frame). A non-default setting carries its name in the file names, as for OpenMM. A killed run resumes from the last frame of `md.traj` and appends. The driver's records (`meta.json`, `progress.json`, `frames.npy`) go to `_records/md_ase/<setting>/basinNN/`. Collect, the ensemble report and 02d read `ase/` through the same trajectory reader, choosing the route by what the molecule directory holds (`--route auto`).

**Blocked by:** 04 (trajectory reader), 07 (basins from mace/).

**Status:** done 2026-09-15 (integration test + the 02b test chain on the ASE route end to end locally)

- [x] `md_ase/basinNN/` holds exactly `start.extxyz`, `md.traj`, `md.log` (default setting) or their `_<setting>` forms
- [x] `md.traj` frame count equals `md.log` row count after every flush; a second run over a partial trajectory resumes from the last frame and appends only new frames; over a finished one it does nothing
- [x] `state.npz` is gone; `meta.json`, `progress.json`, `frames.npy` sit in `_records/md_ase/<setting>/basinNN/`
- [x] `trajectory_reader.read_trajectory` reads an `ase/` folder (positions, symbols, masses, table from `md.log`) and `trajectory_dirs` finds either route; collect, report and 02d take `--route auto|openmm|ase` and put their records under `_records/<route>/<setting>/`
- [x] Test point: a few hundred CPU steps on a shipped molecule through the ASE driver; exactly the three files; frames = rows; resume appends; the reader's positions equal the driver's float64 frames; collect runs on it with `--route ase`
