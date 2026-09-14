# 03: OpenMM engine folder

**What to build:** the trajectory driver leaves, under `openmm/<setting>/basinNN/` of the molecule directory, exactly the seven OpenMM files: `start.pdb`, `system.xml`, `integrator.xml`, `traj.dcd`, `state.csv`, `state.xml`, `state.chk`. DCD and CSV are written at the sampling interval; state XML and checkpoint at every flush. A killed run resumes from the state and appends. Everything the driver used to write beside the frames (`frames.npy`, `meta.json`, `summary.json`) goes to `_records/` with the same content.

**Blocked by:** 01 (Layout module).

**Status:** done 2026-09-14

- [x] Topology built from the atomic numbers: one chain, one residue `MOL`, elements only; `start.pdb` is the post-relax structure
- [x] `traj.dcd` frame count equals `state.csv` row count after every flush
- [x] Second invocation over a finished basin does nothing; over a partial one it resumes from `state.xml` (checkpoint when it loads) and appends only new frames
- [x] No seed level: one folder per basin; the setting is `default` for the qha and identity chains and the row name for the 02d-2 array
- [x] The driver's records (`meta.json`, `summary.json`, `frames.npy`) land in `_records/` and nothing but the seven files is in the engine folder
- [x] Test point 3: a few hundred CPU steps on a shipped molecule; exactly the seven files; frames = rows; resume appends; no `.npy` or record in the folder
