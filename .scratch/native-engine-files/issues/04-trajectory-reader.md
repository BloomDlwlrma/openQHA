# 04: Trajectory reader

**What to build:** one function that, given an OpenMM engine folder, returns positions in Å as float64, symbols, masses and the per-frame table, read from `start.pdb`, `traj.dcd` and `state.csv`. The collect analysis, the ensemble report and the 02d identity check read trajectories through it; nothing reads `frames.npy` any more.

**Blocked by:** 03 (OpenMM engine folder).

**Status:** done 2026-09-14

- [x] Positions from the DCD equal the positions the driver had in memory to float32 tolerance
- [x] A folder without `traj.dcd` (or without `start.pdb`) is refused with the folder named in the error
- [x] Collect, ensemble report and 02d run unchanged in their numbers on a trajectory written by ticket 03
- [x] Test point 6
