# 05: MACE engine folder

**What to build:** branch A leaves, under `mace/` of the molecule directory, `confNN/opt.traj`, `confNN/opt.log` and `confNN/conf.extxyz` for every tightened conformer (CREST's plus the pooled reference), and `basinNN/basin.extxyz` plus `basinNN/hessian.npy` for every surviving basin. The branch A record keeps its content and goes to `_records/`.

**Blocked by:** 01 (Layout module).

**Status:** done 2026-09-14

- [x] The ASE optimiser is given a trajectory file and a log file per conformer; `conf.extxyz` carries energy and forces as ASE writes them
- [x] `basin.extxyz` comment names the conformer index it came from and CREST's comment line for that frame
- [x] `hessian.npy` is the raw analytic Hessian, 3N x 3N, eV/Å², float64, neither mass-weighted nor projected
- [x] `basins.json` and `basins.xyz` go to `_records/`; the store write is gone
- [x] Test point 4: the census on a prepared ensemble leaves three files per conformer and two per basin; the extxyz names its conformer; the Hessian is 3N x 3N and symmetric
