# 07: Branch B reads basins from mace/

**What to build:** `--basins auto` lists `mace/basinNN/basin.extxyz` under the molecule directory of the basin tag and starts one trajectory per basin from those geometries. The basin store, its shard rule and the relocated-xyz helper are retired; the example confs no longer name a store.

**Blocked by:** 03 (OpenMM engine folder), 05 (MACE engine folder).

**Status:** ready-for-agent

- [ ] The basin count and geometries come from `mace/basinNN/`; a molecule with no `mace/` folder is refused naming the folder
- [ ] The basin store module and `S0_BASIN_ROOT` are gone from code, confs and docs
- [ ] The 02d-2 array reads basins from the basin tag's molecule directory and writes each row under `openmm/<row>/`
- [ ] The existing branch B tests pass against the new source of basins
