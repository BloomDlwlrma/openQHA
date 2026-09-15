# 21: Docs, local rerun, sample records

**What to build:** the documentation shows the new `_records/` tree and the Batch table (output inventory, branch A and B workflow docs, the runbook, the example READMEs, README); the delete-old-layout plan also lists the step-2 leftovers (`_records/**/default/`, `basins.toml`, the three JSON batch files) so a cluster tree can be cleaned before a rerun. 02b on the OpenMM and ASE routes and 02d on OpenMM are rerun locally into `$HOME/runs/openQHA` in WSL, and each `_records/` tree is copied into `examples/<example>/sample_records/<route>/` (text only, no engine files).

**Blocked by:** 19, 20.

**Status:** ready-for-agent

- [ ] no document describes `md_<route>/<setting>/`, `basins.toml`, `meta.json` or a JSON batch summary as current
- [ ] the delete-old-layout plan lists the step-2 leftovers under a given root (unit test)
- [ ] `examples/02b_.../sample_records/openmm/`, `.../ase/` and `examples/02d_.../sample_records/openmm/` exist, each with `branchA.toml`, `md_<route>/basinNN/md.toml`, `collect.toml`, `ensemble.toml` and the `.out` files, every `.toml` starting with `[Calculation_Status]`
- [ ] the suite passes with `--all`
