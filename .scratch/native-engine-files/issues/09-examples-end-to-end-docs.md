# 09: Examples end to end, docs

**What to build:** the shipped chains (02b qha, 02d identity, 02d-2 array) run into the new tree on both clusters from their confs; the array submitter and array job no longer compose scratch tags or keep directories; the runbook's storage section and the branch A/B workflow docs describe the new tree and the root rule.

**Blocked by:** 02, 04, 06, 07.

**Status:** ready-for-agent

- [ ] `run_chain.sh` with the 02b and 02d confs on a100x leaves one molecule directory each under the AI root with `crest/`, `mace/`, `openmm/default/`, `_records/`
- [ ] The 02d-2 array leaves nine `openmm/<row>/` folders in one molecule directory
- [ ] `submit_array.sh` and `array.slurm` pass no `S0_SCRATCH_TAG` / `S0_KEEP_DIR`
- [ ] Runbook §0b/§6 and the workflow docs show the tree and the root rule; `docs/output_inventory.md` gains a "since 2026-09-14" section pointing at the new layout
- [ ] Full test suite green
