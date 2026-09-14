# 09: Examples end to end, docs

**What to build:** the shipped chains (02b qha, 02d identity, 02d-2 array) run into the new tree on both clusters from their confs; the array submitter and array job no longer compose scratch tags or keep directories; the runbook's storage section and the branch A/B workflow docs describe the new tree and the root rule.

**Blocked by:** 02, 04, 06, 07.

**Status:** done locally 2026-09-15; the two cluster runs are the user's (commands in the note)

- [x] Locally (WSL, real CREST, OpenMM route on CPU, `S0_RUNS_ROOT=/tmp/openqha_e2e`): 02b `branchA.conf` then `test30.conf` (A → B → collect → report) and 02d `branchA.conf` then `test30.conf` (A → B → identity) run end to end through `chain_body.sh`, one molecule directory each with `crest/`, `mace/`, `openmm/basinNN/`, `_records/`, per-tag records under `<root>/<tag>/_records/`, nothing new under the checkout's `analysis/` or `logs/`
- [ ] `run_chain.sh` with the 02b and 02d confs on a100x leaves the same tree under the AI root (USER runs it)
- [ ] The 02d-2 array leaves nine settings' files (`*_<row>.*`) in each `openmm/basinNN/` of one molecule directory (USER runs it)
- [x] `submit_array.sh` and `array.slurm` pass no `S0_SCRATCH_TAG` / `S0_KEEP_DIR`
- [x] Runbook §0b/§6 and the workflow docs show the tree and the root rule; `docs/output_inventory.md` gains a "since 2026-09-14" section pointing at the new layout
- [x] Full test suite green (28/28 with `--all`, 2026-09-15)

Found and fixed while running locally: the collect driver's `main()` did not pass `--basin-tag`/`--setting` to its tasks and referenced a removed variable (my ticket-04 patch); parsl's run directory under the checkout fails on a Windows drive (certificates must be 0700) — it now goes to `<root>/<tag>/_records/parsl/<job>.<pid>/`; the `local` resource config had no collect executor.
