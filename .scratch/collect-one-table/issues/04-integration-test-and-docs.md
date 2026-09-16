# 04: End to end on the ASE chain, and the docs name one Table

**What to build:** the ASE engine-folder integration test checks that the chain ends with `collect.out`, `collect.toml`, `collect.dat` (three sections, every column commented) and that the ensemble runs on it. Every document that lists the four `.dat` names the one: the output inventory (section 8), the records inventory, the branch B production page, the repository README, the propanal example README, and the docstrings of the collect record and table modules.

**Blocked by:** 03 (The ensemble reads the Table and the Property file).

**Status:** ready-for-agent

- [ ] integration test green on propanal, ASE route, checking the Record on disk
- [ ] `grep` for `.trajectories.dat`, `.criteria.dat`, `.assembly.dat`, `.blank.dat` finds nothing outside `_backup/`, `.scratch/` history and `.mem/`
- [ ] the output inventory's section 8 tree shows `collect.dat` with its three sections
