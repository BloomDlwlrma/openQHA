# 20: Batch: the Slurm log is the report

**What to build:** the three parsl drivers (branch A, branch B trajectories, collect) write nothing about the Batch. After the run each prints one header and one aligned line per Calculation with the common columns first, `species basin seed rc seconds STATUS record`, then its own columns (frames, T mean, card; criteria count; basins, CREST count, fallback), and a footer with the batch wall, the pass count and the Slurm job id. `rc` is the subprocess return code, STATUS is read from the Calculation's Property file after the subprocess returns (FAILED when the file is absent and rc is not zero, RUNNING when it was cut), `record` is the Property file's absolute path. `branchB_parsl_summary.json`, `collect_batch.json`, `analysis/branchE/<tag>/batch.json` and their writers are gone.

**Blocked by:** 17, 18.

**Status:** ready-for-agent

- [ ] the branch B parsl driver on the local resource config with one basin prints the header and one line whose `record` is the absolute path of an existing `md.toml` with STATUS NORMAL TERMINATION, and nothing is written under `<root>/<tag>/_records/` except `parsl/` (new integration test)
- [ ] the branch A and collect drivers print the same first seven columns (unit test on the line formatter)
- [ ] no batch summary is dumped as JSON by the three drivers; `analysis/branchE` is not created
