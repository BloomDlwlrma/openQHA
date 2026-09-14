# 02: Root by partition, no copy-back

**What to build:** on Tianhe the root of the molecule tree is derived from the partition and the account, printed and exported at environment start, and the run tree is written there once. The per-job scratch stops being the runs root, the exit-trap copy into `logs/node_local/` and the keep directory are removed, and the MACE server sockets go to a node-local directory.

**Blocked by:** None (can start immediately).

**Status:** done 2026-09-14

- [x] `<prefix>/HDD_POOL/<acct>/<user>/sherwin/runs` with prefix `/XYFS02` for partitions ai and cn, `/XYAIFS00` for a100x, h100x, hx, a800x, v100x; acct and user from `HOME`; the tail `sherwin/runs` is one variable
- [x] Partition from `OPENQHA_PARTITION`, then `SLURM_JOB_PARTITION`; on a login node with neither, from which prefix is mounted
- [x] An explicit `S0_RUNS_ROOT` wins; when nothing resolves the environment file prints why and exits non-zero (never falls back to HOME)
- [x] The resolved root is printed on the `runs root` line the environment already prints
- [x] The chain body no longer installs the exit trap, the keep directory or the MANIFEST; `S0_KEEP_DIR` and `S0_SCRATCH_TAG` are no longer read by the array job
- [x] Sockets: `/tmp/<user>/<jobid>` (pid when not in a job); the socket module's bind-test fallback unchanged
- [x] Test point 2: sourcing the environment file under a fake HOME and each partition prints the expected root; neither partition nor mount is an error; explicit `S0_RUNS_ROOT` wins
