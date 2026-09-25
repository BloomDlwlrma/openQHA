# 01: The ORCA child environment is Slurm-blind

**Why (round 13 rulings, 2026-09-25):** the draw300 labels array on TianheXY-CN produced
zero labels -- every frame's ORCA died in "Startup" because the job let OpenMPI's Slurm
resource allocator arm itself out of the scheduler variables it left behind, then removed
the one variable the allocator needed; the frames' own outputs end in
`.... aborting the run`. Evidence and primary sources: the feature's research note and
grilling round.

**What to build:** every ORCA subprocess openQHA launches runs in a child environment that
carries no `SLURM*`/`PMI*` variable at all, built in one seam, so no scheduler environment
can ever abort ORCA before it starts -- on the labels route, the msRRHO level jobs, the
single-point/composite launchers, everywhere. The same seam switches OpenMPI's own CPU
binding off, so placement has exactly one owner: the worker's `taskset` range. The policy
is written down (ADR 0008 -- renumbered from the planned 0007; plus the HPC docs); the
job scripts keep their existing narrow
unset untouched -- the ORCA invariant no longer depends on it.

**Blocked by:** None (can start immediately)

**Status:** done 2026-09-25. Evidence: unit suite 64/64 (`t_orca_child_env.py` included;
its 7 seam checks + 5 totality checks pass); `py_compile` clean on the four edited Python
files; **site verification passed** -- job 7673439 on cnode2448 (`debug`, TianheXY-CN) via
`.scratch/orca-slurm/verify-orca-one.sh`: the in-job seam self-check read `none / none`
with `SLURM_JOBID` armed, `dsgdb9nsd_006885 displaced_b01_k3` retried once -> `labelled
185 65`, and the fresh `.out` ends `****ORCA TERMINATED NORMALLY****` (ORCA 3 min 5 s,
exit 0). The two additions in the table below keep their OPEN dispositions.

- [x] The ORCA child environment seam deletes every variable whose name starts with `SLURM`
      or `PMI`, keeps `S0_ORCA_PATH`/`S0_ORCA_LIB` prepended to PATH / LD_LIBRARY_PATH, and
      sets `OMPI_MCA_hwloc_base_binding_policy=none` (`orca.subprocess_env()`;
      `t_orca_child_env` A).
- [x] Every ORCA launch goes through the seam: the labels worker, the msRRHO level jobs,
      the single-point/composite launcher (which passed no environment at all) and the
      two direct launchers outside the interface (the branch-B opt+freq driver and the
      package2 high-level calibration driver) -- plus a **sixth site the ticket's table
      missed** (`examples/02c_hessian_benchmark_levels/s0_level_benchmark.py`); see the
      additions table below. `t_orca_child_env` B scans the live tree and fails on any
      textual spawn without the seam; a binary resolved at runtime (the `capabilities`
      version probe) is invisible to the scan and is listed below as a reported extra.
- [x] Offline unit tests: a fabricated environment full of Slurm/PMI variables comes back
      stripped, with the shims and the binding knob intact; a check that no ORCA launch
      bypasses the seam (the launchers pass its environment) -- `tests/unit/t_orca_child_env.py`.
- [x] Site verification: one known-failed draw300 frame, re-run by hand with the existing
      per-frame retry on Tianhe, terminates normally (the fresh `.out` carries
      `****ORCA TERMINATED NORMALLY****`) -- the gate the retry round waits for. **Done
      2026-09-25, job 7673439 (cnode2448, `debug`), `dsgdb9nsd_006885 displaced_b01_k3`**:
      the old `.out` read the incident's `ras_base_allocate` force-terminate before the
      retry; the retry printed `labelled 185 65`; the fresh `.out` ends with the terminal
      line; the old output is kept at
      `$HOME/orca_verify_backup/orca.wb97m-d3bj_def2-tzvppd.displaced_b01_k3.old.out`.
      The frame's one manual retry is spent; the next step per the sequencing rule is the
      failure count (`s0_hl_progress --tag draw300`), then ticket 02's retry round.
- [x] ADR 0007 records "ORCA children are Slurm-blind; the worker owns placement", with the
      rejected alternatives (Slurm-shaped allocation, `srun` steps, MCA exclusions) and the
      OpenMPI-source evidence. **Number changed to 0008**: 0007 was taken the same day by
      the msrrho imaginary-mode ADR (`docs/adr/0007-one-imaginary-mode-policy-floor-and-the-two-principles.md`);
      the misnumbering is noted in the record itself.
- [x] The HPC README rule 2, the ORCA env script header and the labels script's loop
      comment state the two layers (job-level narrow unset for non-ORCA payloads; the full
      strip in the seam); the CREST/MACE/training scripts and the nine sbatch filters are
      unchanged; `OMPI_MCA_rmaps_base_oversubscribe=1` is documented as the fallback only,
      not set. The pipeline-debug script's twin loop carries the same comment.

## Additions beyond this ticket's text (dispositions OPEN)

| # | addition | where | basis | recommendation |
|---|----------|-------|-------|----------------|
| 1 | the examples level-benchmark ORCA launch routed through `subprocess_env()` | `examples/02c_hessian_benchmark_levels/s0_level_benchmark.py` | the ticket's criterion "Every ORCA launch goes through the seam" + the totality test; this sixth site passes no environment today and was absent from the round-13 table | keep (one line; makes the totality check truthful) or revert and exempt it in the test |
| 2 | ADR number 0008 instead of 0007 | `docs/adr/0008-orca-children-are-slurm-blind.md` | 0007 is the msrrho ADR of the same day -- forced by the collision, not a scope choice | keep (already noted in the ADR) |
| 3 | reported, NOT changed: `openqha/capabilities.py::_binary_runs("orca")` runs `orca --version` with the inherited environment (a runtime-resolved binary, invisible to the static scan) | `openqha/capabilities.py` | the ticket's "every ORCA subprocess" and the ADR's absolute sentence; the spec's route table lists only the five launch sites | options: (a) route it through a lazy seam import in a tiny follow-up -- makes the invariant literal; (b) leave it -- a `--version` run cannot reach the MPI-startup path the incident lived in. **Recommendation: (a), later; not needed for the fix or the retry round** |
| 4 | the ADR gained one "Not yet wired" sentence documenting row 3 | `docs/adr/0008-...md` | a known deviation left undocumented would make the ADR false | keep |

