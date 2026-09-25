---
status: accepted
date: 2026-09-25
---

# ORCA children are Slurm-blind; the worker owns placement

(The orca-slurm ticket named this ruling ADR 0007; that number went the same day to the
msrrho imaginary-mode ADR, `0007-one-imaginary-mode-policy-floor-and-the-two-principles.md`.
This record is 0008.)

## Context

The draw300 labels array on TianheXY-C produced zero labels: every frame's ORCA died
within seconds of starting, in "Startup", with
`[file orca_tools/qcmsg.cpp, line 394]: .... aborting the run`. The frames' own outputs
name the mechanism: ORCA's bundled OpenMPI 4.1 saw `SLURM_JOBID`, selected its Slurm
resource allocator, and that allocator force-terminated because `SLURM_TASKS_PER_NODE`
was absent. The job script's narrow unset loop
(`grep -E '^(PMI|SLURM_(CPU|TASK|NTASKS|NPROCS|STEP))'`, `hpc/slurm/hl_labels.slurm`)
had deleted exactly the variable the allocator demanded while keeping the `SLURM_JOBID`
and `SLURM_NODELIST` that armed it.

The OpenMPI v4.1.8 source fixes the rules (research note, primary sources):
`SLURM_JOBID` alone makes `ras/slurm` (priority 50), `plm/slurm` (priority 75, which
launches the ORTE daemons through `srun`) and `ess/slurm` eligible; `ras/slurm` then
requires both `SLURM_NODELIST` and `SLURM_TASKS_PER_NODE`, and `ras_base_allocate`
force-terminates on `ORTE_ERR_NOT_FOUND`. With no component selected, ORTE falls back to
"just add the local host". ORCA's own manual adds the placement rule: its driver must
own its `mpirun` ("Do not start the ORCA driver with mpirun!"), and OpenMPI's own CPU
binding is switched off with `--bind-to none` or, as used here,
`OMPI_MCA_hwloc_base_binding_policy=none` (ORCA 6.1 manual §2.5.2.1).

The deployment shape is 16 independent frames per 64-core node, one `mpirun -np 4` per
frame, each worker pinned by `taskset` to its own four cores: Slurm neither launches nor
binds these ranks (Slurm's MPI guide modes 2/3), so placement has no owner unless one is
designated.

## Decision

**No ORCA subprocess ever sees a scheduler variable, and the environment it does see is
built in one seam.** `openqha.qm_interfaces.orca.subprocess_env()` -- the single
enforcement point -- deletes every variable whose NAME starts with `SLURM` or `PMI`,
keeps `S0_ORCA_PATH` / `S0_ORCA_LIB` and prepends them to PATH / LD_LIBRARY_PATH as
before, and sets `OMPI_MCA_hwloc_base_binding_policy=none`. The prefix rule, not a
pattern list, is deliberate: the incident was a pattern list removing the wrong subset of
a half-armed environment.

**Every ORCA launch route goes through the seam**: the labels worker
(`frame_labels._run_orca`), every msRRHO level job (`orca._run_job`), the single-point /
composite launcher (`orca.single_point`), the branch-B opt+freq driver
(`scripts/production/s0_branch2_opt_freq.py`), the package2 high-level driver
(`scripts/calibration/s0_package2_highlevel_freq.py`) and the level benchmark under
`examples/02c_hessian_benchmark_levels/`. A unit test (`t_orca_child_env`) pins the child
environment offline and scans the live tree, so a Slurm variable cannot leak back in and
a new launcher that forgets the seam fails.

**The worker keeps its own Slurm view.** The strip is for ORCA children only: the frame
lock, the submit directory and the reporting still read `SLURM_JOB_ID`,
`SLURM_SUBMIT_DIR` and `SLURM_JOB_PARTITION` in the worker process.

**The job scripts' narrow unset stays as it is** (ruling Q4, 2026-09-25): it serves the
non-ORCA payloads' scheduling variables, and the ORCA invariant no longer depends on it.
One comment at the two step-03 scripts' loops points at the child seam so no reader
"repairs" the payload issue there.

**Placement's only owner is the worker's `taskset` range**, with OpenMPI's own binding
switched off through the ORCA-documented knob. `OMPI_MCA_rmaps_base_oversubscribe=1` is
the documented fallback only -- set if a "not enough slots" line ever appears -- and is
not set by default.

## Considered options

* **A correctly shaped Slurm allocation** (`--ntasks=16`, one task per frame, so every
  `SLURM_*` variable the allocator reads is present). Rejected: the ranks are 16
  independent `mpirun`s the job script starts, not tasks Slurm launches; shaping the
  allocation to pretend otherwise has Slurm bind and account processes it did not start,
  and `ras/slurm`'s managed path sets no-oversubscribe, so `mpirun -np 4` against a small
  task shape still fails after the first error is fixed (research note §2.7). It also
  keeps the ORCA child dependent on the allocation's exact shape -- the class of coupling
  that broke draw300.
* **`srun` steps**: launch each frame's ORCA under an `srun` step. Rejected: ORCA's
  driver must call its own `mpirun` (manual, boxed warning), and an `srun` step
  re-introduces exactly the step-scoped scheduler environment the child must not see
  (`SLURM_TASKS_PER_NODE`, `SLURM_PROCID`, `SLURMD_NODENAME`); it would also re-shape the
  frame-lock and pid model.
* **MCA component exclusions** (`OMPI_MCA_ras=^slurm`, and the same for `plm` and
  `ess`). Rejected: `SLURM_JOBID` is a three-way switch, so all three must be excluded
  and kept in step with the OpenMPI version (a future OpenMPI can add a fourth
  component); deleting the variable disarms every one of them in one stroke, from the
  same source logic (research note §2.1-2.5).
* **Leaving the job scripts' narrow unset as the fix** (widening it to the full
  `^(PMI|SLURM)_` filter). Rejected: the filter is the non-ORCA payloads' business, the
  worker's own Slurm reads would need captures, and the ORCA invariant would again depend
  on which shell happened to keep which variable -- the dependency that produced the
  incident.

## Consequences

* An ORCA child runs on the node it was started on (ORTE's local fallback) and is
  outside Slurm's binding and accounting; placement is the worker's `taskset` alone.
* The scheduler can no longer abort ORCA before it starts; the draw300 frames are
  recovered by the one-shot retry of orca-slurm ticket 02, after one frame is verified by
  hand on site.
* The policy is read from one function; the test suite fails offline if a Slurm variable
  can leak back or if a new launcher bypasses the seam.
* Not yet wired: the capability probe that runs `orca --version` to test installation
  (`capabilities._binary_runs("orca")`) still inherits the worker environment. It cannot
  reach the MPI-startup path the incident lived in, and it is listed in ticket 01 for
  disposition (a lazy seam import would route it).
* The hkuhpc bundle's `ORCA_SKIP_CPU_BIND` (not an ORCA-documented variable) is to be
  replaced with the documented knob when that file is next touched -- a separate cleanup,
  not this change.
* Evidence: `research-orca-slurm-primary-sources.md` (ORCA 6.1 manual §2.5, OpenMPI
  v4.1.8 source, slurm.schedmd.com); incident and rulings in
  `grilling-round-13-orca-parallel-framework.md` (2026-09-25).
