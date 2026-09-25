# The ORCA child environment is Slurm-blind, and a failed frame is retried at most once

Status: ready-for-agent
Date: 2026-09-25
Vocabulary: CONTEXT.md (Frame set, Batch, Calculation; the reference levels of `orca.LEVELS`).
Decisions: ADR 0007 (to be written with ticket A: ORCA children are Slurm-blind; the worker owns placement).
Grilling: `grilling-round-13-orca-parallel-framework.md` (Q1-Q7, Q5a-Q5d; rulings 2026-09-25).
Evidence: `research-orca-slurm-primary-sources.md` (ORCA 6.1 manual, OpenMPI v4.1.8 source, slurm.schedmd.com).

## Problem Statement

A full labels round on TianheXY-CN (`TAG=draw300`, twelve array tasks, three days of
queue) produced **zero labels**. Every frame's ORCA died within seconds of starting; the
worker log repeats the one line

    dsgdb9nsd_006885 displaced_b01_k3 FAILED RuntimeError:   .... aborting the run

The frame's own ORCA output says why: ORCA's bundled OpenMPI picked a Slurm resource
allocator (the `SLURM_JOBID` the job leaves in the environment arms it), that allocator
demanded `SLURM_TASKS_PER_NODE` and could not find it (the job script's unset loop had
removed exactly that variable while keeping the ones that arm the allocator), and
`ras_base_allocate` treats the mismatch as fatal — before any chemistry. Seconds per
frame, no label written, and every frame the round touched is now in the **failed** state.

The failed state is a dead end by design (ticket 24: a frame is attempted once; a failed
frame is not rerun by any round or driver; `--retry` was the human's lever). Two further
consequences make the situation worse than it looks:

- **The round signal lies.** The list of pending work skips failed frames, and the
  assemble exit code counts only *unlabelled* frames — so resubmitting the round as it
  stands reports "nothing pending" and can exit 0 while thousands of frames are failed.
- **There is no safe way to recover the mass.** Rerunning the failures needs a deliberate,
  bounded lever; and any rerun must not (a) destroy the failed output — today a retry
  overwrites it — or (b) loop: a frame that fails again must never be re-selected by a
  later round.

Underneath both: the parallel-computation policy is undocumented. Nothing states that an
ORCA subprocess must never see the scheduler's environment, that CPU placement belongs to
the worker's `taskset` alone, or that OpenMPI's own binding must stay off — so the next
`hl_*` script or ORCA launcher can quietly re-import the same defect.

## Solution

ORCA runs **blind to Slurm**: the ORCA child environment is built in one place and carries
no `SLURM*`/`PMI*` variable at all, while the worker process keeps its own Slurm view (frame
lock owner, submit directory, reporting). CPU placement has exactly one owner — the
worker's `taskset` range — with OpenMPI's own binding switched off through the
ORCA-documented MCA knob. Every ORCA launch route goes through that one seam, and a unit
test pins it offline.

The mass-failed frames are recovered by **one deliberate, once-only retry round**: a new
`--retry-failed` flag makes the ordinary array/worker round select frames whose `.out`
lacks the terminal line and for which no archive exists; before ORCA runs, the previous
failed `.out` is saved beside the file group as `<stem>.failed.out`; successes are skipped
exactly as today. After that single retry the frame is **final** whatever the outcome — a
retry that fails stays failed and is never selected again, so no round can loop. Nothing
deletes evidence, the campaign page's round procedure stays valid, and the incident's
outputs are preserved on disk.

The policy is written down (ADR 0007 plus the HPC README and the ORCA env script header):
the job script's narrow unset serves the non-ORCA payloads; the ORCA child's full strip
lives in the child-environment seam.

## User Stories

1. As the campaign operator, I want ORCA subprocesses to see no Slurm variables, so that a
   scheduler environment can never abort an ORCA job before it starts.
2. As the campaign operator, I want the whole site's ORCA work fixed by one seam, so that
   the labels round, the msRRHO level jobs and the parsl worker all get the fix without
   route-specific patches.
3. As the campaign operator, I want OpenMPI's own CPU binding switched off, so that the
   worker's `taskset` range is the only thing deciding where the four ranks run.
4. As a developer, I want the ORCA child to keep its PATH and library shim (the shared
   build), so that hiding Slurm does not break finding `mpirun` or the ORCA libraries.
5. As a developer, I want a unit test on the child environment, so that a future edit that
   lets a Slurm variable back in fails offline.
6. As a developer, I want every ORCA invocation — including the single-point launcher and
   the direct scripts — to use that one child environment, so that no route is exempt.
7. As the next reader, I want an ADR that says why the scheduler is unset inside a
   scheduler job, so that the decision is not re-litigated or silently undone.
8. As the campaign operator, I want one deliberate flag (`--retry-failed`) that
   re-attempts failed frames, so that the mass failure from this incident can be recovered
   without ad-hoc commands.
9. As the campaign operator, I want each failed frame retried at most once, so that no
   round or driver can ever loop on a permanently broken frame.
10. As the campaign operator, I want the previous failed ORCA output archived before it is
    overwritten, so that the evidence of the failure survives the retry.
11. As the campaign operator, I want finished frames skipped exactly as today, so that a
    retry round never touches labelled work.
12. As the campaign operator, I want a retried frame that fails to stay failed and be
    listed, so that the remaining failures are a short list for a human.
13. As the campaign operator, I want the retry to run through the ordinary array/worker
    path, so that the campaign page's round procedure stays valid and the locks keep
    partitioning the work.
14. As the progress reader, I want failed frames and their archived outputs to be visible
    on disk, so that the progress table and the per-molecule Record can describe the state
    without a database.
15. As a reviewer, I want "failed" to remain different from "cut", so that walltime losses
    are still rerun whole and the retry cap counts only real failures.
16. As the campaign operator, I want a `--force` switch for the deliberate "re-run all
    ORCA labels" operation, kept off by default, so that the extraordinary operation exists
    without becoming routine.
17. As a developer, I want the retry semantics locked by tests at the frame-labels seam, so
    that "archive, then run, once" cannot regress.
18. As a developer, I want the task list to carry the retry intent to the worker, so that
    the same `xargs` dispatch machinery runs retries unchanged.
19. As the site operator, I want the fix verified on one frame before any mass retry round,
    so that no frame's single retry is burned on an unverified fix.
20. As the campaign operator, I want the campaign page and the HPC README to state the two
    environment layers and the retry rule, so that operators do not "repair" the narrow
    loop or invent their own retries.
21. As the maintainer of the other payloads, I want the CREST/MACE/training scripts
    untouched by this change, so that working stages are not disturbed.
22. As the user, I want the incident, its evidence and its tickets to live in one tracker
    (`.scratch/orca-slurm/`), so that the story is found in one place.

## Implementation Decisions

### The child environment (ticket A)

- The ORCA interface module's `subprocess_env()` becomes the single enforcement point. It
  builds the ORCA subprocess environment and now:
  - deletes every variable whose **name starts with `SLURM` or `PMI`** (a prefix rule, not
    a pattern list — the incident was caused by a pattern list removing the wrong subset;
    this covers `SLURM_JOBID`, `SLURM_NODELIST`, `SLURM_TASKS_PER_NODE`, `SLURMD_NODENAME`,
    `PMI_FD`, `PMIX_*`);
  - keeps `S0_ORCA_PATH` / `S0_ORCA_LIB` and everything else, and still prepends them to
    PATH / LD_LIBRARY_PATH as today;
  - sets `OMPI_MCA_hwloc_base_binding_policy=none`, the ORCA-documented binding
    off-switch, so placement's only owner is the worker's `taskset`.
- Rationale (source-cited in the research note): in OpenMPI 4.1, `SLURM_JOBID` alone arms
  `ras/slurm`, `plm/slurm` and `ess/slurm`; with none of them selected, `ras` falls back to
  the local host and `mpirun -np 4` runs against the pinned cores.
- `single_point()` (today it passes no environment at all, so it would not even receive
  the site's ORCA paths) and the two direct ORCA launchers in `scripts/` are routed
  through `subprocess_env()` in the same ticket.
- `OMPI_MCA_rmaps_base_oversubscribe=1` is documented as the **fallback** only, to be set
  if an "not enough slots" line ever appears; it is not set by default.
- The nine `hl_*`/`branch*` sbatch scripts' narrow unset filter is **unchanged** (ruling
  Q4): it serves the non-ORCA payloads' scheduling variables, and the ORCA invariant no
  longer depends on it. One comment at the labels script's loop (and its twin in the
  debug-pipeline script) points at the child seam so no reader "repairs" the payload issue
  there.
- No behaviour change for CREST/MACE/torch payloads.

### The retry round (ticket B)

- The labels driver gains `--retry-failed`. When set, its task list additionally includes
  every failed frame that has **no archive**: `pending` today skips finished and failed
  frames; the flag re-includes failed frames whose file group carries no
  `<stem>.failed.out`.
- The task list carries the per-frame retry intent to the worker; the worker maps it to
  the frame CLI's existing `--retry` (`label_one(retry=True)` already exists). The
  `xargs`/lock/assemble machinery is otherwise untouched.
- **Archive before overwrite.** On a retry run, if the stem's `.out` exists without the
  terminal line, it is renamed to `<stem>.failed.out` **before ORCA starts**; the slot is
  replaced each time it is written (one slot per frame, per ruling Q5a). A finished frame
  is never archived or touched.
- **Once-only.** The archive is the durable marker: a frame is retryable iff its `.out`
  lacks the terminal line AND no `<stem>.failed.out` exists. After the single retry the
  frame is final whatever the outcome:

      failed + no archive   --(--retry-failed)-->  archive .out -> .failed.out; ORCA runs
        success -> finished (terminal line; skipped forever after)
        failure -> failed (archive present; never selected again)
        cut     -> never run (no .out; ordinary policy reruns it whole; the cap counts
                   failures, not cuts)

- A cut retry leaves the archive untouched and the frame reads "never run" for the
  ordinary policy — unchanged from ticket 24's cut rule.
- The archive file sits beside the file group and is inert: `finished()`/`failed()` and the
  progress walk read the exact `<stem>.out`, so no parser, the Dataset and the progress
  table need changes. No Record schema change; failed frames keep `STATUS=failed` and the
  reason naming the ORCA output.
- A `--force` switch on the frame CLI bypasses the finished/failed skip for the deliberate
  "re-run all ORCA labels" operation only; **off by default**, never wired into a round or
  the driver. `--retry-failed` remains the only retry lever.

### Documentation and ADR

- ADR 0007: "ORCA children are Slurm-blind; the worker owns placement" — the surprising
  question it answers ("why is the scheduler unset inside a scheduler job?") is exactly
  what a future reader will ask; the alternatives (Slurm-shaped allocation, `srun` steps,
  MCA exclusions) were on the table and rejected for source-cited reasons.
- `hpc/slurm/README.md` rule 2, the ORCA env script header and the labels script comment
  state the two layers: the job scripts remove scheduling variables for their payloads;
  the ORCA child's full strip lives in `subprocess_env()`.
- The campaign page's §3/§4 (frame states, retry) state the once-only retry rule and the
  sequencing below.
- No CONTEXT.md change (implementation policy, not domain vocabulary).

### Tickets (for `to-tickets`)

Two tickets, per ruling Q7: **A — the ORCA child environment** (strip + binding knob +
routing + unit test + ADR/README), **B — the retry round** (`--retry-failed` + archive +
once-only + tests + docs). B depends on A only for the sequencing of live operation, not
for code.

## Testing Decisions

Good tests here assert **external behaviour at existing seams**: a fabricated environment
in, a stripped child environment out; files on disk in, selection or archive out. They run
offline and deterministically, and none of them needs ORCA or Slurm.

- **Seam A — the child environment.** A unit test calls `subprocess_env()` with a
  fabricated environment containing `SLURM_JOBID`, `SLURM_NODELIST`,
  `SLURM_TASKS_PER_NODE`, `SLURMD_NODENAME`, `PMI_FD`, `PMIX_X`, `S0_ORCA_PATH` and
  `S0_ORCA_LIB`; it asserts no `SLURM*`/`PMI*` survives, the two shims survive (prepended,
  in order), and `OMPI_MCA_hwloc_base_binding_policy=none` is present. New test file (the
  existing ORCA-jobs test is being edited by another workstream — do not touch it).
- **Seam B — the frame's retry semantics.** Extend the frame-labels unit test, which
  already injects a fake `runner` into `label_one`: (i) a failed frame + retry → the
  archive exists while the fake runner is "running", and the new files land after; (ii) a
  retry when an archive already exists → refused, `status=failed`, no run; (iii) a failed
  retry → archive stays, still failed; (iv) a finished frame + retry → skipped, no archive.
- **Seam C — the round's selection.** A listing test on a fake tree: `--retry-failed`
  includes failed+no-archive, excludes failed+archive, excludes finished and never-run,
  and the counts reported match. Prior art: the frame-labels `pending` tests and the
  campaign test's fake-tree pattern.
- Prior art to follow: the frame-labels unit test (fake `runner`, fake `squeue`, file
  fixtures) and the campaign test (a small fake molecule tree with counts).

## Out of Scope

- MCA component exclusions (`OMPI_MCA_ras=^slurm`, …); the 16×4 allocation reshape
  (`--ntasks=16`); any `srun`-step-based rewrite (ORCA's driver must own its `mpirun`).
- The CREST/MACE/training payloads and the other sbatch scripts' unset filter.
- The hkuhpc bundle's undocumented `ORCA_SKIP_CPU_BIND` (a separate cleanup when that file
  is next touched).
- A bulk snapshot/export of the incident's failed outputs (ruled unnecessary; in-place
  preservation plus the archive is the save).
- Relabelling finished frames outside the deliberate `--force` operation.
- Anything msRRHO-specific beyond the shared child-environment seam.

## Further Notes

- **Sequencing is a rule, not a preference.** With the once-only cap, running
  `--retry-failed` on unfixed code would burn every frame's single shot: deploy ticket A →
  count the failures (`s0_hl_progress`) → verify **one** frame by hand on Tianhe → only
  then run the retry round.
- The incident's failed outputs stay in place until the retry archives them; one
  molecule's full file set and the failure signature are already preserved outside the
  tree (the `dbg/` copies and the research note).
- Open questions carried from the research note: which `ess` component the real launches
  select was not measured; why the 2026-09-19 smoke run passed is a reconstruction; the
  ORCA forum was unreachable, so `ORCA_SKIP_CPU_BIND` remains unverified as an ORCA
  variable.
- Measured, recorded, not relied upon: the ORTE components key on the legacy `SLURM_JOBID`
  while the frame lock reads the modern `SLURM_JOB_ID` — a future maintainer could exploit
  the split, but the policy deliberately does not depend on it.
