# Install and transport: the three-artifact stack, mace-md style

Type: task
Status: resolved
Serves: 05
Part of: [hessian-learn-framework](../map.md)

> The `/to-spec` output for decision [Install and transport](../decisions/05-install-and-transport.md)
> — the brief [05b](05b-the-install-script.md), [05c](05c-the-environment-files.md),
> [05d](05d-the-tianhe-path.md) and [05e](05e-fresh-env-acceptance.md) execute against.
> Per the effort's conventions in `docs/agents/issue-tracker.md` (Wayfinding operations),
> wayfinder tickets carry no `ready-for-*` triage labels; readiness is `Status: open`.

## Problem Statement

A machine that must fine-tune MACE-OFF23 on Hessian Labels needs three things pip cannot resolve on
its own — the openQHA checkout (a library, not a distribution), the mace fork (`BloomDlwlrma/mace`,
branch `openqha-hessian`; the only mace the training side accepts), and the `openqha-hessian`
package. Today the fork install is a line of prose, no script exists in the package repository, the
environment files still name the PyPI wheel as if the fork did not exist, the Tianhe transfer tool
does not actually carry the package although the documentation says it does, and no documented
sequence puts a fresh environment into the state where the training fork check passes.

## Solution

One install script in the package repository (name fixed: `install.sh`), in the pattern of the
companion project mace-md: it installs the fork from its branch URL (or from a local checkout path
on machines with no GitHub access), replaces any `mace-torch` wheel with the editable fork,
editable-installs the package itself, and verifies its own work. The environment and requirements
files stop naming the wheel and depend on the fork by git URL. On Tianhe the transfer tool carries
the two checkouts beside the repository, and the environment-install job installs both artifacts
into every environment it manages — no outbound network needed where the jobs run. No wheels are
built, transported, or kept as a fallback; the documentation tells exactly this story.

## User Stories

1. As a developer setting up a fresh WSL environment, I want one documented sequence of commands to end with the editable fork and the package installed, so that training readiness never depends on hand-assembled steps.
2. As a developer, I want the install script to replace any installed PyPI wheel of mace, so that a stale `mace-torch` can never shadow the fork.
3. As a developer, I want the script to verify its own end state (import location, version, `.git` beside the package, `mace_fork_info()`), so that a broken install fails at install time, not mid-training.
4. As a developer, I want re-running the script to be safe, so that updating a machine is the same single command.
5. As a developer, I want the fork source to default to the branch URL, so that no commit pin ever rots and updates are simply re-runs.
6. As a developer on a machine with no GitHub access, I want to hand the script a local checkout path instead, so that the same recipe installs without network.
7. As a developer, I want the script to validate a local path before touching the environment, so that a bad path cannot leave the environment with no mace at all.
8. As a trainer, I want the installed fork to answer `mace_fork_info()` with a clean commit, so that `check_fork(strict=True)` passes and every run is traceable.
9. As a maintainer, I want exactly one install recipe shared by the docs, the Tianhe installer, and the local flow, so that script and documentation cannot drift apart.
10. As the Tianhe operator, I want the transfer tool to carry the fork and package checkouts (`.git` included) beside the repository, so that the AI-side install needs no outbound network.
11. As the Tianhe operator, I want the environment-install job to install both artifacts into every environment it manages, so that "which environment can train" is never a question.
12. As the Tianhe operator, I want a clear skip message when the checkouts are absent, so that eval-only builds still succeed and the missing training stack is explicit.
13. As an environment builder, I want the environment and requirements files to depend on the fork by git URL instead of the wheel, so that building an environment never installs the wrong mace.
14. As a pure-eval user, I want one stated sentence that non-training work tolerates the wheel (provenance recorded as `unknown`), so that I don't chase a fork installation I don't need.
15. As a future initial user of the package, I want README-level instructions in the mace-md style (pip lines, defaults, verification), so that setup is trivial.
16. As a Record consumer, I want the fork commit recorded per run, so that any model traces back to the code that produced it.
17. As a documentation reader, I want the Tianhe install document to match the transfer tool's actual behavior, so that the documented path is executable as written.
18. As a maintainer, I want no wheels or release artifacts for this stack yet, so that there is one story and publication is owned by its own ticket.
19. As a maintainer, I want the old world's pins and comments (wheel floors, version notes) rewritten, so that no file still names the wheel as how a machine gets mace.
20. As a developer, I want the offline constraint stated once (installation happens at a networked moment; compute nodes have none), so that nobody designs a download into a compute job.

## Implementation Decisions

1. **Deliverable.** A single install script at the package repository root, named `install.sh` (name
   fixed by the package-line decision). Contract: it runs inside an already-created, activated
   environment; it acts on that environment through `python -m pip`; it never creates environments
   and never touches openQHA's importability.
2. **Fork source.** Default `git+https://github.com/BloomDlwlrma/mace.git@openqha-hessian`; an
   optional local checkout path argument replaces it (the offline / Tianhe mode). The script does no
   cloning of its own — pip's editable VCS install performs a blobless partial clone (default on
   git ≥ 2.17) and keeps the checkout with `.git` in place, which is what the provenance contract
   needs.
3. **Validation before mutation.** A local path must exist and be a git checkout whose root carries
   the mace package (so `mace_fork_info()`'s ".git beside the package" contract will hold). Failures
   abort before any uninstall/install.
4. **Steps, in order.** Uninstall any `mace-torch` distribution; editable-install the fork;
   editable-install the package (the script resolves its own repository root); verify.
5. **Verification block** (non-zero exit on any failure): `import mace` resolves to a path whose
   package directory has `.git` one level up; `mace.__version__` is `0.3.16+openqha`; the checkout
   is tracked-clean; when the openQHA checkout is importable, `mace_fork_info()` returns a full
   commit; the fork and package commits are printed.
6. **Editable for both artifacts, always.** There is no separate "debug" variant of the script: the
   local-path mode of the same script is the developer/offline mode. No `--no-deps` (mace-md
   precedent: the environment owns the dependencies; anything missing is filled by the same pip
   run).
7. **Ref form.** Branch, never a commit pin; the run-time Record (via `mace_fork_info()`) is the
   reproducibility handle. Docs may mention the tip at time of writing as orientation only.
8. **Environment / requirements files.** The `mace-torch` pins become a bare git-URL pip requirement
   for the fork (mace-md style). Non-editable by design: the env files guarantee "import mace is the
   fork" for eval; machines that train run `install.sh` (editable) regardless. Comments naming the
   wheel or version oddities are rewritten.
9. **Dependency-installer housekeeping.** The installer script's explicit mace pip line is dropped
   (the environment files now carry the fork; re-installing the wheel there would overwrite it);
   the other packages on that line stay. It prints a one-line pointer to `install.sh` for machines
   that train; it never invokes it (users of other campaigns must not be silently switched).
10. **Machine policy (the training-vs-eval question, settled).** The editable fork + package are
    required exactly where the training side runs (`05_train`, the training SLURM job, `judge`);
    pure-eval machines may keep the PyPI wheel, with provenance recorded as `unknown` (already
    tolerated outside training). On machines that carry the checkouts, every managed environment
    gets both artifacts.
11. **Tianhe environment-install §9.** Rewritten to install BOTH artifacts per managed environment
    by delegating to the package's `install.sh` in local-path mode (run through the environment's
    own python). `MACE_FORK` keeps its name and now points at the mace checkout (default: sibling
    of the repository); a sibling variable names the package checkout. Absent checkouts produce the
    explicit skip message (the wheel path remains for eval).
12. **Transport.** The transfer tool's push list gains the two checkouts (sibling layout) plus the
    workflows directory (and docs) — `.git` must ride along (no exclusion may touch it). The Tianhe
    install document is corrected so its "the checkout travels beside the repository" claim is true
    of the tool.
13. **Offline statement, written once into the docs.** Installation happens at a networked moment
    (WSL; the Tianhe login side with its proxy); the AI side installs from the carried checkouts;
    the local-path argument is the hedge when GitHub is unreachable.
14. **No wheels.** Nothing is built, shipped, or kept as a fallback; release artifacts for the
    package are deferred to the publication ticket.
15. **Docs surface owned here.** The Tianhe install document's fork/install block (including its
    stale old-fork narrative), openQHA's README fork section, §9 comments, the dependency
    installer's messages, the transfer tool's usage text, and the package's own README section for
    initial users.

## Testing Decisions

- **What makes a good test here:** external behavior at the highest point — the state of a fresh
  environment after the documented lines — asserted through pre-existing interfaces, never through
  the script's internals.
- **The single seam:** the install entry point, exercised black-box in a scratch environment. The
  assertions are the pre-existing probes: `import mace` + `engine.mace_fork_info()` (clean full
  commit) and `run.check_fork(strict=True)` (passes), plus `import openqha_hessian`. This seam is
  the ticket's acceptance sentence verbatim; no new seam is introduced.
- **The script's verification block is the same seam made executable:** running the script IS the
  test; its exit code is the verdict.
- **Negative cases, same seam, cheap:** a local path that is not a git checkout, and a missing path
  — non-zero exit, environment untouched (mace import state unchanged).
- **Prior art:** the training SLURM job's pre-flight block (imports the stack, calls
  `check_fork(strict=True)`, prints the fork identity); the package's test runner, which prints the
  provenance of the three artifacts; the weights check script, which prints the fork identity and
  explains `unknown`.
- **Deliberately not done:** unit tests around pip operations (a suite must not mutate the active
  environment); a Tianhe execution (decision + docs are in scope, the run is manual and later); new
  test infrastructure.
- **Acceptance run:** once, in a fresh WSL environment, recorded (commands + outputs) on the ticket.

## Out of Scope

- Publication artifacts (wheels, GitHub Releases) and the upstream PR cycle.
- Fixing the stale loss-module address mentions and retired sha256 keys spotted in scripts and docs
  (reported to the map; their owners differ).
- Executing an install on the Tianhe AI side.
- Changes to `mace_fork_info()` / `check_fork()` contracts (ticket 06's; unchanged under this
  design).
- Anything about the campaign beyond round 1.

## Further Notes

- **Findings with primary-source citations:** the mace-md install-pattern research note in this
  effort's research folder — pip's editable VCS clone location and blobless partial clone verified
  by `GIT_TRACE`; conda pip sections are requirements files (so `-e` entries are legal); mace-md's
  own script and env file quoted verbatim.
- **Done means:** the ticket's acceptance — a fresh WSL env from the documented lines ends with
  `import mace` resolving to the fork checkout, `mace_fork_info()` clean, the fork check passing;
  script and documentation agree; the Tianhe path is decided with the offline constraint stated.
- **Watch-list (unverified):** GitHub reachability from the Tianhe login node through the site
  proxy (if it fails, the Tianhe environment files fall back to the wheel line and §9's
  carried-checkout install does the rest — a one-line change); the clone location for editable VCS
  installs inside a conda environment (pip's documented default is the environment's `src`; only
  the location is affected, never the provenance contract).
- **Cross-ticket:** ticket 06 can take "no change" for its `mace_fork_info()`-contract question
  under this design.
- **Facts the implementation leans on:** the WSL environment's pip is 26.2.1 (accepts the direct
  `-e git+…@branch` form; older pips need the `#egg=` spelling); pip's partial-clone behavior needs
  git ≥ 2.17.

## Answer

Resolved as the brief, 2026-09-29: the design was settled in the grilling round (the mace-md
simplification — git URLs, editable where provenance needs it, no wheels) and this spec was
approved with granularity; execution is sliced as [05b](05b-the-install-script.md),
[05c](05c-the-environment-files.md), [05d](05d-the-tianhe-path.md) and
[05e](05e-fresh-env-acceptance.md). The findings it leans on: [the mace-md install
pattern](../research/mace-md-install-pattern.md).
