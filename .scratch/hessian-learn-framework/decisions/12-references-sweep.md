# References sweep: docs, slurm drift, tutorials

Type: task
Status: resolved
Blocked by: 04, 05, 06, 11
Part of: [hessian-learn-framework](../map.md)
Spec: [spec-references-sweep.md](../spec-references-sweep.md)
Slices: [12a](../implementation/12a-the-slurm-drift.md) · [12b](../implementation/12b-the-refusals-and-identity-text.md) · [12c](../implementation/12c-the-public-documents.md) · [12d](../implementation/12d-the-library-pass.md) · [12e](../implementation/12e-the-tests-pass.md) · [12f](../implementation/12f-the-scripts-and-root-pass.md) · [12g](../implementation/12g-the-hpc-and-workflows-pass.md) · [12h](../implementation/12h-the-ci-and-the-close-out.md) · [12i](../implementation/12i-the-package-pass.md)

## Question / work

The mechanical sweep of everything that still names the old arrangement; the content decisions come from 04/05/06/07, this ticket lands them.

- `docs/tianhe_install.md` (fork note 284–291; install lines 277/331); openQHA `README.md` "The mace fork" (202–219); `check_dependency.py`; `environment-cuda.yml:160` stale comment; `docs/hessian_learning_campaign.md`; `workflows/hessian_learning/README.md` (its `RETRY_FAILED` line 77 is already stale — decide whether that belongs here or in the campaign page's own upkeep).
- Tutorials `T04`/`T05`: module imports and the fork-commit assertions (mapping from [Identity after the split](06-identity-after-the-split.md)).
- The three stale Slurm scripts (`hpc/slurm/branchA_debug.slurm:58–60`, `branchA_deimos.slurm:73–74`, `branchB_traj_tianhe_a.slurm:108–109`) reading `prov["sha256"]` keys that no longer exist — a KeyError today.
- `.scratch/hessian-learning-set` references that would mislead a reader (ticket 10/36 texts): a pointer note, not a rewrite — old tickets are history.
- Reported by [Round-1 xyz](08-round-1-xyz.md) when its CLI fix landed (2026-09-26):
  - `workflows/hessian_learning/README.md` step-04 section: "frame, the default (production)" / "molecule (the smoke set)" are backwards since S0-C-65 (and the "(0.1)" fraction claims).
  - `hpc/slurm/hl_labels.slurm` tail (task 0): the `04_dataset.py` call's exit code is never captured (`exit "$rc"` is the assemble's), so a failed Dataset build exits 0 silently; the call also leans on the CLI's `--split-by` default (correct now, but implicit).
  - `openqha/data/dataset.py:640`: the refusal message says "Pass resplit=True (04_dataset.py --resplit)" — the workflow deliberately exposes no such flag (ruling 2026-09-26); keep the substring "resplit" (`tests/unit/t_dataset.py:395` pins it) while re-wording.
- Flagged by [04a](../implementation/04a-strip-weight-identity.md)'s Answer (2026-09-27): `hpc/slurm/hl_branchA.slurm:63`, `hl_pipeline_debug.slurm:67` and `q5-rerun-parked.sh:420` grep the reduced `s0_check_weights.py` stdout for `registry pin` / `params sha`; those strings are gone, so the `|| echo "... weights check failed"` fallback now fires on every branch-A job — the tool prints `path` / `bytes` lines instead.
- Left by [01d](../implementation/01d-the-switchover.md): `openqha_hessian/run.py`'s `check_fork` refusal still says `pip install -e <path>/openQHA-Hessian` — the retired fork checkout's name (today that path is the package, not mace). Reword to the current install (`openQHA-Hessian/install.sh`, or `pip install -e <path>/mace` for the fork checkout), and while rewording add the cwd self-diagnosis (grilling 2026-09-29): from a directory that contains a `mace/` subdirectory (the workspace root) the editable install is shadowed and this same check reads a false `unknown` — tell the reader to `cd` into a repository root first.

## Answer

**The sweep.** Eight passes (everything in 12a–12i but this close-out) each applied one
rule set: every public-facing comment, docstring, user-visible string and document now
describes current behaviour — process tokens (ticket numbers, study codes, ruling dates,
grilling/decision/round references, scratch paths) out; event sentences read as
current-state sentences; measured dates kept; user-visible strings and the tests pinning
them changed together; behaviour untouched. The content fixes rode the same passes: the
three Slurm preflights print the engine name and the resolved weights path; the
weights-check pipelines watch the check tool's current output; the labels job fails on a
failed Dataset build and names its split mode; the Dataset and fork-guard refusals offer
remedies that exist; the workflow README, the campaign page and the tutorials' source
cells match the code and the build actually run.

**The public-face boundary.** Public = the two repositories minus the scratch trackers,
the ADRs, `CONTEXT.md` and `AGENTS.md`; only the public face was cleaned. The mace fork
is frozen — its comments, messages and anchors untouched (`1110ffb` / `5c2d761` verified
at the close-out). Two exclusions are explicit in the residual scan's scope:
`docs/tutorials/archive/T03` (an archived, superseded tutorial kept as history) and the
notebooks' stored outputs (historical runs; their cell sources are clean).

**Two reported facts (not acted on):** the scratch tree is git-tracked (180 files ship in
the repository tree); the commit history keeps its internal ids — historical messages are
not rewritten.

**The CI and its first run.** `openQHA/.github/workflows/ci.yml` — ubuntu-latest, Python
3.11, numpy, ase and PyYAML (the clean-env dry-run found `t_msrrho_presets` needs the
repository's YAML config — one light dependency more than the plan assumed), the eight
smoke tests run directly (repository bootstrap, module entry points, the store TOML
round-trip, the QHA harmonic limit, mode matching, the msrrho presets, the md
record-resume read, the Hessian screen floor). A clean-checkout dry-run of the eight
passed before the push; the first CI run is green (https://github.com/BloomDlwlrma/openQHA/actions/runs/36665595358).

**The retirement baseline.** The last full-suite runs: openQHA `tests/run_tests.py --all`
**73/73**, the package `tests/run_tests.py --all` **9/9** (2026-09-30, the `openqha`
env). Thereafter the CI is the standing verification.

**The residual scans.** One scan over both repositories' public faces — run on a clean
clone of the close-out tree with the citation-file comment fixes applied (510 files
scanned, 193 excluded; families: ticket, ruling, grilling, ADR, CONTEXT.md, .scratch,
S0/D0 codes, plan files, set names, spec names, numbered decision/checkpoint/round
references; notebook sources only): **0 violations** and 143 kept-class lines, each with
a recorded reason (the old-assets sample records, the artifact index's / the baseline's /
the store's decision values, the mem-decide namespace, the decision-format grammar and
its API strings, the functional context-document reads, the campaign's own round
vocabulary). The scan's wider file-type net also caught two comment leftovers in
`docs/cite/cite_openQHA.bib` (a study code and a ticket range) — fixed in this close-out.

**Delivery shape.** The spec's "three commits and one on the package" materialised finer:
one commit per slice plus its two-axis review annotations (as separate commits), the
package pass as `cc183eb` + `669f823`, and the close-out as `f368bd6` (the CI) plus the
commit this record rides. Both remotes carry the tip; the first CI run is green
(https://github.com/BloomDlwlrma/openQHA/actions/runs/36665595358).

**Reported per the operating rule.** Files changed: `.github/workflows/ci.yml` (new),
`docs/cite/cite_openQHA.bib` (two comments), and the ticket files this close-out edits.
Checks run: the workflow's YAML parse; the eight smoke tests in a clean clone under a
scratch env; both suites (`--all` 73/73 and 9/9); the residual scans (0 violations); the
`mace` freeze check; the first remote CI run (green). Not verified: the `pull_request`
trigger path (no PR existed to exercise it); the dry-run's Python was conda 3.11, not the
runner image — the remote run is the authoritative check.

**Recorded, not fixed:** the `RETRY_FAILED` stale reference has no live counterpart (the
retry contract reads `RETRY_ONLY`); the `.scratch/hessian-learning-set` pointer note from
the work list is superseded by the boundary ruling (scratch trackers take no notes, no
rewrites).
