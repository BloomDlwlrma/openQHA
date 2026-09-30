# Spec: the references sweep (ticket 12)

Ticket: [12 — References sweep: docs, slurm drift, tutorials](decisions/12-references-sweep.md). Drafted from the Q1–Q19 grilling rulings, approved 2026-09-30. Seams: none new (confirmed).
Slices: [12a](implementation/12a-the-slurm-drift.md) · [12b](implementation/12b-the-refusals-and-identity-text.md) · [12c](implementation/12c-the-public-documents.md) · [12d](implementation/12d-the-library-pass.md) · [12e](implementation/12e-the-tests-pass.md) · [12f](implementation/12f-the-scripts-and-root-pass.md) · [12g](implementation/12g-the-hpc-and-workflows-pass.md) · [12h](implementation/12h-the-ci-and-the-close-out.md) · [12i](implementation/12i-the-package-pass.md).

## Problem Statement

The repos are being readied for their final, public form, but the public face still speaks in the internal shorthand of the effort. Code comments and docstrings cite tickets, ruling dates and study codes ("Records redesign (user ruling 2026-09-15, tickets 16-18)"); user-visible strings — error messages, CLI help — carry the same codes; several documents still describe retired mechanics (the SHA-256/parameter fingerprint and the registry pin) or the wrong split story. Some of it is not merely confusing but broken: three Slurm preflights read a provenance key that no longer exists (a KeyError that kills the job early), two weights-check pipelines grep for output lines the tool no longer prints (their fallback warning fires on every job), and a failed Dataset build in the labels job exits 0 silently. The package's fork-guard refusal still names the retired fork checkout as the install. And the standing verification habit — run the full suite on one machine and quote the baseline — cannot survive the hand-off to a public repository.

## Solution

One sweep (ticket 12) that makes the public face self-contained and trustworthy:

- fixes the broken and stale references (Slurm preflights, weight-check pipelines, the labels job's exit code, the dataset and fork-guard refusals, stale checksum claims);
- rewrites comments, docstrings and user-visible strings across the public code to describe current behavior in plain language — no internal process citations, no retired mechanics described as current;
- aligns the public documents' stale narratives (split modes, weight identity, fork arrangement, install, the campaign's dataset guidance);
- updates the tutorials' source cells and remaps the test's fork-commit pin onto the rebuilt fork's history;
- adds a minimal CI workflow to openQHA that mirrors the package's — dependency-light, running a representative subset of the existing tests on push and pull request;
- records one final full-suite run as the retirement baseline of the old habit;
- records the public-face boundary (scratch trackers, ADRs, the context and agent documents are not public) and leaves the frozen mace fork untouched.

## User Stories

1. As a public reader of the openQHA source, I want comments and docstrings that describe what the code does today, so that I can follow it without the internal ticket archive.
2. As a user reading an error message, I want it to name a real, actionable remedy in plain words, so that I can act without decoding study codes.
3. As a user of the dataset CLI, I want its help text free of internal study codes, so that it reads like a normal tool.
4. As a reader of the Hessian-learning workflow README, I want the two split modes described correctly — whole-molecule split as the production default, per-frame split as the smoke/fit mode — so that I pick the right flag.
5. As a reader of the same README, I want the fraction defaults to match the code (5% in both modes), so that my expectations are right.
6. As a reader of the same README, I want the table's module addresses to point at the current package, so that the imports I copy work.
7. As an operator of the campaign, I want the campaign page's dataset guidance to match the dataset that was actually built (split by molecule), so that I do not rebuild it with the wrong mode.
8. As a reader of the campaign's log-expectation table, I want the sample "split by …" log line to show the production mode, so that I can recognise a correct build log.
9. As an operator submitting the trajectory and branch Slurm jobs, I want their preflight to print the engine name and the resolved weights path, so that the job does not die on a retired key.
10. As a branch-A job owner, I want the weights-check pipeline to match the check tool's current output, so that the fallback warning fires only when a weight file is actually missing.
11. As a user of the parked-rerun script, I want its warning text to name what it actually checks, so that the message is not misleading.
12. As an operator of the labels job, I want a failed Dataset build to make the job exit non-zero, so that a broken build cannot pass silently.
13. As an operator, I want the labels job to pass the dataset's split mode explicitly, so that the dataset it builds does not depend on a CLI default.
14. As a user hitting the dataset's split-change refusal, I want the message to offer remedies that exist (rebuild through the build API, or under a new name) rather than a flag the workflow deliberately does not expose, so that I can act.
15. As a user hitting the fork-guard refusal, I want the install line to point at the current install (the install script, or an editable checkout of the fork), so that I fix it right.
16. As a user running the fork guard from a directory that shadows the install (one containing a `mace/` subdirectory), I want the refusal to diagnose the shadowing and tell me to change directory, so that a false "unknown" is explained.
17. As a tutorial reader, I want T04/T05's source cells to import the current package and name the current checkout, so that the notebooks run against the tree as it is.
18. As a maintainer, I want the integration test's fork-commit pin mapped onto the rebuilt fork's history, so that the assertion still means what it says.
19. As a reader of the README's weight-register section, I want the retired fingerprint/pin narrative gone, so that the identity model is described as it now is (engine name + resolved path).
20. As a user of the dependency-check script, I want its claims to match what it does (imports, paths, presence), so that nothing promises hashing that no longer happens.
21. As a user reading the Slurm house README, I want its "SHA-256 before doing any work" line corrected, so that it matches the reduced check tool.
22. As a maintainer, I want the test fixture that pretends to be a mace checkout to stop reusing the retired checkout name, so that nothing in tests points readers at a dead path.
23. As a maintainer, I want the environment files' stale "hash-checked on load" comment corrected, so that the environment docs do not claim retired behaviour.
24. As a public reader of the package's source, I want its module docstrings to describe the modules' roles rather than the internal ticket that produced them, so that the package reads as a library.
25. As a user of the package, I want its user-visible error strings free of study codes, so that failures read in plain language.
26. As a maintainer, I want the tests that pin those strings updated in the same change, so that the contracts stay enforced.
27. As a maintainer, I want the comments across the public face reworded by one consistent rule — current behaviour, no internal citations, measurement dates kept — so that the result is uniform and reviewable.
28. As a developer, I want no behaviour change from the comment and document passes, so that the sweep is provably safe.
29. As a maintainer of the openQHA repository, I want a minimal CI workflow (like the package's) on push and pull request, so that changes are verified automatically.
30. As a maintainer, I want the CI to run a dependency-light, representative subset of the existing tests, so that real regressions are caught without heavy installs (no torch, mace, OpenMM, weights or network).
31. As the developer, I want a final full-suite run recorded in the ticket, so that the retirement of the baseline habit is documented.
32. As the person preparing the public release, I want the public-face boundary recorded (scratch trackers, ADRs, the context and agent documents are not public), so that nobody overshoots the cleanup or leaks the wrong files.
33. As a reviewer of git history, I want three focused commits — the sweep, the comment/string pass, and the CI — so that each concern is reviewable, accepting that historical commit messages keep their internal references.
34. As the maintainer of the mace fork, I want its history frozen — no comment edits, no tip moves — so that the identity anchors stay stable.
35. As an auditor, I want a residual-citation scan over the public face at the end, so that completeness is provable.

## Implementation Decisions

- **Public-face boundary.** Public = the repositories minus the scratch trackers, the ADRs, the context document and the agent document; only the public face is cleaned. The mace fork is frozen (ADR 0013) — its comments and messages are not touched. Two facts are reported, not acted on: the scratch tree is currently git-tracked; commit history carries internal ids.
- **Reference fixes.** Slurm preflights print the engine name and the resolved weights path (the retired digest fields are dropped). The weights-check sentinel becomes the check tool's weights-path line; the fallback wording names that. The labels job captures the Dataset build's exit status and fails the job when either it or the assemble fails; the split flag becomes explicit. The dataset's split-change refusal keeps the substrings its unit test pins and offers the two remedies that exist (the build API with a resplit, or a new dataset name) — the workflow deliberately exposes no resplit flag. The fork-guard refusal points at the install script / an editable fork checkout and adds a conditional note when the working directory contains a `mace/` subdirectory; its pinned substrings are preserved.
- **Narrative fixes.** The workflow README's split story, fractions, package addresses and fork naming are corrected; the campaign page's split narrative is aligned with the dataset actually built; the retired weight-identity narrative is removed from the README, the dependency-check script, the environment files and the Slurm README; the test-only mace-checkout fixture loses the retired name; a stale retry constant reference is recorded only (it has no live counterpart).
- **Comment and string rules** (one rule set, applied everywhere public): strip process tokens (ticket numbers, study codes, ruling dates, grilling/decision/round references, scratch paths); event sentences become current-state sentences; measured-fact dates are kept; user-visible strings and the tests pinning them change together; behaviour is unchanged; identifiers are untouched.
- **Tutorials.** Source cells updated (imports, checkout naming, loss address); the fork-commit pin remapped via the ADR 0011 old→new table; notebooks are not re-executed — stored outputs stay as historical runs.
- **Docs pass.** All public documents get the same treatment; content fixes and citation stripping are merged into a single editing pass per file; a per-file count table precedes the pass.
- **CI.** A new minimal workflow for the openQHA repository, shaped like the package's: ubuntu-latest, Python 3.11, triggers on push to main and pull requests; installs only the light dependencies (numpy, ase); runs eight existing unit tests directly (repository bootstrap, module entry points, store serialization, the QHA harmonic limit, mode matching, msrrho presets, record resume, the screen floor). Acceptance is the first green run, recorded in the ticket.
- **Baseline retirement.** One final full-suite run is recorded; thereafter the CI is the standing verification.
- **Delivery.** Three commits on openQHA (the sweep; the comment/string pass; the CI and the ticket record) and one on the package; both remotes pushed; the ticket record is the Answer + a map line + resolved status.

## Testing Decisions

- **What a good test is here:** external behaviour only. This work is comments, documents and a small set of message contracts — the existing pinned-string assertions are exactly the right seam for the message changes; no new seam is introduced.
- **Seams (confirmed):** none new. The verification surface is the existing suites (the runner's groups), plus one new repo-level seam — the CI workflow — plus mechanical checks (a compile pass over every touched file, and a residual-citation scan over the public face proving the process tokens are gone).
- **Which modules are exercised:** the dataset module's refusal path, the package's fork guard, the package's loss error string (all through their existing unit tests), and both suites once each — the openQHA suite as the retirement baseline, the package suite as the release check.
- **Prior art:** the package's minimal CI (install, then two unit tests); the package suite's pinned-string assertions; the runner's design of independent per-file test programs.

## Out of Scope

- The scratch trackers (both sets): no notes, no rewrites — history stays as written.
- ADRs, the context document and the agent document: not public, untouched.
- The mace fork: frozen — comments and commit messages untouched.
- Untracked archival directories; rewriting commit history.
- Re-running notebooks; widening the CI beyond the smoke subset; running the full unit group in CI; training or Tianhe jobs.

## Further Notes

- The canonical example of the comment rewrite: keep the description ("Records layout: no setting level under the records folder…"), drop the provenance parenthetical.
- The rerun script's fixed commit keeps its hash but gains a comment saying why it is fixed (the hash alone would be unexplainable).
- The ticket's Answer carries: the public-face boundary ruling, the two reported facts above, the first CI run URL, the final suite results, and the residual-scan result.
