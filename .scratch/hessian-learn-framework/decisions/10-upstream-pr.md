# Upstream: submit the mace-side changes

Type: task
Status: resolved
Blocked by: 02
Part of: [hessian-learn-framework](../map.md)

## Question / work

Q4a: the six commits were written generic and upstreamable ("nothing openQHA-specific ever lands there"). Turn them into a PR against `ACEsuit/mace`.

1. From the rebuilt branch ([The real fork](02-the-real-fork.md)), prepare the PR's commit set: drop fork-only commits (e.g. the version bump `0ae78c4`; and a binaries-drop commit if one exists) or mark them clearly; keep the four feature commits A–D plus their tests.
2. Write the PR text: the problem (a per-structure Hessian label cannot travel mace's data path; no external-loss hook; multihead mode drops `--loss external`), what the commits do, and the evidence (the tests in the branch).
3. Check current `main` first for overlap — v0.3.16 had none of this, but `main` moves.
4. Acceptance: PR URL recorded, or a written decision (with the user) not to submit.

## Answer

**Decision (grilling 2026-09-29, with the user): do not submit upstream — the fork becomes a
permanent private fork.** This closes the ticket through its acceptance alternative (a written
decision not to submit). Findings that drove it:
[`research/mace-md-fork-practice.md`](../research/mace-md-fork-practice.md); durable record:
ADR 0013.

Why not submit:

- **The route this ticket assumed is closed for these files.** Upstream is mid-v1-rewrite; its
  current `CONTRIBUTING.md` (same text on `main` and `develop`) puts exactly our files on the
  "core and shared — v1 only" surface (`mace/data/atomic_data.py`, `mace/tools/arg_parser.py`,
  `mace/tools/train.py`), takes v0.3 features only with a v1-impossibility note plus a
  self-contained directory and number-pinned tests, and routes v0.3 fixes through `develop`. A PR
  against `main` is not a route (precedent: #1445 retargeted main→develop before merge); the
  realistic outcomes were "v1-only, please redo" or a close.
- **The reference private fork of the same trade shows the alternative works.** `mace-md`'s mace
  fork (`jharrymoore/mace@softcore`) is release-line based, frozen since 2024-05-01, referenced
  by bare branch name, and was never offered upstream; the author's merged upstream PRs are
  unrelated small fixes.
- **The project need is already served by the fork** (installs pin
  `BloomDlwlrma/mace@openqha-hessian`); the submission bought only optionality (upstream adoption
  → future maintenance relief) and a public record, at the cost of a rebase across ~124 commits /
  318 files of drift, scrubbing, and test-suite adaptation. The optionality was judged not worth
  the cost.

**Adopted posture (the reference way, with our stronger anchors):**

1. **Release-line base** — already held: upstream tag `v0.3.16` (`4d2da09`) + D + the six commits
   (a tag anchor, unlike the reference fork's non-tag base).
2. **Frozen** — no upstream tracking and no rebases; `openqha-hessian` @ `1110ffb` and
   `base-v0.3.16` @ `5c2d761` are the permanent anchors Records, installs and the old→new mapping
   rely on. A future base move is a new decision, not upkeep.
3. **Bare branch-name reference** — already the case (`pip install git+…@openqha-hessian`); the
   identity layer on top (`0.3.16+openqha`, `mace_fork_info()`, the mapping) is kept.
4. **Never upstream** — Q4's "organised as upstream-PR-able" is revised by this ticket. The branch
   stays exactly as written — no scrubbing, the two AI-attribution trailers included (upstream
   rules no longer apply), because SHA stability is what the Records rely on.

Consequences: the 02 rebuild's PR motive retires (its other gains — upstream-comparable history,
lean clones from D, the mapping — stand). Accepted costs: optionality foregone, no public review,
base-rot risk on a future MACE upgrade (a new decision then). All submission-path questions
(target branch `main`/`develop`, rebase, scrubbing, test adaptation, PR text, mechanics) are void.
The overlap check was run before deciding: no upstream match for `hessian_key` /
`hessian_weight` / `loss_module` / external loss (research note §3). Residual forward-looking PR
sentences in [02](02-the-real-fork.md) and ADR 0011 stay as history; the revision of record is
this Answer, the map, and ADR
[0013](../../../docs/adr/0013-permanent-private-mace-fork.md).
