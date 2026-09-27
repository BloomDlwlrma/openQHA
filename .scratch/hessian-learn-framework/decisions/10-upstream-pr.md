# Upstream: submit the mace-side changes

Type: task
Status: open
Blocked by: 02
Part of: [hessian-learn-framework](../map.md)

## Question / work

Q4a: the six commits were written generic and upstreamable ("nothing openQHA-specific ever lands there"). Turn them into a PR against `ACEsuit/mace`.

1. From the rebuilt branch ([The real fork](02-the-real-fork.md)), prepare the PR's commit set: drop fork-only commits (e.g. the version bump `0ae78c4`; and a binaries-drop commit if one exists) or mark them clearly; keep the four feature commits A–D plus their tests.
2. Write the PR text: the problem (a per-structure Hessian label cannot travel mace's data path; no external-loss hook; multihead mode drops `--loss external`), what the commits do, and the evidence (the tests in the branch).
3. Check current `main` first for overlap — v0.3.16 had none of this, but `main` moves.
4. Acceptance: PR URL recorded, or a written decision (with the user) not to submit.

## Answer

<!-- resolver: append the PR URL or the decision; set Status: resolved; add a line to the map's Decisions so far -->
