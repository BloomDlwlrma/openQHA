# 02d: Clone, verify, re-home — the fork's working home and the old checkout's exit

Type: task
Status: resolved
Serves: 02
Blocked by: 02c
Part of: [hessian-learn-framework](../map.md)

> Execution slice for [The real fork](../decisions/02-the-real-fork.md); the brief is
> [02a](02a-the-mace-fork.md). Agent-run. Wayfinder conventions apply: the `Status`
> protocol, no triage labels (`docs/agents/issue-tracker.md`, ADR 0009).

## What to build

A fresh blob-filtered clone of the fork lands as the sibling `mace/` checkout, and that
clone demonstrates the acceptance: editable install, version `0.3.16+openqha`, and a
clean fork-info reading. The WSL `openqha` env resolves its `mace` import to the new
checkout, and the old `openQHA-Hessian/` checkout retires to `_to_delete/` — leaving the
`openQHA-Hessian/` path clear for the package.

## Acceptance

- [x] The fresh clone installs editable; `mace.__version__` is `0.3.16+openqha`; `openqha.potentials.engine.mace_fork_info()` reports a clean 40-hex commit (not dirty), read from that clone.
- [x] `mace/` is that clone: branch `openqha-hessian` tracking the fork; an `upstream` remote pinned to the needed tag refspec only (no upstream mass fetch).
- [x] The WSL `openqha` env's imported `mace` resolves to the `mace/` checkout; the obligations handed to ticket 05 are written down (the install/transport documentation).
- [x] The old checkout is archived at `_to_delete/openQHA-Hessian-old-fork-2026-09-26/`; the `openQHA-Hessian/` path is clear.
- [x] Reported: clone commit, install evidence, the old→new SHA table (ticket 06's mapping material), anything unverified.

> Until this lands, the WSL env's `mace` still resolves to the old checkout path; the
> refresh here keeps the env importable after the move.

## Answer

Resolved 2026-09-27, agent-run in the WSL `openqha` env.

- **Clone.** Fresh blob-filtered clone of `https://github.com/BloomDlwlrma/mace.git` at
  the sibling `mace/` (`git clone --filter=blob:none --branch openqha-hessian`). HEAD =
  `1110ffbafd651a74d1d4678deb4748056d1dff0a` (the fork's pushed tip, tracking
  `origin/openqha-hessian`); both tags resolve (`base-v0.3.16` = `5c2d761`, D; `v0.3.16`
  = `4d2da09`); `.git` 2.7 MiB, whole checkout 5.1 MiB (blobless). `origin` keeps the
  fork's all-heads refspec; `upstream` = `https://github.com/ACEsuit/mace.git` pinned to
  the old checkout's tag-only refspec `+refs/tags/v0.3.16:refs/tags/v0.3.16` (no
  upstream mass fetch).
- **Install evidence.** `python -m pip install -e .` in the WSL `openqha` env replaced
  the old editable install (`mace-torch 0.3.16` → `0.3.16+openqha`); `mace.__version__`
  = `0.3.16+openqha` from the checkout's `mace` package (literal in `mace/__version__.py`);
  `openqha.potentials.engine.mace_fork_info()` = commit `1110ffb...`, `mace_fork_dirty`
  false, `mace_fork_path` = the `mace/` checkout — re-checked after the archive move,
  import also verified from a neutral cwd. The checkout is tracked-clean (pip's
  untracked `mace_torch.egg-info/` does not count as dirty).
- **Archive.** The old checkout moved to `_to_delete/openQHA-Hessian-old-fork-2026-09-26/`
  (still `1110ffba`, tracked-clean); the `openQHA-Hessian/` path is clear. The move
  needed a **native rename** (PowerShell `Move-Item`): VS Code's git extension held the
  checkout's `.git` (handle dump: one `Code.exe` watcher handle, share-delete) and the
  WSL-side `mv` returned EACCES while it was held; a temporary
  `git.ignoredRepositories` workspace setting did not release it and was removed again.
- **Old → new SHA table** (ticket 06's mapping material; the full version with commit
  subjects is in [02b](02b-the-rebuild.md)): base `8fac5d1` → `5c2d761` (D); the six —
  `0ae78c4` → `a31d0a6`, `1b9c382` → `1b53032`, `e68390f` → `61582b0`, `904dd3b` →
  `cce52c0`, `a37f8b6` → `66a68f0`, `f14a56f` → `1110ffb` (old tip → new tip).
- **Handed to ticket 05** (install/transport documentation): the recipe that worked in
  this WSL env — inside the env, `git clone --filter=blob:none
  https://github.com/BloomDlwlrma/mace.git mace`, then `python -m pip install -e mace`.
  The documented URL form stays
  `git+https://github.com/BloomDlwlrma/mace.git@openqha-hessian`. A fresh env must end
  with `import mace` → the checkout and a clean `mace_fork_info()`; the editable shape
  (`.git` beside the package) is load-bearing for provenance; the pinned upstream tag
  refspec must survive any re-clone recipe. The offline Tianhe transport stays 05's
  decision.
- **Unverified:** nothing outstanding for this slice's acceptance.
