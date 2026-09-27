# hessian-learn-framework

Label: wayfinder:map — local-markdown tracker, this file at `.scratch/hessian-learn-framework/map.md`
Tickets: `decisions/NN-<slug>.md` — the map's route (`Type:` grilling/research/prototype/task; `Status:` open/claimed/resolved; `Blocked by: NN`). Execution slices land in `implementation/<parent><letter>-<slug>.md` (`Serves: NN`), findings in `research/`, superseded tickets in `archive/`. Convention: workspace-root `docs/agents/issue-tracker.md` → Wayfinding operations (ADR 0009).

## Destination

`openQHA-Hessian` stands on its own: the training side lives in it as a package, openQHA imports it, and the mace-internal changes live on the real fork `BloomDlwlrma/mace` (rebased on upstream history, PR-ready). Model weights load the native mace/mace-off way (SHA256 pinning retired). Round 1 of Hessian training runs and is debugged on one assembled xyz of every labeled frame.

## Notes

- The map is an index. A decision lives in exactly one place — its ticket; this file only gists it and links.
- Skills per session: `grilling` + `domain-modeling` for decision tickets; `research` for research tickets. Vocabulary: `openQHA/CONTEXT.md`. Scope rule: `openQHA/AGENTS.md` (2026-09-25): finish the ticket's scope; report anything beyond it, do not do it unsolicited.
- Layout (adopted 2026-09-26, [ADR 0009](../../docs/adr/0009-wayfinder-ticket-layout.md)): `decisions/` holds the map's tickets — the route; `implementation/` holds execution slices derived with `/to-tickets` (`<parent><letter>`, e.g. `07a-…`) plus work that never belonged on the route, one `Serves:` line each; findings live in `research/`; only superseded tickets go to `archive/`. The frontier scan covers both ticket folders. Execution is in scope for this effort, so map tickets may carry work as well as decisions.
- Seeded rulings (user, 2026-09-25/26 grilling; they stand unless a ticket revisits them):
  - Q1 = split: `BloomDlwlrma/mace` = a true GitHub fork of ACEsuit/mace for the mace-internal changes; `openQHA-Hessian` = the independent package.
  - Q2 = move the training side (`openqha/training/` or the whole 05_train side) out of openQHA into `openQHA-Hessian`.
  - Q3 = openQHA's reference layer may be edited as part of this effort.
  - Q4 = the mace-side changes are organised as upstream-PR-able.
  - Q5 = keep the `openQHA-Hessian` name and URL; replace its content; old history only as a local bundle; new commit ids accepted; old Records keep their old hashes, with a mapping note added.
  - Work items added 2026-09-26: (1) remove SHA256, call models the native mace/mace-off way; (2) assemble every labeled frame into one xyz for round 1 of Hessian training and the debug run.
- The old fork, for reference: 7 commits = squashed base `8fac5d1` (tag `base-v0.3.16` = upstream v0.3.16 `4d2da09` minus the three bundled binaries) + 6 commits (`0ae78c4` version bump; `1b9c382` A hessian label; `e68390f` B external loss; `904dd3b` C multihead/eval; `a37f8b6` D probes; `f14a56f` probe default); whole change = 12 files, +869/−18.

## Decisions so far

<!-- one line per closed decision ticket (decisions/): [<ticket title>](decisions/NN-slug.md): gist of the answer -->

- [The package line: what moves, what stays, and what the package is called](decisions/01-the-package-line.md): the whole `openqha/training/` (7 modules, 2,662 lines) and its 9 tests move into the `openqha-hessian` package (`openqha_hessian`, version 0.1.0); direction package → openqha — the library never imports it — and consumers keep their files, changing addresses only; no shim for `openqha.training`; `--loss_module` becomes `openqha_hessian.phl_loss:build`. Slice: [01a](implementation/01a-the-package-line.md); durable record: ADR 0010.

- [The real fork: `BloomDlwlrma/mace`, rebuilt on upstream history](decisions/02-the-real-fork.md): the true fork carries `openqha-hessian` @ `1110ffb` + `base-v0.3.16` @ D `5c2d761` (default branch switched to `openqha-hessian`, decision 3 revised); a fresh blobless clone at the sibling `mace/` installs editable in the WSL env (`0.3.16+openqha`, clean `mace_fork_info`); the old checkout retired to `_to_delete/openQHA-Hessian-old-fork-2026-09-26/`. Slice: [02a](implementation/02a-the-mace-fork.md); durable record: ADR `0011`.

- [Native model loading: what mace/mace-off offer without SHA256](decisions/03-native-model-loading.md): mace's runtime load path hashes nothing (upstream's sha256 lives in its test goldens); `mace_off()` downloads/caches when bundles are absent; `engine.calculator()` is already native — ticket 04 must pick minimal-strip vs delegate-to-`mace_off`. Findings: [`research/native-model-loading.md`](research/native-model-loading.md).

## Implementation

<!-- one line per landed execution slice (implementation/): [<ticket title>](implementation/NNx-slug.md): gist -->

- [02b: The rebuild](implementation/02b-the-rebuild.md): the branch rebuilt on upstream history — `base-v0.3.16` @ D `5c2d761` (v0.3.16 minus the three bundled binaries); six commits replayed to tip `1110ffb` (old-tip content identical); old history bundled in `_backup/openQHA-Hessian-old-history-2026-09-26.bundle`; push is 02c.
- [02c: Fork and push](implementation/02c-fork-and-push.md): the true fork `BloomDlwlrma/mace` carries the rebuilt refs — `openqha-hessian` @ `1110ffb`, `base-v0.3.16` @ `5c2d761` (push ≈63 KiB; upstream's refs inherited unchanged, nothing force-pushed); the checkout's `origin` points at it; old fork untouched; default branch switched `develop` → `openqha-hessian` (see 02c Postscript).
- [02d: Clone, verify, re-home](implementation/02d-clone-verify-rehome.md): fresh blobless clone of the fork at `mace/` (`1110ffb`; branch tracks `origin`; `upstream` pinned to the v0.3.16 tag refspec only); editable install in the WSL env (`0.3.16+openqha`, clean `mace_fork_info`); old checkout archived under `_to_delete/`; the install obligations for ticket 05 are in the Answer.
- [01b: The package skeleton](implementation/01b-the-package-skeleton.md): `openQHA-Hessian/` is a live repo (`main`; origin = the old URL, first commit unpushed) carrying `pyproject.toml` (0.1.0), `hvp.py`, the twin runner (provenance header) and the fixture locator (`openqha_src`/`OPENQHA_SRC`); `phl_loss`'s sibling import is openQHA's one consumer change; the WSL env has the editable install + the `openqha-dev.pth` dev line; openQHA's 80-test suite green, `05_train --dry-run` byte-identical.

## Not yet specified

<!-- fog: in scope, not sharp enough to ticket yet; graduates as the frontier advances -->

- What to do with old artifacts whose recorded paths point at the old checkout (`mace_fork_path` in Records, run configs under `~/runs/openQHA`): migrate, note, or leave as history. Sharpens once *Identity after the split* and *The repo swap* settle the new paths.
- Whether the package needs release artifacts of its own (wheels on GitHub Releases?) for offline Tianhe installs. Sharpens once *Install and transport* picks a mechanism.
- What `s0_check_weights.py` becomes in reduced form (`--json`/`--compare` semantics) after the fingerprint machinery retires. Sharpens once *SHA256 retirement* lands.

## Out of scope

<!-- work ruled beyond the destination; closed, never graduates -->

- The campaign beyond round 1: the R0–R3 replay rows, the `w_H` scan, reopening the judgement gate, later training rounds (the parked active-learning round proposal in `hessian-learning-set/grilling-round-12`).
- The upstream review cycle after the PR is opened, and any switch of the mace base to a newer release.
- The remaining `hessian-learning-set` wrap-up (its planned ticket 38, the Algorithm-5 cleanups) — that set owns it.
