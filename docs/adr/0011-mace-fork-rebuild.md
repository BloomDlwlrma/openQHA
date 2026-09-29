---
status: accepted
date: 2026-09-27
---

# The mace-side changes live on a true fork of upstream, rebuilt on upstream history

## Context

The openQHA Hessian work carries six mace-internal commits (hessian label, external loss
hook, multihead/eval, probes). They sat on `BloomDlwlrma/openQHA-Hessian`, a fork whose
base is one orphan commit `8fac5d1` (upstream v0.3.16 `4d2da09` re-rooted, the three
bundled foundation-model binaries left out) tagged `base-v0.3.16` — so nothing compares
against upstream and the commits cannot go upstream as a PR. The split ruling (2026-09-25/26)
gave the mace side its own true fork; branch name, rebuild mechanics and local layout were
settled in ticket 02 of `.scratch/hessian-learn-framework/` (grilling 2026-09-26/27).

## Decision

The six commits are rebased onto upstream v0.3.16 (`4d2da09`) on a true GitHub-side fork
`BloomDlwlrma/mace`, branch `openqha-hessian`. Topping `4d2da09` is one **fork-only**
commit D — "drop the three bundled foundation-model binaries" — because every checkout of
this fork (dev, Tianhe tarball, lazy/partial clones) should stay source-sized; the binaries
are not needed to build, to run mace's tests, or to run openQHA (weights live in
`data/potentials/`; `mace_mp`/`mace_off` download-or-cache when bundles are absent). The
tag **`base-v0.3.16` points at D** and is pushed with the branch, so
`git diff base-v0.3.16..openqha-hessian` keeps its old meaning: the six feature commits,
nothing else. **No default-branch setting is changed**; the fork simply keeps whatever
GitHub creates (the parent's current default). The old orphan history survives only as a
**local bundle**
(`_backup/openQHA-Hessian-old-history-2026-09-26.bundle`), made before the rebase. Commit
ids of the six change; the old→new mapping is under Consequences below.

## Considered options

* **Keep the binaries and ship a pure rebase.** Rejected: every checkout and every
  blob-filtered workflow would then materialize or fetch ~160 MB of files this project
  never uses; the old fork's lean clones were a property worth carrying over. The
  PR-side cost (one extra commit to exclude) is already anticipated by ticket 10.
* **A rebuild was needed at all** (vs. leaving the orphan base): rejected because the
  orphan base makes the branch non-comparable to upstream and non-PR-able — the point of
  the split's Q4 ruling.
* **Squash-and-repush convenience of any kind** (dropping upstream history to keep pushes
  tiny): rejected on a true fork, where the server already holds upstream's objects, the
  old 153 MB push failure cannot recur, and an object-diff against upstream is only
  meaningful with the real history present.

## Consequences

* The branch push from this machine is small (~1 MB) — the fork shares upstream's object
  pool; the old failure mode (squashed base with no shared objects) is gone.
* Any fresh clone still carries upstream's ~160 MB of history in `.git`; only
  `--filter=blob:none` + a tip checkout avoids the binaries, and the binaries-drop commit
  is what makes that checkout cheap.
* Old Records cite the old commit ids and the old `base-v0.3.16`. The old→new mapping,
  full 40-hex (the six commits of the branch, and the moved tag):

  | # | old | new | subject |
  |---|-----|-----|---------|
  | 1 | `0ae78c4f47c7a6220edf33e23fc836f1e7f82c7f` | `a31d0a6e3634896c349190c6e19aa9fb67e77049` | version 0.3.16+openqha |
  | 2 | `1b9c382f6cde289795781a0582de9c9b31e62942` | `1b530325313cb058efe36abed9976be54c9c744c` | hessian label |
  | 3 | `e68390fca2b9c055609726f8d0897cc90a4ee08d` | `61582b0efb61cc251457c52367bb8239fe1cb2b5` | external loss hook |
  | 4 | `904dd3b2d57e31f9265f71c64e4d25f3e0bd7020` | `cce52c0963d36b493549a3b78e32c174f98d3c2e` | multihead/eval |
  | 5 | `a37f8b67a1936ecbcdf70b12dd3606e4f0858d05` | `66a68f0e4ad2158fcba9e0c64a890bdf5d4ec2d9` | probes |
  | 6 | `f14a56fe40330766219cb83f9ebbf2d813b99e05` (old tip) | `1110ffbafd651a74d1d4678deb4748056d1dff0a` (new tip) | probe default |

  and the tag moved `8fac5d11bb34954e17ed7a41e7a4bb6f908017be` →
  `5c2d7612eed88dc1463b5588a79c2d5f5718d322` (D). Verified by the 02b rebuild
  (`.scratch/hessian-learn-framework/implementation/02b-the-rebuild.md`): old tip vs
  new tip, and each old commit vs its rebased counterpart, are empty diffs. A reader of
  an old Record should note that its `MACE_FORK` string,
  `BloomDlwlrma/openQHA-Hessian@openqha-hessian`, now names the GitHub address the
  `openqha-hessian` package occupies, not the fork that trained the number. The mapping
  can be checked against the archived bundle
  `../../../_backup/openQHA-Hessian-old-history-2026-09-26.bundle`.
* The fork's installs consume the branch by URL (`pip install git+https://github.com/BloomDlwlrma/mace.git@openqha-hessian`);
  the install/transport story is ticket 05's.
* Any later switch of the base to a newer upstream release is out of this effort's scope;
  a future rebase replays D together with the six commits.
