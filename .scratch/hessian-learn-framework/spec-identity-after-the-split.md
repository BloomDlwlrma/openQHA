# Spec: Identity after the split — the package in the Records, the fork constant, and the old→new mapping

Label: `ready-for-agent`. Tracker: `.scratch/hessian-learn-framework/`. Spec for
[06-identity-after-the-split](decisions/06-identity-after-the-split.md), rulings of 2026-09-29;
findings: [research/mace-identity-practice.md](research/mace-identity-practice.md); durable
records: [ADR 0011](../../docs/adr/0011-mace-fork-rebuild.md) (amended here) and
[ADR 0010](../../docs/adr/0010-hessian-learning-lives-in-openqha-hessian.md).

**Rulings taken with this spec.** Q1: `MACE_FORK` = `BloomDlwlrma/mace@openqha-hessian`,
`MACE_FORK_BASE` value unchanged. Q2: the package is recorded as version + best-effort commit,
never gated. Q3: the refusal rules stay exactly as they are; the package is not added to any
gate. Q4: the old→new mapping lives in ADR 0011. Q5: the fork reader's contract is unchanged.

Settled elsewhere, not this spec's work: the fork rebuild ([02](decisions/02-the-real-fork.md));
the package line and the move ([01](decisions/01-the-package-line.md),
[07](decisions/07-the-move.md)); the weight-identity strip ([04](decisions/04-sha256-retirement.md));
the install and transport story ([05](decisions/05-install-and-transport.md)).

## Problem Statement

After the split, three code bases produce a training or judge number: openQHA (the library —
its `VERSION` is in the Record), the mace fork (its commit is in the Record as
`MACE_FORK_COMMIT`), and `openqha-hessian`, the package — which the Record does not name at
all. The package is where the loss and the judge live now; its version stays `0.1.0` while its
code keeps moving through round 1, so two Records made a day apart cannot be told apart by the
code that made them, and when a number moves nothing on disk says whether the package changed.

The Record also misdirects. `MACE_FORK` still reads
`BloomDlwlrma/openQHA-Hessian@openqha-hessian` — a repository name that now belongs to the
package, not to the fork that trained the number — so new Records, prints and repair messages
send a reader to the wrong repository. And old Records are immutable and keep citing commit
ids (`904dd3b`, …) that the rebuilt fork no longer contains; the ids changed by design
(accepted with [02](decisions/02-the-real-fork.md)), and nothing on disk maps them.

## Solution

Each new training and judge Record carries the package's identity next to the fork's:
`HL_PACKAGE_VERSION` (the installed distribution's version) and `HL_PACKAGE_COMMIT` (the
package checkout's commit, read best-effort — `unknown` when there is nothing to read, never a
refusal). The `MACE_FORK` constant is corrected to `BloomDlwlrma/mace@openqha-hessian`;
`MACE_FORK_BASE` keeps its value and gains a comment saying where the tag now lives. Nothing
new gates anything: `check_fork` keeps exactly its current meaning, and the package is not
added to it. [ADR 0011](../../docs/adr/0011-mace-fork-rebuild.md) gains the old→new commit
table, and the references sweep ([12](decisions/12-references-sweep.md)) updates the tests and
tutorial wording that pinned the old hashes. Old Records and fixtures are untouched.

## User Stories

1. As the operator, I want every new training Record to name the package version and commit
   that computed its numbers, so that any run can be repeated against the same loss code.
2. As the operator, I want every new judge Record to carry the same two fields, so that a
   verdict says which judge code produced it.
3. As the operator, I want the package commit read best-effort with `unknown` as the fallback,
   so that a wheel install or a missing `.git` degrades the Record instead of blocking the run.
4. As the operator, I want no refusal attached to the package's identity, so that round-1 work
   never stops because of provenance bookkeeping.
5. As the operator, I want the version read from the installed distribution metadata, so that
   the project file is the single place a version is bumped.
6. As the operator, I want `MACE_FORK` to read `BloomDlwlrma/mace@openqha-hessian`, so that new
   Records, prints and repair messages name the repository that actually exists.
7. As the operator, I want `MACE_FORK_BASE` to keep naming `base-v0.3.16` while its comment
   says where the tag now lives, so that identity strings do not silently change meaning.
8. As the operator, I want `check_fork` unchanged — strict training still refuses an mace
   checkout that is unknown or has modified tracked files — so that the loss stays
   "reproducible from a commit".
9. As a reader of an old Record citing `904dd3b`, I want a durable old→new table, so I can
   find the rebuilt commit that carries the same content.
10. As a reader seeing the old `MACE_FORK` string in an old Record, I want the note that this
    name is now the package's address, so I am not sent to the wrong repository.
11. As the operator, I want the archived bundle named next to the mapping, so the mapping can
    be checked against the history it came from.
12. As the operator, I want the new fields pinned by the existing tests — the writers' schema
    checks and the reader's fallback cases — so a future edit cannot drop them silently.
13. As the reviewer, I want the same test seams the fork identity already uses (no new seam),
    so the suite keeps one kind of shape.
14. As the operator, I want old Records and fixtures untouched, so history reads exactly as it
    did.
15. As the operator, I want the sweep to update the training integration test's pinned old
    commit B to the rebuilt `61582b0e…` and T05's fork-commit wording, so no test or tutorial
    claims a hash that no longer exists.
16. As the operator on Tianhe, I want nothing here to need network or a new install shape, so
    this change is portable by construction.

## Implementation Decisions

- **The constants.** `MACE_FORK` becomes `"BloomDlwlrma/mace@openqha-hessian"` — the rebuilt
  true fork, same branch, same `owner/repo@branch` shape. `MACE_FORK_BASE` keeps
  `"base-v0.3.16"`; its comment is refreshed to say the tag now lives on the fork's rebuilt
  history at the binary-drop commit (`5c2d761` = upstream v0.3.16 minus the three bundled
  model binaries). Both are report-only strings — messages, prints, Record fields; nothing
  parses them — so existing consumers follow automatically once the value flips. Any
  assertion that pins the old value is updated in the same change.
- **The two new fields, both Records.** `HL_PACKAGE_VERSION` and `HL_PACKAGE_COMMIT` are added
  to the training writer's Record (the `Calculation_Info` schema and the info dict the run
  writes, including the dry-run path the tests read) and to the judge writer's Record the
  same way. The names are exactly these; no `EXTENSION_*` prefix (the package is not
  `openqha.extensions`).
- **The values.** The version comes from the installed distribution metadata of
  `openqha-hessian` (the hyphenated distribution name), `"unknown"` when it cannot be read.
  The commit is read best-effort from the package's own checkout with the fork reader's
  semantics: the module file's grandparent must hold `.git`, `rev-parse HEAD` must answer a
  full 40-hex commit, and every failure — no git, not a repository, a timeout, a stray answer
  — becomes `"unknown"`; the read never raises. The package's working-tree state is not read:
  there is no dirty field, and the dirty question is not asked.
- **One reader, shared rules.** The reader is a generalization of the fork reader so the rules
  live once; `mace_fork_info` keeps its public shape (module file and git runner injectable;
  its current keys and behaviour). The package read is the same rules run on the package's
  module file. The two identities must never disagree about what "a checkout" means.
- **The refusal rules do not move.** `check_fork` keeps its current behaviour exactly (strict
  training refuses an unknown or dirty mace checkout; `smoke_fit` passes `strict=False`); the
  package is not added to any gate; the judge keeps recording whatever it can read.
- **No new mechanisms.** No PEP 610 / `direct_url.json` reading, no build-time stamping, no
  checksums, no dirty recording, nothing that changes what must be installed where. The
  mace-release shape applies: a version string as the identity, git capture best-effort, and
  nothing gated (see the research note).
- **The mapping.** [ADR 0011](../../docs/adr/0011-mace-fork-rebuild.md) is amended: the
  consequence line "the old→new mapping note is ticket 06's obligation" is replaced by the
  table itself — the six replayed commits and the moved base tag, old→new full 40-hex, taken
  from [02b](implementation/02b-the-rebuild.md) (whose old-tip ↔ new-tip diff is empty) —
  plus one note that the old `MACE_FORK` string names the address the package now occupies,
  and the archive bundle's path
  (`../../../_backup/openQHA-Hessian-old-history-2026-09-26.bundle`). No separate docs page.
- **Where it lands.** The fields and the reader ride the move of the training writers into the
  package ([07](decisions/07-the-move.md), the 01d slice), so they are not churned twice; the
  constant edit is independent and small; the sweep items are
  [12](decisions/12-references-sweep.md)'s: the training integration test's pinned old commit
  B → `61582b0efb61cc251457c52367bb8239fe1cb2b5`; the fork reader test's cosmetic fake id → the
  new base prefix; T05's fork-commit wording; the `.scratch/hessian-learning-set` pointer
  notes; and the training gate's repair message, which still offers
  `pip install -e <path>/openQHA-Hessian` (the retired name — it would install the package,
  not the fork) — re-pointed at the fork's current home with
  [05](decisions/05-install-and-transport.md)'s install story, its message tests kept aligned.
- **Old data stays.** Old Records, old fixtures and every old hash in history files stay
  readable as-is; only assertions that pin the old world's values are updated.

## Testing Decisions

- A good test asserts external behaviour only: that the Record a writer produced contains the
  two keys with values of the right shape, and that the reader returns `unknown` for defective
  checkouts and a 40-hex commit for a real one. Nothing asserts the reader's internals.
- Seams — both existing, no new seam:
  1. **The writers' Record checks** (the highest practical seam for "what the Record
     carries"): the training writer's unit test pins the exact key set the run writes and
     asserts it sits inside the Record schema; extend that set with `HL_PACKAGE_VERSION` and
     `HL_PACKAGE_COMMIT`. The judge side's Record test gets the same containment check beside
     its schema round-trip, so both writers are pinned the same way.
  2. **The identity reader's unit test** (prior art: the fork reader's test — a temporary
     checkout with an injected git runner): a 40-hex answer is recorded; no `.git` beside the
     package answers `unknown`; a failing git answers `unknown`; a non-commit answer answers
     `unknown`; a `.git` further up is not found. Run the same cases on the package's module
     file.
- The integration test that reads a real training Record asserts the two keys exist (value:
  40-hex or `unknown` — the environment decides).
- Not tested: the constant's literal value (it is exercised through the message and print that
  interpolate it), and the ADR text.

## Out of Scope

- Wheel / pip provenance mechanics: PEP 610 reads, build-time stamping, checksums. Settled
  against for this round; a future need is [05](decisions/05-install-and-transport.md) /
  [13](decisions/13-publication.md)'s.
- Softening `check_fork` or extending any refusal to the package.
- Recording the package's dirty state.
- Migrating old artifacts whose recorded paths name the old checkout (the map's "Not yet
  specified"; sharpens elsewhere).
- The Tianhe transport ([05](decisions/05-install-and-transport.md)), the repo swap
  ([11](decisions/11-repo-swap.md)) beyond names, publication ([13](decisions/13-publication.md)).

## Further Notes

- The evidence for this shape: [research/mace-identity-practice.md](research/mace-identity-practice.md)
  — the shipped mace records almost nothing and never refuses; its develop branch reads pip
  metadata only in tests. This spec takes the release's shape and adds only the one thing our
  Records exist to say: which code produced a number.
- The write-once asymmetry: a field missing from an immutable Record can never be back-filled,
  so the commit is recorded best-effort even though a wheel will answer `unknown`.
- The purest-minimal variant (version only — no commit, no reader) was considered when this
  spec was written and was not taken, because round-1 package changes would become invisible
  in the Records. If it is ever preferred, the change is: drop `HL_PACKAGE_COMMIT` and the
  reader call — nothing else moves.
