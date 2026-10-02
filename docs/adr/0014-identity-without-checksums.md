---
status: accepted
date: 2026-10-02
---

# Identity without checksums: the sha256 machinery comes out, the PRNG seed material stays

## Context

The project's sha256 uses had accumulated beyond the weight layer: ticket 04's ruling
retired the model-file fingerprints and the registry pin but deliberately kept
`CONFIG_SHA256` and the non-weight reference-data hashes, ADR 0012 had the environment
files carry the mace fork as a bare non-editable git-URL requirement, and the Tianhe
process verified copies and read receipts with `sha256sum` (the campaign page's "both
sides size+sha equal" transfer rule; the deployment receipts). The user's 2026-10-02
ruling (`.scratch/hessian-learn-framework/decisions/16-identity-without-checksums.md`,
slices 16a-16e) drew the line again: sha256 is not how this project establishes
correspondence.

## Decision

**Every sha256 use the project adds of its own retires.** The load path was already
native (mace's `torch.load` hashes nothing -- ADRs 0011/0012); the remaining
project-side uses come out: the Record's `CONFIG_SHA256`, the reference-data digests
(the QM9 list's pinned digest and provenance match flag, the curated pack's hash
attribute, the artifact index's hash column, the prep tool's head-hash, the edge-list
hash and its pin, the mace-patch source fingerprint), and the receipt steps that
computed or compared digests (including the campaign's "both sides size+sha equal"
transfer rule). Old Records and recorded receipt values stay byte-identical as records
-- the immutable-Records rule -- they are history, not a mandate.

**One exception: PRNG seed material.** The sha256 calls that *seed random draws* stay:
`openqha.data.frames.frame_seed` (the first 8 bytes of the digest seed a frame's
displacement stream) and `openqha.data.dataset`'s `_rng`, which seeds `valid_probes`
and `frame_draw`. Seed derivation is not a correspondence check; the numbers they
generate are the reproducibility handle. Upstream mace/CREST code is never touched
("mace/crest native").

**The kept product identity (Q9=A).** The fork identity stays: `check_fork`'s refusals
(a non-fork mace, or a dirty checkout, refuses training) and the Record's fork/package
commit fields (`MACE_FORK_COMMIT`, `HL_PACKAGE_VERSION`/`HL_PACKAGE_COMMIT`). Identity
is name + resolved path + commit; the checksum was never the gate.

**Process layer.** The two-side commit-locking deployment discipline retires:
deployment receipts name each tree's `git log -1` and a clean `git status --porcelain`;
no cross-side equality and no digest is required. The fork-by-URL requirement retires
from the environment/requirements files and from the install default: the carried
(sibling) checkout is the install path, and the URL survives only as an explicit
opt-in.

## Considered options

* **Keep the reference-data digests because they are cheap.** Rejected: the ruling is a
  boundary, not a cost decision; the seed exception shows the line is drawable.
* **Replace sha256 with size (or mtime) where copies are verified.** Rejected (user
  ruling): no substitute -- the archive's read-back verification is the check, not a
  number compared on two sides.
* **Purge historical values from Records and answers.** Rejected: Records are immutable
  and old answers are history; the machinery retires for what the project does and
  requires from here.

## Consequences

* `CONFIG_SHA256` retires from the fine-tune Record (slice 16a); the recipe's identity
  is the config file the Record names (`CONFIG_FILE`), and the registry entry's source
  is the index plus that path.
* The reference-data hashes and the curated-QM9 verifier's hash modes retire (slice
  16b).
* Requirements/environment files stop carrying the fork-by-URL pin;
  `install_dependency.sh` installs no mace and points at `install.sh` (slice 16c).
* The install defaults move to the carried checkout -- `install.sh` with no argument,
  never the network; the URL is explicit (slice 16d).
* Tianhe lands the carried commits with a names/ids/clean-status receipt, no checksums
  (slice 16e).
* Narrows ADR 0012: "the environment files carry the fork as a bare non-editable
  git-URL requirement", "Default source: the fork's branch URL", and the pure-eval
  non-editable ride are retired; the one-script install and the
  editable-where-provenance-needs-it rule stand. Narrows ticket 04's "kept" list and
  ADR 0013's bare-branch-name installation note.
