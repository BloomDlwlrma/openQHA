---
status: accepted
date: 2026-09-29
---

# The training stack installs by one script, from git URLs, editable where provenance needs it

## Context

After the split (ADRs [0010](0010-hessian-learning-lives-in-openqha-hessian.md),
[0011](0011-mace-fork-rebuild.md)), a machine that fine-tunes needs three artifacts pip cannot
resolve on its own: the openQHA checkout (a library, not a distribution), the mace fork
`BloomDlwlrma/mace@openqha-hessian`, and the `openqha-hessian` package. The old world's story —
a PyPI wheel pin plus a line of prose about the fork — described none of this: the environment
files still named the wheel, the Tianhe transfer tool did not carry the package although its
document said it did, and no documented sequence put a fresh environment into the state where
the training fork check passes. Ticket 05 of `.scratch/hessian-learn-framework/` settled the
machinery; the mace-md pattern (`jharrymoore/mace@softcore`, installed by git URL from a short
script) is the model, with the one deviation this project cannot skip: `mace_fork_info()` reads
the fork's commit from the `.git` beside the imported package, so wherever training runs, the
fork must be an *editable* install.

## Decision

One install script, `openQHA-Hessian/install.sh`, is the single recipe — shared by both
READMEs, the Tianhe install job and the local flow. It runs inside an already-activated
environment, acts on that environment through `python -m pip`, and never creates environments
or touches openQHA. Default source: the fork's branch URL,
`git+https://github.com/BloomDlwlrma/mace.git@openqha-hessian#egg=mace-torch` (the fragment
names the requirement for older pips; a branch never rots under a pin's feet — the run-time
Record is the reproducibility handle). One optional argument replaces the URL with a local
checkout path — validated before anything is uninstalled or installed — which is the offline
and Tianhe mode. Steps, in order: uninstall any `mace-torch` distribution; editable-install the
fork (pip's blobless VCS clone keeps `.git`); editable-install the package; verify the end state
(import location with `.git` one level up, version `0.3.16+openqha`, tracked-clean,
`mace_fork_info()` full commit, `import openqha_hessian`) and exit non-zero on any failure.
Re-running is safe. No `--no-deps`; no wheels — nothing is built, shipped or kept as a fallback
(release artifacts are the publication ticket's, ticket 13 of the same effort).

The environment and requirements files carry the fork as a bare **non-editable** git-URL pip
requirement (the mace-md form): building an environment installs the fork, never the wheel, and
pure-eval machines ride it with the fork commit recorded as `unknown` — already tolerated
outside training. The editable install is required exactly where the training side runs
(`05_train`, the training Slurm job, `judge`); `install_dependency.sh` no longer installs any
mace and ends with a one-line pointer to `install.sh` (it never invokes it, so users of other
campaigns are not silently switched).

On Tianhe the two checkouts travel beside the repository — the transfer tool's push list
carries `../mace` and `../openQHA-Hessian` with `.git` protected — and the environment-install
job's section 9 installs both artifacts into every environment it manages by delegating to
`install.sh` in local-path mode through each environment's own python. Absent checkouts print
the explicit skip message, so eval-only builds still succeed. Installation happens at a
networked moment (WSL; the Tianhe login side with its proxy); the AI side installs from the
carried checkouts, and the local-path argument is the hedge when GitHub is unreachable.

## Considered options

* **Wheels or tarballs somewhere on a shelf.** Rejected: they go stale the moment the branch
  moves, and release artifacts are the publication ticket's business; the acceptance shows the
  URL install working.
* **A commit pin in the requirement.** Rejected: it rots; the run-time Record (the fork commit
  `mace_fork_info()` reads) is the reproducibility handle instead.
* **A separate offline / debug script.** Rejected: the local-path mode of the same script *is*
  the offline mode — one recipe, no drift between developer and cluster.
* **Editable installs on eval machines too.** Rejected: eval needs only "`import mace` is the
  fork"; the non-editable URL install answers that without a script, and the stricter contract
  stays where it acts.

## Consequences

* Script and documentation tell one story: openQHA's README fork section, the package's README,
  the Tianhe install document and the dependency installer's messages all say the same three
  things — fork by git URL, `install.sh` to train, run it last.
* The fork check is the boundary made executable: a Record made on an eval machine carries
  `mace_fork_commit: unknown`; a training run refuses it.
* Two deviations from ticket 05's spec letter, both recorded on the slices and kept: the
  editable requirement carries the `#egg=mace-torch` fragment (bare `-e git+…@branch` fails on
  pip 24.0), and pure-eval machines ride the environment-built non-editable fork rather than a
  mace wheel.
* Where the editable clone lands is pip's business, not the script's: `<env>/src` in a
  virtualenv, `<cwd>/src` under conda (pip classifies a conda environment as a global install).
  The acceptance recorded the practical consequence — from the README's implied cwd the fork
  clone lands inside the openQHA checkout; a fix is open (ticket 05's Answer has the evidence).
* Old Records keep their old strings and paths; what they meant is answered by ADR 0011 (the
  fork mapping) and the field-history ADRs (0004, 0006, 0010).
