# 12b: The refusals and the identity text

Serves: [12](../decisions/12-references-sweep.md) · spec: [spec-references-sweep.md](../spec-references-sweep.md).

**What to build:** The dataset's split-change refusal offers remedies that exist (the build API's resplit, or a new dataset name) while keeping the substrings its unit test pins; the fork-guard refusal points at the current install and adds the conditional working-directory shadow diagnosis (pinned substrings kept); the synthetic mace-checkout fixture loses the retired checkout name; the integration test's fork-commit pin is remapped onto the rebuilt fork's history via the ADR 0011 table.

**Blocked by:** None (can start immediately).

**Status:** resolved

- [x] The split-refusal unit test passes unchanged; `split by`, `asks for` and `resplit` remain in the message.
- [x] The fork-guard unit test passes; `pip install -e` and the fork identity remain in the message; the install script is named.
- [x] The shadow note appears exactly when the working directory contains a `mace/` subdirectory — never otherwise.
- [x] `FORK_COMMIT_B` is the rebuilt history's commit-B hash (ADR 0011); the integration check still passes.
- [x] Both suites green (`--all`).

## Answer (2026-09-30, implemented in this commit and the package commit `16d3e7a`)

**The Dataset refusal** (`openqha/data/dataset.py:637–642`) keeps the three substrings its
unit test pins and now offers only remedies that exist — `resplit=True` through
`dataset.build` (with the note that `04_dataset.py` deliberately exposes no such flag),
a rebuild under a new `--name`, or keeping the mode. `t_dataset` passes unchanged.

**The fork guard** (`openqha_hessian/run.py`, the package commit) refuses with
`bash <path>/openQHA-Hessian/install.sh` (or `pip install -e <path>/mace`) in place of the
retired checkout name, keeping `pip install -e` and the `MACE_FORK` identity; when the
working directory holds a `mace/` subdirectory the refusal appends the shadow note
("cd into a repository root and retry"). `t_train_run` pins the install script beside the
kept substrings and checks the note both ways (present with the subdirectory, absent
without).

**The fixture and the pin.** The synthetic mace-checkout fixture in `t_engine_fork.py`
is renamed `mace-checkout` (the retired checkout's name out). `t_train_engine.py`'s
`FORK_COMMIT_B` is the rebuilt history's commit B,
`61582b0efb61cc251457c52367bb8239fe1cb2b5` (ADR 0011's old→new table); the integration
check passes.

Reported per the operating rule — files changed: `openqha/data/dataset.py`,
`tests/unit/t_engine_fork.py` (this repo); `openqha_hessian/run.py`,
`tests/unit/t_train_run.py`, `tests/integration/t_train_engine.py` (the package commit
`16d3e7a`). Checks run: the three unit programs green (`t_train_run` 41 checks, 0 failed);
`t_train_engine` 30 checks, 0 failed (and again inside the package suite, 104 s); suites
`--all` green — openQHA 73/73 (`C:\Users\10704\AppData\Local\Temp\suite12a.log`),
package 9/9 (`C:\Users\10704\AppData\Local\Temp\t12b-suites.log`). Not verified: nothing
pending — no behaviour change beyond the refusal strings is expected, and the suites are
the confirmation. The working tree also carries other slices' edits mid-flight (the
documents/scripts passes over `README.md`, `docs/`, `hpc/`); they are not part of this
commit.

## Review record (2026-09-30, annotations)

Two-axis review of the two commits (openQHA `8ed521a`, package `16d3e7a`), per the
`code-review` skill.

**Standards.** No hard violations: the pinned substrings are intact in both messages
(`split by` / `asks for` / `resplit`; `pip install -e` + `MACE_FORK`), no new text uses a
`CONTEXT.md` `_Avoid_` term, and the tracker follows the 12a template (the CRLF endings
match the tracker's existing `i/crlf` convention — the code files stay `i/lf`). Kept
judgement calls, all accepted as-is: `check_fork` now probes the cwd inside an identity
check — the requested self-diagnosis — while its docstring and the README's `run` line
still describe identity only; the Dataset message names the workflow's deliberate
non-flag (the sentence denies the flag, it does not offer it; the old text already named
`04_dataset.py`); `mace-checkout` slightly under-reads the fixture (it also hosts the
fake package — still "the synthetic checkout"); the test comment "changing it needs
`--resplit`" (`tests/unit/t_dataset.py:401`) is left for the tests pass — this slice's
contract is the message, and the pin is deliberately unchanged.

**Spec.** Every 12b requirement delivered, no scope creep: the split-change refusal keeps
the pinned substrings and offers only remedies that exist (the build API's resplit — with
`04_dataset.py` verified to expose no such flag — or a new `--name`); the fork guard
names the install script / an editable fork checkout and appends the cwd shadow note
exactly when the working directory holds `mace/` (tested both ways); the mace-checkout
fixture loses the retired name; `FORK_COMMIT_B` is ADR 0011's rebuilt commit B
(`61582b0e…` against the retired `e68390f…`), integration green. Story 17 (tutorials)
belongs to 12c, not here.
