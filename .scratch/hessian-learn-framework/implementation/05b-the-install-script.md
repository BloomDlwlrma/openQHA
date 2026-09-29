# 05b: The install script — the fork and the package, one `install.sh`

Type: task
Status: resolved
Serves: 05
Blocked by: None
Part of: [hessian-learn-framework](../map.md)

> Execution slice of [05a](05a-install-and-transport.md): spec decisions 1–7 and 14.
> Wayfinder conventions apply: the `Status` protocol, no triage labels
> (`docs/agents/issue-tracker.md`, ADR 0009).

## What to build

The package repository gains its `install.sh` — the mace-md pattern (pip-install the fork from its
branch URL, then the package itself) plus the two things our stack adds: editable installs (the
`mace_fork_info()` contract needs `.git` beside the imported package) and a local-path mode for
machines without GitHub.

- Default source: `git+https://github.com/BloomDlwlrma/mace.git@openqha-hessian`; one optional
  argument (or variable) supplies a local checkout path instead — the offline/Tianhe mode. The
  script never clones by hand: pip's editable VCS install keeps the clone, `.git` included.
- Order and safety: validate a local path first (exists, `.git`, the mace package at its root);
  then uninstall any `mace-torch` distribution; then editable-install the fork; then
  editable-install the package (the script resolves its own repository root).
- Verification block, non-zero exit on any failure: `import mace` lands inside a checkout whose
  package directory has `.git` one level up; version `0.3.16+openqha`; the checkout is
  tracked-clean; when the openQHA checkout is importable, `mace_fork_info()` returns a full commit;
  the fork and package commits are printed.
- No `--no-deps` (mace-md precedent); `python -m pip` on the active environment; re-running is
  safe.
- Docs in this slice: the package's README gains the install section for initial users; openQHA's
  README fork section is rewritten to the same story (no `pip install -e ../openQHA-Hessian`
  old-world naming), including the one-sentence eval-vs-training note.

## Acceptance

- [x] Local-path mode run twice in the WSL `openqha` env: both runs end with `import mace` at the
      checkout, `0.3.16+openqha`, tracked-clean, `mace_fork_info()` clean, `import openqha_hessian`
      OK; the second run is visibly idempotent
- [x] URL mode runs in a throwaway venv (never the live env — it would move the dev import): the
      clone lands with `.git`, the fork contract holds
- [x] Negative cases: a non-checkout path and a missing path exit non-zero BEFORE any
      uninstall/install; the environment's mace import is unchanged after the refusal
- [x] `bash -n` clean; the header states the contract (active env only, no env creation, no openQHA
      handling)
- [x] READMEs updated
- [x] Reported per the operating rule: files changed, checks run, anything not verified

## Answer

Resolved 2026-09-29, agent-run. One script puts the training stack into an activated
environment -- the mace fork and the package, both editable, verified -- and one story lives in
both READMEs.

**What changed** (package repo: new `install.sh` + `README.md`; openQHA: `README.md` fork
section):

- `openQHA-Hessian/install.sh` -- `bash install.sh [MACE_CHECKOUT_PATH]`. Default source
  `git+https://github.com/BloomDlwlrma/mace.git@openqha-hessian#egg=mace-torch` (see the
  finding below); one optional argument replaces it with a local checkout (offline / Tianhe),
  validated first (exists, `.git`, `mace/__init__.py` at the root) so a bad path cannot leave
  the environment without mace. Then: uninstall `mace-torch` -> editable-install the fork ->
  editable-install this repository (root resolved from the script's own location) -> the
  verification block: `import mace` lands in a checkout whose package directory has `.git` one
  level up, `0.3.16+openqha`, tracked-clean, `mace_fork_info()` full commit when openQHA
  imports, `import openqha_hessian`, fork + package commits printed; non-zero exit on any
  failure. Header states the contract (active env only, never creates environments, never
  touches openQHA); no `--no-deps`; re-running is safe.
- `openQHA-Hessian/README.md` -- the install and verify sections for initial users (mace-md
  style): the script, the by-hand pip lines, the ordering note vs `install_dependency.sh`
  (05c's hand-off sentence), the eval sentence.
- `openQHA/README.md` -- the fork section rewritten: the fork is `BloomDlwlrma/mace` @
  `openqha-hessian`, install via `../openQHA-Hessian/install.sh`, the ordering note (run it
  last -- or again after any re-run of `install_dependency.sh`), the eval sentence; no
  `pip install -e ../openQHA-Hessian` old-world naming remains.

**Finding that decides the URL spelling** (differs from decision 2's bare string in the fragment
only): for *editable* VCS requirements the bare `-e git+…@openqha-hessian` form fails on pip
24.0 ("Could not detect requirement name"); `#egg=mace-torch` passes on pip 24.0 (stock venv)
and on 26.2.1 (the WSL env's pip) -- the research note's "portable spelling keeps `#egg=`".
Repo and branch are exactly decision 2's. (05c's venv check saw the bare form pass
*non-editable* -- the fragment is only needed for the editable install.)

**Evidence** (WSL, `openqha` env; throwaway venvs under `/tmp`; 2026-09-29):

- `bash -n install.sh` -- clean.
- Negative cases: missing path / directory without `.git` / git checkout without the mace
  package -- all exit 1 with a named error, zero `==>` step lines in the logs (nothing ran),
  and a before/after probe (`import mace` location + `pip show mace-torch`) byte-identical.
- Local-path mode x2 (WSL env): both exit 0; run 2 shows the full replace cycle again
  (visibly idempotent); end state both times: `import mace` ->
  `/mnt/c/.../mace/mace/__init__.py`, `0.3.16+openqha`, tracked-clean, `mace_fork_info()` =
  `1110ffbafd651a74d1d4678deb4748056d1dff0a` dirty=False, `import openqha_hessian` OK;
  before/after snapshots identical.
- URL mode in throwaway venvs (the live env's dev import never moved): `install.sh` with no
  argument, in a stock venv (pip 24.0) and in a pip-26.2.1 venv -- both exit 0; the clone lands
  at `<venv>/src/mace-torch` with `.git`; `mace_fork_info()` full commit, clean. The live env
  still imports the workspace checkout afterwards; pip's "Not uninstalling ... outside
  environment" guard kept the conda copies untouched throughout.
- Suites: `openQHA-Hessian` unit 3/3 and `--all` 4/4 (integration included); openQHA unit 63/63.

**Not verified / deferred:**

- `shellcheck` is not installed (no lint beyond `bash -n`).
- No Tianhe execution (05d's scope); a fresh environment built from the documented lines is
  [05e](05e-fresh-env-acceptance.md)'s proof.
- openQHA's full `--all` group was not run here (no Python changed; `t_train_engine` still
  waits on 01d and the group rewrites shared fixtures) -- its unit group was run instead.
