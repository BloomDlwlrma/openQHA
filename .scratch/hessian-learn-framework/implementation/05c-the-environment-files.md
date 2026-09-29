# 05c: The environment files — the fork replaces the wheel pin

Type: task
Status: resolved
Serves: 05
Blocked by: None
Part of: [hessian-learn-framework](../map.md)

> Execution slice of [05a](05a-install-and-transport.md): spec decisions 8–10.
> Wayfinder conventions apply: the `Status` protocol, no triage labels
> (`docs/agents/issue-tracker.md`, ADR 0009).

## What to build

Building an environment must install the fork, never the wheel.

- The five environment files and the two requirements files replace their `mace-torch` pins with
  the fork's git-URL pip requirement — bare and non-editable (mace-md style): eval rides the
  non-editable fork fine; machines that train run [05b](05b-the-install-script.md)'s script
  regardless. Comments naming the wheel or odd versions are rewritten.
- `install_dependency.sh` drops mace from its explicit pip line (pymsym and the rest stay — the
  environment files now carry the fork; re-installing the wheel there would overwrite it) and ends
  with a one-line pointer to `install.sh` for machines that train. It does not invoke the script.

## Acceptance

- [x] No `mace-torch` requirement remains in the five environment files or the two requirements
      files; no comment still names the wheel as how mace arrives
- [x] A scratch venv pip-installing the fork URL line resolves `import mace` to the fork, version
      `0.3.16+openqha`
- [x] `install_dependency.sh` no longer pip-installs mace, prints the pointer; `bash -n` clean
- [x] Full fresh-environment proof explicitly deferred to [05e](05e-fresh-env-acceptance.md)
- [x] Reported per the operating rule

## Answer

Resolved 2026-09-29, agent-run. Building an environment now installs the fork, never the wheel.

**What changed** (the 8 dependency files, +50/-36):

- The five environment files (`environment.yml`, `environment-cuda.yml`, `environment-tianhe.yml`,
  `environment-tianhe-gpu.yml`, and a branch-B-only fifth, since retired to `_superseded/`) and the
  two requirements files (`requirements.txt`, `requirements-minimal.txt`) carry
  `git+https://github.com/BloomDlwlrma/mace.git@openqha-hessian` in place of `mace-torch>=0.3.6`
  -- bare and non-editable, mace-md style. Each site's comment now tells the fork story (fork by
  git URL, non-editable by design, training runs `openQHA-Hessian/install.sh` regardless).
- Comments rewritten with the pins: `environment.yml`'s wheel-range note; `requirements.txt`'s
  "pin `torch==` and `mace-torch==`" advice is now "pin `torch==` and keep to one mace fork
  commit", citing `engine/torch_version` + `engine/mace_fork_commit`; the reproducibility block's
  `mace 0.3.17`/`mace 0.3.16` identifiers are gone (the two environments it compares are now
  identified by their BLAS/torch stacks); the Tianhe "pip layer" records name the fork instead of
  the wheel; the GPU file's "0.3.17 does not exist on PyPI" NOTE is retired with the pin it
  policed.
- `install_dependency.sh`: mace dropped from the explicit pip line (`pymsym` stays; the `parsl`
  line is unchanged), the pip-pass comment says why (a wheel here would overwrite the fork `$REQ`
  installs), the `say` banner and the pip-index comment name the fork, and the final message ends
  with the one-line pointer: `3. to TRAIN (Hessian Labels): run ../openQHA-Hessian/install.sh in
  the active environment`. The script does not invoke `install.sh`.

**Evidence** (WSL, `openqha` env, 2026-09-29; the venv check in a throwaway `/tmp` venv):

- Grep over the seven files: zero `mace-torch` / `0.3.17` / `0.3.6`; each carries exactly one
  fork URL line. The only `mace-torch` string left in the eight touched files is the version
  probe's distribution-name label in `install_dependency.sh` (stdout, not a requirement or
  comment).
- `bash -n install_dependency.sh` -- clean.
- All five environment files parse (`yaml.safe_load`), each with exactly one `git+https://` pip
  entry.
- Scratch venv (`python -m venv --system-site-packages` on the env's python, then
  `pip install --no-deps <fork URL>`): pip resolved the URL, cloned blobless
  (`git clone --filter=blob:none`) at the branch tip `1110ffbafd651a74d1d4678deb4748056d1dff0a`,
  built `mace_torch-0.3.16+openqha-py3-none-any.whl`, and `import mace` resolved INTO the venv at
  `0.3.16+openqha` (venv pip 24.0 -- the bare URL form needs no `#egg=`; the conda env's editable
  install was left untouched, pip noting "Not uninstalling ... outside environment").
  `--no-deps`: the dependency set is the environment's business; the check scopes to the
  URL -> install -> import chain.
- The same venv answers `mace_fork_info()` = `unknown` (non-editable: no `.git` beside the
  package) -- the eval-side behaviour the rewritten comments state; only the editable install
  (via `install.sh`) yields a commit.

**Not verified / deferred:**

- A fresh environment built end-to-end from these files is [05e](05e-fresh-env-acceptance.md)'s
  proof, per this ticket's acceptance -- no environment was rebuilt here.
- `install.sh` itself lands with [05b](05b-the-install-script.md); the pointer is the spec'd
  forward reference.

**Reported beyond scope (not changed here):** re-running `install_dependency.sh` AFTER
`install.sh` reinstalls mace non-editable from `$REQ`, replacing the editable fork (pip compares
the direct git URL against the editable's local-path `direct_url.json`). Training then refuses
loudly at `check_fork(strict=True)`; the fix is re-running `install.sh`. The docs (05b/05d) should
phrase it as "`install.sh` last -- or again after any re-run of `install_dependency.sh`".

**Review (two axes, `1e1af86..88a9cab` openQHA):** one hard finding, caught on both axes and
fixed in this commit: the new pip-pass comment's two continuation lines had lost their leading
`#`, so every run would have executed them as commands ("command not found" before the pymsym
install) -- invisible to `bash -n` (bare words are valid syntax) and to the greps. The lines are
comments again. Judgement calls left standing: (a) the fork comment block repeats across the six
files -- kept, each file stands alone and this is the mace-md pattern; (b) "keep to one mace fork
commit" cites `engine/mace_fork_commit`, which the same files' non-editable install answers
`unknown` -- kept, the advice addresses training-side reproducibility (the editable install), and
eval Records carry `unknown` by design (05a decisions 8/10). Spec: every requirement and all five
acceptance boxes borne out by the diff and the recorded evidence; no scope creep. Carried, not
touched: `hpc/slurm/install_env_tianhe.slurm:285`/`:302` still call the environment files' mace
"the wheel" -- false now, and already [05d](05d-the-tianhe-path.md)'s file (§9, decisions 11-13).
