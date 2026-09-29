# The mace-md install pattern (and the openQHA-Hessian install it implies)

Type: research — feeds ticket 05 ("Install and transport")
Date: 2026-09-28
Method: primary sources only — the mace-md copy in this workspace, pip documentation, conda source,
and empirical checks in a throwaway WSL venv (commands and observed outputs below). No repository
edits; scratch under `/tmp` cleaned up.

## 1. What mace-md does (verbatim, from `source-code/mace-md-master/`)

`mace-install.sh` (whole file — 8 lines):

```bash
#!/bin/bash -l
mamba env create -f mace-openmm.yml

conda activate mace-openmm

pip install git+https://github.com/jharrymoore/mace.git@softcore
pip install git+https://github.com/jharrymoore/openmm-ml.git@ml_alchemy
pip install .
```

`mace-openmm-full.yml:19-24` (the pip section; the other yml has none):

```yaml
  - pip
  - pip:
    - mpi4py
    - git+https://github.com/jharrymoore/mace.git@softcore
    - git+https://github.com/jharrymoore/openmm-ml.git@ml_alchemy
    - git+https://github.com/jharrymoore/mace-md.git
```

`README.md:8` (verbatim, including the original's "build" typo): "The base conda environment can be
build from `mace-openmm.yml`, additional requirements can be found in `mace-install.sh`".

Observation: the project declares its two forks **and itself** as bare git-URL pip dependencies —
branch names, no commit pin, no `-e`, no wheels, no hashes. The whole repo says nothing about
provenance, editable installs, clusters, or offline installs.

## 2. The pip mechanics this pattern rests on (verified)

### (a) Non-editable `git+…@branch` leaves nothing behind

pip docs (VCS Support, pip.pypa.io/en/stable/topics/vcs-support/): "For non-editable installs, the
project is built locally in a temp dir and then installed normally."
Empirical (`pypa/sampleproject`, pip 23.2.1 venv): clone goes to `/tmp/pip-*`, code lands in
`site-packages`, no `.git` anywhere on disk afterwards. So this form is, for our provenance purposes,
identical to the PyPI wheel: `mace_fork_info()` finds no `.git` and answers `unknown`.

### (b) Editable `-e git+…@branch` KEEPS the clone, with `.git`

pip docs: "The default clone location (for editable installs) is: `<venv path>/src/SomeProject` in
virtual environments; `<cwd>/src/SomeProject` for global Python installs. The `--src` option can be
used to modify this location."
Empirical: `Cloning … to /tmp/piptest-ed2/src/sampleproject`; `<src>/sampleproject/.git` present;
the import resolves from inside the clone.

This is exactly what `mace_fork_info()` needs — `openqha/potentials/engine.py:350-351`:
`root = Path(module_file).resolve().parent.parent` / `if not (root / ".git").exists(): return unknown`.
mace is a **flat** layout (`mace/mace/__init__.py`), so a clone at `<src>/mace` gives
`parent.parent = <src>/mace`, where `.git` sits: the editable VCS install satisfies the fork contract
without any manual clone recipe. (A src-layout project could not; mace is not one.)

### (c) pip does a blobless partial clone by default

pip docs: "Pip requests partial clones by default when using Git 2.17 or later… setting
`PIP_NO_PARTIAL_CLONE_FOR_BROKEN_GIT_SERVER=1` will force pip to ignore the claim and use a full clone."
Empirical `GIT_TRACE=1` (git 2.43.0, pip 26.2.1) caught the real command lines:

```
trace: built-in: git clone --filter=blob:none --quiet https://github.com/pypa/sampleproject /tmp/piptest-ed3/src/sampleproject
trace: run_command: git … fetch origin … --filter=blob:none --stdin
```

plus `remote.origin.promisor=true`, `partialclonefilter=blob:none`, full commit graph. i.e. the URL
form is the same weight class as the 02d recipe (`git clone --filter=blob:none`), not a full-history
download.

### (d) Syntax sensitivity across pip versions

- pip 23.2.1 rejects `-e "name @ git+…@branch"` ("… is not a valid editable requirement");
  `-e "git+…@branch#egg=name"` works.
- pip 26.2.1 (the WSL `openqha` env's pip) accepts the direct form.
Portable spelling keeps `#egg=`, or the docs state the floor.

### (e) Conda YAML pip sections are requirements files

conda 24.1.2 source, `conda/env/installers/pip.py:17-56` (read-only): the pip specs are written into
a temporary requirements file and installed with `pip install -U -r <file>`; each YAML list entry is
one requirements-file line, and pip's Requirements File Format supports `[-e] <vcs project url>`.
⇒ `- git+…` entries in an environment yml are legal (mace-md proves it empirically); `- -e git+…`
entries are legal by the same mechanism. (mamba's behaviour was not separately tested.)

## 3. Implications for ticket 05 (proposal this note supports; not yet settled)

1. `openQHA-Hessian/install.sh` — the mace-md script with the one deviation we cannot skip
   (provenance needs `-e`):

   ```bash
   python -m pip uninstall -y mace-torch || true
   python -m pip install -e "git+https://github.com/BloomDlwlrma/mace.git@openqha-hessian"
   python -m pip install -e .
   # then verify: mace.__file__ inside the checkout, version 0.3.16+openqha, .git beside the
   # package, mace_fork_info() clean; print the fork and package commits.
   ```

   A local-checkout path argument replaces the URL where GitHub is not reachable (Tianhe AI side) —
   one optional argument; no second "debug" script is needed.
2. Env / requirements files: the `mace-torch` pins become the fork's git URL
   (`git+https://github.com/BloomDlwlrma/mace.git@openqha-hessian`), mace-md style. Note the split
   the env-file form cannot remove: a non-editable URL install has no `.git`, so it only guarantees
   "import mace is the fork" for eval; training machines run `install.sh` (editable) regardless.
   mace-md has no such split because it records no provenance.
3. No wheels anywhere: not built, not transported, no fallback. Install happens where network exists;
   where it does not, the carried checkouts are handed to the script.

## 4. Unverified / cautions

- Conda envs: the `<venv>/src` default was verified in a venv; for a conda env the same rule is
  expected (pip's virtualenv test) but not tested — deliberately no install into the conda env. The
  only consequence is where the clone lands; pass `--src` to pin it if it matters.
- mamba vs conda pip-section behavior: not tested.
- GitHub reachability from Tianhe's login node through the site proxy: still unverified; the
  local-path argument is the hedge.
- mace clone size via pip: not measured (deliberately no mace clone in the test); blobless makes it
  metadata-bound, same class as the 02d clone.

## Sources

- `source-code/mace-md-master/`: `mace-install.sh` (whole file), `mace-openmm-full.yml:19-24`,
  `README.md:8`. Copies elsewhere in the workspace are byte-identical (27-file SHA-256 check).
- pip docs: https://pip.pypa.io/en/stable/topics/vcs-support/ (quotes in §2a–2c).
- `openQHA/openqha/potentials/engine.py:350-351` (the `.git` check `mace_fork_info()` performs).
- conda 24.1.2 `conda/env/installers/pip.py:17-56` (temporary requirements file; `install -U -r`).
- Empirical session: WSL, git 2.43.0, pip 26.2.1 (openqha env) and 23.2.1 (temp venv); repo
  `pypa/sampleproject`; scratch cleaned.
