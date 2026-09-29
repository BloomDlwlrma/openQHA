# Install and transport: the new install.sh, Tianhe, and the xfer gap

Type: task
Status: resolved
Blocked by: 01, 02
Part of: [hessian-learn-framework](../map.md)

## Question / work

After the split, one documented path must put a machine into the working state — the mace fork from `BloomDlwlrma/mace` plus `openQHA-Hessian` as the package — locally in WSL and on Tianhe, where GitHub may not be reachable. The mace-md pattern is the model: its `mace-install.sh` runs `pip install git+https://github.com/jharrymoore/mace.git@softcore` and then installs its own package.

Work:

1. Write `openQHA-Hessian/install.sh` (name per [The package line](01-the-package-line.md)): install the fork (git URL pinned to the rebuilt branch; `--no-deps`? both orders?) and the package itself — editable where provenance needs it (`mace_fork_info()` requires `.git` at the mace checkout root; training refuses `unknown`/dirty).
2. Rewrite `hpc/slurm/install_env_tianhe.slurm` §9 (lines 284–303) for the two-artifact install and the new meaning of `MACE_FORK`.
3. Fix the transport truth: `hpc/tools/xfer_tianhe_ai.sh`'s push-repo list (lines 127–133) does **not** carry `openQHA-Hessian/`, although `docs/tianhe_install.md:284–291` says it does. Decide how the fork and the package reach Tianhe (xfer the checkouts? wheels? a tarball cache?), and make the script and the docs agree.
4. Clean up the pins that name the old world: `environment*.yml` / `requirements*.txt`'s `mace-torch>=0.3.6`, `install_dependency.sh`'s handling — the fork is no longer just "the PyPI wheel".
5. Decide what "install" means for a machine that must run `05_train` (editable mace checkout required) vs one that only evaluates.

## Acceptance

- A fresh WSL env from the documented lines ends with `import mace` resolving to the fork checkout, `mace_fork_info()` clean, and `05_train`'s fork check passing.
- Script and documentation tell the same story; the Tianhe path is decided with the offline constraint stated.

## Answer

Resolved 2026-09-29, agent-run, on the fresh-environment acceptance
[05e](../implementation/05e-fresh-env-acceptance.md) — the effort's single testing seam,
green. One install script puts the training stack into an activated environment (the fork and
the package, both editable, verified; a local-path mode is the offline/Tianhe form); the
environment and requirements files carry the fork by git URL, non-editable, so building an
environment installs the fork and never the wheel; Tianhe's transfer tool carries the two
checkouts (`.git` included) and its install job delegates to the script per managed
environment; no wheels. Durable record:
[ADR 0012](../../../docs/adr/0012-install-and-transport.md). Findings the design leaned on:
[`research/mace-md-install-pattern.md`](../research/mace-md-install-pattern.md).

**The recorded acceptance run** (WSL, 2026-09-29 12:40–12:45 CST; full log 380 lines, zero
non-zero exit codes: `C:\Users\10704\AppData\Local\Temp\oqha-05e\accept.log`; solver and
pip-dependency noise elided, everything else verbatim). Two environment-local deviations from
the documented lines, both stated: the environment name is overridden to `oqha-05e` (the
documented `openqha` is this machine's live development environment), and `env_openqha.sh`
does not exist on this machine, so the README's "or by hand" exports stand in.

```text
$ conda env create -y -n oqha-05e -f .../openQHA/environment.yml
  ...
  Resolved https://github.com/BloomDlwlrma/mace.git to commit 1110ffbafd651a74d1d4678deb4748056d1dff0a
  Successfully installed ... mace-torch-0.3.16+openqha ...
(exit 0)

$ conda activate oqha-05e
$ python -V
Python 3.11.16
$ python -m pip --version
pip 26.2.1 from /home/ubuntu/anaconda3/envs/oqha-05e/lib/python3.11/site-packages/pip (python 3.11)

$ python -m pip show mace-torch          # what the env file installed, before install.sh
Name: mace-torch
Version: 0.3.16+openqha
Location: /home/ubuntu/anaconda3/envs/oqha-05e/lib/python3.11/site-packages
                                         # non-editable -- no .git beside it (the eval state)

$ export OPENQHA_ROOT=.../openQHA; export PYTHONPATH=$OPENQHA_ROOT
$ python -c 'import openqha; print(openqha.__file__)'
/mnt/c/.../openQHA/openqha/__init__.py

$ bash .../openQHA-Hessian/install.sh    # documented default: the fork from its branch URL
==> fork source: git+https://github.com/BloomDlwlrma/mace.git@openqha-hessian#egg=mace-torch
==> environment: Python 3.11.16 at /home/ubuntu/anaconda3/envs/oqha-05e/bin/python
==> [1/4] uninstalling any mace-torch distribution
  Successfully uninstalled mace-torch-0.3.16+openqha
==> [2/4] installing the mace fork editable
  Cloning https://github.com/BloomDlwlrma/mace.git (to revision openqha-hessian) to ./src/mace-torch
  Running command git clone --filter=blob:none --quiet https://github.com/BloomDlwlrma/mace.git .../openQHA/src/mace-torch
  Successfully installed mace-torch-0.3.16+openqha
==> [3/4] installing openqha-hessian editable
  Successfully installed openqha-hessian-0.1.0
==> [4/4] verifying the end state
  mace            0.3.16+openqha  .../openQHA/src/mace-torch/mace/__init__.py
  fork commit     1110ffbafd651a74d1d4678deb4748056d1dff0a
  mace_fork_info  1110ffbafd651a74d1d4678deb4748056d1dff0a  dirty=False  path=.../openQHA/src/mace-torch
  openqha_hessian .../openQHA-Hessian/openqha_hessian/__init__.py  (package commit ff92141)

ok: the fork is installed editable and openqha-hessian imports
==> done
(exit 0)

$ python -c 'import mace; print(mace.__version__); print(mace.__file__)'
0.3.16+openqha
/mnt/c/.../openQHA/src/mace-torch/mace/__init__.py

$ python -c 'from openqha.potentials.engine import mace_fork_info; print(mace_fork_info())'
{'mace_fork_commit': '1110ffbafd651a74d1d4678deb4748056d1dff0a', 'mace_fork_dirty': False, 'mace_fork_path': '/mnt/c/.../openQHA/src/mace-torch'}

$ python -c 'from openqha_hessian import run as train_run; print(train_run.check_fork(strict=True))'
{'mace_fork_commit': '1110ffbafd651a74d1d4678deb4748056d1dff0a', 'mace_fork_dirty': False, 'mace_fork_path': '/mnt/c/.../openQHA/src/mace-torch', 'mace_version': '0.3.16+openqha'}

$ python -c 'import openqha_hessian; print(openqha_hessian.__file__)'
/mnt/c/.../openQHA-Hessian/openqha_hessian/__init__.py

$ python -m pip show mace-torch openqha-hessian
  mace-torch      0.3.16+openqha  Editable project location: .../openQHA/src/mace-torch
  openqha-hessian 0.1.0           Editable project location: .../openQHA-Hessian
(exit 0)

===== SENTINEL-05E-DONE failures=0 =====
```

**Findings recorded with the run.**

- *The clone location — 05a's watch-list item, resolved, with a consequence for the docs.*
  05a expected a conda editable VCS clone in the environment's `src`; it lands in `<cwd>/src`:
  pip 26.2.1 classifies a conda environment as a global install
  (`running_under_virtualenv() == False`; `pip._internal.locations.get_src_prefix`), so the
  run from the README's implied cwd (the openQHA root) put the clone at
  `openQHA/src/mace-torch`; `src/` is not git-ignored, so a doc-following user sees `?? src/`.
  **Reported, not fixed here** — candidates: `install.sh` passing `--src "$CONDA_PREFIX/src"`
  to the fork step, or a docs line.
- *The state the acceptance exercised.* The package checkout's HEAD at run time was `ff92141`
  ("The switchover: judge, run and smoke_fit move in — the Records carry the package identity",
  the [01d](../implementation/01d-the-switchover.md) session, 12:42:00) — so the probe exercised
  the moved `run.check_fork` as shipped. The openQHA side of the switchover was still in flight
  in the working tree.

**Not verified / deferred.** No suite ran in the fresh env (the spec's testing decision: the
probes are the seam; no suite tests were added); no weights were fetched or loaded (nothing
weights-related was probed); the Tianhe path was not executed (05d's evidence stands) and
Tianhe's GitHub reachability through the site proxy stays unverified (05a's watch-list: the
fallback is a one-line change to the environment files' fork requirement; §9's carried-checkout
install is the primary path); URL mode only — the local-path mode is 05b's and 05d's evidence.
The scratch environment and the clone (4.9 MB) were removed after the run; nothing remains in
the working tree (`git status -- src` clean).

**Review (two axes, `890bade..0baaedd` openQHA):** no hard findings. Standards: tracker
conventions, the `Status` protocol and the ADR's structure/voice all clean; one near-convention
item fixed — the map's "Not yet specified" fog line for package release artifacts had its
trigger met ("Sharpens once *Install and transport* picks a mechanism") and is retired (the
subject is ticket 13's), the map's "Slices:" line now links like its peers, and the run
findings were thinned to a single home here with 05e pointing at it. Spec: every requirement
and all four acceptance boxes borne out by the recorded log; no scope creep; three
record-level corrections in this commit — the ADR no longer says the refusal "names the fix"
(the repair hint still names the pre-split install; reported by 05d, owned by 12), the ADR now
marks the two recorded deviations from the spec's letter (the `#egg=` fragment; eval riding
the non-editable fork rather than a wheel), and the watch-list record is now truthful in both
directions: this Answer restates the still-open Tianhe GitHub-reachability item, and 05a
carries a dated amendment resolving the clone-location item.
