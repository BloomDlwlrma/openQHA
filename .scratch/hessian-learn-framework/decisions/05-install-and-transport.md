# Install and transport: the new install.sh, Tianhe, and the xfer gap

Type: task
Status: open
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

<!-- resolver: append what was done/decided; set Status: resolved; add a line to the map's Decisions so far -->
