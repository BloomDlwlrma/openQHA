# 16e: The Tianhe landing

Type: task
Status: open
Blocked by: [16a](16a-the-record-without-the-checksum.md), [16b](16b-the-reference-data-without-the-checksums.md), [16c](16c-the-rulebook-and-the-dates.md), [16d](16d-the-install-without-the-github-default.md), [15](../decisions/15-the-balance-on-the-probe-estimator.md).
Serves: [16](../decisions/16-identity-without-checksums.md).

**What to build:** with 16a-16d committed, the Tianhe side is brought to the new state
in one sitting -- both code trees are overlaid in place (merge-only: `unzip -o` of the
new-commit ZIPs carried from GitHub, downloaded after the workstation push) and the AI
environment's `openqha_hessian` editable install is re-pointed directly at the XYFS02
tree, after which the repair-era AI-side symlink is deleted. Tianhe gets no operator
git action (no receipts, no writes, no tree surgery) and no cleanup: the site `.git`
is left exactly as it is (a stale, "modified" status is expected and harmless), and
every old / renamed-aside directory stays for a later cleanup.

- [ ] Both trees overlaid from the new ZIPs; the receipt is the content probes below (no git receipts, no sha256). The deployed heads are recorded workstation-side in this ticket's Answer.
- [ ] `import openqha_hessian` resolves to `$S2/openQHA-Hessian` (`/XYFS02/...`) with no symlink in the path; the AI-side link is deleted (`rm`; renamed aside only if the mount refuses the unlink).
- [ ] Spot check passes: AI env `check_fork(strict=True)` + package runner `--all`; the two CN envs' import check.
- [ ] No operator git receipt or write ran on Tianhe; the guard's built-in read-only reads stayed; nothing old was cleaned.

## Notes

- **Revision (2026-10-02, from the grilling) -- deltas against the first write:** the
  tarball-carried checkout replacement became a plain overlay of code ZIPs (no `.git`
  is carried or updated); the git receipts (`git log -1`, `status`, `remote -v`) are
  out, replaced by content probes + the workstation-side record of the pushed heads;
  the "remotes stripped" item is retired (no operator git on Tianhe: no receipts, no
  writes); the timing gate "wait for the arms" became "as soon as the ZIPs are on
  site"; the link removal is a delete after the re-point, not a directory swap.
- **Timing (user ruling 2026-10-02):** do it as soon as the ZIPs are on site; avoid
  only each arm's stage-1 -> stage-2 handoff moment (watch `squeue`; expected around
  2026-10-05). A running stage-1 is unaffected (its code is loaded); a pending stage-2
  will load the new code -- 16a-16d touch no loss numerics, so the run's numbers are
  unchanged (only its Record differs in the retired `CONFIG_SHA256` field and the
  registry `source` string).
- **Why the overlay is safe (checked 2026-10-02):** both diffs (site state -> tip)
  carry no file-level deletions or renames, so the merge-only copy leaves no stale
  file to shadow the new run; untracked site data (weights, SPICE, logs) and every
  `.old-*` directory are untouched by construction.
- **Steps** (`S2=/XYFS02/HDD_POOL/hku2021_fos4/hku2021_fos4xy_2/sherwin`):
  1. Carry `openQHA-main.zip` + `openQHA-Hessian-main.zip` to the login node's `~`
     (workstation downloads them after the push; the download page names the heads --
     at writing `7c74b1a` / `306313b`; record them in the Answer).
  2. Overlay both trees (nothing is deleted):
     `unzip -o -q openQHA-main.zip -d $S2/_incoming` (and the package zip);
     `cp -a $S2/_incoming/openQHA-main/. $S2/openQHA-main/` (and the package).
  3. AI side, in the environment that carries the old editable -- order matters,
     re-point then verify then cut: `source ~/init_conda.sh && conda activate
     openqha-gpu`; `pip install -e $S2/openQHA-Hessian` (`--force-reinstall --no-deps`
     if pip skips; `--no-build-isolation` if the index is unreachable); verify the
     import path prints `/XYFS02/...` and `check_fork(strict=True)` is green;
     `rm ~/HDD_POOL/sherwin/openQHA-Hessian` (the LINK only; if the mount refuses,
     `mv` it aside as `...link-old-<date>`).
  4. Receipt probes + spot check below; paste the outputs back.
- **Receipt probes** (from each tree's root; expected value in brackets; `grep -c`
  exits 1 on count 0 -- expected): openQHA: `test -f
  docs/adr/0014-identity-without-checksums.md` [exists]; `sha256_head` in
  `scripts/tooling/s0_prepare_data.py` [0]; `edge_list_sha256` in `openqha/config.py`
  [0]. Package: `CONFIG_SHA256` in `openqha_hessian/run.py` [0]; `sha256_file` in
  that file [0]; `fork source: local checkout` in `install.sh` [1].
- **Spot check:** AI env, after the re-point: the import-path line,
  `check_fork(strict=True)`, then `cd $S2/openQHA-Hessian && python tests/run_tests.py --all`
  (9/9). CN side: the import-path line in both `openqha` and `openqha-gpu` (their
  installs are untouched; both already point at `$S2`).
- **The site's git is not a surface:** no git write command may ever run on the trees
  (`checkout` / `reset` / `stash` would restore the old code); `git status` showing a
  mass of modified files after the overlay is expected. The only git that runs on site
  is the product's own built-in read-only guard (the fork check's `rev-parse` /
  `status` reads), which is not an operator action. `mace` is untouched (no new
  commit).
- The old AI-side directories (already renamed aside) stay; their deletion is a later
  cleanup (storage there is unstable). `$S2/_incoming/` stays as well.
