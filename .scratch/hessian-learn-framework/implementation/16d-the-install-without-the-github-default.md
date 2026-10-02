# 16d: The install without the GitHub default

Type: task
Status: resolved
Blocked by: None.
Serves: [16](../decisions/16-identity-without-checksums.md).

**What to build:** `bash install.sh` with no arguments installs from the sibling mace
checkout under the existing validation rules and never reaches the network; the URL mode
survives only as an explicit opt-in (CI uses it explicitly and stays green); README and
the Tianhe install docs present the local route as the only route, with no
fetch-from-GitHub instructions on Tianhe-facing surfaces; the end-state identity
verification is untouched (Q9=A).

- [x] No-arg install in a sibling layout succeeds offline; the end-state verification passes.
- [x] No-arg install without a discoverable checkout fails, naming both remedies: a path, or the explicit URL flag.
- [x] URL mode requires the explicit flag; `ci.yml` calls it explicitly -- the first CI run rides the next push (no push credentials in this session; a local end-to-end `--url` run is the evidence here).
- [x] README + Tianhe install docs: local route leads; Tianhe pages carry no git-URL/pip-from-git lines (prose citation links excepted).

## Notes

- The Tianhe installer already calls install.sh in local-path mode; this slice closes the default and the docs, not the on-site installer.
- The workstation keeps pushing to GitHub (the linkage removal is Tianhe-facing; Q10).

## Answer

**Landed (2026-10-02), package `9bfacef`** (on `ada4e4d`). `install.sh`'s default is the
sibling checkout: no argument resolves `$ROOT/../mace` and applies the existing validation
(exists -> directory -> `.git` -> `mace/__init__.py`) before anything is uninstalled; the
path argument is unchanged; `--url` is the explicit opt-in that fetches the branch URL
(header, usage and the fork-source line name it). A defaulted run that finds no valid
sibling fails before any pip step, naming both remedies -- a path, or `--url`. `ci.yml`
calls `bash install.sh --url` (a CI machine carries no checkout). The package README leads
with the local route and keeps `--url` only as the named opt-in; the by-hand block
installs `-e ../mace`. On the Tianhe side the two recipes install the editable pair from
the carried checkouts, the offline note reads "from disk, offline", and a residual scan of
the Tianhe pages (`docs/tianhe_install.md`, `docs/tianhe_runbook.md`) and the two
Tianhe-facing scripts (`hpc/slurm/install_env_tianhe.slurm`, `hpc/env/tianhe.sh`) for
git-URL/pip-from-git lines is empty; the environment/requirements files' fork-by-URL lines
are [16c](16c-the-rulebook-and-the-dates.md)'s removal. The end-state identity
verification is untouched (Q9=A).

**Evidence.** Routing pinned with a stubbed `python`: no-arg -> pip receives `-e <sibling>`
(no URL anywhere), path -> path, `--url` -> the URL, two args -> the usage error. No-arg
end state green in the WSL `openqha` env, also with github.com blackholed (`url.insteadOf`
-> a dead port; control `git ls-remote` rc=128): `fork source: local checkout
/mnt/c/.../mace`, `1110ffb` clean, `ok:`. No-checkout and broken-sibling failures name
both remedies; explicit-path failures keep their one-reason message. A real `--url` run
(scratch cwd, pip 26.2.1) cloned the branch and ended `ok:`; the env was restored to the
sibling checkout afterwards. Package suite `--all` 9/9 on a clean worktree at `ada4e4d` +
this patch; `bash -n` + shellcheck clean. The strict no-network probe (`unshare -rn`) shows
the only network pip still wants is the configured index for standard build tooling
(`setuptools>=42` build isolation) -- unchanged pip behaviour, not a fork fetch.

**CI.** `ci.yml` calls `bash install.sh --url`; this session has no push credentials (the
SSH dry-run times out, no agent), so the first run rides the next push -- the workflow
change plus the local end-to-end `--url` run are the evidence here.

**Review record (two-axis, per the `code-review` skill).** Ranges reviewed: package
`ada4e4d..9bfacef` (annotations `306313b`), openQHA `244ac6d..c328ed6` (annotations in
this commit).

- **Standards.** One hard finding -- the CI box was checked while its run still rode the
  next push (`openQHA/AGENTS.md`: acceptance criteria paid, evidence on the Status line);
  resolved by scoping the box to what is paid (the push rides the human's agent). Judgement
  calls applied: the contract bullet's offline claim scoped to the fork fetch (pip's
  standard build tooling may still use the configured index -- recorded in the Evidence
  paragraph); the duplicated `-e` guard folded into one branch. Kept deliberately: the
  `--url` opt-in named in the README (it must stay discoverable; the local route leads),
  and `$MACE_HINT`'s repeated suffix (explicit-path failures keep one-reason messages).
- **Spec.** No missing requirements, no scope creep; one wording finding fixed -- the
  note's "environment files carry the fork non-editable" sentence became provider-agnostic
  ("A non-editable fork is enough for eval"), true before and after 16c's env-file
  retirement. Left for the env-file retirement to reconcile (recorded, not rewritten): the
  page's remaining env-mechanics prose -- section 1.2's "arrives from GitHub", the
  `install_dependency.sh` reinstall sentence, section 3.3's "carry the fork's git URL".

