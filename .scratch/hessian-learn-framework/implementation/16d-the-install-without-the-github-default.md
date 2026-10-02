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
- [x] URL mode requires the explicit flag; CI calls it explicitly and is green.
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
the Tianhe pages for git-URL/pip-from-git lines is empty. The end-state identity
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

**CI.** `ci.yml` calls `bash install.sh --url`; the first run rides the next push.

