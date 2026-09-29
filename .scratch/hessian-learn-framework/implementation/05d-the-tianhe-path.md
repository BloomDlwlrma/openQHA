# 05d: The Tianhe path — §9, the transfer list, and one story

Type: task
Status: resolved
Serves: 05
Blocked by: 05b, 05c
Part of: [hessian-learn-framework](../map.md)

> Execution slice of [05a](05a-install-and-transport.md): spec decisions 11–13 and the Tianhe half
> of 15. Wayfinder conventions apply: the `Status` protocol, no triage labels
> (`docs/agents/issue-tracker.md`, ADR 0009).

## What to build

Tianhe's install job becomes the two-artifact install, and transport and documents are made to tell
one story:

- §9: install BOTH artifacts into every environment it manages (the CPU and GPU envs when present)
  by delegating to [05b](05b-the-install-script.md)'s script in local-path mode through each
  environment's own python. `MACE_FORK` keeps its name and now points at the mace checkout
  (default: sibling of the repository); a sibling variable names the package checkout. With no
  checkouts present: the explicit skip message, not a failure — eval environments remain fine.
- Transfer tool: the push list gains `../mace`, `../openQHA-Hessian`, `workflows` and `docs`;
  `.git` must ride along (no exclusion may skip it). Usage text updated.
- Documents: the Tianhe install document's fork/install block corrected to the two checkouts and
  the script; its stale old-fork narrative removed; the offline constraint stated once (installation
  at a networked moment; the AI side installs from the carried checkouts; the local-path argument
  is the hedge). The spec's §11 wording ("wheel path remains") is clarified to the post-05c
  reality — eval rides the environment-built non-editable fork.
- No Tianhe execution in this slice.

## Acceptance

- [x] `DRY_RUN=1 … push-repo` lists the two checkouts plus workflows and docs; nothing in the
      exclusion list can skip `.git`
- [x] §9 parses (`bash -n`); its environment loop covers openqha and openqha-gpu; the
      absent-checkout path prints the skip message instead of failing
- [x] The Tianhe install document matches the tool and the script (checkout names, no false claims,
      offline statement present)
- [x] No Tianhe run attempted
- [x] Reported per the operating rule

## Answer

Resolved 2026-09-29, agent-run. Tianhe's install job now puts the two-artifact training stack into
every environment it manages, the transfer tool carries exactly what that needs, and the document
tells the one story.

**What changed** (+95/-48 across the three files; this ticket and the map ride along):

- `hpc/slurm/install_env_tianhe.slurm` -- section 9 rewritten: per managed environment it runs
  `conda run -n <env> bash <pkg>/install.sh <mace>` -- [05b](05b-the-install-script.md)'s script
  in local-path mode, acting on that environment through its own python -- replacing the old
  inline wheel-replacement. `MACE_FORK` keeps its name and now points at the mace checkout
  (default `../mace`); the package checkout gets `HESSIAN_PKG` (default `../openQHA-Hessian`).
  Missing checkouts = the explicit skip message, exit 0, eval unaffected (the environment files
  install the fork non-editable by git URL, per 05c); an install failure is reported per
  environment without ending the job. The header usage block updated; `cd "$REPO"` gained its
  exit guard (SC2164 closed).
- `hpc/tools/xfer_tianhe_ai.sh` -- `push-repo` gains `../mace`, `../openQHA-Hessian`, `workflows`
  and `docs`; the checkouts ride as siblings of the repository (the loop's relative-path handling
  carries the `..` entries unchanged). `.git` protection: the old unanchored `--exclude 'logs'`
  also matched `.git/logs` inside a pushed checkout -- it is anchored (`'/logs'`) now, and no
  other entry in the list can touch `.git`. Usage and comments updated; the remote probe's nested
  quoting rewritten (SC2016/SC2027/SC2086) and the final idiom is an if/else (SC2015): the script
  is shellcheck-clean at default severity.
- `docs/tianhe_install.md` -- both hand-built pip blocks corrected to the post-05c reality (pip
  line = pymsym/parsl; mace = the fork's git URL, non-editable; machines that TRAIN run
  `bash ../openQHA-Hessian/install.sh ../mace` last). The "The mace fork" block rewritten: fork =
  `BloomDlwlrma/mace@openqha-hessian`, training side = the `openQHA-Hessian` package, the two
  checkouts + section 9 + the transfer tool; the stale old-fork narrative (old repo name, wheel
  story, wrong tool claim) is gone; the offline constraint is stated once; the §3.3 wheel-pin note
  replaced by the post-05c reproducibility note; the §1.2 mirror line no longer names the wheel.

**Evidence** (WSL, 2026-09-29; harnesses under `/tmp`; no Tianhe contacted, nothing installed):

- `bash -n` clean on both scripts; `shellcheck` v0.11.0: zero findings on both (before: SC2164 /
  SC2016+SC2027+SC2086+SC2015).
- `DRY_RUN=1 push-repo` with a stubbed `ssh` on PATH: lists `../mace`, `../openQHA-Hessian`,
  `workflows`, `docs` plus every previous entry; ends `done`.
- Exclusion proof (rsync itemized dry run with the file's own array eval'd): `.git/logs/HEAD`
  transferred under the new list (1) and withheld under the old unanchored list (0 -- the bug);
  root `logs/` still excluded.
- Section 9 extracted and run standalone: absent checkouts -> skip message, rc 0; present
  checkouts with a stubbed `conda` -> one `conda run -n <env> bash <pkg>/install.sh <mace>` call
  for `openqha` and one for `openqha-gpu`; the loop line carries both names literally.

**Not verified / deferred:**

- No Tianhe execution (the slice's prohibition) and no real `install.sh` run: the §9 harness
  stubbed `conda` on purpose, so no environment was mutated; the exact transfer of the sibling
  checkouts to a real remote is likewise unexercised (the dry run proves the list and the local
  path resolution only). The fresh-environment acceptance is
  [05e](05e-fresh-env-acceptance.md).
- The blobless fork checkout makes `git log -S` fetch historical blobs (it hangs): the flag
  spelling behind the doc's "`--hessian_probe modes` removed" was verified by filesystem greps
  (`--hessian_probe`, matching the parser) instead.
- No Python suite run: no Python changed.

**Reported beyond scope (not changed here):** `docs/tianhe_runbook.md:277` ("serves `mace-torch`
too") and `:296` ("`mace-torch`, `pymsym` and `parsl` ...") still carry wheel-era wording;
`openqha/training/run.py:215`'s fork-refusal message still advises the pre-split install
(`pip uninstall -y mace-torch && pip install -e <path>/openQHA-Hessian`) -- the training side's
callers are 07/01d territory. The tracker files `05a-install-and-transport.md`,
`05e-fresh-env-acceptance.md`, the two research notes and `spec-identity-after-the-split.md`
remain untracked in the working tree (pre-existing; not this slice's).
