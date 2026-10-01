# 15e: The close-out -- commits, suites, and the resume handoff

Type: task
Status: resolved
Blocked by: [15c](15c-the-amendments.md), [15d](15d-the-dbg-gate.md).
Serves: [15](../decisions/15-the-balance-on-the-probe-estimator.md) · spec: [spec-the-balance-on-the-probe-estimator.md](../spec-the-balance-on-the-probe-estimator.md).

**What to build:** the change lands and round 1 can resume. Commits on both repos (the
package: the estimator path + tests; openQHA: driver help + the amendments); suites
recorded (package `--all`, openQHA `--all`); the Tianhe deployment list (which commits /
files to sync to the checkouts); ticket 15 resolved with its Answer and one map line;
the timing-resume handoff -- the v3 submit block is unchanged (the balance is now
minutes-scale), and the runbook's held-submission sequencing is confirmed.

- [x] Commits landed with recorded hashes (pushes stay the user's).
- [x] Suite results recorded (`--all` for both repos).
- [x] Deployment list written: repos, commits, files to sync to both Tianhe checkout sides.
- [x] Ticket [15](../decisions/15-the-balance-on-the-probe-estimator.md): Answer + `Status: resolved` + one map line (`Implementation`).
- [x] Resume notes: the timing job's submit block as-is; then cap -> the two arms.

## Notes

- The timing/arms submissions themselves are ops (spec Out of Scope); this slice only
  makes them unblocked and hands the steps back.

## Answer (2026-10-01)

**Commits landed** (recorded hashes; pushes stay the user's):
- `openQHA-Hessian` (main) -> tip **`547c394`**: `bf0abd6` (15a -- `openqha_hessian/hvp.py`
  and `smoke_fit.py` + `tests/unit/t_smoke_fit.py`, `tests/integration/t_hvp_engine.py`),
  `05ac7b1` (15b -- `run.py` + `tests/unit/t_train_run.py`,
  `tests/integration/t_train_engine.py`), `547c394` (the 05ac7b1 review annotation).
  Pushed (reflog 2026-10-01 17:51; `ls-remote` verified `547c394` on
  `BloomDlwlrma/openQHA-Hessian`).
- openQHA (main) -> tip **`81e51e6`** + this record's commit: `3551273` (15b --
  `workflows/hessian_learning/05_train.py`: `--hessian-weight` help + balance print),
  `326dcc5` (15c -- `spec-phl-verbatim.md`, 09g), `81e51e6` (15d -- the gate numbers into
  ticket 15), plus the tracker commits `c66833c` (15a's record, which opens the series)
  and this record's commit (which also brings
  `spec-the-balance-on-the-probe-estimator.md` into the tree; untracked until now).
  Pushed through `81e51e6` (reflog "update by push": 17:50 and 18:35); this record's
  commit stays local for the user's push.
- The untouched side: `mace` @ `1110ffb` (frozen; ticket 15 changes no fork file).

**Suites (2026-10-01, the close-out tree).** Package `--all` **9/9** (done 18:42:50);
openQHA `--all` **73/73** (done 18:49:51); both rc 0 -- log `%TEMP%\oqt15e-suites.log`,
sentinel `OQT15E-SUITES-RC pkge=0 oq=0 2026-10-01 18:49:51`; the header names the two
checkouts and `mace-torch 0.3.16+openqha`. Same verdict as 15d's run earlier today
(18:12-18:22); these are the close-out numbers.

**Deployment list -- both Tianhe checkout sides** (CN/A side root
`/XYFS02/HDD_POOL/hku2021_fos4/hku2021_fos4xy_2/sherwin`; AI side root
`~/HDD_POOL/sherwin`; the three checkouts under each):

| repo | from (09e/09g state) | to | files | notes |
|---|---|---|---|---|
| `openQHA-Hessian` | `411abe1` | **`547c394`** | `openqha_hessian/hvp.py`, `smoke_fit.py`, `run.py`; `tests/unit/t_smoke_fit.py`, `tests/unit/t_train_run.py`, `tests/integration/t_hvp_engine.py`, `tests/integration/t_train_engine.py` | THE change -- the envs' editable installs read this checkout; no rebuild (the import picks the new code up) |
| `openQHA-main` | `5b0b5b9` (ZIP-time record; the id is not checked, per 09e) | the close-out tip (after the user's push) | ticket 15's file: `workflows/hessian_learning/05_train.py`; the post-09e window also carries `hpc/slurm/hl_train.slurm`, `workflows/hessian_learning/README.md`, the `docs/` pages + T05 notebook, `README.md`, `openqha/potentials/engine.py`, `openqha/store/record.py`, two test files | help/print only -- the runs read the package |
| `mace` | `1110ffb` | `1110ffb` | (unchanged) | frozen |

Route as in 09e (network git stays unreliable through the proxy): repack the two `.git`
tars from the workstation trees, carry them + the fresh ZIPs (after the user's push) via
the usual channel, overlay, then verify -- receipt 1's ids (09g's 前置) plus the
identity checkpoints (`mace` clean, package clean); end-to-end evidence: the timing run's
Record reads `HL_PACKAGE_COMMIT = 547c394...`. `install.sh` re-run optional (idempotent);
the editable installs need no env rebuild.

**Resume -- the timing job, then the cap, then the two arms** (09g unchanged):
- Submit the held timing job with 09g step 1's block AS-IS (`yhbatch -p a800x --gpus=1
  -t 12:00:00 hpc/slurm/hl_train.slurm`, `MAX_EPOCHS=2`, the `_w1` files, no `--register`);
  the balance now runs minutes-scale. Receipt 2 reads `SECONDS_PER_EPOCH` +
  `HESSIAN_WEIGHT` + `BALANCE_PROBE`/`BALANCE_N_PROBES` + `HESSIAN_CURVE_MOVED`.
- Then 09g step 2's cap (the workstation confirms the `CAP=` before the arms), then step
  3's two production arms in one submission (`--register --register-copy` each).
- Sequencing confirmed: the refresh is the only gate before the timing resumption.

**Review (two-axis, per the `code-review` skill).** Two read-only sub-agents over the
close-out change set (working tree vs `81e51e6`; the diff beside the records in
`%TEMP%\oqt15e-review.diff`):
- Standards: one hard finding fixed in-pass -- the operating-rule report (files changed /
  checks run / not verified) was missing and is now below. Judgement calls kept with
  reasons: the tracker-required duplication across ticket / map / record (the documented
  index design overrides; 15d's review recorded the same), the spec's `ready-for-agent`
  label and header-paragraph format (inherited from the sibling specs), and the 15e
  checklist's "one map line (`Implementation`)" read with `docs/agents/issue-tracker.md`'s
  Resolve rule -- ticket 15's line in `Decisions so far`, 15e's in `Implementation`, one
  each.
- Spec: no missing requirements (all five boxes evidenced); the deployment section's
  route/evidence lines judged consistent with the handoff asked for (kept). Fixed
  in-pass: "trailer commits" miscounted `c66833c` (it opens the series; now "tracker
  commits"), the "mid-day" slip (the 15d suites ran 18:12-18:22), and ticket 15's
  paragraph now names `c66833c`. The "v3 submit block" label kept as the brief's
  wording -- the operative reference is 09g step 1, submitted as-is.

**Reported per the operating rule.** Files changed (this close-out):
`decisions/15-the-balance-on-the-probe-estimator.md`, `map.md`,
`implementation/15e-the-close-out.md` (new), `spec-the-balance-on-the-probe-estimator.md`
(new to the tree). Checks run: the two suites (`--all`; 9/9 + 73/73, rc 0); the
working-tree diff vs `81e51e6` and `git status` (exactly the four files); the two-axis
review above.

**Not verified.** The Tianhe refresh and the three submissions are the user's ops (spec
Out of Scope); this slice hands them the list above.
