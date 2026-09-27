# 01c: The loss core — `phl` and `phl_loss` follow

Type: task
Status: open
Serves: 01
Blocked by: 01b
Part of: [hessian-learn-framework](../map.md)

> Execution slice for [The package line](../decisions/01-the-package-line.md); the brief is
> [01a](01a-the-package-line.md). Repeats 01b's pattern for the remaining openqha-free
> modules; after this slice, every module of the training side that does not touch openQHA
> has left. Agent-run, no user action required. Wayfinder conventions apply: the `Status`
> protocol, no triage labels (`docs/agents/issue-tracker.md`, ADR 0009).

## What to build

`phl` and `phl_loss` join `hvp` in `openqha_hessian` under their names (`phl_loss`'s
sibling imports become package-internal), with `t_phl` and `t_phl_loss` moving to the
package's groups under the twin runner.

Because `phl_loss` is the module the fork's `--loss_module` address names, the address
flips **in this slice**, the moment the module leaves: the training run's loss-module
constant carries `openqha_hessian.phl_loss:build`, and every test or script that pins the
old address follows — including `t_train_run` and `t_train_engine`, which do not move
until [01d](01d-the-switchover.md) but must keep passing in between. Old Records stay
untouched; only new writes carry the new address.

Everything still living in openQHA gets its outside-facing imports for the two modules
pointed at the package while its remaining `openqha.training` imports stay until 01d:
`judge`, `run` and `smoke_fit` (which read them), the tooling scripts that reach in
(`s0_probe_calibration`, `s0_spice_pt_draw`, and `s0_hl_smoke_fit`'s share),
`t_probe_calibration` (which stays put — its subject is the tooling script), and the few
tests whose references sit inside functions. Mixed addresses across these files are
expected in this slice and must not bend any seam.

## Acceptance

- [ ] The package runner passes `t_phl` and `t_phl_loss` beside 01b's tests
- [ ] openQHA's `tests/run_tests.py --all` passes, `t_train_run`'s and `t_train_engine`'s loss-address assertions included
- [ ] `05_train --dry-run` prints `--loss_module openqha_hessian.phl_loss:build` — the effort's one argv change — and nothing else differs
- [ ] The new address resolves end to end: the string the argv carries imports and its `build` runs (no training run needed)
- [ ] No module in `openQHA/training/` imports `phl` or `phl_loss` through the old path
- [ ] Reported per the operating rule: files moved, checks run, anything not verified
