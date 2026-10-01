# 15b: The driver and the Record carry the estimator

Type: task
Status: resolved
Blocked by: [15a](15a-the-balance-estimator.md).
Serves: [15](../decisions/15-the-balance-on-the-probe-estimator.md) · spec: [spec-the-balance-on-the-probe-estimator.md](../spec-the-balance-on-the-probe-estimator.md).

**What to build:** a fine-tune with `--hessian-weight balance` (the default) produces a
Record that states its provenance -- the rule stays `balance`, and `BALANCE_PROBE` /
`BALANCE_N_PROBES` say which estimator produced the value -- with a value an independent
recomputation reproduces; the driver's balance print and help stop saying "Cartesian
target"; a `05_train --dry-run` on the dbg subset shows the balance line in minutes,
not hours.

- [x] The balance helper passes the run's probe kind, `k`, and seed through; `run_training` wires the run's settings into it.
- [x] The Record gains `BALANCE_PROBE` / `BALANCE_N_PROBES`; the `HESSIAN_WEIGHT_RULE` and `BALANCE_L_H` descriptions carry the new reading, including the comparability note for pre-amendment Records (their value is exact); `L_E` / `L_F` unchanged.
- [x] Integration check: a fixture run's Record shows `rule = balance` + the two fields, and the value matches a direct estimator recomputation within the closed-form tolerance.
- [x] The driver's `--hessian-weight` help, the module's "THE WEIGHT" paragraph and the helper's docstring no longer say "Cartesian target, the full matrix".
- [x] `05_train --dry-run` on `draw300_r1dbg` prints the balance line and finishes in minutes (recorded).

## Notes

- The print surface itself is unchanged mechanically (the loss line already carries
  `w_H <value> (balance)`); only the provenance around it changes.
- Prior art: `t_train_engine`'s dry-run and Record checks; the 09f dry-run procedure.

## Answer (2026-10-01, package `05ac7b1` + annotations `547c394`)

**What landed.** The balance runs under the run's probe setting. `hessian_weight_balance`
takes `probe` / `n_probes` / `seed` (defaults: gaussian, k=4, seed 123) and passes them into
`smoke_fit.epoch_zero_balance`; `run_training` wires the run's `PROBE` / `N_PROBES` / `SEED`
into it. The Record gains `BALANCE_PROBE` (String) / `BALANCE_N_PROBES` (Integer) -- `-` / 0
when the rule is `given` -- beside the amended `HESSIAN_WEIGHT_RULE` / `BALANCE_L_H`
descriptions (measured with the run's probe setting; the exact full-matrix reading stays the
anchors' and the judge's job; Records written before the estimator amendment hold the exact
full-matrix reading); `L_E` / `L_F` untouched. The train.out balance note names the
estimator, and the driver's `--hessian-weight` help and balance print say "the run's probe
setting" / `(gaussian k=4)`. No new modules, no dependencies, no fork changes.

**Evidence.**
- `t_train_run` **42/42** (executed; the four paired refusal branches contribute one site
  each): the helper's pass-through -- explicit `rademacher k=7 seed=42` and the gaussian k=4
  seed=123 defaults land in `epoch_zero_balance` (patched calculator + balance) -- and the
  schema checks (both fields typed; the amended descriptions pinned).
- `t_train_engine` **32/32**: the given rule's Record carries `("-", 0)`; the balance run's
  Record carries `BALANCE_PROBE = gaussian`, `BALANCE_N_PROBES = 4` = the run's
  `(PROBE, N_PROBES)`, and `BALANCE_L_H` reproduces a direct `hessian_weight_balance`
  recomputation to 1e-12 (`3.201798e-02` vs `3.201798e-02`; the same `w_H`).
- The dbg dry-run, default balance rule:
  `05_train.py --tag draw300 --name draw300_r1dbg --run dry15b --dry-run` ->
  `DRYRUN-RC=0 WALL=59s`; the line:
  `balance base model on the train file: L_E 5.190e-07 L_F 1.734e-04 L_H 2.122e-03 (gaussian k=4) -> w_H = w_F L_F / L_H`,
  resolving `w_H = 8.170329761632427` (seed 123; run dir `<mirror>/draw300_r1dbg/train/dry15b`).
  For scale: the exact full-matrix pass on this same 92-frame train file was ~7 min in 09f's
  mini arms, and >=8h13m unfinished on the full train file. The seed-123 draw sits +3.4 %
  from the stored exact `w_H = 7.9005` -- inside the dbg noise band the spec predicts
  (~1-7 %); the s.e.'d verdict is [15d](15d-the-dbg-gate.md).
- Suites: package `--all` **9/9** (17:23:45 -> 17:26:33); openQHA `--all` **73/73**
  (17:30:47 -> 17:38:38); logs `%TEMP%\hl15b-pkgall.log` / `hl15b-oqsuite.log` (workstation).

**Review record (two-axis, per the `code-review` skill).** Ranges: package
`bf0abd6..05ac7b1`, openQHA working tree (`05_train.py`); two read-only sub-agents.
Fixed in-pass: the `t_train_run` count line now reports the executed checks (42; annotation
commit `547c394`). Judgement calls recorded, no change: the three-place default duplication
(flat-kwargs house style), the `(probe, k, seed)` clump (the schema is flat by convention),
the wording-substring test pins (the ticket asks for the amended descriptions), and the
report kv/note + the given-rule dry-run check as in-spirit provenance (the Record states its
estimator).

**State.** Package: `05ac7b1` + annotations `547c394` on `main` (local; the push stays the
user's). openQHA: the driver help/print change rides this ticket's commit. Next slices:
[15c](15c-the-amendments.md) (the amendments), [15d](15d-the-dbg-gate.md) (the dbg gate).
