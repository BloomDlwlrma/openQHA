# 15d: The dbg gate -- the closed-form-qualified numbers

Type: task
Status: resolved
Blocked by: [15b](15b-the-driver-and-the-record.md).
Serves: [15](../decisions/15-the-balance-on-the-probe-estimator.md) · spec: [spec-the-balance-on-the-probe-estimator.md](../spec-the-balance-on-the-probe-estimator.md).

**What to build:** the evidence that the amended balance is wired right. On
`draw300_r1dbg` (92-frame train file), compute the exact per-frame matrices once (sanity:
re-derive the stored reference `w_H = 7.9005`; produce the closed-form s.e.), then run
the amended balance at >=3 seeds; each seed's deviation from the exact value is within
4x the closed-form standard error, and the seed spread is consistent with it (nonzero,
within a loose factor). All numbers go into ticket 15's Answer -- the script itself is a
one-off in the OS temp dir (house rule); the numbers are the durable part.

- [x] >=3 seeds, each with its `w`, the closed-form s.e. (relative), and `z`; the spread report; the 7.9005 re-derivation matches (sanity).
- [x] All `|z| <= 4`; spread nonzero and consistent; a systematic deviation beyond the bound stops the change and comes back for review.
- [x] The numbers are written into [15](../decisions/15-the-balance-on-the-probe-estimator.md)'s Answer.

## Notes

- The s.e. is a property of the file (from the exact matrices), the same for every
  seed; `w` relative error equals `L_H` relative error to first order (seed 123:
  +3.4148 % vs 3.3020 %; the z above uses the w side).
- Yardstick rationale (spec Further Notes): per-frame k=4 noise is ~9-71%; the dbg
  mean's is ~1-7%; the full train file's is ~0.1-0.5% -- hence s.e.-relative, not a
  fixed +-1%.

## Answer (2026-10-01)

**Ran.** One-off harness `oqt15d-gate.py` in the OS temp dir (house rule), log
`oqt15d-gate.log`, evidence `oqt15d-gate.json`. The 92-frame train file
(`train/dry15b/train.wb97m-d3bj_def2-tzvppd.extxyz`, sha256 `773296dc...`; 44 x 16-atom
+ 48 x 14-atom frames) on base `MACE-OFF23_medium`; package `0.1.0` @ `547c394`
(annotations); fork `1110ffb` clean.

**The exact reference (sanity).** One full-matrix pass over the 92 frames (3N HVPs per
frame, 412.8 s): L_E 5.190180543955543e-07, L_F 1.7335776671230068e-04, L_H
2.194250882215115e-03 -> w_H = 7.900544469067027, **bit-identical** to the stored 09f
reference (rel. diff +0.000e+00). The same pass gives the closed-form Gaussian s.e.:
sd(mean L_H) = 1.0445657443767679e-04 -> rel s.e. = 0.047604663297310376 (4.7605 %).

**The amended balance at 5 seeds** (the wired `hessian_weight_balance`, gaussian k=4;
~52 s each): w = 7.719748782410633 / 7.847536027788529 / 7.853701698100101 /
8.170329761632424 / 7.880595272458871 at seeds 0 / 1 / 2 / 123 / 2026; rel -2.2884 /
-0.6709 / -0.5929 / +3.4148 / -0.2525 %; z = -0.481 / -0.141 / -0.125 / +0.717 /
-0.053. **Verdict: pass** -- max |z| = 0.717 <= 4, spread nonzero and consistent
(sample sd 0.16634559940563676 vs predicted 0.3761027593153636, ratio 0.44; chi^2 =
0.78 at dof 5), no systematic deviation. Seed 123 reproduces 15b's dry-run value to
~2 ulp (8.170329761632424 vs 8.170329761632427 recorded there; torch-CPU last-digit
process variance -- the same value at 15 significant digits).

**Cost.** The exact pass 412.8 s vs ~52 s per amended pass (~8x end to end on this
file; the Hessian term alone scales 3N : k -- 10.5-12x on its 14-16-atom frames); the
whole gate 674.5 s.

**Determinism (extra).** A fresh-process rerun of seed 123 (two calls in one process):
both exactly `8.170329761632427` (L_H 2.1217964484907632e-03) -- bit-identical within
the process, and equal to 15b's dry-run value to 0 ulp (the gate's own seed-123 value,
the 4th call in its process, sat ~2 ulp lower: +3.55e-15; ~4e-16 relative torch-CPU
last-ulp jitter). Probe log `oqt15d-repro.log`.

**Suites (2026-10-01).** Package `--all` 9/9; openQHA `--all` 73/73 -- both rc 0; log
`%TEMP%\oqt15d-suites.log`.

**Review (two-axis, per the `code-review` skill).** Working-tree review of the three
files by two read-only sub-agents: no hard failures; the acceptance boxes were
discharged and every recorded number reproduced from the evidence files. Fixed in-pass:
the "3N : k = 57 : 4 ~ 14x" cost line re-scoped to this file (10.5-12x at 14-16
atoms), the spread sentence re-worded ("sits on the low side of the prediction, well
inside the loose factor"), and the Notes' `w`/`L_H` relation qualified "to first
order". Recorded, not changed: the numbers appear in both the ticket's and the slice's
Answers (the house pattern of 15a-c), and `decisions/15` keeps `Status: open` until
[15e](15e-the-close-out.md)'s close-out.

**Not verified.** Nothing of this slice's scope is left; the deployment list, the
suites record and the timing resumption are [15e](15e-the-close-out.md)'s.
