# The balance on the probe estimator: w_H without the full-matrix pass

Type: task
Status: open
Blocked by: None.
Part of: [hessian-learn-framework](../map.md)

## Question / work

The balance rule's `L_H` is the exact full-matrix value read over the run's train file
(`epoch_zero_balance` with the Cartesian target). That pass is the run's dominant fixed
head — ~3N Hessian-vector products per labelled frame over 27,740 frames, ≈10
epoch-equivalents (observed ≥8h13m unfinished on v100x, job 259832) — and it is paid per
arm and, in active learning, per round. The user's ruling (2026-10-01): `w_H` is the
trained loss term's own weight, and it should be measured under the same estimator
family as the trained term — PHL's random-probe projection — not by an exact-matrix
pass that no training step performs.

This ticket carries that change: the balance's `L_H` is measured with the run's probe
setting (gaussian k=4 by default), drawn per frame from a dedicated generator seeded by
the run's `SEED`; the rule name and formula are unchanged; the Record gains
`BALANCE_PROBE` / `BALANCE_N_PROBES`; the exact anchors and the judge keep their
full-matrix readings; `spec-phl-verbatim.md` (story 5, Derivation 3.6) and the round
materials (09g) get dated amendments; the local gate is re-run on the `draw300_r1dbg`
subset against the stored exact reference (w_H = 7.9005), judged against the
closed-form standard error.

Spec: [spec-the-balance-on-the-probe-estimator.md](../spec-the-balance-on-the-probe-estimator.md).

Also recorded (not this ticket's work): the per-run **full exact evaluation** job (the
user's Q3, 2026-10-01) becomes a future ticket; its affordable form is a batched-exact
evaluator (E2), and the registered-model surface already allows loading a fine-tune by
name.

## Answer

**The dbg gate (15d, 2026-10-01).** One-off harness in the OS temp dir (house rule; the
repo keeps none): `oqt15d-gate.py`, log `oqt15d-gate.log`, evidence `oqt15d-gate.json`.
Environment: `draw300_r1dbg`'s 92-frame train file (sha256 `773296dc...`; 44 x 16-atom +
48 x 14-atom frames), base model `MACE-OFF23_medium`, package `0.1.0` @ `547c394`
(annotations), fork `1110ffb` clean.

*The exact reference, re-derived in one full-matrix pass* (92 frames, 3N Hessians per
frame; 412.8 s): L_E 5.190180543955543e-07, L_F 1.7335776671230068e-04, L_H
2.194250882215115e-03 -> **w_H = 7.900544469067027, bit-identical to 09f's stored
reference** (rel. diff +0.000e+00). The same pass yields the file's closed-form
Gaussian s.e.: sd(mean L_H) = 1.0445657443767679e-04 -> **rel s.e. = 0.047604663297310376
(4.7605 %)** -- the gate's yardstick.

*The amended balance (the wired `hessian_weight_balance`; gaussian k=4) at 5 seeds* --
about 52 s each:

| seed | w_H | L_H | rel. vs exact | z |
|---|---|---|---|---|
| 0 | 7.719748782410633 | 2.2456e-03 | -2.2884 % | -0.481 |
| 1 | 7.847536027788529 | 2.2091e-03 | -0.6709 % | -0.141 |
| 2 | 7.853701698100101 | 2.2073e-03 | -0.5929 % | -0.125 |
| 123 | 8.170329761632424 | 2.1218e-03 | +3.4148 % | +0.717 |
| 2026 | 7.880595272458871 | 2.1998e-03 | -0.2525 % | -0.053 |

*Verdict: pass.* max |z| = 0.717 <= 4; the spread is nonzero and consistent with the
closed form (sample sd 0.16634559940563676 vs predicted 0.3761027593153636, ratio
0.44; chi^2 = 0.78 at dof 5 -- the scatter sits on the low side of the prediction,
well inside the loose factor, n=5 making the sd itself noisy); no systematic deviation
-- the change stands. Seed 123 reproduces 15b's dry-run value to ~2 ulp
(`8.170329761632424` vs `8.170329761632427` recorded there, torch-CPU last-digit
process variance; the same value at 15 significant digits).

*Cost on the file:* exact 412.8 s vs about 52 s per amended pass (about 8x end to end
here; the Hessian term alone scales 3N : k -- 10.5-12x on this file's 14-16-atom
frames). The whole gate ran 674.5 s.

*Determinism (extra; the follow-up probe of 15b's 2-ulp question).* Seed 123 rerun in a
fresh process, two calls in one process: both exactly `8.170329761632427` (L_H
2.1217964484907632e-03) -- bit-identical within the process, and equal to 15b's dry-run
value to 0 ulp; the gate's seed-123 value (the 4th balance call in that process) sat
~2 ulp lower (`8.170329761632424`, +3.55e-15). Torch-CPU last-ulp process jitter
(~4e-16 relative, ~14 orders below the gate's noise scale); no effect on any verdict.

*Suites (2026-10-01, this tree).* Package `--all` **9/9**; openQHA `--all` **73/73** --
both rc 0 (log `%TEMP%\oqt15d-suites.log`: `OQT15D-SUITES-RC pkge=0 oq=0`; the header
names the two checkouts and `mace-torch 0.3.16+openqha`).

<!-- 15e (the close-out) carries: the deployment list, the suites record, Status: resolved, the map lines -->
