# Spec: The balance on the probe estimator (w_H without the full-matrix pass)

Label: `ready-for-agent`. Tracker: `.scratch/hessian-learn-framework/`. Spec for
[15-the-balance-on-the-probe-estimator](decisions/15-the-balance-on-the-probe-estimator.md);
ruling of 2026-10-01 (user). Amends the balance reading of `hessian-learning-set`'s
`spec-phl-verbatim.md` (story 5, Derivation 3.6) for the balance only; S0-C-64's
loss-side reading (PHL verbatim; "projected" means the random-vector projection and
nothing else) stands.

## Problem Statement

The balance rule — the driver's default `--hessian-weight balance` — measures `w_H` from
an exact full-matrix `L_H` over the run's train file. That pass, not the training loop,
is the run's dominant fixed head: ~3N Hessian-vector products per labelled frame over
27,740 frames (≈10 epoch-equivalents; ≥8h13m observed unfinished on v100x), paid by the
timing job and by each of the two production arms, and to be paid again by every future
round that re-measures the balance. It is also semantically odd: the trained term is
PHL's random-probe estimator, so the weight that calibrates it is defined from a
quantity the loss never computes — while PHL itself sets weights under the projected
scheme. The result is a wall-time risk that has already cost two nights and a timing
submission that had to be held.

## Solution

Measure `L_H` with the same estimator family the loss trains with: per frame, k
standard-normal probes (k = the run's probe count, 4), `L_H` = mean over the train
file's labelled frames of `sum_j ||H_theta v_j - H_r v_j||^2 / (9 N^2 k)`, drawn from a
dedicated generator seeded by the run's `SEED`. The rule name and formula do not
change (`HESSIAN_WEIGHT_RULE = balance`, `w_H = w_F L_F / L_H`); the estimator is
unbiased for the exact value and ≈10×+ cheaper, turning a multi-hour head into a
minutes-scale one. The Record gains `BALANCE_PROBE` and `BALANCE_N_PROBES` so a
balance value is never read without its estimator. The exact anchors and the judge
keep their full-matrix readings, and the local gate re-validates the new value on the
`draw300_r1dbg` subset against the stored exact reference before anything is submitted.

## User Stories

1. As the operator, I want the balance to finish in minutes rather than hours, so that
   the timing job and both arms are not gated by a multi-hour fixed head.
2. As the operator, I want the timing job and the arms to run the same pipeline, so
   that the cap's wall check describes the runs that actually train.
3. As the operator, I want the balance to stay affordable when it is re-measured every
   round, so that the campaign is not taxed ≈10 epoch-equivalents per arm per round.
4. As the operator, I want both arms to share one `w_H` by construction (same train
   file, same seed), so that the cross-arm reading stays clean.
5. As the operator, I want the held timing submission unblocked once this change is
   deployed, so that round 1 resumes on the amended pipeline.
6. As the reviewer, I want the Record to state which estimator produced `BALANCE_L_H`
   (`BALANCE_PROBE`, `BALANCE_N_PROBES`), so that values are never compared across
   estimators unknowingly.
7. As the reviewer, I want the exact anchors to remain exact, so that each run keeps a
   noise-free before/after pair and the probe offset stays measurable.
8. As the reviewer, I want a comparability note for pre-amendment Records (their
   `BALANCE_L_H` is an exact reading), so that the change's boundary is explicit.
9. As the reviewer, I want the balance value to be deterministic for a given (base
   model, train file, probe, k, seed), so that reruns reproduce the same `w_H`.
10. As the reviewer, I want the estimator's realized deviation measured on the dbg
    subset against the stored exact reference, so that adoption rests on evidence.
11. As the maintainer, I want `epoch_zero_balance`'s Cartesian path to stay
    byte-for-byte, so that the anchors, the judge and the fixture tests keep their
    exact readings.
12. As the maintainer, I want no new modules, no new dependencies and no fork changes,
    so that the fork freeze (`1110ffb`) is untouched and the change rides existing
    seams.
13. As the maintainer, I want dated amendments in `spec-phl-verbatim.md` (story 5,
    Derivation 3.6) and a re-baselined 09g, so that the documents stay single-sourced.
14. As the gate runner, I want the acceptance yardstick to be the closed-form standard
    error, not a guessed tolerance, so that the small dbg set's noise is not mistaken
    for a bias.
15. As the gate runner, I want ≥3 seeds reported, so that the seed spread is known,
    not assumed.
16. As the next session, I want the spec, the amendment requirements and the gate
    numbers recorded, so that deployment and the timing resumption are mechanical.

## Implementation Decisions

- **The estimator.** `L_H` = the mean over the train file's labelled frames of the
  run's probe estimator, `sum_j ||H_theta v_j - H_r v_j||^2 / (9 N^2 k)`; `v_j`
  standard normal (PHL Algorithm 1); `k` = the run's probe count (4); drawn per frame
  from a dedicated generator seeded by the run's `SEED`. Unbiased for the exact value;
  the realized deviation on the full train file is ≈0.1–0.5% (closed-form extrapolation;
  measured at the gate on the dbg subset, where it is ≈1–7% — hence the s.e. yardstick).
- **Unchanged rule surface.** `HESSIAN_WEIGHT_RULE` stays `balance`; `w_H = w_F L_F /
  L_H`; `L_E`/`L_F` computations unchanged (deterministic); a numeric
  `--hessian-weight` is still `given`.
- **The balance helper passes the run's probe setting** (probe kind, k, seed) into the
  smoke-fit balance function; its `probe` parameter becomes functional: `cartesian` is
  today's exact path (kept byte-for-byte for callers that need it), `gaussian` /
  `rademacher` select the estimator.
- **Record schema.** Add `BALANCE_PROBE` (String) and `BALANCE_N_PROBES` (Integer);
  update the `HESSIAN_WEIGHT_RULE` and `BALANCE_L_H` descriptions ("measured with the
  run's probe setting …; the exact full-matrix reading stays the job of the anchors and
  the judge"); pre-amendment Records' `BALANCE_L_H` are exact readings (comparability
  note in the descriptions' text).
- **Module surfaces (house references, no paths).** The package's smoke-fit module (the
  balance function gains the estimator path); the package's run module (the balance
  helper passes probe/k/seed; the Record schema and info gain the two fields; the
  module's "THE WEIGHT" paragraph and the helper's docstring drop "Cartesian target,
  the full matrix"); the openQHA driver's `--hessian-weight` help string (drop "with
  the Cartesian target").
- **Estimator mechanics.** Reuse the same HVP path the training loss uses (the fork's
  force-graph HVP), without the third-order graph; batching frames is permitted as an
  implementation freedom provided the tests hold; no new dependencies; no fork changes.
- **Spec amendments (drafted, to land with the code).** `spec-phl-verbatim.md` story 5:
  "measured by the balance rule on the full Cartesian matrix only" becomes "measured by
  the balance rule with the run's probe setting (gaussian k=4 by default; the exact
  full-matrix reading remains the anchors' and the judge's job)". Derivation 3.6's
  parenthetical "(the estimator's noise must not enter a weight)" becomes the amended
  rationale: a fixed-seed k=4 estimate carries a ≈0.1–0.5% offset on the full train
  file — far below the ≈5–35% per-step noise the loss itself trains through — and the
  weight is thereby measured under the trained term's own estimator; the Record keeps
  the evidence to re-identify it. The implementation section's "`epoch_zero_balance`
  computes the Cartesian value only" line gains the same qualifier.
- **09g re-baseline.** The balance no longer reads as "a full-matrix pass over the
  whole train file"; the wall check's fixed head becomes split + probe-balance +
  anchors; the timing submission stays held until the change is deployed (user ruling,
  2026-10-01).
- **Deliberately not changed.** The trained loss; the validation estimator (4 fixed
  stored probes); the exact anchors; the judge; the smoke-fit instrument's own balance
  (its Cartesian reading stands and its historical numbers stay valid).

## Testing Decisions

- **What a good test is here.** Assert external behavior: the estimator's agreement
  with the exact reading within the closed-form standard error; determinism for a fixed
  seed; the Record's provenance fields and rule; never the internals of the HVP calls.
- **Seams.** No new seams. Unit tests extend the package's existing smoke-fit test
  surface; the integration check extends the existing run-to-Record surface (the
  fixture fine-tune); the acceptance gate runs the existing driver dry-run path on the
  `draw300_r1dbg` subset (the 09f procedure).
- **The dbg gate.** Compute the amended balance on the dbg train file at ≥3 seeds;
  compare each against the stored exact reference (`w_H = 7.9005`); require each within
  a small multiple (≈4) of the closed-form standard error and the seed spread
  consistent with it; record all numbers in the ticket's Answer. A systematic deviation
  beyond the bound stops the change.
- **Prior art.** `t_smoke_fit`'s balance arithmetic checks; `t_train_engine`'s
  "within 4 s.e." comparison of the epoch −1 validation term against the Cartesian
  value; the 09f gate's dbg procedure.
- **The wiring check.** A probe-arithmetic identity joins the unit surface: for sample
  frames, the estimator's HVP path (`H_theta v` through the fork's machinery) agrees
  with the exact matrix applied to `v` to ≤1e-8 relative — wiring errors are caught
  below the statistical noise floor.
- **Suites before deploy.** Package `--all` and openQHA `--all` green; then deploy the
  checkouts and resume the round.

## Out of Scope

- The batched-exact (E2) evaluator and the per-run full exact evaluation job (Q3,
  2026-10-01) — recorded for a future ticket; the registered-model surface already
  allows loading a fine-tune by name.
- The smoke-fit instrument's own balance semantics (stays Cartesian; its historical
  numbers stand).
- k or probe-kind changes: the balance follows the run's probe setting; round 1 is
  gaussian k=4.
- The round's production knobs, the anchors, the judge, active-learning scheduling.
- The timing/arms submissions themselves (ops; they resume after this change deploys).

## Further Notes

- **Cost.** Full : k-probe = 3N : K = 57 : 4 ≈ 14× for the Hessian term (Derivation
  3.5); ≈10×+ end to end with E/F; the observed exact pass was ≥8h13m unfinished
  (v100x, 259832).
- **Noise.** Closed form (`estimator_variance`): per-frame relative sd at k=4 ∈
  [sqrt(2/(3N k)), sqrt(2/k)] ≈ 9.4%–70.7%; over 27,740 frames ≈ 0.1–0.5%; over the
  92-frame dbg train set ≈ 1–7% (the gate's yardstick is the s.e., not a fixed ±1%).
- **Scale.** Training's own per-step Hessian-term noise (4-frame batches, k=4) ≈
  5–35%; the weight scan's meaningful steps are 0.3×/1×/3× — the estimator's offset is
  invisible at both.
- **PHL context.** The paper trains with HVPs only (probes per minibatch; k=1 per
  molecule in its fixed-probe regime) and sets its weights by tuning under the
  projected scheme (λF 0.30 / λH 0.09); this change moves the balance onto that
  scheme's estimator — still measured, not quoted.
- **Sequencing.** Implement → suites → amend the specs/runbook → deploy the checkouts →
  resume: submit the held timing job → cap → the two arms.
