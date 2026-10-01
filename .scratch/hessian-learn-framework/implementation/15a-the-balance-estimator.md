# 15a: The balance's estimator path -- the function gains the probe branch

Type: task
Status: resolved
Blocked by: None.
Serves: [15](../decisions/15-the-balance-on-the-probe-estimator.md) · spec: [spec-the-balance-on-the-probe-estimator.md](../spec-the-balance-on-the-probe-estimator.md).

**What to build:** the balance function can produce its `L_H` with the run's probe
estimator beside the existing exact path: given the probe kind (gaussian), `k`, and a
seed, it returns a deterministic, unbiased estimate -- the estimator family the trained
loss itself uses -- while the Cartesian call stays byte-for-byte what it is today (the
anchors and the judge keep their exact readings). The wiring arithmetic is checked
directly, not only statistically.

- [x] Calling the estimator path returns the same value on repeat for a given seed, and the value is the mean over the train file's labelled frames of `sum_j ||H_theta v_j - H_r v_j||^2 / (9 N^2 k)`, `v_j` standard-normal, drawn per frame from a dedicated generator seeded by the run's seed.
- [x] On the fixture, the estimate sits within 4x the closed-form standard error (`estimator_variance`, gaussian) of the Cartesian value -- the `t_train_engine` precedent's construction, applied to the mean over frames at k=4.
- [x] The wiring check: for sample frames, the estimator's HVP path (`H_theta v` through the fork's machinery) equals the exact matrix applied to `v` to <=1e-8 relative.
- [x] The Cartesian path and its pins are unchanged (existing tests green).

## Notes

- Reuse the same HVP machinery the loss uses (`hvp_from_forces` pattern, no third-order
  graph); no new dependencies; **no fork changes**. Batching frames is permitted as an
  implementation freedom provided the tests hold.
- TDD seam: the package's existing smoke-fit unit surface (prior art: `t_smoke_fit`'s
  balance arithmetic; `t_train_engine`'s 4 s.e. comparison).
- Spec authority for every number and definition; the caller surface (who passes
  probe/k/seed) is 15b's scope.

## Answer (2026-10-01, `openQHA-Hessian` @ `bf0abd6`)

**What landed.** `epoch_zero_balance` reads `L_H` with the run's probe setting beside
the exact path: `cartesian` is the full-matrix reading, the same path as before;
`gaussian` / `rademacher` draw `n_probes` (k=4) probes per frame from ONE dedicated
generator (`seed`, file order) and average `sum_j ||H_theta v_j - H_r v_j||^2 / (9 N^2 k)`
over the labelled frames, computed `chunk` frames at a time through the new
`hvp.hvp_from_atoms_batch` -- the same probe packing and force-graph HVP the training
loss runs every step (inference: no third-order graph). The frames are collated through
the calculator's own `_atoms_to_batch`, their fields re-wrapped as base `Data` (mace's
`AtomicData` cannot be empty-constructed, so its own `to_data_list` cannot split a frame
back out; the fork does not override the merge's offset semantics). The return dict
gains `N_PROBES` (0 for the exact path); unknown probe names and `chunk < 1` are
refused. Chunk batching: the spec's permitted implementation freedom, `chunk=8` default
(approved with the plan). No fork changes; no new dependencies.

**Evidence.** `t_smoke_fit` **23/23**: the estimator deterministic (bit-identical);
one frame = the manual chain to 1e-12; within 4 s.e. of the Cartesian value (z = 1.22);
wiring per frame <= 1e-8 (measured 3.0e-16) and a two-frame chunk = the single-frame
calls per frame (1.5e-16); chunk=1 vs 8 the same value (5.9e-16); a bad probe name
refused; rademacher runs; the old cartesian pins untouched (`N_PROBES = 0` recorded).
Package suite `--all` **9/9** (fresh run 17:02:52). The post-review integration check in
`t_hvp_engine` -- two separate frames collated like the training batch = per-frame HVPs
on the REAL engine (5.7e-14) -- first exposed a real defect (`to_data_list` cannot
reconstruct `AtomicData`); the re-wrap fix landed before the green run.

**Review record (two-axis, per the `code-review` skill).** Range `411abe1`..working
tree, two read-only sub-agents; findings fixed in-pass:
- Spec: the real-collate path had no test (fixed: the `t_hvp_engine` check); the wiring
  check used one frame duplicated (fixed: different probes per frame, both slices
  checked); 15d's dbg gate stays scheduled (not this slice).
- Standards: `zip` truncated unpaired lists silently (fixed: `ValueError`); the "same
  batched HVP the loss runs" wording overclaimed (fixed: "same probe packing and
  force-graph HVP ... inference here: no third-order graph"); the duplicated-ladder
  note kept as a judgement call (the single-frame helper stays the production-proven
  primitive; both share `hessian_vector_products`).

**State.** `bf0abd6` on the package repo's `main` (local; the push stays the user's).
`run.py`'s balance helper still reads `probe="cartesian"`; the driver wiring, the
`BALANCE_PROBE` / `BALANCE_N_PROBES` Record fields and the help strings are
[15b](15b-the-driver-and-the-record.md)'s scope.
