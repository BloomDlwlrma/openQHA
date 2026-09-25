# 39: The Reference level retries a soft saddle before any verdict

**What to build:** at the reference level, a basin whose relaxation ends with its lowest mode inside the inversion window is re-optimised once from the ORCA-relaxed geometry — the same-method rule: interpret frequencies only at a stationary point of that method. If the retry reaches a minimum the basin enters normally; if it remains a saddle it is excluded, listed, and the merge map records `soft_saddle = true` together with the lowest frequency. A saddle below the floor is excluded as today. `ITHR_POLICY` in the reference record reads `invert_below`. The classification uses the shared floor classifier; the retry reuses the existing per-basin ORCA machinery (no new engine-file convention). The unit tests cover the branches; the ORCA path itself is exercised by the reference-level integration fixture.

**Blocked by:** 35 (One production policy on the CREST floor; the three-policy spread removed) — the floor classifier and the record changes live there

**Status:** done 2026-09-25 (full suite `python tests/run_tests.py --all`: all 77 tests pass — 63 unit + 12 integration + 2 regression — including `tests/integration/t_reference_level_retry.py`; the fixture run is recorded below)

- [x] Classification branches unit-tested: a minimum; a below-floor saddle; a window saddle that retries into a minimum; a window saddle that stays a saddle (`soft_saddle = true` + lowest frequency in the merge map and the record). `t_reference_level.py`: `relaxation_verdict` is checked on the shared floor classifier (minimum, below-floor saddle, window soft saddle — the floor line itself in the window; a preset with no floor has no window); `relax_with_retry` is checked with a fake `run` on all branches, including a retry that ends below the floor and a job already on disk (`seconds is None`) that is not retried again.
- [x] `ITHR_POLICY = invert_below` in the reference record (read from `thermo.PRODUCTION_POLICY`); the record-key test is updated. The `[[Basin]]` schema gains `SOFT_SADDLE` (written false on the clean fixture, true on a stay-saddle exclusion) and the merge map gains `lowest_frequency_cm` + `soft_saddle` on every row.
- [x] The retry is wired through the existing per-basin ORCA machinery; one integration fixture run is recorded in the ticket. `relax_with_retry` calls `orca.optimise_and_hessian` with the basin's own stem (`layout.orca_level_stem`) and the new `rerun=True`, which runs the job again and replaces the file group; `tests/integration/t_reference_level_retry.py` (fake ORCA binary) runs the real `_run_job` → publish → `parse_hess` → `verify_hess_frequencies` → retry → Records path on the propanal fixture.

## Notes (implementer, 2026-09-25)

**The records.** The merge map row of a basin that stays a saddle gains `soft_saddle = true`
and `lowest_frequency_cm` (ORCA's own lowest vibrational mode — the number the classification
used, tied to our projection by the existing `roundtrip_cm` column); every row carries both
columns, so a reader sees the lowest mode of kept and merged basins too. `status` stays
`saddle` (the flag is the extra bit, as in the prototype). The merge map's columns now carry
their comment lines (`MERGE_MAP_SCHEMA`, ADR 0003's form — the new two included; the file
was the one `.dat` written without them). The `thermo_msrrho` record's `[[Basin]]` row
gains `SOFT_SADDLE` (false everywhere a saddle is not soft) and the excluded saddle now
carries its `N_IMAGINARY` and its lowest frequency; the Report's exclusion line names the
soft saddle, the window and, when one ran, the retry.

**Decisions not spelled out in the ticket, and why.**

1. *The retry replaces the basin's file group.* `rerun=True` on the same stem — no new
   engine-file convention, and the file group keeps its one meaning: the basin's final job
   at this level. Downstream readers (`--start-from`, `hessian_compare`, `mode_curvature`)
   therefore always read the geometry the verdict was taken on, never the discarded saddle.
2. *`soft_saddle` describes the final relaxed spectrum.* The flag is true when the lowest
   mode lies in the inversion window [ithr, 0) — a stay-saddle retry, or a reused job found
   that way — and false when the mode is below the floor. A retry that settles below the
   floor is therefore excluded as an ordinary saddle, its reason saying it was retried
   ("... below the floor (ithr = -50); retried once from the relaxed geometry"); the
   ticket's "if it remains a saddle ... records soft_saddle = true" holds for the case it
   describes (it remains a *soft* saddle) and "a saddle below the floor is excluded as
   today" holds for the other. The flag never asserts a retry: the reason line says whether
   one ran ("retried once from the relaxed geometry" vs "the finished job on disk was not
   re-run"), so a Batch resume cannot claim work it did not do.
3. *The retry runs only when the job was produced in this call* (`record["seconds"] is not
   None`): a Batch resume reads the finished file group and does not re-run ORCA, as the CLI
   promises ("a finished basin is never recomputed"). A job already on disk that is still a
   window saddle is written `soft_saddle = true` with the "not re-run" reason; a first-pass
   job not produced by this code (pre-ticket-39 files) is the same case.
4. *Classification uses ORCA's printed lowest vibrational mode* (`verify_hess_frequencies`
   `["lowest_cm_inv"]`), the same estimator as the existing `n_imaginary` column, rather than
   a second projection of our own; the round trip to our projection stays recorded per basin.
5. *A lowest mode in [-1, 0) is a window saddle, not a sub-1 drop.* The classification is the
   shared floor classifier (the ticket's words), so the reference level retries it rather
   than dropping it the way the per-mode thermochemistry's ORCA-style sub-1 rule drops a
   mode from the sums; only if it stays in the window is the basin excluded. The prototype's
   `referenceVerdict` reads any negative mode the same way. The user may want this band
   exempted from the retry (the thermochemistry could keep the basin by dropping the mode);
   it is written down here so the ruling can be taken on numbers, not silently.

**The fixture run** (`wsl` in the `openqha` env, repo root; the fake ORCA binary writes
attempt 1 = the fixture basin-00 Hessian with its lowest eigenvalue shifted to −6.84 cm⁻¹,
attempt 2 = the fixture's real minimum job or the saddle again, in the window or below the
floor):

```
$ python tests/integration/t_reference_level_retry.py
  the generated window saddle round-trips and its lowest mode is -6.84 cm^-1 ok
  A: the window saddle was retried once (two ORCA calls; basins 1-2 reused)  ok
  A: the retry's input is the first job's ORCA-relaxed geometry (max dev 5.0e-11 A) ok
  A: the basin is kept/merged with soft_saddle false, lowest_frequency_cm = 131.14 ok
  A: the retry REPLACED the file group (the .hess is the fixture's, byte for byte) ok
  A: no basin is excluded and every [[Basin]] row says SOFT_SADDLE = false   ok
  A: the merge map lists every MACE basin once with the new columns          ok
  B: two ORCA calls; the merge map is status saddle, soft_saddle true, lowest -6.84 ok
  B: the record's excluded row carries SOFT_SADDLE = true and the lowest frequency ok
  B: the ensemble counts it excluded (N_BASINS 3, N_INCLUDED 2, N_EXCLUDED 1) ok
  B: the Report says the soft saddle was retried and names the window        ok
  B: the other merge-map rows are not soft saddles and carry their lowest mode ok
  C: below the floor after the retry: status saddle, soft_saddle false, lowest -61.68 ok
  C: the record says the saddle below the floor was retried, not that it is soft ok
PASS
```

`t_reference_level.py` additionally runs the propanal fixture unchanged (three clean
minima): merge map rows carry `soft_saddle = false` with their lowest modes (131.14, …),
every `[[Basin]]` row carries `SOFT_SADDLE = false`, and the tiers are untouched. The
integration file also checks that the generated saddle round-trips through
`verify_hess_frequencies` and that the first attempt's geometry is the second's input.

