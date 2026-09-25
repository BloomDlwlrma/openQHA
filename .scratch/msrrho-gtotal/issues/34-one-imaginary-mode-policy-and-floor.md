# 34: The imaginary-mode regime — one production policy on a CREST floor; census floor; ORCA-style sub-1 cm⁻¹; reference-level retry

**Triage:** ready-for-agent
**Blocked by:** none
**Supersedes:** the open item of 28 (the three-policy spread is decided: removed; `refuse` is removed)

**Status:** done 2026-09-25 — implemented by tickets 35-41: 35 (one production policy `invert_below` on the floor, plus the sub-1 cm⁻¹ drop with its tripwire), 37 (the census floor and the inversion window), 38 (the certified tighten with the one bounded second pass), 39 (the reference-level soft-saddle retry), 40 (ADR 0007, the campaign guard, the verified suite) and 41 (the batch screen wired to the floor, with the close-out bundle). Evidence in each ticket's Status; the suite and integration transcripts are in tickets 40 and 41.

## Problem Statement

Branch A loses whole molecules to sign flips at the bottom of the spectrum. On Tianhe,
five molecules died with "no basin survives the tightening and the imaginary-frequency
filter" while the candidate's lowest mode sat at −6.84 cm⁻¹ — indistinguishable from
numerical noise at the decision resolution. Meanwhile the thermochemistry layer carried
three imaginary-mode policies with no production decision (`refuse` was the default "until
the ruling"), every record carried a three-policy spread nobody acted on, the census had
no frequency floor at all, sub-1 cm⁻¹ anomalies had no defined semantics, a reference-level
relaxation that landed on a soft saddle was thrown out without a second look, and a frame
whose tighten stopped short of `fmax` was never judged against any stated certification
line. The regime as a whole was documented only as a spread of options.

## Solution

One production policy, everywhere: `invert_below` with the frequency floor
`ithr = −50 cm⁻¹` (CREST's own default). The census admits candidates whose lowest mode
lies in (ithr, 0) — recorded as an **inversion window**, inverted by the thermochemistry —
and ejects only what lies below the floor; a molecule all of whose candidates lie below
the floor is still refused, with every condemned candidate's lowest frequency listed.
Sub-1 cm⁻¹ modes in a projected spectrum are dropped from the thermochemistry sums the
way ORCA's `CutOffFreq` and CREST's `vibthr` drop them, with the dropped values recorded;
three or more in one spectrum raise (the signature of an unprojected spectrum). A
reference-level relaxation that lands on a soft saddle is re-optimised once before any
verdict; if it stays a saddle it is excluded, listed, and annotated `soft_saddle` with its
lowest frequency. Non-converged frames are judged against ORCA's default convergence
(`TolMaxG = 3e-4 Eh/bohr = 1.543e-2 eV/Å`): inside the line they are certified; outside,
one second optimisation pass runs; still outside, they enter with `tighten_converged=false`
and the residual recorded. `refuse` and `[Imaginary_Spread]` are removed. Every number
keeps its provenance in the records; CONTEXT.md and ADR 0006 record the vocabulary and
the ruling.

## User Stories

1. As the production operator, I want molecules whose only imaginary mode sits in (−50, 0) cm⁻¹ to finish branch A instead of being refused, so that the flexible tail of a draw stops losing whole molecules.
2. As the production operator, I want the −6.84 cm⁻¹ molecule to come out as a basin list with the mode recorded in an inversion window, so that a night of Tianhe time is not spent to learn a number we can now act on.
3. As the production operator, I want a molecule whose candidates all lie below the floor to be refused with every candidate's lowest frequency and residual listed, so that one read tells me whether it is a real finding or a tightening problem.
4. As a records reader, I want every msRRHO record to name the policy and the floor it used, so that no free-energy number's provenance is a guess.
5. As a records reader, I want per-basin `N_INVERTED` with the inverted frequencies and `N_BELOW_FLOOR` with the dropped frequencies, so that I can recompute what was done to the spectrum.
6. As a records reader, I want an excluded basin to carry its reason string (below floor, tripwire), so that "excluded" is never confused with "missing".
7. As the safety reviewer, I want a projected spectrum with three or more modes below 1 cm⁻¹ to raise, so that a broken projection can never be absorbed by silent dropping.
8. As the safety reviewer, I want one or two sub-1 cm⁻¹ modes to be dropped ORCA-style and counted, so that a genuinely ultra-soft mode does not kill a molecule while still leaving a trace.
9. As the census author, I want one pure verdict function (spectrum + floor in; verdict + counters out), so that the floor rule is testable without an engine.
10. As the census author, I want the census floor number to come from the same value the thermochemistry uses, with a test asserting equality, so that the two can never drift apart.
11. As the census author, I want `tighten_converged = false` recorded (with the residual) on basins that exceeded ORCA's default line even after a second pass, so that non-stationarity is recorded rather than hidden.
12. As the census author, I want a frame whose residual is inside ORCA's default max-gradient certified as `converged_orca_default` without extra work, so that slow-but-fine frames do not pay for a second pass.
13. As the reference-level user, I want a wB97M relaxation that ends with a soft saddle (lowest mode in (−50,0)) to be re-optimised once before any verdict, so that "the optimisation had not arrived" is handled as a geometry problem, not a frequency policy.
14. As the reference-level user, I want the merge map to distinguish `saddle` from `soft_saddle` and to record the lowest frequency, so that a human can see which exclusions deserve a look.
15. As the maintainer, I want `refuse` gone from the code, the tests and the docs, so that there is exactly one production policy and no dead branch to mislead.
16. As the maintainer, I want `[Imaginary_Spread]` gone from every record and schema, so that no record promises three policies that production no longer has.
17. As the GFN2-seam maintainer, I want `crest_native` untouched (the seam keeps its line-for-line CREST claim) with only the removed spread block absent, so that the seam's validation stays true.
18. As an archivist, I want records written before 2026-09-25 that carry `ITHR_POLICY = refuse` to stay readable as data, so that nothing historical is rewritten; their recomputation is out of scope by ruling.
19. As the CLI user, I want no `--policy` flag on the msRRHO production step (an impossible choice cannot be wrong), and the per-basin drop counts printed instead of the spread.
20. As the docs keeper, I want CONTEXT.md to define "frequency floor (ithr)", "inversion window" and the two-policy vocabulary, and ADR 0006 to record the ruling with its provenance, so the next reader does not re-litigate it.
21. As the Hessian-learning campaign owner, I want a guard sentence in the workflow notes that labels are computed at their own geometry and never optimised, so no future maintainer "fixes" the displaced frames (D1 scope).
22. As the future implementer, I want the validated state machine quoted in this spec, so that the code lands with the exact shape that was clicked through in the prototype.
23. As the operator, I want the reference-level ORCA inputs to pin `ProjectTR` / `TransInvar` / `CutOffFreq` explicitly, so that the projection state of every Hessian is declared rather than inherited.
24. As a test author, I want the config's dead `reject_imaginary*` keys removed and one live floor key added, so the config stops documenting a flag no code reads.

## Implementation Decisions

- **One policy, two implementations.** The imaginary-mode policy set becomes
  `{invert_below, crest_native}`. `refuse` is deleted from the thermo module, its tests are
  deleted or rewritten, and production callers stop passing a policy at all (the default
  *is* production: `invert_below`, floor from the `crest` preset). `crest_native` remains,
  reachable only by the GFN2 seam.
- **Sub-1 cm⁻¹ semantics (ORCA-style drop with recording, ≥3 = tripwire).** In the policy
  layer: a mode with `|nu| < 1 cm⁻¹` is removed from every thermochemistry sum (S, Cp,
  H, ZPE) and counted (`n_below_floor`, with the values listed); if three or more such
  modes appear in one spectrum the calculation raises — that count has no plausible
  physical explanation at a converged, projected minimum and is the signature of an
  unprojected spectrum. `VIBTHR_CM = 1.0` stays the constant; its role changes from
  "assertion" to "drop floor + tripwire".
- **The policy layer returns counters.** `msrrho` records `n_inverted`, `n_below_floor`
  and the dropped frequencies; `basin_thermochemistry_from_frequencies` carries them into
  the basin record; an excluded basin carries `excluded_reason`. The strict
  `vibrational()` helper stays strict (harmonic over all modes, rejects non-positive);
  the policy is applied by the assembly entry (`g_minus_eel`, which gains the policy
  parameter) *before* the strict sums, so branch A's RRHO labels and the census's
  optional free energy both follow the same rule as the ensemble.
- **Ensemble.** `assemble` still raises when no basin survived; `basin_thermochemistry*`
  and `ensemble`'s defaults become `invert_below`; the three-policy spread function and
  the `[Imaginary_Spread]` record block are deleted (their purpose was the ruling;
  ticket 28's remaining item is closed as superseded). Record schema and the TOML
  round-trip test lose the block; basin rows gain the drop fields.
- **Census floor.** `census_from_frames` gains `ithr_cm` (read from the `package2` config
  section, which mirrors `MSRRHO_PRESETS["crest"].ithr_cm`; a unit test asserts the
  equality). Ejection condition becomes `lowest < ithr`; candidates in (ithr, 0) are
  admitted with `n_inversion_window` and their lowest frequency recorded; the record also
  gains each basin's `lowest_frequency_cm_inv`, `n_below_ithr = 0`, `convergence_class`
  and, when needed, `tighten_converged = false`. The refusal (all candidates below the
  floor) keeps raising, with the message listing every condemned candidate and the
  tighten residual line, reworded to the floor vocabulary. Acceptance criterion 4 is
  reworded accordingly ("no mode below ithr; window counted; 6 rigid modes; separation
  unchanged"). The verdict itself is extracted as a pure helper (below) so it is testable
  without an engine.
- **Convergence certification.** Classes: residual ≤ repo target (1e-4 eV/Å) →
  `converged`; ≤ ORCA's default line (3e-4 Eh/bohr = 1.543e-2 eV/Å) →
  `converged_orca_default`; above → a second, bounded optimisation pass is offered by the
  pipeline; if it still exceeds the line the basin enters with `tighten_converged = false`
  and the residual recorded. Nothing is auto-ejected for non-convergence; the counts are
  reported in the census summary.
- **Reference level.** A relaxed basin with imaginary modes is classified by the floor:
  below the floor → `saddle` (excluded, listed, as now). Inside the inversion window →
  one retry: re-run one ORCA relaxation from the ORCA-relaxed geometry (reusing the
  existing engine files machinery); if it reaches a minimum it enters normally; if it
  remains a saddle it is excluded, listed, and the merge map row gains `soft_saddle =
  true` plus the lowest frequency. `ITHR_POLICY` in the reference record becomes
  `invert_below`. ADR 0004's level-folder conventions are unchanged.
- **ORCA interface.** The reference-level input block pins `%freq ProjectTR true`,
  `TransInvar true`, `CutOffFreq 1.0` explicitly (defaults today; declared from now on),
  and the module documents that its imaginary count uses the same 1 cm⁻¹ floor. Whether
  the `.hess` matrix we read has the acoustic-sum-rule correction applied is verified and
  stated in the module's notes rather than assumed.
- **Verdict logic (validated in the prototype; lift as-is).**

  ```js
  // prototype_imaginary_mode_state_machine.html — decision-rich excerpt
  censusVerdict(nu, ithr)  -> min(nu) < ithr ? "saddle" : "basin"
                              counters: {lowest, n_below_ithr, n_in_window}
  convergenceClass(residual, secondPass) ->
      residual <= 1e-4            -> "converged"
      residual <= 1.543e-2        -> "converged_orca_default"
      secondPass === "converged"  -> "converged (second pass)"
      secondPass === "still"      -> "admitted_flagged"   // tighten_converged = false
      else                        -> "needs_second_pass"
  thermoOutcome(nu, {ithr, vibthr, mode}) ->
      min(nu) < ithr                       -> excluded ("below floor, not invertible")
      count(|nu| < vibthr) >= 3            -> raised  ("unprojected signature")
      |nu| < vibthr && mode === "assert"   -> excluded (kept only for comparison runs)
      otherwise -> included: drop the |nu| < vibthr modes (counted),
                   invert the (ithr, 0) modes (counted), sums harmonic above
  ```
- **CLI and scripts.** The msRRHO production step loses `--policy`; its printout reports
  per-basin inverted/dropped counts instead of the spread. The branch-A script passes the
  floor through and stops hard-coding the old flag. Its acceptance-criteria text and the
  branch-A report wording follow the floor vocabulary.
- **Config.** The dead keys `reject_imaginary` (conformers section) and
  `reject_imaginary_frequencies` (hessian section) are removed; `package2` gains
  `ithr_cm = -50.0` with a comment naming CREST's `-ithr` default as its origin and the
  preset it mirrors.
- **Documentation.** CONTEXT.md: "Imaginary-mode policy" rewritten to the two policies
  with the removal dated; "Basin" redefined via the frequency floor; new term "Inversion
  window"; the "Every record carries `[Imaginary_Spread]`" sentence removed. ADR 0006
  written: one policy, the floor, the removal of `refuse`, the sub-1 semantics, the
  reference retry — with the provenance table (−50 = CREST `-ithr`; 1 = CREST `vibthr` /
  ORCA `CutOffFreq`; 25 = `-sthr` / `QRRHORefFreq`). The Hessian-learning workflow notes
  gain the D1 guard sentence. README's policy paragraph follows.
- **Bookkeeping.** Ticket 28's remaining open item and spec.md's "until the ruling"
  sentence are updated to point at this ticket; the research note
  (`research-soft-modes-primary-sources.md`) and the prototype HTML stay in the feature
  directory as primary sources.

## Testing Decisions

- **What a good test is here:** only external behaviour — a spectrum, a residual, a level
  in; a verdict record, counters, a reason string, a message out. No assertions on private
  helpers, no engine runs in unit tests, no record-shape drift beyond the deliberate
  schema changes. Every new field gets exactly one test that would fail if the field
  disappeared.
- **Seams (please confirm; amend if they do not match expectations):** the existing pure
  seams are reused — `msrrho` (policies, drop/tripwire, counters), the per-basin
  thermochemistry and `assemble` (exclusion, listing, empty-ensemble refusal), the census
  message helper. Two small new pure helpers are proposed at the highest point that needs
  no engine: the census floor verdict (spectrum + floor → verdict + counters) and the
  reference-level relaxation classification (relaxed record in → kept/saddle/soft_saddle +
  retry decision out). Everything else stays covered by the one existing integration test
  that runs a real MACE census.
- **Modules under test:** thermo (policy layer + drop/tripwire + `g_minus_eel` policy
  threading), the msRRHO ensemble (per-basin, assemble, record schema/round-trip),
  the census screen (verdict helper, message, record fields, criterion text),
  the reference level (classification helper, record/merge-map fields), the config seam
  (floor single-source equality assertion).
- **Prior art to follow:** `t_msrrho_presets.py` (synthetic spectra with literal
  frequencies, one `check(...)` per behaviour); `t_thermo_msrrho_calculation.py`
  (plain-dict basins, record key sets, `run_calculation` re-writes);
  `t_reference_level.py` (record block key sets); `t_branch_a_crash.py` (census message
  text); `t_crest_entropy_seam.py` (fixture-driven seam, updated for the removed block);
  `t_toml_record_roundtrip.py` (schema round-trip); the MACE integration test
  (`tests/integration/`) for the wired path.
- **Cases to pin down (at least):** the −6.84 candidate admitted with one window mode and
  the thermodynamics inverting it; a −30 mode likewise; a −0.5 single mode dropped with
  `N_BELOW_FLOOR = 1` and the value stored; the same spectrum under the assertion mode
  excluded with the reason (kept as a comparison test only); a 3× sub-1 spectrum raising;
  a −195.79 candidate ejected and, as the only candidate, refusing with the message; a
  below-floor basin excluded from the ensemble with its reason and the ensemble refusing
  when it was the only basin; the four convergence classes including the flagged admit; a
  soft-saddle retry landing in a minimum versus staying a saddle (merge-map fields); the
  seam record (no spread block, unchanged counters); the config↔preset floor equality.

## Out of Scope

- Rerunning the five parked Tianhe molecules and any cluster operations.
- Changing the tighten target (`fmax = 1e-4 eV/Å` stays) or the analytic-Hessian default.
- The msRRHO presets' interpolation values (τ = 25, rotor caps) — untouched.
- The GFN2 seam's policy (`crest_native`) and its comparisons — untouched apart from the
  removed spread block in its record.
- Branch B draws, the Hessian-learning campaign's data (basin-only training, held-out
  generators) — only the D1 guard sentence is added.
- `hessian_compare` / `mode_curvature` semantics beyond passing the new default and
  dropping the deleted `refuse` argument.
- Deleting `crest_native`, or any change to `Thermo` presets' `grimme2012` (no ithr).
- Rewriting historical records or supporting their recomputation under `refuse`.

## Further Notes

- Primary sources kept in the feature directory: the research note
  (`research-soft-modes-primary-sources.md`, with the CREST/ORCA/VASP/Gaussian quotes and
  the Q9 literature section on off-stationary Hessian labels) and the prototype
  (`prototype_imaginary_mode_state_machine.html`). When this ticket is implemented, the
  prototype's pure module is lifted into the real modules and the HTML rides to a
  throwaway branch as the primary source of the validated state machine.
- Provenance, for the record comments: `ithr = −50 cm⁻¹` is CREST's `-ithr` default
  (`crest --entropy`); `1 cm⁻¹` is CREST's `vibthr` and ORCA's `CutOffFreq`; `25 cm⁻¹` is
  CREST's `-sthr` and ORCA's `QRRHORefFreq`; ORCA's default `TolMaxG = 3e-4 Eh/bohr`
  (manual 7.26) is the certification line; ORCA `ProjectTR`/`TransInvar` are the defaults
  pinned explicitly.
- The two decisions were taken on 2026-09-25 (grilling rounds; D1: interpret frequencies
  only at a same-method stationary point; D2: always project, never compute thermochemistry
  with projection off).
