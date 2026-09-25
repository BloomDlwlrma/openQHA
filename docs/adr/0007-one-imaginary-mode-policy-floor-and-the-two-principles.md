---
status: accepted
date: 2026-09-25
---

# One production imaginary-mode policy on CREST's floor; the census floor, the certified tighten and the reference retry; frequencies are interpreted only at a same-method stationary point

(Ticket 34 planned this ruling as ADR 0006; that number is the PHL-verbatim ADR of
2026-09-23. This is 0007.)

## Context

Branch A lost whole molecules to sign flips at the bottom of the spectrum. On Tianhe five
molecules died with "no basin survives the tightening and the imaginary-frequency filter"
while the candidate's lowest mode sat at -6.84 cm^-1 -- indistinguishable from numerical
noise at the decision resolution. Meanwhile the thermochemistry layer carried three
imaginary-mode policies with no production decision (`refuse` was the default "until the
ruling"), every record carried a three-policy spread nobody acted on, the census had no
frequency floor at all, sub-1 cm^-1 anomalies had no defined semantics, a reference-level
relaxation that landed on a soft saddle was thrown out without a second look, and a frame
whose tighten stopped short of `fmax` was never judged against any stated certification
line (ticket 34).

The program documentation is unanimous that a frequency is only meaningful at a
stationary point of the method that produced it (Gaussian's `Freq` page: "meaningless ...
other than a stationary point for the method used for frequency determination"; VASP:
tighten the relaxation and recompute before drawing conclusions; ORCA: "Significant
negative frequencies ... prove that the optimization has not resulted in an energy
minimum"), and that projection is not optional for thermochemistry (ORCA manual 7.27:
with `PROJECTTR` off the entropy "exhibits a finite jump" as the rigid-body modes cross
`CutOffFreq`). The primary sources and the quotations are collected in
`.scratch/msrrho-gtotal/research-soft-modes-primary-sources.md`, with the prototype state
machine beside it.

## The two principles, dated 2026-09-25

(The user's rulings, restated in the research note; D1 and D2 are its labels.)

* **D1 -- interpret frequencies only at a stationary point of the same method.** A sign
  flip on an unconverged geometry is a geometry problem, not a frequency policy: the
  census tightens first and reads the spectrum after, the reference level re-optimises
  before any verdict, and a frequency is never computed at a foreign geometry. D1 governs
  the census minimum verdict, the thermochemistry at basins and the reference-level
  basins. It must NOT be applied to the Hessian-learning labels: a displaced frame's
  label is deliberately off-stationary curvature at the frame's own geometry, never
  optimised (`workflows/hessian_learning/README.md`; ADR 0005).
* **D2 -- always project; never compute thermochemistry with projection off.** Our
  spectra are Eckart-projected; the reference-level ORCA inputs pin `ProjectTR true`,
  `TransInvar true` and `CutOffFreq 1.0` explicitly (ticket 36), and the ORCA interface
  records that the stored `$hessian` is the PRE-ASR matrix -- the sum-rule correction is
  internal to ORCA's frequency step.

## Decision

**One production policy, two implementations.** The policy set is
`{invert_below, crest_native}`. `invert_below` with the frequency floor
`ithr = -50 cm^-1` is the default everywhere (the per-basin thermochemistry, the
ensemble, branch A's RRHO label, the census's optional free energy); it takes a mode in
`[ithr, 0)` as `|omega|` and excludes the basin only when a mode lies below the floor.
`crest_native` remains, reachable only by the GFN2 seam, reproducing CREST 3.0.2 line for
line (a mode below ithr is kept negative with zero entropy but stays in ZPE, H and Cp).
The third policy, `refuse`, is removed from the code, the CLI and the tests; records
written before 2026-09-25 that name it stay readable as data, and their recomputation is
out of scope. No `--policy` flag remains on the production step: an impossible choice
cannot be wrong.

**The sub-1 cm^-1 rule (ORCA-style drop with recording, three-or-more tripwire).** A mode
with `|omega| < 1 cm^-1` is removed from every thermochemistry sum (S, Cp, H, ZPE) and
counted (`N_BELOW_FLOOR`, the values recorded). Three or more in one spectrum raise: that
count has no plausible physical explanation at a converged, projected minimum and is the
signature of an unprojected spectrum.

**The census floor and the inversion window.** The census screen (`census_from_frames`)
admits a candidate whose lowest mode lies in `[ithr, 0)` as a Basin, records it as an
**inversion window** (`n_inversion_window`, `lowest_frequency_cm_inv`), and ejects only
what lies below the floor as a saddle. The floor is single-sourced: the configuration's
`package2.ithr_cm` mirrors the `crest` preset's `ithr_cm`, and a unit test asserts the two
are equal. A molecule all of whose candidates lie below the floor still refuses, with the
message listing every condemned candidate's lowest mode and the tighten residual.

**The tighten certification.** Every candidate's tighten is judged against two lines: the
repository target (`fmax = 1e-4 eV/A`) and ORCA's default max-gradient (`TolMaxG =
3e-4 Eh/bohr = 1.543e-2 eV/A`). Classes: at or below the target `converged`; inside ORCA's
line `converged_orca_default`, admitted and marked `tighten_converged = false`; above the
line ONE bounded second optimisation pass runs (the same `conformers.optimise` with the
census's own bounds, appending to the same engine files -- the conformer's own `opt.traj`
and `opt.log`, so no new engine-file name is born): `converged_second_pass` when it
brings the residual inside the line, and `not_certified` -- rejected and listed with its
residual, a class of its own next to the frequency-floor saddles -- when it does not.
Nothing is ever admitted under a silent flag.

**The reference-level retry.** A reference-level relaxation whose lowest mode lies in the
inversion window is re-optimised once from the ORCA-relaxed geometry (D1 as a repair); if
it reaches a minimum the basin enters normally, if it stays a soft saddle it is excluded,
listed, and the merge map records `soft_saddle = true` with the lowest frequency. A saddle
below the floor is excluded as before. The reference record's `ITHR_POLICY` reads
`invert_below`.

## Provenance

| number | source |
|---|---|
| `ithr = -50 cm^-1` | CREST `-ithr` default (man page; `src/entropy/thermocalc.f90`); ORCA's "numerical error ... may reach 50 cm^-1" and VASP's 40-160 cm^-1 noise class bracket it |
| `1 cm^-1` | CREST `vibthr = 1.0` (drop) = ORCA `CutOffFreq 1.0`; our rule uses the same number as the drop floor with the three-or-more tripwire |
| `25 cm^-1` | CREST `-sthr` default = ORCA `%freq QRRHORefFreq 25` -- the `crest` preset's tau, unchanged |
| `1.543e-2 eV/A` | ORCA's default `TolMaxG = 3e-4 Eh/bohr` -- the certification line; the repository's own target stays `fmax = 1e-4 eV/A` |

## Considered options

* Keep the three-policy spread and decide later: rejected; it carried no production
  decision and every record promised a choice nobody acted on (ticket 34).
* Keep `refuse`: rejected; it loses whole molecules to noise at the decision resolution
  (the -6.84 cm^-1 candidate).
* Admit a doubly-non-certified frame with a flag (the prototype's `admitted_flagged`):
  rejected by the ruling of 2026-09-25; a frame that misses ORCA's own line after its
  second pass must not enter the sum.
* Apply D1 to the Hessian-learning frames (optimise before labelling): rejected; it would
  destroy the off-stationary labels the fine-tune trains on and invalidate the held-out
  rows (ADR 0005).

## Consequences

* A molecule whose only imaginary mode sits in `(-50, 0) cm^-1` finishes branch A as a
  basin list with the window recorded, instead of being refused; the thermochemistry
  inverts the window modes.
* A spectrum with three or more sub-1 modes raises instead of being silently dropped into
  a wrong entropy.
* Non-stationarity is visible: `converged_orca_default` carries `tighten_converged =
  false`; a double failure is rejected and listed, never admitted.
* One record per fact: `[Imaginary_Spread]` is gone; per-basin `N_INVERTED`,
  `N_BELOW_FLOOR` (with values) and `EXCLUDED_REASON` replace it; the census record
  carries `convergence_class` and `tighten_converged` per basin plus the four class counts
  and the `not_certified` list; the merge map carries `soft_saddle` and
  `lowest_frequency_cm`.
* The GFN2 seam keeps its line-for-line CREST claim under `crest_native`; only the
  removed spread block is absent from its record.
* The D1 principle must not be over-applied to the Hessian-learning campaign: labels are
  computed at the frame's own geometry and never optimised; training rows are stationary
  basin frames, the off-stationary generators are held out (ADR 0005; Rodriguez 2025;
  PHL's NMS rationale).

Records: tickets 34-40 (`.scratch/msrrho-gtotal/issues/`) and
`research-soft-modes-primary-sources.md`; CONTEXT.md carries the vocabulary (frequency
floor, inversion window, imaginary-mode policy).
