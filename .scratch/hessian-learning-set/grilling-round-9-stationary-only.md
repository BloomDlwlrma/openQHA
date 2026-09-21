# Hessian learning -- grilling, round 9: a training set of stationary frames only (2026-09-21)

The user's ask: the training set holds stationary frames only -- `basin + merged + saddle`,
each with E, F and H -- and no displaced frame; validated through the four extrapolation
views of Rodriguez 2025 (held-out stationary / off-minimum by RMS bin / MD temperature ramp /
vibrational rows) plus the forgetting and in_distribution rows. Displaced-frame Hessians
are bought only if the judge says the off-minimum bins fall short.

## Facts checked before asking

1. **The label rule already is stationary-only.** `frame_labels.HESSIAN_GENERATORS =
   ("basin", "merged", "saddle")`; displaced frames get E/F today. What changes is the
   Dataset: `dataset.build` has no generator filter -- every labelled frame of a training
   molecule goes to train / valid / test by the 90/5/5 frame draw. Excluding a generator
   from train/valid is a code change (a `train_generators=` argument, recorded).
2. **"Stationary" means stationary on the ENGINE's surface.** A `basin` frame is the
   MACE-tightened minimum (fmax 1e-4); a `merged` frame is a CREST conformer that branch A's
   deduplication merged into an existing basin, geometry = its own tightened
   `conf.extxyz`, "no displacement"; a `saddle` frame is a tightened conformer rejected for
   imaginary modes. At the reference level these are single points with a small non-zero
   gradient (methyloxirane basin: |g_r|max 0.067 eV/A), never re-optimised. So a merged
   frame is a near-duplicate of its basin at the engine level; how near (RMSD, energy)
   is not yet measured on any set.
3. **Counts.** Smoke set: 65 frames / 7 molecules with 1 + 4 per basin -> ~13 basins,
   merged/saddle a handful. draw300 estimate (round 5): ~17 frames per molecule, ~5
   stationary -> ~32,000 stationary frames on 6,458 molecules, ~19,000 of them basins.
   ORCA at ~4 core-h per Hessian (19 atoms): ~130 k core-h = ~7 days of 12 x 64 cores.
4. **Off-minimum judge frames need H labels somewhere.** The smoke set's 52 displaced
   frames have H (classical 298 K draw). The campaign's displaced frames would need a
   labelled subset held out from training for the RMS-bin rows; E/F on all ~70,000
   displaced frames is ~19 k core-h (~1 day), H on 1 per basin for 500 stratified
   molecules ~6 k core-h (~8 h).
5. **Rodriguez's stationary set is one third transition states**; ours is nearly all
   minima (saddles are incidental). His NEB result rests on the TS third.
6. **The by-frame split changes meaning.** With stationary frames only, an
   "interpolation" test frame is another stationary point of a trained molecule (a
   different basin or a merged near-duplicate), not a displaced neighbour; valid at 5 %
   of ~32,000 = ~1,600 frames.
7. The judge has no RMS-displacement axis and no MD line yet (round 7/8 proposals);
   `rms_displacement_A` is on every frame and branch B's `calibration_md_stability` runs
   fixed-temperature MD with a failure criterion.

## Questions (round 9) -- see the reply of 2026-09-21; answers recorded here when they arrive

## Ruling (2026-09-21, S0-C-54) -- the round closes

The user: the saddle question is outside the project's scope; **train on basins only; the
deliverable is the msRRHO result; extrapolation (NMS, MD, vibration) is post-hoc
validation and reference only.** Consequences for the questions above:

| Q | outcome |
|---|---|
| Q1 | displaced frames are not trained on; the judge's off-minimum reference rows use the labelled displaced frames that exist (the smoke set's 52). Whether any campaign displaced frame gets a label at all is a cost line, not a training decision (open, minor). |
| Q2 | `merged` frames: the training generator is `basin`; merged frames are OUT unless the user says otherwise (open, one yes/no; their near-duplicate RMSD can be measured on the smoke set when needed). |
| Q3 | no TS location; saddle frames stay incidental and are not trained on; the Record reports their count. Closed. |
| Q4 | by-frame split kept; "interpolation" = held-out basin frames of trained molecules; the RMS-bin and MD-ramp rows are REFERENCE rows (reported, no PASS/FAIL). |
| Q5 | no second-stage trigger: displaced Hessians are not bought on the judge's say-so. Closed. |
| Q6 | the noun **Held-out generator** is still needed (displaced frames exist, are judged, never trained); to be written into CONTEXT.md with the Dataset change. |
| Q7 | ADR 0005 "basin Hessians only; the msRRHO entropy is the deliverable; extrapolation rows are reference" -- to be written with the ticket. |

Cost: ~19,000 basin frames x ~4 core-h = ~76 k core-h = ~4 days of 12 x 64 cores.
Superseded by this ruling: round 8 (from scratch / RI-MP2 / surface), `design-nms-frames.md`
(the NMS generator and ANI's S x K), the "every frame carries H" rule of
`framework-phl-efh-wb97m.md` section 2, and the stage-2 trigger of round 9 Q5.
