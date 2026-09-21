---
status: accepted
date: 2026-09-21
---

# The fine-tune learns basin Hessians only; the msRRHO result is the deliverable; the extrapolation rows are reference, not gates

## Context

PHL (Rodriguez et al., the "projected Hessian loss" work openQHA's loss module follows)
trains ANI from scratch on OpenREACT with a Hessian on EVERY frame, stationary and
off-stationary alike. Our campaign labels frames at wB97M-D3(BJ)/def2-TZVPPD with ORCA
analytic Hessians, and a Hessian frame costs about 25x an energy-force frame: labelling
every campaign frame (basin + displaced + merged + saddle) was costed at ~22 days of 12
nodes; the basin frames alone (~19,000) at ~4 days (grilling rounds 9-10).

Rodriguez 2025 ("Does Hessian data improve the performance of MLIPs", JCTC) trained on
STATIONARY points only (RTP: 35,087 minima and transition states) and evaluated on IRC
and normal-mode-sampled frames, MD stability ramps and vibrational frequencies: Hessian
training at stationary points cut the off-stationary Hessian RMSE by 70-90 % and halved
the force error, without any off-stationary Hessian in the training set (tables 2-4).

The deliverable of the Hessian-learning set is the msRRHO thermochemistry of molecules
at their own minima (S0-C-54): entropy, enthalpy and free energy from the Hessian at the
basin. Nothing downstream reads a Hessian away from a minimum.

## Decision

`dataset.build(train_generators=("basin",))` is the default: only basin frames enter
`train` and `valid`. Every labelled frame of another generator -- `displaced`, `merged`,
`saddle` -- is a **Held-out generator** frame: routed to `test` above the frame draw and
above any previous index, marked `held_out_generator = yes`, counted separately
(`N_TEST_HELD_OUT`, per class and per molecule). The label rule
(`frame_labels.HESSIAN_GENERATORS`) is untouched: which frames carry Hessians is one
question, which frames train is another.

The training target is the Cartesian matrix (S0-C-53), validated on four probes fixed
per frame (S0-C-55). The judge's verdict reads the thermochemistry rows at the seven
pinned molecules' own minima, the in_distribution no-degradation row and the forgetting
line; the off-stationary rows -- the H / E / F metrics of the held-out generator's frames
binned by RMS displacement, the MD temperature ramp, the frequency rows -- are REFERENCE
rows: reported, never gated (ticket 22). No transition-state location, no NMS generator,
no "Hessian on every frame" second stage is planned; round 8's from-scratch / RI-MP2 /
surface programme is shelved.

## Consequences

* The Hessian bill is the basin frames' (~4 days of 12 nodes on draw300), not the
  campaign's (~22).
* The displaced frames keep their E/F labels where they have them and become the
  extrapolation readout Rodriguez used: what a basin-only Hessian fine-tune buys away
  from the minimum is measured, in the report, and never promised.
* A Dataset built before this ADR that had displaced frames in `train` moves them to
  `test` on the next rebuild (the rule is above the kept decisions); its
  `TRAIN_GENERATORS` names the change.
* Training merged or saddle frames later is one flag (`--train-generators basin merged`),
  and their frames are drawn fresh then, since a held-out routing was never a draw.
* If the reference rows show the basin-only model degrading the forces or the MD
  stability of the seven relative to the base, that is a finding to report beside the
  msRRHO gate, not a trigger for a second training stage.
