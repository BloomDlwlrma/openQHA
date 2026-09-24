# Hessian learning -- grilling, round 8: a from-scratch MACE at RI-MP2 with the PHL strategy, surface first, thermochemistry second (2026-09-21)

The user's ask: adopt PHL's strategy (Hessian-vector-product supervision on every frame) to
train MACE FROM SCRATCH at RI-MP2 accuracy; build (1) the full Hessian potential-energy
surface first, not only the thermochemistry; (2) the thermochemistry potential as a second
step on top of it.

## What this overturns, so it is said before anything is built

- S0-C-2 / S0-C-21: fine-tune MACE-OFF23 (base -> pretrain -> RI-MP2 fine-tune). A
  from-scratch model gives up the base's 951,813 SPICE E/F frames and its MD stability.
- The campaign's Level: wB97M-D3(BJ)/def2-TZVPPD (chosen so that the base's labels and ours
  agree, S0-C-33). RI-MP2 labels put the whole Dataset on a different energy zero: the
  `--E0s foundation` line of `run.py` and the base's E0s no longer apply.
- Round 6's order: the thermochemistry potential first, the surface set decided from its
  judge table (Q9 (a)). Here the order is reversed.

## Facts (all from this repo's own measurements)

- From scratch, the literature's scale: MACE-OFF23 951,813 E/F frames over 17,132
  molecules, 190 epochs; PHL trains ANI (a far smaller model) on a set where every
  configuration has a Hessian, ~10^5 configurations, up to 5,000 epochs, five seeds.
- RI-MP2/RIJK/cc-pVTZ on our own molecules: gradient 44.26 s per structure (D0-P3-23);
  `NumFreq` Hessian 1,613 s (10 atoms) / 3,360 s (15 atoms) on 4 ranks (S0-C-11), i.e.
  0.45-0.93 h per Hessian frame. ORCA has ANALYTIC RI-MP2 second derivatives (RHF
  reference, RIJK) -- not yet timed here; the wB97M analytic Hessian of propanal took 226 s
  on 8 cores (S0-C-38).
- Round 5's arithmetic: Hessians on every campaign frame (~100,000) at ~1 h x 4 cores =
  400,000 core-hours = ~22 days of 12 x 64-core nodes; RI-MP2 NumFreq is in the same band.
  A from-scratch set of 3-5 x 10^5 Hessian frames is therefore 2-4 months of the same
  12 nodes, or one month with analytic RI-MP2 Hessians if they time at ~1/3 of NumFreq.
- What already exists and is level-agnostic: the loss (`phl.py`, `phl_loss.py`: four
  weightings, exact `modes` probe), the fork (A: Label in the batch; B: external loss),
  `hvp.py`, the judge, `02_frames` / `03_labels` / `04_dataset`. `03_labels` is ORCA at one
  Level string; RI-MP2 is a different input template (`! RI-MP2 RIJK cc-pVTZ cc-pVTZ/C
  cc-pVTZ/JK TightSCF`) and `orca.parse_hess` reads the same `.hess`.
- Round 6's band table: the thermochemistry and surface potentials differ mainly by DATA
  (Hessians away from the minima) and by the judge; the weighting alone moves 1-8 % of the
  loss. A surface potential trained with Hessians everywhere already carries the
  thermochemistry information; step (2) is a re-weighting, not a second campaign.

## Questions (round 8) -- see the reply of 2026-09-21; answers recorded here when they arrive

## Facts added 2026-09-21 (second ask: PHL as-is on MACE-OFF23; MACE-OFF23-SC's method)

- MACE-OFF23-SC (Moore, arXiv:2405.18171; thesis §4.5 p. 55; stage-2 T05 notebook): a
  64-channel, L = 1 MACE trained FROM SCRATCH on SPICE v1 (augmented) + 55 synthetic
  atomic dimer curves that put a monotone repulsive wall below the shortest sampled
  distance. From scratch beat transfer learning from the OFF23 checkpoints "since the
  learned radial embeddings will differ substantially" -- the new data lives on a domain
  (r < r_s) the pretrained radial embedding never saw. Stage 2's corollary: fine-tune FROM
  the SC checkpoint with the radial embedding frozen (`--freeze`), because its fine-tune
  adds level shift and chemistry, no new short-range physics.
- The criterion this gives: scratch when the new labels force the low-level embedding onto
  a domain outside its support; fine-tune otherwise. Hessian Labels at basins / 298 K
  displaced frames are on the manifold the base already fits (S0-C-39: base Hessian MAE
  0.026-0.033 eV/A^2, frequencies 2.9-4.7 cm^-1); an RI-MP2 level shift is smooth and
  global (S0-C-35: systematically softer, band by band). Neither is an r < r_s event.
- PHL's loss as-is (Cartesian Gaussian probe, no projection, no mass weighting,
  `/(9N^2)`, weights 1 / 0.30 / 0.09 tuned for ANI in Hartree from scratch) is one setting
  of our loss -- `probe = gaussian, mode_weighting = none` -- except the Eckart projection,
  which we keep (S0-C-43/44: the rigid block is 29-100 cm^-1 at displaced geometries and
  is not curvature). Its weights do not transfer (the balance on MACE-OFF23 is measured,
  ticket 15: flat w_H = 1.75 at w_F = 100, entropy 1139.6).
