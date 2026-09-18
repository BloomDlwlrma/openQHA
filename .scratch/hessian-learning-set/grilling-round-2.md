# Hessian learning: grilling, round 2 (2026-09-17, open)

Opened at the close of the msrrho-gtotal proposal (`../msrrho-gtotal/closing-report.md`).
What is settled and feeds this round: HIP's metric set is the ruler, PHL is the loss
candidate, neither is the potential (round 1 rulings); MACE-OFF23_medium is the base to
fine-tune (plan_C §5.1); the reference level is wB97M-D3(BJ)/def2-TZVPPD (S0-C-33, ADR
0004); targets = every gated QM9 molecule, held-out = `IN_TRAINING = false` molecules,
active learning may only add from outside the held-out set (round 5 rulings); the four
shipped molecules are in SPICE, the three rings and 99.85 % of the QM9 targets are not
(S0-C-40/41); out-of-distribution error is curvature in the low modes, in-distribution
error is relative energy (closing report §5); oxetane's puckering mode is anharmonic at
every level (the correlated levels see a double well -- RI-MP2 at its own planar geometry has
an imaginary puckering mode -- wB97M and MACE a single well; far-IR barrier 15.5 cm^-1).

Facts checked for this round (not asked of the user):
- `mace-torch 0.3.16` in `openqha`: the model forward has `compute_hessian` (AD Hessian
  output, `models.py:284`) but `modules/loss.py` has no Hessian or HVP term -- a loss has to
  be written either way (full E-F-H or PHL).
- SPICE / MACE-OFF23 training data is on disk (Apollo parquet, E and F per frame, split by
  frame) -- plan_C ★16 chose PEFT "because we do not have the upstream set"; that reason
  no longer holds.
- Analytic wB97M Hessian: 156-223 s per 10-atom basin at 8 ranks (S0-C-38); DLPNO-CCSD(T)
  NumFreq: ~4.5 days per basin -- a judge, never a label.
- CONTEXT.md has no entry for model error / level error / approximation error, Dataset,
  Label, held-out; `[Training_Set]` and the tiers exist only as record keys.
- `mode_curvature` measures the finite-difference self-check (|omega_fd - omega_r|): 8.4
  cm^-1 on oxetane's puckering mode (quartic), 1.1 cm^-1 for MACE on the same line, < 0.01
  on a harmonic mode -- a usable anharmonicity flag exists at zero cost.
- Added after the ticket-32 dry run (same day, issues/32 findings 1-5): (i) the Cartesian
  curvature of a level at a foreign geometry carries a gradient term -- RI-MP2 136.2 cm^-1
  at its own minimum (experiment 135.1) but 93 at the wB97M geometry 0.009 A away, canonical
  CCSD(T) -65 there; MACE at x_r vs own minimum differs by 10 / 3 / 28 cm^-1 on the rings'
  lowest modes; (ii) the analytic wB97M Hessian's softest mode moves 131.3 -> 137.2 between
  DefGrid2 and DefGrid3 (energies' finite difference: 140.8) -- ~10 cm^-1 of label noise
  with the production grid; (iii) DLPNO-CCSD(T) energies carry ~5e-6 Eh PNO noise between
  distinct geometries: no numerical Hessian, no numerical optimisation at that level;
  canonical CCSD(T)/cc-pVTZ is smooth at 6 min a point (10 atoms).

- `mace-torch 0.3.16`, the Hessian code path read in full (2026-09-18): `modules/utils.py`
  `compute_hessians_vmap(forces, positions)` = one vjp of `-forces.view(-1)` per Cartesian
  component (`torch.vmap` over `I_N`, chunk 1 or 16, loop fallback with `.detach()`), with
  `create_graph=False` -- the returned Hessian is NOT differentiable w.r.t. the parameters,
  so it cannot sit in a loss as shipped; `get_outputs(..., compute_hessian)` only makes the
  force graph with `create_graph=True` so that the vjps exist. `models.py` `MACE.forward`/
  `ScaleShiftMACE.forward` expose `compute_hessian` and return `"hessian"`
  ([3N_batch, n_nodes, 3], block-diagonal over the batch); `calculators/mace.py`
  `get_hessian()` is inference only. `modules/loss.py` has energy / forces / stress /
  virials / dipole / polarizability terms, per-config weights from `ref.weight *
  ref.<prop>_weight`, no `hessian`; `data/atomic_data.py` has no `hessian` field or weight;
  `tools/train.py` calls the model with `compute_force/virials/stress` only;
  `scripts_utils.get_loss_fn` has no Hessian branch. Everything for a Hessian loss --
  data field + key, weight, loss term, `output_args["hessian"]`, the model call -- is to be
  written; the HVP itself is one `torch.autograd.grad(-(forces * v).sum(), positions,
  create_graph=True)` on the existing training graph.

## Round 2 questions

Q1 Label level. (a) wB97M-D3(BJ)/def2-TZVPPD analytic (the base model's own reference: the
   label corrects curvature without moving the energy level; 3 min/basin); (b) RI-MP2
   (plan_C §5.3 as written in August); (c) DLPNO-CCSD(T) numerical. Recommended (a); (b)
   is a level shift on top of a model correction (S0-C-33) and (c) is unaffordable as a
   label; CCSD(T) stays the judge of the reference itself (ticket 33).
Q2 Set composition. (a) the 7 known molecules only; (b) 7 + an out-of-distribution draw
   from the QM9 targets (IN_TRAINING = false), stratified by ring size and heteroatom
   pattern, whole molecules held out; (c) the topology-disagreement sites only.
   Recommended (b) with the smoke set = the 7, first draw 200 molecules (~1-3 basins each,
   ~1 day of wB97M Hessians at 8 ranks per 100 basins), the data-efficiency scan of
   plan_C acceptance 5 at 50/100/200 molecules; (c) as a weighted subset, not the set.
Q3 Geometries per basin. (a) basins only; (b) basins + n thermal displacements (<= 0.15
   A, seeded) with the H label at every displaced point (PFT table 1: E/F alone on
   displaced points is harmful); (c) + the along-mode line points of `mode_curvature`.
   Recommended (b) n = 4, plus (c) where a profile exists -- they are already labelled.
Q4 Loss. (a) full E-F-H (AD Hessian, exact, 25x per epoch, 3N <= 60 affordable);
   (b) PHL E-F-HVP with projected mass-weighted random probes (plan_C corrections 1-2);
   (c) full-H on the smoke set to measure the ceiling and the cost, then PHL for the
   200-molecule set. Recommended (c); the cost of one full-Hessian epoch on the 7
   molecules is the number that decides, measured before the set is labelled.
Q5 Low-mode weighting (plan_C ★2, C3-a/b/c/d). The rings put the error at < 300 cm^-1
   where dS/d omega ~ 1/omega. (a) weight each projected mode by |dS_msRRHO/d omega| at
   298 K (C3-a spectral shaping, the entropy's own sensitivity); (b) flat; (c) band
   weights. Recommended (a), with the msRRHO interpolation weight so the rotor-treated
   modes below tau are not over-driven.
Q6 Anharmonic modes: approximation error vs model error. Oxetane's -1.39 cal/mol/K is the
   difference of two harmonic numbers on a non-harmonic mode. (a) the judge excludes a
   mode whose FD self-check exceeds a threshold (e.g. 5 cm^-1) or omega_r < 30 cm^-1 from
   the thermochemical tier, reports it in an `[Anharmonic]` block with the 1-D level count
   from the profile (oxetane: fundamental 52.92 cm^-1 is the experimental check); (b) keep
   it in the tier and footnote; (c) a 1-D anharmonic correction enters G_total. Recommended
   (a) now, (c) as a later proposal; the training label on such a mode is kept (round 1
   Q3: never refused) but its loss weight is what Q5 gives it.
Q7 Forgetting. (a) co-training with a SPICE subsample (K = 4 upstream E/F steps per
   Hessian step, PFT algorithm 1) now that the data is on disk; (b) PEFT/LoRA; (c) both,
   compared on the SPICE test split. Recommended (a), judged by acceptance 7a rewritten
   as: E/F error on the SPICE test split (50,195 frames, or a fixed 5,000-frame draw)
   within 15 % of MACE-OFF23_medium's own.
Q8 The judge's thresholds, by distribution. Held-out out-of-distribution molecules (rings
   and the withheld draw): low-mode frequency MAE must fall below the in-distribution
   value (8.5 cm^-1 propanal) and |MODEL_ERROR_S_REF| <= 0.2 cal/mol/K excluding Q6 modes;
   in-distribution (four shipped): no degradation beyond 15 % on any HIP metric; 7a/7b/7c
   reported together, in- and out-of-distribution in separate rows. Recommended as
   stated; thresholds are must-fail on the 0.9x-scaled potential and must-pass on
   identical-to-base.
Q9 Nouns and files. (a) CONTEXT.md gains Model error / Level error / Approximation error
   (the three subtractions), Label (a reference-level E, F, H at one geometry), Dataset
   (an aggregate of Labels over many Calculations with a split column), Held-out;
   (b) the Dataset lives at `<root>/<tag>/_datasets/<name>/` with `index.dat` naming
   every contributing Property file and the split; (c) training input format: extxyz
   with `hessian` as a per-frame flattened info key (MACE's loader reads extxyz; the
   Hessian key needs a loader change either way) rather than the OpenREACT HDF5 of plan_C
   §4.3, which then becomes an export. Recommended all three.
Q10 Where it runs. Smoke (7 molecules, full-H loss) on the local WSL CPU or the A800
   (74 ms/step measured); the 200-molecule fine-tune on tianhe-ai A800 by the existing
   submission path; wB97M labels locally at 8 ranks (100 basins/day) or as an hkuhpc
   bundle via `orca_jobs`. Recommended: labels local first (they start today), training
   on the A800.

Q11 The CCSD(T) reference after the dry run (ticket 32 rewrite). (a) canonical
    CCSD(T)/cc-pVTZ: NumGrad optimisation from the RI-MP2 geometry (~600 points, 2-3 days
    at 8 ranks per basin), then along-mode lines at the CCSD(T) minimum for the modes
    below 300 cm^-1 (7 points, ~45 min per mode) with the wB97M-DefGrid3 Hessian for the
    rest: a per-mode CCSD(T) reference for the modes that carry the entropy; (b) RI-MP2
    (analytic gradients, TightOpt + NumFreq 15 min per basin) as the wavefunction tier in
    `level_compare`, CCSD(T) dropped; (c) both, MP2 now and CCSD(T) low modes as the
    Batch; (d) drop the wavefunction tier, experiment is the judge of wB97M.
    Recommended (c): MP2 is a day's work and already agrees with experiment on the
    torsion; the CCSD(T) low-mode lines are the only affordable CCSD(T) Hessian
    information and they answer the question ticket 32 asked.
Q12 Reference Hessian settings. The production wB97M records use ORCA's default grid;
    the softest mode carries ~10 cm^-1 of grid noise, DefGrid3 halves it at 2.1x cost
    (8.7 vs 4.1 min per basin). (a) recompute every reference Hessian with DefGrid3 and
    declare the grid in the level's keywords (`wB97M-D3BJ def2-TZVPPD TightOpt Freq
    TightSCF DefGrid3`); (b) keep DefGrid2 and carry the 10 cm^-1 as the label noise
    floor in the judge; (c) DefGrid3 for Hessian-learning labels only.
    Recommended (a): one hour of ORCA for the seven molecules; the labels and the
    reference tiers then share one grid.
Q13 Two curvature quantities, both reported. The Cartesian Hessian at a fixed geometry
    (gradient term included) is the Hessian-learning target and what `hessian_compare`
    measures at x_r; the harmonic frequency at each level's own minimum is what the
    thermochemistry uses. (a) `hessian_compare` adds, per reference mode, the engine's
    own-minimum frequency matched by block overlap (`OMEGA_ENGINE_OWN_CM`) beside D_ii,
    and the judge's low-mode threshold (Q8) is stated for both; (b) at-x_r only (as
    ruled for ticket 27); (c) own-minimum only.
    Recommended (a); the rings' lowest-mode errors at x_r (-13 / -57 / +66) are -10 / -29 / +76
    at the engine's own minimum, and a reader must see both.

Round 3 waits on Q1, Q2, Q4 (set size and loss fix the code), Q9 (CONTEXT wording),
Q11-Q12 (they rewrite ticket 32 and the reference records).
