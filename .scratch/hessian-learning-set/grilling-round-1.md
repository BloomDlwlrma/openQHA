# Hessian-learning set and the judge: grilling, round 1 (2026-09-15, open)

Parked here so the questions survive a context boundary. Tickets 25 and 26 of the msRRHO
proposal were split out of `.scratch/msrrho-gtotal/spec.md`; they get their own spec once
this grilling closes. Facts checked before asking: plan_C §4.3 and §7.4 fix the HDF5 shape
(OpenREACT keys; three files: pretrain GFN2, finetune RI-MP2, base MACE; level in file
attrs; never mixed); `thermal_displacements` exists (0.15 A cap, seeded); the ORCA
interface has NumFreq and `verify_hess_frequencies`; the xtb interface has `--ohess`
only; `hessian.npy` per basin exists in the mace engine folder (raw eV/A^2, unprojected);
CONTEXT.md has no noun for an aggregate over many Calculations.

## Ticket 25, the Hessian-learning set

- Q1 Which geometries get labels: (a) basins only; (b) basins + n thermally displaced
  structures per basin; (c) basins + branch B frames. Recommended (b), n = 4.
- Q2 GFN2 label at a MACE geometry: (a) all `--bhess`, shift measured once per molecule;
  (b) `--hess` unless imaginary, then `--bhess`; (c) both stored. Recommended (a).
- Q3 Are imaginary modes a reason to refuse a label? Recommended no: labels are never
  refused; count and lowest frequency stored; refusal is a thermochemistry policy only.
- Q4 Train / hold-out: (a) whole molecules; (b) by displacement seed; (c) both.
  Recommended (c), split is a column in the index set at write time.
- Q5 Where the RI-MP2 labels run: recommended deimos (D0-75), four shipped species first
  (~85 NumFreq jobs, 40-80 h serial), the 23 existing conformers re-run for one
  provenance status.
- Q6 GFN2 file scope: recommended same geometries as RI-MP2 for the shipped species
  (pure subtraction) plus a larger GFN2-only superset as a separate file.
- Q7 Where the HDF5 files live and what they are called: recommended a new noun
  **Dataset** (aggregate over Calculations) under `<root>/<tag>/_datasets/`, never in
  the checkout; `index.parquet` names every contributing Property file.

## Ticket 26, the judge

- Q8 The reference for thermochemical deltas when RI-MP2 has imaginary modes at the
  MACE geometry: (a) frequency-level quantities only there; (b) `invert_below(-50)`
  labelled; (c) re-optimise (never). Recommended (a) as rule, (b) as labelled column.
- Q9 Thresholds: per-band signed bias reported only; per-basin dS_msRRHO <= 1.0
  cal/mol/K; dG_total <= 1.0 kcal/mol; 7a force RMS on basins within 15 % of base; 7b
  force RMS and projected mass-weighted Hessian error on held-out displaced structures
  both below base. Base always run alongside; identical-to-base gives zero deltas
  (must-pass); 0.9x frequency-scaled potential fails 7a (must-fail).
- Q10 Output: (a) per-molecule records, aggregate = Slurm table; (b) aggregate record
  under the Dataset with the verdict. Recommended (b).

Round 2 waits on Q1, Q4, Q7 (displacement counts per file, index columns, CONTEXT.md
wording for Dataset / Label).

## Candidate selection rule from the Step-1 grilling (2026-09-15, Q13)

Topology-disagreement points: basins that merge, or become saddles, when re-optimised at wB97M are where MACE and the reference disagree about the surface topology, i.e. where curvature differs most. Candidate label sites for ticket 25, with weights, not as the whole set. Also parked here from Q2: "how to use the wB97M Hessian to correct the MACE Hessian at the MACE geometry".

## Rulings parked from the msrrho-gtotal round 4 (2026-09-16): metric set and loss

- **Metric set = HIP's `scripts/eval_horm.py`** (hip-main, `hessian-train/hip-main`),
  ported name for name in ticket 27 of msrrho-gtotal: `hessian_mae` (element-wise,
  eV/A^2), `eigval_mae` and the Eckart variants (`analyze_frequencies_np`: mass-weight,
  Eckart-project, `neg_num`), `eigvec1_cos` / `eigvec2_cos` at sorted index,
  `eigvec_overlap_error = ||abs(Q_m^T Q_t) - I||_F`, `neg_num_agree`, `asymmetry_mae`.
  Our additions on top: degenerate-block overlap (HIP's per-index cosine is undefined
  inside a degenerate block), the low-mode (< 300 cm^-1) statistics that carry the
  entropy, the softening slope, and the curvature along each DFT mode
  `D_ii = L_r,i^T K_m L_r,i` -- which is PHL's Hessian-vector product with the DFT
  normal modes as deterministic probe vectors. The judge (Q9) reads these numbers.
- **HIP's architecture is not our potential**: its Hessian head is a direct prediction,
  not the second derivative of the energy that gives G_i; the thermochemistry needs one
  energy surface (Gonnheimer AD route on MACE). HIP is the ruler, not the model.
- **Loss for the MACE Hessian-learning step: PHL (E-F-HVP, Hutchinson)** is the
  candidate, ported into MACE fine-tuning (PHL-main is hippynn code; MACE has no
  Hessian loss upstream -- to verify). Open for round 2: for 10-20-atom QM9 molecules
  (3N <= 60) the full E-F-H loss (Rodriguez 2025) is affordable and is the accuracy
  ceiling PHL approaches; PHL's 24x matters when the set grows. Decide on the measured
  cost of one full-Hessian epoch on the four molecules before choosing.

## Input from round 5 of msrrho-gtotal (2026-09-17): training-set membership

Tickets 30/31 there produce, per molecule, whether MACE-OFF23 saw its conformers
(SPICE membership at three strictnesses, frame counts, config_type). The judge (Q9)
must report its thresholds separately for in-distribution and out-of-distribution
molecules; the held-out set for 7b should contain molecules of both kinds. Open for
round 2: whether the Hessian-learning set itself should prefer out-of-distribution
targets, and how the SI states the split.
