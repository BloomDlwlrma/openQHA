# Spec: Hessian learning for MACE-OFF23 (`workflows/hessian_learning/`, steps 00-06)

Concluded 2026-09-20. **The training side ("The loss, in one place", the training rows of the
Implementation decisions, tickets 10-16) is superseded by `spec-fine-tune-basins.md`
(S0-C-53..57, amended 2026-09-22 to S0-C-58/59/60): the Cartesian matrix is the target, basin
frames only, one production row R4 with `w_H` = the balance by default, the judge gates on the
matrix itself and the gate is closed.** Steps 00-04 and tickets 01-09 stand as written here.
Synthesised from grilling rounds 1-5 (`grilling-round-{1..5}*.md`)
and their rulings, the loss design `design-phl-loss.md` (approved 2026-09-18; its long form
with every derivation executed is `docs/tutorials/T03_openQHA_Theory_Projected_Hessian_Loss.ipynb`),
and tickets 01-09 as built. The 2026-09-18 version of this file covered steps 01-04 only
("steps 05 and 06 wait on round 2"); this version covers the whole Workflow. Where a
round-2 question is still formally unruled the spec states the option the tickets assume
and marks it OPEN (section "Rulings"), so approving a ticket rules it.

## Problem statement

MACE-OFF23_medium's out-of-distribution error on QM9 rings is curvature at the minimum:
on the three rings the lowest-mode error at the reference geometry is -13 / -57 / +66
cm^-1 (-10 / -29 / +76 at the engine's own minimum) against 4-8.5 on in-distribution
propanal, and oxetane's puckering mode alone costs -1.39 cal/mol/K (S0-C-41). An E/F loss
does not constrain curvature between its samples (the PFT finding: E/F fine-tuning on
displaced frames improved forces and degraded phonons), so correcting it needs reference
Hessians in the loss, at and around the basins of many QM9 molecules that MACE-OFF23 has
never seen, a loss that reads them without forming the 3N x 3N model Hessian at every
step, a judge that reads the shipped full Hessian, and a guard against forgetting SPICE.
The reference is the base model's own level, wB97M-D3(BJ)/def2-TZVPPD: the Label corrects
curvature without moving the energy level (round-2 Q1 (a)); the wavefunction ladder
(S0-C-45) is another tier, not a better reference.

## Solution

Seven numbered drivers under `workflows/hessian_learning/`, each a Batch (CONTEXT
*Workflow*) over `openqha/` code, plus the xargs-mode Slurm stage scripts that run 02/03/
branch A as arrays on tianhe (round-5 Q6 (a)) and the training job on the A800 cluster:

0. `00_draw.py` -- the campaign's molecule list: 23 structure classes
   (`configs/structure_classes.yaml`, SMARTS / ring rules, census-checked), 300 per class
   drawn from the gated QM9 targets with `in_training = false` at every match level, the
   UNION over classes, short classes taken whole with the shortfall recorded, the pinned
   seven always in; `draw.{out,toml,dat}`. Built (ticket 05): `draw300` = 6,458 molecules.
1. `01_select.py` -- the selection under a tag: every drawn molecule as a row with its
   classes, stratum, SPICE membership, pin status, `has_basins` / `has_frames` (the
   campaign's progress table); `select.{out,toml,dat}`. Built (04); classes column (07).
2. `02_frames.py` -- per molecule the Frame set at the MACE level: `basin` (1 per basin),
   `displaced` (4 per basin, classical 298.15 K harmonic draw from the stored MACE
   Hessian, RMS <= 0.15 A, seed = hash(qm9_index, basin, generator, k)), `merged`,
   `saddle` (one per merge-map row); MACE E, F, H at every frame; ANI-1's 275 kcal/mol
   energy window as the only filter, bond-graph change reported not dropped;
   `frames/<generator>.mace-off23_medium.extxyz` + Record. Built (02, S0-C-46).
3. `03_labels.py` -- per frame the reference Label at a FIXED geometry, never optimised:
   `wB97M-D3BJ def2-TZVPPD TightSCF EnGrad Freq` (analytic Hessian) for basin / merged /
   saddle frames, `EnGrad` only for displaced frames (round-5 Q7 (b)); the raw Cartesian
   Hessian, never pre-projected; ORCA files as a file group of the molecule directory
   `orca.<level>.<generator>_bBB_kK.{inp,out,hess,engrad}`, full `.out` kept, finished
   skipped, concurrent Batches claim frames; `frames/<generator>.<level>.extxyz` with
   positions identical to the MACE file (1e-7 A after the COM shift) + Record with wall
   time, memory, route, noise floor (basin frames). Built (03, 06, 09; S0-C-47, S0-C-50).
4. `04_dataset.py` -- the Dataset: the production split BY FRAME 90 / 5 / 5 with a
   per-frame seeded draw, the pinned seven a whole-molecule test set, `pool` for frames
   without the level; `{train,valid,test,pool}.<level>.extxyz`, the merged
   `mace_<name>.<level>.extxyz` in MACE's key form (`REF_energy`, `REF_forces`,
   `REF_hessian` where a Hessian exists, `has_hessian`, `split`), `index.dat` (molecule,
   basin, generator, k, split, classes, levels, seed, params_sha256, ORCA version),
   `dataset.{out,toml}` with a `[[Class]]` table, `--export openreact` (Eh / Eh/A / Eh/A^2).
   Built by molecule (04, S0-C-48); by frame + MACE form + classes = ticket 07.
5. `05_train.py` -- the fine-tune: openQHA's own entry point (mace-md's shape) driving
   `mace.cli.run_train.run(args)` of the fork `BloomDlwlrma/openQHA-Hessian@openqha-hessian` with `--loss external
   --loss_module openqha.training.phl_loss:build`, the projected Hessian-vector-product loss of
   `design-phl-loss.md` (eq. 6: Hutchinson probes `v~ = M^-1/2 P v`, HVP by one extra
   backward over the force graph, eq. 9; Rademacher k = 4 for training, `probe = modes`
   / the full matrix for exact evaluation, eq. 10), entropy-weighted modes (eq. 3,
   `w_i = |dS_msRRHO/d omega_i|` at 298 K), the Hessian term masked on frames without a
   Label, multihead replay on a SPICE subsample against forgetting (round-2 Q7 (a) as
   mace's `--multiheads_finetuning`); one run = `<root>/<tag>/_datasets/<name>/train/<run>/`
   with the mace config, log, checkpoints, and `train.{out,toml}` (loss curves per term,
   the epoch-0 balance, wall per epoch, the resulting model's `params_sha256`, the config
   SHA); the model registered in `ENGINES` by ticket 01's path. Tickets 10-13, 15.
6. `06_judge.py` -- the ruler: on the held-out frames the shipped full Hessian
   (`get_hessian`) against the Label through `hessian_compare` (HIP families 1-4, the
   along-reference-mode curvature `D_ii`, `MIXING`, the noise floors), rows per class and
   per distribution (the by-frame test = interpolation within drawn molecules; the pinned
   seven = the only out-of-molecule test; the four shipped molecules = in-distribution),
   thermochemistry at the fine-tuned model's OWN minima for the seven through the msRRHO
   pipeline (`S0_ENGINE`, `OMEGA_ENGINE_OWN_CM` beside `D_ii`, round-2 Q13 (a)), the
   forgetting judge (E/F on a fixed 5,000-frame SPICE test draw within 15 % of the base
   model), must-pass on the base model and must-fail on the 0.9x-scaled potential;
   `judge.{out,toml,dat}`. Ticket 14.

Stage scripts (`hpc/slurm/hl_*.slurm`, ticket 06): `hl_branchA` -> `hl_frames` ->
`hl_labels` as `--array=0-11` over `draw.dat`, each proved on `debug` with `LIMIT=16`;
`hl_train.slurm` on the A800 cluster (ticket 13). Progress: `s0_hl_progress.py` (08).

## The loss, in one place (design-phl-loss.md, verified in T03)

    L_H(theta)  = (1/n_vib) || P M^-1/2 (H_theta - H_r) M^-1/2 P ||_F^2                  (1)
                = (1/n_vib) ( sum_i (D_ii - lambda_i^r)^2 + sum_{i!=j} D_ij^2 )          (2)  = hessian_compare family 4
    L_H^W       = (1/n_vib) sum_ij w_i w_j (D_ij - lambda_i^r delta_ij)^2,  P_W = L_r W^1/2 L_r^T   (3)
    L^_H^(k)    = (1/(n_vib k)) sum_j || P M^-1/2 (H_theta v~_j - H_r v~_j) ||^2,  v~_j = M^-1/2 P v_j   (6)
    E[L^_H] = L_H (7);  Var = 2 (||B||_F^2 - sum_i B_ii^2) / (n_vib^2 k), Rademacher (8)
    H_theta v~  = -grad_x (F_theta . v~), create_graph=True                              (9)
    probe = modes, k = n_vib  ->  L^_H == L_H exactly                                    (10)
    L = w_E L_E + w_F L_F + w_H L^_H                                                     (11)

Algorithms 1-4 (probes; one training step; the ruler; acceptance A1-A7) are in
`design-phl-loss.md` section 3 and T03 section 9. The verified numbers T03 gives on the
propanal fixture are the oracles of the unit tests (section "Testing decisions").

What T03 changed in the rewrite plan (section 4 of the design): the HVP needs no change to
the MACE forward. In training mode the forces are already built with `create_graph=True`
and `batch.positions` is the leaf they were differentiated against, so the loss itself
takes `-grad_x (F . v~)` from `pred["forces"]` and `ref["positions"]` (T03 section 10:
this on the unmodified MACE-OFF23 graph equals a column of `get_hessian()` to 1.4e-14
eV/A^2). Design 4.1-4.3 (`compute_hessian_vector_products` in `modules/utils.py`, the
`hessian_probes` argument of the forwards, `calculators/mace.py get_hessian_vector_products`)
are therefore NOT added to mace: the inference-side HVP the acceptance tests need is
`openqha/training/hvp.py` over the calculator's model. Design 4.5 (the loss class in
`modules/loss.py`) moves to openQHA too, because mace's `train()` takes any `nn.Module`.
What remains inside mace is generic -- data fields and an external-loss hook -- and lives
in the fork (next section): `data/utils.py`, `data/atomic_data.py` (commit A, ticket 12);
`tools/arg_parser.py`, `tools/scripts_utils.py`, `tools/train.py` (commit B, ticket 13).

## User stories

- As the person training, I run `05_train.py --tag draw300 --probe rademacher
  --n-probes 4 --hessian-weight W` and get a run directory whose Record says the loss per
  term per epoch, what `w_H` balanced against `w_F` at epoch 0, the wall per epoch, and the
  fingerprint of the model it produced; `mace_run_train` without the new flags is unchanged.
- As the judge, I run `06_judge.py --tag draw300 --run R` and read one table: per class
  and per distribution the low-mode MAE, the HIP metrics, `D_ii` against `lambda_r`, the
  msRRHO entropy at the model's own minima for the seven, the SPICE forgetting number,
  and the must-pass / must-fail lines; nothing in it comes from the estimator.
- As the person paying for the campaign, I know from the smoke fit (ticket 15) the cost
  of one epoch with the full-matrix loss and with k = 4 probes, and the ceiling `L_H`
  reaches on the seven, before the A800 job for 6,458 molecules is submitted.
- As a reader of the paper, I find the loss derived in T03, the frame recipe in CONTEXT.md
  and `workflows/hessian_learning/README.md`, and every number of the judge in a Record.

## Implementation decisions

Frames, labels, Dataset (rounds 3-5, tickets 01-09; unchanged):
- Frame set keys, generators and counts (basin 1, displaced 4 classical 298 K, merged,
  saddle), ANI-1's energy window as the only filter; Labels at a fixed geometry; Hessian
  Labels on basin / merged / saddle frames only, `EnGrad` on displaced frames; KEEP =
  `.inp .out .hess .engrad`; the flat molecule tree `<root>/<tag>/<qid>/` with the frame
  jobs as file groups; one campaign = one tag = one Dataset; engine identity by
  `params_sha256`; the draw = union over classes, 300 per class, none in SPICE.
- Split: BY FRAME 90 / 5 / 5, per-frame seeded, the pinned seven whole-molecule test; the
  by-molecule split stays as `--split-by molecule` for the smoke Dataset.

The loss and the mace rewrite (round 2 Q4/Q5/Q7/Q13 as approved in the design, T03):
- PHL for the loss, the shipped full Hessian for the ruler. Training probes Rademacher
  k = 4 (Gaussian is measurably worse: variance 4.1e-2 vs 2.0e-2 on the same operator,
  T03 section 4); `probe = modes` is the exact loss and is used for the smoke fit's
  ceiling and cost; evaluation inside training computes (1) EXACTLY from the full matrix
  (`compute_hessian=True` in mace's forward at validation, `create_graph=False`) -- the
  logged validation Hessian loss is the ruler's quantity, never sampled.
- Mode weighting `entropy` (eq. 3) on by default; `none` available; the weight of a mode
  below `PROFILE_BELOW_CM` is what the formula gives it (the judge, not the loss, sets
  such modes aside -- Q6).
- Masking: a frame without a Label contributes E and F only; `n_vib` normalisation and
  the batch mean run over labelled graphs (round-5 Q7 (b)).
- `w_H`: set so that `w_H L^_H ~ w_F L_F` on the base model at epoch 0 (measured by the
  smoke fit), scanned x0.3 / x3; the Cartesian 0.25-0.30 band of PHL / Rodriguez does not
  transfer under mass weighting and is not used.
- Forgetting: mace's multihead fine-tuning (`--multiheads_finetuning True --pt_train_file
  <spice_subset.xyz> --num_samples_pt 5000`): the Hessian term applies to the fine-tuning
  head, the pretraining head keeps E/F on SPICE frames that are on disk.
- Delivery of the mace changes (ruled 2026-09-20, mace-md's pattern): a FORK.
  `BloomDlwlrma/openQHA-Hessian` (as `jharrymoore/mace@softcore` is for mace-md): mace-torch at upstream tag
  `v0.3.16` (`4d2da09`) re-rooted as one orphan commit = tag `base-v0.3.16` (the three bundled
  foundation-model binaries, 153 MB, left out), branch `openqha-hessian`, upstream kept locally, the
  base tag pushed so `git diff base-v0.3.16..openqha-hessian` is the whole change; package name
  `mace`, `__version__ = 0.3.16+openqha`; installed with `pip install -e` in every env
  (the tree travels with the xfer scripts; no GitHub needed on tianhe). What must touch
  mace's internals goes into the fork as generic, upstreamable commits (data fields;
  `--loss external --loss_module module:factory`; `evaluate` with `compute_hessian` when
  the loss asks for it); everything else -- probes, weights, HVP, the loss class, the
  driver -- uses mace's public API from `openqha/training/` and `05_train.py`, exactly as
  `mace_md` uses `MACECalculator` / `mace.tools` and installs its own `mace@softcore`
  branch. The fork's identity goes into every Record: `provenance()` gains `mace_version`
  and `MACE_FORK_COMMIT` (from the editable install's checkout; `unknown` or dirty refuses
  to train). Not `openqha/extensions/`: that subpackage is for optional cross-checks no
  production number depends on. (Considered and dropped: a patch set applied to
  site-packages with SHA-256 checks -- a home-made tool where git already exists; import-
  time hooks -- implicit and unreadable.)
- Where it runs (round-2 Q10, OPEN as stated): the smoke fit on the local WSL CPU (7
  molecules, 65 frames, minutes per epoch) or the A800; the campaign fine-tune on the
  tianhe A800 cluster through `hpc/resource_configs/tianhe_a.py`'s submission path,
  `hl_train.slurm`, one card; labels on tianhe CPU as the arrays.
- The smoke fit needs training frames: `04_dataset.py --no-pinned` builds `smoke_fit`
  from the same seven molecules with the by-frame split (the pinned rule off), used ONLY
  for the fit / cost / ceiling measurement, never for a judge claim about generalisation.
- Registration: a fine-tuned model enters `ENGINES` with `filename`, `source` = the
  Dataset index path + the config SHA, `params_sha256` from `s0_check_weights.py --pin`
  (ticket 01); `06_judge` and the msRRHO pipeline select it with `S0_ENGINE`.

## Rulings: what the tickets rest on

| question | status | what the tickets assume |
|---|---|---|
| R2 Q1 label level | de facto (a) | wB97M-D3(BJ)/def2-TZVPPD, ticket 03 built on it |
| R2 Q2 set | superseded by R5 | `draw300`, 6,458 molecules, per class |
| R2 Q3 geometries | (b), n = 4; Q14 (b) classical | S0-C-46; (c) line points not in the set |
| R2 Q4 loss | design approved; target re-ruled S0-C-53 | PHL trains on the **Cartesian** matrix (B = I); `modes` / full matrix measure; validation by four fixed probes (S0-C-55) |
| R2 Q5 weighting | superseded by S0-C-53 | `cartesian` is the default; entropy / none are diagnostics |
| R2 Q6 anharmonic modes | OPEN, assumed (a) | the judge sets aside modes with `omega_r < 30` or an FD self-check > 5 cm^-1 into `[Anharmonic]`; the loss keeps their Label with their weight |
| R2 Q7 forgetting | design approved (a); the Replay ruled S0-C-56/60 | multihead replay = a drawn SPICE train-split file (R4: 4 x `N_TRAIN_HESSIAN` frames at `config_weight` 10); judge = E/F on a 5,000-frame SPICE test draw within 15 % (the 15 % is a placeholder, never ruled) |
| R2 Q8 thresholds | OPEN, assumed as stated; **the gate re-ruled S0-C-59 and CLOSED S0-C-60** | gate rows: the held-out Hessian matrix itself (engine / base - 1 <= 0), in-distribution no HIP metric worse by > 15 %, forgetting <= 1.15x; reference rows: low-mode MAE 8.5 cm^-1, abs(`MODEL_ERROR_S_REF`) <= 0.2 cal/mol/K, RMS bins, the ramp; with the gate closed every row is reported and `VERDICT = REPORTED`; must-fail on 0.9x / must-pass on base holds with the gate open |
| R2 Q9 nouns/files | built | CONTEXT Frame / Frame set / Dataset / Workflow; `_datasets/<name>/`; extxyz with the flattened Hessian, HDF5 as export |
| R2 Q10 where | OPEN, assumed | smoke fit local or A800; campaign on the A800 |
| R2 Q11 CCSD(T) route | OPEN | msRRHO ticket 32/33, not this Workflow |
| R2 Q12 reference grid | OPEN, must precede the `hl_labels` array | the smoke labels were made with DefGrid2 (the level's `single_point` as declared); a DefGrid3 ruling is one word in `orca.LEVELS[...]["single_point"]` and a `LEVEL` string, and the judge's floor cannot be tighter than the grid noise (~10 cm^-1 on the softest mode at DefGrid2, S0-C-44) |
| R2 Q13 two curvatures | design approved (a) | `06_judge` reports `D_ii` at x_r and `OMEGA_ENGINE_OWN_CM` |
| R5 Q1-Q8 | ruled 2026-09-19 (S0-C-49) | as built in tickets 05-06 |
| tree, name | ruled 2026-09-20 (S0-C-50) | ticket 09 |

## Testing decisions

Unit (propanal fixture: ORCA `.hess` + stored MACE Hessian at the same geometry; no
engine, no ORCA), oracles from T03:
- probes: `P V = 0`; `v~ = M^-1/2 P v`; the reference side `r_j = P M^-1/2 H_r v~_j` is a
  matvec; entropy weights equal the analytic HO derivative to 4 digits at 131 cm^-1
  (1.463e-2 vs 1.466e-2 cal/mol/K per cm^-1) and eq. 3 holds to 1e-18.
- A2 exactness: `probe = modes` and `probe = cartesian` reproduce (1) to 1e-14; eq. 2 =
  `hessian_compare`'s `D_ii` / `MIXING` to 1e-14 (0.3427 <= 0.3697 <= 0.5448 is the Weyl
  check).
- A1 unbiasedness: 4000 Rademacher draws, mean within 1 %, variance within 5 % of eq. 8;
  Gaussian variance larger.
- A3 projection: `eps sqrt(m m^T) V V^T` added to the Label (300 cm^-1 rigid block) changes
  the loss by 0. A5 must-pass: `H_r := H_base` gives 0 loss and zero gradient. A6 must-fail:
  `H_r := 0.81 H_base` gives `0.19^2 ||K~||_F^2 / n_vib` and a non-zero gradient.
- A4 gradient: autograd `dL/dtheta` vs central finite difference to 1e-6 (T03: 1.5e-8) on
  a toy potential; A7: reverse-over-reverse = forward-over-reverse = explicit Hessian to
  1e-12.
- batch identity: a two-molecule batch's HVP equals the per-molecule HVPs (1e-16);
  a batch of two labelled + one unlabelled frame gives the loss of the two alone.
- data path: a `mace_<name>.<level>.extxyz` frame round-trips `REF_hessian` into
  `AtomicData.hessian` (flattened, concatenated like `ptr`), `sqrt_masses`, `has_hessian`.
- the fork: mace's own `pytest tests/` green on `openqha-hessian`; the external-loss hook
  exercised in the fork's tests with a fake loss module; `provenance()` reports
  `0.3.16+openqha` and a 40-hex `MACE_FORK_COMMIT`, `unknown` on a pip-installed 0.3.16.
Integration (engine): `openqha.training.hvp` on MACE-OFF23_medium vs a column of
`get_hessian()` at the fixture's basin to 1e-12 eV/A^2 (T03: 1.4e-14); `mace_run_train`
one epoch on the smoke Dataset on the fork and on pip's 0.3.16, `--loss energy_forces`,
gives the same loss to 1e-10 (the default path is untouched).
Smoke fit (ticket 15, engine, minutes-hours): `probe = modes` vs `k = 4` per-epoch wall and
the loss curve on `smoke_fit`; `06_judge` must-pass on the base model and must-fail on the
0.9x-scaled potential; the fine-tuned seven's low-mode MAE falls below the base model's.

## Out of scope

- The CCSD(T) / wavefunction reference tier (msRRHO tickets 32-33, round-2 Q11).
- The curvature-away-from-minimum probe of round-5 Q7 (c) (a judge-only measurement later).
- PEFT / LoRA (round-2 Q7 (b)), the 1-D anharmonic correction to G_total (Q6 (c)), active
  learning from the pool, the along-mode line points as one-probe Labels (design section 5,
  last bullet), hot / cooled / NMS frames.

## Tickets, in order (`issues/`)

| # | ticket | status | blocked by |
|---|---|---|---|
| 01 | engine identity by parameter fingerprint; registering a fine-tuned potential | done 09-18 | -- |
| 02 | the Frame set at the MACE level (`frames.py`, `02_frames.py`) | done 09-18 | 01 |
| 03 | reference E-F-H labels per frame (`frame_labels.py`, `03_labels.py`, tianhe Batch) | done 09-19 | 02 |
| 04 | the Dataset: split, files, index (`dataset.py`, `01_select.py`, `04_dataset.py`) | done 09-18 | 02, 03 |
| 05 | structure classes and the draw (`structure_classes.py`, `00_draw.py`) | done 09-19 | -- |
| 06 | xargs-mode stage scripts; displaced frames `EnGrad` only | code done 09-19; tianhe gates open | 05 |
| 07 | classes in the selection and the Dataset; by-frame split; the MACE-form file | open | 05 |
| 08 | campaign runbook, cost, progress | open | 05, 06, 07 |
| 09 | the flat molecule tree, NAME = TAG | code done 09-20; tianhe migration open | 06 |
| 10 | the mace fork `BloomDlwlrma/openQHA-Hessian@openqha-hessian` and the HVP entry points (`openqha/training/hvp.py`, `provenance()`, the install line) | done 09-20 | -- |
| 11 | probes, weights and the projected loss (`openqha/training/phl.py`, `phl_loss.py` with `build(args)`) | done 09-20; Cartesian path added by 21 | 10 |
| 12 | the Label in the batch -- fork commit A (`data/utils.py`, `data/atomic_data.py`, `--hessian_key`) | done 09-20 | 07, 10 |
| 13 | fork commit B (`--loss external`, `evaluate` full matrix, the flags) and `05_train.py` as openQHA's entry point, `hl_train.slurm` | done 09-20; commit C by 18 | 11, 12 |
| 14 | the judge, `06_judge.py` | done 09-21; rows re-ordered by 22 (S0-C-58/59/60) | 01, 07 (runs on the base model before 13) |
| 15 | the smoke fit: cost, ceiling, `w_H`, must-pass / must-fail | done 09-21 (the 65-frame programme is the tianhe run) | 13, 14 |
| 16 | the campaign fine-tune on `draw300` and its judge table | rewritten in `spec-fine-tune-basins.md`: ONE row R4, code done 09-22 | 08 (labels), 15 |
| 18-23 | the fine-tune as ruled (S0-C-53..60) | see `spec-fine-tune-basins.md` | -- |

## Further notes

- SPICE: 10 RDKit conformers -> 100 ps 500 K OpenFF MD -> 25 max-min RMSD hot + 25 cooled,
  by-frame split, no Hessian. OpenREACT: RTP stationary points (train), IRC paths and NMS
  hot frames (test), A / Eh / Eh/A / Eh/A^2 HDF5. PHL: E-F-HVP with random probes at
  stationary points. PFT: finite-displacement phonons at minima, co-training against
  forgetting. Ours: MACE basins + classical 298 K displacements + merged / saddle, E-F-H
  at two levels (Hessian on stationary frames), by-frame split with seven whole molecules
  held out, the projected mass-weighted loss with the entropy's own weights.
