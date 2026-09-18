# Spec: the frame set and the first Workflow (`workflows/hessian_learning/`, steps 01-04)

Synthesised 2026-09-18 from grilling rounds 3-4 (`grilling-round-3-frames.md`,
`grilling-round-4-frames.md`) and their rulings. Steps 05 (train) and 06 (judge) wait on
round-2 Q1-Q13 and are out of scope here.

## Problem statement

MACE-OFF23's out-of-distribution error on QM9 rings is curvature at the minimum
(hessian_compare, S0-C-41); correcting it needs reference E-F-H labels at and around the
basins of many QM9 molecules, in a form MACE can train on and a judge can read. Nothing
in the repository produces frames, splits, or a training file; SPICE's frame recipe
(hot MD + cooling) and OpenREACT's (stationary points + hot NMS) were both read and
neither fits: our seeds are MACE basins and the entropy error lives within 0.15 A of them.

## Solution

Four numbered drivers under `workflows/hessian_learning/` that call `openqha/` code:

1. `01_select.py` -- the molecule list: gated QM9 targets with branch A done under a
   tag, membership in SPICE from `qm9_targets_membership.dat`, stratification keys (ring
   count, heteroatom pattern, heavy atoms), the 7 known molecules pinned; writes
   `<root>/<tag>/_datasets/<name>/select.{out,toml,dat}`.
2. `02_frames.py` -- per molecule, the Frame set at the MACE level: `basin` (1 per
   basin), `displaced` (4 per basin, `hessian.thermal_displacements` at 298.15 K, RMS
   <= 0.15 A, seed = hash(qm9_index, basin, generator, k)), `merged` and `saddle` (one
   per merge-map row of that status, geometry from branch A's engine files); MACE E, F,
   H at every frame (`get_hessian`, raw Cartesian eV/A^2, flattened); filter |F|max > 10
   eV/A or a changed bond graph -> dropped and counted; writes
   `<molecule>/frames/<generator>.mace-off23_medium.extxyz` and
   `<molecule>/frames/frames.{out,toml}` (the Record: counts per generator, dropped,
   seeds, `engine.provenance()` incl. the parameter fingerprint).
3. `03_labels.py` -- per frame, the reference E, F, H at `wb97m-d3bj_def2-tzvppd`
   (`orca.LEVELS`, single point + analytic Hessian, `! wB97M-D3BJ def2-TZVPPD TightSCF
   Freq`, DefGrid per round-2 Q12), full `.out` kept under
   `<molecule>/orca/<level>/frames/<generator>_<basin>_<k>/`; writes
   `<molecule>/frames/<generator>.<level>.extxyz` with positions identical to the MACE
   file to 1e-8 A (refuses otherwise). Runs as a tianhe Batch: 4 ORCA ranks per frame,
   16 frames concurrently per node, 1 node for the smoke set, 12 nodes for the
   200-molecule draw; a local run is the same driver with `--local`.
4. `04_dataset.py` -- the Dataset: reads every Frame set of the selected molecules,
   assigns the split (test = whole molecules incl. the pinned 7; valid = 10 % of the
   training molecules' frames by frame; train = the rest; pool = frames without a
   reference label), writes `{train,valid,test,pool}.<level>.extxyz` and `index.dat`
   (molecule, basin, generator, k, split, levels present, seed, engine fingerprint,
   ORCA version), and `dataset.{out,toml}`; `--export openreact` writes the HDF5 form.

## User stories

- As the person training, I run `run.sh --tag rings --limit 7` and get a Dataset whose
  every frame has MACE and wB97M E-F-H at the same positions, a split I can read from one
  table, and a Record saying what was dropped and why.
- As the judge, I read `index.dat` and know for every frame which molecule, basin,
  generator and split it came from, and whether that molecule is in SPICE.
- As the person resubmitting, I re-run `03_labels` and finished frames are skipped (the
  `.out` terminal line), unfinished ones are rerun; nothing is recomputed at MACE.
- As a reader of the paper, I find the frame recipe in CONTEXT.md (Frame, Frame set,
  Dataset, Workflow) and in `workflows/hessian_learning/README.md`, with SPICE's and
  OpenREACT's recipes stated beside ours.

## Implementation decisions

- Frames live under `<molecule dir>/frames/`, one extxyz per generator per level (Q1,
  Q5); keys `qm9_index`, `basin`, `generator`, `k`, `seed`, `level`, `energy`, `forces`,
  `hessian` (3N x 3N flattened, eV/A^2), `smiles`.
- Generators and counts: basin 1, displaced 4, merged and saddle one each (Q2); no MD,
  no cooling, no hot NMS (round-3 rulings; recorded as considered).
- Split: valid by frame within training molecules, test by molecule, pool for
  unlabelled (Q3); set at write time, a column of `index.dat`.
- Order: smoke set (7 molecules) end to end on tianhe, 1 node, 4 x 16; then the
  stratified 200 on 12 nodes (Q4 as ruled); `01`/`02` run over everything branch A has
  produced, `03 --limit` draws.
- Engine identity: `engine.parameter_fingerprint()` (SHA-256 over the sorted
  `state_dict`: name, dtype, shape, bytes -- container-insensitive) is restored and
  written into `provenance()`, every Frame-set Record and `index.dat`; a fine-tuned
  potential is registered in `ENGINES` with its filename, its Dataset index path and its
  training config hash as `source`, and `S0_ENGINE` selects it (Q7 and the user's
  question on weights).
- `thermal_displacements` gains `hessian=` (precomputed) so the basin's stored MACE
  Hessian is reused, and returns the seed per sample.
- MACE filter at generation (Q8): |F|max, bond graph vs the basin's SMILES (RDKit).
- Reproducibility: seeds hashed from (qm9_index, basin, generator, k); ORCA and MACE
  versions and the fingerprint in the Records.
- tianhe layout for `03_labels`: Parsl role `labels` in `hpc/resource_configs/tianhe_cpu.py`
  (16 workers x 4 cores per node, 1 node smoke / 12 nodes draw), partition `deimos`
  (64 cores, 512 GB, 7 days), `%pal nprocs 4`, `%maxcore 6000`, `$TMPDIR` scratch,
  ORCA 6.1.1 from `~/env_orca611.sh` (conda env `orca611`; verified on the login node
  2026-09-18); the tianhe `openqha` env needs `qc-procrustes` added.

## Testing decisions

- Unit (propanal fixture, no engine, no ORCA): a Frame set built from the fixture's
  basins and stored MACE Hessians has 1 + 4 + (merged) frames per basin with the
  declared keys and seeds; the same seed reproduces the same positions; a frame with a
  broken bond graph is dropped and counted; the wB97M file written from a fake label
  with positions perturbed by 1e-6 A is refused; `04_dataset` on two fake molecules
  gives valid frames only from train molecules, test only whole molecules, pool = frames
  without the level; `index.dat` round-trips; `parameter_fingerprint()` is identical
  across a `torch.save`/`load` round trip of the same weights and differs after one
  tensor is changed.
- Integration (engine): `02_frames` on propanal reproduces `hessian_at_<level>.npy` at
  the basin frame to 0 and its displaced frames have RMS <= 0.15 A.
- Smoke on tianhe: 7 molecules, every frame labelled, `index.dat` complete, wall time
  and memory per frame recorded in the Dataset Record.

## Out of scope

- `05_train`, `06_judge` (round 2 open); the loss (PHL vs full E-F-H); co-training;
  DefGrid choice for the reference (Q12 of round 2 -- the label step takes the level's
  keywords as declared, and a change there is one line); hot / cooled / NMS frames.

## Further notes

- SPICE: 10 RDKit conformers -> 100 ps 500 K OpenFF MD -> 25 max-min RMSD hot + 25 cooled,
  by-frame split, no Hessian. OpenREACT: RTP stationary points (train), IRC paths and
  NMS hot frames (test), Eh / Eh/bohr^2 HDF5. Ours: MACE basins + 298 K harmonic
  displacements + merged/saddle, E-F-H at two levels, by-molecule test.
