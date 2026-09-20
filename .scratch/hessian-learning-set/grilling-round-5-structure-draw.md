# Hessian learning -- grilling, round 5: the structure-class draw (2026-09-19)

The user's ask: a Dataset of QM9 molecules with special structures, **500 per structure
class, none of them in MACE-OFF23's SPICE training file**, labelled through the xargs-mode
Slurm array (`sbatch --array=0-11 hpc/slurm/hl_labels.slurm`), every script proved on
`debug` first. Classes named: rings (three-membered, small rings, bicyclic, aromatic,
polycyclic, eight-membered -- only 36 in QM9), carbonitrile, secondary alcohol,
trialkylamine, alkyne, tertiary amine, aromatic compound, dialkyl ether, heterocyclic,
epoxide, aldehyde, amide, carboxylic acid, ester, cyclopropane; also aliphatic / aromatic
amines, primary alcohols, ethers, carbonyls (COO- and other C=O).

Facts checked so you don't have to: the gated target table
(`data/training_sets/qm9_targets_membership.dat`) holds 119,450 molecules with their
relaxed SMILES, **119,275 outside the SPICE training file** -- the exclusion costs
almost nothing, the classes decide the draw. The smoke set's tianhe numbers: ~5 min and
~90 MB per rank per frame, 16 frames per node, no contention penalty; branch A ~8-10 min
per molecule, 16 per node. The Frame set of a molecule today is 1 basin + 4 displaced
frames per basin + the merged/saddle conformers (smoke set: 65 frames for 7 molecules,
9.3 per molecule).

---

❓ **Q1** - **Overlap between classes.** A molecule is usually in several classes
(a heterocyclic epoxide is small-ring, heterocyclic, epoxide, dialkyl ether). (a) draw
500 per class independently and take the UNION (a molecule counts for every class it
belongs to; total well under 20 x 500); (b) 500 DISJOINT per class, each molecule assigned
to its rarest class first; (c) the union, but classes with fewer than 500 candidates
outside SPICE (eight-membered rings: 36) take all of them and the shortfall is reported,
never filled from elsewhere.

➡️ (a) + (c). The judge asks "how does the fine-tuned model do on epoxides", and a
molecule that is also an ether is still an epoxide; disjointness would make the class
statistics depend on an arbitrary priority order. Shortfalls are a fact of QM9, stated in
the Record.

---

❓ **Q2** - **The class definitions.** SMARTS on the relaxed SMILES with RDKit, kept in
`configs/structure_classes.yaml` (name, SMARTS or ring rule, note), so a definition is a
config line and the census reprints. Proposed:

    three_membered_ring   [r3]                          small_ring   [r3,r4]
    bicyclic              >= 2 rings sharing >= 1 atom   polycyclic   >= 3 rings
    aromatic              CalcNumAromaticRings >= 1     eight_membered_ring  [r8]
    heterocyclic          [r;!#6]                       carbonitrile [#6][CX2]#[NX1]
    secondary_alcohol     [CX4H1]([#6])([#6])[OX2H]     primary_alcohol  [CX4H2][OX2H]
    trialkylamine         [NX3]([CX4])([CX4])[CX4]      tertiary_amine   [NX3;H0;!$(NC=O)]([#6])([#6])[#6]
    aliphatic_amine       [NX3;!$(NC=O);!$(N-a)][CX4]   aromatic_amine   [NX3]-a
    alkyne                [CX2]#[CX2]                   dialkyl_ether    [CX4][OX2][CX4]
    epoxide               C1OC1                         aldehyde         [CX3H1](=O)[#6]
    amide                 [CX3](=O)[NX3]                carboxylic_acid  [CX3](=O)[OX2H1]
    ester                 [CX3](=O)[OX2][#6]            ketone           [#6][CX3](=O)[#6]
    cyclopropane          [C;r3]1[C;r3][C;r3]1 (all-carbon three-ring)

(a) this table, checked by a census over all 133,885 QM9 SMILES against the counts you
quoted (carbonitrile 10,315, secondary alcohol 10,668, trialkylamine 10,687, alkyne
10,873, tertiary amine 11,057, aromatic 15,863, dialkyl ether 24,012, heterocyclic
61,904, eight-membered 36): a pattern whose count is off by more than 10 % is reported
as suspect and shown to you before the draw; (b) you supply the SMARTS.

➡️ (a). The census is the test of the definitions; the numbers you quoted are the
oracle.

---

❓ **Q3** - **Frames per molecule, i.e. the cost.** Union of ~20 classes x 500 with heavy
overlap: estimate 5,000-8,000 molecules. At 9.3 frames per molecule that is 45,000-75,000
labels; at 5 min per frame on 192 slots (12 nodes) = **20-33 hours of 12 nodes** for
step 03, plus 4-7 hours for branch A. (a) keep 1 + 4 displaced per basin (the smoke
recipe); (b) 1 + 2 displaced per basin for the draw, the smoke set keeps 4 (halves step
03); (c) 1 + 4, but cap at 8 frames per molecule (drop merged/saddle beyond the first).

➡️ (a). The recipe is what the smoke set was built and judged with; changing it between
the smoke set and the draw makes the two incomparable, and 30 hours of 12 nodes is
inside deimos's 7-day limit as one array job with resubmission. If the queue says
otherwise, (b) is one flag (`--n-displaced 2`) and a Record column.

---

❓ **Q4** - **The split of the draw.** Test today = the pinned 7 + 10 % of the other
molecules per stratum (ring count x heteroatom pattern). (a) as is; (b) 10 % per CLASS
(a molecule in several classes is drawn once, credited to each), so every class has a
held-out set of ~50 molecules for the judge; (c) 10 % per stratum, and the judge reports
per class whatever landed there.

➡️ (b). The judge's question is per class; a class with 500 molecules must have its own
held-out ~50 or the answer for that class is noise.

---

❓ **Q5** - **"Not trained in SPICE".** `in_training` in the membership table is true at
ANY match level (isomeric, stereo-removed, InChIKey connectivity). (a) exclude
`in_training = true` (119,275 remain); (b) exclude only isomeric matches (looser; a few
hundred more molecules); (c) also exclude `in_test_only`.

➡️ (a). "Never seen at any level" is what held-out means (round-2 ruling); the pool
loses 175 molecules, nothing.

---

❓ **Q6** - **What runs where, and the debug gate.** Every stage is a Slurm array in the
xargs shape (no parsl in a job): `hl_branchA.slurm` (one line = one molecule,
`s0_A_pipeline.py`, 16 x 4 threads per node), `hl_frames.slurm` (02_frames per
molecule, MACE on CPU, 16 per node), `hl_labels.slurm` (as now), then 01/04 on one node.
Gate: each script once on `debug` with `LIMIT=16` (16 molecules / 16 frames = one round,
< 10 min) before its array. (a) this; (b) one `hl_pipeline.slurm` array that does A -> 02
-> 03 per molecule line (a molecule's labels start as soon as its frames exist; fewer
submissions, but a node mixes CREST, MACE and ORCA and the layout arithmetic gets murky).

➡️ (a). Stages with one layout each are what the hkuhpc scripts do and what the smoke
set proved; the cost of (a) is three submissions instead of one.

---

The ALF/parsl mode (driver in `tmux`, elastic blocks) stays available for every stage as
the alternative to the array, unchanged by these tickets.

---

## Rulings (user, 2026-09-19)

- **Q1 (a)+(c). Q2 (a). Q5 (a)** -- and the screening is to be explained (below).
- **Q3: PRODUCTION** -- every basin and every frame of the union, integrated into MACE
  training xyz. No thinning of the recipe.
- **Q4: production split, BY FRAME** (train / valid / test drawn per frame, as SPICE).
  Assumption until you say otherwise: the pinned seven stay a whole-molecule test set
  (they are the msRRHO judge's molecules, ruled earlier); everything else splits by frame
  90 / 5 / 5. Consequence stated once: a by-frame test measures interpolation within the
  drawn molecules, not generalisation to unseen ones -- the per-class judge numbers are
  interpolation numbers, and the seven are the only out-of-molecule test.
- **Q6 (a)** -- one array per stage, `LIMIT=16` debug gate each -- with three things
  measured first (below): the filesystem footprint, the time on 12 x 64 cores, and how the
  result becomes one xyz for MACE Hessian learning.

## How the SPICE screening works (Q5)

Source: MACE-OFF23's released training data (Moore et al., Apollo 10.17863/CAM.107498):
`train_large_neut_no_bad_clean.xyz` (951,005 frames, 17,132 molecules) and
`test_large_neut_all.xyz` (50,195 frames). Every frame header carries an atom-mapped,
explicit-hydrogen `smiles="..."` and a `config_type`. `openqha/data/training_set.py`
streams the headers once, clears the atom maps, removes explicit H with RDKit, and
files each distinct molecule under three keys: canonical isomeric SMILES (with stereo),
canonical SMILES with stereo removed, and the InChIKey connectivity block (tautomer- and
charge-insensitive); a dimer frame (`DES370K Dimers`, two fragments) files each fragment
separately and its frames count as dimer frames. That index is
`data/training_sets/mace-off23_spice_index.dat` (in git, ~3 MB). The QM9 side: each of the
119,450 gated targets' relaxed SMILES (the SMILES line of its curated QM9 xyz) gets the
same three keys; `membership()` looks each up, the strictest level that hits is
`match_level`; `in_training = true` when that identity has ANY monomer or dimer frame in
the TRAIN file at ANY level; `in_test_only` when only the test file has it. The result
over all targets is `data/training_sets/qm9_targets_membership.dat` (one row per
molecule); the draw takes `in_training = false`: 119,275 molecules. The 175 excluded are
almost all 1-4 heavy atoms (water, methane, acetone, propanal ...): SPICE's QM9-size
content is the DES370K monomer set, which is small.

Heavy-atom histogram of the 119,275: 9 heavy atoms 99,075 (83 %), 8: 16,587, 7: 2,943,
6: 556, <= 5: 114. **The draw will be 9-heavy-atom molecules (~18-20 atoms) -- twice the
atoms of the smoke set, which is why the cost below is not the smoke set's 5 min per
frame.**

## Footprint and time (Q6), measured where possible

Per frame, measured on the smoke set (10 atoms, wB97M-D3BJ/def2-TZVPPD, 4 ranks):
ORCA files kept = 231 KB (`job.out` 99, `job.hess` 45, `job.property.txt` 85, inp/engrad
2); the scratch-only `job.gbw` + `job.densities` = 2.5 MB are not copied back on tianhe;
the extxyz row per frame per level = 19 KB (the flattened 30 x 30 Hessian). For a 19-atom
molecule (3N = 57, 3,249 Hessian elements): `.hess` ~160 KB, `property.txt` ~300 KB,
`.out` ~150-250 KB, extxyz ~70 KB per level -> **~0.9 MB per frame kept, ~0.5 MB with
`property.txt` dropped from KEEP** (it duplicates the Hessian; proposed in ticket 06).

Frames per molecule: the smoke set had 1-2 basins (9.3 frames per molecule). A 9-heavy
QM9 molecule with chains gives CREST 1-10 conformers; a working guess is 3 basins -> 1 +
4 displaced per basin + ~2 merged = ~17 frames. The EXACT count is known after
`hl_branchA` + `hl_frames` (cheap, hours), before any label is bought.

Union size: 21 classes x 500 with heavy overlap (heterocyclic and aromatic alone cover
most of QM9): 5,000-8,000 molecules; ticket 05's Record states it.

Time per label for 19 atoms: not measured. Basis functions ~2.2x the smoke set's (def2-
TZVPPD: ~527 vs ~256); the analytic Hessian (CP-KS) scales ~N_bf^3-4 -> 8-16x -> **40-80
min per frame at 4 ranks**. The `LIMIT=16` gate on 16 draw molecules measures this.

Putting it together (N_mol = 6,000, 17 frames, 1 h per frame at 4 ranks):

    branch A    6,000 x ~10 min / 192 slots           ~5 h of 12 nodes
    02 frames   6,000 x ~1 min  / 192                 < 1 h
    03 labels   100,000 frames x 4 cores x 1 h        400,000 core-hours = **~22 days of 12 nodes x 64**
    storage     100,000 x 0.9 MB (0.5 MB trimmed)     ~90 GB (~50 GB) on XYFS02 (1 TB, no backup)
                + branch A ~1 MB per molecule         ~6 GB
                + Dataset files (2 levels + merged)   ~15 GB text, ~2.6 GB HDF5

Ranks per frame do not change the core-hours (8 ranks halves the wall per frame, halves
the frames per node). What changes it: the frame count (Q3 ruled: all), the level (an HF
or a smaller-basis label is 5-10x cheaper -- not the reference), or the molecule count.
So the plan that respects Q3: run branch A and 02 for the whole union first (a day), read
the exact frame count and the gate's per-frame time, and then submit `hl_labels` as
`--array=0-11` resubmitted week by week (7-day walltime, finished frames skipped) --
ticket 08 carries the measured column.

## One xyz for MACE Hessian learning (Q6)

`04_dataset` already writes `{train,valid,test}.<level>.extxyz` with `energy` / `forces`
(ASE calculator) and `hessian` (info, 3N x 3N flattened, eV/A^2). For MACE-torch the
loader's default keys are `REF_energy` (info) and `REF_forces` (arrays); with
`energy_key=energy` MACE warns and rewrites them itself. Ticket 07 therefore writes, in
addition, `REF_energy`, `REF_forces`, `REF_hessian` (flattened) and a `split` key, and
one merged file `mace_<name>.<level>.extxyz` (all splits, `split` per frame; MACE's
`--train_file` takes it, our 05_train reads `REF_hessian`), beside the per-split files and
the OpenREACT HDF5 (`molecules-<name>.h5`, float64: 100,000 frames x 3,249 x 8 B = 2.6 GB,
the compact form). Text size: ~70 KB per 19-atom frame -> ~7 GB for 100,000 frames.

---

## Q7 / Q8 (asked 2026-09-19 after the cost table; ruled the same day)

❓ **Q7 -- a reference Hessian at every frame?** Literature: Rodriguez (OpenREACT) and
PHL train Hessians at stationary points only (NMS/IRC frames are tests); PFT at minima
only; HIP at NEB path frames (TS search). Our target is the curvature at the minima.
Per frame at 19 atoms: analytic Hessian 40-80 min, energy + gradient 3-5 min.
(a) E-F-H everywhere (100k Hessians, ~22 days of 12 nodes); (b) E-F-H at basin / merged /
saddle frames (the stationary conformers, ~5 per molecule), E-F only at the displaced
frames (30k Hessians + 70k gradients, ~8 days); (c) (b) + one Hessian at one displaced
frame per basin as a curvature-away-from-minimum probe (~11 days).

➡️ **Ruled (b).** `keyword_line(level, hessian=False)` = `single_point + " EnGrad"` for
the displaced generator; the Dataset writes `REF_hessian` only where it exists and the
loss masks it (as PHL's does); the smoke set keeps its 65 Hessians. (c) stays available as
a judge-only measurement later.

❓ **Q8 -- `.gbw` / `.loc` retention.** `.gbw` ~5 MB at 19 atoms: every frame 500 GB,
Hessian-labelled frames only ~150 GB; `.loc` exists only with a `%loc` block.

➡️ **Ruled: both kept for the basin frames only** (basin / merged / saddle = the
Hessian-labelled jobs). So: `KEEP` gains `.gbw` and `.loc` for those jobs, drops
`property.txt` everywhere; the Hessian jobs get a `%loc` block -- the repository's
existing convention (`orca.LEVELS` dlpno: `LocMet AHFB`, occupied orbitals) unless you
name another localisation; displaced (gradient-only) jobs keep `.inp .out .engrad`.

### What the "curvature-away-from-minimum probe" of (c) is (asked, for the record)

An MLIP trained on energies and forces is fitted to E and to the FIRST derivative at
the sampled points. The Hessian is the derivative of the force field between those
points, and nothing in an E/F loss penalises how the model's forces bend between
samples: two models can agree on E and F at every training geometry and still differ in
curvature, because curvature is the small-scale roughness of the fitted surface. That is
the PFT finding (Phonon Fine-Tuning, arXiv 2025): fine-tuning on E/F of displaced /
MD structures IMPROVED forces and DEGRADED phonons (the Hessian at the minimum) -- the
fit bought force accuracy with curvature roughness -- and the cure was to add a
second-derivative term (finite-displacement phonons) to the loss. PEFT/Rodriguez report
the same mechanism from the other side: Hessian labels at the minima fix the curvature
there.

Under (b) our training data constrain the curvature AT the basins (H labels) and the
forces AROUND them (E-F at the displaced frames), but no label says what the curvature
is at a displaced geometry. Taylor: H(x0 + d) = H(x0) + (d^3E) . d + ...; the model's
H at x0 + d is whatever its fit does between the basin Hessian and the displaced forces.
The probe of (c) is one displaced frame per basin with a REFERENCE Hessian, used by the
judge (not the loss): the projected low-mode eigenvalues and the Frobenius error of
H_model(x0 + d) against H_ref(x0 + d), compared with the same error at the basin. If
the off-minimum error is much larger than the basin error, E/F training on the displaced
frames has roughened the curvature between the samples (the PFT symptom) and the answer
is (c) in the loss, or fewer / smaller displacements; if the two errors are alike, (b)
is enough. It also measures the third derivative -- whether the model's anharmonicity
H(x0 + d) - H(x0) matches the reference's -- which is what the classical 298 K frames
were drawn for. Cost of the probe: one Hessian per basin, ~18k for the draw (+3 days),
or a sample of basins for the judge alone.

**Final rulings (2026-09-19): Q7 (b) without the probe; Q8: NO `.gbw`, NO `.loc`, no `%loc` -- `KEEP` = `.inp .out .hess .engrad` (Hessian jobs), `.inp .out .engrad` (gradient jobs), `property.txt` dropped. Tickets 05-08 approved; implementation starts with 05.**
