# Hessian learning: grilling, round 4 -- the frame set, what is still open (2026-09-18)

Rulings received on round 3 (2026-09-18): seeds are the **MACE basins of branch A**
(CREST samples on GFN2, MACE enters as `refine = "sp"`, then MACE tightening to fmax 1e-4
and the MACE Hessian; no GFN2 minimum is ever a seed); the displaced frames carry the
entropy error (the three Hessian-learning papers train near minima), so SPICE's
`md500` / `cooled` generators and the `nms_hot` idea are **considered, not built** --
round-3 Q2, Q3, Q4 (as "MACE single points as a filter only"), Q9, Q12 close on that.
What follows is the frontier that those rulings did not touch, in the round format.

---

❓ **Q1** - **The nouns and where a frame lives**: three new CONTEXT.md entries --
**Frame** = one geometry of one molecule, born from one Basin by one named generator
(`basin`, `displaced`, `merged`, `saddle`), carrying labels at whatever levels have been
run on it; **Frame set** = one molecule's frames, engine-independent extxyz under
`<molecule dir>/frames/<generator>.<level>.extxyz` with `qm9_index`, `basin`,
`generator`, `seed`, `level` keys; **Dataset** = a split of Frame sets over many
molecules under `<root>/<tag>/_datasets/<name>/{train,valid,test}.extxyz` + `index.dat`
(molecule, basin, generator, split, levels present). (a) as proposed; (b) frames stay in
the engine folder that labelled them (`mace/basinNN/frames/`, `orca/<level>/...`) and
the Dataset only aggregates; (c) keep SPICE's word "conformation", no new nouns.

➡️ (a). A frame is not an engine file (three engines may label the same geometry) and
not a Basin (most are not minima); it needs its own noun and its own folder.
"Conformation" collides with CREST's "conformer".

---

❓ **Q2** - **Generators and counts per basin**: (a) `basin` (1) + `displaced` (4 per
basin, harmonic quantum at 298.15 K along the MACE modes, RMS <= 0.15 A, the
`thermal_displacements` defaults) + `merged` (every CREST conformer branch A merged
away, one frame each, no displacement) + `saddle` (every merge-map `saddle`, one frame
each); (b) the same with 2 displaced per basin (Rodriguez's NMS density); (c) 8.

➡️ (a). 4 is the smallest count that samples both signs of every soft mode with
some redundancy; Rodriguez's 2 was for a test set. At ~2 basins per molecule this is
~10-12 frames per molecule; 8 doubles the label bill for frames that lie in the same
0.15 A shell.

---

❓ **Q3** - **The split**: (a) train / valid / test all by **molecule**, 80/10/10,
stratified by ring count and heteroatom pattern, the 7 known molecules pinned to test;
(b) **valid by frame inside the train molecules** (early stopping sees interpolation, as
SPICE), **test by whole molecules** (generalisation, `IN_TRAINING = false` by
construction); (c) by frame throughout, as SPICE.

➡️ (b). Early stopping on an out-of-distribution valid set stops the wrong thing; the
test must be molecules the model never saw (round-2 held-out ruling). The split is a
column of `index.dat` set at write time (round-1 Q4), never recomputed.

---

❓ **Q4** - **Which molecules, in what order**: (a) the 7 known molecules as the smoke set
(~80 frames, one local day of wB97M labels) through all six steps first, then a
stratified 200 drawn from the 119,451 gated targets outside SPICE with branch A run for
them (tianhe Batch), labels for all their frames (~2,500 frames, ~150 h at 8 ranks:
an hkuhpc bundle); (b) 200 directly; (c) frames for all 119,451 (the unlabelled *pool*,
~1 s per basin once branch A exists) and labels for the 200 -- later active-learning
draws pick from the pool.

➡️ (a) now, (c) as the standing shape of the workflow: `01_select` and `02_frames` run
over everything branch A has produced, `03_labels` over `--limit N` stratified; the
pool costs nothing beyond branch A, which is the one Batch that is expensive at 119 k.

---

❓ **Q5** - **What a frame file carries, per level**: (a) one extxyz per generator per
level: `<generator>.mace-off23_medium.extxyz` (E, F, H from the engine, written at
generation time) and `<generator>.wb97m-d3bj_def2-tzvppd.extxyz` (E, F, H labels),
identical positions checked to 1e-8 A at write time (as `hessian_at_<level>.meta.toml`
does), `hessian` as a flattened 3N x 3N per-frame info key in eV/A^2, raw Cartesian
(never projected); (b) one file per molecule with level-prefixed keys; (c) plan_C's
OpenREACT HDF5 (`coordinates/energies/forces/hessian` per group, Eh and Eh/bohr^2).

➡️ (a). MACE's loader consumes one `energy_key` / `forces_key` per file, so one level
per file is what training reads; the geometry-identity check is plan_C acceptance
5.4(2). The HDF5 form becomes `04_dataset --export openreact` if it is ever needed
for HIP / PHL code that reads it.

---

❓ **Q6** - **`workflows/` as a thing**: (a) `openQHA/workflows/hessian_learning/` =
numbered drivers `01_select.py`, `02_frames.py`, `03_labels.py`, `04_dataset.py`,
`05_train.py`, `06_judge.py` + `README.md` + `run.sh`, each calling `openqha/` code and
writing Records into molecule directories; CONTEXT gains **Workflow** = an ordered set
of Batches that turns a molecule list into one deliverable (here: a Dataset and a
judged potential); (b) `scripts/production/s0_H_*` beside the other drivers; (c) a
Parsl / Snakemake DAG.

➡️ (a). The `s0_*` scripts are per-Calculation drivers and stay; a workflow is the
layer above them, and a reader should find the six steps in one folder in run order.

---

❓ **Q7** - **Seeds and reproducibility**: (a) every generator seeded from
hash(`qm9_index`, basin, generator, k), the seed written into each frame, and
`index.dat` recording the engine versions (MACE weights SHA, ORCA version) so a
rebuild is byte-comparable; (b) a global seed only.

➡️ (a); it is the rule `thermal_displacements(seed=)` already follows.

---

❓ **Q8** - **MACE on the frames**: (a) the MACE E-F-H at every frame is written at
generation time (it is free: the Hessian is one `get_hessian` per frame) and doubles as
the filter -- a frame whose MACE |F|max exceeds 10 eV/A or whose bond graph (RDKit,
from the basin's SMILES) changed is dropped and counted in the Frame set's Record;
(b) MACE E-F only, Hessian later; (c) nothing at MACE until the judge.

➡️ (a). The MACE Hessian at the frame is the very quantity the loss compares to the
label; having it on disk lets `06_judge` and `hessian_compare` run without an engine,
and the filter costs nothing extra.

Round 5 (spec) waits on Q1, Q3, Q4, Q6; Q2, Q5, Q7, Q8 can be answered together.
Round-2 Q1-Q13 (label level, loss, weighting, forgetting, judge thresholds, CCSD(T)
route, DefGrid3) remain open and are needed before `03_labels` and `05_train` are
written; `01_select`, `02_frames`, `04_dataset` need only this round.
