# Hessian learning: grilling, round 3 -- the frame set (2026-09-18, open)

Scope: the first step of `openQHA/workflows/`: QM9 molecules -> basins -> frames -> one
extxyz per split (train / valid / test), first at GFN2-xTB, then (the open question)
MACE refinement, before any wB97M label is spent. Rounds 1-2 (`grilling-round-1.md`,
`grilling-round-2.md`) are still open on Q1-Q13; this round only asks what those did not.

## Facts checked (not asked of the user)

- **How SPICE makes frames** (`openmm/spice-dataset/pubchem/createPubchem.py`, read
  2026-09-18; the local copy `hessian-train/SPICE/*.xyz` is the Apollo extxyz export with
  `energy`, `forces`, `MACE_forces`, `smiles`, `config_type`): per PubChem molecule
  10 RDKit conformers (`generate_conformers(n_conformers=10, rms_cutoff=0)`) -> each is
  minimised, then 100 ps Langevin MD at **500 K**, OpenFF 2.0.0 unconstrained, 1 fs, a
  state saved every 10 ps (100 candidates) -> states with E > 1e4 kJ/mol dropped -> **25
  "high-energy" frames by greedy max-min all-atom RMSD** (`filterByRMSD`) -> each of the
  25 is cooled: 5 minimisation iterations + 1 ps at 100 K -> **25 "low-energy" frames**.
  50 per molecule, no dedup of the 10 seeds against each other, no basin notion, no
  Hessian, no split column (the split is by frame, 95/5, in the Apollo release). DES370K
  monomers: ~49 geometries taken from DES370K itself. Test file: 34,093 PubChem, 14,797
  DES370K, 1,025 dipeptide, 144 QMugs, 84 water, 52 solvated frames.
- **What we have**: branch A gives per molecule a basin list (MACE-relaxed, fmax 1e-4,
  dedup 0.30 A, analytic Hessian, sigma, weights) with the CREST GFN2 ensemble as engine
  files; `hessian.thermal_displacements(atoms, calc, T, n_samples, seed, <= 0.15 A RMS)`
  samples the harmonic quantum distribution along the modes; branch B has `md_ase` /
  `md_openmm` MACE trajectories at 298.15 K for the example molecules only; `xtb.py` has
  `optimise_and_hessian` and `hessian_at`, **no MD wrapper**; CREST's own metadynamics
  frames are not kept. Curated QM9: 133,660 `.xyz`; gated targets 119,451, 176 in SPICE.
- **Labels**: wB97M analytic Hessian 3-5 min per 10-atom basin at 8 ranks (DefGrid2;
  DefGrid3 8.7 min, round-2 Q12); the reference Hessian at a *displaced* frame carries the
  gradient term (yesterday) and that is what a Hessian label at a non-stationary point is
  -- correct, but its rigid block is not zero (project with P on both sides in the loss).
- **How OpenREACT (Rodriguez, Smith, Mendoza-Cortes 2025) makes frames** -- the three
  `molecules-{RTP,IRC,NMS}.h5` read 2026-09-18 (`hessian-train/`, wb97xd/6-31g(d),
  Gaussian 16, doi 10.1038/s41597-020-0460-4 as source of the reactions): one group per
  reaction (`rxnNNNNNN`, attrs `crg`, `mult`, `nat`), datasets `coordinates` [n, N, 3],
  `energies` [n] (Hartree), `forces` [n, N, 3], `hessian` [n, 3N, 3N] (Eh/bohr^2:
  checked -- the IRC point with max|F| = 3e-5 gives -1106 / 212 / 286 cm^-1 under that
  unit, a transition state, and nonsense under the other two), `species`. **RTP** = 11,954
  reactions x (reactant, TS, product) = 35,087 stationary points -- **the only training
  set**; **IRC** = 574 reactions x ~60 path points (34,248), test only; **NMS** = 524
  reactions x ~120 frames (62,527), test only: each frame is one IRC point displaced along
  its normal modes with random amplitudes (ANI-style), ~2 per IRC point, **RMS
  displacement 0.04-0.83 A (mean 0.27), energy above the parent mean 0.062 Eh = 39
  kcal/mol, max 254 kcal/mol** -- far hotter than SPICE's 500 K MD and than our
  `thermal_displacements` (<= 0.15 A). So OpenREACT's design is the opposite of SPICE's:
  train on stationary points *with* Hessians, test on paths and hot frames; SPICE trains
  on hot frames *without* Hessians and tests by frame. Rodriguez's headline (plan_C §8.1:
  E-F-H Hessian error 37.8 vs 128 on NMS) is an extrapolation number from minima + TS to
  39-kcal/mol frames. Neither has a basin notion or a by-molecule split.
- **Vocabulary**: CONTEXT.md has Basin, Calculation, Batch, Molecule directory, Level;
  no Frame, Dataset, Split, Workflow. `workflows/` does not exist; the ordered drivers
  live in `scripts/production/s0_*`.

## Round 3 questions

❓ **Q1** - **What a Frame is, and where it lives**: SPICE has no basin notion; we do.
Proposed nouns for CONTEXT.md: **Frame** = one geometry of one molecule, born from one
Basin by one named generator (`basin`, `displaced`, `md500`, `cooled`), carrying labels
at the levels that have been run on it; **Frame set** = the Frames of one molecule, an
engine-independent extxyz under `<molecule dir>/frames/<generator>.extxyz` with `basin`,
`generator`, `seed`, `level` keys; **Dataset** = a split of Frame sets over many
molecules, written under `<root>/<tag>/_datasets/<name>/{train,valid,test}.extxyz` +
`index.dat` (molecule, basin, generator, split, levels present). (a) as proposed; (b)
frames stay inside the engine folder that made them (`xtb/md/`), Dataset only aggregates;
(c) no Frame noun, "conformation" as SPICE says.

➡️ (a). A frame is not an engine file (three engines may label it) and not a Basin
(most are not minima); it needs its own noun and its own folder. "Conformation" collides
with CREST's "conformer".

---

❓ **Q2** - **Generators per basin** (SPICE: MD 500 K + RMSD max-min + cooling, 50 per
molecule; plan_C §8.2/8.4: PFT and PEFT used rattle / relaxation-trajectory frames only,
Rodriguez stationary points + normal-mode displacements, none used MD): (a) mirror
SPICE: from every basin, GFN2-xTB Langevin MD 500 K 100 ps, 10 candidates, max-min RMSD,
25 hot + 25 cooled; (b) harmonic-quantum `thermal_displacements` only (n per basin,
<= 0.15 A); (c) both, with the MD count small: per basin n_disp = 4 displaced + 4 hot
(max-min RMSD out of 100 ps at 500 K) + their 4 cooled frames + the basin itself = 13
frames per basin; (d) (c) plus the along-mode line points where `mode_curvature` ran.

➡️ (c). The displaced frames are where the entropy error lives (curvature at the
minimum, all three published Hessian-learning papers); the hot/cooled frames are what
SPICE trained on and what keeps the fine-tune from forgetting off-minimum forces (Q7 of
round 2). 13 x ~2 basins x 200 molecules ~ 5,000 frames ~ 250 h of wB97M Hessians at 8
ranks -- an hkuhpc bundle, not a laptop; the smoke set (7 molecules) is ~40 frames, a day
locally. (d) adds nothing new (those points are already labelled at three levels; they
join by construction).

---

❓ **Q3** - **The MD engine for the hot frames**: (a) xtb GFN2 `--md` (the "first
GFN2-xTB" of the request; the CREST env has it; SHAKE off, 1 fs, 500 K, no MACE needed);
(b) OpenFF 2.0.0 via OpenMM exactly as SPICE (needs a force field for every QM9 species:
OpenFF parametrises most CHON but fails on some rings/charges); (c) MACE-OFF23 MD (branch
B's `md_ase`, real potential, 74 ms/step on the A800, but the frames are then MACE's own
distribution -- the thing we are correcting).

➡️ (a). It is the cheapest surface that respects bonds, needs no parametrisation, and
sampling on a surface that is *not* the model under test is a feature (SPICE's OpenFF is
the same idea).

---

❓ **Q4** - **What "MACE refinement" of the GFN2 frames means**: (a) relax every frame on
MACE-OFF23 -- no: that collapses hot frames to minima and destroys the generator's
meaning; (b) MACE single points on every frame to rank / filter: drop frames with MACE
force > F_max (unphysical), keep the committee-variance ranking as a `mace_var` key for
later active-learning draws, never move atoms; (c) MACE-relaxed **basins** only (already
branch A) and no MACE on frames at all; (d) MACE as the cooling step instead of GFN2
(the 25 "low-energy" frames of SPICE become MACE-minimised toward the basin).

➡️ (b). Refinement = a label and a filter, not a geometry change. (d) is tempting but
puts MACE's curvature into the frame *positions* -- the cooled frames would sit in MACE's
wells, which are exactly the wrong wells on the rings.

---

❓ **Q5** - **Seeds: basins only, or CREST conformers too?** SPICE seeds from 10 RDKit
conformers without dedup. (a) basins only (branch A's list, 1-3 for these molecules);
(b) basins + every CREST conformer within 3 kcal/mol that branch A merged away (they are
the saddle-side geometries where topology disagrees, round-1 Q13); (c) basins + the
merged conformers as `merged` generator frames, no MD from them.

➡️ (c). Merged conformers are free (already computed), informative (MACE and the
reference disagreed there), and one Hessian each; MD from them would double the cost for
frames that mostly fall back into the same basins.

---

❓ **Q6** - **The split**: SPICE splits by frame (95/5), i.e. interpolation. (a) train /
valid / test all by **molecule** (whole molecules held out; the 3 rings + propanal +
acetone/acetamide/N-methylformamide pinned to test), stratified by ring count and
heteroatom pattern, 80/10/10; (b) valid by frame inside the train molecules (early
stopping sees interpolation, as SPICE), test by molecule; (c) by frame throughout.

➡️ (b). Early stopping on an out-of-distribution valid set stops the wrong thing;
the test must be molecules the model never saw (round-2 rulings: held-out =
`IN_TRAINING = false` molecules, and the split is a column in `index.dat` set at write
time, round-1 Q4).

---

❓ **Q7** - **Which molecules, and how many, for the first Dataset**: (a) the 7 known
(smoke set) only, first; (b) smoke set + 200 drawn from the 119,451 gated targets
outside SPICE (stratified as in Q6a), branch A run for all of them first (CREST + MACE:
~10 min per molecule locally, 33 h; or the tianhe Batch); (c) 1,000.

➡️ (a) then (b) as the same workflow with `--limit`; the smoke set proves every step
end to end (frames, labels, split, loader, one epoch) before 200 molecules of branch A
are spent. (c) is the second draw after acceptance 5's data-efficiency scan.

---

❓ **Q8** - **What a frame file carries at each level**: (a) one extxyz per generator per
molecule at the GFN2 level (`energy`, `forces`, `hessian` flattened 3N x 3N as an info
key, `level="gfn2"`), the wB97M labels added later as a second file `<generator>.wb97m.extxyz`
with identical geometries (positions must match to 1e-8 A, as `hessian_at_<level>.meta`
does now); (b) one file per molecule with all levels as prefixed keys
(`gfn2_energy`, `wb97m_energy`, ...); (c) plan_C's OpenREACT HDF5.

➡️ (a). MACE's loader reads one `energy_key` / `forces_key` per file, so one level per
file is what training consumes; two files with a geometry identity check is the pairing
acceptance of plan_C 5.4 (2). HDF5 becomes an export if ever needed.

---

❓ **Q9** - **Frame filters** (SPICE: E < 1e4 kJ/mol only): (a) SPICE's energy cut plus
a bond-topology check (RDKit connectivity of the frame equals the basin's; a frame that
reacted at 500 K is dropped and counted) plus a max GFN2 force (e.g. 10 eV/A); (b)
energy cut only; (c) none.

➡️ (a). QM9 has strained rings; 500 K GFN2 will open some of them and a reacted frame
is a different molecule with the wrong `smiles` key.

---

❓ **Q10** - **`workflows/` as a thing**: (a) `openQHA/workflows/<name>/` = one ordered,
numbered set of drivers (`01_select.py`, `02_frames.py`, `03_labels.py`, `04_dataset.py`,
`05_train.py`, `06_judge.py`) plus a `README.md` and a `run.sh`, each step calling
`openqha/` code and writing Records into molecule directories; the first is
`workflows/hessian_learning/`; CONTEXT gains **Workflow** = an ordered set of Batches
that turns a molecule list into one deliverable; (b) keep everything under
`scripts/production/` with an `s0_H_` prefix; (c) a Parsl / Snakemake DAG.

➡️ (a). The `s0_` scripts are per-Calculation drivers and stay; a workflow is the layer
above them, and a reader should find the six steps in one folder in the order they run.

---

❓ **Q11** - **Random seeds and reproducibility of the frame set**: (a) every generator
seeded from `(qm9_index, basin, generator)` hashed, seed written into each frame; the
Dataset `index.dat` records generator versions (xtb, RDKit) so a rebuild is
byte-comparable; (b) seeds only.

➡️ (a); it is the same rule as `thermal_displacements(seed=)` already follows.

❓ **Q12** - **Displacement amplitude, and whether hot NMS frames are trained on or only
tested on**: OpenREACT trains on stationary points only and tests on NMS frames 39
kcal/mol above their parents; SPICE trains on 500 K frames; plan_C §8.2 (PFT) says E/F
alone on displaced frames hurts curvature. (a) train on `displaced` at 298 K (harmonic
quantum, <= 0.15 A, E-F-H) + `md500`/`cooled` (E-F-H), and keep a separate **`nms_hot`
generator (ANI-style random-amplitude normal-mode sampling, T_eff ~ 1000 K, E-F-H) as a
test-only Frame set** -- the off-equilibrium acceptance 7b; (b) train on `nms_hot` too;
(c) no hot frames at all.

➡️ (a). It reproduces both published designs' *test* conditions (Rodriguez NMS,
acceptance 7b) without importing OpenREACT's training regime, which was chosen for
reactions, not conformers; whether hot frames should enter training is then a measured
question (train with / without, compare on the held-out rings), not a ruling.

---

❓ **Q13** - **Transition states between conformers**: OpenREACT's RTP includes the TS of
every reaction (one third of the training set). Our basins are conformational minima;
the barriers between them (torsional TSs, ring inversions -- oxetane's planar saddle at
MP2) are where MACE and the reference disagreed most. (a) no TS frames (the deliverable
is G of minima; round-1 Q13 kept only merged/saddle-side minima); (b) add a `saddle`
generator: the CREST-merged conformers that re-optimise to a saddle at the reference
(branch A's `merge_map` status `saddle`) as frames, one Hessian each, no TS search; (c)
run a TS search between neighbouring basins.

➡️ (b). They are already found (merge map), cost one Hessian each, and are the
conformational analogue of RTP's TS column; (c) is a different project.

Round 4 waits on Q1, Q2, Q6, Q10 (they fix the nouns, the folder and the counts);
Q3-Q5, Q8, Q9 can be answered together; round-2 Q1-Q13 remain open.
