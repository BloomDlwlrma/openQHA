# openQHA

Conformational free energy of small organic molecules: CREST finds the basins, MACE
relaxes them and gives their Hessians, OpenMM samples each basin, and a quasi-harmonic
analysis turns the samples into an entropy. This glossary fixes the words used for what
those steps leave on disk.

## Language

**Engine**:
One of the programs that does the physics: CREST, MACE (through ASE), OpenMM, xtb, ORCA.
_Avoid_: backend, calculator (for the program as a whole), tool

**Engine file**:
A file an engine writes itself, in its own format, or the input that engine was given and
the stdout it printed. CREST's `crest_conformers.xyz`, ORCA's `.gbw`, an ASE `opt.traj`,
an OpenMM `.dcd`.
_Avoid_: raw output, native output, product (that word is reserved for the answer)

**Record**:
What this repository writes about one Calculation: a Report, a Property file and, when the
Calculation produces rows, a Table, in the molecule directory. Nothing is written about a
Batch.
_Avoid_: metadata, artefact, sidecar, summary

**Report**:
The `.out` text a Calculation leaves for a human, in the form CREST's `crest.out` and
ORCA's `.out` take; its last line says the Calculation terminated normally.
_Avoid_: log (that is an engine's or Slurm's stdout), summary

**Property file**:
The `.toml` beside a Report holding what a later step reads back, in the form ORCA's
`.property.txt` takes: a status block, the inputs, and the result blocks; nothing else.
_Avoid_: metadata file, meta, record file, settings file

**Table**:
The one whitespace `.dat` beside a Report holding a Calculation's per-row numbers, in the
form CREST's `crest.energies` takes, with every section named and every column explained
in a comment above its header. Collect is the one Calculation that leaves one.
_Avoid_: csv, parquet, dataframe, sidecar, the four tables

**Calculation**:
One step applied to one molecule or one basin: a CREST run, a MACE relax and Hessian,
one MD trajectory, one collect, one ensemble. The unit that owns a Record.
_Avoid_: task, job (that is Slurm's word), run

**Batch**:
One driver invocation that runs many Calculations inside one Slurm job or array row.
Its only trace is the Slurm log, which lists every Calculation's return code and Record.
_Avoid_: campaign, chain, pipeline (for the invocation), summary

**Molecule directory**:
The one directory that holds everything for one molecule under one tag, with one folder
per engine inside it. Engine folders contain engine files only.
_Avoid_: run directory, output directory, work directory (CREST's own term is kept for
the folder CREST runs in)

**Engine folder**:
The folder inside a molecule directory that one engine wrote into: `crest/`, `mace/`
(the potential's relax and Hessian), `md_openmm/` and `md_ase/` (the two implementations
of the sampling step), `xtb/`, `orca/`. The MD folders are named by role because "mace"
and "ase" side by side read as one thing twice.
_Avoid_: stage folder, branch folder, route folder

**Basin**:
One local minimum of a molecule that survived branch A's deduplication and Hessian
screen; numbered from 00 in the order branch A lists them.
_Avoid_: conformer (that is what CREST reports before the screen), minimum, well

**Tag**:
The label of one campaign: the same molecule under two tags is two molecule directories.
_Avoid_: run name, experiment, label

**Root**:
The directory the molecule directories are written under, on the shared filesystem.
_Avoid_: runs root, scratch, keep dir

**Level**:
The model chemistry that produced a number: `gfn2`, `wb97m-d3bj_def2-tzvppd`,
`dlpno-ccsdt_cc-pvtz`, `mace-off23_medium`. Lower-case, method first, basis second,
joined by `_`, dispersion inside the method token. An engine may run several levels
(ORCA runs two); a level is run by one engine. The reference level for model error is
`wb97m-d3bj_def2-tzvppd`, the level MACE-OFF23 was trained to.
_Avoid_: level of theory (as a folder or key name), method, theory

**Thermo folder** (was "level folder" until 2026-09-20):
The one folder in a molecule directory, `msrrho/thermo/`, that holds every level's msRRHO
results for that molecule, FLAT: `<level>.<step>.{out,toml,dat}` per level
(`wb97m-d3bj_def2-tzvppd.thermo_msrrho.toml`, `mace-off23_medium.merge_map.dat`), the
cross-level Records bare (`level_compare.toml`, `hessian_compare.toml`). Engine files stay
in engine folders (`msrrho/orca.<level>.basinNN.*`); the thermo folder holds the Records of
the Calculations that turn a level's Hessians into thermochemistry, and the merge map of
that level's basins. A level that was not computed for a molecule has no file: absence is
stated, never a zero (`layout.levels_present`).
_Avoid_: level folder, reference folder, benchmark folder, per-level records

**Numerical reference level**:
A reference level whose geometry and Hessian ORCA can only produce from energies
(`Opt NumGrad` + `NumFreq`), because the method has no analytic gradient:
`dlpno-ccsdt_cc-pvtz` is one. Its records carry what an analytic level does not need:
the noise floor of the Hessian (the largest rigid-body eigenvalue of the unprojected
Hessian, in cm^-1: 5-30 for an analytic Hessian at a tight minimum, where the rotational
block feels the residual gradient, plus the finite-difference noise for a numerical one)
and the final numerical-gradient RMS; a low-mode difference below the larger floor of
the two Hessians compared is unresolved, not model error. Before its (6N)^2 single points are spent, the curvature along a chosen
reference mode is measured from a line of energies (`mode_curvature`), which is also
what the returned Hessian must reproduce.
_Avoid_: CCSD(T) Hessian (as if it were analytic), NumFreq level, high level

**Imaginary-mode policy**:
What a `thermo_msrrho` Calculation does with a negative projected frequency, named in
every record: `refuse` (the basin is excluded and listed), `invert_below` (a mode in
(ithr, 0) takes |omega|, a mode below ithr excludes the basin), `crest_native` (CREST
3.0.2 line for line: inverted in (ithr, 0), kept negative with zero entropy below ithr,
still in ZPE, H and Cp). Every record carries `[Imaginary_Spread]`, S_abs under all
three. Production uses `refuse`; the GFN2 seam uses `crest_native`.
_Avoid_: ithr handling, frequency cleaning, fixing imaginary modes

**Enantiomer degeneracy**:
The factor `g'` of a basin in the Gibbs-Shannon sum: 2 for a conformer whose rotamer
group holds a sampled mirror pair, or that is chiral while another chiral conformer
does (CREST's `enantiofac` rule), 1 otherwise; a basin whose mirror image is itself a
basin counts once. Obtained by porting CREST's `intraconfRMSD` onto `crest_rotamers.xyz`
grouped by `cre_members`, with the mirror test and the conformer's chirality as
Procrustes numbers (rotational vs orthogonal RMSD; self-mirror RMSD against the
threshold 0.75 RTHR, `meng2022procrustes`) rather than a point-group label; never from a
rotamer count (methyl wells are one harmonic mode in one well).
_Avoid_: degeneracy (alone -- that is the electronic `g0`), rotamer number, multiplicity

**Frame**:
One geometry of one molecule, born from one Basin by one named generator -- `basin`
(the minimum itself), `displaced` (a harmonic classical draw along the basin's modes at
the target temperature -- equipartition, the distribution branch B's 298 K MD samples;
the quantum draw is available by name -- within a stated RMS displacement), `merged`
(a CREST conformer branch A merged into a basin), `saddle` (a merged conformer that re-optimised to a
saddle at the reference) -- carrying, per Level that has been run on it, the energy,
forces and Cartesian Hessian at that geometry. A frame's Hessian is the raw Cartesian
matrix at a fixed geometry, gradient term included; it is not a frequency. "Stationary"
of a frame means stationary on the ENGINE's surface (the basin was optimised with MACE):
at the reference level the same geometry carries a gradient, which its Label keeps.
_Avoid_: conformation (SPICE's word; collides with CREST's conformer), sample, structure

**Normal-mode sampling** (`nms`):
How a Frame set's displaced Frames are drawn from a Basin's modes since 2026-09-23, and the
workflow's only draw: mode k is given the harmonic energy c_k (3/2) N_a k_B T from a random
partition (c_k >= 0, their sum s <= 1) and a random sign, so a frame's total harmonic energy
is (3/2) s N_a k_B T -- bounded by (3/2) N_a k_B T, mean (3/4) N_a k_B T -- and the
displacement is sqrt(2E_k)/omega_k per mode. Drawn at 450 K, where it puts the amplitude the
equipartition draw put at 298 K. It is the SCALE OF THE DIAGNOSTICS and nothing else: msRRHO
thermochemistry is computed at Basins and the training set is Basin frames only, so the draw
touches neither; the displaced Frames carry an EnGrad reference label and serve the judge's
held-out rows, the in-distribution and forgetting checks and the curvature change from the
Basin. One campaign therefore carries one draw (`FORCE=1` rebuilds every Frame set when it
changes; a Frame set's own Record says which draw and which temperature made it). It does not
bound the geometry -- the amplitude is sqrt(2E)/omega -- so a Basin with a near-zero mode
displaces by angstroms and loses its displaced Frames to the energy window, which is the only
filter. The equipartition and zero-point draws remain available to calibration and branch B,
not to this workflow.
_Avoid_: calling the 450 K a physical temperature of the frames (the draw is a bounded random
partition, not a Boltzmann ensemble); reading a msRRHO or training consequence into it.

**Engine Hessian of a Frame**:
Which Frames carry one, since 2026-09-23: a Basin frame reuses branch A's stored matrix,
`merged` and `saddle` get one from the engine, a `displaced` frame gets NONE (energy and
forces only; its `LOWEST_FREQ` is blank and its `has_hessian` is false). The engine Hessian
is 3N backward passes -- 13.4 s of the 13.6 s a 19-atom frame costs -- and ~22 of a
molecule's ~30 Frames are displaced, so this is 3.4x of the Frame-set step; nothing
downstream reads it, because a labelled Frame enters the Dataset as its reference Label, a
displaced Frame has no reference Hessian, and training predicts its own.
_Avoid_: confusing it with the Label Hessian (the reference one) or with the model's
prediction; assuming a `displaced` Frame can be compared matrix-to-matrix.

**Frame set**:
One molecule's Frames, one file per generator and per Level, engine-independent,
under the molecule directory's `frames/` folder; identical positions across the Levels
of one generator, checked (a reference label whose `.hess` geometry differs in shape by more than 1e-7 A
from the engine file is refused). A Frame set has a Record like any Calculation (what
was generated, what was dropped and why, the seeds, the engine identity) and one Record
per reference Level labelled (`labels.<level>`: what was labelled, reused, refused or
failed, ORCA's wall time and memory per frame, the noise floor of every Hessian). A frame's
reference job is attempted ONCE and bounded (`TIMEOUT_S`, 8 h): a job that did not terminate
normally leaves its `.out` and is a failed frame -- read, not rerun, until a human asks
(`--retry`); a job cut before anything came back (walltime, a dead node) left nothing and is
rerun whole -- no unit checkpoints or resumes (round 11, S0-G-96). While it runs the frame is
held by a `.running` lock that names its Slurm job and is touched every minute; another
process treats the frame as held only while Slurm does not call that job dead AND the lock
was touched within 30 min. A reference
label is a single point at the frame's fixed geometry -- energy, gradient and analytic
Hessian -- never an optimisation.
_Avoid_: trajectory, ensemble (that is a thermodynamic average), training file

**Dataset**:
A split of Frame sets over many molecules -- `train`, `valid`, `test`, and `pool` for
frames that exist but carry no reference label yet -- written under the root as one
file per split and one index naming every contributing Frame set, its molecule, basin,
generator, split and the Levels present. The split is set when the Dataset is written
and never recomputed: `train` and `valid` hold frames of the training generators only
(basin, since S0-C-54), `valid` drawn by frame from the training molecules; `test` is
whole molecules the model has never seen, the by-frame test draw, and every labelled
frame of a Held-out generator.
_Avoid_: training set (alone -- that is one split), benchmark, corpus

**Replay**:
The base model's own training frames that a fine-tune concatenates into the same training
set as the Dataset -- a fixed draw from SPICE's train split, one seed one file for a whole
campaign, energies and forces only, no Hessian -- so that the fine-tune keeps what the base
knew. Its size is reported as frames per Hessian frame (the same file is a different ratio
on every Dataset), its weight is each frame's `config_weight`; it never overlaps the
forgetting set's molecules nor the in_distribution molecules: the tool that draws it
skips their frames (SPICE's test split is by frame, so most molecules have frames on both
sides) and refuses to write when what it drew would still overlap.
_Avoid_: co-training (PFT's step loop, which mace does not do), pretraining head (the
implementation's `pt_head`), fine-tune set

**Held-out generator**:
A Frame generator whose frames enter the Dataset, may carry Labels and are read by the
Judge, but never enter `train` or `valid` -- today `displaced`, `merged` and `saddle`
(S0-C-54, ADR 0005: the fine-tune learns basin Hessians only; the msRRHO result is the
deliverable). Their labelled frames sit in `test` with `held_out_generator = yes` in the
index, and the Judge reports them as reference rows (binned by RMS displacement), never
as gates. Which generators train is the Dataset's `TRAIN_GENERATORS`, set at build time.
_Avoid_: test generator, extrapolation set (that is what the rows measure, not the frames)

**Held-out**, and the three distributions:
What a judged number is worth depends on what the model had already seen, so every judge
table has three rows and never one. **interpolation**: a test frame of a TRAINING
molecule -- held out by frame, so it measures interpolation within those molecules, not
generalisation to new ones; the production split holds out whole MOLECULES (S0-C-65), so
`interpolation` is EMPTY in production and only the smoke / fit Datasets (the by-frame
split) produce one. **out_of_molecule**:
whole molecules held out, which the fine-tune never saw in any frame. **in_distribution**:
molecules the BASE model was trained on (the four shipped ones, S0-C-40) -- there the
question is not accuracy but damage, and the line to watch is "no worse than the base".
A number quoted without its row is not a claim about anything.
_Avoid_: held-out (alone), test set (alone -- which test?), generalisation (for the by-frame split)

**Loss**:
The fine-tune's Hessian term, PHL verbatim (S0-C-64): the mean squared error per matrix
element of the Cartesian matrix, `||H_theta - H_r||_F^2 / (9 N^2)` (eq. 1'), sampled by
`K` random probes through Hessian-vector products (`sum_j ||H_theta v_j - H_r v_j||^2 /
(9 N^2 K)`, eq. 6'), the probes PHL's standard normal (S0-C-68) and the reference side a
matvec on the stored Label. Nothing is mass-weighted or projected in training; the mass
weighting and Eckart projection are evaluation only -- what a frequency is -- and live in
`hessian_compare`, never the loss. The full matrix is the deterministic limit (3N unit
probes = `get_hessian`), which is what the Judge reads exactly.
_Avoid_: projected Hessian loss (the 2026-09-18 design, superseded by S0-C-53/64),
entropy-weighted loss, mode-weighted loss

**Judge**:
The step that decides whether a potential is better, reading only shipped paths: the full
Cartesian Hessian from the engine's own `get_hessian` against the reference Label through
`hessian_compare`, never the training loss's estimator. It reports per structure class and
per distribution, sets anharmonic modes aside from the entropy tier, reads the
thermochemistry from the msRRHO Records rather than recomputing it, and ends in one line
per row. GATE rows decide the verdict (S0-C-58/59): the Hessian MATRIX itself against the
Label on the held-out Hessian frames -- the training target's own number, engine no worse
than base -- the in_distribution no-degradation, the forgetting line. Everything computed
FROM the matrix afterwards is post-processing and a REFERENCE row, measured against a
number and reported, never gated: the low-mode frequency line (the standard vibrational
analysis of the trained matrix -- mass weighting + Eckart projection, as `hessian_compare`
computes a frequency -- never the training loss), the msRRHO entropy at the
engine's own minima, the Held-out generator's frames binned by RMS displacement (H, E, F
against the base), the MD temperature ramp (run only when asked for). The gate is CLOSED
for now (S0-C-60): every row is reported against its number and the verdict reads
`REPORTED`; `--gate` reopens it. It is calibrated in both directions with the gate open:
the base model against itself reads exactly 0 on the Hessian gate and must not fail a gate
row, and a deliberately scaled potential must fail the Hessian gate and with it the
verdict.
_Avoid_: evaluation (alone), validation (that is inside training), benchmark

**Workflow**:
An ordered set of Batches that turns a molecule list into one deliverable, kept as
numbered drivers in one folder under `workflows/`, each step writing Records into the
molecule directories it touches. The per-Calculation drivers it calls stay where they
are.
_Avoid_: pipeline (that word is branch A's own driver), DAG, chain
