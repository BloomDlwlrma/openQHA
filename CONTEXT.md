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

**Level folder**:
The one folder in a molecule directory, `levels/`, that holds every level's results for
that molecule, one sub-folder per level named by the level. Engine files stay in engine
folders; the level folder holds the Records of the Calculations that turn a level's
Hessians into thermochemistry, and the merge map of that level's basins. A level that was
not computed for a molecule has no sub-folder: absence is stated, never a zero.
_Avoid_: reference folder, benchmark folder, per-level records

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
matrix at a fixed geometry, gradient term included; it is not a frequency.
_Avoid_: conformation (SPICE's word; collides with CREST's conformer), sample, structure

**Frame set**:
One molecule's Frames, one file per generator and per Level, engine-independent,
under the molecule directory's `frames/` folder; identical positions across the Levels
of one generator, checked (a reference label whose `.hess` geometry differs in shape by more than 1e-7 A
from the engine file is refused). A Frame set has a Record like any Calculation (what
was generated, what was dropped and why, the seeds, the engine identity) and one Record
per reference Level labelled (`labels.<level>`: what was labelled, reused or refused,
ORCA's wall time and memory per frame, the noise floor of every Hessian). A reference
label is a single point at the frame's fixed geometry -- energy, gradient and analytic
Hessian -- never an optimisation.
_Avoid_: trajectory, ensemble (that is a thermodynamic average), training file

**Dataset**:
A split of Frame sets over many molecules -- `train`, `valid`, `test`, and `pool` for
frames that exist but carry no reference label yet -- written under the root as one
file per split and one index naming every contributing Frame set, its molecule, basin,
generator, split and the Levels present. The split is set when the Dataset is written
and never recomputed: `valid` is drawn by frame from the training molecules, `test` is
whole molecules the model has never seen.
_Avoid_: training set (alone -- that is one split), benchmark, corpus

**Workflow**:
An ordered set of Batches that turns a molecule list into one deliverable, kept as
numbered drivers in one folder under `workflows/`, each step writing Records into the
molecule directories it touches. The per-Calculation drivers it calls stay where they
are.
_Avoid_: pipeline (that word is branch A's own driver), DAG, chain
