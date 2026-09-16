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
What this repository writes about one Calculation: a Report and a Property file, in the
molecule directory. Nothing is written about a Batch.
_Avoid_: metadata, artefact, sidecar, summary

**Report**:
The `.out` text a Calculation leaves for a human, in the form CREST's `crest.out` and
ORCA's `.out` take; its last line says the Calculation terminated normally.
_Avoid_: log (that is an engine's or Slurm's stdout), summary

**Property file**:
The `.toml` beside a Report holding what a later step reads back, in the form ORCA's
`.property.txt` takes: a status block, the inputs, and the result blocks; nothing else.
_Avoid_: metadata file, meta, record file, settings file

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

**Enantiomer degeneracy**:
The factor `g'` of a basin in the Gibbs-Shannon sum: 1, or the number of RMSD-distinct
core structures CREST finds inside the basin's rotamer group (2 for a geometric
enantiomer pair such as gauche-propanal). Obtained by porting CREST's `intraconfRMSD`
onto `crest_rotamers.xyz` grouped by `cre_members`; never from a rotamer count (methyl
wells are one harmonic mode in one well).
_Avoid_: degeneracy (alone -- that is the electronic `g0`), rotamer number, multiplicity
