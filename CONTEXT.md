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
A file this repository writes about a run: settings, verdicts, timings, logs, summaries.
`meta.json`, `basins.json`, the collect parquet tables, `driver.log`, completion markers.
_Avoid_: metadata, artefact, sidecar

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
