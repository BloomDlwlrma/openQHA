# Engine files only: one molecule directory, one folder per engine

Status: ready-for-agent
Date: 2026-09-14
Vocabulary: CONTEXT.md. Decisions: docs/adr/0001, docs/adr/0002.

## Problem Statement

A run of the chain leaves its results in three trees (a runs root, `analysis/`, and a
basin store that is a byte-identical second copy), and beside the few files the engines
produced it leaves about twenty-five files this repository wrote about the run. On the
cluster the whole run tree is then copied a second time at job exit. Reading a finished
molecule means knowing this repository's conventions instead of the engines'. What the
user wants is what ORCA leaves behind: the `.out`, the `.gbw`, the `.hess`, in one place,
and nothing else in that place.

## Solution

Every molecule under a tag gets ONE molecule directory on the shared filesystem, with one
engine folder per engine and engine files only inside those folders. CREST's working
directory is kept verbatim; MACE's relax and Hessian are kept in ASE's own forms; OpenMM's
trajectory is kept through OpenMM's own reporters. Everything this repository writes about
a run goes into a single `_records/` folder inside the same molecule directory, unchanged
in content for now (step 2 decides its future). The old trees are deleted, after a plan
that lists them, and the examples are re-run into the new layout.

    <root>/<tag>/<range>/<chunk>/<qid>/
      crest/                      CREST working directory, verbatim (input.toml, <qid>.xyz,
                                  crest.out, crest_conformers.xyz, crest_best.xyz, ...)
      crest_shake1/               only when the SHAKE=2 run terminated early
      mace/confNN/                opt.traj  opt.log  conf.extxyz     every tightened conformer
      mace/basinNN/               basin.extxyz  hessian.npy          every surviving basin
      openmm/<setting>/basinNN/   start.pdb  system.xml  integrator.xml  traj.dcd
                                  state.csv  state.xml  state.chk
      xtb/basinNN/  orca/basinNN/ 02c only, the engines' own files
      _records/                   everything this repository writes about the run

    root = <prefix>/HDD_POOL/<acct>/<user>/sherwin/runs
           prefix /XYFS02 for partitions ai and cn, /XYAIFS00 for a100x h100x hx a800x v100x;
           S0_RUNS_ROOT set explicitly still wins; off-cluster default unchanged.
    range = 16 000 molecules, chunk = 1 000 (1_16000/1_1000, 1_16000/1001_2000, ...).

## User Stories

1. As the user, I want one directory per molecule under a tag, so that everything about a molecule is found by one path.
2. As the user, I want an engine folder to contain only what that engine produced (its input, its stdout, its own output files), so that I can read it the way I read an ORCA directory.
3. As the user, I want CREST's working directory kept verbatim, so that CREST's own restart, ensemble and topology files are there.
4. As the user, I want CREST to run node-local and its finished directory moved once into the molecule directory, so that Lustre never sees CREST's thousands of small writes.
5. As the user, I want the first SHAKE attempt kept beside the fallback, so that an early termination is evidence and not lost.
6. As the user, I want every tightened conformer's relaxation kept as an ASE trajectory and log, so that a conformer that lost the deduplication can still be inspected.
7. As the user, I want each surviving basin's geometry as an extxyz whose header names the conformer it came from, so that the basin list is readable without a JSON.
8. As the user, I want each basin's raw analytic Hessian as a numpy array, so that projection and frequencies can be redone from the matrix.
9. As the user, I want the OpenMM trajectory as a DCD with a PDB topology, so that any trajectory tool reads it.
10. As the user, I want the state CSV row-aligned with the DCD frames, so that energy and temperature per frame are one join away.
11. As the user, I want the serialised system and integrator kept, so that the physics that ran is the physics on disk.
12. As the user, I want both a checkpoint and a portable state file, so that a run resumes fast on the same hardware and still resumes on other hardware.
13. As the user, I want a killed trajectory to resume from its state and append to its DCD, so that no frame is produced twice or lost.
14. As the user, I want one trajectory per basin with no seed level, so that the tree reflects the one-seed ruling.
15. As the user, I want several trajectory settings on one branch A product to sit side by side under `openmm/`, so that one search and nine samplings read as exactly that.
16. As the user, I want branch B to read the basins from the molecule directory, so that there is no second copy to keep in step.
17. As the user, I want the collect step and the 02d identity check to read the DCD, so that the analysis runs on the engine file and not on a private format.
18. As the user, I want the root chosen by the cluster's partition, so that the same conf runs on XY-A and XY-AI without editing.
19. As the user, I want the resolved root printed at job start, so that I can see where the tree is going.
20. As the user, I want the environment to refuse to run when the root cannot be derived, so that nothing lands in a home directory by accident.
21. As the user, I want records kept in one `_records/` folder inside the molecule directory, so that nothing is lost before step 2 decides their form.
22. As the user, I want no copy of the run tree at job exit, so that a trajectory exists once.
23. As the user, I want the MACE server sockets in a node-local directory, so that Lustre is never asked to hold a socket.
24. As the user, I want a script that lists the old trees it would delete, so that I can check the plan before anything is removed.
25. As the user, I want the same script to delete them only when I say so, so that a plan is never a deletion.
26. As the user, I want the shipped examples to run end to end into the new layout, so that the change is proven on the molecules the repository ships.
27. As the user, I want the seven test points to exist as tests, so that the tree shape is a checked fact rather than a convention.
28. As the user, I want each step to tell me what to commit, so that I can track the work on GitHub myself.

## Implementation Decisions

- **One layout module answers every path question.** A new pure module in the store package returns the molecule directory, each engine folder, the records folder and the shard directories from (root, tag, qid). Every writer and every reader asks it; no other code composes these paths. The basin store module and its shard rule (chunk 4 000) are retired; the new rule is range 16 000, chunk 1 000.
- **Root resolution lives in the Tianhe environment file.** Prefix by partition (`OPENQHA_PARTITION`, then `SLURM_JOB_PARTITION`; on a login node with neither, the mounted prefix); account and user from `HOME`; the tail `sherwin/runs` is one variable. An explicit `S0_RUNS_ROOT` wins. The resolved root is printed and exported; failure to resolve is an error, not a fallback to `HOME`. The scratch-based override of the runs root, the exit-trap copy to `logs/node_local/`, and the keep directory are removed.
- **Sockets are node-local.** The socket directory is `/tmp/<user>/<jobid>`; the socket module's existing fallback logic stays, the Tianhe file stops pointing it at the shared filesystem.
- **CREST runs node-local and is moved once.** The branch A pipeline gives CREST a working directory under `/tmp/<user>/<jobid>/<qid>/`; when CREST returns, that directory is copied into `crest/` (and the fallback's into `crest_shake1/`), and the node-local copy removed. The record CREST's parser produces keeps reading from `crest/`. Scratch reuse and settings matching keep working against `crest/`.
- **MACE engine files come from the census.** The tightening loop passes an ASE trajectory file and log file per conformer into the optimiser and writes the tightened geometry as extxyz with energy and forces; the Hessian screen writes the raw analytic Hessian per surviving basin and the basin's extxyz whose comment names its source conformer (index and the CREST comment). Nothing else is written by the census; the branch A record still goes to `_records/`.
- **OpenMM engine files come from the trajectory driver.** The driver builds an OpenMM Topology (one chain, one residue `MOL`, elements only) and uses OpenMM's reporters or file classes directly against its Context: PDB of the post-relax start, XML of system and integrator, DCD and state CSV at the sampling interval, checkpoint and XML state at every flush. Resume reads the XML state (checkpoint if it loads), counts the DCD frames already present, and appends. `frames.npy`, `summary.json` and the per-trajectory record move to `_records/` with the same content as today.
- **Readers go through one trajectory reader.** A function that, given an OpenMM engine folder, returns positions (Å, float64 array), symbols and masses read from `start.pdb` and `traj.dcd`, plus the sampling metadata from `state.csv`. The collect analysis, the ensemble report and the 02d identity check call it instead of loading `frames.npy`.
- **Branch B reads basins from `mace/`.** The `--basins auto` path lists `mace/basinNN/basin.extxyz` under the molecule directory of the basin tag; the store lookup and the relocated-xyz helper are removed. One reader module in the store package answers every basin question (existence, count, geometries, record, completed set) and every reader goes through it.
- **The submit scripts find the molecule directory with `find`** (two shell functions beside the root resolution: find the directory, count its basins), so the shard rule is spelled only in the layout module. Recorded fact, 2026-09-14: this was built on my false premise that a login node has no usable python; Tianhe login nodes can `conda activate openqha` / `openqha-gpu` and run any driver. The user accepted the helpers as part of the work because they are useful, not because they are needed, and ruled that a change of this kind (special handling for an environment nobody asked about) is theirs to decide next time.
- **Settings under `openmm/`.** The setting name is `default` for the qha and identity chains and the settings-table row name for the 02d-2 array; the array's per-row tag stays for the records but the engine files of all rows sit in one molecule directory.
- **Records are moved, not changed.** Every file this repository wrote about a run keeps its name and content and goes to `_records/` with a flat structure: the branch A record, the trajectory records per setting and basin, the collect tables and log, the completion marker, the ensemble report, the driver logs, the parsl summaries. Their future form is step 2.
- **Old trees are deleted by a script.** One script, `--plan` by default listing every old runs root, `analysis/` tree, basin store and `logs/node_local/` on the machine it runs on with sizes; `--delete` removes them. It never touches the new root.
- **Examples and confs.** The example confs stop naming a basin store; the array submitter and array job stop composing scratch tags; the runbook's storage section and the branch A/B workflow docs describe the new tree.

## Testing Decisions

A good test here checks the tree that appears on disk and what a reader gets back from it,
never how the writer built it. Prior art: the single-file tests under the unit test
directory, one defect story each, under a second, run by the repository's test runner;
the user intends to change that style later and this spec keeps to it for now.

1. **Layout**: from (root, tag, qid) the module returns the documented directories; shard boundaries at 1 000 and 16 000; the seven shipped species land in `1_16000/1_1000`.
2. **Root resolution**: the Tianhe environment file sourced under a fake `HOME` and each partition prints the expected root; no partition and no mount is an error; an explicit `S0_RUNS_ROOT` wins.
3. **OpenMM engine folder**: a few hundred CPU steps on a shipped molecule leave exactly the seven files; DCD frames equal CSV rows; a second run appends and the frame count grows by the new frames only; no `.npy` or record appears in the folder.
4. **MACE engine folder**: the census on a prepared ensemble leaves three files per conformer and two per basin; the basin extxyz names its conformer; the Hessian is 3N × 3N, symmetric, float64.
5. **CREST relocation**: a stand-in crest binary writing the known file set; the molecule directory ends with `crest/` holding that set, and `crest_shake1/` beside it when the fallback ran; the node-local directory is gone.
6. **Readers**: positions read back from DCD equal the written float64 positions to float32 tolerance; a folder without a DCD is refused with the folder named.
7. **Delete script**: on a temporary copy of the old layout, `--plan` lists exactly the trees and removes nothing; `--delete` removes them and nothing else.

## Out of Scope

- The form, names and content of the records (step 2): they move to `_records/` as they are.
- Where logs that are not engine files go (Slurm output, driver stdout, parsl run directories, socket logs): step 2; Slurm output keeps its current place meanwhile.
- The 02c ORCA and xtb folders beyond moving them into the molecule directory: their files are already engine files.
- Any change to the physics, the protocol, the seeds or the criteria.
- Converting old trees: they are deleted, not migrated.
- The test style change the user intends.

## Further Notes

- DCD stores positions as float32 (about 1e-7 Å at these coordinates); the quasi-harmonic covariances are of fluctuations of 0.1 Å, so the analysis is unaffected; the record of the run can note it.
- An OpenMM checkpoint is bound to the platform and hardware that wrote it; the XML state is the portable resume and is written as well.
- CREST's directory is the only engine output that goes through node-local disk; the reason is the small-file I/O measured on 2026-09-12, recorded in the runbook.
- The user tracks the work on GitHub; every step ends with the commit message to use.
