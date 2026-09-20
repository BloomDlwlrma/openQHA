---
status: accepted
date: 2026-09-14
---

# One directory per molecule, one folder per engine, engine files only

Until 2026-09-14 a run was spread over three trees (a runs root, `analysis/`, and a
`data/basins/` store holding a byte-identical second copy of the basin record), and the
engine files sat beside about twenty-five records written by this repository. The user
ruled that a molecule's results live in ONE directory, `<root>/<tag>/<qid>/`, with one
folder per engine (`crest/`, `mace/`, `openmm/`, `xtb/`, `orca/`) that holds engine files
and nothing else; branch B reads the basins from `mace/`, and the store is retired. Where
the records go is a separate decision (step 2), taken after this one.

## Considered options

- Keep the split and add the missing engine files: rejected, the split itself was the
  objection.
- Engine files and records in the same folder, as before: rejected, a reader cannot tell
  what the engine produced from what this code wrote about it.

## Amendment (user, 2026-09-14, same day)

The `openmm/<setting>/basinNN/` level is dropped: one `md_openmm/basinNN/` per basin holds
every trajectory setting of that basin, the default setting under the bare file names
and any other under `<stem>_<setting>.<ext>` (`traj_p1500_s5.dcd`). The suffix is the
setting and not the job id, so a later job can resume the file by name. Records about a
whole tag (batch summaries) live at `<root>/<tag>/_records/`.

## Amendment 2 (user, 2026-09-15)

The two MD folders are named by role, `md_openmm/` and `md_ase/`, because `mace/` and
`ase/` side by side read as the same thing twice while they are different steps (relax +
Hessian, and sampling). Route identifiers (`--route openmm|ase`) are unchanged; the
layout module's `md_folder()` is the only place the folder name is spelled.

## Amendment 3 (user, 2026-09-20): a frame's label is a file group, not a folder

The Hessian-learning campaign labels ~15 frames per molecule with one ORCA job each, and
the rule "engine files only in engine folders" put every job at
`orca/<level>/frames/<generator>_bBB_kK/job.{inp,out,hess,engrad}` -- ten path levels
below the runs root and 100,000 directories for the campaign, which the user ruled too
long. The exception: a single-job engine run whose products are a fixed, small set of
files is a FILE GROUP of the molecule directory, the path fields folded into the name,
`<molecule>/orca.<level>.<generator>_bBB_kK.{inp,out,hess,engrad}` (`layout.orca_frame_stem`).
The fields are joined by `.` (the repository's `<thing>.<level>.<ext>` convention) because
the level and the frame tag both contain `_`. ORCA itself still runs in a run directory
(node-local scratch, or `<molecule>/.<stem>/`) that is removed after the copy-back, so no
engine scratch file reaches the molecule directory. Later the same day (ticket 09b) the
rule became general and the two studies got their own sub-folders: `frames/` holds the
Hessian-learning set (Frame sets, labels and the per-frame file groups), `msrrho/` the
msRRHO study -- its basin-level jobs as `orca.<level>.basinNN.{inp,out,hess,xyz}` (the
probes of `mode_curvature` as `orca.<level>.basinNN.<tag>.*`), its Records flat in
`msrrho/thermo/` as `<level>.<step>.<ext>` (replacing ADR 0004's `levels/<level>/`), and
`crest_entropy/`, `xtb/` moved whole. Every ORCA job is a file group; `orca/` and
`levels/` no longer exist. In the same ruling the two shard layers of
2026-09-14 go: the molecule directory is `<root>/<tag>/<qid>/`, the tag directory flat --
a campaign's tag holds thousands of molecules, not 133 885 -- and a label-only molecule
sits under its label the same way (`_label/` is gone). One campaign is one tag and one
Dataset: every step's `--name` defaults to the tag. `scripts/tooling/s0_flatten_tree.py`
moves an existing tag to this form.

## Consequences

MACE and OpenMM have no file format of their own, so their folders hold the closest
conventions: ASE's `opt.traj` / `opt.log` / extxyz / a Hessian `.npy`, and OpenMM's
reporters (`.dcd`, PDB topology, state CSV, serialised system and integrator, checkpoint).
