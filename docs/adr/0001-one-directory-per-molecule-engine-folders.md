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

## Consequences

MACE and OpenMM have no file format of their own, so their folders hold the closest
conventions: ASE's `opt.traj` / `opt.log` / extxyz / a Hessian `.npy`, and OpenMM's
reporters (`.dcd`, PDB topology, state CSV, serialised system and integrator, checkpoint).
