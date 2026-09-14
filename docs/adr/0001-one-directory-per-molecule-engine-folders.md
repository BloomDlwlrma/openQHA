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

## Consequences

MACE and OpenMM have no file format of their own, so their folders hold the closest
conventions: ASE's `opt.traj` / `opt.log` / extxyz / a Hessian `.npy`, and OpenMM's
reporters (`.dcd`, PDB topology, state CSV, serialised system and integrator, checkpoint).
