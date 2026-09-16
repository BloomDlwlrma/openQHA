# 23: Enantiomer degeneracy: port of CREST's intraconfRMSD

**What to build:** a `degeneracy` Calculation that gives every basin its `g'` (CONTEXT.md: Enantiomer degeneracy) by running CREST's own algorithm on CREST's own files from the branch A crest engine folder: rotor groups from the topology heuristics (equivalent nuclei bonded to one common neighbour that has at most one other neighbour; ring groups once), those atoms excluded, proper-rotation Kabsch RMSD between all rotamers of one conformer (`cre_members` grouping of `crest_rotamers.xyz`), greedy clustering at 0.125 A -> `g_core`, mirror check (x -> -x, 0.094 A) recorded as a flag, `unsampled` when a rotamer group has one member. The result is mapped onto branch A's basins by the conformer each basin came from and written as `levels/mace-off23_medium/degeneracy.out` and `degeneracy.toml` (`[[Basin]]`: INDEX, CONFORMER, G_PRIME, G_PRIME_SOURCE, N_ROTAMERS, N_CORES, MIRROR_FLAG). No rotamer factor is ever computed into `g'`.

**Blocked by:** None (can start immediately).

**Status:** done 2026-09-16

- [x] propanal's branch A CREST files give g' = (1, 2, 1) for (cis, gauche, third) with the gauche mirror flag set and the third marked `unsampled`
- [x] a rotamer group whose members differ only by methyl rotation gives g' = 1 (the rotor-group exclusion works)
- [x] the Property file starts with `[Calculation_Status]` and the Report ends with the terminal line; both live in the level folder, not in `_records/`
- [x] a molecule directory without `crest_rotamers.xyz` or `cre_members` fails with a message naming the missing file, never with g' = 1
- [x] unit tests on a hand-built rotamer file (one achiral, one chiral pair, one single-member group) run green
