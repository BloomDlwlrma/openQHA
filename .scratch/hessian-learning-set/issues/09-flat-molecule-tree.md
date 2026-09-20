# 09: A flat tree: NAME defaults to TAG, no shard layers, frame labels as files of the molecule directory (`store/layout.py`, `data/frame_labels.py`, steps 00-04, `hpc/slurm/hl_*`)

**Ruling (2026-09-20, three messages):** the tree is too long. (A) one campaign = one tag = one Dataset: `--name` defaults to `--tag`. (B) a frame's ORCA files are not a directory two levels down but FILES of the molecule directory, the path's fields folded into the name. (C, "同时删除这两层") the two shard layers `<range>/<chunk>` go: `<root>/<tag>/<qid>/`.

**What to build:**

*A. NAME = TAG by default.* `00_draw / 01_select / 03_labels / 04_dataset --name` and `hl_list.py --name` default to the tag; `hl_branchA.slurm` / `hl_frames.slurm` get `NAME="${NAME:-$TAG}"` (labels / pipeline_debug already have it). The Dataset directory stays `<root>/<tag>/_datasets/<name>/`, so `TAG=draw300` alone gives `<root>/draw300/_datasets/draw300/`. A second subset on the same tree still names itself. No data moves; the README's commands lose their `NAME=`.

*B. Frame labels as files.* Today (10 levels):

    <root>/<tag>/<range>/<chunk>/<qid>/orca/<level>/frames/displaced_b00_k3/job.{inp,out,hess,engrad}

After (5 levels; the four fields engine, level, frame, kind become one name; no shard layers):

    <root>/<tag>/<qid>/orca.<level>.displaced_b00_k3.{inp,out,hess,engrad}

- `layout.orca_frame_stem(level, generator, basin, k)` -> `"orca.<level>.<gen>_b<BB>_k<K>"`; `layout.orca_frame_file(molecule, level, generator, basin, k, ext)`; `orca_frame_dir` removed (its one caller outside `frame_labels` is `03_labels`). The field separator is `.` -- the repository's convention for `<thing>.<level>.<ext>` (`frames/displaced.<level>.extxyz`, `labels.<level>.toml`); `_` cannot separate fields because the level (`wb97m-d3bj_def2-tzvppd`) and the frame stem (`displaced_b00_k3`) contain it. The tokens `frames` and `job` are dropped: they carry nothing (the generator names the frame; there is one job per frame). Basin-level ORCA jobs of the msRRHO study (`orca/<level>/basinNN/`, four readers in `thermochem/`) are NOT touched: closed study, existing trees and fixtures.
- `frame_labels`: `finished / running_elsewhere / _claim / _release / parse_label / label_one / assemble` take `(molecule, stem)` instead of `(workdir, "job")`; the lock is `<stem>.running`; ORCA still runs `job.inp` in the node-local scratch (or, without scratch, in a temporary `<molecule>/.orca_<stem>/` removed after the copy-back), and the copy-back renames `job.<ext>` -> `<stem>.<ext>` for the KEEP kinds. `label_one`'s returned `workdir` / `out` keys become the molecule directory and the `.out` path.
- ADR 0001 amended (an ADR 0001a paragraph in `docs/adr/`): "engine files only in engine folders" gets the exception "a single-job engine run whose products are a fixed, small set of files (the frame labels) is a file group of the molecule directory named `<engine>.<level>.<job>.<ext>`"; decision S0-C-50 records it.
*C. No shard layers.* `layout.molecule_dir(root, tag, qid)` = `<root>/<tag>/<qid>`; `shard`, `CHUNK`, `RANGE` and the `_label/` branch removed (a label-only molecule sits under its label); `basins.completed / census` and the readers of `dataset`, `02_frames`, `03_labels`, `s0_delete_old_layout` glob one level (`*/`); `worklist.completed / remaining` lose `chunk_dir`. A tag holds thousands of molecules, not 133,885: one directory is fine on Lustre (a `ls` of 6,458 entries).
- Migration: `scripts/tooling/s0_flatten_tree.py --tag <tag> [--apply]`, two idempotent stages: the molecules up from `<range>/<chunk>/` and `_label/`, then every `orca/<level>/frames/<frame>/job.<ext>` to the file group; empty folders removed, an existing target is a reported conflict. Run on tianhe for `smoke`, `rings`, `propanal` before any campaign job; the fixture `tests/data/propanal_molecule` gains no frame files (the frame-label tests build theirs).

**Blocked by:** 06 (code). Must land BEFORE the campaign's `hl_labels` array: after 100,000 frames the move is 400,000 renames.

**Delivers:** paths a person can read; 100,000 fewer directories in the campaign; one name on the command line.

- [x] unit: `orca_frame_stem` spells `orca.wb97m-d3bj_def2-tzvppd.displaced_b00_k3`; a level with a `/` is refused; `t_frame_labels.py` (25 checks) passes on the flat names, including the lock `<stem>.running`, the scratch copy-back rename, `finished` on `.hess` / `.engrad`, `assemble` finding both kinds, and no `orca/` directory left in the molecule
- [x] unit: `s0_flatten_tree` stage 1 on a copied tag in the shard form (+ `_label/acetone`) moves both up and removes the three folders; stage 2 on the molecule leaves no `orca/<level>/frames/`, keeps `orca/<level>/basin00/`, a second plan is empty, `assemble` reproduces the label files byte for byte
- [x] unit: `NAME="${NAME:-$TAG}"` in all four stage scripts, no `NAME=draw` in the headers or the README, `--name` optional in 00/01/04 and `hl_list.py` (03: the default mode when neither `--species` nor `--all`); `t_layout_molecule_directory` holds the flat tree (`molecule_dir` = `<root>/<tag>/<qid>`, no `shard`); unit group 45/45
- [ ] tianhe (user): `s0_flatten_tree.py --tag smoke` (and `rings`, `propanal`) dry run then `--apply`, then `04_dataset --tag rings --tag propanal --name smoke` reproduces `test 65 / pool 0`

**Closing (code, 2026-09-20):** A (`--name` defaults; `run.sh`), B (`layout.orca_frame_stem / orca_frame_file`; `frame_labels.finished / lock_file / running_elsewhere / _claim / _release / parse_label / label_one / assemble` on `(molecule, stem)`; ORCA always in a run directory, KEEP copied back under the stem; `03_labels.pending`), C (`layout.molecule_dir` flat; `basins`, `worklist`, the four globs), ADR 0001 amendment 3, README / output inventory / branchA workflow doc, `s0_flatten_tree.py`, tests. Decision S0-C-50. The tianhe migration of the three small trees is the user's run, before any campaign job.
