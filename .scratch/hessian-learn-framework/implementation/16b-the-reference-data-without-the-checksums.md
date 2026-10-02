# 16b: The reference data and the tools without checksums

Type: task
Status: resolved
Blocked by: None.
Serves: [16](../decisions/16-identity-without-checksums.md).

**What to build:** no workflow computes or compares sha256 over reference data or the
project's own products: the QM9 list's pinned digest and provenance match flag, the
curated pack's hash attribute, the artifact index's hash column, the prep tool's
head-hash, the edge-list helper and its benchmark consumer, the stale edges pin, and the
mace-patch source fingerprint are all gone; the affected tests assert the new shapes;
the openQHA suite is green.

- [x] QM9 provenance reports names/paths/counts, no digest and no match flag; its test reflects that.
- [x] The curated pack, the artifact index and the prep tool write/print no hashes; their tests assert the new shape (the index/prep tools had none -- `tests/unit/t_data_tooling_shape.py` added).
- [x] The edge-list hash helper and the consumer field are removed; the stale pin in the edges config is deleted.
- [x] The curated-QM9 verifier's hash modes are removed (user ruling: delete, no size substitute; retired if nothing non-hash remains).
- [x] The mace-patch provenance report carries the behavioral probe result only.
- [x] openQHA suite --all green.

## Notes

- Historical fixtures/records with old hash fields stay byte-identical; `_backup/`, `analysis/` archives untouched.
- This is the widest slice; if it sprawls at build time, split into (b1) data pins/packers and (b2) index/prep/mace-patch.

## Answer (2026-10-02, openQHA; rides this ticket's commit)

**What landed.** The project's own sha256 machinery is out of the reference-data surfaces
and tools; seed material and upstream code untouched.

- **QM9 provenance** (`openqha/data/qm9_uncharacterized.py`): `SOURCE_SHA256`, the digest
  computation and `sha256_matches_recorded` are gone; `load()` still reports the path, the
  source URL/citation and the counts, and the not-found error names the source without a
  hash. `t_filters_f7` asserts the new shape (no digest key, no match flag).
- **Curated pack** (`s0_pack_curated_qm9.py`): writes no `sha256` attribute (and reads each
  file once); `t_curated_qm9_h5` asserts `"sha256" not in attrs`.
- **Artifact index** (`openqha_index_artifacts.py`): `sha256_of` and the `sha256` column
  are gone; the on-disk `analysis/INDEX.*` archives stay byte-identical (historical).
- **Prep tool** (`s0_prepare_data.py`): `sha256_head` and both digest prints are gone; the
  tool computes nothing.
- **Edges**: `config.edge_list_sha256()` (and its `hashlib` import) removed; the package-2
  benchmark no longer prints a checksum nor stores `edges_sha256`; the stale
  `sha256_of_edge_list` pin is deleted from `configs/edges_testset.yaml`; the policy
  sentence in `openqha/__init__.py` / `openqha/config.py` / `configs/openqha.yaml` and the
  F7 note in `configs/filters.yaml` read without a hash.
- **Curated-QM9 verifier** (`s0_verify_curated_qm9.py`): `--sha` and `--digest` deleted
  (delete, no size substitute); `verify_archive` returns `(checked, failures)`; the group
  read-back (the Tianhe damage check) stays. Its test damages one group's OBJECT HEADER
  deterministically (`h5py.h5o.get_info(...).addr + 8`) and asserts the checksum error, the
  exit codes, the absence of both flags, and the `find()` message.
- **mace-patch** (`openqha/potentials/mace_patch.py`): `installed_variant()` carries the
  probe only -- `module_path`, `defect_present`, `n_edges`, `probe_shift_A`, `error`; the
  source sha256, the marker-based `sizing`, the `_MARKER_*` constants and the `inspect`
  use are gone. Consumers updated (`s0_B_md_stability`, the 02a demo, `engine.py` prose);
  the regression test asserts the probed identity and the retired fields' absence.
- **New test** (`tests/unit/t_data_tooling_shape.py`): the index/prep tools had no test;
  they now have one (no digest helper, no sha256 column/report line). `tests/README.md`
  and `.gitignore` residue fixed.

**Evidence.**
- Targeted: `t_data_tooling_shape` (new), `t_verify_curated_qm9`, `t_curated_qm9_h5`,
  `t_filters_f7` PASS standalone; `t_mace_translation_invariance` 4/4. Suite: `--all`
  **74/74** rc 0 (log `%TEMP%\oqt16b-all2.log`; the pre-review unit-only run was 60/60).
- Residual scan: live `sha256`/`hashlib` remain only in the seed material
  (`frames.frame_seed`, `dataset._rng`) and HDF5's own error strings.

**Review record (two-axis, per the `code-review` skill).** Range: this slice's working
tree (24 files + this ticket); two read-only sub-agents. Fixed in-pass: the missing
index/prep shape test; the duplicated header-flip snippet in `t_verify` (one helper now);
the stability script's provenance chain (one `installed` local); `tests/README.md`'s
"pinned SHA-256"/"named" residue; `.gitignore`'s "pinned by sha256" line. Judgement calls
recorded, no change: `read_text` vs `read_bytes().decode` in the packer
(`parse_qm9_text` splits lines either way); the dated-rationale repetition (the register);
`configs/stage0_production.yaml.bak` stays as history.

**State.** openQHA: rides this ticket's commit. 16a landed (package `ada4e4d`); 16d
landed (`9bfacef` / `c328ed6`); 16c in flight; 16e open. Note for 16c: live `--sha`
instructions still stand in `docs/hessian_learning_campaign.md` and
`.scratch/orca-slurm/runbook-tianhe-hl_labels.md:85` (runbooks; 16c deletes them with a
dated line).

