# 16a: The Record without the checksum

Type: task
Status: resolved
Blocked by: None.
Serves: [16](../decisions/16-identity-without-checksums.md).

**What to build:** a fine-tune Record and its registry entry carry no hash of any kind;
the recipe's identity in the Record is the config's path; the package suite is green
with the corresponding test pins rewritten; `check_fork` and the fork/package identity
fields are untouched (Q9=A).

- [x] The Record schema carries no `CONFIG_SHA256` (or any other hash field); the registry entry is written without a hash-derived source.
- [x] The driver module computes no digest for this, and its prose points at "the config file the Record names", not at a digest.
- [x] `t_train_run` / `t_train_engine` pins rewritten (kept-keys list, registry shape); package suite --all green.
- [x] Records already written stay readable -- no reader keys off the removed field.

## Notes

- Old records keep their field; nothing is rewritten (house rule).
- The package repo's only `hashlib` use lives in the deleted helper; the module ends hash-free.

## Answer (2026-10-02, package `ada4e4d`)

**What landed.** The Record carries no hash of any kind and the registry entry's `source`
is no longer hash-derived: `CONFIG_SHA256` leaves `SCHEMA` and the info dict, the digest
helper (`sha256_file`) and `import hashlib` leave `run.py`, and `registry_entry`'s
`source` names the config file the Record names (`<index.dat> + config
<run>/config.yaml`) instead of a digest prefix. The module docstring, the schema comment
and both functions' docstrings point at "the config file the Record names", not at a
digest. `check_fork` and the fork/package identity fields are untouched (Q9=A). Old
Records keep their field and stay readable -- `prop.load` is a raw `tomllib` read and no
live-code reader in either repo keys off it (the only remaining mentions are this
slice's test pins, the tracker and backups).

**Evidence.**
- Red first: the rewritten pins against the old module -- the retired-keys and
  hash-free checks FAIL and `registry_entry` dies with `KeyError: 'CONFIG_SHA256'`.
- `t_train_run` **45/45**: the kept-keys set loses `CONFIG_SHA256` and the retired-keys
  set gains it; the `source` string is pinned exactly; a pre-retirement Record (the
  field injected on disk) still reads; the module source holds no `hashlib` and no
  `sha256_file`.
- `t_train_engine` **32/32**: the run's `train.toml` names the config file on disk and
  carries no `CONFIG_SHA256`; the ENGINES entry's source is the config file.
- Package suite `--all` **9/9** (rc 0; `t_train_engine` 114.8 s CPU), log
  `/tmp/t16a-pkgall.log` (WSL side); residual scan: zero `hashlib` / `sha256_file` /
  `CONFIG_SHA256` in `openqha_hessian/`.
- The openQHA-side driver sentence (`05_train.py`'s Record paragraph) is already updated
  in the shared tree by a concurrent session; it is not in this slice's commit.

**Review record (two-axis, per the `code-review` skill).** Range: package working tree
vs `bb810d0`, two read-only sub-agents. Standards: no documented-standard violations;
the source-text hash-free pin mirrors `t_phl_loss.py`'s convention (judgement call kept).
Spec: the `05_train.py` prose pointer (covered outside this commit, above) and the
old-Record pin injecting the removed field into a new-format fixture (kept -- the
field-tolerant read is the property under test). The package tree also carried a
parallel session's 16d edits (`install.sh` / `README.md` / `ci.yml`); commit `ada4e4d`
excludes them.

**State.** Package: `ada4e4d` on `main` (local; the push stays the user's). Tracker: this
ticket + the map line. Sibling slices 16b-16d are landing in concurrent sessions; 16e
waits on the arms.
