# 07: The assemble walk goes quiet

**What to build:** A whole-tree `03_labels.py --assemble` stops shouting and stops writing
what does not exist. A molecule with nothing on disk at its level -- no ORCA file group, no
label file, no Record -- comes back untouched from `frame_labels.assemble`: no engine file
is read, neither Record is written. The driver lists only molecules with labels or failures
and closes with one tally --
`assemble  W s over N molecules: X with labels, Y with failures or refusals, Z untouched (nothing on disk yet)`
-- reports the assemble phase's own wall (the summary's old walls measured the empty Batch),
renders NaN statistics as `-`, and drops the empty-Batch table and its `0.0 s / nan / 0/0
frames / slurm (none)` footer in assemble mode. The exit-code gate keeps its meaning: 0 only
when every in-scope frame has an ORCA job.

**Blocked by:** None (can start immediately)

**Status:** implemented 2026-09-27 (commit e1f675b). Evidence: a red repro first -- a /tmp
copy of two untouched molecules gave 2 per-molecule `nan` lines, four new Record files and
`rc=1`; after the fix the same tree gives 0 lines, 0 new files, `no Batch ran (--assemble
reads the disk)` and the tally `assemble  0.0 s over 2 molecules: 0 with labels, 0 with
failures or refusals, 2 untouched (nothing on disk yet)`, still `rc=1`.
`tests/unit/t_frame_labels.py` pins the three behaviours; the full unit suite is 66/66.

- [x] An untouched molecule is left alone by `assemble`: no engine read, no file written,
      the all-unlabelled counts returned.
- [x] The driver lists only molecules with labels or failures, closes with the tally,
      prints no NaN statistic and no empty-Batch block; the exit-code gate is unchanged.
- [x] Unit tests pin the skip, the `-` rendering and the quiet assembly; the full suite
      passes.
