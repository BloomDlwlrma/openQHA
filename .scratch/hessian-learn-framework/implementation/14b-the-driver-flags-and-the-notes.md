# 14b: The driver's flags and the dated notes -- openQHA

Type: task
Status: resolved
Part of: [hessian-learn-framework](../map.md)
Serves: [14](../decisions/14-driver-truth-follow-ups.md) · spec: [spec-driver-truth-follow-ups.md](../spec-driver-truth-follow-ups.md).

**What happened.** The openQHA side landed as `e2f2be1` (review annotations `e4b18b3`):
`05_train.py` gains `--ema-decay` (float; default None -- mace's 0.99 -- and the help
names the fork's multihead 0.99999) and `--force-mh-ft-lr` (store_true; the native
escape for the fork's multihead lr/EMA rule, the help carrying mace's "not
recommended"), both passed through to `run_training`. The three pinned launch surfaces
carry a dated note that the single-token `--key=value` hazard is fixed -- the two-token
form stays pinned -- and spec story 14 gains the same amendment; the annotation pass
dropped the ticket/round process tokens from those notes (the date alone marks the
change).

## Surfaces touched

- `05_train.py`: the two flags and their help; the call site on the multiheads line.
- `hpc/slurm/hl_train.slurm` header, `workflows/hessian_learning/README.md` production
  block, the T05 production cell: the dated fix note beside the two-token pin.
- `spec-round-1-run.md` story 14: the dated amendment.
- Tracker: the ticket's Status note and the published spec ride the code commit.

## Evidence

- openQHA suite `--all` 73/73, rc 0 (`%TEMP%\oqt14-oq-suite.log`); the annotations
  touch comments only, so the suite was not rerun (recorded in [14c](14c-the-close-out.md)).
- The dbg dry-runs ([14c](14c-the-close-out.md)) exercised the driver end to end: the
  forgotten-flags run shows the mirror, the forced run flips it, the file-less
  multiheads run refuses.
