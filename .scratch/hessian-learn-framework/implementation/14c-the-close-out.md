# 14c: The close-out -- the dbg acceptance, the review and the records

Type: task
Status: resolved
Part of: [hessian-learn-framework](../map.md)
Serves: [14](../decisions/14-driver-truth-follow-ups.md) · spec: [spec-driver-truth-follow-ups.md](../spec-driver-truth-follow-ups.md).

**What happened.** The workstation gate for ticket 14 ran on the `draw300_r1dbg` subset
with three dry-runs on the annotated tree (`%TEMP%\oqt14-dbg.log`):

| run | flags | result |
|---|---|---|
| dry14 | the explicit flags forgotten; `--mace-arg=--clip_grad=1.0` | the mirror warns (`EMA_DECAY 0.99 -> 0.99999, LR 0.01 -> 0.0001`); the argv carries `--lr 0.0001`, `--ema_decay 0.99999` and the extra verbatim (`--clip_grad=1.0`); `config.yaml` reads `lr: 0.0001`, `ema: True`, `ema_decay: 0.99999`, `clip_grad: 1.0` (not `clip_grad=1.0: True`) |
| dry14f | `--lr 0.002 --force-mh-ft-lr` | the argv carries `--lr 0.002` and `--force_mh_ft_lr True`; `config.yaml` reads `lr: 0.002`, `force_mh_ft_lr: True` |
| dry14r | `--multiheads` without a Replay file | refused, naming `--pt-train-file` (rc 1) |

Suites at the tree: package `--all` 9/9 rc 0 (`%TEMP%\oqt14-pkg-suite.log`), openQHA
`--all` 73/73 rc 0 (`%TEMP%\oqt14-oq-suite.log`; pre-annotation -- the annotations touch
comments only).

## The two-axis review

Both axes ran over `f8e6dea` + `e2f2be1` before this record. **Standards**: the public
register held except three process tokens in the new comments and the notes'
"ticket 14"/"round 1" wording -- all dropped in the annotation commits, the dated fix
notes staying; smells: the force verdict riding the controls bag (documented and kept --
the spec's single-resolution design), a duplicated `multiheads` read (hoisted), one test
local renamed. **Spec**: the EMA / EMA_DECAY descriptions lacked the pre-change
comparability note (added); the extras fold had no run_training-level coverage
(ov1/fx1/fz1 added); and the sharp catch -- the extras force verdict was short-circuited
by the driver flag, so a conflicting pair could have restated the lie class the ticket
exists to kill -- fixed in `d4640d6` and pinned by fz1.

## Landed (unpushed -- pushes are the user's)

- openQHA-Hessian `f8e6dea` + review annotations `d4640d6`.
- openQHA `e2f2be1` + review annotations `e4b18b3` (the tracker rides a later commit:
  the records, the ticket's Answer, the map lines).
- Deployment rides the next Tianhe sync; the round-1 arms are unaffected either way
  (their launch lines' explicit values coincide with the mirror).
- Not verified here: the pushes themselves, and any GPU-side run (the gate is the
  dry-run plus the suites; the trained values are unchanged by construction).
