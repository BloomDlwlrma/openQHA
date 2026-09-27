# Round 1: the first training + debug run

Type: task
Status: open
Blocked by: 08
Part of: [hessian-learn-framework](../map.md)

## Question / work

Run the first Hessian training round on [Round-1 xyz](08-round-1-xyz.md)'s assembled dataset, and debug what breaks.

Settle at claim time (with the user):

- **Stack**: run under the current fork chain (old fork checkout, `openqha.training` still inside openQHA), or after the split ([The move](07-the-move.md) / [Install and transport](05-install-and-transport.md))? The data ticket does not depend on the split; this run may or may not.
- **Entry and knobs**: `workflows/hessian_learning/05_train.py` — multiheads/replay settings, `--hessian-weight`, run name, `--register` policy ([SHA256 retirement](04-sha256-retirement.md) may change what registering means).
- **Where**: Tianhe GPU (the campaign site) vs local WSL CPU (debug tier); and what counts as "debug" here (dry-run argv? 1–3 epochs? the full R4 row?).
- **Done looks like**: a `train.{out,toml,dat}` Record whose validation curves are non-degenerate and whose Hessian curve moves, with the identity lines from [Identity after the split](06-identity-after-the-split.md) present; plus a written list of bugs found and where each was fixed or ticketed.

## Answer

<!-- resolver: append what was run + evidence; set Status: resolved; add a line to the map's Decisions so far -->
