# Round 1: the first training + debug run

Type: task
Status: open
Blocked by: 08, 11c
Part of: [hessian-learn-framework](../map.md)

## Question / work

Run the first Hessian training round on [Round-1 xyz](08-round-1-xyz.md)'s assembled dataset, and debug what breaks.

Settled (grilling, 2026-09-29 — unless the run itself says otherwise):

- **Stack**: the latest tree only — openQHA and the package at their current `main` (the package's README rewrite `a5f8103` atop `ff92141`), fork `BloomDlwlrma/mace` @ `openqha-hessian` `1110ffb`; no old fork chain, no sha256-era code, no maintenance of old versions; old Records stay byte-identical and readable (no migration, no repair).
- **Entry and knobs**: `workflows/hessian_learning/05_train.py` on the campaign's one production row — run name `R4`, multiheads on; the Replay is 4 x the canonical (post-cancel) rebuild's `N_TRAIN_HESSIAN` at `config_weight` 10, and that build's printed `REPLAY_R4_FRAMES` and draw command are the authority (not the earlier build's 63,004); `--hessian-weight` stays the default; the epoch cap is the optional parameter (`--max-epochs`; Slurm `MAX_EPOCHS`, default 100) set to the budget at claim time and recorded; `--register-copy` on the production run and on the debug runs (stamped `<campaign>-<run>+<stamp>` entries).
- **Where / the gate**: one bounded Tianhe A800 run is "done"; before it, the cheap local gate (`--dry-run` argv + a 1-epoch subset run) and the 05e-style fresh-env acceptance at Tianhe (`install.sh`, `check_fork(strict=True)`, the package runner 9/9); the session's three commits go into the Answer.
- **Done looks like**: a `train.{out,toml,dat}` Record whose validation curves are non-degenerate and whose Hessian curve moves, with the identity lines (fork `1110ffb`; `HL_PACKAGE_VERSION`/`HL_PACKAGE_COMMIT`); plus a written list of bugs found and where each was fixed or ticketed.
- **Old-assets boundary** (what "not maintained" means): no new compatibility writes, no shim, no backfill; readers stay tolerant (old frames read as `-`; old Records readable); stale-checkout deletions only ever through 11b's verified process (already done for the old fork); the rest of `_to_delete/` and `_backup/` stays as history; old artifacts whose records cite old paths stay as history — no migration, no repair (closes the map's old-path fog item).
- **Waits for**: 11c (the repo swap) — resolved 2026-09-29 (`088adfe`: the remote serves only `main`); 11d (the CI) does not block.

## Answer

<!-- resolver: append what was run + evidence; set Status: resolved; add a line to the map's Decisions so far -->
