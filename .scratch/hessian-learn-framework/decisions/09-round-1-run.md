# Round 1: the first training + debug run

Type: task
Status: open
Blocked by: 08, 11c
Part of: [hessian-learn-framework](../map.md)

## Question / work

Run the first Hessian training round on [Round-1 xyz](08-round-1-xyz.md)'s assembled dataset, and debug what breaks.

Settled (grilling, 2026-09-29 — unless the run itself says otherwise; amended by the 2026-09-30 re-ruling, see the spec):

- **Stack**: the latest tree only — openQHA and the package at their current `main` (the package's README rewrite `a5f8103` atop `ff92141`), fork `BloomDlwlrma/mace` @ `openqha-hessian` `1110ffb`; no old fork chain, no sha256-era code, no maintenance of old versions; old Records stay byte-identical and readable (no migration, no repair).
- **Entry and knobs**: `workflows/hessian_learning/05_train.py` on the campaign's two production arms — run names `replay30k_w1` / `replay30k_w10`, multiheads on; the Replay is a 30,000-frame draw of SPICE's train split (the mace-docs guidance, user ruling 2026-09-30), drawn twice with seed 0 at `--weight 1` / `--weight 10` so the two frame sets are identical by construction (the weight lives in the file); `--hessian-weight` stays the default; the epoch cap is the optional parameter (`--max-epochs`; Slurm `MAX_EPOCHS`, default 100) set to the budget at claim time and recorded; `--register-copy` on the production runs and on the debug runs (stamped `<campaign>-<run>+<stamp>` entries).
- **Where / the gate**: two bounded Tianhe A800 runs (one per arm, submitted together) are "done"; before them, the cheap local gate (`--dry-run` argv + a 1-epoch subset run) and the 05e-style fresh-env acceptance at Tianhe (`install.sh`, `check_fork(strict=True)`, the package runner 9/9); the session's three commits go into the Answer.
- **Done looks like**: per arm, a `train.{out,toml,dat}` Record whose validation curves are non-degenerate and whose Hessian curve moves, with the identity lines (fork `1110ffb`; `HL_PACKAGE_VERSION`/`HL_PACKAGE_COMMIT`), plus the cross-arm w1-vs-w10 line; and a written list of bugs found and where each was fixed or ticketed.
- **Old-assets boundary** (what "not maintained" means): no new compatibility writes, no shim, no backfill; readers stay tolerant (old frames read as `-`; old Records readable); stale-checkout deletions only ever through 11b's verified process (already done for the old fork); the rest of `_to_delete/` and `_backup/` stays as history; old artifacts whose records cite old paths stay as history — no migration, no repair (closes the map's old-path fog item).
- **Waits for**: 11c (the repo swap) — resolved 2026-09-29 (`088adfe`: the remote serves only `main`); 11d (the CI) does not block.

Spec: [spec-round-1-run.md](../spec-round-1-run.md) — the 2026-09-30 multihead-standard
re-selection (Q1–Q8 approved) and the same day's Q2 re-ruling (the Replay: 30,000 frames in two
arms, weights 1 and 10). Where it differs from the settled block above (learning rate and EMA
truth, Stage Two off, clip/weight decay, the epoch-cap rule, the Record checks, the Replay), the
spec is the operative text.

## Answer

<!-- resolver: append what was run + evidence; set Status: resolved; add a line to the map's Decisions so far -->

**Progress (2026-09-30, the workstation legs -- [09f](../implementation/09f-the-local-gate.md)).**
The local gate ran end to end: the fetched `draw300_r1` dataset is placed at the mirror
(`~/runs/openQHA/draw300/_datasets/draw300_r1/`; the five SHA256s re-verified), the dry-run on
the real merged file is green (splits 27,740 train / 868 valid; production argv mirrored), the
`draw300_r1dbg` subset (2 molecules / 96 frames) backs three 3-epoch mini arms -- both replay
arms (`--weight 1` / `10`, one seed) and the naive comparison -- and all three Records pass the
truth battery the production runbook reuses (LR 0.0001 / EMA True / SWA False; `config.yaml`
carrying clip_grad 1.0 / weight_decay 0.0 / ema_decay 0.99999; the two fork log lines;
`HESSIAN_CURVE_MOVED`; exact anchors 0 < AFTER < BEFORE; the soft E0s look). One gate finding
was fixed at the run level: the single-token `--mace-arg=--key=value` form reaches mace but
`config.yaml` can only read it as a bare flag -- the pinned EXTRA on the four live surfaces and
spec story 14 now use two tokens per mace flag; the driver-side fix and the `control_settings`
mirror are ticketed as [14](../decisions/14-driver-truth-follow-ups.md). The Tianhe legs (timing
job -> cap -> the two production arms -> evidence fetch-back) are the runbook
[09g](../implementation/09g-the-timing-and-production-runbook.md), delivered and awaiting
execution; Status stays open until the two production Records land.
