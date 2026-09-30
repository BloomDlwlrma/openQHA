# 09b: The build's production Replay line -- 30,000 frames, both weights

Type: task
Status: resolved
Blocked by: 09a.
Serves: [09](../decisions/09-round-1-run.md) · spec: [spec-round-1-run.md](../spec-round-1-run.md).

**What to build:** Re-running the dataset build's print step shows the production Replay
instruction in its new form — 30,000 frames, two weights (1 and 10), the two draw
commands — and no live code path still describes the production Replay as 4 x
`N_TRAIN_HESSIAN`. The S0-era ladder recipe fields stay, clearly labelled as history.

- [x] The 30,000 is defined once in the data layer; the build print emits the two
      production commands from it (same seed; `--weight 1` / `--weight 10`; the
      `spice_pt_replay30k_w1` / `_w10` outputs).
- [x] The build print no longer reads as "the production Replay = the R4 recipe": the
      R4 line is replaced or unmistakably marked historical (the S0 2026-09-21/22 scan
      rows).
- [x] The schema descriptions stop claiming the production Replay is 4 x
      `N_TRAIN_HESSIAN` (the `N_TRAIN_HESSIAN` description; `REPLAY_R4_FRAMES`
      re-labelled as the S0 ladder recipe).
- [x] The affected unit tests are updated and the suite is green.

## Answer (2026-09-30, implemented in this commit)

**The print step** (`workflows/hessian_learning/04_dataset.py`). The build's print no
longer offers the R4 recipe as the production Replay. It states the production
instruction — the same 30,000 SPICE train frames in both arms, one seed, written twice;
only the stored `config_weight` differs — and emits both draw commands from the data
layer's constants:

    python scripts/tooling/s0_spice_pt_draw.py --n 30000 --seed 0 --weight 1 --out <root>/spice/spice_pt_replay30k_w1.extxyz
    python scripts/tooling/s0_spice_pt_draw.py --n 30000 --seed 0 --weight 10 --out <root>/spice/spice_pt_replay30k_w10.extxyz

The R4 mention stays as one unmistakably historical sentence — "R4 (the S0 replay
ladder's scan row, 2026-09-21/22 -- historical; not the production Replay): 4 x N train
frames with a Hessian = M frames at config_weight 10" — with no draw command of its own
(the old `spice_pt_R4.extxyz` instruction is gone).

**The data layer** (`openqha/data/dataset.py`). The production standard is defined once:
`REPLAY_N_FRAMES = 30000` (the mace-docs multihead guidance's "30000 is a good value"),
`REPLAY_SEED = 0`, `REPLAY_WEIGHTS = (1.0, 10.0)`; the R4 pair
(`REPLAY_PER_HESSIAN_FRAME_R4`, `REPLAY_CONFIG_WEIGHT_R4`) stays, re-labelled as the S0
replay ladder's historical scan row. The schema descriptions stop claiming the production
Replay is 4 x `N_TRAIN_HESSIAN`: `N_TRAIN_HESSIAN` says the R4 row *was* 4 x this, and
`REPLAY_R4_FRAMES` reads as the S0 ladder's R4 scan row (2026-09-21/22), historical — not
the production Replay (a fixed 30,000-frame draw, s0_spice_pt_draw.py). Verified end to
end: a synthetic `property.write` renders exactly those texts as the `dataset.toml`
comments beside the two keys.

**The tests** (`tests/unit/t_dataset_cli.py`). A second phase stubs `build` with a
minimal Record, runs `main()` to the end and captures stdout; the checks pin the spec's
literals — both 30k commands (one seed; weights 1 and 10; the `spice_pt_replay30k_w1` /
`_w10` outputs), the three data-layer constants at their settled values, and that no line
reads as "the production Replay = the R4 recipe" (the R4 mention is the S0 ladder's scan
row, marked historical; `spice_pt_R4.extxyz` appears nowhere).

**Checks.** `t_dataset_cli.py` PASS (5 checks); the related unit files green (`t_dataset`
PASS; `t_dataset_generators` 15/15; `t_dataset_mace_form` 10/10; `t_frame_labels` PASS;
`t_module_entry_points` 2 entry points); `py_compile` clean on the three changed files;
the suite `--all` **all 73 test(s) passed** (60 unit / 11 integration / 2 regression;
`SUITE_RC=0`; log `C:\Users\10704\AppData\Local\Temp\oqt09b_suite.log`). The run sat on
the shared worktree while the parallel 09c session landed its commits; that range touches
none of this slice's three files (`git log d155ad3..HEAD --` the three paths is empty).

**Not verified.** Nothing on Tianhe: the two draws are [09d](09d-the-two-30k-draws.md),
and the acceptance and the production runs belong to
[09](../decisions/09-round-1-run.md)'s execution. The live documents are
[09c](09c-the-live-documents-pass.md) — outside this slice by design.
