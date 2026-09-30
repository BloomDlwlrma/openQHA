# 09a: The Replay 30k standard -- two production arms, config_weight 1 and 10

Type: task
Status: resolved
Blocked by: None (can start immediately).
Serves: [09](../decisions/09-round-1-run.md) · spec: [spec-round-1-run.md](../spec-round-1-run.md).

**What to build:** The operative round-1 plan states the new production Replay standard
end to end. The Replay is a **30,000-frame draw** of SPICE's train split (the mace-docs
guidance, user ruling 2026-09-30 — the two quoted bullets recorded), consumed by **two
arms** identical except for the frames' `config_weight`: `replay30k_w1` (weight 1) and
`replay30k_w10` (weight 10), submitted together — concurrent when the queue allows. The
spec's rulings block, the draw/launch stories, the cap text, the scale comparison and
the budget line all tell that story, and the mechanism verdict (with a user
`--pt_train_file`, the file IS the size) is recorded where the spec relies on it.

- [x] `spec-round-1-run.md`: the rulings block carries the 2026-09-30 ruling (30,000
      frames; two arms at weights 1 / 10; run names `replay30k_w1` / `replay30k_w10`;
      the mace-docs quotes as its basis); Q2 and every other production `R4 = 4xN`
      passage (stories 5, 9, 14, 18; the Replay decision; the scale comparison;
      Further Notes) rewritten to the new standard.
- [x] The draw story names the two commands — `--n 30000 --seed 0 --weight 1` and
      `--weight 10`, the same seed so the two frame sets are identical by construction —
      with `--out .../spice_pt_replay30k_w1.extxyz` / `..._w10.extxyz`.
- [x] The launch story submits both arms together (one job each) with the same EXTRA
      line (the weight lives in the file; nothing else differs between arms); the done
      bar applies per arm plus a cross-arm line (the w1-vs-w10 reading: coverage vs
      pull).
- [x] The cap rule stays `min(60, floor(0.9 x 86400 / s_per_epoch))` with the docs'
      10-30-epoch window as the expectation (not the ceiling); the timing job stays one
      measurement (config_weight does not change per-epoch cost); the budget text says
      two jobs (two arms double the training draw on the reserve — recorded).
- [x] `decisions/09-round-1-run.md` stops citing the build's `REPLAY_R4_FRAMES` as the
      Replay authority; `research/replay-ratio-provenance.md` gains the mechanism
      addendum (user file = the size; `--num_samples_pt` acts only on the
      `mp/omat/...` download path; verified in the fork at `1110ffb`).

## Answer (2026-09-30, implemented in this commit)

**The rewrite.** The round-1 plan states the production Replay in its new form end to end.
`spec-round-1-run.md`: the header and the rulings block carry the 2026-09-30 ruling — Q2
re-ruled to the **30,000-frame draw** in two arms (`replay30k_w1` / `replay30k_w10`, weights
1 / 10, submitted together), with both mace-docs bullets quoted as the basis ("Dataset size
ratio: It usually gives best performance to use as many replay sample as you can. Use
`--num_samples_pt` to control this (30000 is a good value)."; "Number of epochs for
training: The number of epochs for convergence is between 10 and 30 epochs."). Every
production `R4 = 4xN` passage rewritten: the Solution bullets; stories 5 (the two draw
commands, same seed, the two `--out` files), 9 (local runs mirror both arms at mini scale),
12 (one timing measurement serves both arms), 13 (the docs' 10–30-epoch window is the
expectation; the 60 ceiling its doubled margin), 14 (both arms, one job each, same `EXTRA`
line), 17 (done bar per arm plus the cross-arm w1-vs-w10 reading), 18 (both arms' evidence
+ registry names `replay30k_w1+<stamp>.model` / `replay30k_w10+<stamp>.model`); the Replay
decision (the 30k draw, the two weights, **the mechanism verdict** with its fork file:line
evidence); the scale comparison (30,000 vs 27,740 — ≈1.1:1 by frame count, 10:1 / 1:1 by
weight, the mace-docs anchor); Further Notes (two jobs — the draw doubles on the reserve).

**The mechanism verdict** (verified in the fork at `1110ffb`, the checked-out `mace/`):
with a user `--pt_train_file` the file is the replay set as-is —
`mace/mace/tools/multihead_tools.py:105-128` (`prepare_pt_head`: the special values
`mp`/`omat`/`matpes_pbe`/`matpes_r2scan` take the download branch; every other value lands
as `train_file: args.pt_train_file`), `mace/mace/cli/run_train.py:297-314` (the dispatch:
special values → `assemble_replay_data`; ase-readable files → `get_dataset_from_xyz`, read
whole), and `--num_samples_pt` read only in `assemble_replay_data`
(`mace/mace/tools/multihead_tools.py:188`, passed to `select_samples` as `num_samples`;
default 10000 in `mace/mace/tools/arg_parser.py:587-591` — never in play for our user
files).

**decision 09** (`decisions/09-round-1-run.md`): the Entry-and-knobs line names
`replay30k_w1` / `replay30k_w10`, the 30,000-frame draw and the two weights drawn with one
seed; the build's printed `REPLAY_R4_FRAMES` is cited nowhere as the Replay authority; the
gate and done lines say two arms; the Spec pointer lists the Replay re-ruling.

**The provenance note** (`research/replay-ratio-provenance.md`): §6 "Mechanism addendum
(2026-09-30)" added — the supersession note (the 30k two-arm standard) plus the fork
file:line evidence above; the historical sections stay as provenance.

**Checks.** Residual scan over `spec-round-1-run.md` and `decisions/09-round-1-run.md` for
`R4` / `110960` / `110,960` / `REPLAY_R4_FRAMES` / `4×N_TRAIN_HESSIAN`: 0 hits (the
provenance note keeps its historical sections by design; 09b–09d carry their own rewrite
scopes). The two quoted bullets are present in the spec's rulings block. This is a
documentation-only slice — no runnable code touched, so no `py_compile` or single-file test
applies; the openQHA suite ran on the tree: **all 73 test(s) passed** (60 unit / 11
integration / 2 regression; `SUITE_RC=0`; log
`C:\Users\10704\AppData\Local\Temp\oqt09a_suite.log`).

**Not verified.** Nothing runnable changed; the Tianhe side (the two draws, the timing job,
the two production jobs) stays to be executed — 09b/09c/09d carry the build print, the
live-documents pass and the draws.

**Carried.** The commit also carries the shared worktree's pending one-line 12i refresh in
`map.md` (the annotations `669f823` now verified on the package remote) — `map.md` was
touched for this slice's line and the edit already sat in the working tree.
