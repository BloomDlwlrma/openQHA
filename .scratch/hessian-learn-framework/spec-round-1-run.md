# Spec: Round 1 — the first Hessian training + debug run, under the multihead-replay standard

Label: `ready-for-agent`. Tracker: `.scratch/hessian-learn-framework/`. Spec for
[09-round-1-run](decisions/09-round-1-run.md); rulings of 2026-09-30 ("Q1–Q8 approved", with the
multihead-standard re-selection of every training knob, and Q2 re-ruled the same day: the Replay
becomes a 30,000-frame draw in two arms, weights 1 and 10). Supersedes the corresponding lines of
the ticket's 2026-09-29 settled block wherever they differ.

**Rulings taken with this spec.** Q1: the fine-tuning parameter set is re-selected with **multihead
replay as the standard** (MACE docs' selection logic for "transfer to a different chemistry / broad
screening / structure search" + the benchmarked study behind it): lr `1e-4` (explicit — the fork
itself forces this value in multihead mode; the Record must say the truth), EMA on with the fork's
forced decay `0.99999`, **Stage Two off**, `clip_grad 1.0`, `weight_decay 0.0`, constant weights
`E1/F100` + `w_H = balance`. Q2 (re-ruled 2026-09-30): the Replay is a **30,000-frame draw** of
SPICE's train split — the mace-docs multihead guidance ("Dataset size ratio: It usually gives best
performance to use as many replay sample as you can. Use `--num_samples_pt` to control this (30000
is a good value)."; "Number of epochs for training: The number of epochs for convergence is between
10 and 30 epochs.") — consumed by **two arms** identical except the frames' `config_weight`:
`replay30k_w1` (weight 1) and `replay30k_w10` (weight 10), submitted together; Mode 1 (original
labels; SPICE and the target labels share the ωB97M-D3(BJ)/def2-TZVPPD level — no pseudolabel).
Q3: dataset placement, the Tianhe visit and the two Replay draws are unchanged, plus log checks.
Q4: the local gate mirrors the production argv for both arms (plus the optional naive arm). Q5: the
launch lines carry the re-selected flags (both arms); the epoch cap stays
`min(60, floor(0.9 × 86400 / SECONDS_PER_EPOCH))`. Q6: the done bar keeps its thresholds and gains
four record/config/log consistency checks. Q7: governance unchanged; the record-truth fix is
run-level now (explicit `--lr 0.0001`) and a `control_settings` mirror is ticketed. Q8: out of
scope, as ruled.

## Problem Statement

Round 1 is the first Hessian fine-tune of MACE-OFF23_medium on the assembled `draw300_r1` dataset.
The originally settled knob block was written against the mace/OFF23 from-scratch recipe; the
fine-tuning literature (MACE's own 2026 fine-tuning guidance and benchmarks) and the fork's own
code say a multihead-replay fine-tune wants different values — and, worse, the fork **silently
forces** `lr = 1e-4` and a different EMA decay in multihead mode while our Records, `config.yaml`
and printouts report the nominal `0.01`/`0.99`. Left unhandled: the first production Record would
be mis-described, the run would carry a schedule the guidance warns can destabilise fine-tuning,
and every downstream decision (cap, budget, comparisons) would be anchored to values that were
never actually used.

## Solution

Run round 1 with **multihead replay as the explicit standard**, and with every knob either taken
from or consciously deviating from that standard, each deviation recorded:

- a single Tianhe visit: refresh the three checkouts, run the 05e-style acceptance, sweep the
  08 leftovers, draw the two Replay files (30,000 frames; `--weight 1` and `--weight 10`, one seed);
- a local gate that mirrors the production argv exactly (both replay arms at mini scale plus an
  optional naive comparison arm), short (1–5 epochs) and cheap;
- a Tianhe timing job that measures `SECONDS_PER_EPOCH` (one measurement — `config_weight` does not
  change per-epoch cost, so it serves both arms), then the two bounded production runs (one job per
  arm) with the cap derived from it;
- a done bar that checks each arm's Record truth (curves, Hessian movement, identity lines, and the
  effective lr/EMA/SWA actually trained) plus the cross-arm line (the w1-vs-w10 reading: coverage
  vs pull), then hands the evidence back to the workstation.

## User Stories

1. As the operator, I want the fetched dataset placed at the local runs mirror
   (`~/runs/openQHA/draw300/_datasets/draw300_r1/`) with all five SHA256s re-verified, so that every
   gate runs against the canonical artifact and not a copy of unknown provenance.
2. As the operator, I want the three Tianhe checkouts (openQHA tree, `openQHA-Hessian`, `mace`)
   refreshed to the current mains and their commit ids recorded, so that the production Record's
   identity lines match the workstation's.
3. As the operator, I want the 05e-style acceptance at Tianhe (`install.sh` local-path mode →
   `check_fork(strict=True)` → the package runner 9/9) run before any training submission, so that
   a broken environment is found on the login node, not inside a GPU allocation.
4. As the operator, I want the 08 leftovers cleaned in the same visit (stale fetch tarball; the
   confirmed pre-rebuild "old extxyz" set), so that only canonical files remain on Tianhe.
5. As the operator, I want the **two Replay files** drawn on this visit — `--n 30000 --seed 0
   --weight 1` and `--n 30000 --seed 0 --weight 10` (the same seed, so the two frame sets are
   identical by construction), `--out .../spice_pt_replay30k_w1.extxyz` /
   `.../spice_pt_replay30k_w10.extxyz` — their `[Replay]` Records pasted back, so that each
   production arm's replay is exactly a settled artifact — original labels, no pseudolabel (SPICE
   and the target share the level).
6. As the operator, I want the pre-flight checks for the draw (SPICE source, forgetting ids,
   in-distribution index, assertions pass, `.valid.extxyz` present), so that a refused draw is not
   mistaken for a missing one.
7. As the operator, I want a local dry-run against the real merged file with a fixed
   `--hessian-weight` (balance skipped), so that argv, splits and the record header are checked on
   the real artifact without a full-set balance pass.
8. As the operator, I want a `draw300_r1dbg` subset dataset (whole molecules from train + valid)
   built by a throwaway script, so that the local 1–5 epoch run is cheap but structurally faithful.
9. As the operator, I want the local runs to mirror the production argv (multiheads on; two
   spice-tiny mini replays, `--weight 1` and `--weight 10` from the same seed; `--lr 0.0001`,
   `--no-swa`, `clip 1.0`, `wd 0.0`, `ema_decay 0.99999`, E1/F100, balance), so that what passes
   locally is what flies at Tianhe.
10. As the operator, I want an **optional** naive arm (multiheads off, same data/loss) beside the
    multihead arm, so that the docs' "compare the arms on the observable you care about" is
    available at negligible local cost.
11. As the operator, I want the local run's Record to pass the same truth checks as the production
    one, so that the check battery itself is validated before the expensive run.
12. As the operator, I want a Tianhe **timing job** (real environment, real dataset, production
    knobs, short cap) that records `SECONDS_PER_EPOCH` — one measurement: `config_weight` does not
    change per-epoch cost, so it serves both arms — so that the production cap is derived from a
    measurement instead of a guess.
13. As the operator, I want the production cap set to `min(60, floor(0.9 × 86400 / s_per_epoch))`
    (the docs' 10–30-epoch multihead convergence window is the expectation; the 60 ceiling is its
    doubled margin), so that each run fits the 24 h wall with margin and does not tread water past
    convergence.
14. As the operator, I want the production launch lines — both arms submitted together, one job
    each (`replay30k_w1`, `replay30k_w10`), the same `EXTRA` line (the weight lives in the replay
    file; nothing else differs between arms) — to carry the re-selected flags via the Slurm `EXTRA`
    variable (`--register --register-copy --lr 0.0001 --no-swa --mace-arg=--clip_grad=1.0
    --mace-arg=--weight_decay=0.0 --mace-arg=--ema_decay=0.99999`), so that both runs are
    reproducible from one line.
15. As the reviewer, I want the accepted-log evidence ("Multihead finetuning mode, setting learning
    rate to 0.0001 and EMA to True"; "Param group 0: lr = 0.0001"), so that the fork's silent
    override is *observed* rather than assumed.
16. As the reviewer, I want the Record to state `LR = 0.0001`, `SWA = False`, `EMA = True` and the
    `config.yaml` to carry `clip_grad 1.0`, `weight_decay 0.0`, `ema_decay 0.99999`, so that the
    nominal-vs-effective mismatch cannot recur in an artifact.
17. As the reviewer, I want the done bar unchanged for physics and applied **per arm**
    (non-degenerate validation curves, `HESSIAN_CURVE_MOVED = True`, exact anchors `AFTER < BEFORE`,
    identity lines) and extended with a soft E0s sanity look (initial energy error well under
    500 meV/atom) plus the cross-arm line (the w1-vs-w10 reading: coverage vs pull), so that
    "done" stays operational rather than vibes.
18. As the next session, I want the evidence fetched back (each production arm's
    train.{out,toml,dat} + config.yaml, and the local gate records) and each arm's model copied
    into the local registry as `mace_off23_draw300/replay30k_w1+<stamp>.model` /
    `mace_off23_draw300/replay30k_w10+<stamp>.model`, placed under the mirror run directory, so
    that round 2 starts from a workstation that holds everything.
19. As the operator, I want attempts, walls and failures listed in the Answer (with the bug list
    and where each was fixed or ticketed), so that "debugged" has a record.
20. As the maintainer, I want the fork-boundary ruling carried: baseline frozen (v0.3.16, no
    rebase, no release switch, never upstream); our fork may be updated when needed, with verified
    code runs, recorded as new anchors.
21. As the maintainer, I want the record-truth fix run-level now (`--lr 0.0001` explicitly) and the
    `control_settings` mirror (multihead lr/EMA/decay rule) as a small ticket, so that truth is
    immediate and the driver stops being able to lie later.
22. As the effort, I want the Answer + a map line at resolution, so that the effort's index stays
    the single place the outcome lives.

## Implementation Decisions

- **The knob set (old nominal → new standard).** lr `0.01` → **`1e-4` explicit** (the fork forces
  this value in multihead mode anyway; explicit makes the Record true). EMA decay `0.99` (nominal)
  → **`0.99999`** (the fork's forced value; the docs' table says `0.9999` — the code's value wins
  and is recorded). Stage Two: on (3/4) → **off** (`--no-swa`; FT guidance: no two-stage schedule).
  `clip_grad 10` → **`1.0`**. `weight_decay 5e-7` → **`0.0`**. Weights stay **constant E1/F100 +
  `w_H = balance`** (the multihead page's own loss example; the balance rule is this campaign's
  dimension calibration, not comparable to PHL's λ or OFF23's absolute weights). Unchanged: batch 4,
  seed 123, float64, `eval_interval 1`, `scheduler_patience 20`, `patience 50`, gaussian probe
  `k=4`, exact anchors on, `--register-copy` on production and debug runs.
- **The Replay *is* the file.** A **30,000-frame draw** of SPICE's train split (the mace-docs
  guidance, Q2 above), drawn by `s0_spice_pt_draw.py` with `--n 30000 --seed 0`, twice: `--weight 1`
  and `--weight 10` — the same seed, so the two files hold identical frame sets and only the stored
  `config_weight` differs (`spice_pt_replay30k_w1.extxyz` / `spice_pt_replay30k_w10.extxyz`); Mode 1
  (original labels; SPICE and the target share the ωB97M-D3(BJ)/def2-TZVPPD level — no pseudolabel).
  The weight lives in the file, never a runtime flag. **Mechanism verdict** (fork `1110ffb`): with a
  user `--pt_train_file` the file *is* the replay set — `mace/mace/cli/run_train.py:297-314` routes
  the special values (`mp`/`omat`/`matpes_*`) to the download path and every user file to the plain
  ase-read path, and `--num_samples_pt` is read only inside `assemble_replay_data`
  (`mace/mace/tools/multihead_tools.py:188`); 30,000 frames is what both arms see.
- **Scale comparison (recorded for context).** MACE-MP-0's own fine-tuning used a 100,000-frame
  MPtrj replay against its ≈1.5M-frame pretraining corpus (≈6.7 %, 1:15) and ≈100-config target
  sets; ours is 30,000 replay frames against a 27,740-frame Hessian target (≈1.1:1 by frame count;
  10:1 by weight in the w10 arm, 1:1 in the w1 arm). The mace-docs guidance ("use as many replay
  sample as you can"; 30,000) is the anchor, and 30k sits in the same order as MP-0's 100k. Our
  replay is a same-corpus draw (SPICE→SPICE), theirs a same-corpus draw (MPtrj→MPtrj); no
  pseudolabel needed on either side.
- **Gates and seams.** Existing seams only: the `05_train` CLI (argv as the contract), the Slurm
  wrapper's env surface, the package test runner (acceptance), and the Record writers (the files
  the checks read). No new seams; no code changes are required for the run itself.
- **Cap rule.** `cap = min(60, floor(0.9 × 86400 / SECONDS_PER_EPOCH))` from the timing job (one
  measurement — `config_weight` does not change per-epoch cost); the documented 10–30-epoch
  multihead convergence window is the expectation, and the 60 ceiling is its doubled margin — a
  margin, not a number to plan on.
- **Record-truth handling.** Run-level: pass `--lr 0.0001` (and the other flags above) so the
  Record/config describe what trains. Driver-level: mirror mace's multihead rule in
  `control_settings` (+ an `EMA_DECAY` field question) as a small ticket — not in this run's scope.
- **Governance.** Small and blocking → fixed in session + committed + listed; larger → ticketed
  with a workaround; nothing blocking left unresolved. Fork updates only with verified code runs.
- **Old-assets boundary** unchanged from the ticket's settled block.

## Testing Decisions

- **What a good check is here.** The run's external behavior is what it leaves on disk: the record
  trio, `config.yaml`, the logs. Checks assert those artifacts (values present, consistent,
  truthful) and never inspect training internals.
- **The battery.** (1) Local dry-run on the real merged file. (2) Local subset 1–5 epoch runs with
  the production argv (both replay arms at mini scale; + optional naive arm): record trio written;
  `HESSIAN_CURVE_MOVED`; truth checks. (3) Tianhe acceptance: `install.sh`,
  `check_fork(strict=True)`, package runner 9/9. (4) Timing job: `SECONDS_PER_EPOCH` recorded → cap
  computed and recorded. (5) Production runs (both arms): the done bar per arm + the four
  consistency checks + the two log lines, plus the cross-arm line (the w1-vs-w10 reading).
- **Prior art.** The 05e acceptance pattern (env + fork + tests); the `t_train_engine` integration
  test (record-level assertions on a real mini fine-tune); the 08a runbook (Tianhe stepwise
  execution with paste-back receipts).

## Out of Scope

- cueq anywhere in the round-1 path (its equivalence battery and benchmarks remain a separate
  workstream; no production use).
- The R0–R3 replay rows, the `w_H` scan, and any campaign work past round 1 (the map's
  out-of-scope list).
- Round 2 (the judge) and the `06` step.
- The `control_settings` mirror and any other driver/package code change (ticketed; the run-level
  flags carry round 1).
- "OFF23 value replication" runs and any further naive-vs-replay comparison beyond the optional
  local arm.
- The Q8 topic (still out of this effort's ticket, as ruled).

## Further Notes

- Parameter sources and links: `research/production-training-parameters.md` (OFF23 scratch + PHL +
  MACE defaults), `research/mace-finetuning-parameters.md` (the fine-tuning guidance, MP-0, Kaur,
  Equitrain, PFT, hippynn; the fork's forced `lr=1e-4` / `ema_decay=0.99999` documented with
  file:line), and `research/replay-ratio-provenance.md` (the replay size/weight history, with the
  mechanism addendum: with a user `--pt_train_file` the file IS the size).
- Budget facts: the Tianhe account's remaining reserve is the training budget (same account as
  labeling); the production draw is **two jobs** — one per arm; two arms double the training draw
  on the reserve (recorded) — each bounded to 24 h; every attempt is listed.
- The Answer records: commits made, attempts, the cap derivation, the check outputs, the cross-arm
  reading, and where the evidence lives locally.
