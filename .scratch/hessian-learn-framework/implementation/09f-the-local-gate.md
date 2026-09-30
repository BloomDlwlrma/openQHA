# 09f: The local gate -- the dataset mirror, the dry-run, the debug subset, the mini arms

Type: task
Status: resolved
Blocked by: None.
Serves: [09](../decisions/09-round-1-run.md) · spec: [spec-round-1-run.md](../spec-round-1-run.md).

**What to build:** the workstation legs of round 1 before any Tianhe production submission --
spec stories 1 (the fetched dataset placed at the local mirror), 7 (the dry-run on the real
merged file), 8 (the `draw300_r1dbg` subset), 9-10 (the local mini arms: both replay weights
plus the naive comparison) and 11 (the truth battery validated on the local Records) -- with
whatever the gate broke fixed in session.

- [x] Story 1: the five fetch files placed at `~/runs/openQHA/draw300/_datasets/draw300_r1/`; all five SHA256s re-verified against 08's recorded set.
- [x] Story 7: `05_train --dry-run` on the real merged file (production argv mirrored; fixed `--hessian-weight` so the full-set balance is skipped); the splits written and counted.
- [x] Story 8: `draw300_r1dbg` (whole molecules from the real train+valid; throwaway script in the workspace temp dir).
- [x] Story 9: `mini_w1` + `mini_w10` -- two mini replays (spice-tiny, one seed, weights 1 / 10), the production argv mirrored, 3 epochs each.
- [x] Story 10: `mini_naive` -- multiheads off, same data/loss/knobs.
- [x] Story 11: the truth battery passes on the local Records (the same battery 09g runs on the production arms).
- [x] Findings fixed in session: (1, the gate) the single-token `--mace-arg=--key=value` form — two tokens per mace flag pinned on the four surfaces + spec story 14; (2, the review) the launch lines' missing Dataset name — `NAME=draw300_r1` / `--name draw300_r1` on every surface, and T05's production cell's stray `--split-by frame` corrected to `molecule`. Driver-side items ticketed: [14](../decisions/14-driver-truth-follow-ups.md).

## Answer (2026-09-30, implemented in this commit)

**Story 1 -- the mirror.** The five fetch files copied into
`~/runs/openQHA/draw300/_datasets/draw300_r1/` (21 s) and `sha256sum`-verified -- all five match
08's recorded set: `72e9d364...` (merged), `74d20775...` (h5), `4e62a0a3...` (index.dat),
`ce62a5ca...` (dataset.out), `ce119e9e...` (dataset.toml). The dry-run's derived split files land
beside them under `train/dry1/` (4.3 GB, derived data).

**Story 7 -- the dry-run.** `05_train.py --tag draw300 --name draw300_r1 --run dry1 --dry-run
--hessian-weight 0.01 --multiheads --pt-train-file <mini w1> --pt-valid-file <mini w1 valid>
--lr 0.0001 --no-swa --mace-arg=...` (6 min 15 s, rc 0). Header: `train 27740 frames, 27740 with
a reference Hessian`; `valid 868 frames, 868`; `lr 0.0001 ... ema True swa False`; replay 4 frames
`config_weight 1.0`; `mace 0.3.16+openqha fork 1110ffbafd65...`. The mace argv carries the pinned
set (`--E0s foundation`, `REF_*` keys, `--loss external --loss_module
openqha_hessian.phl_loss:build`, `--default_dtype float64`, `--multiheads_finetuning True`,
`--real_pt_data_ratio_threshold 0.0`, `--clip_grad=1.0 --weight_decay=0.0 --ema_decay=0.99999`).

**Finding (story 7's catch).** The single-token extras form `--mace-arg=--clip_grad=1.0` reaches
mace correctly (argparse reads `--opt=value`), but `argv_pairs` -- and therefore `config.yaml` --
can only read it as a bare flag: dry1's `config.yaml` carries `clip_grad=1.0: True`,
`weight_decay=0.0: True`, `ema_decay=0.99999: True`, so story 16's config truth (the file
carrying `clip_grad 1.0` etc.) would fail. **Fixed at the run level**: two tokens per mace flag
(`--mace-arg=--clip_grad --mace-arg=1.0`), verified first in `argv_pairs` (no stray `key=value`
keys) and then end-to-end in the three mini arms' `config.yaml` (`clip_grad: 1.0`,
`weight_decay: 0.0`, `ema_decay: 0.99999`). The four pinned surfaces updated together -- the
workflow README's production block, `hl_train.slurm`'s header example, T05's source cell -- and
spec story 14 amended with the dated note; the driver-side fix and the `control_settings`
mirror are ticketed ([14](../decisions/14-driver-truth-follow-ups.md)).

**Second finding (the two-axis review of `016f95a`, fixed in the annotations commit).** Every
pinned launch line omitted the Dataset's NAME: `05_train.py`'s `--name` defaults to the tag, so
`TAG=draw300 RUN=...` addresses `_datasets/draw300` -- not the canonical `draw300_r1` -- and the
production jobs would have failed on the absent merged file. Fixed on every surface:
`NAME=draw300_r1` on the README / `hl_train.slurm` / T05 launch lines and [09g](09g-the-timing-and-production-runbook.md)'s
timing + both production commands; `--name draw300_r1` in the README build line, the README and
T05 judge lines, T05's hand-run line, `05_train.py`'s docstring example and the campaign page's
command 6. The same sweep caught T05's production cell carrying a stray `--split-by frame` (the
smoke mode) where the production split is `molecule` -- corrected to `--split-by molecule`, as
the campaign page's own note requires. `decisions/14` gained its `Blocked by: None.` line.

**Story 8 -- the subset.** Throwaway script `oqt09-subset1.py` (workspace temp; the repo keeps no
debug piles): picks whole molecules from the real `dataset.toml` Molecule table (the first two
train-side molecules with `N_VALID >= 2`, then further train-only molecules with `N_TRAIN >= 3`),
streams the real merged file (29,979 frames, ~4 min) and
re-writes the kept frames through `dataset._write_split(..., reference=True)` into
`.../draw300_r1dbg/mace_draw300_r1dbg.wb97m-d3bj_def2-tzvppd.extxyz` -- **2 molecules, 96 frames
(92 train / 4 valid), all with Hessians**; the real `REF_*` keys, the fixed per-frame probes and
the `split` keys survive. The picks: `dsgdb9nsd_000547` (stratum r0_O2, 44 train + 2 valid) and
`dsgdb9nsd_001478` (r0_N1O2, 48 + 2). The script's stop rule (8 molecules / 80 frames) bounds
ADDITIONS after the valid providers; the two providers alone reached 96 frames, so it stopped
there.

**Stories 9-11 -- the mini arms.** Two mini replays drawn with the real tool from the spice-tiny
fixture (the tool's own documented rehearsal source): `--n 4 --seed 0 --weight 1|10 --n-valid 2`,
with benign exclusion files under `~/runs/openQHA/spice/_oq09_tmp/`. The two frame sets are
identical by construction (ids files row-for-row equal); the `[Replay]` Records differ only in
`{WEIGHT, FILE, IDS_FILE, VALID_FILE, SECONDS}`. Three 3-epoch arms on `draw300_r1dbg`, the
production argv mirrored (CPU):

| arm | replay | epochs/wall | exact anchors before -> after | probe offset | battery |
|---|---|---|---|---|---|
| `mini_w1` | 4 frames @ w 1 | 3 / 374.7 s | 2.332031e-03 -> 1.748669e-03 (-25.02%) | 16.94% | PASS 19/19 |
| `mini_w10` | 4 frames @ w 10 | 3 / 384.7 s | 2.332031e-03 -> 1.748964e-03 (-25.00%) | 16.94% | PASS 19/19 |
| `mini_naive` | none (multiheads off) | 3 / 446.9 s | 2.332031e-03 -> 1.779127e-03 (-23.71%) | 13.09% | PASS (naive-mode) |

All three arms share the balance `w_H = 7.900544469067027` (the same train file), LR 0.0001,
EMA True, SWA False; the two replay arms' logs carry the fork's lines ("Multihead finetuning
mode, setting learning rate to 0.0001 and EMA to True"; "Param group 0: lr = 0.0001"); `HESSIAN_CURVE_MOVED = True`; the
identity lines name fork `1110ffb` and the package version/commit; the soft E0s look is green
(epoch -1 valid RMSE E/atom 4.86 / 4.87 meV). Registration: `mace_off23_draw300/mini_w1+20260930-093201.model`
and `mini_w10+20260930-094711.model` copied into `openQHA/data/potentials/` (git-ignored). At this
scale the two weights move the target ~identically (the weight enters the Replay head's loss
alone); one curiosity recorded: mace's log shows a single spiky train-loss row (68,905 w1 /
689,055 w10 -- the ~10x ratio tracks `config_weight`), both arms complete and validate cleanly.

`mini_naive` (multiheads off; the optional comparison arm) exercises the same battery with the
two multihead-specific expectations inverted -- `config.yaml` carries `multiheads_finetuning
False` (and no `real_pt_data_ratio_threshold` key, which is emitted only in multihead mode), and
the fork's multihead forced-lr line is absent because non-multihead mode does not force `lr` --
which makes its explicit `--lr 0.0001` recording load-bearing; under those expectations all
checks pass (`rc=0`). Its numbers: the same balance `w_H = 7.900544469067027`, anchors -23.71%
(slightly less target movement than the replay arms' -25.02% / -25.00%), probe offset 13.09%,
3 epochs in 446.9 s (149.0 s per epoch), and a much quieter energy term (final valid
E term 3.874e-08 -> 0.20 meV/atom, against the replay arms' ~2.37e-05 -> 4.86 / 4.87 meV/atom:
the Replay head's pull shows on E first). Registration:
`mace_off23_draw300/mini_naive+20260930-100330.model`. This is the local cross-arm reading
(recorded for the record); the production reading belongs to the two replay arms (09g step 5).

**The walls and attempts** (spec story 19): dry-run 6 min 15 s; subset scan ~4 min; `mini_w1`
~14 min total (balance ~7 min + 3 epochs 374.7 s + anchors); `mini_w10` ~14.5 min; `mini_naive`
~15.5 min (149.0 s per epoch -- the suite ran concurrently for most of its window); no refusals,
no failed attempts, no re-runs. Suite on the tree (after the doc edits): openQHA `--all`
**73/73** (SUITE_RC=0; log `C:\Users\10704\AppData\Local\Temp\oqt09-suite.log`).

**Not verified.** The Tianhe legs -- the timing job, the cap, the two production arms and the
fetch-back -- are the runbook [09g](09g-the-timing-and-production-runbook.md), delivered and
awaiting execution. The mini replays are the tiny fixture (the production pair lives on Tianhe,
[09d](09d-the-two-30k-draws.md)) and `--device cpu` is a local-only deviation; both ride the
receipts.

## Review record (2026-09-30, annotations -- the two-axis review of `016f95a`)

Two read-only sub-agents over `48ce3ba..016f95a`, per the `code-review` skill. The Spec axis
verified the receipts (five SHA256s, splits 27,740/868, the subset counts, the three arms'
arithmetic, the suite 73/73) and the story-14 amendment's token-for-token consistency; it found
the launch lines' missing Dataset name (fixed above) and flagged 09g's per-step wall check as an
unrecorded deviation from story 13's formula (now noted in 09g as a review annotation). The
Standards axis checked the house conventions (ticket shape, `(user, <date>)` register, EOL,
one-map-line rules): the same two fixes plus `decisions/14`'s missing `Blocked by` line; it
recorded the EXTRA literal living on six pinned surfaces as a judgement call -- deliberate, and
this session is the case for the consistency invariant it must hold. Recorded, not changed: the
README's S0 scan-row example keeps its historical `TAG=draw300 RUN=R1` form (it is labelled
history); `hl_labels.slurm`'s tail build keeps `--name "$NAME"` (operator-set, not the training
launch); the Answer's subset-cap phrasing was corrected for accuracy (the stop rule bounds
additions after the valid providers, it does not truncate). Both fixes landed with the tests
that pinned the old text updated in the same commit: `t_frame_labels.py`'s NAME guard now
forbids only the bare `NAME=draw300` (the tag used as the Dataset name) and
`t_hl_campaign.py`'s command-6 literal gains `--name draw300_r1`; unit group all 60 test(s)
passed.
