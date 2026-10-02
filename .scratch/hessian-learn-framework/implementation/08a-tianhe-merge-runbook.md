# 08a: The Tianhe merge runbook -- draw300 -> draw300_r1

Type: task
Status: claimed
Serves: 08
Part of: [hessian-learn-framework](../map.md)

> Session artifact, 2026-09-27. Runbook V2, delivered in chat and saved here at the
> user's request. Execution split per the 08 grilling ruling Q3: the Tianhe steps are
> run by the user; the workstation reconciles the paste-back into ticket 08's Answer
> and a map line. Ticket 07's quiet-assemble fix (commit `e1f675b`) does not block this
> merge -- the tree is already assembled; copy it over before any future assemble
> refresh (then the flood of per-molecule `nan` lines and the empty-Batch block are gone).

## State (updated 2026-09-28)

- Steps 0-1: done on Tianhe (the fixed `04_dataset.py`, commit `7536e72`, is what step 4
  runs; the signature/`_datasets` prechecks passed before step 2).
- Step 2: done. Measured: `select: 6443 molecules (0 drawn, 6443 with basins, 6017 with
  a Frame set, 7 pinned) -> .../draw300/_datasets/draw300_r1/select.dat`.
- Step 3: done. First pass (2026-09-27; the flood ticket 07 fixes): `molecules 6017`,
  `frames 37788 to label ... 9335 finished already, 0 failed frames on disk`.
  Refresh (2026-09-28; corrected spelling, quiet ticket-07 output -- see Amendment (2)):
  `molecules 6048 (selection 'draw300_r1')`, `frames 32339 to label ... 16490 finished
  already, 0 failed frames on disk`, `assemble 4437.4 s over 6048 molecules: 2876 with
  labels, 0 with failures or refusals, 3172 untouched (nothing on disk yet)` -- the trees
  now carry every label finished so far. (`6048` vs step 2's `6017`: the driver filters
  the selection against the Frame sets actually on disk, and 31 more landed since.)
- Step 4: merge done (2026-09-28). Measured: `dataset 'draw300_r1' at wb97m-d3bj_def2-tzvppd
  (split by molecule): 6048 molecules (304 test), 324978 frames: train 15751 valid 486 test
  587 pool 308154; 16824 with a Hessian`; `merged  .../mace_draw300_r1.wb97m-d3bj_def2-tzvppd.extxyz
  (16824 labelled frames, keys REF_energy / REF_forces / REF_hessian / split)`; R4 (S0-C-60)
  line: `Replay = 63004 frames at config_weight 10`. Full command block re-run 2026-09-29:
  `rc=0`, `exported  .../molecules-draw300_r1.h5 (16824 labelled frames)` -- the two runs
  produced identical numbers because both read the SAME assembled label files (the
  2026-09-28 assemble refresh; the merge reads the assembled extxyzs, nothing newer);
  frames finished since are simply not assembled yet.
- Step 5: done (2026-09-29). `s0_hl_progress.py --tag draw300`: TOTAL labelled 29728 /
  failed 10554 (the legacy archives) / unlabelled 284692 / running 208 -- the
  29728-vs-16824 delta is exactly the finished-but-unassembled set this step predicted.
  Refreshing steps 3+4 again absorbs it (do that after the labelling sweep stops).
- Steps 6-7: next (fetch once the final refresh is in).

## Budget stop plan (2026-09-29)

Measured unit cost: mean 1862 s/frame (p50 1687, p90 2621, max 12267; 4 ranks) -> **2.07
core-hours per frame**. The running sweep: 13 blocks (832 cores; block-0..12, all started
09-26 23:50, all time out **09-29 23:50**). Burn since 09-25: 67,755 core-hours (48,922 of
them this parsl round); the round burns ~20k core-hours/day.

User ruling (2026-09-29): the remaining budget goes to the hessian training first, and the
training draws from the SAME core-hour account (both confirmed) -- everything left after
tonight is the training reserve. Plan:

1. DONE (2026-09-29 morning: user ruling -- the labelled set, 5,018 molecules / ~29.9k
   frames, is enough for training; cancelled EARLY, before the 23:50 wall). Receipts:
   driver killed on ln201 (`tmux send-keys -t hl-labels C-c`; `pgrep` clean; the tmux
   session that remains is just its empty shell), all 13 `parsl.*` blocks scancelled
   (COMPLETING -> final `squeue` empty). Burn stopped at ~59.3 h per block (~49.3k
   core-hours for the round; ~10.6k saved vs the natural end).
2. NEXT: refresh (step 3) + build (step 4) on the now-static tree; fetch with the
   corrected step-6 list (`index.dat` in, `dataset.dat` out -- never a build product).
   The per-class labelled-molecule table (the 2026-09-29 check): 5,018/6,048 molecules
   (83.0%), 29,858 frames; every class 79-100% except `primary_alcohol` 32.3% (221/684 --
   the lone low class, candidate for any future top-up).
   Remaining labels after tonight ~ 14k frames ~ 29k core-hours if ever wanted (2.07/frame).

---

# The runbook (V2)

**步骤 0 — 拷贝修复版**（工作站）：把 `openQHA/workflows/hessian_learning/04_dataset.py`（`7536e72` 版）覆盖到天河 `~/openQHA-main/workflows/hessian_learning/04_dataset.py`。只此一个文件。

**步骤 1 — 环境 + 预检**（天河登录节点）

```bash
cd ~/openQHA-main && export OPENQHA_PARTITION=deimos && source hpc/env/common.sh && source hpc/env/tianhe.sh
python -c "import inspect; from openqha.data import dataset; print(inspect.signature(dataset.build))"
ls $S0_RUNS_ROOT/draw300/_datasets/draw300/ 2>/dev/null || echo no-draw300-dir
ls $S0_RUNS_ROOT/draw300/_datasets/draw300_r1/ 2>/dev/null || echo no-draw300_r1-dir
```

预期：签名含 `keep_previous`、`train_generators`、`resplit`；`draw300` 目录**无** `mace_*`（那次尾调用静默崩的印证）；`draw300_r1` 不存在。任一不对 → 停下贴回。

**步骤 2 — 选择表**（秒级）

```bash
python workflows/hessian_learning/01_select.py --tag draw300 --name draw300_r1
```

预期结尾：`select: N molecules (N drawn, N with basins, N with a Frame set, N pinned) -> …/draw300_r1/select.dat`。实测已通过：`6443 molecules (0 drawn, 6443 with basins, 6017 with a Frame set, 7 pinned)`。

**步骤 3 — assemble**（把已完成 ORCA 帧落成 04 的唯一原料）

```bash
python workflows/hessian_learning/03_labels.py --tag draw300 --name draw300_r1 --level wb97m-d3bj_def2-tzvppd --generators basin --assemble
```

预期：跑完无 traceback；各分子 `frames/basin.wb97m-d3bj_def2-tzvppd.extxyz` 出现/刷新。（与本轮 `GENERATORS=basin` 一致；若更早还标过别的 generator 且未 assemble，去掉该 flag 跑全量。）实测已通过：`9335 finished already`。刷新重跑见文末 Amendment（2026-09-27 (2)）：`--level` 必须用注册名（下划线），别用文献式的连字符。

**步骤 4 — 合成**

```bash
python workflows/hessian_learning/04_dataset.py --tag draw300 --name draw300_r1 --split-by molecule --export openreact
```

预期关键行：`dataset 'draw300_r1' at … (split by molecule): … frames: train … valid … test … pool …` -> `merged  …/mace_draw300_r1.wb97m-d3bj_def2-tzvppd.extxyz (… labelled frames, keys REF_energy / REF_forces / REF_hessian / split)` -> `exported …h5`。全新名字，无需 resplit；若 >10 分钟仍在跑，Ctrl-C 后改 debug 队列跑同一条命令。

**步骤 5 — 对账**

```bash
python scripts/tooling/s0_hl_progress.py --tag draw300
```

尾行 labelled 总数应与步骤 4 的 labelled（train+valid+test）对齐；差异只会来自"ORCA 完成但没 assemble/半途失败"的帧——两个数都贴回。

**步骤 6 — 回传**

```bash
D=$S0_RUNS_ROOT/draw300/_datasets/draw300_r1
du -sh $D; ls -lh $D
tar -czf ~/draw300_r1_fetch.tgz -C $D mace_draw300_r1.wb97m-d3bj_def2-tzvppd.extxyz molecules-draw300_r1.h5 index.dat dataset.out dataset.toml
```

> 2026-10-02:本步原含 `sha256sum` 收据,按 checksum 退役裁定删除(decision 16;执行于
> 2026-09-29,五枚值在 08 号票 Answer 保留为历史);tar 路径与 listing 即收据。

`molecules-draw300_r1.h5` 是 OpenREACT 布局的导出（训练阶段用），一并带回；训练若直接在天河上跑可去掉。`index.dat` 是逐帧索引（split 的权威记录，下次重建要读它）——必带。train/valid/test/pool 是派生物，不带。（早期清单里的 `dataset.dat` 不存在——那是 select 步骤的 `select.dat` 混淆。）把 tar 弄回工作站，路径告诉我。

**步骤 7 — 贴回**：步骤 1-6 的全部输出（签名行、select 汇总、04 三行、进度尾行、tar 路径）。之后我写 08 号票 Answer + map 一行。将来跑 `RETRY_ONLY=1` 补了标签，重跑步骤 3+4 即刷新（同名 keep_previous；本地已验证两次跑逐字节幂等）。

## Amendment 2026-09-27: step 4 goes through a log

Step 3's flood is why the console alone is not trusted for step 4 -- run it through a log
file and grep the three summary lines instead:

```bash
L=$S0_RUNS_ROOT/logs/dataset_draw300_r1_$(date +%F_%H%M).log
python workflows/hessian_learning/04_dataset.py --tag draw300 --name draw300_r1 --split-by molecule --export openreact > "$L" 2>&1
echo rc=$?
grep -E "^(dataset|merged|exported)" "$L"
```

Paste back `rc`, the three grepped lines, then steps 5-6 unchanged. (Before any future
assemble refresh, also copy `03_labels.py` + `openqha/data/frame_labels.py` from ticket
07, commit `e1f675b`.)

## Amendment 2026-09-27 (2): the level spelling, and step 3's refresh

Step 3 was re-run to refresh the labels with the frames finished since the first pass.
The command came back with the level spelled the literature way -- a hyphen joining the
method and the basis -- and the run stopped at startup:

    KeyError: "no ORCA specification for level '…'; known: dlpno-ccsdt_cc-pvtz,
    hf_cc-pvtz, ri-mp2_aug-cc-pvtz, ri-mp2_cc-pvtz, wb97m-d3bj_def2-tzvppd"

`orca.LEVELS` holds exactly ONE spelling per level and the driver validates `--level`
through `frame_labels.keyword_line` before any file is written and before any ORCA job,
so the stop is fail-safe with no side effects (the tee'd log is the only product): a
hyphen-joined level would otherwise have matched zero frame stems and reported every
molecule untouched. The registered name is the underscore form (`wb97m-d3bj_def2-tzvppd`
-- method `wb97m-d3bj`, basis `def2-tzvppd`, joined by `_`; CONTEXT.md spelling). Earlier
deliveries of this runbook carried the hyphen form in the step-3 command and in the
expected-output lines; every spelling in this file is now the registered one.

Corrected step 3 -- the only change is the spelling:

```bash
python workflows/hessian_learning/03_labels.py --tag draw300 --name draw300_r1 --level wb97m-d3bj_def2-tzvppd --generators basin --assemble
```

`--assemble` exits 0 only when no frame without an ORCA job is left over the chosen
molecules; while the sweep is still filling in labels, rc=1 with the closing tally is
the expected result.
