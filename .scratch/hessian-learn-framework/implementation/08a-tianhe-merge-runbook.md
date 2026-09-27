# 08a: The Tianhe merge runbook -- draw300 -> draw300_r1

Serves: 08
Part of: [hessian-learn-framework](../map.md)

> Session artifact, 2026-09-27. Runbook V2, delivered in chat and saved here at the
> user's request. Execution split per the 08 grilling ruling Q3: the Tianhe steps are
> run by the user; the workstation reconciles the paste-back into ticket 08's Answer
> and a map line. Ticket 07's quiet-assemble fix (commit `e1f675b`) does not block this
> merge -- the tree is already assembled; copy it over before any future assemble
> refresh (then the flood of per-molecule `nan` lines and the empty-Batch block are gone).

## State (2026-09-27)

- Steps 0-1: done on Tianhe (the fixed `04_dataset.py`, commit `7536e72`, is what step 4
  runs; the signature/`_datasets` prechecks passed before step 2).
- Step 2: done. Measured: `select: 6443 molecules (0 drawn, 6443 with basins, 6017 with
  a Frame set, 7 pinned) -> .../draw300/_datasets/draw300_r1/select.dat`.
- Step 3: done. Measured: `molecules 6017 (selection 'draw300_r1')`, `frames 37788 to
  label ... 9335 finished already, 0 failed frames on disk`. This assemble run is the
  flood ticket 07 fixes; the merge itself is unaffected.
- Steps 4-7: next. Step 4 needs the fixed `04_dataset.py` on Tianhe (step 0) -- without
  it the build dies with `AttributeError: 'Namespace' object has no attribute 'resplit'`.

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
du -sh $D
sha256sum $D/mace_draw300_r1.wb97m-d3bj_def2-tzvppd.extxyz $D/dataset.out $D/dataset.toml $D/dataset.dat
tar -czf ~/draw300_r1_fetch.tgz -C $D mace_draw300_r1.wb97m-d3bj_def2-tzvppd.extxyz dataset.out dataset.toml dataset.dat
```

把 tar 弄回工作站，路径告诉我。

**步骤 7 — 贴回**：步骤 1-6 的全部输出（签名行、select 汇总、04 三行、进度尾行、sha256 + tar 路径）。之后我写 08 号票 Answer + map 一行。将来跑 `RETRY_ONLY=1` 补了标签，重跑步骤 3+4 即刷新（同名 keep_previous；本地已验证两次跑逐字节幂等）。

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
