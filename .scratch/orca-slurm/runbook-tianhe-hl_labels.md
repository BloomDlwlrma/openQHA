# hl_label 天河运行手册（TianheXY-CN）

**范围**：orca-slurm 集（tickets 01-06）里 step 03「reference labels」——对每帧跑 ORCA
`! <level single point> EnGrad [Freq]` 出参考标签——在 TianheXY-CN 上的**提交、判读、恢复**操作。
**行为版本**：orca-slurm tickets 01-06 全部落地后（`256f1ef` / `3263349` / `69040ff`）：
一轮默认携带未归档失败帧的一次重试、`RETRY_ONLY=1` 只扫失败、旧 `RETRY_FAILED` 已删除。**更新**：2026-09-26。

> 这份手册只讲**怎么操作**；出处随各节标注，第一手清单在 §12。要读全量叙述，第一手资料是
> [`docs/hessian_learning_campaign.md`](../../docs/hessian_learning_campaign.md)（campaign 页，"the page it is run from"）；
> 站点事实（分区、文件系统、登录）在 [`docs/tianhe_runbook.md`](../../docs/tianhe_runbook.md)；
> 每步算什么在 [`workflows/hessian_learning/README.md`](../../workflows/hessian_learning/README.md)。
> 卡住或与现场不符时：**直接开会话问 agent，不要凭感觉改脚本。**

**目录**：[§0 速查](#0-一页速查) · [§1 流水线](#1-这是一条什么流水线30-秒) · [§2 提交规矩](#2-提交规矩违反会把作业送进坑里) · [§3 部署与前置检查](#3-部署与前置检查每台每次更新后过一遍) · [§4 六个命令](#4-六个命令照抄即可) · [§5 阶段一览](#5-阶段一览) · [§6 日志判读](#6-日志判读) · [§7 核心契约](#7-03-的核心契约一次重试怎么走) · [§8 恢复剧本](#8-恢复剧本) · [§9 进度与节奏](#9-进度与节奏) · [§10 故障排查](#10-故障排查表) · [§11 术语表](#11-术语表) · [§12 来源与维护](#12-来源第一手资料与维护)

---

## 0. 一页速查

```bash
cd ~/openQHA-main; export OPENQHA_PARTITION=deimos; source hpc/env/common.sh && source hpc/env/tianhe.sh

# 进度（随时可跑，登录节点，1-2 分钟）
python scripts/tooling/s0_hl_progress.py --tag draw300

# 03 标签的两个合法路线（见 §4）
tmux new -s hl-labels
python -u workflows/hessian_learning/03_labels.py --tag draw300 --resource tianhe_cpu --generators basin \
    --max-blocks 12 --walltime 3-00:00:00 2>&1 | tee $S0_RUNS_ROOT/logs/labels_draw300_$(date +%F_%H%M).log
# 或（fallback，手动 rounds）：
GENERATORS=basin TAG=draw300 sbatch --array=0-11 --time=3-00:00:00 hpc/slurm/hl_labels.slurm
GENERATORS=basin RETRY_ONLY=1 TAG=draw300 sbatch --array=0-11 --time=3-00:00:00 hpc/slurm/hl_labels.slurm  # 只清失败
```

**三条铁律**

1. 提交用**默认环境 + 命令行前缀变量**（`TAG=draw300 sbatch …`）；**绝不写受限 `--export` 清单**（会丢掉
   `USER`/`LOGNAME`，曾在 torch import 阶段直接崩：job 7675703）。
2. 每个 stage 只在上一步**结束后**手动启动；每轮 ranges 自己从盘上列 pending——顺序错了只会空跑，不会坏事。
3. 数组的**每个 task 都算一次 submission 配额**（本组 32）：同一时刻只放一个 12 任务的 array，不预排队、不用
   `--dependency` 链。

---

## 1. 这是一条什么流水线（30 秒）

```
00 draw ──► A branchA ──► 02 frames ──► 01 select ──► 03 labels ──► 04 dataset
 (选分子)    (CREST 盆地)   (MACE 构象集)   (选名单)      (ORCA 参考标签)   (数据集)
```

- **hl_label = 03**：对 Frame set 的每一帧，ORCA 在该帧的**固定几何**跑一次单点 + 梯度 (+ 解析 Hessian)。
  一帧一个 ORCA job（4 ranks，`%maxcore 6000`），失败**最多重试一次**（§7）。
- 数据落点：分子树在 **runs root**，`/XYFS02/HDD_POOL/<acct>/<user>/sherwin/runs`（由分区推导，
  作业 banner 的 `runs root` 行会打印；显式 `S0_RUNS_ROOT` 优先）。标签文件：
  `<root>/<TAG>/<qid>/frames/orca.<level>.<gen>_bBB_kK.{inp,out,hess,engrad}`（文件组），
  Record `labels.<level>.{out,toml}`，每 generator 的 `*.extxyz`。
- 谁在哪里跑：**只有 00/01 和 03 的 parsl driver 在登录节点**（driver 不跑化学，只投块）；其余全部 `sbatch`。
  03 是唯一需要多轮（rounds）的 stage。
- draw300 规模：6,458 分子；本轮范围 = **basin 帧 44,480 个**（scope ruling 2026-09-25：训练和 msRRHO 只要
  basin Hessian，其余 generator 留在 pool）。

---

## 2. 提交规矩（违反会把作业送进坑里）

| 规矩 | 为什么 |
|---|---|
| 默认环境 + 命令前缀变量（`TAG=draw300 sbatch hpc/slurm/hl_branchA.slurm`），不写 `--export` 清单 | 受限清单会丢掉身份变量 `USER`/`LOGNAME`；计算节点 passwd map 解析不了 uid 时，torch import 的 `getpass.getuser()` 直接 `KeyError`（job 7675703，2026-09-25）。`--export=ALL` 也不要手写（它就是默认） |
| 同一时刻只提交一个 12 任务的 array；rounds 一轮一轮手动来 | tenant `hku2021_fos4` 配额 32 submissions，**每个 array task 算一个**；预排 3 轮 = 36 被直接拒（`AssocMaxSubmitJobLimit`） |
| 测试一律 `--partition=debug --time=00:30:00` | 短队列排队快、不占 deimos |
| 一个 campaign 一个 `TAG`（目录、Dataset 同名） | 分子树、选名单、数据集都按 tag 落盘 |
| 登录节点只干秒级活 + 03 的 driver（在 `tmux` 里） | 登录节点是共享的；化学在计算节点上 |

> 出处：`AGENTS.md`「Submitting jobs」；campaign 页 §1「为什么不是一串数组」。

---

## 3. 部署与前置检查（每台/每次更新后过一遍）

1. **同步代码**（Tianhe 上的 checkout 是**文件拷贝**、没有 `.git`）：把本地 checkout 拷贝到
   `~/openQHA-main`。**整棵树保持一致**——半新半旧会让站点验收 gate（第 4 步）说不清是哪版在跑。
2. **几何来源**：`data/qm9/curated_qm9.h5` 不在 git 里，缺失时全分子 rc=1
   （`reference geometry not found … curatedQM9 at None`）。拷贝**先临时名再 `mv` 原子换入**，然后两侧
   `python scripts/tooling/s0_verify_curated_qm9.py --sha`；集群侧再跑一次全量校验（`--only <qid>` 查单个）。
3. **ORCA 环境**：由 `hpc/env/orca.sh` 从 `~/env_orca611.sh` 定位（共享 build
   `orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg`，OpenMPI 4.1.8 在 conda env `orca611`），记下
   `S0_ORCA_BIN / S0_ORCA_PATH / S0_ORCA_LIB`。换用别处安装的 ORCA：提前导出这三个变量即可。
4. **站点验收 gate（ticket 01）**：在 checkout 里 `sbatch verify-orca-one.sh`（debug，30 分钟；脚本在
   `orca-slurm/verify-orca-one.sh`，需拷到 checkout）。它拿一个**已知失败**的 frame 手动 `--retry` 一次，
   **必须正常终止**；会先自检这段 checkout 带不带 ORCA-child 修复、再自检 Slurm-blind 缝（期望输出
   `seam stripped: none` / `seam binding knob: none`），并能读出 `FIX-NOT-DEPLOYED` / `SEAM-FAIL` 两种拒收
   结论。**换过 checkout / ORCA / 怀疑环境时重跑它。**
5. **30 分钟首测（每个 stage 一次）**：

```bash
TAG=draw300 LIMIT=16 sbatch --partition=debug --time=00:30:00 hpc/slurm/hl_branchA.slurm
TAG=draw300 LIMIT=16 sbatch --partition=debug --time=00:30:00 hpc/slurm/hl_frames.slurm
python workflows/hessian_learning/01_select.py --tag draw300
TAG=draw300 LIMIT_FRAMES=16 sbatch --partition=debug --time=00:30:00 hpc/slurm/hl_labels.slurm
```

   每步先读日志（§6）再进下一步。整条流水线也可以一键冒烟：`sbatch hpc/slurm/hl_pipeline_debug.slurm`
   （七个固定分子、tag smoke；`SPECIES="…" TAG=t1` 可换）。

6. **03 的路由 gate（5 分钟，命令 5 之前）**：`tmux` 里跑一帧
   `python -u workflows/hessian_learning/03_labels.py --tag draw300 --resource tianhe_cpu --debug --limit-frames 1`。
   看三件事：login 上出现一个 `parsl.*` block、几分钟内一行 `<qid> <frame> labelled <s> <MB>`、driver 退出 0
   → 路线可用。worker 连不回来（block 跑完却没标签）时，按 §10 修 `address_by_interface`。

---

## 4. 六个命令（照抄即可）

```bash
cd ~/openQHA-main; export OPENQHA_PARTITION=deimos; source hpc/env/common.sh && source hpc/env/tianhe.sh

# 00  抽分子（登录节点，秒级；同一 seed = 同一批 6,458 个）
python workflows/hessian_learning/00_draw.py --tag draw300 --per-class 300 --seed 0

# （首测见 §3.5；重建全部 Frame set 用 FORCE=1 … hl_frames.slurm，见 campaign 页）

# 1  A：12 个 submission，~10 h（柔性尾巴单独重投，见 §9）
TIMEOUT_S=14400 TAG=draw300 sbatch --array=0-11 --time=1-00:00:00 hpc/slurm/hl_branchA.slurm
# 2  02：2 个 submission，~1 h，A 之后
TAG=draw300 sbatch --array=0-1 --time=04:00:00 hpc/slurm/hl_frames.slurm
# 3  01：秒级，02 之后
python workflows/hessian_learning/01_select.py --tag draw300

# 4  起一个 driver 活得下去的会话
tmux new -s hl-labels
#    （新 tmux shell 是裸的，先重source：）
#    export OPENQHA_PARTITION=deimos; source hpc/env/common.sh && source hpc/env/tianhe.sh
# 5  03：parsl driver，≤12 个 3 天块，basin only，~3.5 天
python -u workflows/hessian_learning/03_labels.py --tag draw300 --resource tianhe_cpu --generators basin \
    --max-blocks 12 --walltime 3-00:00:00 2>&1 | tee $S0_RUNS_ROOT/logs/labels_draw300_$(date +%F_%H%M).log

# 6  04：driver 结束后
python workflows/hessian_learning/04_dataset.py --tag draw300 --split-by molecule --export openreact
```

**driver 会话操作**：`Ctrl+b d` 离开（driver 继续）；`tmux attach -t hl-labels` 回来；`Ctrl+b [` 滚动 (`q` 退出)。
停止：attach 后 `Ctrl+C`（parsl 会把块取消掉），再 `squeue -u $USER`、`scancel` 残留的 `parsl.*`。
driver 被登录节点重启弄死：**原样重开命令 5**——它会从盘上列 pending、跳过已完成/已归档失败、接管
Slurm 判死的块，几分钟内把在飞帧重跑（§7 锁）。租户有多个登录节点：`tmux ls` 空不代表没有会话，
换台机器 `pgrep -u $USER tmux` 看看。

> 出处：campaign 页 §1。ranges 的 array 路线（fallback、省心）就是 §0 速查里的两条 `hl_labels.slurm` 命令。

---

## 5. 阶段一览

| 步 | 命令 | 规模 | 结束判据（日志/退出码） |
|---|---|---|---|
| A | `TIMEOUT_S=14400 TAG=draw300 sbatch --array=0-11 --time=1-00:00:00 hpc/slurm/hl_branchA.slurm` | 12 submissions，~10 h `[estimate]` | 每分子 `… -> B basins`；尾部 `== A  K done, 0 not done…` |
| 02 | `TAG=draw300 sbatch --array=0-1 --time=04:00:00 hpc/slurm/hl_frames.slurm` | 2 submissions，~2 h | `== 02  K Frame sets written, 0 not…` |
| 01 | `python workflows/hessian_learning/01_select.py --tag draw300` | 登录节点，秒级 | 写出 `select.dat` |
| 03 | driver（命令 5）或 arrays（§0） | ≈ 64,400 core-h，~3.5 天，两轮 | task 0 的 assemble `exit 0`（= 无 in-scope 帧缺少 ORCA job） |
| 04 | `python workflows/hessian_learning/04_dataset.py --tag draw300 --split-by molecule --export openreact` | 分钟级 | `dataset 'draw300' …` + `per class:` 表 |

**耗时（draw300，12 节点）**：03 用**实测** 1,303 s / basin Hessian job、185 s / displaced 梯度 job；A 的
柔性尾巴实测单分子到 19.8 h、task 墙 15–22 h（重投要 2–3 天）；02 实测 122 s/分子 → ~2 h。
数字的基线与推导见 campaign 页 §2。

---

## 6. 日志判读

日志落在**提交目录**的 `logs/slurm/`：数组 = `openqha_hl_labels_<jobid>_<task>.out`，无数组的 = `…_4294967294.out`；
driver 的 log 是命令 5 里 tee 的那个文件。

### 6.1 每个 03 task 日志里你会看到（从上到下的顺序）

| 行 | 含义 |
|---|---|
| `runs root  …` / `tag/name … level …` / `allocation …` | banner：每个都过一次目 |
| `frames     T pending, K for this task` | 本轮待办总数 T，本 task 分到 K |
| `timeout    ORCA per frame TIMEOUT_S=28800 s; a failed frame is re-attempted once by a later round unless its archive exists` | 单帧上限与重试规则 |
| `retry      the failed frames without an archive are re-attempted ONCE in this round; the failed .out is archived as <stem>.failed.out before ORCA starts; a failed retry stays final` | **每轮都打**：本轮携带的失败重试规则 |
| `retry-only the list holds ONLY those failed frames; nothing else is queued` | 仅 `RETRY_ONLY=1` 时 |
| `generators basin only (scope ruling 2026-09-25): the other frame generators stay unlabelled` | 仅 `GENERATORS=basin` 时 |
| `retries    R in this task` + 每帧一行 `<molecule>  <gen>_bBB_kK` | **本 task 的重试切片**：它这轮要重试哪几帧 |
| 工作行 `<qid> <frame> <status> <s> <MB>` | 每帧一行；`status` ∈ `labelled` / `reused` / `refused` / `failed`（先前失败；本次未跑——重试未请求或已花）/ `running`（别处持锁）/ `FAILED …`（本次 ORCA 失败） |
| `== 03 wall W s for K frames` | task 墙时 |
| task 0 续：`== 03 assemble` → 每分子表（含 `failed` 计数）→ `== 04 Dataset` → `== assemble exit 0\|1` | **exit 0 = 无 in-scope 帧缺少 ORCA job**；1 = 还有（对 retry-only 轮，1 属预期，见 §7） |

### 6.2 坏信号

| 症状 | 含义 / 处置 |
|---|---|
| `refused` | `.hess` 几何与 MACE 文件不一致（A/02 重跑过而标签没重跑）→ 重跑该帧的标签 |
| `FAILED` 且 `.out` 以 `openQHA: ORCA killed after TIMEOUT_S=…` 结尾 | Hessian 比估计大：读 `.out`；要给它更多时间就**手动** `--retry` 并加大 `--timeout`（下一轮只会在同样的 `TIMEOUT_S` 下重试它） |
| `MB` 接近 6000 | `%maxcore` 用尽 → 降 `CONCURRENCY` |
| `assemble exit 1` | 还有帧没有 ORCA job：看进度表 `unlabelled`，或直接再投一轮 |
| 整份日志只见 banner 不见工作行、或 block 跑完没标签 | driver 路由问题 → §10 |

> 出处：campaign 页 §4（log 行表）；`hpc/slurm/hl_labels.slurm` 的 echo 行。

---

## 7. 03 的核心契约：一次重试怎么走

一帧的 ORCA job 有界（`TIMEOUT_S=28800 s`，8 h）且**只承诺重试一次**。四种盘面状态：

| 盘上 | 状态 | 下一轮 |
|---|---|---|
| `<stem>.out` 有终止行 + 产物 (`.hess`/`.engrad`) | **finished** | 跳过 |
| `<stem>.out` 无终止行（崩了，或 TIMEOUT 杀） | **failed** | **重试一次**（每轮默认携带未归档失败帧；`RETRY_ONLY=1` 则可以只扫它们） |
| 旁边有 `<stem>.failed.out`（归档） | **retry 已花** | 不再被任何一轮选中（归档 = 最终性标记） |
| 没有 `<stem>.out` | 从未跑 / **cut**（墙时、SIGKILL、节点死） | 整帧重跑（没有任何断点续算） |

- **归档**：重试启动前，旧的失败 `.out` 被改名成 `<stem>.failed.out`（一个槽位，每次写入即替换）——失败证据不丢。
- **finality**：重试再失败 → 保持 failed、档案在、永不重选；重试被 cut → 无 `.out`、档案不动、按普通策略整帧重跑。
- **人工扳手**：单帧 `python -m openqha.data.frame_labels <molecule_dir> <generator> <basin> <k> --retry`
  （档案在了会被拒绝）；`--force` 只留给"重跑全部标签"的刻意操作，**绝不接进任何一轮**。
- **时序规则（硬规则）**：修复部署并验证 → 数失败 → **手动验一帧** → **任意一轮**都行。
  为什么改成"any round"：从 2026-09-26 起 retry 是每轮默认携带的——**你投出的任何一轮都会烧掉它摸到的失败帧的那一次重试**。
- **锁**：ORCA 运行期间 `frames/<stem>.running` 持锁（记 Slurm job、每分钟心跳）；别的进程只在
  "Slurm 不认为该 job 死了 **且** 30 分钟内心跳过"时才当它被持有。两轮并行安全，锁会把帧分区。
- **retry-only 轮的小怪癖（保留）**：`unlabelled` 还有存量时，task 0 的 assemble 仍 exit 1——exit 信号讲的是
  "还有帧没有 ORCA job"，不是"这一轮做完没有"。

> 出处：campaign 页 §3；`openqha/data/frame_labels.py`；ticket 02 + 04 + 06。

---

## 8. 恢复剧本

**A. 大规模失败（2026-09-25 型事故重演）** —— 按硬规则走：
1. 部署修复（checkout 同步 + §3.4 的站点 gate 过一遍）；
2. 数失败：`python scripts/tooling/s0_hl_progress.py --tag draw300`（读 `failed` 列）；
3. **手动验一帧**：`python -m openqha.data.frame_labels … --retry`（它会花掉该帧的一次重试——可以接受）；
4. 扫尾或全量：`GENERATORS=basin RETRY_ONLY=1 TAG=draw300 sbatch --array=0-11 --time=3-00:00:00 hpc/slurm/hl_labels.slurm`
   （只清失败，几小时）——或干脆投常规一轮（`RETRY_ONLY` 不设），失败帧会随行。
5. 之后 `failed` 里剩下的就是"重试也失败"的最终名单：人工读 `.out`（旁边有 `.failed.out`）。

**B. 少量失败清零**：直接上面的 `RETRY_ONLY=1` sweep；投之前先预览（打印摘要与规则行，不提交）：
`python workflows/hessian_learning/03_labels.py --tag draw300 --all --generators basin --retry-only --dry-run`。

**C. 一轮被墙时杀掉**：**原样重投**（git 里什么都不用动）——cut 的帧没有 `.out`，会整帧重来；卡在锁上的帧由下一轮的判死接管（§7），无需手删。

**D. 单帧问题帧**：先 `python -m openqha.data.frame_labels <…> --retry`；若它已被自动重试花掉
（档案在），用 `--force` 是唯一的翻越方式——但那是你的刻意决定，不是轮次行为。

**E. 卡在 `running`**：看 `squeue -j <lock 里的 job>` 与 `frames/<stem>.running` 的 mtime；
`COMPLETING`/终态或 30 分钟没心跳 = 已自由，下一轮会接管。

**F. driver 死了/换登录节点**：原样重开命令 5；或换 ranges 路线（§0）。

> 出处：campaign 页 §3/§5；`verify-orca-one.sh`（"a cut retry just did not complete"）。

---

## 9. 进度与节奏

```bash
python scripts/tooling/s0_hl_progress.py --tag draw300
```

一张表：每结构类一行 + 合计的可读列：分子数 / branch A / `A failed` / Frame set / 帧数 / **labelled** /
**failed** / unlabelled / running——全部从盘上读（不需要数据库）。

- `unlabelled` 归零 = 计算做完；`failed` 就是 §8 要读的名单；`running` 为 0 而 `unlabelled` 不为 0 =
  没有作业在做它：driver 死了（重启）或投一轮 fallback。
- 03 的节奏：basin-only ≈ 3.5 天 = **两轮 3 天**；driver 路线自动补块，ranges 路线一轮接一轮手动投。
- 每轮 task 日志量：工作行 ≈ 43,637/12 ≈ 3.6k 行 + 摘要/切片（listing 已不打印整册；ticket 03 的
  "≤ ~2,000 行"估算没算进工作行，实际以"几千行"为准）。

> 出处：campaign 页 §2/§5；ticket 03 的 Status 注记。

---

## 10. 故障排查表

| 症状 | 原因 | 处置 |
|---|---|---|
| 每个 `*.out` 开头即 `[file orca_tools/qcmsg.cpp, line 394]: … aborting the run` + ORTE `SLURM_TASKS_PER_NODE` 报错 | 2026-09-25 事故：ORCA 的 OpenMPI 看到半套 Slurm 变量。**现已从设计上修复**（ORCA child Slurm-blind） | 复现说明这份 checkout 没有修复或环境被改：跑 §3.4 的验收 gate，按结论同步 checkout |
| 作业在 torch import 处 `KeyError` / `getpass.getuser` 崩 | 提交时写了受限 `--export`，丢了 `USER`/`LOGNAME` | 用**命令前缀变量**重投（§2 规矩 1） |
| ORCA 报 "not enough slots" | mpirun 槽位不足 | 该次启动加 `OMPI_MCA_rmaps_base_oversubscribe=1`（文档化 fallback，默认不开） |
| driver 的块跑完却没有标签；driver 停在 frame 列表 >5 min | worker 连不回 login（多网卡主机名问题） | 在 `hpc/resource_configs/tianhe_cpu.py` 给 `HighThroughputExecutor` 加 `address=address_by_interface("<nic>")`（`ip -4 addr` 找计算网卡），再走 5 分钟 gate |
| `sbatch` 被拒 `AssocMaxSubmitJobLimit` | 组配额 32 满（数组 task 也计数） | 等/问；先 `squeue -u $USER` 看自己排了什么 |
| 分子在 A 阶段超 `TIMEOUT_S` 或 `rc=124` | CREST 尾巴重 | 该分子下一轮再来（timeout 保持 pending；`WALL_S` 是 opt-in）；尾巴单独重投 2–3 天 |
| 登录节点 curl/conda 挂住 | 代理/DNS | 按 `docs/tianhe_runbook.md` §1 先 `setproxy`；挂住的作业照收全额墙时 |
| 帧被判 `running` 但没人在跑 | 锁的持有者死了 | 等 §7 的判据（Slurm 终态或 30 分钟无心跳）后下一轮接管；不要手删锁（让机制干活） |

> 出处：spec.md（事故与修复）+ 研究笔记（OpenMPI 机制）；`hpc/slurm/README.md` rule 2 与 fallback；
> campaign 页 §1（gate 表）；`verify-orca-one.sh`。

---

## 11. 术语表

| 词 | 在这里的意思 |
|---|---|
| **Frame / Frame set** | 一个分子的一个几何（basin/displaced/merged/saddle 由 generator 命名）/ 一个分子的全部帧 |
| **Label** | 一帧的参考值：能量、梯度、（basin/merged/saddle 的）解析 Hessian，连同 Record |
| **Record** | 一棵分子树里由某次 Calculation 写下的 `.out`（人读）+ `.toml`（机器读） |
| **Batch / round（轮）** | 一次 driver 调用 / 一次数组提交 = 一个 round |
| **failed vs cut** | failed = 跑过、无终止行（有 `.out`，可重试一次）；cut = 什么都没留下（整帧重跑） |
| **retry 已花 / archive** | 一次重试启动前把旧失败 `.out` 改名为 `<stem>.failed.out`；档案在 = 该帧 final |
| **RETRY_ONLY / retry-only** | `RETRY_ONLY=1`（脚本）/ `--retry-only`（driver）：只列未归档失败帧的 sweep 轮 |
| **quiet listing** | 日志不再打印全册 frame list / 每分子 walk；每 task 只打自己的重试切片 + 摘要 |
| **slice（切片）** | 某 task 分到的帧子集（数组按 stride 分） |
| **block** | parsl 向 deimos 投的一个单节点 3 天作业（≤12 个） |
| **driver** | 登录节点 tmux 里的 `03_labels.py --resource tianhe_cpu`，只调度不计算 |
| **runs root** | 分子树根：`/XYFS02/HDD_POOL/<acct>/<user>/sherwin/runs`（见 §1） |
| **TIMEOUT_S** | 单帧 ORCA 墙时上限（28800 s）；到点杀 → failed（有 TIMEOUT trailer） |

---

## 12. 来源（第一手资料）与维护

**第一手**（行为以它们为准）：

- [`docs/hessian_learning_campaign.md`](../../docs/hessian_learning_campaign.md) —— campaign 页：序列、成本、状态、日志行、恢复。
- [`workflows/hessian_learning/README.md`](../../workflows/hessian_learning/README.md) —— 每步的机制与 one-attempt 契约。
- [`spec.md`](spec.md) + `issues/01-06`（尤其 [03](issues/03-retry-default-and-retry-only.md)、[04](issues/04-retry-rides-every-round.md)、[06](issues/06-the-record-catches-up.md)）—— 规则与其裁定。
- [`research-orca-slurm-primary-sources.md`](research-orca-slurm-primary-sources.md) —— Slurm-blind 修复的来源笔记。
- [`verify-orca-one.sh`](verify-orca-one.sh) —— 站点验收 gate（§3.4）。
- 脚本：`hpc/slurm/hl_labels.slurm`、`hl_label_worker.sh`、`hl_pipeline_debug.slurm`；`workflows/hessian_learning/03_labels.py`；`openqha/data/frame_labels.py`；`scripts/tooling/s0_hl_progress.py`。
- 站点：[`docs/tianhe_runbook.md`](../../docs/tianhe_runbook.md)；安装：[`docs/tianhe_install.md`](../../docs/tianhe_install.md)。

**外部**（机制性论断的出处）：

- ORCA 6.1 手册·并行与绑定：<https://www.faccts.de/docs/orca/6.1/manual/contents/essentialelements/parallel.html>
- Slurm 文档（`sbatch` / FAQ，配额与数组语义）：<https://slurm.schedmd.com/sbatch.html> · <https://slurm.schedmd.com/faq.html>
- ORCA 论坛（问题与公告）：<https://orcaforum.kofo.mpg.de/>

**维护规则**：这份手册**不定义行为**、只汇总操作。代码或 ticket 改了行为 → 先改 campaign 页/README，
再顺手更新本手册对应小节与"行为版本"。不确定的地方**宁可不写**，也别从记忆里补——去读脚本。
有任何一步和现场对不上：**开个会话问 agent**（它就是这份手册的作者）。
