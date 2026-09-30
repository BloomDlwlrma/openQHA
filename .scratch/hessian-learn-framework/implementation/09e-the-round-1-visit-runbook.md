# 09e: The round-1 visit runbook -- refresh, acceptance, leftovers, the two draws

Type: task
Status: resolved
Blocked by: None.
Serves: [09](../decisions/09-round-1-run.md) · spec: [spec-round-1-run.md](../spec-round-1-run.md).

> Session artifact, 2026-09-30, delivered in chat and saved here at execution time. The
> Tianhe steps are run by the user; the workstation reconciles the paste-back into this
> file's steps and [09d](09d-the-two-30k-draws.md)'s Answer.
> 范围 = spec 的 single visit 四步:**刷新三个 checkout → 05e 式验收 → 08 残留清理 →
> 两次 Replay draw**。timing job 与两条生产提交不在本清单(spec 列在访问之后);
> 数据集的工作站镜像(story 1)也不在此,它是工作站侧步骤。

## 状态

- 步骤 0:**通过**(两侧)。三个 checkout 由 GitHub ZIP + 工作站打包的 `.git` 叠合;ids 逐字
  核对一致(mace `1110ffb…`、openQHA-Hessian `411abe1…`、openQHA-main `5b0b5b9…`),两侧
  `status` 全净。
- 步骤 1:**通过**(CN/A 两个 env + AI `openqha-gpu`)。各:install.sh 结尾 `ok: …` +
  `check_fork(strict=True)` 绿(fork `1110ffb`,dirty False)+ 包 runner 9/9;CN cpu env 的
  numpy 被 install 升到 2.4.6、CN gpu 到 2.2.6,测试均绿。
- 步骤 2:**通过**。`$R`/`~`/`_datasets` 的 tgz 点名均空;`draw300_r1/` 五个 canonical 未动、时间戳全对(2026-09-29);
  两个 pre-rebuild 副本(`draw300/_datasets/draw300/pool.wb97m-d3bj_def2-tzvppd.extxyz`、
  `molecules-draw300.h5`,09-21)确认后删除。
- 步骤 3:**通过**。输入:SPICE 发布集(工作站两份 Apollo tar.gz,1.62 GB + 85 MB)以 1.70 GB
  传至 `$R/openQHA-main/data/training_sets/mace-off23_spice/`,tar 与解出 xyz 的四条 sha256 全部
  核对通过(test 50195 帧);`s0_spice_test_draw.py --n 5000` 生成 forgetting ids。3.0 preflight
  三行 OK;3.1 两条 Replay draw 一次通过(两条 stdout、四条 sha256、3.3 验证输出与两份
  `[Replay]` Record 全文见 [09d](09d-the-two-30k-draws.md) 的 Answer)。同一套也从 AI 侧走了
  一遍(test draw 在 `/XYAIFS00` 也生成;两侧 ids 仅 `# source` 头行不同,CN draw 用 `/XYFS02` 份;
  AI 侧数据副本按新政策可删,留给收口后处理)。**拒绝/墙**:无(此前的“输入缺失”是前置阻塞,
  补救记录在上;工具本身未发生拒收)。
- **站点事实勘误(2026-09-30 实测)**:**tianhexy-ai 的计算节点对 `/XYFS02` 与 `/XYFS03` 均直连
  可读写**(探针作业 259412,v100x 节点 vn205:`findmnt` 两挂载在场;sherwin 读 OK + marker 写
  OK;XYFS03 建目录 + marker OK)。df:XYFS02 7.6P 空闲(28%),XYFS03 1018T 空闲(24%)。据此:
  XYFS02 的挂载方含 `tianhexy-ai`;新增 **XYFS03 = `/XYFS03/SSD_POOL/hku2021_fos4/hku2021_fos4xy_2`**
  (SSD 池,`tianhexy-ai` + `k8s_xingyiAI_2`);2026-09-15 的“AI 集群读不到 /XYFS02”记录作废。
  后续政策(用户 2026-09-30):**新文件只建 `/XYFS02`**(XYFS03 备用);AI 侧作业显式
  `export S0_RUNS_ROOT=/XYFS02/HDD_POOL/hku2021_fos4/hku2021_fos4xy_2/sherwin/runs`(需要时加
  `S0_TRAINING_SETS_ROOT=…/openQHA-main/data/training_sets/mace-off23_spice`)。`docs/tianhe_runbook.md`
  §0b、`hpc/resource_configs/*` 的同步改录与 `root.sh` 映射是否纳入 XYFS03 = 后续单独一票,不在本次。
- 证据(留在盘上):探针日志 `~/probe_xyfs_v100x_259412.log`(AI home);markers
  `sherwin/probe_xyfs02_from_v100x.ok` 与 `/XYFS03/…/probe_xyfs03_from_v100x.ok`。

## 参考:三个 checkout 与预期 id(2026-09-30 实测)

**本机布局(2026-09-30 实测):两侧都已铺好,三个 checkout 分别在各自的 `$R` 下
(`openQHA-main` / `openQHA-Hessian` / `mace`);两侧的 `$HOME` 里都没有树。下文命令的树
路径一律用 `$R`(步骤 0 里出现的 `~/HDD_POOL/sherwin` 就是当日那一侧的 `$R`)——开跑前
按登录侧设一次:**

- CN/A 侧:`export R=/XYFS02/HDD_POOL/hku2021_fos4/hku2021_fos4xy_2/sherwin`
- AI 侧:`export R=~/HDD_POOL/sherwin`(= `/XYAIFS00/HDD_POOL/hku2021_fos4/hku2021_fos4xy_2/sherwin`)

登录节点必须先 `export OPENQHA_PARTITION`(这台登录节点两挂载都在场,不设则根解析失败):
**CN/A 侧 `deimos`(→ `/XYFS02/.../sherwin/runs`),AI 侧 `h100x`(→ `/XYAIFS00/.../sherwin/runs`)**;
换侧或改分区后先 `unset S0_RUNS_ROOT` 再 source(解析只在它未设时重算,旧值会粘住)。

| 树 | 路径 | 分支 | commit(当前预期) |
|---|---|---|---|
| openQHA | `$R/openQHA-main` | main | `5b0b5b9a70eb8f4fb9f8f40deaa62e1627c26304`(ZIP 时点为准) |
| mace(冻结) | `$R/mace` | `openqha-hessian` | `1110ffbafd651a74d1d4678deb4748056d1dff0a` |
| openQHA-Hessian | `$R/openQHA-Hessian` | main | `411abe1b2610e51c1d2e977c4475576d1b4eb981` |

两条身份线从**后两个** checkout 的 `.git` 读出(`mace_fork_info()`、`package_identity()`),
所以它们必须带 `.git` 且 tracked-clean;`openQHA-main` 的 id 只做记录。

---

# The runbook

## 步骤 0 — 三个 checkout 到位(代码 ZIP + 工作站 `.git` 叠合;无网络 git)

**路由(2026-09-30 实测):** 网络 git 过代理不可靠(503 / HTTP2 framing 错)→ 不再用;
代码用 GitHub 网页的 **ZIP**(已在 `~`:`mace-openqha-hessian.zip`、`openQHA-Hessian-main.zip`、
`openQHA-main.zip`),`.git` 由**工作站打包**随你的通道过去,两边拼成完整 checkout。
注意:这栈里 git 的**本地只读**调用去不掉——install.sh 的校验、`mace_fork_info()`、
`check_fork`/训练守卫都跑 `git rev-parse` 与 `git status`(不碰网络);ZIP 单独不够
(没有 `.git`,步骤 1 会拒绝),但 ZIP + 工作站的 `.git` 已在本机逐字验证:rev-parse
得到锚点、`status` 全净、断网也不触发取物。

**工作站(已生成,三个文件在 `C:\Users\10704\Downloads\`):**

    mace-dotgit.tar              155M   <- mace/.git(1110ffb,分支 openqha-hessian)
    openQHA-Hessian-dotgit.tar   1.1M   <- openQHA-Hessian/.git(411abe1,main)
    openQHA-main-dotgit.tar       14M   <- openQHA/.git(main)

把它们用你惯用的通道放到天河 `~`。(之后如需重打:`tar -C mace -cf mace-dotgit.tar .git`,
三棵同理。)

**天河(登录节点;不需要代理、不需要网络 git;在树所在目录操作):**

```bash
cd "$R"                                     # 当日那一侧的项目目录($HOME 下没有树)
# 目录名对齐(以你实际解压出的名字为准)
mv mace-openqha-hessian mace
mv openQHA-Hessian-main openQHA-Hessian
rmdir openQHA-main 2>/dev/null || true      # 空目录让位给 unzip
unzip -n openQHA-main.zip                   # 产出 openQHA-main/

# 叠 .git(与 ZIP 同层的隐藏目录)
tar -xf mace-dotgit.tar -C mace
tar -xf openQHA-Hessian-dotgit.tar -C openQHA-Hessian
tar -xf openQHA-main-dotgit.tar -C openQHA-main
```

**核对 + 记录(天河;同样在 `$R` 里跑):**

```bash
cd "$R"
ls openQHA-main | head -3                   # 应见到 README.md 等;只有 .git 就先 unzip -n openQHA-main.zip
for d in openQHA-main openQHA-Hessian mace; do printf '%-18s ' "$d"; git -C "$d" rev-parse HEAD 2>/dev/null || echo 'no .git'; done
git -C mace branch --show-current
git -C mace status --porcelain --untracked-files=no && echo "mace clean"
git -C openQHA-Hessian status --porcelain --untracked-files=no && echo "package clean"
```

预期(2026-09-30):`mace` = `1110ffbafd651a74d1d4678deb4748056d1dff0a`(分支
`openqha-hessian`,clean)、`openQHA-Hessian` = `411abe1b2610e51c1d2e977c4475576d1b4eb981`
(clean)、`openQHA-main` = `5b0b5b9a70eb8f4fb9f8f40deaa62e1627c26304`(若你的 ZIP 是更早
时点下载的,rev-parse 不同就如实记录——它不参与任何检查;`status` 若有噪音同理)。前两个
任一不符 → 原样贴回,停。

**非 repo 文件(你自行传):** 步骤 3 要 `data/training_sets/spice_test_5000.extxyz.ids.dat`
(缺就先跑 `s0_spice_test_draw.py --n 5000`)与 SPICE release(`S0_TRAINING_SETS_ROOT`
指向它);runner 的 integration 组要 MACE-OFF23 权重(`data/potentials`,不在 git;缺了
先试 mace 的下载,网络不行就把权重文件传上去)。ZIP + `.git` 都不携带这些。

**回执**:三行 id + 分支行 + 两条 `status` 行。

## 步骤 1 — 05e 式验收(登录节点;cpu 与 gpu 两个 env 各一遍)

`install.sh` 用 local-path 模式(离线);这是 §9 的委托命令按手执行
(`conda run -n <env> bash <pkg>/install.sh <mace>` 的等价物,通过激活达成),然后再跑
`check_fork(strict=True)` 与包测试 runner。

```bash
cd $R/openQHA-main
source ~/init_conda.sh
source /APP/u22/ai_x86/toolshs/setproxy.sh 172.16.31.200 3138     # pip 的构建步骤要 DNS
export OPENQHA_PARTITION=deimos    # 本侧:CN/A=deimos,AI=h100x;换侧先 unset S0_RUNS_ROOT
source hpc/env/common.sh

# --- cpu env (openqha): 抽签/判官侧
unset OPENQHA_ENV                 # tianhe.sh 只在 OPENQHA_ENV 未设置时按 ROLE 选环境
export OPENQHA_ROLE=cpu
source hpc/env/tianhe.sh
python -V; command -v python
bash ../openQHA-Hessian/install.sh ../mace
python -c "from openqha_hessian import run; print(run.check_fork(strict=True))"
( cd ../openQHA-Hessian && python tests/run_tests.py --all )

# --- gpu env (openqha-gpu): 训练侧
unset OPENQHA_ENV
export OPENQHA_ROLE=gpu
source hpc/env/tianhe.sh
python -V; command -v python
bash ../openQHA-Hessian/install.sh ../mace
python -c "from openqha_hessian import run; print(run.check_fork(strict=True))"
( cd ../openQHA-Hessian && python tests/run_tests.py --all )
```

预期(每个 env 都如此):

- `python -V` 是环境自己的解释器;`install.sh` 四步后打印
  `fork commit 1110ffb…`(无 `DIRTY`)、`mace_fork_info` 一行、`openqha_hessian … (package commit 411abe1)` 或当次,最后 `ok: the fork is installed editable and openqha-hessian imports`。
- `check_fork(strict=True)` 打印 info dict:`mace_fork_commit` 40-hex、
  `mace_fork_dirty: False`。
- runner 头部是 provenance(openqha / openqha_hessian / mace-torch 的路径与版本),
  末尾 `all 9 test(s) passed`。

任一 FAIL / traceback → 原样贴回,停。注意:probes 从 `$R/openQHA-main` 出发(它的
**上层** `$R` 里就有 `mace/`——别从那里跑,cwd 里的 `mace/` 会影子化 editable 安装;
`check_fork` 会指名这一点)。

**回执**:两个 env 的 `python -V`、install.sh 末尾块、check_fork 行、runner 末行。

## 步骤 2 — 08 残留清理(天河)

规则(spec Q5):**canonical 文件永不触碰**;stale fetch tarball 删除;任何 dataset 级
pre-rebuild "old extxyz" 副本先列出、逐条确认后删除。

```bash
ls -lh "$R"/*.tgz ~/*.tgz 2>/dev/null || echo "no tgz found"
D=$S0_RUNS_ROOT/draw300/_datasets/draw300_r1
du -sh "$D"; ls -l --time-style=long-iso "$D"
find $S0_RUNS_ROOT/draw300/_datasets -maxdepth 3 \( -name '*.extxyz' -o -name '*.h5' -o -name '*.tgz' \) \
     -printf '%TY-%Tm-%Td %TH:%TM  %10s  %p\n' | sort
```

预期:`draw300_r1/` 内五个 fetch 文件(merged extxyz、h5、index.dat、dataset.out、
dataset.toml)时间戳 = **2026-09-29**(merged 13:46–14:58 档;h5 re-export 20:15;sha256
见 08 号票 Answer 的五枚)。任何**早于 2026-09-29** 的 dataset 级副本、以及已回传后的
`~/draw300_r1_fetch.tgz` 都是候选:tar 直接删(已裁决),其余**先贴回候选清单再删**。

```bash
rm ~/draw300_r1_fetch.tgz          # 已回传的 stale tar
# rm <逐条确认过的 pre-rebuild 副本>   # 只删确认的;canonical 不动
```

**回执**:删除前的三段 listing + 实际删除的清单(条数与路径)。

## 步骤 3 — 两次 Replay draw(天河登录节点;9d 的执行物)

前置:步骤 0/1 已过(draw 要 `import openqha_hessian`;两 env 都已装)。全程在
`$R/openQHA-main`;用 cpu 环境。

```bash
cd $R/openQHA-main
export OPENQHA_PARTITION=deimos    # 本侧:CN/A=deimos,AI=h100x
unset S0_RUNS_ROOT                 # 解析只在未设时重算;换侧/改分区后必须清
unset OPENQHA_ENV
export OPENQHA_ROLE=cpu
source ~/init_conda.sh
source hpc/env/common.sh && source hpc/env/tianhe.sh

# --- 3.0 preflight:解析三个输入(任一 MISSING -> 停,原样贴回)
python - <<'PY'
from pathlib import Path
from openqha.data import training_set, dataset
from openqha_hessian import judge
s = training_set.settings()
for name, p in [("source", Path(s["train"])),
                ("forgetting-ids", Path("data/training_sets/spice_test_5000.extxyz.ids.dat")),
                ("membership", Path(dataset.MEMBERSHIP_FILE))]:
    print(name, "OK    " if p.is_file() else "MISSING", p)
print("in_distribution:", ", ".join(judge.IN_DISTRIBUTION))
PY

# --- 3.1 the two draws(恰好这两条命令;同 seed、同一排列,只差落盘 weight)
python scripts/tooling/s0_spice_pt_draw.py --n 30000 --seed 0 --weight 1  --out $S0_RUNS_ROOT/spice/spice_pt_replay30k_w1.extxyz
python scripts/tooling/s0_spice_pt_draw.py --n 30000 --seed 0 --weight 10 --out $S0_RUNS_ROOT/spice/spice_pt_replay30k_w10.extxyz

# --- 3.2 产物(存在性 + 来源指纹;两份 extxyz/ids 的 sha256 必不相等,这不是相等证据)
ls -lh $S0_RUNS_ROOT/spice/spice_pt_replay30k_w*
sha256sum $S0_RUNS_ROOT/spice/spice_pt_replay30k_w1.extxyz $S0_RUNS_ROOT/spice/spice_pt_replay30k_w10.extxyz \
          $S0_RUNS_ROOT/spice/spice_pt_replay30k_w1.extxyz.ids.dat $S0_RUNS_ROOT/spice/spice_pt_replay30k_w10.extxyz.ids.dat

# --- 3.3 验证(天河本地跑;30k 行不取回):Records 只差 {WEIGHT,FILE,IDS_FILE,VALID_FILE,SECONDS},
#     ids 行逐行相等(整文件不等:header 有 `# weight` 行)
python - <<'PY'
import os
from pathlib import Path
from openqha.store import dat, property as prop
sp = Path(os.environ["S0_RUNS_ROOT"]) / "spice"
w = "spice_pt_replay30k_w{}.toml"
a = prop.load(sp / w.format(1))["Replay"]; b = prop.load(sp / w.format(10))["Replay"]
skip = {"WEIGHT", "FILE", "IDS_FILE", "VALID_FILE", "SECONDS"}
differ = sorted(k for k in a if a[k] != b[k])
print("WEIGHT", a["WEIGHT"], "vs", b["WEIGHT"], "| fields differing:", differ,
      "| UNEXPECTED:", [k for k in differ if k not in skip] or "none")
ra = dat.read_table(sp / "spice_pt_replay30k_w1.extxyz.ids.dat")
rb = dat.read_table(sp / "spice_pt_replay30k_w10.extxyz.ids.dat")
print("ids rows", len(ra), "vs", len(rb), "| row-for-row identical:", ra == rb)
PY
```

产物命名:每个 draw 写出 `….extxyz` + `….extxyz.ids.dat` + `….valid.extxyz`(200 帧)
+ `….toml`(Record 名**不带** `.extxyz`)。预期 stdout 摘要:`drawn 30000 frames with
seed 0, config_weight 1/10: … molecules` 与 `written …` 四行。

**拒收怎么读(exit 2、零写出、重跑安全):** 源缺 → 设 `S0_TRAINING_SETS_ROOT`(或
`--source`)后重跑;forgetting ids 缺 → **先** `python scripts/tooling/s0_spice_test_draw.py --n 5000`,
再原样重跑 3.1 两条;断言失败/合格帧不足 → 真发现,原样贴回即停。不预跑小 `--n` 试验
(扫描/分类是固定成本,省不了时间)。

**回执**:3.0/3.1/3.2/3.3 的全部输出 + 两份 `cat $S0_RUNS_ROOT/spice/spice_pt_replay30k_w{1,10}.toml` 全文。

---

## 回执清单(汇总)

1. 步骤 0:工作站与天河两侧 id、cleanliness、分支。
2. 步骤 1:两个 env 的 python/install.sh/check_fork/runner 输出。
3. 步骤 2:删除前 listing + 删除清单。
4. 步骤 3:四段输出 + 两份 `.toml` 全文。

回执到齐后:09d 的 Answer(两 Record、验证结论、sha256、时间/拒收记录)、本文件
`状态`、map 两行(09e/09d)、单次提交;套件 `--all` 73/73 与两轴 review 批注随本次提交并入。
