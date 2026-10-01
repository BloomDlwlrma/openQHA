# 09g: The timing and production runbook -- one measurement, two 24 h arms

Type: task
Status: claimed
Blocked by: None (the visit's acceptance is green -- [09e](09e-the-round-1-visit-runbook.md)).
Serves: [09](../decisions/09-round-1-run.md) · spec: [spec-round-1-run.md](../spec-round-1-run.md).

> Session artifact, 2026-09-30, delivered in chat and saved here. The Tianhe steps are run
> by the user; the workstation reconciles the paste-back into [09](../decisions/09-round-1-run.md)'s
> Answer. 范围 = spec stories 12-14 (the timing job, the cap, the two production
> submissions) + 15-17 (the done bar) + 18 (the evidence fetch-back).
> 范围:**tianhexy-ai 侧**(a800x)`yhbatch` 三条作业 -- timing(1 条)→ cap → 生产(2 条,一起交)。
> 不在本清单:**工作站侧**的数据集镜像/本地 gate(09f)、round 2、judge。
>
> **修订 2026-10-01(ticket 15 -- the balance on the probe estimator):balance 改按运行的 probe
> 设置估计** -- 不再是全矩阵一次过整个 train 文件;分钟级,固定头 = split + probe-balance + 两锚点。
> **timing 作业在改动部署到两侧 checkout 之前保持暂缓(held,user 裁定 2026-10-01)**;部署后按本
> 清单原样提交(提交块不变)。被替换的原文以删除线保留(步骤 2);Record 与 balance 行新增
> `BALANCE_PROBE` / `BALANCE_N_PROBES`,回执 2 一并读回。
>
> **修订 2026-10-01(二,yhbatch 的 NAME 注入)**:站点包装脚本 `/usr/bin/yhbatch` 把每个作业的
> `NAME` 覆盖成 `yhbatch`(实测:显式导出也被覆盖;`TAG`/`RUN`/位置参数无恙;`--export=ALL,NAME=…`
> 可夺回但生产不采用,user 2026-10-01)。提交行的 Dataset 名改用 **`DSET=draw300_r1`**
> (`hl_train.slurm` 改读 `${DSET:-$TAG}`);本清单所有提交行不再出现 `NAME=`。

## 前置(登录 ln301;逐条执行,输出即回执 1)

09e 已绿(两侧 checkout ids 一致、验收三条链绿);数据集与两枚 Replay 在 `/XYFS02` 的 runs root 下(见下面的 export)。

```bash
export R=~/HDD_POOL/sherwin        # AI 侧 checkout 根 = /XYAIFS00/HDD_POOL/hku2021_fos4/hku2021_fos4xy_2/sherwin
export S0_RUNS_ROOT=/XYFS02/HDD_POOL/hku2021_fos4/hku2021_fos4xy_2/sherwin/runs    # 数据与产物都在 /XYFS02
export OPENQHA_PARTITION=a800x     # 登录节点两挂载都在;不设则根解析失败

# 三个 checkout 的 id(后两个的 .git 是训练记录身份行的来源)
for d in openQHA-main openQHA-Hessian mace; do printf '%-18s ' "$d"; git -C "$R/$d" rev-parse HEAD; done
git -C "$R/mace" branch --show-current        # 应为 openqha-hessian

# 数据落点核对(必须看到正确的落点)
ls "$S0_RUNS_ROOT/draw300/_datasets/draw300_r1/"   # 五个 canonical(dataset.out/.toml、index.dat、mace_*.extxyz、molecules-*.h5)+ train/
ls "$S0_RUNS_ROOT/spice/"                          # spice_pt_replay30k_w{1,10}.extxyz + .ids.dat/.valid.extxyz/.toml
```

- **执行侧** = tianhexy-ai 的 `a800x`(≠ TianheXY-A 的 `ai`;user, 2026-10-01)。提交形式:`yhbatch -p a800x --gpus=1 ...`;`--gpus` 必需、`--exclusive` 禁用;`set-XY-I.sh`/`-G` 是 `ai` 分区的入口,本侧不用。
- **三个 checkout** = `openQHA-main` / `openQHA-Hessian` / `mace` 的 git 工作树;两侧均有(`/XYFS02/HDD_POOL/hku2021_fos4/hku2021_fos4xy_2/sherwin/` 与 AI 侧 `$R`)。默认用 **AI 侧那套**提交(mace 与包的身份行从环境 `openqha-gpu` 的安装读出;09e 验收也是在这一侧绿的)。要连日志/注册副本也落 `/XYFS02`:把 `R` 换成 `/XYFS02/HDD_POOL/hku2021_fos4/hku2021_fos4xy_2/sherwin`(两树 id 相同;`openqha` 改从那棵树加载,包与 mace 仍来自环境)。

## 步骤 1 -- timing job(一次测量服务两臂)

生产旋钮、短 cap;`config_weight` 不改每 epoch 成本,所以用 `_w1` 文件跑一次即可。
**不注册**(测量,不是产品;`--register` 不在 EXTRA 里)。

```bash
cd $R/openQHA-main
export S0_RUNS_ROOT=/XYFS02/HDD_POOL/hku2021_fos4/hku2021_fos4xy_2/sherwin/runs
EXTRA='--lr 0.0001 --no-swa --mace-arg=--clip_grad --mace-arg=1.0 --mace-arg=--weight_decay --mace-arg=0.0 --mace-arg=--ema_decay --mace-arg=0.99999'
TAG=draw300 DSET=draw300_r1 RUN=timing1 MAX_EPOCHS=2 MULTIHEADS=1 EXTRA="$EXTRA" \
    PT_TRAIN_FILE=$S0_RUNS_ROOT/spice/spice_pt_replay30k_w1.extxyz \
    PT_VALID_FILE=$S0_RUNS_ROOT/spice/spice_pt_replay30k_w1.valid.extxyz \
    yhbatch -p a800x --gpus=1 -t 12:00:00 hpc/slurm/hl_train.slurm
```

注意 EXTRA 的三个 mace 旋钮每个是**两个 token**(`--mace-arg=--clip_grad --mace-arg=1.0`):
单 token 的 `--key=value` 能到 mace,但 run 的 `config.yaml` 只能读成裸 flag(本地 gate
发现;spec 故事 14 已按此改写)。

若 12 h 被墙切断(日志无收尾块):把 `-t` 提到 24:00:00 原样重交。

作业结束后从 `logs/slurm/openqha_hl_train_<jobid>.out` 的收尾读:
`epochs N in S s (S/E s per epoch)` 的 **SECONDS_PER_EPOCH**、`balance` 行
(`BALANCE_L_E/F/H` + `({BALANCE_PROBE} k={BALANCE_N_PROBES})` + 解出的 `HESSIAN_WEIGHT`)、`HESSIAN_CURVE_MOVED`。

**回执 2**:job id + 该输出块(从 `run timing1 ...` 到 epoch 表尾)+
`SECONDS_PER_EPOCH` + `HESSIAN_WEIGHT` + `BALANCE_PROBE`/`BALANCE_N_PROBES` + `HESSIAN_CURVE_MOVED`。

## 步骤 2 -- cap

```
cap = min(60, floor(0.9 * 86400 / SECONDS_PER_EPOCH))
```

(S/E 是 mace 训练循环自己的墙钟 ÷ epoch 数 -- balance 与两次锚点不在其中,它们是固定
头部(split + probe-balance + 两锚点),额外加在墙钟上(~~balance 是全矩阵一次过整个 train 文件~~
**2026-10-01 修订(ticket 15):balance 改按运行的 probe 设置估计 -- gaussian k=4、固定 seed,分钟级**)。
cap 照 spec 的公式取,另用 timing run 自己的数字做一次墙钟检查:`cap × S/E + 固定头 ≤ 0.9 × 86400`
(固定头 ≈ slurm 日志里 wrapper 的 `wall N s` − MAX_EPOCHS × S/E);超了就把 cap 下调到
满足。数字贴回后由工作站复核(2026-10-01 约定:复核确认的 `CAP=` 回发后才提交两臂)。)

(评审批注 2026-09-30:spec 故事 13 钉的是公式本身;这条墙钟检查是批注加入的下限保护,
只会在固定头会擑破 24 h 墙时下调 cap,两个数都贴回记录。)

**回执 3**:`SECONDS_PER_EPOCH` → 算出的 `CAP`。

## 步骤 3 -- 两条生产臂(一起交,各一条作业)

`EXTRA` 与 timing 相同,加上 `--register --register-copy`(每臂自注册,印出 ENGINES 条目;
注册必须在 run 上 -- 事后补 `05_train --register-copy` 会重新训练)。

```bash
cd $R/openQHA-main
export S0_RUNS_ROOT=/XYFS02/HDD_POOL/hku2021_fos4/hku2021_fos4xy_2/sherwin/runs
EXTRA='--register --register-copy --lr 0.0001 --no-swa --mace-arg=--clip_grad --mace-arg=1.0 --mace-arg=--weight_decay --mace-arg=0.0 --mace-arg=--ema_decay --mace-arg=0.99999'
CAP=<步骤 2 的数>
TAG=draw300 DSET=draw300_r1 RUN=replay30k_w1 MAX_EPOCHS=$CAP MULTIHEADS=1 EXTRA="$EXTRA" \
    PT_TRAIN_FILE=$S0_RUNS_ROOT/spice/spice_pt_replay30k_w1.extxyz \
    PT_VALID_FILE=$S0_RUNS_ROOT/spice/spice_pt_replay30k_w1.valid.extxyz \
    yhbatch -p a800x --gpus=1 -t 24:00:00 hpc/slurm/hl_train.slurm
TAG=draw300 DSET=draw300_r1 RUN=replay30k_w10 MAX_EPOCHS=$CAP MULTIHEADS=1 EXTRA="$EXTRA" \
    PT_TRAIN_FILE=$S0_RUNS_ROOT/spice/spice_pt_replay30k_w10.extxyz \
    PT_VALID_FILE=$S0_RUNS_ROOT/spice/spice_pt_replay30k_w10.valid.extxyz \
    yhbatch -p a800x --gpus=1 -t 24:00:00 hpc/slurm/hl_train.slurm
```

**回执 4**:两个 job id + 提交行原文。

## 步骤 4 -- 每条臂的 done bar(检查电池)

把检查脚本写到 `~`(不碰 checkout;只读 run 目录的产物),对每条臂跑。全部 `ok` 且末行
`PASS` 才算过。

```bash
cat > ~/oq09_check_train.py <<'PY'
"""Round-1 done bar: the check battery for one fine-tune run directory.

Usage: python oq09_check_train.py <run_dir> [expected_replay_weight]
Reads only on-disk artifacts (train.{out,toml,dat}, config.yaml, logs)."""
import sys
from pathlib import Path
from openqha.store import dat, property as prop

run_dir = Path(sys.argv[1])
exp_w = sys.argv[2] if len(sys.argv) > 2 else None
n_checks = n_fail = 0


def check(label, ok, detail=""):
    global n_checks, n_fail
    n_checks += 1
    if not ok:
        n_fail += 1
    print("  {:76s} {}".format(label, "ok" if ok else "FAIL " + str(detail)[:260]))


rec = prop.load(run_dir / "train.toml")["Calculation_Info"]
rows = dat.read_table(run_dir / "train.dat")
cfg = {}
for line in (run_dir / "config.yaml").read_text(encoding="utf-8").splitlines():
    k, _, v = line.partition(":")
    cfg[k.strip()] = v.strip()
log = "\n".join(p.read_text(errors="replace") for p in sorted((run_dir / "logs").glob("*.log")))

print("run dir: {}".format(run_dir))
check("record trio written (train.out/.toml/.dat) + config.yaml",
      all((run_dir / n).is_file() for n in ("train.out", "train.toml", "train.dat", "config.yaml")))
check("Record: LR = 0.0001 (explicit; the fork's forced value)", rec["LR"] == 0.0001, rec["LR"])
check("Record: EMA = True and SWA = False (Stage Two off)", rec["EMA"] is True and rec["SWA"] is False,
      (rec["EMA"], rec["SWA"]))
check("Record: HESSIAN_WEIGHT_RULE = balance and HESSIAN_WEIGHT > 0",
      rec["HESSIAN_WEIGHT_RULE"] == "balance" and rec["HESSIAN_WEIGHT"] > 0,
      (rec["HESSIAN_WEIGHT_RULE"], rec["HESSIAN_WEIGHT"]))
check("Record: HESSIAN_CURVE_MOVED = True", rec["HESSIAN_CURVE_MOVED"] is True)
check("Record: exact anchors 0 < AFTER < BEFORE",
      rec["VALID_HESSIAN_EXACT_BEFORE"] > 0 and 0 < rec["VALID_HESSIAN_EXACT_AFTER"] < rec["VALID_HESSIAN_EXACT_BEFORE"],
      (rec["VALID_HESSIAN_EXACT_BEFORE"], rec["VALID_HESSIAN_EXACT_AFTER"]))
check("Record: identity lines (fork 1110ffb..., HL_PACKAGE_VERSION/COMMIT)",
      str(rec["MACE_FORK_COMMIT"]).startswith("1110ffb") and bool(rec["HL_PACKAGE_VERSION"])
      and (rec["HL_PACKAGE_COMMIT"] == "unknown" or len(str(rec["HL_PACKAGE_COMMIT"])) == 40),
      (rec["MACE_FORK_COMMIT"], rec["HL_PACKAGE_VERSION"], rec["HL_PACKAGE_COMMIT"]))
check("config.yaml: clip_grad 1.0 / weight_decay 0.0 / ema_decay 0.99999",
      cfg.get("clip_grad") == "1.0" and cfg.get("weight_decay") == "0.0" and cfg.get("ema_decay") == "0.99999",
      {k: cfg.get(k) for k in ("clip_grad", "weight_decay", "ema_decay")})
check("config.yaml: lr 0.0001 / ema True / multiheads_finetuning True / ratio threshold 0.0",
      cfg.get("lr") == "0.0001" and cfg.get("ema") == "True" and cfg.get("multiheads_finetuning") == "True"
      and cfg.get("real_pt_data_ratio_threshold") == "0.0",
      {k: cfg.get(k) for k in ("lr", "ema", "multiheads_finetuning", "real_pt_data_ratio_threshold")})
check("config.yaml: no Stage Two keys", not [k for k in cfg if k == "swa" or k.startswith(("start_swa", "swa_"))],
      [k for k in cfg if k == "swa" or k.startswith(("start_swa", "swa_"))])
check("config.yaml: float64 / E0s foundation / REF keys / external loss",
      cfg.get("default_dtype") == "float64" and cfg.get("E0s") == "foundation"
      and cfg.get("energy_key") == "REF_energy" and cfg.get("forces_key") == "REF_forces"
      and cfg.get("hessian_key") == "REF_hessian" and cfg.get("loss") == "external"
      and cfg.get("loss_module") == "openqha_hessian.phl_loss:build",
      {k: cfg.get(k) for k in ("default_dtype", "E0s", "loss")})
check("log: the fork's multihead line ('... setting learning rate to 0.0001 and EMA to True')",
      "setting learning rate to 0.0001 and EMA to True" in log)
check("log: 'Param group 0: lr = 0.0001'", "Param group 0: lr = 0.0001" in log
      or "lr = 0.0001" in log, [l.strip() for l in log.splitlines() if "lr = " in l][:3])
valid = [r for r in rows if r["split"] == "valid"]
check("epoch table: >= 4 valid rows", len(valid) >= 4, [r["epoch"] for r in valid])
hcurve = [(r["epoch"], r["valid_hessian"]) for r in valid if r["valid_hessian"] is not None]
moved = len(hcurve) >= 2 and (max(v for _e, v in hcurve) - min(v for _e, v in hcurve)) > 1e-6 * max(abs(v) for _e, v in hcurve)
check("valid Hessian curve present, finite and moved", moved and rec["HESSIAN_CURVE_MOVED"] is True, hcurve)
check("all three validation terms on every valid row",
      all(r["valid_energy"] is not None and r["valid_forces"] is not None and r["valid_hessian"] is not None for r in valid),
      [(r["epoch"], r["valid_energy"]) for r in valid])
e0 = next((r["rmse_e_per_atom_meV"] for r in valid if r["epoch"] == -1), None)
check("soft E0s look: epoch -1 valid energy RMSE well under 500 meV/atom", e0 is not None and e0 < 500, e0)
if exp_w is not None:
    check("replay: 30,000 frames, config_weight {}, mace's head counts parsed".format(exp_w),
          rec["MULTIHEADS"] is True and rec["PT_N_FRAMES"] == 30000 and rec["PT_CONFIG_WEIGHT"] == exp_w
          and rec["PT_HEAD_TRAIN"] == 30000,
          (rec["MULTIHEADS"], rec["PT_N_FRAMES"], rec["PT_CONFIG_WEIGHT"], rec["PT_HEAD_TRAIN"]))
print("\n{} checks, {} failed".format(n_checks, n_fail))
print("PASS" if not n_fail else "FAIL: {} checks not ok".format(n_fail))
sys.exit(1 if n_fail else 0)
PY

# run the battery, once per arm (gpu env; from the checkout root)
cd $R/openQHA-main
export OPENQHA_PARTITION=a800x    # AI 侧;两挂载都在,登录节点必须显式设
source ~/init_conda.sh            # AI 侧环境载入(user, 2026-10-01);下面 tianhe.sh 据此激活 openqha-gpu
unset OPENQHA_ENV; export OPENQHA_ROLE=gpu
source hpc/env/common.sh && source hpc/env/tianhe.sh
export S0_RUNS_ROOT=/XYFS02/HDD_POOL/hku2021_fos4/hku2021_fos4xy_2/sherwin/runs
python ~/oq09_check_train.py $S0_RUNS_ROOT/draw300/_datasets/draw300_r1/train/replay30k_w1 1.0
python ~/oq09_check_train.py $S0_RUNS_ROOT/draw300/_datasets/draw300_r1/train/replay30k_w10 10.0
```

再加两条日志行证据(spec 故事 15),原样贴回:

```bash
grep -h "setting learning rate" $S0_RUNS_ROOT/draw300/_datasets/draw300_r1/train/replay30k_w1/logs/*.log
grep -h "Param group 0" $S0_RUNS_ROOT/draw300/_datasets/draw300_r1/train/replay30k_w1/logs/*.log
# 同两条对 replay30k_w10
```

**回执 5**:每臂 checker 全文输出 + 每臂两行 grep。

## 步骤 5 -- 跨臂线(w1 vs w10:coverage vs pull)

两条 Record 的同一批数并排贴回(工作站写成 09 的答案段):`HESSIAN_WEIGHT`(应两臂相同
-- 同一 train 文件、同一 probe/k/seed)、`VALID_HESSIAN_EXACT_BEFORE/AFTER` 与 `VALID_PROBE_OFFSET_RUN`、
各自 `train.out` 的 valid 曲线尾行、`N_EPOCHS`/`SECONDS_PER_EPOCH`。一句话读法:哪条臂把
目标(Hessian 项)压得更多 = coverage;哪条臂把 base 的 E/F 保持得更稳 = pull。

**回执 6**:上述数字两列。

## 步骤 6 -- 证据回传

```bash
D=$S0_RUNS_ROOT/draw300/_datasets/draw300_r1/train
for r in replay30k_w1 replay30k_w10 timing1; do
  tar -czf ~/09_${r}_evidence.tgz -C $D/$r train.out train.toml train.dat config.yaml
done
# 两枚模型(注册形式,来自 --register-copy)
ls -l $R/openQHA-main/data/potentials/mace_off23_draw300/
tar -czf ~/09_models.tgz -C $R/openQHA-main/data/potentials mace_off23_draw300
sha256sum ~/09_*_evidence.tgz ~/09_models.tgz
```

tar 三枚 + 模型 tar 走你的通道回工作站;工作站把它们解到镜像
`~/runs/openQHA/draw300/_datasets/draw300_r1/train/<run>/` 与本地注册表
`openQHA/data/potentials/mace_off23_draw300/`,然后收尾 09 的 Answer、写跨臂段。

**回执 7**:四枚 tar 的路径 + sha256 + 注册表 ls。

## 回执清单(汇总)

1. 三个 checkout id(AI 侧)。
2. timing1:job id + 输出块 + `SECONDS_PER_EPOCH` + `HESSIAN_WEIGHT` + `BALANCE_PROBE`/`BALANCE_N_PROBES` + `HESSIAN_CURVE_MOVED`。
3. cap 算式与结果。
4. 两条生产 job id + 提交行。
5. 每臂 checker 输出 + 两行日志。
6. 跨臂数字两列。
7. 回传 tar 路径 + sha256 + 注册表 ls。

Fail / 拒绝一律原样贴回停下;不要自行改旋钮重交。
