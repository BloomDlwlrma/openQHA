# 09d: The two 30k draws -- one seed, weights 1 and 10

Type: task
Status: resolved
Blocked by: 09a.
Serves: [09](../decisions/09-round-1-run.md) · spec: [spec-round-1-run.md](../spec-round-1-run.md).

**What to build:** On the Tianhe visit, the production Replay is drawn twice from
SPICE's train split — `--n 30000 --seed 0 --weight 1` and `--weight 10` — so the two
frame sets are identical by construction and only `config_weight` differs. Each draw's
`[Replay]` Record comes back, and the visit checklist carries exactly these two
commands.

- [x] Both files exist on Tianhe with their `.ids.dat`, `.valid.extxyz` and `[Replay]`
      Records: `spice_pt_replay30k_w1.extxyz`, `spice_pt_replay30k_w10.extxyz`.
- [x] The two Records differ only in WEIGHT (same SEED / N / N_MOLECULES / exclusions;
      the ids rows identical) — verified and reported. 实测差集 = {WEIGHT, FILE,
      IDS_FILE, VALID_FILE, SECONDS}(后四者按设计不同;`UNEXPECTED: none`);
      `ids rows 30000 vs 30000 | row-for-row identical: True`。
- [x] The paste-back receipts (both Records + the drawn summary lines) are in this
      ticket's Answer and the runbook's step list.

## Answer (2026-09-30, executed on Tianhe; recorded in this commit)

CN 侧(ln233)CPU 环境,两条命令原样执行:

    python scripts/tooling/s0_spice_pt_draw.py --n 30000 --seed 0 --weight 1  --out $S0_RUNS_ROOT/spice/spice_pt_replay30k_w1.extxyz
    python scripts/tooling/s0_spice_pt_draw.py --n 30000 --seed 0 --weight 10 --out $S0_RUNS_ROOT/spice/spice_pt_replay30k_w10.extxyz

**两条 stdout**(w10 的 source/excluded 与 w1 相同,`config_weight` 与路径不同):

    source     /XYFS02/HDD_POOL/hku2021_fos4/hku2021_fos4xy_2/sherwin/openQHA-main/data/training_sets/mace-off23_spice/train_large_neut_no_bad_clean.xyz (951005 frames; 526484 eligible after the exclusions)
    excluded   3416 forgetting-draw molecules, 4 in_distribution; skipped 420135 + 0 frames of the permutation, 4386 unparsed
    drawn      30000 frames with seed 0, config_weight 1.0: 10580 molecules, at most 10 frames of one
    written    /XYFS02/HDD_POOL/hku2021_fos4/hku2021_fos4xy_2/sherwin/runs/spice/spice_pt_replay30k_w1.extxyz
               /XYFS02/HDD_POOL/hku2021_fos4/hku2021_fos4xy_2/sherwin/runs/spice/spice_pt_replay30k_w1.extxyz.ids.dat
               /XYFS02/HDD_POOL/hku2021_fos4/hku2021_fos4xy_2/sherwin/runs/spice/spice_pt_replay30k_w1.valid.extxyz (200 frames)
               /XYFS02/HDD_POOL/hku2021_fos4/hku2021_fos4xy_2/sherwin/runs/spice/spice_pt_replay30k_w1.toml

    # w10(source/excluded 两行同 w1;drawn/written 仅 w1→w10、config_weight 10.0):
    source     /XYFS02/HDD_POOL/hku2021_fos4/hku2021_fos4xy_2/sherwin/openQHA-main/data/training_sets/mace-off23_spice/train_large_neut_no_bad_clean.xyz (951005 frames; 526484 eligible after the exclusions)
    excluded   3416 forgetting-draw molecules, 4 in_distribution; skipped 420135 + 0 frames of the permutation, 4386 unparsed
    drawn      30000 frames with seed 0, config_weight 10.0: 10580 molecules, at most 10 frames of one
    written    /XYFS02/HDD_POOL/hku2021_fos4/hku2021_fos4xy_2/sherwin/runs/spice/spice_pt_replay30k_w10.extxyz
               /XYFS02/HDD_POOL/hku2021_fos4/hku2021_fos4xy_2/sherwin/runs/spice/spice_pt_replay30k_w10.extxyz.ids.dat
               /XYFS02/HDD_POOL/hku2021_fos4/hku2021_fos4xy_2/sherwin/runs/spice/spice_pt_replay30k_w10.valid.extxyz (200 frames)
               /XYFS02/HDD_POOL/hku2021_fos4/hku2021_fos4xy_2/sherwin/runs/spice/spice_pt_replay30k_w10.toml

**3.0 preflight(原文)**:

    source OK     /XYFS02/HDD_POOL/hku2021_fos4/hku2021_fos4xy_2/sherwin/openQHA-main/data/training_sets/mace-off23_spice/train_large_neut_no_bad_clean.xyz
    forgetting-ids OK     data/training_sets/spice_test_5000.extxyz.ids.dat
    membership OK     /XYFS02/HDD_POOL/hku2021_fos4/hku2021_fos4xy_2/sherwin/openQHA-main/data/training_sets/qm9_targets_membership.dat
    in_distribution: dsgdb9nsd_000018, dsgdb9nsd_000019, dsgdb9nsd_000035, dsgdb9nsd_000036

**产物与指纹**(`$S0_RUNS_ROOT/spice/` = `/XYFS02/HDD_POOL/hku2021_fos4/hku2021_fos4xy_2/sherwin/runs/spice/`;大小 234 MB / 8.5 MB / 1.6 MB / 2.8 KB):

| 文件 | sha256 |
|---|---|
| `spice_pt_replay30k_w1.extxyz` | `46fc9e00dbc8931dfa888508479fcbf8a74af29683f123431f0d49dcba087cce` |
| `spice_pt_replay30k_w10.extxyz` | `df11f7da281b2c65de1944af964e934b26cd19cbcf92209addc426fd947a1a1d` |
| `spice_pt_replay30k_w1.extxyz.ids.dat` | `1aff09309b226e00b260461cf9b01085751080ff4560c2f4be13df0bd6d1e117` |
| `spice_pt_replay30k_w10.extxyz.ids.dat` | `f3778e27daf72fafd44661a660eb7d9cc9cd3546c3ec28f3561a1a62e89d6334` |

(四枚两两不等——文件本就不同;相等证据是下面的逐行比对。)

**验证(3.3)**:`WEIGHT 1.0 vs 10.0 | fields differing: ['FILE', 'IDS_FILE', 'SECONDS', 'VALID_FILE', 'WEIGHT'] | UNEXPECTED: none`;`ids rows 30000 vs 30000 | row-for-row identical: True`。

**两份 `[Replay]` Record(全文)**:

```toml
[Calculation_Status]
PROGNAME = "openQHA spice_pt_draw"  # String: the step that wrote this file
VERSION  = "0.3.0"                  # String: openQHA version
STATUS   = "NORMAL TERMINATION"     # String: the completion marker

[Replay]
DOI                       = "10.17863/CAM.107498"  # String: the SPICE release's DOI
SPLIT                     = "train"             # String: train: the file the frames come from
SOURCE                    = "/XYFS02/HDD_POOL/hku2021_fos4/hku2021_fos4xy_2/sherwin/openQHA-main/data/training_sets/mace-off23_spice/train_large_neut_no_bad_clean.xyz"  # String: the source file
N_SOURCE                  = 951005              # Integer: frames in the source file
SEED                      = 0                   # Integer: the permutation's seed
N                         = 30000               # Integer: frames written (the first N eligible of the permutation)
WEIGHT                    = 1.0                 # Double: config_weight on every frame
N_MOLECULES               = 10580               # Integer: distinct molecules (connectivity keys) in the draw
FRAMES_PER_MOLECULE_MAX   = 10                  # Integer: the most frames one molecule contributes
N_VALID                   = 200                 # Integer: frames in the companion validation file (from the end of the permutation)
KEY_LEVEL                 = "connectivity"      # String: the identity level of the exclusions
FORGETTING_IDS            = "/XYFS02/HDD_POOL/hku2021_fos4/hku2021_fos4xy_2/sherwin/openQHA-main/data/training_sets/spice_test_5000.extxyz.ids.dat"  # String: the forgetting draw's ids file the molecules were excluded against
N_FORGETTING_MOLECULES    = 3416                # Integer: molecules of the forgetting draw
N_IN_DISTRIBUTION         = 4                   # Integer: in_distribution molecules excluded
N_SKIPPED_FORGETTING      = 420135              # Integer: frames of the permutation skipped for a forgetting-draw molecule
N_SKIPPED_IN_DISTRIBUTION = 0                   # Integer: frames skipped for an in_distribution molecule
N_SKIPPED_UNPARSED        = 4386                # Integer: frames skipped because RDKit could not read their SMILES (no identity, no promise)
N_ELIGIBLE                = 526484              # Integer: frames of the source that could have been drawn
FILE                      = "/XYFS02/HDD_POOL/hku2021_fos4/hku2021_fos4xy_2/sherwin/runs/spice/spice_pt_replay30k_w1.extxyz"  # String: the Replay file
IDS_FILE                  = "/XYFS02/HDD_POOL/hku2021_fos4/hku2021_fos4xy_2/sherwin/runs/spice/spice_pt_replay30k_w1.extxyz.ids.dat"  # String: the ids file
VALID_FILE                = "/XYFS02/HDD_POOL/hku2021_fos4/hku2021_fos4xy_2/sherwin/runs/spice/spice_pt_replay30k_w1.valid.extxyz"  # String: the companion validation file
SECONDS                   = 242.4443085193634   # Double, s: wall time
```

```toml
[Calculation_Status]
PROGNAME = "openQHA spice_pt_draw"  # String: the step that wrote this file
VERSION  = "0.3.0"                  # String: openQHA version
STATUS   = "NORMAL TERMINATION"     # String: the completion marker

[Replay]
DOI                       = "10.17863/CAM.107498"  # String: the SPICE release's DOI
SPLIT                     = "train"             # String: train: the file the frames come from
SOURCE                    = "/XYFS02/HDD_POOL/hku2021_fos4/hku2021_fos4xy_2/sherwin/openQHA-main/data/training_sets/mace-off23_spice/train_large_neut_no_bad_clean.xyz"  # String: the source file
N_SOURCE                  = 951005              # Integer: frames in the source file
SEED                      = 0                   # Integer: the permutation's seed
N                         = 30000               # Integer: frames written (the first N eligible of the permutation)
WEIGHT                    = 10.0                # Double: config_weight on every frame
N_MOLECULES               = 10580               # Integer: distinct molecules (connectivity keys) in the draw
FRAMES_PER_MOLECULE_MAX   = 10                  # Integer: the most frames one molecule contributes
N_VALID                   = 200                 # Integer: frames in the companion validation file (from the end of the permutation)
KEY_LEVEL                 = "connectivity"      # String: the identity level of the exclusions
FORGETTING_IDS            = "/XYFS02/HDD_POOL/hku2021_fos4/hku2021_fos4xy_2/sherwin/openQHA-main/data/training_sets/spice_test_5000.extxyz.ids.dat"  # String: the forgetting draw's ids file the molecules were excluded against
N_FORGETTING_MOLECULES    = 3416                # Integer: molecules of the forgetting draw
N_IN_DISTRIBUTION         = 4                   # Integer: in_distribution molecules excluded
N_SKIPPED_FORGETTING      = 420135              # Integer: frames of the permutation skipped for a forgetting-draw molecule
N_SKIPPED_IN_DISTRIBUTION = 0                   # Integer: frames skipped for an in_distribution molecule
N_SKIPPED_UNPARSED        = 4386                # Integer: frames skipped because RDKit could not read their SMILES (no identity, no promise)
N_ELIGIBLE                = 526484              # Integer: frames of the source that could have been drawn
FILE                      = "/XYFS02/HDD_POOL/hku2021_fos4/hku2021_fos4xy_2/sherwin/runs/spice/spice_pt_replay30k_w10.extxyz"  # String: the Replay file
IDS_FILE                  = "/XYFS02/HDD_POOL/hku2021_fos4/hku2021_fos4xy_2/sherwin/runs/spice/spice_pt_replay30k_w10.extxyz.ids.dat"  # String: the ids file
VALID_FILE                = "/XYFS02/HDD_POOL/hku2021_fos4/hku2021_fos4xy_2/sherwin/runs/spice/spice_pt_replay30k_w10.valid.extxyz"  # String: the companion validation file
SECONDS                   = 222.48331236839294  # Double, s: wall time
```

**事实与算式**:N_ELIGIBLE 526484 = 951005 − 420135(跳过 forgetting 帧) − 4386(unparsed);
两条同 seed 0、同一排除集(3416 forgetting 分子 + 4 个 in_distribution);10580 分子、
FRAMES_PER_MOLECULE_MAX 10;排除键 = connectivity。时长 242.4 s / 222.5 s。**拒绝/墙**:无。

**检查**:openQHA 套件 `--all` **73/73**(60 unit / 11 integration / 2 regression;`all 73
test(s) passed`,EXIT 0,约 7 分钟;日志 `C:\Users\10704\AppData\Local\Temp\oqt09d-suite.log`)。
更改文件:本票、[09e](09e-the-round-1-visit-runbook.md)、`map.md`(两行);**无运行代码改动**。
**未验证**:Tianhe 盘上内容以贴回回执为准(本工作站未再触盘);两轴 review(Standards/Spec
子代理)已过本切片,批注随本提交并入。

**Carried**:提交随带 `map.md` 的一条既有未提交编辑(时间戳 13:16,晚于 HEAD `5b0b5b9`;
内容 = 移除 decision-12 行与 12h 实现行,**非本切片改动**)——按 09a 先例原样带上,提交信息有注。
