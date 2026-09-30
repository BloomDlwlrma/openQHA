# Production training parameters: MACE-OFF23, MACE-OFF23-SC, PHL, and the MACE upstream defaults

Findings note for the hessian-learn-framework effort ([09-round-1-run](../decisions/09-round-1-run.md)):
what the primary sources actually used for training, so the round-1 knob discussion compares
against the real thing, not paraphrase. Compiled 2026-09-30.

Method: read-only survey of the workspace's own copies, plus fetched arXiv pages and `pypdf`
extractions of the local PDFs. Every claim carries `path:line` (line numbers of the extracted
text for PDFs — extraction is not durable; the PDFs are). Where something was not verifiable,
it is marked NOT FOUND rather than guessed.

## 1. MACE-OFF23 — the script that trained the released models

Primary: `source-code/ref-papers/mace-training/mace-off-main/mace_off23/train_scripts/medium.sh`.
**Caution:** the shipped `small.sh` / `large.sh` are content-identical to `medium.sh` (every
line compared; all three carry `--name="SPICE_medium_neut_E0"`); the copy under
`lambda-qm9-reaction-deltan_0-mace_v1/related-papers/...` is identical too. So this is ONE
recipe; the released S/M/L architecture differences live in the paper's Table 2 (see below),
not in the repo.

Exact flags (line numbers from `medium.sh`):

    :4   --name="SPICE_medium_neut_E0"
    :9   --E0s="{35: ..., 6: ..., 17: ..., 16: ...}"     # explicit SPICE isolated-atom dict
    :10-15 --model="MACE" --num_interactions=2 --num_channels=128 --max_L=1 --correlation=3 --r_max=5.0
    :16-17 --forces_weight=1000 --energy_weight=40        # Stage-One weights (from scratch)
    :18-19 --weight_decay=5e-10 --clip_grad=1.0
    :20-21 --batch_size=128 --valid_batch_size=128
    :22-24 --max_num_epochs=190 --scheduler_patience=20 --patience=50
    :25    --eval_interval=1
    :26-27 --ema --swa                                    # --ema_decay not passed -> mace default 0.99
    :28-30 --start_swa=115 --swa_lr=0.00025 --swa_forces_weight=10
    :31-37 --num_workers=32 --error_table='PerAtomMAE' --default_dtype="float64" --device=cuda
           --seed=123 --restart_latest --save_cpu

Not passed: `--lr` (mace default 0.01), `--amsgrad` (fork default True), `--optimizer` (adam).
Effective Stage Two: start at 115/190 ≈ **60.5 %**, `swa_lr` = 0.01/40 = 2.5e-4 exactly,
Stage-Two weights **1000 (default) / 10 (script)** — not 1000/100.

Dataset facts (paper, fetched): arXiv 2312.15211 (v5; JACS version local at
`source-code/ref-papers/mace-training/jacs-MACE-OFF Short-Range Transferable Machine Learning
Force Fields.pdf`). Core training set = SPICE v1, molecule-level split, 95 % train+valid /
5 % test (ar5iv §II.2); ≈85 % of SPICE kept (ten elements, neutral, ion pairs removed);
augmented with QMugs 50–90-atom molecules and water clusters; COMP6 tripeptides test-only.
Table 2 (ar5iv): OFF23(S) cutoff 4.5 Å / 96 channels / max L 0; OFF23(M) 5.0 / 128 / 1;
OFF23(L) 5.0 / 192 / 1; OFF24(M) 6.0 / 128 / 1. Training-recipe pointer: "More information
about the training is provided in Section S1" (ar5iv) — the SI is not in the workspace; the
script above IS the workspace's primary recipe.

## 2. MACE-OFF23-SC — no recipe in the workspace

`lambda-qm9-.../related-papers/.../MACE-OFF23-SC-main/` holds only `README.md`, licence and
`MACE-OFF23-SC_swa.model` (binary). Paper = arXiv 2405.18171 / JACS "Computing Solvation Free
Energies…" (Moore, Cole, Csányi), local PDF extracted: they **augmented the MACE-OFF24
training set with synthetic softened-dimer curves and fitted the softcore potential from
scratch**, finding lower errors than transfer learning from the OFF checkpoints (extracted
text, "SC fitting context"). No lr/epochs/batch are stated for that fit in the article text.
Training flags: NOT FOUND.

## 3. PHL — Projected Hessian Learning (the curvature-supervision reference route)

Code: `source-code/final-workflow-design/hessian-train/PHL-main/` (duplicate under
`…/ref-papers/mace-training/hessian-traning/`). Paper: `PHL-arxiv-2026-Projected Hessian
Learning (PHL).pdf` = arXiv **2603.04523** ("Projected Hessian Learning: Fast Curvature
Supervision…", Rodriguez, Smith, Matin, Lubbers, Barros, Mendoza-Cortes) — confirmed by the
extracted first page; NOT `2604.01017` (that is an Equitrain/LoRA paper).

Notebook facts (`PHL_training.ipynb`; lines of the raw file): config `max_epochs=5000,
batch_size=400` (:96); loader split 80/10/10 (:107, :376); optimizer AdamW on weights (no
`lr=` → torch default **1e-3**) + SGD on biases `lr=1e-3` (:242, :473), both with
`ReduceLROnPlateau(factor=0.5, patience=100, threshold=0)` (:244-245); stop when
`lr < 1e-5` (:484-486); loss `E + 0.30·F + 0.09·HVP`, terms normalized /√N, /3N, /(3N)²
(:571-584); **k = 1** Gaussian Hutchinson vector per molecule, resampled every step (:292,
:546; one-hot variant commented out); seed 7289038; model = TorchANI `ANIModel`
(4 × 256-64-256-1), trained **from scratch**.

Paper facts (fetched arXiv 2603.04523; extracted text lines cited): "λF = 0.30 and λH = 0.09
were tuned to balance…" (line 382); datasets — RTP 35,087 geometries from 11,961 reactions,
IRC 34,248 from 600 trajectories, NMS 62,527 (lines 311-318); Appendix B (fetched): 5,000-epoch
cap never reached, AdamW+SGD **lr = 1e-4 both**, β₁ 0.9 β₂ 0.999 ε 1e-8, weight decay 1e-2,
batch 400, ensembles of five seeds; runtimes on A6000: E-F ≈ 4 s/epoch, E-F-HVP ≈ 13.3-13.6
s/epoch, full E-F-H ≈ 326 s/epoch (>24× the E-F cost); text also reports "700 epochs,
resulting in smoother and more stable convergence trajectories" (§4, line 400); SI:
"trained using mini-batch gradient descent with adaptive learning-rate schedules" (line 1322).
**Discrepancy flagged:** notebook AdamW effective lr = 1e-3 vs paper 1e-4 (everything else
agrees); unresolved which produced the published figures.

Caveat for us: PHL numbers come from a **from-scratch TorchANI** model with its own loss
normalizations; the λ values are not transferable to a MACE fine-tune as-is.

## 4. AD-Hessian paper (kept in the same folder)

`source-code/final-workflow-design/hessian-train/MACE-chemrxiv-2025-Automatic differentiation
(AD) Hessian.pdf` = "Beyond Numerical Hessians: Higher-Order Derivatives for MLIPs via
Automatic Differentiation" (Gönnheimer, Reuter, Margraf; 8 Jan 2025). It is a Hessian-
computation performance/accuracy study (AD vs numerical; pre-trained MACE-MP-0; heat
capacities of 31,000 porous materials; "all results achieved without any fine-tuning").
**Not a training-recipe source** — relevant here as the AD-Hessian cost/accuracy reference.

## 5. MACE upstream defaults (fork `0.3.16+openqha`) and README guidance

`mace/mace/tools/arg_parser.py` defaults: `--lr` 0.01 (:931); `--scheduler` ReduceLROnPlateau,
`--scheduler_patience` 50, `--lr_factor` 0.8 (:963-975); `--patience` **2048** (:1015-1018);
`--max_num_epochs` 2048 (:1012); `--batch_size` 10 (:926-928); `--weight_decay` 5e-7 (:942);
`--clip_grad` 10.0 (:1066); `--amsgrad` **True** (:957-961); `--swa` **False**, `--start_swa`
None, `--swa_lr` 1e-3 (:934-941, :978); `--ema` **False**, `--ema_decay` 0.99 (:1000-1009);
`--hessian_weight` 1.0, `--n_hessian_probes` 4, `--hessian_probe` gaussian (fork surface,
:795-819). When `--swa` is on with no `--start_swa`, mace sets `start_swa = max(1,
max_num_epochs // 4 * 3)` (`mace/tools/scripts_utils.py:755-756`) — the "3/4" is **mace's
fallback**, not the OFF23 script value.

`mace/README.md`: training example `--stage_two --start_stage_two=1200 / 1500` epochs (= 80 %,
"last ~20 %"); finetune guidance `--E0s="estimated"` (:118); float64 default (:120); the
"≈200 000 gradient updates" heuristic `epochs × N / batch` (:122); a small finetune example
(`--lr=0.01 --batch_size=2 --max_num_epochs=6 --ema --amsgrad`, :311-336).

## 6. Our driver vs the sources (comparison)

Ours (`openQHA-Hessian/openqha_hessian/run.py:81-89`, `:295-312`; `05_train.py:74-104`):
lr 0.01, scheduler_patience 20, patience 50, eval_interval 1, EMA on (decay = mace default
0.99), Stage Two on at `max(1, 3·max_epochs//4)`, `swa_lr = lr/40`, Stage-Two weights
1000/100, batch 4/4, seed 123, float64, `--E0s foundation`, w_H = balance.

| parameter | ours | OFF23 script | verdict |
|---|---|---|---|
| lr | 0.01 | not set → mace 0.01 | ✓ |
| scheduler_patience | 20 | 20 | ✓ (OFF23 overrides mace's 50) |
| early-stop patience | 50 | 50 | ✓ (mace default is 2048) |
| swa_lr | lr/40 = 2.5e-4 | 2.5e-4 | ✓ (ratio is OFF23's choice; mace default is 1e-3) |
| EMA | on, decay 0.99 | `--ema`, decay default | ✓ |
| **Stage-Two start** | **3/4 of epochs** | **115/190 ≈ 60.5 %** | ✗ — ours is mace's fallback |
| **Stage-Two weights** | **1000 / 100** | **1000 / 10** | ✗ — ours are mace defaults |
| **Stage-One weights** | **1 / 100** | **40 / 1000** | ✗ — ours are mace defaults (fine-tune convention) |
| weight_decay / clip_grad | 5e-7 / 10 (mace defaults) | 5e-10 / 1.0 | ✗ |
| batch | 4 (Hessian-loss memory) | 128 (from scratch) | expected difference |
| seed, dtype, eval_interval | 123, float64, 1 | same | ✓ |

So the driver comment "the base's recipe: …" is **partly** OFF23 (patience/scheduler/ratio/EMA)
and partly mace's *defaults* (Stage-Two start and both weight sets). Correcting the comment's
attribution is worth doing when the ticket next touches `run.py`; adopting any OFF23 value
itself is a round-1 knob decision (with the w_H stage-two rule noting that `swa_forces_weight`
10 would make Stage-Two w_H = w_H/10).

## 7. Explicit gaps

- OFF23 SI §S1 (the block the paper defers to for training detail) — not in the workspace.
- `small.sh`/`large.sh`: duplicates of `medium.sh` in the released repo; S/L architecture from
  the paper's Table 2 only.
- OFF23-SC training flags — not found anywhere local; the article describes the fit
  qualitatively, no numbers.
- PHL notebook-vs-paper lr (1e-3 vs 1e-4) unresolved.
- The AD-Hessian paper's folder-mates (`qha-hessian/*.pdf`) were not part of this question.
