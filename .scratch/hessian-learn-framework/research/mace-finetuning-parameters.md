# Fine-tuning MACE: parameter-setting schemes from the literature (EF and EFH)

Findings note for the hessian-learn-framework effort ([09-round-1-run](../decisions/09-round-1-run.md)):
how the literature sets fine-tuning parameters for MACE-family models, split into **EF** (energy +
forces [+ stress]) and **EFH** (energy + forces + Hessian / second derivatives). Companion to
[production-training-parameters.md](production-training-parameters.md), which covers the MACE-OFF23
from-scratch scripts, the PHL numbers and the mace defaults — this note is about *fine-tuning*.
Compiled 2026-09-30 from primary sources (official docs, papers, official examples); links inline.

## A. EF fine-tuning

### A1. The official MACE docs — three protocols, and a hyperparameter table (2026)

- Overview: https://mace-docs.readthedocs.io/en/latest/guide/finetuning.html — three protocols:
  **naive** (continue from foundation weights), **multihead replay** (target + replay data, separate
  heads), **LoRA** (adapters only). Warning: "experimental and under active development".
- **Guidance page** (the parameter source):
  https://mace-docs.readthedocs.io/en/latest/guide/finetuning_guidance.html — itself a summary of
  **Tompa et al., arXiv:2606.12704** ("Fine-tuning MLIP foundation models: strategies for accuracy
  and transferability"; data + checkpoints on HuggingFace, linked from the page).

Recommended starting hyperparameters (verbatim table on the guidance page):

| Method | Learning rate | EMA decay | Grad clip | Trainable params |
|---|---|---|---|---|
| From-scratch | `1e-2` | `0.99` | `10.0` | 100 % |
| **Naive** | `1e-3` | `0.999` | `1.0` | 100 % |
| Layer freezing | `1e-3` | `0.999` | `1.0` | ~5 % |
| **LoRA** (r = 4–64) | `1e-2` | `0.99` | `10.0` | 2.5–30 % |
| **Multihead / Pseudolabel** | `1e-4` | `0.9999` | `1.0` | 100 % |

Rules of thumb (same page, verbatim/near-verbatim):

- **Weight decay = 0** ("weight decay pulls parameters toward zero, i.e. away from the pretrained
  solution").
- **Constant loss weights, no two-stage schedule**: "the force-then-energy schedule common in
  from-scratch training *can destabilise* fine-tuning at the transition point"; suggested constant
  example `--energy_weight=10 --forces_weight=10`.
- **E0s**: prefer explicit isolated-atom DFT at the fine-tuning level of theory; else
  `--E0s="estimated"` (model-aware least-squares reestimation — recommended default); **never
  `"average"`**. Symptom checklist: initial dataset error > 500 meV/atom ⇒ E0 mismatch.
- **Multihead lr is applied automatically** (`1e-4`; `--force_mh_ft_lr=True` to override, "not
  recommended").
- Layer freezing: **freeze=5 is the preferable setting** (embedding + interactions frozen; product +
  readouts trainable); freeze=6 "too restrictive". LoRA+replay combined "worse than either alone".
- Selection logic: naive for a single system; multihead replay when breadth/OOD robustness or
  preservation of foundation behaviour matters; validate **beyond pointwise RMSE** (long MD, PES
  holes / repulsive wall, RDFs, barriers).
- Naive example command (overview page, verbatim): `--foundation_model="small"
  --multiheads_finetuning=False --valid_fraction=0.05 --energy_weight=10.0 --forces_weight=10.0
  --E0s="estimated" --lr=0.001 --weight_decay=0.0 --scaling="rms_forces_scaling" --batch_size=2
  --max_num_epochs=6 --ema --ema_decay=0.999 --amsgrad --clip_grad=1.0 --default_dtype="float64"
  --seed=3`. "The hyperparameters will be automatically extracted from the model" for a custom
  `--foundation_model=$path`.
- **Multihead replay page** (https://mace-docs.readthedocs.io/en/latest/guide/multihead_finetuning.html):
  `--pt_train_file` (special value `mp`); tips: `--num_samples_pt` "30000 is a good value";
  "number of epochs for convergence is between 10 and 30"; weight decay "5e-7 to 0.0"; loss example
  `energy_weight 1.0 / forces_weight 100.0`, `swa_energy_weight 10 / swa_forces_weight 100`
  (note: docs' own example still shows SWA weights, while the guidance says avoid the schedule).
  Code path: multihead mode auto-sets lr 0.0001 + EMA (`mace-main/mace/cli/run_train.py:204-212`,
  quoted in the local upstream copy).
- **LoRA page** (https://mace-docs.readthedocs.io/en/latest/guide/lora_finetuning.html):
  `--lora=True --lora_rank=4 --lora_alpha=1.0` (rank typical 2–16); example lr 0.005, EMA 0.995,
  clip 10, energy/forces 1.0/1.0; base weights auto-frozen; adapters merged at save.

### A2. MACE-MP-0 paper — the original fine-tuning experiments (arXiv:2401.00096v3)

- Fine-tuning is done on configurations "generated using MACE-MP-0, typically via molecular
  dynamics"; **"approximately 100 new configurations for each application"**; introduces
  **multi-head replay** to prevent forgetting; FT beats from-scratch in almost all cases.
- Application recipe (Appendix C.2): **replay 100,000 MPtrj configurations; lr 1e-4; EMA 0.9999;
  Adam; batch 16; 10 % validation**; trained on 1× H100.
- Level-of-theory upgrade task: 17,000 configs; energy MAE **scratch 0.543 eV → naive 0.162 eV →
  replay+heads 0.032 eV**; forgetting ratios (Fig. S65): replay ≈1.1–1.4 vs naive ≈4.0 (forces).
- Pretraining baseline for context (Methods): (λE,λF,λσ) = (1,10,10), Huber δ = 0.01; Adam AMSGrad
  lr 1e-3; EMA 0.99999; clip 100; 100 epochs; MPtrj ≈1.5M configs. Also: "After fine-tuning with
  higher weights for energies for an additional 50 epochs, the small model is able to achieve an
  energy MAE of 13 meV".

### A3. Kaur et al. — data-efficient FT for sublimation enthalpies (arXiv:2405.20217; Faraday Discuss. 256, 120 (2025))

- Base: **large MACE-MP-0**; strategy: **"continue training from the last checkpoint … using the
  same hyperparameters"** — no freezing, no LoRA; FT differs from scratch only by "a self-connection
  … only at the first layer".
- Data: 50–400 nested training structures (+100 val) from coarse-GGA NPT MD; 100/polymorph; RPA
  upgrade 75 configs. Density converges with **50–100 structures** (vs 400 from scratch); at 50
  structures energy RMSE ≈2 % of the validation spread and F ≈1 meV/Å.
- Targets include **stress**. Exact lr/epochs and E0s handling: NOT STATED ("same as pretrained").

### A4. Equitrain study — PEFT of MLIPs (Grandel, Benner, George; arXiv:2604.01017v2)

- Base: **MACE-MP-0b3**, 53 materials, **16 train + 4 val configs/material** (rattled/volume-scaled,
  relaxed with the base).
- Methods (§4.3): AdamW + ReduceLROnPlateau, **up to 200 epochs**; loss = weighted Huber
  (E/F/σ) = **(10, 100, 1000)**, δ = 0.01. All schemes lr **0.01**; the differentiator is weight
  decay: transfer learning `(lr, wd) = (0.01, 5e-3)`; multihead `(0.01, 5e-7)` + replay 100;
  **Equitrain `(0.01, 10)`** — a **full-rank LoRA** parameterization (rank eliminated), base frozen,
  weight decay on ΔW only ("proximal/trust-region" reading). "As few as 10 additional training
  structures already yield significant gains." LoRA rank/alpha not reported (full-rank by design).

### A5. Additional EF fine-tuning works (IDs)

- Barocaloric ammonium sulfate, arXiv:2606.25742 — naive + multihead replay of MACE-MPA-0; as few
  as 5–10 DFT configs.
- Prototype-guided latent alignment, arXiv:2605.29969 — data-efficient FT of MACE/MACE-OFF
  (energy MAE −18 % vs standard FT).
- Beyond Adam, arXiv:2512.05489 — optimizer study for MLIP FT (AdamW/ScheduleFree best; adds a
  brief second-order refinement stage).
- Tompa et al., arXiv:2606.12704 — the study behind A1.

### A6. Local (workspace) fine-tuning evidence

- `lambda-qm9-reaction-deltan_0-mace_v1/stage2-lambda-delta0-mace-learning/docs/mace-training/T02_MACE_Practice_II.ipynb`:
  recorded run — config (lines 4792–4809) `foundation_model "small"`, `multiheads_finetuning
  False`, E 1.0 / F 10.0, batch 10, `max_num_epochs 500`; log shows WeightedEnergyForcesLoss
  E1/F10 (line 4971) and **Stage Two at epoch 375 with weights 1000/100, swa lr 0.001** (line 4978)
  — i.e., a stage-two schedule despite the docs' FT guidance (and despite the guidance saying it
  can destabilise).
- `…/T05_MACE-OFF-SC_learned-softcore-reproduction.ipynb`: a **negative transfer result** — command
  `--foundation_model="MACE-OFF23_medium.model"` (lines 1010–1011), conclusion "transfer into new
  short-range physics fails" (lines 1018–1021); no numbers/protocol recorded (line 1482).
- Upstream `related-papers/…/MACE/mace-main/README.md:309-336`: older naive-FT example
  (`lr=0.01`, `E0s="average"`, weights 1.0/1.0, epochs 6, float32) — superseded by A1.
- `…/docs/mace-training/MACE-training-reference.txt`: **empty (0 bytes)**; the `refs/ref-papers/`
  duplicate is a link stub.

## B. EFH fine-tuning (energy + forces + Hessian)

### B1. PFT — Phonon Fine-tuning (Koker, Gangan, Kotak, Marian, Smidt; arXiv:2601.07742v4, ICML 2026) — the primary EFH fine-tuning scheme

https://arxiv.org/abs/2601.07742 — fine-tunes MLIPs by matching **energy Hessians to DFT force
constants from finite-displacement phonon calculations**; scales via **stochastically sampled
Hessian columns with a single HVP per step**; adds a **co-training scheme** against catastrophic
forgetting.

- Objective (Eq. 8): `L_PFT = λE·LE + λF·LF + λσ·Lσ + λΦ·LΦ`; **λΦ = 100** (swept 10/100/1000,
  kept 100); EF force constants are the target of LΦ, E>MAE, forces ℓ2.
- Hyperparameters (Table A.2): **lr 1e-4** (selected from {3e-3, 1e-3, 3e-4, 1e-4}), **AdamW**,
  wd 1e-3, **batch 16**, **200 epochs**, 1× A100; **co-training ratio K = 4** (4 co-train steps per
  phonon step) with co-train weights E/F/σ = 500/200/50; warmup 0/20 epochs.
- Data: MDR Phonon database, **8,510 structures / 301,414 displacement calculations** (95/5).
- **MACE application**: applied to **MACE-MP-0** "using the same hyperparameters as Nequix MP"
  (no co-train), **≈30 A100 h ≈ 1 % of pretraining cost**; third-order force-constant MAE for
  MACE-MP-0 improves 11.41 → 7.86 meV/Å³; thermal conductivity improves too.
- Ablation evidence: **naive FT on the same phonon displacements (E+F+stress only) *increases*
  Hessian error** — direct curvature supervision is the ingredient; co-training preserves upstream
  accuracy.

### B2. PHL — Projected Hessian Learning (arXiv:2603.04523) — from-scratch EF vs EFH reference

(Covered in the companion note; repeated here only as the EFH parameter scheme.) From-scratch
ANI-family training; loss `LE + λF·LF + λH·LH` with **λF = 0.30, λH = 0.09**; Hutchinson **k = 1**,
resampled per minibatch; AdamW+SGD lr 1e-4, wd 1e-2, batch 400, ≤5000 epochs; 5-seed ensembles.
Results vs E-F: −71–88 % Hessian RMSE; one-column vs Hutchinson indistinguishable when randomized
per minibatch; ~24× faster per epoch than full-Hessian training (13.6 s vs 326.5 s on A6000).

### B3. hippynn Hessian-training example (lanl/hippynn) — the reference implementation of B2

`…/source-code/ref-papers/mace-training/hessian-traning/hippynn-development/examples/hessian_training.py`
(upstream https://github.com/lanl/hippynn): `force_coefficient = 0.30`, `hessian_coefficient = 0.09`
(lines 124–134); loss = E-RMSE + 0.30·F-RMSE + 0.09·HVP-RMSE; HVP node uses a **random probe
vector**; full-Hessian loss present but commented out ("very slow"); per-element self-energies
subtracted (E0-analog); dataset OpenREACT-CHON-EFH. Companion HIP-NN paper parameters cited from
work "on ANI".

### B4. HIP — Hessian Interatomic Potentials (Burger et al., arXiv:2509.21624; local `hip-main/`)

From-scratch **direct Hessian prediction** (dedicated head, no autograd) on HORM (1.84M Hessians).
Parameterization includes a **`train_hessian_only`** flag ("freezes the backbone and only trains
the Hessian head" — the closest thing to "Hessian-head fine-tuning" in the set). `configs/train.yaml`:
muon lr 0.02 (AdamW alternative 3e-4), wd 1e-5/0.01, batch 128, loss weights **H 10 / E 1 / F 25
(MAE)**, ≤10000 epochs; `max_epochs` for QM9-Hessian presets. (Not fine-tuning; listed for the
EFH supervision scheme.)

### B5. Other second-order works (IDs)

- Hessian distillation (Amin et al., ICLR 2025, arXiv:2501.09009) — distillation via energy-Hessian
  columns/JVPs.
- Datasets: Hessian QM9 (Sci. Data 2025), HORM (1.84M), OpenREACT-CHON-EFH (PHL/hippynn).
- ELoRA (ICML 2025) — PEFT for SO(3)-equivariant GNNs (no Hessian terms).

## Comparison table (fine-tuning schemes)

| Source | Base model | Data size | LR | Epochs | Loss weights | Freeze/PEFT | Other | Link |
|---|---|---|---|---|---|---|---|---|
| MACE docs — naive | any foundation | user + 5 % val | 1e-3 | 6 (example) | E 10 / F 10 const | none | wd 0, EMA 0.999, clip 1.0, E0s estimated, batch 2 | [A1] |
| MACE docs — LoRA | medium_omat | user | 5e-3–1e-2 | 6 | E 1 / F 1 | LoRA r≈4 (2–16), α 1.0, base frozen | EMA 0.995, clip 10 | [A1] |
| MACE docs — multihead | foundation + replay | replay 30k suggested | **1e-4 auto** | 10–30 | E 1 / F 100 | replay head | EMA forced, wd 5e-7→0 | [A1] |
| MACE-MP-0 apps | MP-0 | ~100 configs/task; replay 100k | 1e-4 | n/s | const per task | none | EMA 0.9999, batch 16 | [A2] |
| Kaur et al. | MP-0 large | 50–400 (+100 val) | same as pretrain (n/s) | continue | E+F+stress | none | — | [A3] |
| Equitrain | MP-0b3 | 16+4/material; replay 100 | 0.01 | ≤200 | Huber E10/F100/σ1000, δ 0.01 | full-rank LoRA (wd 10 on ΔW) | AdamW+ReduceLROnPlateau | [A4] |
| **PFT (EFH)** | Nequix MP / **MACE-MP-0** | 8,510 structs / 301k displ. | **1e-4** | **200** | **λΦ = 100** (+co-train 500/200/50, K=4) | none; co-train replay | AdamW, wd 1e-3, batch 16, 1 HVP/step | [B1] |
| PHL (EFH, scratch) | ANI | 35k+34k+63k | 1e-4 | ≤5000 | E 1 / F 0.30 / H 0.09 | none | k=1 Hutchinson, batch 400 | [B2] |
| hippynn (EFH, scratch) | HIP-NN | OpenREACT-CHON-EFH | n/s | n/s | E 1 / F 0.30 / HVP 0.09 | none | random probe | [B3] |
| HIP (EFH, scratch) | EquiformerV2 | HORM 1.84M | muon 0.02 (AdamW 3e-4) | ≤10000 | H 10 / E 1 / F 25 MAE | `train_hessian_only` option | direct Hessian head, batch 128 | [B4] |

## What this means for our round-1 fine-tune (decision input, not action)

0. **Fork fact (checked 2026-09-30, `mace/mace/cli/run_train.py:204-210`):** in multihead mode
   without `--force_mh_ft_lr`, **our fork itself forces `args.lr = 0.0001`, `args.ema = True`,
   `args.ema_decay = 0.99999`** — every multihead run we have executed thus already trained at
   1e-4 / EMA 0.99999, while our own Record, `config.yaml` and printout say lr 0.01 / EMA 0.99
   (nominal ≠ effective). Round-1 fix: pass `--lr 0.0001` explicitly (the Record becomes true);
   mirroring mace's rule in `control_settings` is a small downstream ticket.

1. **Fine-tuning has its own lr convention** the nominal values did not follow: the 2026 guidance
   uses **1e-3 (naive) / 1e-4 (multihead replay)**, and MACE-MP-0 (1e-4) and PFT (1e-4) agree —
   our *effective* 1e-4 (see 0) matches them; the old nominal 0.01 matched nothing FT-specific.
2. **No stage-two schedule for fine-tuning** ("can destabilise fine-tuning at the transition
   point") vs our default EMA+Stage Two on. Our sweep/stage-two machinery remains available; the
   guidance suggests constant loss weights.
3. **Constant weights example E 10 / F 10** (naive) and E 1 / F 100 (multihead example) vs our
   E 1 / F 100 + `w_H = balance` — the w_H rule stays ours; nothing here overrides it.
4. **clip 1.0 and wd = 0** in the FT guidance vs our clip 10 / wd 5e-7 (mace defaults); EMA decay
   guidance 0.999–0.9999 vs our 0.99.
5. **EFH**: PFT is the first true EFH fine-tuning reference, and its ablation supports our design
   premise — EF-only FT degrades curvature; direct curvature supervision fixes it. Its budget
   shape (λΦ = 100, lr 1e-4, 200 epochs, K = 4 replay co-training, 1 HVP/step) is the closest
   analogue to our R4 row; note its weight is under its own loss normalization (not comparable to
   our `w_H = balance` value).

## Gaps

- `MACE-training-reference.txt` empty; T05 transfer-run numbers not recorded; SC training script unpublished.
- Kaur et al.: lr/epochs/E0s not stated (RSC version not checked; arXiv v1 used).
- Equitrain: LoRA rank/alpha not applicable (full-rank) and trainable counts not reported.
- PFT: no follow-up work found that applies PFT to MACE-OFF23 specifically.
- Docs guidance is new (2026) and "experimental"; numbers tie to Tompa et al. arXiv:2606.12704.

## Links

| # | Source | URL |
|---|---|---|
| A1 | MACE fine-tuning overview | https://mace-docs.readthedocs.io/en/latest/guide/finetuning.html |
| A1 | MACE fine-tuning guidance (hyperparameter table) | https://mace-docs.readthedocs.io/en/latest/guide/finetuning_guidance.html |
| A1 | MACE multihead replay | https://mace-docs.readthedocs.io/en/latest/guide/multihead_finetuning.html |
| A1 | MACE LoRA | https://mace-docs.readthedocs.io/en/latest/guide/lora_finetuning.html |
| A1 | Tompa et al. (study behind the guidance) | https://arxiv.org/abs/2606.12704 |
| A2 | MACE-MP-0 foundation model paper | https://arxiv.org/abs/2401.00096 |
| A3 | Kaur et al., Faraday Discuss. 256, 120 (2025) | https://arxiv.org/abs/2405.20217 |
| A4 | Equitrain / PEFT for MLIPs | https://arxiv.org/abs/2604.01017 |
| A5 | Barocaloric FT | https://arxiv.org/abs/2606.25742 |
| A5 | Prototype-guided latent alignment | https://arxiv.org/abs/2605.29969 |
| A5 | Beyond Adam (optimizers for FT) | https://arxiv.org/abs/2512.05489 |
| B1 | **PFT: Phonon Fine-tuning (ICML 2026)** | https://arxiv.org/abs/2601.07742 |
| B2 | PHL: Projected Hessian Learning | https://arxiv.org/abs/2603.04523 |
| B3 | hippynn Hessian training example | https://github.com/lanl/hippynn |
| B4 | HIP: Hessian Interatomic Potentials | https://arxiv.org/abs/2509.21624 · https://github.com/BurgerAndreas/hip |
| B5 | Hessian distillation | https://arxiv.org/abs/2501.09009 |
