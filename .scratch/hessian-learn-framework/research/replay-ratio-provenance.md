# Replay-ratio provenance: where "R4 = 4 × N_TRAIN_HESSIAN @ config_weight 10" comes from

Question (2026-09-30): "27,740 帧（带 Hessian 的 train）:110,960 帧（同语料抽签）=1:4 的文献来源在哪里？"

**Verdict.** As a replay-frame-count ratio there is **no literature source**. The 4× is a
campaign design value (user rulings S0-C-57 and S0-C-60, 2026-09-21/22). Its nearest
literature anchor is PFT's co-training ratio **K = 4** — an optimizer **step** ratio — read as a
dataset ratio because MACE's multihead loop concatenates the two heads' datasets (the
reinterpretation, and the mechanism mismatch, are both documented in-repo). The companion
`config_weight = 10` traces to PFT's upstream/phonon **force-weight** ratio (λ_F 200/20). The
current literature gives replay-size **guidance, not a ratio**: mace-docs "use as many replay
samples as you can (30000 is a good value)"; MACE-MP-0 used 100k replay frames against
~100-config targets; Tompa et al. (arXiv:2606.12704) explicitly leave optimal replay size and
composition to future work.

## 1. The two numbers are ours (not from literature)

- **27,740** = the train split's Hessian-bearing frames of the canonical `draw300_r1` rebuild
  (2026-09-29). Source: `decisions/08-round-1-xyz.md:39` —
  `"324978 frames: train 27740 valid 868 test 1371 pool 294999; 29979 with a Hessian"`
  (build log excerpt). This is what the Dataset Record field `N_TRAIN_HESSIAN` counts.
- **110,960** = 4 × 27,740. First appearance: `decisions/08-round-1-xyz.md:42` —
  `"R4 (S0-C-60): 27740 train frames with a Hessian -> Replay = 110960 frames at config_weight 10"`.
  It occurs nowhere else except as derived from this build line. (An earlier build printed
  63,004 = 4 × 15,751; superseded — `decisions/09-round-1-run.md:15`.)
- The replay **source** is SPICE's train split, drawn uniformly by frame, one seed file for the
  whole campaign (S0-C-56; drawn by `s0_spice_pt_draw.py`) — the "same-corpus draw".

## 2. Where the 4× was decided (in-repo chain)

1. Cost/coverage analysis, 2026-09-21 — `.scratch/hessian-learning-set/grilling-round-10-replay-cost.md:31`:
   `"| 68,000 (PFT's K = 4) | 4.0 |"`; `:51-52`: `"4 frames per molecule = 68,528 frames, exactly the
   PFT-equivalent size; 1 per molecule = 17,132; 5,000 frames = 5,000 molecules x 1."` — the stated
   basis is "PFT-equivalent".
2. User ruling **S0-C-57** — `.mem/decisions/decisions_C_hessian.md:104`: the five-row ladder
   `"R0 无回放；R1 5,000 帧；R2 17,132 帧（每 Hessian 帧约 1）；R3 68,528 帧（约 4，PFT K=4 等价）；
   R4 = R3 的帧 + 回放帧 config_weight = 10（PFT 上游步力权系数比）"`; rationale:
   `"config_weight 取 10 而非 4：PFT A.4 上游步 λ_F 200 对声子步 20（λ_E 0 对 500 无比值），
   mace 的 config_weight 同乘能量与力项，只能对应力项的 10；4 无依据。"` — here "4 had no basis"
   refers to the **weight** value 4, not the frame ratio.
3. Round-10 record — `grilling-round-10-replay-cost.md:82-83`: `"R0 = 0; R1 = 5,000; R2 = 17,132
   (~1 per Hessian frame); R3 = 68,528 (~4, PFT's K = 4); R4 = R3's frames with config_weight = 10
   (PFT's upstream/phonon force-weight ratio; 4 had no basis)."`
4. User ruling **S0-C-60**, 2026-09-22 — `decisions_C_hessian.md:107`: `"生产级别只训练一行：R4——
   回放帧数 = 第一阶段已标注 Hessian 的训练帧数 x 4，回放帧 config_weight = 10，w_H 取 epoch-0
   平衡值；…"` (single production row; no scan).
5. Rule written into the S0 spec — `.scratch/hessian-learning-set/spec-fine-tune-basins.md:69`:
   `"R4 alone is trained -- Replay = 4 x N_TRAIN_HESSIAN, the number the Dataset Record prints as
   REPLAY_R4_FRAMES, at config_weight 10; R0-R3 stay defined and unrun."`
6. The mechanism transfer is documented as a **reinterpretation** —
   `.scratch/hessian-learning-set/spec-phl-verbatim.md:154-159`:
   `"**Derivation 0.4 (R4's arithmetic).** With N_H labelled train frames, Replay = 4N_H frames at
   config_weight = 10 … The two numbers are PFT's (K=4 replay structures per phonon structure,
   upstream E/F at 10x), read as dataset ratios because mace's multihead loop sees every frame of
   either head once per epoch (--num_samples_pt is not read with a user file; the duplication
   threshold is passed as 0)."`
7. And the mismatch is explicit — `.scratch/hessian-learning-set/issues/15-the-smoke-fit.md:29-33`:
   `"mace 0.3.16's --multiheads_finetuning is **not** PFT's algorithm 1. PFT alternates by STEP
   (K = 4 upstream E/F steps per Hessian step); mace concatenates the two heads' training sets into
   one ConcatDataset and shuffles it … so every replay frame and every fine-tuning frame appears
   exactly once per epoch and the ratio is set by DATASET SIZE, not by a step loop."`
8. Code carries it: `openqha/data/dataset.py:137-139` (`REPLAY_PER_HESSIAN_FRAME_R4 = 4`,
   `REPLAY_CONFIG_WEIGHT_R4 = 10.0`), `:249` (`"REPLAY_R4_FRAMES": (…, "4 x N_TRAIN_HESSIAN: …")`);
   `openQHA-Hessian/openqha_hessian/smoke_fit.py:99-106` (`"the number to compare with PFT's 4."`,
   `PFT_REFERENCE = 4.0`); `run.py:30`.
9. Current effort text: `spec-round-1-xyz.md:35-37`, `decisions/09-round-1-run.md:15`,
   `spec-round-1-run.md:13,129-136` ("R4 = 110,960 frames (4×N_TRAIN_HESSIAN of the canonical
   rebuild) at `config_weight` 10").

## 3. What PFT actually says (primary source, arXiv:2601.07742v4)

- **Co-training is a step interleave** (Algorithm 1): per phonon batch — `Update θ using L_PFT;
  for k = 1 to K: Sample batch from D_up; Update θ using L_EFS`.
- `"We perform PFT both with (K=4) and without (K=0) co-training for 200 epochs"` (§4.1).
- Appendix A.3.2 hyperparameters: `"Co-train ratio 4 … Trade off between training time and
  overfitting to phonon data; 4 co-training steps for every phonon step is reasonable."` —
  a **step** ratio.
- Loss weights (same table): co-train λ_F = **200**; PFT (phonon-step) λ_F = **20** → ratio **10**.
  Co-train λ_E = 500 (started from 10, `"increased until co-train validation energy does not
  diverge"`); λ_Φ = 100.
- So: **4 = steps, 10 = force-weight ratio** in PFT. Neither is a replay-frames : target-frames
  ratio. Ours is the dataset-ratio translation of the first, under MACE's concatenation mechanism —
  an approximation, documented as such (items 6-7 above).

## 4. What the current MACE-ecosystem literature says about replay size

- **mace-docs, "Multihead Replay Finetuning"** (`https://mace-docs.readthedocs.io/en/latest/guide/multihead_finetuning.html`):
  `"**Dataset size ratio**: It usually gives best performance to use as many replay sample as you
  can. Use --num_samples_pt to control this (30000 is a good value). As your input data will be
  repeatedly sampled, you can use a smaller number of epochs."` — an absolute-size
  recommendation (≥ ~30k), no 4× ratio; plus convergence 10-30 epochs and the auto lr 1e-4 / EMA
  behavior.
- **Tompa et al., arXiv:2606.12704** (the study behind mace-docs "Fine-tuning Guidance"):
  replay sets are element-matched foundation-data subsets with benchmark-specific sizes —
  2,912 (NaCl), 3,706 (SN2), 196 (ice), 39,459 (SPICE), 10,000 random MPTraj (Li). Hyperparameters:
  multihead lr 1e-4, EMA 0.9999, clip 1.0, wd 0; replay-head weights λ_E = 1, λ_F = 10
  (pretraining objective), target λ_E = 10, λ_F = 10. Cost: multihead ≈3-15× naive compute.
  Crucially, the paper defers the ratio question: `"what is the optimal composition of the replay
  set? In particular, which structures, in what relative proportions, and at what total set size
  most efficiently preserve foundation-model breadth … We leave a systematic investigation of
  optimal replay composition to future work."`
- **MACE-MP-0** (arXiv:2401.00096; via `research/mace-finetuning-parameters.md:71-74`):
  application recipe = a **100,000-frame MPtrj replay** against ≈100-config target sets;
  pretraining corpus ≈1.5M configurations (≈6.7 %, 1:15). No ratio rule.
- Our own notes already say the same: `research/mace-finetuning-parameters.md:131-139`
  (`"co-training ratio K = 4 (4 co-train steps per phonon step)"`), `:218-219`
  (`"its budget shape (λΦ = 100, lr 1e-4, 200 epochs, K = 4 replay co-training, 1 HVP/step) is the
  closest analogue to our R4 row"`).

## 5. How to describe the ratio in any future write-up

- Correct: "R4 = 4 × N_TRAIN_HESSIAN at `config_weight` 10 — a **campaign design choice**
  (S0-C-57/S0-C-60), anchored on PFT's co-training **step** ratio K = 4 translated to a dataset
  ratio for MACE's concatenation mechanism; `config_weight` 10 mirrors PFT's upstream/phonon
  **force-weight** ratio λ_F 200/20."
- Incorrect: presenting 4 : 1 as a literature-prescribed replay size. No source states it; the docs
  advise "as many as you can (≥ ~30k)" and the 2026 benchmark study leaves optimal size/composition
  open. Our 110,960 frames are consistent with that guidance in order of magnitude (≥ 30k; same
  order as MP-0's 100k), but the "4×" itself is our rule.

## 6. Mechanism addendum (2026-09-30): with a user file, the file IS the size

Written for the replay re-ruling (ticket 09a): the round-1 production Replay is now a
30,000-frame draw of SPICE's train split in two arms — `replay30k_w1` / `replay30k_w10`,
identical frame sets (same seed) differing only in the stored `config_weight` — superseding
the single R4 row the sections above trace. The ruling rests on this mechanism, verified in
the fork at `1110ffb`:

- `mace/mace/tools/multihead_tools.py:105-128` — `prepare_pt_head` branches on the special
  values (`mp`, `omat`, `matpes_pbe`, `matpes_r2scan`): those set `"train_file": "mp"` and
  take the download path; any other value (a user path) lands in the `else` branch as
  `"train_file": args.pt_train_file`.
- `mace/mace/cli/run_train.py:297-314` — the matching dispatch: the special values call
  `assemble_replay_data(...)`; anything ase-readable goes to `get_dataset_from_xyz`, which
  reads the file as-is. The whole file is the replay set; nothing downsamples it.
- `--num_samples_pt` is read in exactly one place — `assemble_replay_data`
  (`mace/mace/tools/multihead_tools.py:188`, passed to `select_samples` as `num_samples`) —
  i.e., only on the `mp`/`omat`/`matpes_*` download path. Its default of 10000
  (`mace/mace/tools/arg_parser.py:587-591`) has never applied to our runs: every round-1
  replay is a user file.

So the draw tool's `--n` alone sets the size (both new arms: `--n 30000`), the file carries
it, and no runtime flag can change it.

## Sources

In-repo (all read verbatim; paths relative to `openQHA/`):
`.mem/decisions/decisions_C_hessian.md:104,107` (S0-C-57, S0-C-60);
`.scratch/hessian-learning-set/grilling-round-10-replay-cost.md:31,51-52,82-83`;
`.scratch/hessian-learning-set/spec-fine-tune-basins.md:10,69`;
`.scratch/hessian-learning-set/spec-phl-verbatim.md:154-159`;
`.scratch/hessian-learning-set/issues/15-the-smoke-fit.md:29-33,44`;
`.scratch/hessian-learn-framework/decisions/08-round-1-xyz.md:39-42`;
`.scratch/hessian-learn-framework/decisions/09-round-1-run.md:15`;
`.scratch/hessian-learn-framework/spec-round-1-run.md:13,129-136`;
`.scratch/hessian-learn-framework/spec-round-1-xyz.md:35-37`;
`.scratch/hessian-learn-framework/research/mace-finetuning-parameters.md:131-139,218-219`;
`openqha/data/dataset.py:137-139,249`; `openQHA-Hessian/openqha_hessian/smoke_fit.py:99-106`, `run.py:30`.

Web (fetched 2026-09-30):
- https://mace-docs.readthedocs.io/en/latest/guide/multihead_finetuning.html (replay size guidance, 30k; 10-30 epochs)
- https://mace-docs.readthedocs.io/en/latest/guide/finetuning_guidance.html (hyperparameter table; Tompa et al.)
- https://arxiv.org/abs/2606.12704 (Tompa et al. — replay sizes per benchmark; optimal composition deferred)
- https://arxiv.org/html/2601.07742v4 (PFT — K = 4 co-train steps; λ_F 200/20; λ_Φ = 100)
- https://arxiv.org/abs/2401.00096 (MACE-MP-0 — 100k replay; ≈1.5M-config corpus)

Assembled 2026-09-30: read-only repo sweep (subagent) + direct primary-source fetches. No code or
spec changed; this note records provenance only.
