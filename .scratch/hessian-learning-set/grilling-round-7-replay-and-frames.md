# Hessian learning -- grilling, round 7: the replay and the frames-per-Hessian ratio (2026-09-21)

The user's ask: explain how the original methods (Rodriguez, PHL, PFT) train on Hessians
and how mace 0.3.16's own multihead fine-tuning replays the pretraining data; explain our
current method with the frames : Hessian ratio made explicit; then grill the ticket-15/16
replay arithmetic ("mace concatenates the two heads and shuffles, the ratio is dataset
size, three knobs: `--num_samples_pt`, `--weight_pt_head`, epochs").

## Facts checked in the code before asking (the fork `openQHA-Hessian`, `mace/cli/run_train.py`, `mace/tools/multihead_tools.py`, `mace/tools/arg_parser.py`)

1. **`--loss external` is overridden in multihead mode.** `run_train.py:383` sets
   `args.loss = "universal"` unconditionally when `args.multiheads_finetuning`, before
   `get_loss_fn` at line 751. With `--multiheads` our Hessian term is silently dropped;
   the smoke fit did not see it because it ran with `--multiheads_finetuning False`, and
   `t_train_engine.py` has no replay file. Fork commit B is therefore incomplete for the
   campaign's own command line (`hl_train.slurm` with `MULTIHEADS=1`).
2. **`--num_samples_pt`, `--weight_pt_head`, `--subselect_pt`, `--filter_type_pt` act only
   on the `mp` / `matpes_*` / `omat` replay path** (`assemble_replay_data` ->
   `select_samples`, which writes `weight_pt` into every frame's `info["config_weight"]`).
   With a user file in `--pt_train_file` (`prepare_pt_head`'s else branch, the only branch
   MACE-OFF can take) the WHOLE file is read and neither flag is consulted anywhere else
   (`grep -rn weight_pt_head mace` -> one use, in that path). So on our command line the
   coverage knob is the size of the replay file, the weight knob is the `config_weight`
   written into its frames, and `run.py` passing `--num_samples_pt` beside
   `--pt_train_file` records a number that does nothing.
3. **`--real_pt_data_ratio_threshold` (default 0.1) duplicates the fine-tune head's
   frames** when `(n_ft) / n_pt < 0.1`: `collections.train += train * int(0.1 / ratio)`
   (`run_train.py:392-407`). The fit set (4 train, 5,000 replay): factor 125, 4 -> 504
   frames, each seen 126 times per epoch, so the replay is ~10 per Hessian-frame visit,
   not the 1,000 written in ticket 15. The 65-frame smoke set (52 train): factor 9, 52
   -> 520, ~19 per visit, not 152. The campaign (~99,000 / 5,000 = 19.8 > 0.1, and
   20,000 -> 4.95) is untouched; duplication would start above ~990,000 replay frames.
   So "every frame once per epoch" is true only above the threshold.
4. **There is no tool that makes `spice_pt_5000.extxyz`.** `s0_spice_test_draw.py` draws
   the judge's forgetting set from SPICE's TEST split; the README and `05_train.py` name a
   replay file nobody writes, and nothing states that the replay draw must come from the
   TRAIN split and be disjoint from the judge's draw.
5. The originals' frames : Hessian ratio is 1 : 1 inside the Hessian set. PHL and
   Rodriguez train ANI from scratch on a set where every configuration carries a Hessian
   (loss `L_E + 0.30 L_F + 0.09 L_HVP`, one Gaussian probe per structure per step,
   `hvp = -grad(forces, coords, grad_outputs=v, create_graph=True)`, `/(9N^2)`; batch 400).
   PFT fine-tunes Nequix MP / MACE-MP-0 on MDR Phonon (8,510 supercells, every one with
   force constants), one one-hot column `(b, j)` per structure per step, forward-over-
   reverse HVP, and its co-training (Algorithm 1) is a STEP loop: one PFT batch, then
   `K = 4` E/F/S batches from MPtrj -- 4 upstream frames per phonon frame at equal batch
   size. Without it Matbench F1 0.751 -> 0.301; with it -0.4 % at a small phonon cost.
6. Ours: the fine-tune head is the draw300 Dataset, ~99,000 train frames of which
   ~29,000 carry a Hessian (basin frames) and ~70,000 are displaced E/F-only frames
   (round-5 Q7 (b)); the Hessian term is paid by labelled frames only
   (`has_hessian`, weight 0 elsewhere). Per epoch, per Hessian frame: 1 Hessian term,
   ~2.4 E/F-only frames of the same molecules, and `n_pt / 29,000` SPICE frames
   (0.17 at 5,000). Inside the head the ratio is set by the Frame generators, not by any
   training flag.

## Questions (frontier, round 7)
See the reply of 2026-09-21; answers are recorded here when they arrive.

