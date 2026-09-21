# Hessian learning -- grilling, round 10: the replay -- cost of {0, 5,000, 20,000, 68,000} frames, and the draw (2026-09-21)

Scope after S0-C-54: the training set is basin frames only (~19,000 labelled, ~17,000 in
train after the 90/5/5 frame split, ~850 valid, ~850 test), every frame with E/F/H; the loss
is the Cartesian target with k = 4 Rademacher probes (S0-C-53); replay is mace's multihead
concatenation of a SPICE train-split file (round 7's corrections: size = file, weight =
`config_weight`, threshold 0, fork commit C).

## 1. Cost model, in the unit u = one energy+forces pass with graph, per frame

Two measurements exist and disagree on what a Hessian frame costs, for a reason:

| source | what was measured | Hessian frame (k = 4) | E/F frame |
|---|---|---|---|
| T04 section 6, one propanal frame, CPU | forward + force backward with graph 45 ms = 1 u; each HVP backward 1.0-1.2 u | **~5 u** (1 + 4 x 1.0) + the parameter backward through the third-order graph | 1 u |
| ticket 15, 4 frames, batch 2, CPU, whole epoch | wall per epoch: E/F 10.7 s, k = 4 13.1 s | **1.23 x** the E/F epoch | 2.7 s per frame |

They disagree because the epoch measurement is dominated by fixed overhead at batch 2 on
CPU (data loading, optimiser, validation, Python); the per-frame AD cost is the 5 u. On a
GPU with batches of 32-128, overhead shrinks and the AD model is the one that scales, so it
is the model used below; the epoch model is quoted as the optimistic bound.

The training cost per epoch, T = N_basin x c_H + N_replay x 1 u, with c_H = 5 u (AD model)
or 1.23 u (epoch model):

| replay frames | ratio per Hessian frame (17,000 train) | AD model, u per epoch | vs no replay | epoch model, u per epoch | vs no replay |
|---|---|---|---|---|---|
| 0 | 0 | 85,000 | -- | 20,900 | -- |
| 5,000 | 0.29 | 90,000 | +6 % | 25,900 | +24 % |
| 20,000 | 1.18 | 105,000 | +24 % | 40,900 | +96 % |
| 68,000 (PFT's K = 4) | 4.0 | 153,000 | **+80 %** | 88,900 | +325 % |

In CPU seconds with ticket 15's per-frame numbers (2.7 s E/F, 3.3 s Hessian at k = 4, batch
2 -- an upper bound, no GPU): basin only 15.6 h per epoch; +5,000: 19.3 h; +20,000: 30.6 h;
+68,000: 66.6 h. A 100-epoch fine-tune is therefore GPU-only; the A800 factor g over this
CPU is not measured yet (`hl_train.slurm` on the smoke set is the measurement).

## 2. The cost nobody had counted: validation with the full Hessian

`wants_hessian_at_eval = True` makes mace compute the full matrix (3N unit probes,
~3N u = ~57 u per 19-atom frame) on every valid frame at every `eval_interval` (default 1
epoch). With ~850 valid frames: **~48,000 u per epoch -- 57 % of the basin-only training
epoch** and more than the 5,000-frame replay. Levers: `--eval_interval` 5-10, or a valid
subset for the Hessian term (the E/F metrics stay per epoch), or `probe = modes` on valid
only (n_vib = 51 instead of 57, small).

## 3. The draw

SPICE train split: 951,005 frames, 17,132 molecules (~55 conformers each), 4.9 GB. A uniform
draw by frame gives molecules with many conformers most of the replay; a draw by molecule
(m frames each) covers the chemistry: **4 frames per molecule = 68,528 frames, exactly the
PFT-equivalent size**; 1 per molecule = 17,132; 5,000 frames = 5,000 molecules x 1.
PFT re-draws its upstream batches every step from 1.58 M frames; a fixed file is seen every
epoch. PFT's co-train loss weights are 10x its phonon-step weights (A.4), tuned "until the
co-train validation energy does not diverge" -- our equivalent is `config_weight` on the
replay frames, a scan point, not a value to copy.

## 4. Questions (round 10) -- see the reply; answers recorded here when they arrive

### Q4 ruled (S0-C-55): (d)
Validation Hessian term = the training estimator with 4 Rademacher probes per valid frame,
probes fixed per frame (seed derived from the frame, recorded), entering the total validation
loss that drives the scheduler, the best checkpoint and the Stage Two switch.
`wants_hessian_at_eval = False`; the fork's full-matrix evaluate hook stays available, unused.
Cost ~4,300 u per epoch (vs ~48,000 full matrix); epoch-level noise ~0.9 % over 850 frames.
Q1-Q3, Q5-Q7 remain open.

### Q1, Q3, Q5, Q6 ruled (S0-C-56)
Q1: the replay is a uniform random draw BY FRAME from SPICE's train split (the user's
choice over the by-molecule draw; consequence recorded: molecules with many conformers
weigh more; the ids file lists frames per molecule). Q3: (a) one seed, one file, the whole
campaign. Q5: both assertions go into the draw tool and must be able to fail. Q6: the
**Replay** entry is in CONTEXT.md (2026-09-21). Q2 and Q7 open.

### Q7 ruled: yes
Ticket 16's first item is the measurement job on tianhe (one A800): basin-only smoke set +
5,000-frame replay, a fixed epoch count; Record carries seconds per epoch, u, g and the
validation share. The scan rows' sizes are written after g is known; the scan's design
(ratio per Hessian frame, forgetting line, `config_weight` row) is unchanged. Q2 pending.

### Q2 ruled (S0-C-57) -- the round closes
R0 = 0; R1 = 5,000; R2 = 17,132 (~1 per Hessian frame); R3 = 68,528 (~4, PFT's K = 4);
R4 = R3's frames with `config_weight = 10` (PFT's upstream/phonon force-weight ratio; 4 had no
basis). Nested prefixes of one seeded permutation. Sizes' wall times filled in after Q7's
measurement; if g is small, R0, R1 and a weight row on R1's frames first.
