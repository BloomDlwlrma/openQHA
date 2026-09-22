# Hessian learning -- grilling, round 12: active learning as the stage-one frames grow (2026-09-22)

The user's ask: think about active learning -- the fine-tune as a loop that follows the
stream of stage-one labels (ORCA analytic Hessians on basin frames, arriving over days from
the 12-node array) instead of waiting for all of them, and that decides which molecules to
label next.

## Facts the design rests on

1. **What grows.** The only thing that grows is `N_H`, the number of basin frames with a
   reference Hessian (draw300: ~19,000 at the end, ~6,500 molecules, ~3 basins each; the
   array labels whole molecules in `draw.dat` order, `hl_list.py` round-robin over 12 tasks,
   ~4 days of 12 nodes for the basin frames alone). The E/F labels of the held-out
   generator frames grow in step (cheap, ~1/25 of a Hessian each).
2. **What is already incremental for free.**
   - The Dataset's by-frame split is per-frame seeded (`frame_draw`): a frame's split never
     changes when more frames arrive, and a rebuild keeps every earlier decision. So
     `train / valid / test` at round *t* are prefixes of round *t+1*'s: a learning curve
     over rounds compares like with like.
   - The Replay is nested by construction (S0-C-56): `--n 4 x N_H(t)` at every round is a
     prefix of the next round's file, same seed, same molecules first.
   - The pinned seven never enter training and are labelled first (smoke): an unbiased
     anchor that does not move with any selection rule.
   - `R4`'s recipe (S0-C-60) is fully determined by `N_H`: Replay = 4 x N_H at weight 10,
     `w_H` = the epoch-0 balance on the current train split. One number in, one row out.
3. **What is not there.** No warm start (`05_train.py` always fine-tunes from the base;
   `--foundation` could name a previous fine-tune but its E0s would then be that model's,
   which is the same level -- fine -- yet the Record's identity chain would need the
   parent run); no priority order in the label queue (`hl_list.py` walks `draw.dat`);
   no per-class query score; no learning-curve table across runs.
4. **What an unlabelled molecule already carries.** Branch A's MACE basins and the Frame
   set exist for every drawn molecule long before its ORCA label (a day for the whole
   union, ticket 08): the engine's Hessian at every basin frame is on disk (`pool` split).
   Any query score that needs only model predictions -- not labels -- can be evaluated on
   the whole pool at any round.
5. **Two things the word "active learning" can mean here**, and they are separable:
   (A) *incremental training along the label stream* -- when to retrain, from what, how to
   read the learning curve, when the labels are enough; (B) *query selection* -- which
   molecules the array labels next, so that each Hessian buys the most.
6. **What the literature did.** ANI-1x (Smith et al. 2018) used query by committee (an
   ensemble's disagreement on energies) to pick structures; DP-GEN the same with force
   deviation; Rodriguez / PHL / PFT did not select -- fixed sets. Nobody selected on
   Hessians. The committee's cost is the ensemble: k fine-tunes per round.

## (A) Incremental training: the proposal

- **Rounds on a doubling schedule of `N_H`**, not on wall-clock: round *t* is trained when
  `N_H >= 2^t x N_0` with `N_0` the smoke set's Hessian count (so ~5 rounds to 19,000).
  Doubling makes the learning curve `L_H(N)` readable on a log axis and keeps the number of
  A800 runs small (each is hours). A round = `04_dataset` (keeps the split) -> the Replay
  prefix at `4 x N_H` -> `05_train` R4 -> `06_judge` (gate closed) -> one line in a
  learning-curve table.
- **Cold start from the base every round.** A learning curve only means something if the
  protocol is identical at every point; warm-starting from round *t-1* confounds "more
  labels" with "more epochs". Warm start is the cheaper production choice LATER, once the
  curve has flattened; not during the curve.
- **`w_H` = the balance on the BASE model over the current train split**, recomputed each
  round (the definition is the base's `w_F L_F / L_H`, so it is comparable across rounds;
  it will drift slowly as the frame mix changes, and the Record says the value).
- **The learning curve is read on the judge's Hessian row** (held-out `||dH||^2/(9N^2)`
  vs base, and the per-class version), on the pinned seven (fixed molecules) and on the
  by-frame test frames (a growing random 5 %). Fit `L(N) = a N^{-alpha} + c` over the
  rounds: `alpha` says how fast labels pay, `c` says where it floors.
- **The stopping rule that replaces a placeholder threshold**: stop labelling when the
  improvement between two doublings is smaller than the judge's own noise -- the bootstrap
  s.e. of the held-out row (resample the test frames) -- i.e. when the next 2x of labels
  buys less than what the test set can resolve. This is the first threshold in this set
  that is measured, not assumed.

## (B) Query selection: the proposal, and its price

- **The score** for an unlabelled molecule *m* (whole molecules are the unit the array
  labels), evaluated on its MACE basin frames without any label:
  - **committee disagreement** `q_c(m) = mean over m's basins of ||H_t(x) - H_{t-1}(x)||_F^2 / (9N^2)`,
    the disagreement between the last two rounds' fine-tunes (a committee of two that costs
    nothing extra; with seeds, a proper QBC of k members costs k trainings per round). Where
    the model is still moving as labels arrive, the next label is worth most.
  - **class deficit** `q_k(m) = held-out Hessian row of m's structure class at round t / the
    mean over classes`: the judge's per-class table already exists (`[[Class]]`); classes
    the model is worst on get their molecules first.
  - combined: rank by `q_c` within the classes ordered by `q_k`; ties by `draw.dat` order.
- **A random control fraction** `f_rand` (proposal: 20 %) of every batch is drawn in
  `draw.dat` order regardless of score. Without it the labelled set is biased towards the
  molecules the score liked and the "interpolation" row stops estimating anything about the
  draw as a whole; with it, the random subset gives an unbiased learning curve beside the
  active one -- and a direct measurement of what selection bought (the active curve should
  lie below the random one at the same `N_H`; if it does not, selection is not paying for
  its own bookkeeping and is switched off).
- **The price.** (i) The score needs the previous rounds' models and one forward with
  `get_hessian` per pool basin frame (~19,000 x 3N HVPs ~ a few GPU-hours per round, or the
  cheaper `||F_t - F_{t-1}||` on the same frames as a proxy -- a force disagreement is one
  forward, and where two models disagree on forces they disagree on the Hessian
  nearby); (ii) the label queue must read a priority file (`hl_list.py --priority
  <file>`: the other session's tool; one flag); (iii) the by-frame test split of active
  molecules is biased with them -- the pinned seven and the random control are the only
  unbiased rows, and the report must say which is which.
- **What not to do.** Select on the training loss's own estimator (it is what the optimiser
  reads, and its noise is not model uncertainty); select frames within a molecule (the
  array labels molecules; a basin without its siblings breaks nothing but buys nothing);
  train an ensemble just for the score before the two-round committee has been tried.

## What this changes in the code (a sketch, for the tickets if approved)

- `openqha/training/curve.py`: the learning-curve table over runs (round, `N_H`, replay,
  `w_H`, the judge's Hessian row and its bootstrap s.e., per class), the power-law fit, the
  stopping line. Reads Records only.
- `openqha/training/query.py`: `score(pool_frames, model_t, model_{t-1}, classes, judge_t)`
  -> a priority table (`qm9_index`, class, `q_c`, `q_k`, rank, `control = yes|no`); writes
  `priority.dat` under the Dataset.
- `hl_list.py --priority priority.dat --control-fraction 0.2` (ticket 08's tool): the array
  takes the top of the list, with every fifth slot from `draw.dat` order.
- `07_round.py`: one driver for a round (`04 -> draw prefix -> 05 -> 06 -> curve -> query`).
- The judge's bootstrap s.e. on the held-out Hessian row (the number the stopping rule
  and, later, the reopened gate's thresholds rest on).

## Questions (round 12)

- **Q1 Rounds.** Doubling of `N_H` from the smoke count (a), a fixed label-time cadence
  (every day of the array, b), or only at the end plus one midpoint (c)? Recommended (a).
- **Q2 Start.** Cold start from the base every round for the curve (a), warm start from
  the previous round (b)? Recommended (a) now, (b) as the production option after the
  curve flattens.
- **Q3 Query score.** The two-round committee on Hessians (a), on forces as the cheap
  proxy (b), class deficit only (c), no selection -- `draw.dat` order, incremental training
  only (d)? Recommended (b) ranked within (c): one forward per pool frame, the judge's own
  class table, no ensemble.
- **Q4 Random control.** 20 % of every label batch in `draw.dat` order, reported as its own
  curve (a); none (b)? Recommended (a): it is the only way to know whether selection paid.
- **Q5 `w_H` per round.** The base's balance on the current split (a), fixed at round 0's
  value (b)? Recommended (a), the value in every Record.
- **Q6 Stopping.** Stop labelling when a doubling improves the held-out Hessian row by less
  than its bootstrap s.e. (a); label everything regardless and use the curve only as a
  report (b)? Recommended (a) as the rule, with the array simply continuing in the
  meantime -- the rule decides whether to draw MORE molecules (draw500), not whether to
  stop draw300.
- **Q7 The label worker.** Add `--priority` to `hl_list.py` (the other session's file, one
  flag) (a), or keep the queue as is and only re-order `draw.dat` between array
  submissions (b)? (b) needs no code but rewrites a Record; (a) is cleaner.
