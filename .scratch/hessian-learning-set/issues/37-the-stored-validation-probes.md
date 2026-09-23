# 37 — Algorithm 4: the validation probes come from the dataset, not from a hash of the Label

**Set:** hessian-learning-set · **Spec:** `spec-phl-verbatim.md` (Step 4) · **Ruling:** **S0-C-67** (revises S0-C-55, S0-C-65)

**Status:** done 2026-09-23 · **Blocked by:** 35 (done), 36 (done)

**Why.** The fixed validation probes were seeded from `SHA-1(bytes(H_r))[0:8]`. That is a
hash of the *measured object*, not of an identity, and it has three consequences the
Record cannot show: a Label recomputed at the same level (another ORCA version, another
convergence threshold, another dtype or write precision) has different bytes and therefore
a different probe set, so two runs' validation curves stop being comparable with nothing
said; the source of the randomness is bound to the quantity being measured; and it existed
only because ticket 12 put `has_hessian` and `hessian` into the batch but not the frame's
identity, which the file has carried all along (`_write_split` copies `a.info`).

PHL's own fixed-vector protocol stores the vectors in the dataset (its Algorithm 1 takes
stored `(v, H v)` pairs). We already store the full `H_r`, so we store `v` alone and form
`r = H_r v` in the loss. **The principle, for everything that follows: randomness is
derived from an identity, never from the data being measured.** Hashing an identity —
`frames.frame_seed(qm9_index, basin, generator, k)`, `dataset._rng(seed, ...)` — is
unaffected; that is the ordinary practice this repository already follows.

## The dataset writes them (`openqha/data/dataset.py`, `04_dataset.py`)

- [x] every labelled frame gets `valid_probes`: `[k_max, 3N]` Rademacher, `k_max = 16`,
      drawn from `_rng(seed, "probe", qid, generator, basin, k)` — the frame's identity,
      the Dataset's seed, nothing from `H_r`. *(Written as planned for `valid`, then
      extended to every labelled split while implementing: see "What it came to".)*
- [x] written flat into the labelled extxyz files (`has_valid_probes` true) and as
      `REF_valid_probes`; `pool` has no Label at all and gets none
- [x] **nested**: the K = 4 reading is the first 4 rows of the same set, so a K scan is a
      flag and not a rebuild, and the K = 4 and K = 8 readings are comparable by
      construction
- [x] Record: `VALID_PROBE_SOURCE = file`, `VALID_PROBE_KMAX`, `VALID_PROBE_MODE`,
      `N_VALID_PROBE_FRAMES` (the seed is the Dataset's own `SEED`, already recorded);
      the report note says what a rebuild changes

## The loss reads them (`openqha/training/phl_loss.py`)

- [x] `graph_labels` returns the graph's `valid_probes` beside its Label
- [x] `FrameConstants`: `seed` and the seeded `_fixed` draw go; eval mode takes the first
      `valid_n_probes` rows of the frame's stored set, training mode is unchanged
- [x] **both `hashlib` calls deleted** — `frame_seed` (line 84) and the `constants()` cache
      key (line 180). The cache either keys on the frame's identity or goes: since ticket
      35 a `FrameConstants` costs an `asarray` and two integers
- [x] a labelled valid frame with no stored probes is **refused**, not silently drawn for:
      that state means a stale valid file, and a silent draw is exactly the failure mode
      this ticket removes

## The calibration tool follows (`scripts/tooling/s0_probe_calibration.py`)

- [x] check 1 reads the frame's stored probes when the frames come from a Dataset's valid
      file, and draws its own (documented, seeded from the frame's identity) when it is
      pointed at a raw `--frames` glob; the docstring's "the frame's own seed,
      `phl_loss.frame_seed`" goes with it

## Acceptance

- [x] the same frame gives the same probes twice, under another mace seed, and after the
      Label is rewritten **bit for bit differently at the same level** (the regression this
      ticket exists for); different frames differ; a training draw differs from all of them
- [x] the first 4 rows of the stored set are the K = 4 set (nesting asserted)
- [x] `grep -rn hashlib openqha/training/phl_loss.py` is empty
- [x] `t_dataset`, `t_phl_loss`, `t_probe_calibration`, `t_train_run` green;
      `t_train_engine` green on the real 3-epoch multihead fine-tune reading a valid file
      that carries probes

**Cost, stated.** `k_max * 3N` values per labelled frame: at `3N <= 60` that is <= 960
numbers beside the Hessian's `9N^2 = 3600`, so about 26 % more text in the labelled files —
for `draw300`, roughly 17,000 train frames and 900 valid ones. Changing `k_max`, or the
Dataset's seed, needs the files rebuilt and makes the readings incomparable with earlier
runs' — accepted, and the Record says so.

**Follows:** 38 (Algorithm 5: `LOSS_EXACT`, the vibrational block's last caller, CONTEXT,
ADR 0006, T03 archived, T04/T05 re-executed).

## What it came to

**One correction to the plan, found by the real fine-tune.** The ticket said the valid split
alone carries the probes. mace evaluates the loss on the TRAINING split too -- its final error
table runs `Running inference on train_<head>` with the loss in eval mode -- and a labelled
train frame without a stored set then hit the refusal and stopped the run. So the Dataset
writes a set for **every labelled frame with a Hessian** (pool frames, which have no Label at
all, get none). What is particular to `valid` is the USE: the reading that drives the
schedule. The cost is the one stated in the ticket times the labelled frames rather than the
valid ones: `k_max * 3N` values beside a Hessian's `9N^2`, about 26 % more text in a file that
already carries the matrices.

**The dataset side.** `dataset.valid_probes(seed, qid, key, n_atoms, k_max)` draws
`[16, 3N]` of +-1 from `_rng(seed, "probe", qid, generator, basin, k)` -- the rule the split
itself uses -- and `build` attaches it to every labelled frame; `_write_split` writes it flat
as `valid_probes` with `has_valid_probes`, and as `REF_valid_probes` on the labelled splits
(the fork's `--valid_probes_key` default). Record: `VALID_PROBE_SOURCE = file`,
`VALID_PROBE_KMAX`, `VALID_PROBE_MODE`, `N_VALID_PROBE_FRAMES`, and a report note saying what
a rebuild changes.

**The loss side.** `graph_labels` returns the graph's stored set beside its Label, slicing the
batch by `k_max * 3n` (k_max read once from what the batch holds, since a Dataset stores one
value for every frame). `FrameConstants` is three slots -- the Label, `n3`, `nu` -- with no
seed and no cached draw; `probes(mode, k, rng=None, stored=None)` takes the first k rows in
eval mode and REFUSES when there are none, naming `04_dataset` and commit D. `constants()` is
no longer cached: the cache existed to key on the Label's bytes, and since ticket 35 a
FrameConstants is an `asarray` and two integers. **`import hashlib` is gone from the module.**

**The calibration tool.** `frame_probes(qid, key, n_atoms, k_max, seed_sets, stored=None)`
reads the frame's stored set when it has one and otherwise makes the same draw `04_dataset`
would from its identity; `frame_statistics` takes the set instead of a seed, so the K rows of
its table are readings of ONE nested set, as production takes them.

## Result

Unit 59/59 files (`t_phl_loss` 41 checks, `t_dataset` PASS, `t_probe_calibration` 31,
`t_dataset_mace_form` 10). Integration `t_judge_engine` 15/15 and `t_train_engine` on the real
3-epoch multihead fine-tune, which now reads the probes out of the valid file through fork
commit D.

The regression this ticket exists for is asserted twice: in `t_phl_loss`, rewriting the Label
bit-for-bit differently at the same level leaves the validation reading unchanged (the old
SHA-1 seed would have changed every probe); in `t_dataset`, the stored set is reproduced from
the Record's SEED and the frame's identity alone.

**Follows:** 38 (Algorithm 5: `LOSS_EXACT`, the vibrational block's last caller, CONTEXT, ADR
0006, T03 archived, T04/T05 re-executed).
