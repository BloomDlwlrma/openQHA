# 37 — Algorithm 4: the validation probes come from the dataset, not from a hash of the Label

**Set:** hessian-learning-set · **Spec:** `spec-phl-verbatim.md` (Step 4) · **Ruling:** **S0-C-67** (revises S0-C-55, S0-C-65)

**Status:** ready-for-agent · **Blocked by:** 35 (done), 36 (the fork field)

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

- [ ] every labelled frame of the **valid** split gets `valid_probes`: `[k_max, 3N]`
      Rademacher, `k_max = 16`, drawn from `_rng(seed, "probe", qid, generator, basin, k)`
      — the frame's identity, the Dataset's seed, nothing from `H_r`
- [ ] written flat into the valid extxyz (`has_valid_probes` true), `train` and `test`
      unchanged: training draws fresh probes every step and the judge uses the full matrix
- [ ] **nested**: the K = 4 reading is the first 4 rows of the same set, so a K scan is a
      flag and not a rebuild, and the K = 4 and K = 8 readings are comparable by
      construction
- [ ] Record: `VALID_PROBE_SOURCE = file`, `VALID_PROBE_KMAX`, `VALID_PROBE_SEED`,
      `N_VALID_PROBE_FRAMES`; the report note says what a rebuild changes

## The loss reads them (`openqha/training/phl_loss.py`)

- [ ] `graph_labels` returns the graph's `valid_probes` beside its Label
- [ ] `FrameConstants`: `seed` and the seeded `_fixed` draw go; eval mode takes the first
      `valid_n_probes` rows of the frame's stored set, training mode is unchanged
- [ ] **both `hashlib` calls deleted** — `frame_seed` (line 84) and the `constants()` cache
      key (line 180). The cache either keys on the frame's identity or goes: since ticket
      35 a `FrameConstants` costs an `asarray` and two integers
- [ ] a labelled valid frame with no stored probes is **refused**, not silently drawn for:
      that state means a stale valid file, and a silent draw is exactly the failure mode
      this ticket removes

## The calibration tool follows (`scripts/tooling/s0_probe_calibration.py`)

- [ ] check 1 reads the frame's stored probes when the frames come from a Dataset's valid
      file, and draws its own (documented, seeded from the frame's identity) when it is
      pointed at a raw `--frames` glob; the docstring's "the frame's own seed,
      `phl_loss.frame_seed`" goes with it

## Acceptance

- [ ] the same frame gives the same probes twice, under another mace seed, and after the
      Label is rewritten **bit for bit differently at the same level** (the regression this
      ticket exists for); different frames differ; a training draw differs from all of them
- [ ] the first 4 rows of the stored set are the K = 4 set (nesting asserted)
- [ ] `grep -rn hashlib openqha/training/phl_loss.py` is empty
- [ ] `t_dataset`, `t_phl_loss`, `t_probe_calibration`, `t_train_run` green;
      `t_train_engine` green on the real 3-epoch multihead fine-tune reading a valid file
      that carries probes

**Cost, stated.** `k_max * 3N` values per labelled valid frame: at `3N <= 60` that is
<= 960 numbers, about 900 frames in `draw300`, a few MB of text. Changing `k_max` needs the
valid file rebuilt — accepted, and the Record says so.

**Follows:** 38 (Algorithm 5: `LOSS_EXACT`, the vibrational block's last caller, CONTEXT,
ADR 0006, T03 archived, T04/T05 re-executed).
