# 18: Fork commit C and the replay corrections -- a replay run whose Hessian term acts and whose Record is true (`openQHA-Hessian` commit C, `training/run.py`, `05_train.py`)

**What to build:** a fine-tune with `--multiheads` on that (1) trains the Hessian term at all and (2) records the Replay as it ran. Today neither holds: the fork's `run_train.py` overwrites `args.loss = "universal"` in multihead mode before the loss is built, so `--loss external` is dropped without an error (the smoke fit never saw it because it ran with replay off); and `run.py` passes `--num_samples_pt`, which mace reads only on its Materials-Project download path, records it as if it acted, and leaves mace's `--real_pt_data_ratio_threshold` at 0.1, which silently duplicates the fine-tune frames whenever they are fewer than 10 % of the replay (x125 on the fit set, x9 on the smoke set). After this ticket: fork commit C keeps an external loss in multihead mode; the driver never emits `--num_samples_pt`, always emits `--real_pt_data_ratio_threshold 0`, counts `PT_N_FRAMES` from the replay file, reads `PT_CONFIG_WEIGHT` from its frames, parses mace's "Total number of configurations" lines for both heads into the Record, and prints the replay ratio as frames per Hessian frame. Rulings: round 7 Q1-Q3, S0-C-56/57.

**Blocked by:** None (tickets 10-15 done). **Unblocks:** 16.

**Status:** ready-for-agent

- [ ] fork commit C on `openqha-hessian`: `args.loss = "universal"` only when `args.loss != "external"`; `tests/test_external_loss.py` gains a multihead case with a two-frame replay file asserting the external factory is called and the log names `loss_module` (fails on commit B, passes on C); `engine.MACE_FORK_COMMIT` / provenance pick up C
- [ ] `run.mace_argv`: no `--num_samples_pt`; `--real_pt_data_ratio_threshold 0` whenever `multiheads`; `argv_pairs` and the config file agree
- [ ] `run.write_record`: `PT_N_FRAMES` (frames counted in the replay file), `PT_CONFIG_WEIGHT` (the frames' `config_weight`, one value or `mixed`), `PT_HEAD_TRAIN` / `FT_HEAD_TRAIN` parsed from mace's log, `REPLAY_PER_HESSIAN_FRAME = PT_N_FRAMES / N_TRAIN_HESSIAN`; the `NUM_SAMPLES_PT` key removed from the schema (a stale key is a lie)
- [ ] `05_train.py --pt-train-file` documented as the size knob; `--num-samples-pt` removed; README's example command updated
- [ ] unit (`t_train_run.py`): the argv never contains `num_samples_pt`, contains the threshold 0 with `--multiheads`, the Record's four `PT_*` keys from a two-frame file with `config_weight = 3`; `smoke_fit.replay_ratio` unchanged
- [ ] integration (`t_train_engine.py`, CPU): a 3-epoch basin-only fine-tune on the methyloxirane fixture with a two-frame replay file: the Record's validation Hessian curve is not constant across epochs (the term acts), `MACE_FORK_COMMIT` is C's hash, the log shows both heads' counts
- [ ] `.mem/notes` round note; the ticket's Closing paragraph names the commit hash
