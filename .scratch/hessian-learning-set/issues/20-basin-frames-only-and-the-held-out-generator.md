# 20: Basin frames only -- `dataset.build(train_generators=)`, the Held-out generator, ADR 0005

**What to build:** a Dataset whose train and valid splits hold basin frames only and whose test split holds, beside the by-frame and by-molecule test frames, EVERY frame of every other generator -- displaced, merged, saddle -- so that the judge can read them and training never sees them (S0-C-54). `dataset.build` gains `train_generators`, default `("basin",)`; frames of any other generator are routed to test regardless of the frame draw; the Dataset Record names `TRAIN_GENERATORS` and `HELD_OUT_GENERATORS`; the index's `split` column shows it; the `[[Class]]` table and the report count basin frames and held-out frames separately. The label rule (`frame_labels.HESSIAN_GENERATORS`) is untouched -- what changes is who trains on what. CONTEXT.md gains **Held-out generator**; `docs/adr/0005` records the ruling (basin Hessians only; the msRRHO result is the deliverable; extrapolation rows are reference; why not PHL's every-frame rule).

**Blocked by:** None. **Unblocks:** 22, 16.

**Status:** ready-for-agent

- [ ] `dataset.build(..., train_generators=("basin",))`: every labelled frame whose generator is not in the tuple goes to `test` with `molecule_split` unchanged and a new index column `held_out_generator = yes|no`; `04_dataset.py --train-generators basin [merged ...]`; Record keys `TRAIN_GENERATORS`, `HELD_OUT_GENERATORS`, `N_TRAIN_BASIN`, `N_TEST_HELD_OUT`
- [ ] on the local smoke Dataset: train + valid contain only `basin` frames; all 52 displaced frames are in test; the merged file for mace carries the same routing; `PURPOSE` and the pinned rule unchanged
- [ ] the judge's `frame_rows` reads test frames as before (no change needed) and the report's per-generator counts appear -- verified by running `06_judge` on the base model against the new Dataset
- [ ] CONTEXT.md: **Held-out generator** (a Frame generator whose frames enter the Dataset, may carry Labels and are read by the Judge, but never enter train or valid; today displaced, merged, saddle); the Dataset entry's split sentence updated; the Frame entry notes that "stationary" means stationary on the engine's surface
- [ ] `docs/adr/0005-basin-hessians-only.md` in the ADR format: context (PHL's rule, Rodriguez's stationary-only result, the 22- vs 4-day bill), decision, consequences (reference rows, no second-stage trigger)
- [ ] unit (`t_dataset_mace_form.py` or a new `t_dataset_generators.py`): routing on a synthetic Frame set with four generators; the Record keys; `--train-generators basin merged` moves merged into train
- [ ] `.mem/notes` round note
