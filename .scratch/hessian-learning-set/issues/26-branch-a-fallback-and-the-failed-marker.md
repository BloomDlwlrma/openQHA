# 26: Branch A -- the SHAKE fallback keeps the published run's ensemble when the retry crashes; a molecule that fails leaves `_records/branchA.failed` and is not rerun (`conformer_search/crest.py`, `s0_A_pipeline.py`, `store/basins.py`, `hl_list.py`, `s0_hl_progress.py`)

**Why (2026-09-22, the draw300 branch A array at 12.6 h):** ~5 % of the molecules ended
`FileNotFoundError: CREST produced no ensemble at .../crest_shake1/crest_conformers.xyz ...
terminated_normally=False, terminated EARLY=0`. Read on `dsgdb9nsd_003375` (ethynyl-housane,
`C#CC1C2CCC12`): the published run (SHAKE=2) **terminated normally** in 6 min with an
ensemble and a few `terminated EARLY` metadynamics; the pointwise retry at SHAKE=1 died in
CREST's *trial* MTD (`Trial MTD 6 did not converge ... Automatic MD restart failed 6 times!
ERROR STOP`) -- on a strained bicyclic, fewer constraints make GFN2 MD less integrable, not
more. `run_with_shake_fallback` returns the retry's record unconditionally, so a good
ensemble was thrown away and the molecule got no record at all: pending forever, rerun by
every resubmission (a plain resubmission would in fact succeed through `run_crest`'s reuse
of the finished `crest/`, but the fresh path stays wrong and the record would not mention
the failed retry). Another ~4 % fell back successfully (criterion 10 records it; those are
`done`). Branch A had no memory of a failure, unlike the frames after ticket 24.

**What to build:**

- `crest.run_with_shake_fallback`: when the retry did not terminate normally or left no
  `crest_conformers.xyz`, and the first attempt terminated normally with one, return the
  FIRST attempt's record with `fallback_failed` (retry dir, shake, terminated_normally,
  n_terminated_early, seconds, the retry's last line) and a `fallback_reason` saying the
  published run's ensemble is used with its early terminations declared. Both attempts
  without an ensemble: the retry's record, as today.
- `s0_A_pipeline.run_crest`: the fresh path names the moved retry directory in
  `fallback_failed`; the reuse path (a finished `crest/` beside a `crest_shake1/` without an
  ensemble) records the same from disk. Criterion 10's detail line carries it.
- `store.basins`: `FAILED = "branchA.failed"`, `failed_path`, `failed(qid, tag)`,
  `write_failed(records_dir, text)`, `clear_failed(records_dir)`. `run_species` writes the
  marker where it raises "no ensemble" (reason, both attempts' last lines, the rerun
  command) and clears it when it writes `branchA.toml`.
- `hl_list.py --stage branchA`: a molecule with the marker is not pending (counted and
  printed); `s0_hl_progress.py`: an `A failed` column beside `branchA`.
- Docs: campaign page §4 row A (the two rc=1 kinds, what each means, the marker, the rerun
  command); `workflows/hessian_learning/README.md` branch A paragraph.

**Blocked by:** nothing. **Unblocks:** the branch A resubmission that completes draw300.

**Status:** implemented 2026-09-22; the tianhe item is the user's.

- [x] unit `t_branch_a_fallback.py` (4 checks): a fake `crest.run` -- first attempt normal + 3 EARLY with an
      ensemble, retry not normal without one -> the first record comes back with
      `fallback_failed` and `used_shake_fallback=False`; first attempt without an ensemble -> the
      retry's record as before; retry normal with an ensemble -> the retry (unchanged);
      `basins.write_failed / failed / clear_failed`
- [x] `t_hl_campaign`: a molecule with the marker -> `A_failed` 1, `branchA` 0
- [x] unit group green (see the report)
- [ ] tianhe (user): after A's array ends, `TAG=draw300 sbatch --array=0-11 ...` once more; the
      ~300 molecules reuse their `crest/` and finish; `s0_hl_progress` branchA -> drawn minus the
      marker count
