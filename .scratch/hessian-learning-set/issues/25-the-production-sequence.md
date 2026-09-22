# 25: The production sequence -- sbatch A, sbatch 02, the tmux parsl driver for 03, under the 32-submission quota (`docs/hessian_learning_campaign.md`, `workflows/hessian_learning/README.md`, `tests/unit/t_hl_campaign.py`)

**Why (2026-09-21/22):** the tenant quota on TianheXY-CN is 32 submitted jobs / 32 nodes /
32 running, shared by the group, and Slurm counts every array task as a submission. The
page's "3 rounds of 3 days, same sbatch command" therefore cannot be pre-queued (36 > 32), and
a dependency chain of 12-node arrays holds up to 26 of the 32 for days. The user chose the
ALF mode already in the repository (§6: a parsl driver in tmux on the login node,
`tianhe_cpu` role `labels`, `init_blocks=0, min_blocks=0, max_blocks=12`) for step 03 -- the
only stage that needs rounds -- and plain sbatch for A and 02. Nothing on the login node but
the driver, which polls `squeue`. The chain-job alternative (`--signal=B:USR1@900` + trap) is
recorded as considered, not built.

**What to write (docs; one test; no stage script changes):**

- `docs/hessian_learning_campaign.md` §1 rewritten as the production sequence, each stage sized
  to its cost and submitted only when the previous one is done (a human hand-off; every stage
  lists its own pending work from disk, so a wrong order only runs empty):

      TIMEOUT_S=14400 TAG=draw300 sbatch --array=0-11 --time=1-00:00:00 hpc/slurm/hl_branchA.slurm   # 12 submissions, ~10 h
      TAG=draw300 sbatch --array=0-1 --time=04:00:00 hpc/slurm/hl_frames.slurm                        # 2 submissions, ~1 h
      python workflows/hessian_learning/01_select.py --tag draw300                                     # login node, seconds
      tmux new -s hl-labels
      python -u workflows/hessian_learning/03_labels.py --tag draw300 --resource tianhe_cpu \
          --max-blocks 12 --walltime 3-00:00:00 2>&1 | tee $S0_RUNS_ROOT/logs/labels_draw300_$(date +%F_%H%M).log
      python workflows/hessian_learning/04_dataset.py --tag draw300 --split-by molecule                # after the driver ends

  with the quota table (user 32 / tenant 32-32-32, array tasks counted singly), the
  submission count per stage (12, 2, <= 12 blocks), how a block's time limit is handled
  (parsl replaces it; frames in flight are rerun whole -- ticket 24), how to stop the driver
  (`tmux attach`, Ctrl+C, parsl cancels its blocks; `squeue -u $USER` afterwards), how to
  leave it (`Ctrl+b d`), what a second driver run does (labels what the first left), and the
  §6 content merged into §1 (§6 becomes a pointer or disappears).
- The **tmux gate**, before the campaign's driver, five minutes:
  `tmux new -s hl-gate; python -u workflows/hessian_learning/03_labels.py --tag draw300 --resource tianhe_cpu --debug --limit-frames 1`
  -- it measures the three unverified facts: HTEX workers on a compute node reach the
  interchange on the login node (`tianhe_cpu.config` sets no `address=`; if the gate hangs at
  "waiting for workers", the fix is `address_by_interface(<nic>)` in `tianhe_cpu.py`, one line,
  after the measurement); `tmux` exists on the login node (`command -v tmux`, else `screen`);
  `squeue -u $USER` shows one block. The page tells the reader what each outcome means.
- The cost table's "measured on the campaign" column keeps its place; the first driver
  round's per-frame seconds fill it (ticket 08's open item).
- `workflows/hessian_learning/README.md`: the step-03 paragraph points at §1's sequence; the
  "3 x 3-day rounds" sentence goes.
- `tests/unit/t_hl_campaign.py`: the page carries the sequence's six commands and the quota
  numbers; the sbatch cross-check against the stage scripts' headers still passes (the parsl
  command is not an sbatch).

**Blocked by:** 24 (the page describes the lock's behaviour at a block's time limit and the
`failed` column; both are 24's). The tmux gate itself needs nothing and can run before 24.
**Unblocks:** the labels campaign run.

**Status:** implemented 2026-09-22 (docs + test); the tianhe gate is the user's.

- [x] page §1 rewritten (six commands with sizes and submission counts, the quota table, why not a chain, the
      tmux gate and its three outcomes, living with the driver); §3 = the frames' states + the sbatch rounds as the
      fallback (the header's array command kept there, submitted one round at a time); §4 row 04, §5 wording, §6 a pointer
- [x] README step 03: the pointer paragraph and the ALF paragraph rewritten (the driver is the route, the array the fallback)
- [x] `t_hl_campaign` PASS: the six commands, the quota words, the gate command, `address_by_interface`, the fallback;
      the header cross-check made `hl_frames.slurm`'s header comment say `--array=0-1` too (the one stage-script line
      touched: a comment; the parsl command is not an sbatch and is not cross-checked)
- [ ] tianhe (user): the tmux gate's result pasted -> `address=` decided; then the driver
