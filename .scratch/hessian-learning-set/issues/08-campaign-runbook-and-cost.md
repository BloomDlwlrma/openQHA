# 08: The campaign: runbook, cost, progress (`workflows/hessian_learning/README.md`, `docs/tianhe_runbook.md`, `scripts/tooling/s0_hl_progress.py`)

**What to build:** the sequence as one page -- `00_draw` (login node, seconds) -> the debug gate of each stage (`LIMIT=16`) -> `sbatch --array=0-11 hl_branchA.slurm` -> `sbatch --array=0-11 hl_frames.slurm` -> `01_select` -> `sbatch --array=0-11 hl_labels.slurm` (resubmitted until `--assemble` exits 0) -> `04_dataset --export openreact`; the cost table (round 5): branch A 8-10 min per molecule, 02 ~1 min, labels 5 min per 10-atom frame measured -- and the 19-atom figure from ticket 06's gate (40-80 min by scaling); for N molecules at F frames each at T per frame, step 03 = N x F x T x 4 / 768 core-hours of 12 nodes (round-5 Q7 (b): 6,000 x ~5 Hessian frames x 1 h + 6,000 x ~12 gradient frames x 5 min -> ~140,000 core-hours, ~8 days), storage 0.5 MB per frame kept + 1 MB per molecule + the Dataset (~7 GB text per level per 100,000 frames, 2.6 GB HDF5); the order that buys nothing blind: branch A and 02 for the whole union first (a day), read the exact frame count, then the labels array resubmitted week by week (7-day walltime, finished frames skipped); what each log's last lines must say; how a killed array is resubmitted (as is: finished skipped, `running` claims respected, stale claims taken over after 2 h); the ALF/parsl alternative in `tmux` (`--max-blocks 12`) for the same stages. `scripts/tooling/s0_hl_progress.py --tag --name`: one table on the login node in seconds, per class and in total -- molecules drawn / branch A done / Frame sets / frames labelled / unlabelled / running -- read from disk, not from `squeue`.

**Blocked by:** 05, 06, 07.

**Delivers:** the page the campaign is run from; the progress table.

- [ ] `s0_hl_progress.py` on the smoke set prints 7 / 7 / 7 / 65 / 0 / 0 with the per-class rows; on a fresh draw the drawn count with zeros elsewhere
- [ ] the runbook's commands are the ones in the scripts' headers (a test greps them against each other, as `t_script_taxonomy` does for the taxonomy lines)
- [ ] OPEN until the campaign runs: the cost table's measured column filled from the array logs
