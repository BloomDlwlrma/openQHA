#!/bin/bash
# One reference label: the xargs worker of hl_labels.slurm / hl_pipeline_debug.slurm.
#
#     hl_label_worker.sh <molecule dir> <generator> <basin> <k> <retry|-> <core range>
#
# The hkuhpc shape (qm9_reaction_eng/docs/bash_orca_remain_workflow_lecture.md, worker
# 10.2): idempotent (a finished job is reused, a frame another job holds is skipped --
# both inside `frame_labels.label_one`), the ORCA ranks pinned to the slot's cores with
# `taskset` (HL_TASKSET=0 leaves placement to the OS), scratch on the node-local
# S0_SCRATCH, the full .out and .hess copied back, success judged by ORCA's terminal
# line. Prints one line per frame; exit 1 only when ORCA did not terminate now (the frame
# is then FAILED -- ticket 24; the one-shot retry rides every round, ticket 04, so a LATER
# round re-attempts it once), 143 when the job was cut.
# The task list's 5th column (ticket 02) marks a FAILED frame the round re-attempts
# once: `retry` -> `--retry`, which archives the failed .out as <stem>.failed.out and is
# refused when that archive already exists (the retry is spent -- the frame is final).
mol="$1"; gen="$2"; basin="$3"; k="$4"; retry="$5"; cores="$6"
cmd=(python -m openqha.data.frame_labels "$mol" "$gen" "$basin" "$k"
     --level "${LEVEL:-wb97m-d3bj_def2-tzvppd}" --nprocs "${NPROCS:-4}" --maxcore "${MAXCORE:-6000}"
     --timeout "${TIMEOUT_S:-28800}")
[ "$retry" = "retry" ] && cmd+=(--retry)
if [ "${HL_TASKSET:-1}" = 1 ] && command -v taskset >/dev/null 2>&1 && [ -n "$cores" ]; then
    exec taskset -c "$cores" "${cmd[@]}"
else
    exec "${cmd[@]}"
fi
