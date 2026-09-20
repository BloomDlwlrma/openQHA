#!/bin/bash
# One reference label: the xargs worker of hl_labels.slurm / hl_pipeline_debug.slurm.
#
#     hl_label_worker.sh <molecule dir> <generator> <basin> <k> <core range>
#
# The hkuhpc shape (qm9_reaction_eng/docs/bash_orca_remain_workflow_lecture.md, worker
# 10.2): idempotent (a finished job is reused, a frame another job holds is skipped --
# both inside `frame_labels.label_one`), the ORCA ranks pinned to the slot's cores with
# `taskset` (HL_TASKSET=0 leaves placement to the OS), scratch on the node-local
# S0_SCRATCH, the full .out and .hess copied back, success judged by ORCA's terminal
# line. Prints one line per frame; exit 1 only when ORCA did not terminate.
mol="$1"; gen="$2"; basin="$3"; k="$4"; cores="$5"
cmd=(python -m openqha.data.frame_labels "$mol" "$gen" "$basin" "$k"
     --level "${LEVEL:-wb97m-d3bj_def2-tzvppd}" --nprocs "${NPROCS:-4}" --maxcore "${MAXCORE:-6000}")
if [ "${HL_TASKSET:-1}" = 1 ] && command -v taskset >/dev/null 2>&1 && [ -n "$cores" ]; then
    exec taskset -c "$cores" "${cmd[@]}"
else
    exec "${cmd[@]}"
fi
