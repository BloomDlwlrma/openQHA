#!/bin/bash
# One molecule's branch A: the xargs worker of hl_branchA.slurm.
#
#     hl_branchA_worker.sh <qm9_index> <core range>
#
# `s0_A_pipeline.py` is idempotent (a molecule with its Property file and basins is
# skipped by the list already); CREST's threads and MACE are pinned to the slot's cores
# with `taskset` (HL_TASKSET=0 leaves placement to the OS). One line per molecule.
qid="$1"; cores="$2"
cmd=(python -u scripts/production/s0_A_pipeline.py --species "$qid" --tag "${TAG:-draw}"
     --threads "${THREADS:-4}" --timeout-s "${TIMEOUT_S:-3600}" --hessian-mode analytic)
t0=$(date +%s)
# A HARD CEILING FOR THE WHOLE MOLECULE (2026-09-24). `--timeout-s` bounds ONE CREST
# attempt; a hang in the tighten / Hessian / MACE-socket phase after it holds this worker
# and the task's slot until the JOB's walltime -- the "no status update, nothing to
# cancel until the time limit" shape of 2026-09-24. Default: three CREST timeouts (the
# published attempt, the pointwise retry, and the census after them) plus an hour; set
# WALL_S to say it yourself. rc=124 means the ceiling ended the molecule, nothing was
# written, and the next round reruns it whole.
WALL_S="${WALL_S:-$(( 3 * ${TIMEOUT_S:-3600} + 3600 ))}"
runner=()
command -v timeout >/dev/null 2>&1 && runner=(timeout -k 60 "$WALL_S")
if [ "${HL_TASKSET:-1}" = 1 ] && command -v taskset >/dev/null 2>&1 && [ -n "$cores" ]; then
    "${runner[@]}" taskset -c "$cores" "${cmd[@]}" > "${S0_SCRATCH:-/tmp}/branchA_${qid}.log" 2>&1
else
    "${runner[@]}" "${cmd[@]}" > "${S0_SCRATCH:-/tmp}/branchA_${qid}.log" 2>&1
fi
rc=$?
echo "$qid rc=$rc $(( $(date +%s) - t0 )) s $(grep -m1 -E '^\[(PASS|FAIL)\]|criteria' "${S0_SCRATCH:-/tmp}/branchA_${qid}.log" | cut -c1-80)"
[ "$rc" = 0 ] || tail -5 "${S0_SCRATCH:-/tmp}/branchA_${qid}.log" | sed "s/^/    $qid: /"
exit "$rc"
