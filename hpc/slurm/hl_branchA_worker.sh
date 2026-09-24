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
# A CEILING FOR THE WHOLE MOLECULE -- OPT-IN, NO DEFAULT (2026-09-24). `--timeout-s`
# bounds ONE CREST attempt; a hang in the tighten / Hessian / MACE-socket phase after it
# would otherwise hold this worker and the task's slot until the JOB's walltime, so
# `WALL_S` is here to cut a molecule loose: rc=124, nothing written, rerun next round.
# It has NO default because molecules are legitimately long -- measured on draw300,
# 2026-09-24: one molecule rc=0 in 71 453 s (19.8 h), task walls 15-22 h against a 24 h
# limit, the census after two CREST attempts dominating -- and any ceiling derived from
# TIMEOUT_S would kill real work. Set it when your draw is known small (e.g. WALL_S=21600).
runner=()
if [ -n "${WALL_S:-}" ] && command -v timeout >/dev/null 2>&1; then
    runner=(timeout -k 60 "$WALL_S")
fi
if [ "${HL_TASKSET:-1}" = 1 ] && command -v taskset >/dev/null 2>&1 && [ -n "$cores" ]; then
    "${runner[@]}" taskset -c "$cores" "${cmd[@]}" > "${S0_SCRATCH:-/tmp}/branchA_${qid}.log" 2>&1
else
    "${runner[@]}" "${cmd[@]}" > "${S0_SCRATCH:-/tmp}/branchA_${qid}.log" 2>&1
fi
rc=$?
echo "$qid rc=$rc $(( $(date +%s) - t0 )) s $(grep -m1 -E '^\[(PASS|FAIL)\]|criteria' "${S0_SCRATCH:-/tmp}/branchA_${qid}.log" | cut -c1-80)"
[ "$rc" = 0 ] || tail -5 "${S0_SCRATCH:-/tmp}/branchA_${qid}.log" | sed "s/^/    $qid: /"
exit "$rc"
