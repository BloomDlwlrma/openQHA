#!/bin/bash
# Submit the twelve branch A shards. TianheXY-C, `deimos`, 3 days each.
#
#     bash hpc/slurm/submit_branchA_deimos.sh <START> <END> [TAG] [NODES]
#     bash hpc/slurm/submit_branchA_deimos.sh 1 16000 prod
#
# Twelve jobs x 16 concurrent molecules x 4 threads = 192 molecules in flight, which is
# the production shape (user, 2026-09-07). The site quota allows 32 nodes; 12 is what is
# asked for, and NODES below is the one number to change.
#
# WHAT THIS DOES THAT `for i in 1..12; do yhbatch; done` DOES NOT
#   * makes the log directory FIRST. Slurm opens --output before the job script runs, so
#     a missing directory kills the job with no log saying why (rule 6).
#   * scans once, on the login node, and REFUSES to submit twelve jobs for nothing.
#   * records the submission -- which shards, which tag, which job ids -- next to the
#     logs. A campaign whose parameters live in a shell history is a campaign nobody can
#     reproduce.
# No `set -eo pipefail`: a failing step must not end the job.

START="${1:?usage: submit_branchA_deimos.sh START END [TAG] [NODES]}"
END="${2:?usage: submit_branchA_deimos.sh START END [TAG] [NODES]}"
TAG="${3:-prod}"
NODES="${4:-12}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$HERE"

# The site quota is 32 running jobs and 32 nodes (Starlight, 2026-09-05). Refuse rather
# than submit into a wall: past the quota the scheduler simply holds everything and the
# campaign looks stalled instead of capped.
NODE_QUOTA=32
if [ "$NODES" -gt "$NODE_QUOTA" ]; then
    echo "refusing: $NODES nodes exceeds the $NODE_QUOTA-node quota on this cluster." >&2
    exit 2
fi

LOGDIR="${S0_RUNS_ROOT:-$HOME/HDD_POOL/runs/openQHA}/logs"
mkdir -p "$LOGDIR"

module load anaconda3/2023.09 2>/dev/null || true
source hpc/env/common.sh
source hpc/env/tianhe.sh

echo "scanning $START..$END for tag $TAG ..."
REMAINING=$(python -u scripts/production/s0_E_worklist.py \
                --range "$START" "$END" --tag "$TAG" | wc -l)
echo "remaining: $REMAINING molecules"
if [ "$REMAINING" -eq 0 ]; then
    echo "nothing to do -- not submitting anything."
    exit 0
fi
if [ "$REMAINING" -lt "$NODES" ]; then
    echo "only $REMAINING molecules left; submitting $REMAINING shards instead of $NODES."
    NODES="$REMAINING"
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
RECORD="$LOGDIR/submit_branchA_${TAG}_${STAMP}.json"
IDS=()
for k in $(seq 1 "$NODES"); do
    OUT=$(yhbatch \
        --export=ALL,SHARD="$k/$NODES",RANGE_START="$START",RANGE_END="$END",TAG="$TAG" \
        --output="$LOGDIR/openqha_A_${TAG}_${k}of${NODES}_%j.out" \
        --error="$LOGDIR/openqha_A_${TAG}_${k}of${NODES}_%j.err" \
        hpc/slurm/branchA_deimos.slurm)
    echo "  shard $k/$NODES  -> $OUT"
    IDS+=("$(echo "$OUT" | grep -oE '[0-9]+' | tail -1)")
done

python - <<PY
import json, os
rec = dict(
    submitted_at="$STAMP", tag="$TAG",
    range=[int("$START"), int("$END")],
    n_shards=int("$NODES"), remaining_at_submission=int("$REMAINING"),
    job_ids="""${IDS[@]}""".split(),
    script="hpc/slurm/branchA_deimos.slurm",
    layout="1 node x 16 concurrent molecules x 4 CREST threads, deimos, 3-00:00:00",
    note=("The worklist is re-scanned by each shard when it STARTS, not now, so a job "
          "that waits in the queue does not redo molecules its neighbours finished."),
)
open("$RECORD", "w").write(json.dumps(rec, indent=2))
print("\\nrecorded  $RECORD")
PY

echo
echo "watch with:  yhq -a          # queue"
echo "             yhi             # node states"
echo "             tail -f $LOGDIR/openqha_A_${TAG}_1of${NODES}_*.out"
