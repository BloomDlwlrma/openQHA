#!/bin/bash
# openQHA branch A -- PRODUCTION on `deimos`. TianheXY-C. One node of the twelve.
#
# Do not submit this by hand twelve times. Use the submitter, which passes the shard:
#
#     mkdir -p $HOME/HDD_POOL/runs/openQHA/logs
#     bash hpc/slurm/submit_branchA_deimos.sh 1 16000 prod
#
# WHAT ONE OF THESE JOBS IS
# -------------------------
#     1 node, whole-node exclusive  (deimos allocates per node)
#     16 molecules at once x 4 CREST threads = 64 cores, exactly full
#     3 days walltime
#     shard K of N: every N-th molecule of the remaining list
#
# Twelve of them is the production shape the user asked for: **4 CPUs x 16 concurrent
# jobs x 12 nodes = 192 molecules in flight.**
#
# WHY 4 THREADS AND NOT MORE
# --------------------------
# 4 is the condition the 285 s/species cost was measured under (S0-A-8). Changing it
# invalidates that number, which is the only cost estimate this repository has. It is
# also not free: CREST's threads parallelise independent metadynamics runs, but the MACE
# calls inside them scale 111/90/72/101 ms at 1/2/4/8 threads -- 8 is SLOWER than 4.
#
# WHY A SHARD AND NOT AN ARRAY
# ----------------------------
# Tianhe supports --array. A shard is used instead because the worklist is computed from
# what is ON DISK at the moment each job starts (openqha/worklist.py), so a job that
# begins six hours after its neighbours does not redo what they have finished. An array
# index would have to be mapped to molecules in advance, and that map goes stale.
# =======================================================================================
#SBATCH --job-name=openqha_A
#SBATCH --partition=deimos
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm/%x_%A_%j.out
#SBATCH --error=logs/slurm/%x_%A_%j.err

# `set -eo pipefail` removed 2026-09-13 (user ruling: a failing step must not end the job; .mem/notes/notes_2026-09-13_no-errexit-anywhere.md)
# NOT `set -u` -- see branchA_debug.slurm.

: "${SHARD:?SHARD not set. Submit through hpc/slurm/submit_branchA_deimos.sh, which sets SHARD, RANGE_START, RANGE_END and TAG.}"
: "${RANGE_START:?RANGE_START not set}"
: "${RANGE_END:?RANGE_END not set}"
TAG="${TAG:-prod}"
THREADS="${THREADS:-4}"
CONCURRENT="${CONCURRENT:-16}"
# Per-molecule cap, not per-job. A molecule that has not finished in 4 hours is stuck;
# killing it costs one molecule, while letting it run costs a worker slot for 3 days.
TIMEOUT_S="${TIMEOUT_S:-14400}"

# ---- 1. site environment ---------------------------------------------------------------
module purge 2>/dev/null || true
module load anaconda3/2023.09 || {
    echo "openQHA: module load anaconda3/2023.09 FAILED -- conda is not on PATH" >&2
    exit 1
}
for v in $(env | awk -F= '{print $1}' | grep -E '^(PMI|SLURM_(CPU|TASK|NTASKS|NPROCS|STEP))'); do
    unset "$v"
done

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source hpc/env/common.sh
source hpc/env/tianhe.sh || { echo "openQHA: hpc/env/tianhe.sh refused (root unresolved); stopping." >&2; exit 1; }
openqha_report_env

openqha_require crest xtb python || exit 1
python -c "
import sys; sys.path.insert(0,'.')
from openqha import engine
p = engine.provenance()
print('engine      {}  sha256 {}'.format(p['engine'], p['sha256'][:16]))" || exit 1

echo "shard       $SHARD"
echo "range       $RANGE_START .. $RANGE_END"
echo "tag         $TAG"
echo "layout      $CONCURRENT concurrent x $THREADS threads = $((CONCURRENT*THREADS)) cores"
echo

# ---- 2. scan, then execute -------------------------------------------------------------
# The list is computed HERE, when this job starts, and not when it was submitted. Two
# things follow: a job that waited in the queue does not redo finished molecules, and the
# decision record (--explain, on stderr) lands in this job's own log.
WORK="$S0_SCRATCH/worklist_${SHARD//\//_}.txt"
python -u scripts/production/s0_E_worklist.py \
    --range "$RANGE_START" "$RANGE_END" --tag "$TAG" --shard "$SHARD" --explain \
    > "$WORK"
N=$(wc -l < "$WORK")
echo "worklist    $N molecules  ($WORK)"
if [ "$N" -eq 0 ]; then
    echo "nothing left in this shard."
    exit 0
fi

# ---- 3. the work -----------------------------------------------------------------------
# `xargs -P` rather than `yhrun`: every task is an independent single-node process, so
# there is nothing for the launcher to lay out, and going through it would re-import the
# SLURM_ variables that were just unset.
#
# The pipeline starts its OWN MACE server per molecule (one model behind one lock, so a
# shared server would serialise what looks parallel in every log). Nothing to set up here.
#
# `--halt never`: one molecule that fails must not take the other 15 with it. Failures are
# in the per-molecule driver.log and in the exit code below.
< "$WORK" xargs -P "$CONCURRENT" -I{} \
    bash -c 'python -u scripts/production/s0_A_pipeline.py \
                 --species "$1" --tag "'"$TAG"'" --threads '"$THREADS"' \
                 --timeout-s '"$TIMEOUT_S"' --hessian-mode analytic \
             || echo "FAILED $1" >> "'"$S0_SCRATCH"'/failures.txt"' _ {}

# ---- 4. account for it -----------------------------------------------------------------
echo
if [ -s "$S0_SCRATCH/failures.txt" ]; then
    echo "FAILURES ($(wc -l < "$S0_SCRATCH/failures.txt")):"
    cat "$S0_SCRATCH/failures.txt"
fi
python - <<PY
import sys; sys.path.insert(0, '.')
from openqha.store import basins
c = basins.census(tag="$TAG")
print("basin store: {} molecules under {}".format(c["total"], c["root"]))
PY
echo "shard $SHARD done."
