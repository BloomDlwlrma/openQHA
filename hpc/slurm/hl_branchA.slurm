#!/bin/bash
# openQHA Hessian-learning campaign, stage A -- branch A (CREST + MACE basins) for every
# molecule of a draw, as a Slurm ARRAY of one-node tasks: plain bash, a task list,
# `xargs -P 16` over hl_branchA_worker.sh, NO parsl (user ruling 2026-09-19; the hkuhpc
# shape). TianheXY-C.
#
#     cd $HOME/openQHA-main
#     TAG=draw300 LIMIT=16 sbatch --partition=debug --time=00:30:00 hpc/slurm/hl_branchA.slurm   # the gate
#     TAG=draw300 sbatch --array=0-11 --time=1-00:00:00 hpc/slurm/hl_branchA.slurm             # the campaign
#
# LAYOUT: CONCURRENCY x THREADS = 64 = the node (16 x 4, the measured 285 s/species
# condition). Each array task takes every ARRAY_N-th pending molecule of the draw;
# molecules with branch A done are skipped (`basins.done`), so a resubmission continues.
# The molecule tree is written under $S0_RUNS_ROOT/<TAG>/ by s0_A_pipeline.py itself.
# =======================================================================================
#SBATCH --job-name=openqha_hl_branchA
#SBATCH --partition=deimos
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

# no `set -e` (project rule 2026-09-13); no `set -u` (the conda hooks, common.sh)

TAG="${TAG:-draw}"
NAME="${NAME:-$TAG}"
THREADS="${THREADS:-4}"
CONCURRENCY="${CONCURRENCY:-16}"
LIMIT="${LIMIT:-}"
TIMEOUT_S="${TIMEOUT_S:-3600}"
ARRAY_ID="${SLURM_ARRAY_TASK_ID:-0}"
ARRAY_N="${SLURM_ARRAY_TASK_COUNT:-1}"

# ---- 1. site environment ---------------------------------------------------------------
module purge 2>/dev/null || true
module load anaconda3/202309 2>/dev/null || module load anaconda3/2023.09 2>/dev/null || true
for v in $(env | awk -F= '{print $1}' | grep -E '^(PMI|SLURM_(CPU|TASK|NTASKS|NPROCS|STEP))'); do
    unset "$v"
done
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export OPENQHA_PARTITION="${SLURM_JOB_PARTITION:-deimos}"
source hpc/env/common.sh
source hpc/env/tianhe.sh || { echo "openQHA: hpc/env/tianhe.sh refused (root unresolved); stopping." >&2; exit 1; }
openqha_report_env
openqha_require crest xtb python || exit 1
python scripts/tooling/s0_check_weights.py 2>/dev/null | grep -E "registry pin" || echo "openQHA: weights check failed" >&2
echo "runs root  $S0_RUNS_ROOT"
echo "tag/name   $TAG / $NAME   layout ${CONCURRENCY} x ${THREADS} threads   array task $ARRAY_ID/$ARRAY_N"
if [ $((CONCURRENCY * THREADS)) -gt "$(nproc)" ]; then
    echo "openQHA: layout ${CONCURRENCY} x ${THREADS} exceeds the node's $(nproc) cores" >&2; exit 1
fi

# ---- 2. the task list: this task's share of the draw's molecules without branch A -----
LIST="$S0_SCRATCH/hl_branchA_${SLURM_JOB_ID:-$$}.txt"
python hpc/slurm/hl_list.py --tag "$TAG" --name "$NAME" --stage branchA --array-id "$ARRAY_ID" --array-n "$ARRAY_N" \
    ${LIMIT:+--limit "$LIMIT"} --out "$LIST" || exit 1

# ---- 3. the work: one worker per molecule, a core range per slot ------------------------
echo; echo "== A  branch A"
started=$(date +%s)
export TAG THREADS TIMEOUT_S
awk -v nc="$CONCURRENCY" -v np="$THREADS" '{ slot = (NR - 1) % nc; s = slot * np; printf "%s %d-%d\n", $1, s, s + np - 1 }' "$LIST" \
    | xargs -r -n 2 -P "$CONCURRENCY" bash hpc/slurm/hl_branchA_worker.sh
wall=$(( $(date +%s) - started ))

# ---- 4. the summary line: what is on disk now, for the molecules of this task -----------
python - <<PY
import sys; sys.path.insert(0, '.')
from openqha import config
from openqha.store import basins
root = config.runs_root(config.load())
ids = [l.strip() for l in open("$LIST") if l.strip()]
done = sum(1 for q in ids if basins.done(q, "$TAG", root=root))
print("== A  {} done, {} not done, of {} in this task; wall {} s".format(done, len(ids) - done, len(ids), $wall))
PY
