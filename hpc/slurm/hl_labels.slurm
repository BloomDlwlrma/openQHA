#!/bin/bash
# openQHA Hessian-learning step 03 -- reference labels as a submitted job: plain bash,
# a task list, `xargs -P 16` over hl_label_worker.sh, NO parsl (user ruling 2026-09-19,
# the hkuhpc shape of qm9_reaction_eng/docs/bash_orca_remain_workflow_lecture.md).
# TianheXY-C. Then step 04.
#
#     cd $HOME/openQHA-main
#     sbatch hpc/slurm/hl_labels.slurm                                        # tag smoke, deimos, 1 node, 4 h
#     sbatch --partition=debug --time=00:30:00 hpc/slurm/hl_labels.slurm      # the same on the short queue
#     TAG=draw300 sbatch --array=0-11 --time=3-00:00:00 hpc/slurm/hl_labels.slurm   # 12 nodes: an ARRAY, one node each; one round of three (docs/hessian_learning_campaign.md)
#
# LAYOUT: CONCURRENCY x NPROCS = 64 = the node (16 x 4, user ruling 2026-09-18); MAXCORE
# = 512 GB x 0.75 / 64 = 6000 MB per rank. An ARRAY job splits the task list round-robin
# over its tasks; a frame another task is already running is skipped (its `running`
# claim), a finished frame is reused, so a killed or time-limited job is resubmitted as
# it is. Task 0 (or the only task) assembles and builds the Dataset at the end.
# =======================================================================================
#SBATCH --job-name=openqha_hl_labels
#SBATCH --partition=deimos
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

# no `set -e` (project rule 2026-09-13); no `set -u` (the conda hooks, common.sh)

TAG="${TAG:-smoke}"
NAME="${NAME:-$TAG}"
export LEVEL="${LEVEL:-wb97m-d3bj_def2-tzvppd}"
export NPROCS="${NPROCS:-4}"
export MAXCORE="${MAXCORE:-6000}"
CONCURRENCY="${CONCURRENCY:-16}"
LIMIT_FRAMES="${LIMIT_FRAMES:-}"
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
source hpc/env/orca.sh
openqha_find_orca || { echo "openQHA: no ORCA -- stopping" >&2; exit 1; }
openqha_report_env
echo "runs root  $S0_RUNS_ROOT"
echo "tag/name   $TAG / $NAME   level $LEVEL   layout ${CONCURRENCY} x ${NPROCS} ranks, %maxcore $MAXCORE   array task $ARRAY_ID/$ARRAY_N"
if [ $((CONCURRENCY * NPROCS)) -gt "$(nproc)" ]; then
    echo "openQHA: layout ${CONCURRENCY} x ${NPROCS} exceeds the node's $(nproc) cores" >&2; exit 1
fi

# ---- 2. the task list: pending frames of the selection, this array task's share --------
LIST="$S0_SCRATCH/hl_frames_${SLURM_JOB_ID:-$$}.txt"
python -u workflows/hessian_learning/03_labels.py --tag "$TAG" --name "$NAME" --level "$LEVEL" \
    ${LIMIT_FRAMES:+--limit-frames "$LIMIT_FRAMES"} --list "$LIST" 2>&1 | grep -v "Warning\|_Jd\|cuequivariance\|torch.load"
[ -f "$LIST" ] || { echo "openQHA: no task list written" >&2; exit 1; }
TOTAL=$(wc -l < "$LIST")
MINE="$LIST.task$ARRAY_ID"
awk -v id="$ARRAY_ID" -v n="$ARRAY_N" '(NR - 1) % n == id' "$LIST" > "$MINE"
echo "frames     $TOTAL pending, $(wc -l < "$MINE") for this task"

# ---- 3. the work: one worker per line, a core range per slot (hkuhpc's awk + xargs) ------
echo; echo "== 03 labels"
started=$(date +%s)
awk -v nc="$CONCURRENCY" -v np="$NPROCS" '{
        slot = (NR - 1) % nc; s = slot * np; e = s + np - 1
        printf "%s %s %s %s %d-%d\n", $1, $2, $3, $4, s, e
    }' "$MINE" | xargs -r -n 5 -P "$CONCURRENCY" bash hpc/slurm/hl_label_worker.sh
echo "== 03 wall $(( $(date +%s) - started )) s for $(wc -l < "$MINE") frames"

# ---- 4. assemble + the Dataset (the first array task; the others' frames are on disk) ----
if [ "$ARRAY_ID" = 0 ]; then
    echo; echo "== 03 assemble"
    python -u workflows/hessian_learning/03_labels.py --tag "$TAG" --name "$NAME" --level "$LEVEL" --assemble 2>&1 | grep -v "Warning\|_Jd\|cuequivariance\|torch.load"
    rc=${PIPESTATUS[0]}
    echo; echo "== 04 Dataset"
    python -u workflows/hessian_learning/04_dataset.py --tag "$TAG" --name "$NAME" --level "$LEVEL" --export openreact 2>&1 | grep -v "Warning\|_Jd\|cuequivariance\|torch.load"
    echo; echo "== assemble exit $rc (1 = frames still unlabelled: resubmit); Dataset under $S0_RUNS_ROOT/$TAG/_datasets/$NAME/"
    exit "$rc"
fi
