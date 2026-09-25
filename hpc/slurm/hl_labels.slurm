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
# LAYOUT: the job ASKS for the node (`--exclusive --ntasks=1 --cpus-per-task=64 --mem=0`)
# and DERIVES the layout from what Slurm granted: CONCURRENCY = CORES / NPROCS, so 16 x 4
# ORCA ranks on a 64-core node (user ruling 2026-09-18); MAXCORE = 512 GB x 0.75 / 64 =
# 6000 MB per RANK, so the 64 ranks together stay under the node's memory. An ARRAY job splits the task list round-robin
# over its tasks; a frame another task is already running is skipped (its `running`
# claim), a finished frame is reused, so a killed or time-limited job is resubmitted as
# it is. Task 0 (or the only task) assembles and builds the Dataset at the end.
# =======================================================================================
#SBATCH --job-name=openqha_hl_labels
#SBATCH --partition=deimos
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

# no `set -e` (project rule 2026-09-13); no `set -u` (the conda hooks, common.sh)

# What Slurm granted, READ FIRST: the unset loop below erases SLURM_CPUS_* (it keeps
# Slurm's MPI variables out of ORCA's mpirun), and `nproc` is not a fallback for them --
# it honours OMP_NUM_THREADS, which common.sh sets to 1, and reported "1 core" on a
# 64-core node (2026-09-20). `--cpus-per-task=64 --mem=0` above ASK for the whole node;
# these two lines record what arrived, and the layout below is derived from it.
CORES="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-$(nproc --all)}}"
MEM_MB="${SLURM_MEM_PER_NODE:-0}"

TAG="${TAG:-smoke}"
NAME="${NAME:-$TAG}"
export LEVEL="${LEVEL:-wb97m-d3bj_def2-tzvppd}"
export NPROCS="${NPROCS:-4}"
# %maxcore is per RANK: 0.75 x the node's memory / cores (512 GB / 64 = 8 GB -> 6000 MB),
# so CONCURRENCY x NPROCS ranks together stay below what the allocation holds.
export MAXCORE="${MAXCORE:-6000}"
# One attempt per frame, bounded (round 11, S0-G-96): ORCA is killed after TIMEOUT_S and
# the frame is FAILED (its .out ends with the openQHA trailer); no round reruns it -- read
# it, then `python -m openqha.data.frame_labels <mol> <gen> <b> <k> --retry`. 8 h = 3x the
# 19-atom analytic Hessian's estimated upper end. A frame CUT by the walltime has no .out
# and is rerun whole by the next round.
export TIMEOUT_S="${TIMEOUT_S:-28800}"
CONCURRENCY="${CONCURRENCY:-$(( CORES / NPROCS ))}"
[ "$CONCURRENCY" -ge 1 ] || CONCURRENCY=1
LIMIT_FRAMES="${LIMIT_FRAMES:-}"
ARRAY_ID="${SLURM_ARRAY_TASK_ID:-0}"
ARRAY_N="${SLURM_ARRAY_TASK_COUNT:-1}"

# ---- 1. site environment ---------------------------------------------------------------
module purge 2>/dev/null || true
module load anaconda3/202309 2>/dev/null || module load anaconda3/2023.09 2>/dev/null || true
# The NARROW unset (hpc/slurm/README.md rule 2): scheduling variables for the non-ORCA
# payloads. ORCA children are made Slurm-blind one layer down, in
# openqha.qm_interfaces.orca.subprocess_env() (ADR 0008) -- do NOT widen this loop to
# "repair" ORCA.
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
echo "allocation $CORES cores, ${MEM_MB} MB (0 = the node's memory); $(( CONCURRENCY * NPROCS * MAXCORE / 1024 )) GB is the ranks' ceiling"
if [ $((CONCURRENCY * NPROCS)) -gt "$CORES" ]; then
    echo "openQHA: layout ${CONCURRENCY} x ${NPROCS} exceeds the node's $CORES cores" >&2; exit 1
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
echo "timeout    ORCA per frame TIMEOUT_S=$TIMEOUT_S s; a failed frame is not rerun (--retry by hand)"

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
    echo; echo "== assemble exit $rc (1 = frames with no ORCA job yet: resubmit; failed frames are not rerun); Dataset under $S0_RUNS_ROOT/$TAG/_datasets/$NAME/"
    exit "$rc"
fi
