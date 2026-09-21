#!/bin/bash
# openQHA Hessian-learning workflow -- the WHOLE pipeline on the `debug` partition, one
# node, 30 minutes, nothing on the login node (user ruling 2026-09-19). TianheXY-C.
#
#     cd $HOME/openQHA-main
#     sbatch hpc/slurm/hl_pipeline_debug.slurm                     # the seven pinned molecules, tag smoke
#     SPECIES="dsgdb9nsd_000018 dsgdb9nsd_000035" TAG=t1 sbatch hpc/slurm/hl_pipeline_debug.slurm
#
# Steps, in order, each skipping what is already on disk:
#     A   branch A per molecule (CREST + MACE), xargs -P 16, 4 threads each     ~10 min
#     02  Frame sets (MACE E-F-H at every frame), xargs -P 7                    ~2 min
#     01  the selection (select.dat)                                            seconds
#     03  reference labels: a task list, xargs -P 16 over hl_label_worker.sh (4 ORCA
#         ranks each, taskset-pinned), capped at LIMIT_FRAMES (default 16 = one round,
#         ~5 min) so the job fits; the rest is hl_labels.slurm. No parsl in a job.
#     04  the Dataset (labelled frames -> test, the rest -> pool) + OpenREACT export
#
# The Slurm log (%x_%j.out in the submit directory) is the Batch's report.
# =======================================================================================
#SBATCH --job-name=openqha_hl_debug
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

# no `set -e` (project rule 2026-09-13); no `set -u` (the conda hooks, common.sh)

# what Slurm granted, READ FIRST (the unset loop below erases SLURM_CPUS_*; `nproc` honours
# common.sh's OMP_NUM_THREADS=1 and would say 1): the layouts are derived from it
CORES="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-$(nproc --all)}}"
NPROCS="${NPROCS:-4}"
CONCURRENCY="${CONCURRENCY:-$(( CORES / NPROCS ))}"
[ "$CONCURRENCY" -ge 1 ] || CONCURRENCY=1

TAG="${TAG:-smoke}"
NAME="${NAME:-$TAG}"
SPECIES="${SPECIES:-dsgdb9nsd_000018 dsgdb9nsd_000019 dsgdb9nsd_000035 dsgdb9nsd_000036 dsgdb9nsd_000044 dsgdb9nsd_000046 dsgdb9nsd_000048}"
LIMIT_FRAMES="${LIMIT_FRAMES:-16}"
LEVEL="${LEVEL:-wb97m-d3bj_def2-tzvppd}"

# ---- 1. site environment ---------------------------------------------------------------
module purge 2>/dev/null || true
module load anaconda3/202309 2>/dev/null || module load anaconda3/2023.09 2>/dev/null || true
for v in $(env | awk -F= '{print $1}' | grep -E '^(PMI|SLURM_(CPU|TASK|NTASKS|NPROCS|STEP))'); do
    unset "$v"
done
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export OPENQHA_PARTITION="${SLURM_JOB_PARTITION:-debug}"
source hpc/env/common.sh
source hpc/env/tianhe.sh || { echo "openQHA: hpc/env/tianhe.sh refused (root unresolved); stopping." >&2; exit 1; }
source hpc/env/orca.sh
openqha_find_orca || { echo "openQHA: no ORCA -- step 03 cannot run" >&2; }
openqha_report_env
echo "runs root  $S0_RUNS_ROOT"
echo "tag/name   $TAG / $NAME"
echo "species    $SPECIES"
python scripts/tooling/s0_check_weights.py 2>/dev/null | grep -E "registry pin|params sha" || echo "openQHA: weights check failed" >&2

# ---- A. branch A: whatever is not done yet ---------------------------------------------
echo; echo "== A  branch A (CREST + MACE) under tag $TAG"
TODO=$(python - <<PY
import sys; sys.path.insert(0, '.')
from openqha.store import basins
print(' '.join(q for q in "$SPECIES".split() if not basins.done(q, "$TAG")))
PY
)
if [ -n "$TODO" ]; then
    echo "   to run: $TODO"
    printf '%s\n' $TODO | xargs -P 16 -I{} \
        python -u scripts/production/s0_A_pipeline.py --species {} --tag "$TAG" --threads 4 --timeout-s 1500 --hessian-mode analytic
else
    echo "   all done already"
fi

# ---- 02. Frame sets ----------------------------------------------------------------------
echo; echo "== 02 Frame sets"
printf '%s\n' $SPECIES | xargs -P 7 -I{} python -u workflows/hessian_learning/02_frames.py --tag "$TAG" --species {} 2>&1 | grep -v "Warning\|_Jd\|cuequivariance\|torch.load"

# ---- 01. selection -----------------------------------------------------------------------
echo; echo "== 01 selection"
python -u workflows/hessian_learning/01_select.py --tag "$TAG" --name "$NAME"

# ---- 03. labels: one round of 16, plain bash + xargs (no parsl in a job) ---------------
echo; echo "== 03 labels (at most $LIMIT_FRAMES frames, ${CONCURRENCY} x ${NPROCS} ranks of $CORES cores)"
export LEVEL NPROCS MAXCORE="${MAXCORE:-6000}"
LIST="$S0_SCRATCH/hl_frames_${SLURM_JOB_ID:-$$}.txt"
python -u workflows/hessian_learning/03_labels.py --tag "$TAG" --name "$NAME" --level "$LEVEL"     --limit-frames "$LIMIT_FRAMES" --list "$LIST" 2>&1 | grep -v "Warning\|_Jd\|cuequivariance\|torch.load"
awk -v nc="$CONCURRENCY" -v np="$NPROCS" '{ slot = (NR - 1) % nc; s = slot * np; printf "%s %s %s %s %d-%d
", $1, $2, $3, $4, s, s + np - 1 }' "$LIST"     | xargs -r -n 5 -P "$CONCURRENCY" bash hpc/slurm/hl_label_worker.sh
python -u workflows/hessian_learning/03_labels.py --tag "$TAG" --name "$NAME" --level "$LEVEL" --assemble 2>&1 | grep -v "Warning\|_Jd\|cuequivariance\|torch.load"

# ---- 04. the Dataset ---------------------------------------------------------------------
echo; echo "== 04 Dataset"
python -u workflows/hessian_learning/04_dataset.py --tag "$TAG" --name "$NAME" --level "$LEVEL" --export openreact 2>&1 | grep -v "Warning\|_Jd\|cuequivariance\|torch.load"

echo; echo "== done. Dataset under $S0_RUNS_ROOT/$TAG/_datasets/$NAME/; the rest of the labels: sbatch hpc/slurm/hl_labels.slurm"
