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
#     03  reference labels, IN THIS ALLOCATION: 16 ORCA jobs x 4 ranks at once,
#         capped at LIMIT_FRAMES (default 16 = one round, ~5 min) so the job fits;
#         the rest is hl_labels.slurm
#     04  the Dataset (labelled frames -> test, the rest -> pool) + OpenREACT export
#
# The Slurm log (%x_%j.out in the submit directory) is the Batch's report.
# =======================================================================================
#SBATCH --job-name=openqha_hl_debug
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# no `set -e` (project rule 2026-09-13); no `set -u` (the conda hooks, common.sh)

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
export S0_PARSL_RUN_DIR="$S0_RUNS_ROOT/parsl/${SLURM_JOB_NAME:-hl}_${SLURM_JOB_ID:-$$}"   # one parsl run dir per job: two jobs must never share parsl's numbered runinfo
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

# ---- 03. labels, in this allocation, one round ------------------------------------------
echo; echo "== 03 labels (in this allocation, at most $LIMIT_FRAMES frames)"
python -u workflows/hessian_learning/03_labels.py --tag "$TAG" --name "$NAME" --level "$LEVEL" \
    --resource tianhe_cpu --limit-frames "$LIMIT_FRAMES" 2>&1 | grep -v "Warning\|_Jd\|cuequivariance\|torch.load"

# ---- 04. the Dataset ---------------------------------------------------------------------
echo; echo "== 04 Dataset"
python -u workflows/hessian_learning/04_dataset.py --tag "$TAG" --name "$NAME" --level "$LEVEL" --export openreact 2>&1 | grep -v "Warning\|_Jd\|cuequivariance\|torch.load"

echo; echo "== done. Dataset under $S0_RUNS_ROOT/$TAG/_datasets/$NAME/; the rest of the labels: sbatch hpc/slurm/hl_labels.slurm"
