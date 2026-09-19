#!/bin/bash
# openQHA Hessian-learning step 03 -- reference labels as ONE submitted job, the driver
# and its 16-per-node ORCA workers inside the allocation (user ruling 2026-09-19: no
# Batch holds the login node's command line). TianheXY-C. Then step 04.
#
#     cd $HOME/openQHA-main
#     sbatch hpc/slurm/hl_labels.slurm                                  # tag smoke, deimos, 1 node, 4 h
#     TAG=smoke PARTITION=debug TIME=00:30:00 LIMIT_FRAMES=32 sbatch hpc/slurm/hl_labels.slurm   # two rounds on the short queue
#     TAG=draw NAME=draw200 NODES=12 TIME=7-00:00:00 sbatch hpc/slurm/hl_labels.slurm           # the 200-molecule draw
#
# PARTITION / TIME / NODES are read at submission: `sbatch --partition=debug --time=00:30:00
# --nodes=1 hpc/slurm/hl_labels.slurm` works too (command-line flags beat the #SBATCH lines).
# Finished frames are skipped, so a killed or time-limited job is simply resubmitted.
# =======================================================================================
#SBATCH --job-name=openqha_hl_labels
#SBATCH --partition=deimos
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# no `set -e` (project rule 2026-09-13); no `set -u` (the conda hooks, common.sh)

TAG="${TAG:-smoke}"
NAME="${NAME:-$TAG}"
LEVEL="${LEVEL:-wb97m-d3bj_def2-tzvppd}"
LIMIT_FRAMES="${LIMIT_FRAMES:-}"

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
export S0_PARSL_RUN_DIR="$S0_RUNS_ROOT/parsl/${SLURM_JOB_NAME:-hl}_${SLURM_JOB_ID:-$$}"   # one parsl run dir per job: two jobs must never share parsl's numbered runinfo
echo "runs root  $S0_RUNS_ROOT"
echo "tag/name   $TAG / $NAME   level $LEVEL   nodes ${SLURM_JOB_NUM_NODES:-1}"

echo; echo "== 03 labels, in this allocation"
python -u workflows/hessian_learning/03_labels.py --tag "$TAG" --name "$NAME" --level "$LEVEL" \
    --resource tianhe_cpu ${LIMIT_FRAMES:+--limit-frames "$LIMIT_FRAMES"} 2>&1 | grep -v "Warning\|_Jd\|cuequivariance\|torch.load"
rc=${PIPESTATUS[0]}

echo; echo "== 04 Dataset"
python -u workflows/hessian_learning/04_dataset.py --tag "$TAG" --name "$NAME" --level "$LEVEL" --export openreact 2>&1 | grep -v "Warning\|_Jd\|cuequivariance\|torch.load"

echo; echo "== labels exit $rc; Dataset under $S0_RUNS_ROOT/$TAG/_datasets/$NAME/"
exit "$rc"
