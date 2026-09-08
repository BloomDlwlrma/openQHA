#!/bin/bash
# debug_realmole -- the full grid, on TianheXY-A. Submitted by run_debug_realmole.sh.
#
#     bash examples/02_qha_openmm_acetone/run_debug_realmole.sh --tianhe
#
# WHY THE WHOLE GRID IS ONE JOB
# -----------------------------
# The cells share their trajectories: length is a truncation, interval is a stride, the
# atom set is a mask. Splitting the grid across jobs would either recompute the same
# trajectories in several allocations or need them shared through the filesystem, and
# the second is just this job with extra steps. What IS worth parallelising -- the
# (basin, seed, thermostat) trajectories -- happens inside, one per card share.
# =======================================================================================
#SBATCH --job-name=openqha_debug_realmole
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --time=7-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# NO --exclusive (ruling 2026-09-05) and NO memory request (banned; sized from the GPUs).

set -eo pipefail
CONF="${CONF:-examples/02_qha_openmm_acetone/debug_realmole.conf}"

module purge 2>/dev/null || true
module load anaconda3/2023.09 2>/dev/null || module load miniforge/24.7.1 2>/dev/null || true
module load CUDA/12.3 || {
    echo "openQHA: module load CUDA/12.3 FAILED -- this would run on the CPU" >&2; exit 1; }
for v in $(env | awk -F= '{print $1}' | grep -E '^(PMI|SLURM_(CPU|TASK|NTASKS|NPROCS|STEP))'); do
    unset "$v"
done

cd "${SLURM_SUBMIT_DIR:-$PWD}"
OPENQHA_ENV="${OPENQHA_ENV:-openqha-gpu}"
export OPENQHA_ENV
source hpc/env/common.sh
source hpc/env/tianhe.sh
openqha_report_env

# Refuse before spending the allocation, not one cell at a time.
python - <<'PY' || exit 1
import sys
from openqha.potentials import engine
from openqha.quasi_harmonic import qha
p = engine.provenance()
print("engine   {}  sha256 {}".format(p["engine"], p["sha256"][:16]))
print("protocol {}".format(qha.protocol()["_status"]))
PY

echo
echo "conf $CONF"
python -u examples/02_qha_openmm_acetone/s0_debug_realmole.py --conf "$CONF" --print-conf
echo
python -u examples/02_qha_openmm_acetone/s0_debug_realmole.py --conf "$CONF"

echo
echo "The table is in the RESULT_LOG named by the conf, with the full record beside it"
echo "as .json. Read distinct_crossings before mean_TS_kcal."
