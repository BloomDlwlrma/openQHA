#!/bin/bash
# openQHA branch B -- COLLECTION. TianheXY-C, ONE node, 64 molecules at a time.
#
#     mkdir -p $HOME/HDD_POOL/runs/openQHA/logs
#     TAG=prod yhbatch hpc/slurm/branchB_collect.slurm
#
# ONE CPU PER MOLECULE, 64 CONCURRENT, ONE NODE
# ----------------------------------------------------------------------
# The trajectories are produced on the GPU cluster, one per card. This is the other half:
# reading frames back and diagonalising a 3N x 3N covariance per molecule. That work is
# small, serial and float64 -- a card buys nothing -- and 64 independent molecules on one
# node buy 64x.
#
# **One node, not twelve.** The pass is short and its real cost is metadata traffic on
# Lustre; more nodes would buy contention, not throughput. If collection ever becomes the
# bottleneck, batch more molecules per task rather than asking for more nodes.
#
# The environment is `openqha` (CPU), not `openqha-gpu`: this reads numpy arrays.
# =======================================================================================
#SBATCH --job-name=openqha_B_collect
#SBATCH --partition=deimos
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

# No `set -eo pipefail`: a failing step must not end the job.

TAG="${TAG:-prod}"
CONCURRENT="${CONCURRENT:-64}"
SPECIES_LIST="${SPECIES_LIST:-}"        # optional file, one QM9 id per line

module purge 2>/dev/null || true
module load anaconda3/2023.09 || {
    echo "openQHA: module load anaconda3/2023.09 FAILED" >&2; exit 1; }
for v in $(env | awk -F= '{print $1}' | grep -E '^(PMI|SLURM_(CPU|TASK|NTASKS|NPROCS|STEP))'); do
    unset "$v"
done

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source hpc/env/common.sh
source hpc/env/tianhe.sh || { echo "openQHA: hpc/env/tianhe.sh refused (root unresolved); stopping." >&2; exit 1; }
openqha_report_env

# ---- which molecules have trajectories AND no result yet --------------------------------
# Same completion rule the Parsl collector uses, and it is STATED rather than implied: if
# the scanner and the worker disagree about "done", molecules are skipped for ever.
WORK="$S0_SCRATCH/collect_worklist.txt"
if [ -n "$SPECIES_LIST" ]; then
    cp "$SPECIES_LIST" "$WORK"
else
    python - <<PY > "$WORK"
import sys
sys.path.insert(0, '.')
from pathlib import Path
from openqha import config
cfg = config.load()
traj_root = Path(config.runs_dir("qha", cfg)) / "$TAG"
done_dir = Path("analysis/qha/$TAG/collect")
n_done = n_todo = 0
for d in sorted(traj_root.glob("dsgdb9nsd_*")):
    if not d.is_dir():
        continue
    if (done_dir / (d.name + ".json")).exists():
        n_done += 1
        continue
    print(d.name)
    n_todo += 1
print("collect: {} already done, {} to do, under {}".format(n_done, n_todo, traj_root),
      file=sys.stderr)
print("done means: analysis/qha/$TAG/collect/<species>.json exists", file=sys.stderr)
PY
fi

N=$(wc -l < "$WORK")
echo "tag         $TAG"
echo "worklist    $N molecules"
echo "layout      $CONCURRENT concurrent x 1 core"
echo
[ "$N" -eq 0 ] && { echo "nothing to do."; exit 0; }

mkdir -p "analysis/qha/$TAG/collect"

# --no-gmx: GROMACS is an EXTENSION and no number depends on it. MDAnalysis is the
# independent implementation behind acceptance criterion 2 and is inside the environment,
# so the criterion still has something behind it here.
< "$WORK" xargs -P "$CONCURRENT" -I{} \
    bash -c 'python -u scripts/production/s0_B_qha_analyse.py \
                 --species "$1" --tag "'"$TAG"'" --no-gmx \
                 > "analysis/qha/'"$TAG"'/collect/$1.log" 2>&1 \
             || echo "FAILED $1" >> "'"$S0_SCRATCH"'/collect_failures.txt"' _ {}

echo
if [ -s "$S0_SCRATCH/collect_failures.txt" ]; then
    echo "FAILURES ($(wc -l < "$S0_SCRATCH/collect_failures.txt")):"
    cat "$S0_SCRATCH/collect_failures.txt"
fi
echo "results under analysis/qha/$TAG/"
