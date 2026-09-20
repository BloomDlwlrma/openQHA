#!/bin/bash
# openQHA Hessian-learning campaign, stage 02 -- the Frame set (MACE E-F-H at every frame)
# of every molecule of a draw whose branch A is done, as a Slurm ARRAY of one-node tasks:
# plain bash, a task list, `xargs -P 16` over 02_frames.py --species, NO parsl (user
# ruling 2026-09-19). TianheXY-C. MACE on CPU, one thread per process (common.sh).
#
#     cd $HOME/openQHA-main
#     TAG=draw NAME=draw300 LIMIT=16 sbatch --partition=debug --time=00:30:00 hpc/slurm/hl_frames.slurm   # the gate
#     TAG=draw NAME=draw300 sbatch --array=0-11 --time=04:00:00 hpc/slurm/hl_frames.slurm               # the campaign
#
# Molecules with a Frame-set Record are skipped, so a resubmission continues.
# =======================================================================================
#SBATCH --job-name=openqha_hl_frames
#SBATCH --partition=deimos
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

# no `set -e` (project rule 2026-09-13); no `set -u` (the conda hooks, common.sh)

TAG="${TAG:-draw}"
NAME="${NAME:-draw300}"
CONCURRENCY="${CONCURRENCY:-16}"
THREADS="${THREADS:-4}"
LIMIT="${LIMIT:-}"
ARRAY_ID="${SLURM_ARRAY_TASK_ID:-0}"
ARRAY_N="${SLURM_ARRAY_TASK_COUNT:-1}"

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
echo "runs root  $S0_RUNS_ROOT"
echo "tag/name   $TAG / $NAME   layout ${CONCURRENCY} x ${THREADS} cores   array task $ARRAY_ID/$ARRAY_N"

LIST="$S0_SCRATCH/hl_frames_${SLURM_JOB_ID:-$$}.txt"
python hpc/slurm/hl_list.py --tag "$TAG" --name "$NAME" --stage frames --array-id "$ARRAY_ID" --array-n "$ARRAY_N" \
    ${LIMIT:+--limit "$LIMIT"} --out "$LIST" || exit 1

echo; echo "== 02 Frame sets"
started=$(date +%s)
export TAG
awk -v nc="$CONCURRENCY" -v np="$THREADS" '{ slot = (NR - 1) % nc; s = slot * np; printf "%s %d-%d\n", $1, s, s + np - 1 }' "$LIST" \
    | xargs -r -n 2 -P "$CONCURRENCY" bash -c 'q="$0"; c="$1"; t0=$(date +%s);
        if [ "${HL_TASKSET:-1}" = 1 ] && command -v taskset >/dev/null 2>&1; then run=(taskset -c "$c"); else run=(); fi
        "${run[@]}" python -u workflows/hessian_learning/02_frames.py --tag "$TAG" --species "$q" 2>&1 | grep -v "Warning\|_Jd\|cuequivariance\|torch.load" | sed "s/^/$q  /"
        echo "$q rc=${PIPESTATUS[0]} $(( $(date +%s) - t0 )) s"'
wall=$(( $(date +%s) - started ))

python - <<PY
import sys; sys.path.insert(0, '.')
from openqha import config
from openqha.data import frames
from openqha.store import basins, layout
root = config.runs_root(config.load())
ids = [l.strip() for l in open("$LIST") if l.strip()]
done = sum(1 for q in ids if (layout.frames_dir(basins.molecule_for(q, "$TAG", root=root)) / (frames.STEP + ".toml")).is_file())
print("== 02  {} Frame sets written, {} not, of {} in this task; wall {} s".format(done, len(ids) - done, len(ids), $wall))
PY
