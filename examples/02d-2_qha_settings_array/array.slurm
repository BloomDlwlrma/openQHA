#!/bin/bash
#SBATCH --job-name=openqha_02d2_array
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --time=36:00:00        # submit_array.sh passes --time; see array.conf
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# =======================================================================================
# 02d-2 ARRAY TASK. TianheXY-A fine-grained environment, or TianheXY-AI (a100x/h100x;
# submit_array.sh passes --partition). One task = G cards = G rows of
# settings.tsv running at once, one per card. Submitted ONLY by submit_array.sh, which
# passes --array, --gpus=G and --cpus-per-task=12G on the command line (they override the
# one-card defaults above) and hands this script the manifest and G as arguments.
#
#     $1  manifest: one generated conf per line, in row order
#     $2  G: cards in this task
#
# The rows this task owns are  row = SLURM_ARRAY_TASK_ID * G + k,  k = 0..G-1.
#
# ONE CARD PER ROW, ENFORCED TWICE. `CUDA_VISIBLE_DEVICES=k` is what OpenMM and torch
# see; `S0_CARD=k` tells hpc/resource_configs/tianhe_a.py to run parsl in-allocation on
# ONE card and NOT to pin workers itself -- its pinning arithmetic counts cards with
# `nvidia-smi -L`, which ignores CUDA_VISIBLE_DEVICES and would put every row's workers
# on cards 0..2. Each row also gets its own parsl run_dir (parsl_card<k>) so G drivers
# starting together do not race for runinfo/NNN.
#
# SCRATCH IS PER ROW, and this is not optional. hpc/env/tianhe.sh keys node-local
# scratch on SLURM_JOB_ID and chain_body.sh's EXIT trap copies it back and `rm -rf`s it.
# G rows sharing one scratch means the FIRST row to finish deletes the other G-1 rows'
# running trajectories. S0_SCRATCH_TAG carries the card index, so each row's scratch is
# ~/runs/<jobid>_card<k> and its end state goes to logs/node_local/<jobid>_card<k>.
#
# %A is the array's job id and %a this task's index (Slurm filename patterns). Each row
# additionally writes its own log, named by its tag, so one file per setting.
# =======================================================================================
MANIFEST="${1:?manifest path}"
G="${2:?cards per task}"
cd "${SLURM_SUBMIT_DIR:-.}"
[ -f "$MANIFEST" ] || { echo "no manifest: $MANIFEST" >&2; exit 2; }

TASK="${SLURM_ARRAY_TASK_ID:-0}"
N="$(grep -c . "$MANIFEST")"
echo "02d-2 task $TASK of array ${SLURM_ARRAY_JOB_ID:-?}: $G card(s), rows $((TASK * G))..$((TASK * G + G - 1)) of $N"
nvidia-smi -L 2>/dev/null | sed 's/^/  /' || echo "  (nvidia-smi not available)"

# The partition is whatever this job was submitted to (submit_array.sh passes it); it
# selects the resource config in chain_body (ai -> tianhe_a, a100x/h100x -> tianhe_ai).
export OPENQHA_PARTITION="${SLURM_JOB_PARTITION:-ai}"
export OPENQHA_KIND=gpu
pids=(); names=()
for k in $(seq 0 $((G - 1))); do
    row=$(( TASK * G + k ))
    [ "$row" -lt "$N" ] || break
    conf="$(sed -n "$((row + 1))p" "$MANIFEST")"
    [ -f "$conf" ] || { echo "row $row: missing conf $conf" >&2; continue; }
    name="$(basename "$conf" .conf)"
    log="logs/02d2_${name}_${SLURM_ARRAY_JOB_ID:-x}_${TASK}.log"
    echo "  card $k  row $row  $name  -> $log"
    S0_CARD="$k" CUDA_VISIBLE_DEVICES="$k" \
    S0_SCRATCH_TAG="card$k" \
    S0_KEEP_DIR="logs/node_local/${SLURM_JOB_ID}_card$k" \
        bash examples/chain_body.sh "$conf" > "$log" 2>&1 &
    pids+=($!); names+=("$name")
done

rc=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "  done    ${names[$i]}"
    else
        echo "  FAILED  ${names[$i]}  (see its log)" >&2; rc=1
    fi
done
exit $rc
