#!/bin/bash
# openQHA Hessian learning, step 05 -- the fine-tune on TianheXY-A. ONE CARD.
#
#     # the gate: 2 epochs on the smoke Dataset, exact probe, 30 minutes
#     source /APP/u22/ai_x86/toolshs/set-XY-I.sh            # fine-grained env FIRST
#     TAG=smoke RUN=gate PROBE=modes MAX_EPOCHS=2 \
#         yhbatch -p ai -G 1 -c 12 -t 00:30:00 hpc/slurm/hl_train.slurm
#
#     # a scan row (tickets 16, 18, 19): the Replay file IS its size, s0_spice_pt_draw.py wrote it
#     TAG=draw300 RUN=R1 HESSIAN_WEIGHT=0.01 N_PROBES=4 MAX_EPOCHS=100 \
#         MULTIHEADS=1 PT_TRAIN_FILE=$S0_RUNS_ROOT/spice/spice_pt_5000.extxyz \
#         PT_VALID_FILE=$S0_RUNS_ROOT/spice/spice_pt_5000.valid.extxyz \
#         yhbatch -p ai -G 1 -c 12 -t 24:00:00 hpc/slurm/hl_train.slurm
#
# WHAT RUNS: `workflows/hessian_learning/05_train.py`, which builds mace's arguments and
# calls `mace.cli.run_train.run` from the FORK (BloomDlwlrma/openQHA-Hessian, branch
# openqha-hessian) with `--loss external --loss_module openqha.training.phl_loss:build`.
# The driver REFUSES a mace that is not the fork or whose checkout is dirty, so this job
# fails in seconds rather than producing a model nobody can trace. Install the fork in
# the environment first (`docs/tianhe_install.md`, `install_env_tianhe.slurm` section 9).
#
# ONE CARD. The training set is molecules of <= 20 atoms in batches of a few: the work
# per step is small and the loss adds k Hessian-vector products (k+1 backward passes per
# batch), which are sequential. Several cards would need DDP and would not make one epoch
# of 100,000 frames materially cheaper than the label campaign that produced them; if
# that changes, mace's own `--distributed` is the route, not more of this script.
# =======================================================================================
#SBATCH --job-name=openqha_hl_train
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
# ONE CARD, 12 CPUs -- the fine-grained environment's unit; `-G` is mandatory there and
# `--mem` is forbidden. NO --exclusive (banned on the AI clusters; ruling 2026-09-05).

# No `set -e`: a failing step must not take the job's reporting with it
# (.mem/notes/notes_2026-09-13_no-errexit-anywhere.md).

TAG="${TAG:?TAG not set (the campaign tag; the Dataset lives under it)}"
NAME="${NAME:-$TAG}"
RUN="${RUN:?RUN not set (the run name; the directory is <dataset>/train/<run>/)}"
LEVEL="${LEVEL:-wb97m-d3bj_def2-tzvppd}"
PROBE="${PROBE:-rademacher}"
N_PROBES="${N_PROBES:-4}"
HESSIAN_WEIGHT="${HESSIAN_WEIGHT:-1.0}"
ENERGY_WEIGHT="${ENERGY_WEIGHT:-1.0}"
FORCES_WEIGHT="${FORCES_WEIGHT:-100.0}"
MODE_WEIGHTING="${MODE_WEIGHTING:-cartesian}"      # the target (S0-C-53); entropy / none are diagnostics
MAX_EPOCHS="${MAX_EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SEED="${SEED:-123}"
MULTIHEADS="${MULTIHEADS:-0}"
PT_TRAIN_FILE="${PT_TRAIN_FILE:-}"         # the Replay file (s0_spice_pt_draw.py); its frame count is the size
PT_VALID_FILE="${PT_VALID_FILE:-}"         # its companion validation file (<stem>.valid.extxyz)
EXTRA="${EXTRA:-}"                       # passed to the driver as is (e.g. --register)

# ---- 1. site environment ------------------------------------------------------------------
module purge 2>/dev/null || true
module load anaconda3/2023.09 2>/dev/null || module load anaconda3/202309 2>/dev/null || \
    module load miniforge/24.7.1 2>/dev/null || true
module load CUDA/12.2 || {
    echo "openQHA: module load CUDA/12.2 FAILED -- this job would train on the CPU" >&2
    exit 1
}
case ":${LD_LIBRARY_PATH:-}:" in *stubs*)
    export LD_LIBRARY_PATH="$(printf '%s' "$LD_LIBRARY_PATH" | tr : '\n' | grep -v stubs | paste -sd: -)"
    echo "cuda      removed a stubs/ directory from LD_LIBRARY_PATH" ;;
esac
for v in $(env | awk -F= '{print $1}' | grep -E '^(PMI|SLURM_(CPU|TASK|NTASKS|NPROCS|STEP))'); do
    unset "$v"
done

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/slurm
OPENQHA_ENV="${OPENQHA_ENV:-openqha-gpu}"
export OPENQHA_ENV
source hpc/env/common.sh
source hpc/env/tianhe.sh || { echo "openQHA: hpc/env/tianhe.sh refused (root unresolved); stopping." >&2; exit 1; }
openqha_report_env

# ---- 2. refuse early: the card, and the mace fork -------------------------------------------
python - <<'PY' || exit 1
import sys
import torch
from openqha.potentials import engine
from openqha.training import run as train_run
print("torch       {}  cuda_available={}  n_devices={}".format(
    torch.__version__, torch.cuda.is_available(), torch.cuda.device_count()))
if not torch.cuda.is_available():
    sys.exit("openQHA: torch cannot see a CUDA device. Refusing to fine-tune on the CPU "
             "inside a GPU allocation -- it would cost a card and say nothing.")
try:
    info = train_run.check_fork(strict=True)
except RuntimeError as exc:
    sys.exit("openQHA: {}".format(exc))
print("mace        {}  fork {}  ({})".format(info["mace_version"], info["mace_fork_commit"][:12],
                                             info["mace_fork_path"]))
p = engine.provenance()
print("engine      {}  params {}  pin {}".format(p["engine"], p["params_sha256"][:16], p["params_pin_status"]))
PY

# ---- 3. the fine-tune -----------------------------------------------------------------------
ARGS=(--tag "$TAG" --name "$NAME" --run "$RUN" --level "$LEVEL" --device cuda
      --probe "$PROBE" --n-probes "$N_PROBES" --mode-weighting "$MODE_WEIGHTING"
      --energy-weight "$ENERGY_WEIGHT" --forces-weight "$FORCES_WEIGHT" --hessian-weight "$HESSIAN_WEIGHT"
      --max-epochs "$MAX_EPOCHS" --batch-size "$BATCH_SIZE" --seed "$SEED")
if [ "$MULTIHEADS" = "1" ]; then
    ARGS+=(--multiheads)
    [ -n "$PT_TRAIN_FILE" ] && ARGS+=(--pt-train-file "$PT_TRAIN_FILE")
    [ -n "$PT_VALID_FILE" ] && ARGS+=(--pt-valid-file "$PT_VALID_FILE")
fi
# shellcheck disable=SC2206
[ -n "$EXTRA" ] && ARGS+=($EXTRA)

echo
echo "tag         $TAG    dataset $NAME    run $RUN    level $LEVEL"
echo "loss        w_E $ENERGY_WEIGHT  w_F $FORCES_WEIGHT  w_H $HESSIAN_WEIGHT"
echo "probe       $PROBE  k=$N_PROBES  weighting $MODE_WEIGHTING"
echo "loop        $MAX_EPOCHS epochs, batch $BATCH_SIZE, seed $SEED, multiheads $MULTIHEADS"
echo

T0=$(date +%s)
python workflows/hessian_learning/05_train.py "${ARGS[@]}"
STATUS=$?
T1=$(date +%s)

# ---- 4. the summary line the log is read for ------------------------------------------------
RUN_DIR="$(python - "$TAG" "$NAME" "$RUN" <<'PY'
import sys
from openqha import config
from openqha.data import dataset
tag, name, run = sys.argv[1:4]
print(dataset.datasets_dir(config.runs_root(config.load()), tag, name) / "train" / run)
PY
)"
echo
echo "openQHA hl_train: run $RUN exit $STATUS, wall $((T1 - T0)) s, directory $RUN_DIR"
[ -f "$RUN_DIR/train.out" ] && sed -n '1,40p' "$RUN_DIR/train.out"
exit $STATUS
