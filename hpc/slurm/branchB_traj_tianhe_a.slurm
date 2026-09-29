#!/bin/bash
# openQHA branch B -- quasi-harmonic trajectories on TianheXY-A. EIGHT CARDS, ONE NODE.
#
#     # 30-minute smoke test on `temp`
#     PARTITION=temp WALLTIME=00:30:00 PROD_PS=2 TAG=a_debug \
#         source /APP/u22/ai_x86/toolshs/set-XY-I.sh   # fine-grained env FIRST
        yhbatch -p ai -G 1 -c 12 -t 00:30:00 hpc/slurm/branchB_traj_tianhe_a.slurm
#
#     # production on `ai`
#     bash hpc/slurm/submit_branchB_tianhe_a.sh prod
#
# ONE TRAJECTORY PER CARD (user ruling 2026-09-07)
# ------------------------------------------------
# The node is 2 x 28 = 56 cores and 8 cards. A quasi-harmonic trajectory is a SERIAL
# chain of single-structure force calls, so there is nothing to parallelise inside one;
# the parallelism goes between them, and eight cards run eight trajectories.
#
# This script does the card assignment itself, with CUDA_VISIBLE_DEVICES per background
# subshell. The Parsl route does the same thing through `available_accelerators` -- and
# NOT through a variable set once in worker_init, which is the version that was written
# first and put all 8 workers on card 0. That failure produces correct numbers 8x slower,
# so it reads as "the GPU is slow" rather than as a placement bug. The table this script
# prints at the end is what makes it visible here.
#
# COST IS UNMEASURED ON THIS MACHINE
# ----------------------------------
# The only GPU figure this repository has for branch B is D0-C-5: 3.5x SLOWER than CPU,
# on a T400. That is a 2 GB entry-level card against 80 GB HBM2e, so it does not
# transfer -- but nothing has replaced it either. **Run the 30-minute smoke test and read
# seconds_per_ps_this_run out of meta.json before sizing a campaign.**
# =======================================================================================
#SBATCH --job-name=openqha_B_traj
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00        # 2026-09-12, see tianhe_a.py WALLTIME
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
# **ONE CARD, 12 CPUs -- the fine-grained environment's unit.** Submit from a shell that
# has run `source /APP/u22/ai_x86/toolshs/set-XY-I.sh`; there `-G` is MANDATORY and `--mem`
# is FORBIDDEN (site PDF). Without the script the same prompt talks to the DEFAULT
# controller, where Gres=(null) and this very `--gpus=1` is refused -- measured 2026-09-11.
# Raise --gpus (and --cpus-per-task = 12 x gpus) for more trajectories per job.
#
# NO --exclusive. Banned on TianheXY-AI (manual 6.2.2) and, by the 2026-09-05 ruling,
# not passed here either -- even though this cluster's own manual calls its nodes
# exclusive and its `yhbatch --help` does list the flag. If a submission is ever refused
# for want of exclusivity, this is the line to add back.
# NO memory request: the system sizes it from the GPU count and specifying it is banned.

# `set -eo pipefail` removed 2026-09-13 (user ruling: a failing step must not end the job; .mem/notes/notes_2026-09-13_no-errexit-anywhere.md)

SPECIES="${SPECIES:?SPECIES not set (one QM9 id, e.g. dsgdb9nsd_000018)}"
TAG="${TAG:-prod}"
BASINS_FILE="${BASINS_FILE:-auto}"      # `auto`: branch A's mace/basinNN/basin.extxyz under TAG (since 2026-09-14)
N_BASINS="${N_BASINS:-1}"
N_SEEDS="${N_SEEDS:-1}"       # one per basin (ruling 2026-09-13)
PROD_PS="${PROD_PS:-200}"
EQUIL_PS="${EQUIL_PS:-50}"
GPUS="${GPUS:-8}"
# Stop and flush before the queue kills the job. 90% of the walltime, so the last chunk
# is written and renamed rather than truncated.
WALL_BUDGET_S="${WALL_BUDGET_S:-77760}"      # 0.90 x 24 h, matching --time above.
                                             # Keep these two in step: the budget is
                                             # what makes a task flush instead of
                                             # being killed mid-rename.

# ---- 1. site environment ---------------------------------------------------------------
module purge 2>/dev/null || true
module load anaconda3/2023.09 2>/dev/null || module load miniforge/24.7.1 2>/dev/null || true
# 12.2, not 12.3: TianheXY-A's driver is 535.104.12 = CUDA 12.2, and OpenMM JITs PTX,
# which minor version compatibility does not cover (2026-09-12: every trajectory died
# with CUDA_ERROR_UNSUPPORTED_PTX_VERSION under 12.3). This file is the older direct
# submit path FOR TIANHEXY-A ONLY; examples/chain_body.sh now decides the module from a
# measurement (openqha_cuda_fit) and is what every example uses. Do not run this file on
# TianheXY-AI: its CUDA modules put lib64/stubs on LD_LIBRARY_PATH (CUDA error 34).
module load CUDA/12.2 || {
    echo "openQHA: module load CUDA/12.2 FAILED -- these tasks will not see a card" >&2
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
OPENQHA_ENV="${OPENQHA_ENV:-openqha-gpu}"
export OPENQHA_ENV
source hpc/env/common.sh
source hpc/env/tianhe.sh || { echo "openQHA: hpc/env/tianhe.sh refused (root unresolved); stopping." >&2; exit 1; }
openqha_report_env

# ---- 2. refuse early ---------------------------------------------------------------------
python - <<'PY' || exit 1
import sys
import torch
from openqha import engine
print("torch       {}  cuda_available={}  n_devices={}".format(
    torch.__version__, torch.cuda.is_available(), torch.cuda.device_count()))
if not torch.cuda.is_available():
    sys.exit("openQHA: torch cannot see a CUDA device. Refusing to run a GPU job on the "
             "CPU -- it would produce correct numbers at the wrong cost and nothing "
             "downstream would say so.")
p = engine.provenance()
print("engine      {}  weights {}".format(p["engine"], p["weights_path"]))
import openmm
print("openmm      {}  platforms {}".format(
    openmm.version.version,
    [openmm.Platform.getPlatform(i).getName()
     for i in range(openmm.Platform.getNumPlatforms())]))
PY

echo
echo "species     $SPECIES"
echo "tag         $TAG"
echo "layout      $GPUS trajectories, one per card"
echo "length      $EQUIL_PS ps equilibration + $PROD_PS ps production"
echo "budget      $WALL_BUDGET_S s per task"
echo

# ---- 3. one trajectory per card ---------------------------------------------------------
# (basin, seed) pairs, laid out over the cards round-robin. The seed is a pure function
# of (basin, seed index, seed0) inside the driver, so which card ran it changes nothing
# about the trajectory -- placement must never enter a result.
mkdir -p "$S0_SCRATCH/cards"

# Does this shell have `wait -n`? Asked once, rather than assumed: without it the throttle
# below would fall through and start every trajectory at once, which puts several
# processes on one card and slows all of them down.
HAVE_WAIT_N=1
help wait 2>/dev/null | grep -q -- "-n" || HAVE_WAIT_N=0
[ "$HAVE_WAIT_N" = "1" ] || echo "note: this shell has no 'wait -n'; falling back to " \
    "draining a full batch of $GPUS at a time." >&2

i=0
running=0
for b in $(seq 0 $((N_BASINS - 1))); do
    for s in $(seq 0 $((N_SEEDS - 1))); do
        card=$((i % GPUS))
        args=(--species "$SPECIES" --tag "$TAG"
              --basin "$b" --seed-index "$s"
              --prod-ps "$PROD_PS" --equil-ps "$EQUIL_PS"
              --platform CUDA --wall-budget-s "$WALL_BUDGET_S")
        [ -n "$BASINS_FILE" ] && args+=(--basins "$BASINS_FILE")
        (
            # ONE CARD PER SUBSHELL. Exported here and never once for the whole script:
            # setting it at the top puts all eight trajectories on card 0, which produces
            # correct numbers 8x slower and reads as "the GPU is slow".
            export CUDA_VISIBLE_DEVICES="$card"
            echo "start basin $b seed $s -> card $card"
            python -u scripts/production/s0_B_qha_trajectory_openmm.py "${args[@]}" \
                > "$S0_SCRATCH/cards/b${b}_s${s}_card${card}.log" 2>&1 \
                || echo "FAILED basin $b seed $s (card $card)" >> "$S0_SCRATCH/failures.txt"
        ) &
        i=$((i + 1))
        running=$((running + 1))
        # Never more than GPUS at once: a ninth process on eight cards shares one of them
        # and both trajectories slow down for no gain.
        if [ "$running" -ge "$GPUS" ]; then
            if [ "$HAVE_WAIT_N" = "1" ]; then
                wait -n || true          # || true: a failed trajectory is recorded above,
                running=$((running - 1)) #          it must not take the batch with it
            else
                wait || true
                running=0
            fi
        fi
    done
done
wait

# ---- 4. account for it, and prove the cards were actually used --------------------------
echo
if [ -s "$S0_SCRATCH/failures.txt" ]; then
    echo "FAILURES:"; cat "$S0_SCRATCH/failures.txt"
fi
python - <<PY
import json, sys
from pathlib import Path
sys.path.insert(0, '.')
from openqha import config
root = Path(config.runs_dir("qha")) / "$TAG" / "$SPECIES"
rows = []
for meta in sorted(root.glob("basin*/seed*/meta.json")):
    m = json.loads(meta.read_text(encoding="utf-8"))
    p = m.get("production", {})
    rows.append((m.get("basin_index"), m.get("seed"), p.get("n_frames"),
                 p.get("complete"), p.get("seconds_per_ps_this_run"),
                 m.get("platform")))
print("{:>6} {:>12} {:>8} {:>9} {:>14}  {}".format(
    "basin", "seed", "frames", "complete", "s/ps this run", "platform"))
for r in rows:
    print("{:>6} {:>12} {:>8} {:>9} {:>14}  {}".format(
        *[("-" if x is None else (round(x, 1) if isinstance(x, float) else x))
          for x in r]))
print()
print("output root", root)
if rows and all(r[5] != "CUDA" for r in rows):
    print("WARNING: no trajectory reports the CUDA platform. This job ran on the CPU.")
PY
