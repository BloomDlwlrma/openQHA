#!/bin/bash
#SBATCH --job-name=openqha_chain
#SBATCH --nodes=1
#SBATCH --output=openqha_chain_%j.out
#SBATCH --error=openqha_chain_%j.err
# ^ partition, walltime and --gpus are NOT here. They depend on which cluster was asked
#   for, and a #SBATCH line cannot be parameterised -- so they are passed as yhbatch
#   flags below, where they can be. Only the invariant directives live here.
#
# =======================================================================================
# THE WHOLE CHAIN, ONE FILE, TWO MODES
# =======================================================================================
#     bash examples/run_chain.sh examples/03_qha_openmm_propanal/chain.conf
#     MODE=hpc PARTITION=ai    bash examples/run_chain.sh <conf>
#     MODE=hpc PARTITION=h100x bash examples/run_chain.sh <conf>
#
#   local   run it here, now, at production settings.
#   hpc     submit THIS FILE to Tianhe with yhbatch, at production settings.
#
# **There is no third mode and no smoke mode.** A short run is not a smaller version of
# the answer -- it is a different quantity that looks like one, and this branch refuses
# acceptance criteria 1 and 5 below 20 ps for exactly that reason. If you want to know
# whether the machinery works, run `--stage estimator` of the parameter scan: it is
# seconds, it is exact, and it cannot be mistaken for a result.
#
# =======================================================================================
# HOW THE JOB GETS ITS SETTINGS: `source`, NOT `--export`
# =======================================================================================
# `--export=ALL,VAR=x` carries the SUBMITTING shell's whole environment into the job and
# then patches a few names on top. Two problems: what the job saw depends on the login
# shell that happened to submit it, and the settings exist only in a scheduler record
# that no one reads afterwards.
#
# Here the conf path is passed as the script's ARGUMENT -- Slurm hands a batch script its
# arguments unchanged -- and the job `source`s it. The settings are then a file that can
# be diffed, committed and re-run, and the job's environment is whatever the compute node
# gives it plus that file. Nothing depends on where it was submitted from.
# =======================================================================================
set -eo pipefail
# NOT `set -u`: the conda GROMACS activation hook fails under it and leaves the
# environment half-built while the script carries on. Measured 2026-09-03.

# ---------------------------------------------------------------------------------------
# 1. Where we are, and which conf
# ---------------------------------------------------------------------------------------
# Inside a job the submit directory is the repository; outside, walk up from this file.
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    ROOT="$SLURM_SUBMIT_DIR"
else
    ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$ROOT"

CONF="${1:-}"
if [ -z "$CONF" ]; then
    echo "usage: bash examples/run_chain.sh <conf>" >&2
    echo "   eg: bash examples/run_chain.sh examples/03_qha_openmm_propanal/chain.conf" >&2
    exit 2
fi
[ -f "$CONF" ] || { echo "no such conf: $CONF" >&2; exit 2; }

# shellcheck disable=SC1090
source "$CONF"

SPECIES="${SPECIES:?the conf must set SPECIES}"
TAG="${TAG:-chain}"
SEEDS="${SEEDS:-3}"
THREADS="${THREADS:-4}"
MODE="${MODE:-local}"
PARTITION="${PARTITION:-ai}"

# ---------------------------------------------------------------------------------------
# 2. What each mode and partition means. One table, so the two modes cannot drift.
# ---------------------------------------------------------------------------------------
case "$MODE" in
    local)
        RESOURCE="${RESOURCE:-local}"; ROUTE="${ROUTE:-ase}"; PLATFORM="CPU" ;;
    hpc)
        ROUTE="${ROUTE:-openmm}"; PLATFORM="CUDA"
        case "$PARTITION" in
            # TianheXY-A: whole node, 8 cards, 56 cores. 7 days.
            ai)    RESOURCE="tianhe_a";  SB_GPUS=8; SB_TIME="7-00:00:00" ;;
            # TianheXY-AI: allocation IS one card, 14 CPUs come with it. 3 days.
            h100x) RESOURCE="tianhe_ai"; SB_GPUS=1; SB_TIME="3-00:00:00" ;;
            *) echo "PARTITION must be ai or h100x, got '$PARTITION'" >&2; exit 2 ;;
        esac ;;
    *) echo "MODE must be local or hpc, got '$MODE'" >&2; exit 2 ;;
esac

# ---------------------------------------------------------------------------------------
# 3. Submit, or run. The same file does both.
# ---------------------------------------------------------------------------------------
if [ "$MODE" = "hpc" ] && [ -z "$SLURM_JOB_ID" ]; then
    LOGDIR="${S0_RUNS_ROOT:-$HOME/HDD_POOL/runs/openQHA}/logs"
    mkdir -p "$LOGDIR"          # Slurm opens --output BEFORE the script runs (rule 6)

    # Refuse on the login node rather than in the queue. A missing or truncated weight
    # file is the commonest way to waste an allocation and costs a second to rule out.
    module load anaconda3/2023.09 2>/dev/null || true
    OPENQHA_ENV="${OPENQHA_ENV:-openqha-gpu}" source hpc/env/tianhe.sh 2>/dev/null || true
    python - <<'PY' || exit 1
import sys
try:
    from openqha.potentials import engine
    p = engine.provenance()
except Exception as exc:                                          # noqa: BLE001
    sys.exit("openQHA: cannot load the potential here -- {}: {}\n"
             "  The MACE-OFF weights are NOT downloaded on a login node. Fetch them\n"
             "  where you have bandwidth and copy them in:\n"
             "      rsync -a data/potentials/ <tianhe>:$HOME/openQHA/data/potentials/"
             .format(type(exc).__name__, exc))
print("engine   {}  sha256 {}  pinned={}".format(
    p["engine"], p["sha256"][:16], p["sha256_pinned"]))
PY

    echo "submitting  $CONF  ->  $PARTITION (${SB_TIME}, --gpus=${SB_GPUS})"
    # The conf path is the script's ARGUMENT. No --export: see the header.
    exec yhbatch --partition="$PARTITION" --time="$SB_TIME" --gpus="$SB_GPUS" \
        --output="$LOGDIR/openqha_chain_${TAG}_%j.out" \
        --error="$LOGDIR/openqha_chain_${TAG}_%j.err" \
        "$0" "$CONF"
fi

# ---------------------------------------------------------------------------------------
# 4. Environment. Only the job needs the site modules.
# ---------------------------------------------------------------------------------------
if [ -n "$SLURM_JOB_ID" ]; then
    module purge 2>/dev/null || true
    module load anaconda3/2023.09 2>/dev/null || module load miniforge/24.7.1 2>/dev/null || true
    module load CUDA/12.3 || {
        echo "openQHA: module load CUDA/12.3 FAILED -- this would run on the CPU" >&2
        exit 1; }
    # Inherited task-layout variables make a child process misread its allocation and
    # try to relaunch itself through the scheduler (rule 7). CREST forks its own workers.
    for v in $(env | awk -F= '{print $1}' | grep -E '^(PMI|SLURM_(CPU|TASK|NTASKS|NPROCS|STEP))'); do
        unset "$v"
    done
    OPENQHA_ENV="${OPENQHA_ENV:-openqha-gpu}"
    export OPENQHA_ENV
    source hpc/env/common.sh
    source hpc/env/tianhe.sh
    openqha_report_env
fi

echo
echo "======================================================================"
echo "openQHA chain   species $SPECIES   tag $TAG"
echo "  mode      $MODE${SLURM_JOB_ID:+ (job $SLURM_JOB_ID)}"
echo "  resource  $RESOURCE   route $ROUTE   platform $PLATFORM"
echo "  seeds     $SEEDS      conf $CONF"
echo "======================================================================"

# ---------------------------------------------------------------------------------------
# 5. The chain. Every step is a PRODUCTION driver -- this file adds no science.
# ---------------------------------------------------------------------------------------
echo
echo "---- 1/4  branch A: conformer search -> every basin -------------------"
python -u scripts/production/s0_A_pipeline.py \
    --species "$SPECIES" --tag "$TAG" --threads "$THREADS" --hessian-mode analytic

echo
echo "---- 2/4  branch B: one trajectory per (basin, seed) ------------------"
# --basins auto: the count comes from branch A's own record, so the number of basins and
# the geometries they start from cannot disagree.
python -u scripts/production/s0_E_branchB_parsl.py \
    --species "$SPECIES" --tag "$TAG" --resource "$RESOURCE" \
    --route "$ROUTE" --basins auto --seeds "$SEEDS"

echo
echo "---- 3/4  collect: quasi-harmonic analysis per molecule ---------------"
python -u scripts/production/s0_E_branchB_collect_parsl.py \
    --species "$SPECIES" --tag "$TAG" --resource "${COLLECT_RESOURCE:-$RESOURCE}"

echo
echo "---- 4/4  the answer: F_conf over the ensemble ------------------------"
python -u scripts/production/s0_B_report_ensemble.py --species "$SPECIES" --tag "$TAG"
