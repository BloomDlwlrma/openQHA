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
# THE WHOLE CHAIN, ONE FILE, TWO MODES, THREE CHAINS
# =======================================================================================
#     bash examples/run_chain.sh examples/02b_qha_openmm_propanal/chain.conf
#     MODE=hpc PARTITION=ai     bash examples/run_chain.sh <conf>     # GPU, 7 days
#     MODE=hpc PARTITION=h100x  bash examples/run_chain.sh <conf>     # GPU, 3 days
#     MODE=hpc PARTITION=deimos bash examples/run_chain.sh <conf>     # CPU, 3 days
#
#   local   run it here, now, at production settings.
#   hpc     submit THIS FILE to Tianhe with yhbatch, at production settings.
#
#   CHAIN=qha       branch A -> branch B -> collect -> F_conf        (the default)
#   CHAIN=levels    02c: MACE vs GFN2 vs RI-MP2. **CPU partitions only** -- ORCA has no
#                   GPU path here, and D0-75 puts production quantum chemistry on deimos.
#   CHAIN=identity  02d: may nu_k replace omega_i in ZPE, enthalpy and entropy? Needs
#                   DENSE sampling, so the conf sets SAMPLE_EVERY.
#
#   CHAIN belongs in the conf, not the environment, so a submission is reproducible from
#   the file alone. The conf is re-sourced inside the job.
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
    echo "   eg: bash examples/run_chain.sh examples/02b_qha_openmm_propanal/chain.conf" >&2
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
CHAIN="${CHAIN:-qha}"

# ---------------------------------------------------------------------------------------
# 2. What each mode and partition means. One table, so the modes cannot drift.
# ---------------------------------------------------------------------------------------
case "$MODE" in
    local)
        RESOURCE="${RESOURCE:-local}"; ROUTE="${ROUTE:-ase}"; PLATFORM="CPU"
        KIND="cpu"; SB_GPUS=0 ;;
    hpc)
        case "$PARTITION" in
            # --- GPU: TianheXY-A, whole node, 8 cards, 56 cores ---------------------
            ai)     RESOURCE="tianhe_a";   KIND="gpu"; SB_GPUS=8; SB_TIME="7-00:00:00" ;;
            temp)   RESOURCE="tianhe_a";   KIND="gpu"; SB_GPUS=8; SB_TIME="00:30:00" ;;
            # --- GPU: TianheXY-AI, the allocation IS one card + 14 CPUs -------------
            h100x)  RESOURCE="tianhe_ai";  KIND="gpu"; SB_GPUS=1; SB_TIME="3-00:00:00" ;;
            # --- CPU: TianheXY-C. ORCA has no GPU path here, so 02c lives on these --
            deimos) RESOURCE="tianhe_cpu"; KIND="cpu"; SB_GPUS=0; SB_TIME="3-00:00:00" ;;
            debug)  RESOURCE="tianhe_cpu"; KIND="cpu"; SB_GPUS=0; SB_TIME="00:30:00" ;;
            *) echo "PARTITION must be ai, temp, h100x, deimos or debug, got" \
                    "'$PARTITION'" >&2; exit 2 ;;
        esac
        if [ "$KIND" = "gpu" ]; then
            ROUTE="${ROUTE:-openmm}"; PLATFORM="CUDA"
        else
            ROUTE="${ROUTE:-ase}"; PLATFORM="CPU"
        fi ;;
    *) echo "MODE must be local or hpc, got '$MODE'" >&2; exit 2 ;;
esac

# 2b. Which chain, and whether this partition can run it.
#
# `levels` is quantum chemistry: ORCA RI-MP2, no GPU path in this repository, and by
# D0-75 production quantum chemistry runs on deimos. Refusing here rather than in the
# queue turns a wasted GPU allocation into a one-line error.
case "$CHAIN" in
    qha|identity|levels) ;;
    *) echo "CHAIN must be qha, identity or levels, got '$CHAIN'" >&2; exit 2 ;;
esac
if [ "$CHAIN" = "levels" ] && [ "$MODE" = "hpc" ] && [ "$KIND" != "cpu" ]; then
    echo "CHAIN=levels is ORCA RI-MP2 -- it has no GPU path. Use PARTITION=deimos" >&2
    exit 2
fi

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

    echo "submitting  $CONF  ->  $PARTITION  chain=$CHAIN  (${SB_TIME}, gpus=${SB_GPUS})"
    # The conf path is the script's ARGUMENT. No --export: see the header.
    #
    # CHAIN and MODE reach the job through the CONF, not through the environment: the
    # conf is re-sourced inside the job, so a submission is reproducible from the file
    # alone. `--gpus` is omitted entirely on a CPU partition rather than passed as 0,
    # which some Slurm builds reject.
    GPUFLAG=()
    [ "${SB_GPUS:-0}" -gt 0 ] && GPUFLAG=(--gpus="$SB_GPUS")
    exec yhbatch --partition="$PARTITION" --time="$SB_TIME" "${GPUFLAG[@]}" \
        --output="$LOGDIR/openqha_${CHAIN}_${TAG}_%j.out" \
        --error="$LOGDIR/openqha_${CHAIN}_${TAG}_%j.err" \
        "$0" "$CONF"
fi

# ---------------------------------------------------------------------------------------
# 4. Environment. Only the job needs the site modules.
# ---------------------------------------------------------------------------------------
if [ -n "$SLURM_JOB_ID" ]; then
    module purge 2>/dev/null || true
    module load anaconda3/2023.09 2>/dev/null || module load miniforge/24.7.1 2>/dev/null || true
    if [ "$KIND" = "gpu" ]; then
        module load CUDA/12.3 || {
            echo "openQHA: module load CUDA/12.3 FAILED -- this would run on the CPU" >&2
            exit 1; }
    fi
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
echo "openQHA   chain $CHAIN   species $SPECIES   tag $TAG"
echo "  mode      $MODE${SLURM_JOB_ID:+ (job $SLURM_JOB_ID)}   partition $PARTITION"
echo "  resource  $RESOURCE   route $ROUTE   platform $PLATFORM"
echo "  seeds     $SEEDS      conf $CONF"
echo "======================================================================"

# ---------------------------------------------------------------------------------------
# 5. The chains. Every step is a PRODUCTION driver or an example driver -- this file
#    adds no science of its own, and every science setting comes from the conf or from
#    configs/branchB_protocol.yaml.
# ---------------------------------------------------------------------------------------

# Branch A first in all three chains: every one of them starts from the basins.
echo
echo "---- branch A: conformer search -> every basin ------------------------"
python -u scripts/production/s0_A_pipeline.py \
    --species "$SPECIES" --tag "$TAG" --threads "$THREADS" --hessian-mode analytic

case "$CHAIN" in

# =======================================================================================
qha)   # the conformational free energy: A -> B -> collect -> F_conf
# =======================================================================================
echo
echo "---- branch B: one trajectory per (basin, seed) -----------------------"
# --basins auto: the count comes from branch A's own record, so the number of basins and
# the geometries they start from cannot disagree.
python -u scripts/production/s0_E_branchB_parsl.py \
    --species "$SPECIES" --tag "$TAG" --resource "$RESOURCE" \
    --route "$ROUTE" --basins auto --seeds "$SEEDS" \
    ${PROD_PS:+--prod-ps "$PROD_PS"} ${EQUIL_PS:+--equil-ps "$EQUIL_PS"}

echo
echo "---- collect: quasi-harmonic analysis per molecule --------------------"
python -u scripts/production/s0_E_branchB_collect_parsl.py \
    --species "$SPECIES" --tag "$TAG" --resource "${COLLECT_RESOURCE:-$RESOURCE}"

echo
echo "---- the answer: F_conf over the ensemble -----------------------------"
python -u scripts/production/s0_B_report_ensemble.py --species "$SPECIES" --tag "$TAG"
;;

# =======================================================================================
levels)   # 02c: MACE vs GFN2-xTB vs RI-MP2, energies and forces through to G - E_el
# =======================================================================================
# CPU only. The cost is entirely the RI-MP2 `TightOpt NumFreq`: 1613 s per 10-atom
# structure at 4 processes (analysis/branch2_cost_model.json), and it is one ORCA job per
# basin, run in sequence. MACE and GFN2 together are under a second.
echo
echo "---- 02c: three levels on every basin ---------------------------------"
python -u examples/02c_hessian_benchmark_levels/s0_level_benchmark.py \
    --species "$SPECIES" --tag "$TAG" \
    --levels "${LEVELS:-mace,gfn2,rimp2}" --nprocs "${ORCA_NPROCS:-$THREADS}" \
    ${ORCA_MAXCORE:+--maxcore "$ORCA_MAXCORE"}
;;

# =======================================================================================
identity)   # 02d: may nu_k replace omega_i in ZPE, enthalpy and entropy?
# =======================================================================================
# The trajectories here are DENSELY sampled on purpose. The production protocol's 1.0 ps
# interval is chosen to filter high-frequency noise out of the ENTROPY; the zero-point
# energy needs of order 12 500 frames to settle, which at 1.0 ps would be 12.5 ns. Both
# numbers are measured in 02d stage 1. SAMPLE_EVERY is therefore a conf setting here and
# every meta.json records the value that was used.
echo
echo "---- branch B: dense trajectories, ${PROD_PS:-protocol} ps ------------"
python -u scripts/production/s0_E_branchB_parsl.py \
    --species "$SPECIES" --tag "$TAG" --resource "$RESOURCE" \
    --route "$ROUTE" --basins auto --seeds "$SEEDS" \
    ${PROD_PS:+--prod-ps "$PROD_PS"} ${EQUIL_PS:+--equil-ps "$EQUIL_PS"} \
    ${SAMPLE_EVERY:+--sample-every "$SAMPLE_EVERY"}

echo
echo "---- 02d: G_total from omega and from nu, term by term ----------------"
python -u examples/02d_qha_frequency_identity/s0_frequency_identity.py \
    --species "$SPECIES" --tag "$TAG" --stage all --traj-tag "$TAG" \
    ${NU_CUT:+--nu-cut "$NU_CUT"}
;;
esac
