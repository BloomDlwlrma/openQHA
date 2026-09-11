#!/bin/bash
#SBATCH --job-name=openqha_gpu_smoke
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# =======================================================================================
# TianheXY-A fine-grained environment, 30-MINUTE smoke test. Selected by
# `bash examples/run_chain.sh <conf> temp`; identical to ai.slurm except the walltime.
#
# It runs on `ai`, not on a partition called `temp`, because the fine-grained controller
# HAS no `temp`: `source set-XY-I.sh && sinfo` lists exactly one partition (measured
# 2026-09-11). `temp` (2 h ceiling, 1 node) exists on the DEFAULT controller only, where
# no card can be requested. The file keeps its name so the run_chain keyword is stable.
#
# Same policy as ai.slurm: 1 card = 12 CPUs = 120 GB, -G mandatory, --mem forbidden.
# =======================================================================================
if [ "${S0_PARSL_NESTED:-}" = "1" ]; then
    # shellcheck disable=SC1091
    source /APP/u22/ai_x86/toolshs/set-XY-I.sh || \
        echo "openQHA: could not source set-XY-I.sh; nested blocks would go to the default controller" >&2
fi
export OPENQHA_PARTITION=temp
export OPENQHA_KIND=gpu
exec bash "${SLURM_SUBMIT_DIR:-.}/examples/chain_body.sh" "$1"
