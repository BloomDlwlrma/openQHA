#!/bin/bash
#SBATCH --job-name=openqha_gpu_h100x
#SBATCH --partition=h100x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus=1
#SBATCH --time=24:00:00        # 2026-09-13: 24 h, see tianhe_ai.py QHA_WALLTIME
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# =======================================================================================
# TianheXY-AI. **The allocation IS one card**, and 14 CPUs come with it -- this cluster
# allocates per card, not per node, and `--exclusive` is BANNED here (it is required on
# TianheXY-C; that is why these are separate files and not one with an `if`).
#
# `hpc/resource_configs/tianhe_ai.py` deliberately does NOT pass `available_accelerators`:
# Slurm has already set CUDA_VISIBLE_DEVICES to the one card in this allocation, while
# `nvidia-smi -L` would report all 8 physical cards on the node and Parsl would happily
# schedule onto seven that are not ours.
# =======================================================================================
export OPENQHA_PARTITION=h100x
export OPENQHA_KIND=gpu
exec bash "${SLURM_SUBMIT_DIR:-.}/examples/chain_body.sh" "$1"
