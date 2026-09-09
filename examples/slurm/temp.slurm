#!/bin/bash
#SBATCH --job-name=openqha_gpu_temp
#SBATCH --partition=temp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/openqha_%x_%j.out
#SBATCH --error=logs/openqha_%x_%j.err
# =======================================================================================
# TianheXY-A, 30-MINUTE queue. The gate before a long GPU job: same file, same conf,
# cluster is not stock Slurm).
#
# The allocation is a whole node: 8 cards and 56 cores. NO `--exclusive` -- by the
# 2026-09-05 ruling TianheXY-A follows TianheXY-AI's rules and Parsl's SlurmProvider
# default of True is what gets a submission refused here.
#
# 56 cores against 8 cards is the layout `hpc/resource_configs/tianhe_a.py` builds on:
# 56 workers, 7 per card. `available_accelerators` is passed to Parsl as an INT, never a
# list -- replaying the arithmetic (2026-09-07), an int gives 7 workers per card while a
# hand-built list gave 49 on card 0 and 7 on card 1, which reads as "the GPU is slow".
# =======================================================================================
export OPENQHA_PARTITION=temp
export OPENQHA_KIND=gpu
exec bash "${SLURM_SUBMIT_DIR:-.}/examples/chain_body.sh" "$1"
