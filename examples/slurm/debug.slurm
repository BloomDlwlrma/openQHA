#!/bin/bash
#SBATCH --job-name=openqha_cpu_debug
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=logs/openqha_%x_%j.out
#SBATCH --error=logs/openqha_%x_%j.err
# =======================================================================================
# TianheXY-CN, 30-MINUTE queue. The gate before a long CPU job: same file, same conf,
# is stock Slurm; the GPU ones are not).
#
# `--exclusive` and the whole 64 cores: TianheXY-C allocates by node and requires it
# (docs/tianhe_runbook.md section 0). The chain's own THREADS setting decides how many of
# those cores a driver actually uses -- 4 for one species, more for a campaign -- and that
# lives in the conf, because it is a property of the work rather than of the queue.
#
# Everything below the directives is one line on purpose. The work is
# `examples/chain_body.sh`, shared by every .slurm here, so the four partitions cannot
# drift apart in what they run -- only in how they are allocated.
# =======================================================================================
export OPENQHA_PARTITION=debug
export OPENQHA_KIND=cpu
exec bash "${SLURM_SUBMIT_DIR:-.}/examples/chain_body.sh" "$1"
