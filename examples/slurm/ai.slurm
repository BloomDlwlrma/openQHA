#!/bin/bash
#SBATCH --job-name=openqha_gpu
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# =======================================================================================
# TianheXY-A, **fine-grained GPU environment** ("tianhexy-i"). Submitted with `yhbatch`
# from a shell that has run
#
#     source /APP/u22/ai_x86/toolshs/set-XY-I.sh
#
# `examples/run_chain.sh` refuses to submit until that is true (it checks the effect:
# `sinfo -h -p ai -o %G` must mention gpu). The same login prompt without the script is a
# DIFFERENT Slurm controller -- Gres=(null), by-CPU allocation -- and there `--gpus=1`
# is refused with "Invalid generic resource (gres) specification". Both were measured on
# ln207 on 2026-09-11; the site PDF explains the two environments.
#
# THE POLICY (site PDF "星逸 AI 集群 GPU 细粒度调用使用说明"):
#     1 GPU = 12 CPUs = 120 GB.   -G / --gpus MANDATORY.   --mem FORBIDDEN.
#     billed  max(gpus, ceil(cpus/12)) x hours x weight
#
# So this file asks for one card and the 12 CPUs that come with it. Asking for 4 CPUs
# would cost exactly the same and leave 8 unusable. A conf may raise GPUS (and CPUS
# follows as 12 x GPUS) -- see "THE ALLOCATION" at the foot of `examples/run_chain.sh`.
# **No `--mem` line, ever.**
#
# ONE JOB, DRIVER AND WORKERS TOGETHER. `hpc/resource_configs/tianhe_a.py` sees
# SLURM_JOB_ID and runs parsl in THIS allocation (LocalProvider): no second allocation,
# no driver holding a card it does not use. For a campaign that needs parsl to submit
# further blocks, set S0_PARSL_NESTED=1 in the conf; the environment script is then
# sourced here too, so those submissions reach the same controller.
# =======================================================================================
if [ "${S0_PARSL_NESTED:-}" = "1" ]; then
    # shellcheck disable=SC1091
    source /APP/u22/ai_x86/toolshs/set-XY-I.sh || \
        echo "openQHA: could not source set-XY-I.sh; nested blocks would go to the default controller" >&2
fi
export OPENQHA_PARTITION=ai
export OPENQHA_KIND=gpu
exec bash "${SLURM_SUBMIT_DIR:-.}/examples/chain_body.sh" "$1"
