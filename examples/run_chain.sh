#!/bin/bash
# =======================================================================================
# SUBMIT. Reads a conf, picks the .slurm for the partition, hands it to the scheduler.
# =======================================================================================
#     bash examples/run_chain.sh <conf>                 # run it here, now
#     bash examples/run_chain.sh <conf> deimos          # TianheXY-CN,  CPU, 3 days
#     bash examples/run_chain.sh <conf> debug           # TianheXY-CN,  CPU, 30 min
#     bash examples/run_chain.sh <conf> ai              # TianheXY-A,   8 cards, 7 days
#     bash examples/run_chain.sh <conf> temp            # TianheXY-A,   8 cards, 30 min
#     bash examples/run_chain.sh <conf> h100x           # TianheXY-AI,  1 card, 3 days
#
# WHY THIS IS TWO FILES AND NOT ONE (user ruling 2026-09-09)
# ----------------------------------------------------------
# It used to submit ITSELF: one file carrying `#SBATCH` directives, re-invoked by the
# scheduler. That failed on the machine in three ways at once, and all three are fixed by
# the split:
#
#   * the job's output arrived on the LOGIN NODE's terminal instead of its `--output`
#     file, so the operator could not get their prompt back to submit the next step;
#   * `#SBATCH` lines cannot be parameterised, so partition, walltime and `--gpus` had to
#     be passed as command-line flags -- which meant the file you read was not the job
#     that ran, and `--ntasks` / `--cpus-per-task` / `--exclusive`, which differ per
#     cluster and cannot be flags in a sane way, were simply absent;
#   * the login node did real work first (importing torch, loading the potential, probing
#     the basin store), which is exactly what a login node is not for.
#
# So: **this** file is the submitter and does nothing heavy. `examples/slurm/<part>.slurm`
# carries the full directives for one queue. `examples/chain_body.sh` is the work, shared
# by all of them, so the five queues can differ in allocation and cannot differ in what
# they compute.
#
# The conf reaches the job as the .slurm's ARGUMENT -- Slurm hands a batch script its
# arguments unchanged -- and the job sources it there. No `--export`: that would carry
# the submitting shell's environment in, making the run depend on who submitted it.
# =======================================================================================
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONF="${1:-}"
PARTITION="${2:-${PARTITION:-}}"

if [ -z "$CONF" ]; then
    echo "usage: bash examples/run_chain.sh <conf> [partition]" >&2
    echo "   eg: bash examples/run_chain.sh examples/02b_qha_openmm_propanal/branchA.conf deimos" >&2
    echo "       bash examples/run_chain.sh examples/02b_qha_openmm_propanal/chain.conf   ai" >&2
    echo "   no partition -> run it here, now." >&2
    exit 2
fi
[ -f "$CONF" ] || { echo "no such conf: $CONF" >&2; exit 2; }

# ---------------------------------------------------------------------------------------
# Local: no scheduler, no .slurm, just do the work.
# ---------------------------------------------------------------------------------------
if [ -z "$PARTITION" ] || [ "$PARTITION" = "local" ]; then
    exec bash examples/chain_body.sh "$CONF"
fi

# ---------------------------------------------------------------------------------------
# Which queue, and which command submits to it.
#
# TianheXY-CN is stock Slurm and takes `sbatch`; the GPU clusters take the site's
# `yhbatch` wrapper. Measured on the machine 2026-09-09.
# ---------------------------------------------------------------------------------------
case "$PARTITION" in
    deimos|debug) SUBMIT="sbatch";  KIND="cpu" ;;
    ai|temp|h100x) SUBMIT="yhbatch"; KIND="gpu" ;;
    *) echo "partition must be deimos, debug, ai, temp, h100x or local -- got '$PARTITION'" >&2
       exit 2 ;;
esac
SUBMIT="${OPENQHA_SUBMIT:-$SUBMIT}"
SLURMFILE="examples/slurm/${PARTITION}.slurm"
[ -f "$SLURMFILE" ] || { echo "no job file for '$PARTITION': $SLURMFILE" >&2; exit 2; }
command -v "$SUBMIT" >/dev/null 2>&1 || {
    echo "openQHA: '$SUBMIT' is not on PATH on this login node." >&2
    echo "  partition '$PARTITION' expects it. If this site uses a different submitter," >&2
    echo "  set OPENQHA_SUBMIT=<command>." >&2
    exit 2; }

# ---------------------------------------------------------------------------------------
# Read the conf -- ONLY to check the submission makes sense. No imports, no torch, no
# model loading: this runs on a login node.
# ---------------------------------------------------------------------------------------
# shellcheck disable=SC1090
source "$CONF"
SPECIES="${SPECIES:?the conf must set SPECIES}"
TAG="${TAG:-chain}"
CHAIN="${CHAIN:-qha}"

case "$CHAIN" in
    conformers|conformers_pair|qha|levels|identity) ;;
    *) echo "CHAIN must be conformers, conformers_pair, qha, levels or identity, got '$CHAIN'" >&2
       exit 2 ;;
esac

# Two chains cannot run on a GPU queue, for the same reason: the tool has no GPU path and
# is not even installed in `openqha-gpu`. Refusing here costs a second; refusing in the
# job costs the queue wait and the allocation.
if [ "$KIND" = "gpu" ]; then
    case "$CHAIN" in
        conformers|conformers_pair)
            echo "CHAIN=conformers is CREST + GFN2-xTB: xtb has no GPU path and" >&2
            echo "  openqha-gpu contains neither crest nor xtb. Use deimos or debug." >&2
            exit 2 ;;
        levels)
            echo "CHAIN=levels is ORCA RI-MP2: no GPU path here, and D0-75 puts" >&2
            echo "  production quantum chemistry on deimos. Use deimos." >&2
            exit 2 ;;
    esac
fi

# Does step 1 exist? A pure filesystem question -- the basin store's layout is
# <root>/<tag>/<shard>/<species>.basins.json -- so it needs no python and no imports.
if [ "$CHAIN" != "conformers" ] && [ "$CHAIN" != "conformers_pair" ] && [ "$KIND" = "gpu" ]; then
    BASINS_ROOT="${S0_BASIN_ROOT:-data/basins}"
    if ! find "$BASINS_ROOT/$TAG" -name "${SPECIES}.basins.json" -print -quit 2>/dev/null \
         | grep -q .; then
        echo "openQHA: no branch A product for '$SPECIES' under tag '$TAG'." >&2
        echo "  Looked in $BASINS_ROOT/$TAG/" >&2
        echo "  Branch A is CREST + GFN2-xTB and cannot run on a GPU queue, so this job" >&2
        echo "  would start and then fail. Run step 1 first, with the SAME tag:" >&2
        echo "    bash examples/run_chain.sh $(dirname "$CONF")/branchA.conf deimos" >&2
        exit 2
    fi
fi

mkdir -p logs        # the .slurm files write --output/--error there, relative to here

echo "submit    $SUBMIT $SLURMFILE"
echo "  conf      $CONF"
echo "  chain     $CHAIN   species $SPECIES   tag $TAG"
echo "  logs      logs/openqha_*_<jobid>.{out,err}"
exec "$SUBMIT" "$SLURMFILE" "$CONF"
