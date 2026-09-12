#!/bin/bash
# =======================================================================================
# SUBMIT. Reads a conf, picks the .slurm for the partition, hands it to the scheduler.
# =======================================================================================
#     bash examples/run_chain.sh <conf>                 # run it here, now
#     bash examples/run_chain.sh <conf> deimos          # TianheXY-CN,  CPU, 3 days
#     bash examples/run_chain.sh <conf> debug           # TianheXY-CN,  CPU, 30 min
#     bash examples/run_chain.sh <conf> ai              # TianheXY-A fine-grained, per card, 24 h
#     bash examples/run_chain.sh <conf> ai --gpus 8     # ... or a whole node: 8 cards, 96 CPUs,
#                                                       #     96 branch B trajectories at once
#     bash examples/run_chain.sh <conf> temp            # same queue, 30 min (there is no `temp` there)
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
#     (2026-09-11: the size of the allocation IS passed as flags again, when and only when
#     the conf sets NODES / CPUS / GPUS / WALLTIME. The reason is that the right size is a
#     property of the work -- 3 trajectories do not need 8 cards -- so it belongs in the
#     conf with THREADS and SEEDS. The submit line is echoed in full to keep the job
#     legible; see "THE ALLOCATION" at the foot of this file.)
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
# A conf's settings are plain assignments, so `source` overrides anything the shell set:
# `SPECIES=x bash run_chain.sh conf` submits the conf's species, not x. That is the design
# (what ran is the file), and on 2026-09-11 it silently resubmitted acetone when propanal
# was meant. So it is now a refusal: if the shell had SPECIES or TAG set to something the
# conf then replaced, stop and say which file to use instead.
_SHELL_SPECIES="${SPECIES:-}"; _SHELL_TAG="${TAG:-}"
# shellcheck disable=SC1090
source "$CONF"
SPECIES="${SPECIES:?the conf must set SPECIES}"
for _pair in "SPECIES:$_SHELL_SPECIES:$SPECIES" "TAG:$_SHELL_TAG:${TAG:-}"; do
    _name="${_pair%%:*}"; _rest="${_pair#*:}"; _shell="${_rest%%:*}"; _conf="${_rest#*:}"
    if [ -n "$_shell" ] && [ -n "$_conf" ] && [ "$_shell" != "$_conf" ]; then
        echo "openQHA: your shell has $_name=$_shell but $CONF sets $_name=$_conf, and the" >&2
        echo "  conf wins -- the job sources the FILE, so the shell value would not run." >&2
        echo "  Refusing rather than submitting $_conf under a name you did not intend." >&2
        echo "  Use (or copy) a conf that says $_name=$_shell; the examples carry one per" >&2
        echo "  molecule (branchA.conf / branchA-propanal.conf)." >&2
        exit 2
    fi
done
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

# ---------------------------------------------------------------------------------------
# THE ALLOCATION. The .slurm file carries a default for its queue; a conf may size the
# allocation to the WORK instead. Only what the conf actually sets is passed, so a conf
# that says nothing submits exactly the file you can read -- unchanged behaviour.
#
# This bends the 2026-09-09 rule that the file you read is the job that ran, and it bends
# it on purpose. A `#SBATCH` line cannot say "four cores because THREADS is 4", and the
# alternative was one .slurm per (queue, size). The mitigation is that the exact command
# is echoed below, so the job is still legible -- from the conf plus that echo, rather
# than from the .slurm alone.
#
#   NODES     -N               whole nodes.
#   CPUS      -c               cores for the one task. A queue may still hand over a whole
#                              node anyway (TianheXY-C allocates by node and requires
#                              `--exclusive`), in which case this bounds what is USED, not
#                              what is charged.
#   GPUS      --gpus=N         cards. Mandatory on both GPU clusters' per-card
#                              environments. On ai|temp it defaults to 1 and CPUS follows
#                              as 12 x GPUS, because that is what a card brings and bills.
#                              OPENQHA_GRES=gpu:1 sends the `--gres=` spelling instead
#                              (h100x only; ai|temp always use --gpus).
#   WALLTIME  -t
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# THE JOB NAME carries the settings, so `squeue -o "%.60j"` and the log file name
# (`logs/%x_%j.out`) say what a job IS without opening the conf. A conf may set JOB_NAME
# outright; otherwise:
#
#   identity   openqha_<SPECIES>_identity_e<EQUIL_PS>_p<PROD_PS>_s<SAMPLE_EVERY>_x<SEEDS>_nu<NU_CUT>
#   others     openqha_<SPECIES>_<CHAIN>_<TAG>
#
# Commas become '-' (a comma inside --job-name is legal but breaks every shell pipeline
# that reads squeue). Slurm caps a job name well above this length; squeue's default
# column does not, so widen it: squeue -u $USER -o "%.10i %.70j %.2t %.10M".
# ---------------------------------------------------------------------------------------
if [ -z "${JOB_NAME:-}" ]; then
    case "$CHAIN" in
        identity)
            JOB_NAME="openqha_${SPECIES}_identity_e${EQUIL_PS:-proto}_p${PROD_PS:-proto}_s${SAMPLE_EVERY:-proto}_x${SEEDS:-3}_nu${NU_CUT:-default}" ;;
        *)
            JOB_NAME="openqha_${SPECIES}_${CHAIN}_${TAG}" ;;
    esac
fi
JOB_NAME="$(printf '%s' "$JOB_NAME" | tr ', /' '-_-')"

RES_ARGS=(--job-name="$JOB_NAME")
if [ -n "${NODES:-}" ];    then RES_ARGS+=(--nodes="$NODES"); fi
if [ -n "${CPUS:-}" ];     then RES_ARGS+=(--ntasks=1 --cpus-per-task="$CPUS"); _CPUS_ADDED=1; fi
# ---------------------------------------------------------------------------------------
# TianheXY-A: ONE login node, TWO Slurm environments (site PDF; measured 2026-09-11).
#
#   default        `ai` = an[9..43], Gres=(null), allocated by CPU: ANY card request is
#                  refused -- "Invalid generic resource (gres) specification".
#   fine-grained   after `source /APP/u22/ai_x86/toolshs/set-XY-I.sh`: `ai` = an[44-53],
#                  1 card = 12 CPUs = 120 GB, -G MANDATORY, --mem FORBIDDEN,
#                  billed max(gpus, ceil(cpus/12)).
#
# This project submits to the fine-grained one. The check below is of the EFFECT (what
# `sinfo` says about gres), not of an environment variable, because the script's internals
# are the site's business. OPENQHA_SKIP_ENV_CHECK=1 bypasses it.
# ---------------------------------------------------------------------------------------
case "$PARTITION" in
    ai|temp)
        if [ "${OPENQHA_SKIP_ENV_CHECK:-}" != "1" ] && command -v sinfo >/dev/null 2>&1; then
            if ! sinfo -h -p ai -o %G 2>/dev/null | grep -qi gpu; then
                echo "openQHA: this shell is in TianheXY-A's DEFAULT Slurm environment" >&2
                echo "  (sinfo shows no gres on 'ai'). A card request there is refused." >&2
                echo "  Enter the fine-grained environment first:" >&2
                echo "      source /APP/u22/ai_x86/toolshs/set-XY-I.sh" >&2
                echo "  then submit again. (OPENQHA_SKIP_ENV_CHECK=1 to override.)" >&2
                exit 2
            fi
        fi
        # The bill counts cards; 12 CPUs come with each. Ask for all of them.
        GPUS="${GPUS:-1}"
        CPUS="${CPUS:-$(( GPUS * 12 ))}"
        if [ "$CPUS" -gt $(( GPUS * 12 )) ]; then
            echo "openQHA: CPUS=$CPUS exceeds 12 x GPUS=$GPUS; the site bills the excess" >&2
            echo "  as ceil($CPUS/12) cards. Raise GPUS instead, or lower CPUS." >&2
            exit 2
        fi
        # CPUS was defaulted above, so it is added here (the generic block ran earlier).
        [ -n "${_CPUS_ADDED:-}" ] || RES_ARGS+=(--ntasks=1 --cpus-per-task="$CPUS")
        RES_ARGS+=(--gpus="$GPUS") ;;
    *)
        if [ -n "${OPENQHA_GRES:-}" ]; then
            RES_ARGS+=(--gres="$OPENQHA_GRES")
        elif [ -n "${GPUS:-}" ]; then
            RES_ARGS+=(--gpus="$GPUS")
        fi ;;
esac
if [ -n "${WALLTIME:-}" ]; then RES_ARGS+=(--time="$WALLTIME"); fi

case "$PARTITION" in
    ai|temp)
        for a in "${RES_ARGS[@]}"; do
            case "$a" in --mem*) echo "openQHA: --mem is FORBIDDEN in the fine-grained environment (site PDF)." >&2; exit 2 ;; esac
        done ;;
esac

mkdir -p logs        # the .slurm files write --output/--error there, relative to here

echo "submit    $SUBMIT ${RES_ARGS[*]} $SLURMFILE $CONF"
echo "  conf      $CONF"
echo "  chain     $CHAIN   species $SPECIES   tag $TAG"
echo "  name      $JOB_NAME"
if [ ${#RES_ARGS[@]} -gt 1 ]; then
    echo "  alloc     from the conf: ${RES_ARGS[*]:1}"
    echo "            (overrides the #SBATCH defaults in $SLURMFILE)"
else
    echo "  alloc     the #SBATCH defaults in $SLURMFILE"
fi
echo "  logs      logs/${JOB_NAME}_<jobid>.{out,err}"
exec "$SUBMIT" "${RES_ARGS[@]}" "$SLURMFILE" "$CONF"
