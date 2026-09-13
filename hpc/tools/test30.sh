#!/bin/bash
# =======================================================================================
# Run one chain conf IN THE FOREGROUND on the compute node you are already on, so every
# line it prints is on your terminal and every file it writes is one `ls` away.
#
#     source /APP/u22/ai_x86/toolshs/set-XY-I.sh          # login node, once
#     bash hpc/tools/gpu_shell.sh 1 02:00:00               # -> a shell on a card
#     conda activate openqha-gpu                           # on the compute node
#     bash hpc/tools/test30.sh examples/02b_qha_openmm_propanal/test30.conf
#     bash hpc/tools/test30.sh examples/02d_qha_frequency_identity/test30.conf
#     bash hpc/tools/test30.sh examples/02d-2_qha_settings_array/generated/t5_s2.conf
#
# This is `examples/chain_body.sh` with the two variables a .slurm would export, a log
# that is also on screen, and THREE REFUSALS that exist because each one was a way to
# spoil a production result while "just testing":
#
#   1. not inside an allocation -- chain_body keys its scratch on SLURM_JOB_ID, and a
#      run on the login node is not a test of the compute node;
#   2. the conf's TAG already has branch A basins -- then it IS a production tag, and
#      the trajectory driver would RESUME production from this test's 5 ps of frames;
#   3. no openqha environment active -- the failure would be a torch import 30 s in,
#      indistinguishable at a glance from a real defect.
#
# Nothing here catches a failure and hides it: chain_body's own must() stops the chain
# at the first producing step that fails, and the exit code is passed through.
# =======================================================================================
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || { echo "openQHA: cannot cd to $ROOT" >&2; exit 1; }

CONF="${1:?usage: bash hpc/tools/test30.sh <conf>}"
[ -f "$CONF" ] || { echo "openQHA: no such conf: $CONF" >&2; exit 2; }

# ---- 1. on a compute node, inside an allocation ---------------------------------------
if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "openQHA: SLURM_JOB_ID is not set, so this is not a compute-node shell." >&2
    echo "  Get one first:  bash hpc/tools/gpu_shell.sh 1 02:00:00" >&2
    exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
    echo "openQHA: no card visible here (nvidia-smi). Was the shell allocated with --gpus?" >&2
    exit 2
fi

# ---- 2. the conf must not write under a production tag --------------------------------
# Read TAG / BASIN_TAG the way chain_body will: by sourcing, in a subshell.
# '|' as the separator, not a space: an empty BASIN_TAG would otherwise collapse and
# SPECIES would be read into its slot.
IFS='|' read -r TAG BASIN_TAG SPECIES <<<"$(
    # shellcheck disable=SC1090
    . "$CONF" >/dev/null 2>&1
    printf '%s|%s|%s\n' "${TAG:-}" "${BASIN_TAG:-}" "${SPECIES:-}"
)"
[ -n "$TAG" ]     || { echo "openQHA: $CONF sets no TAG" >&2; exit 2; }
[ -n "$SPECIES" ] || { echo "openQHA: $CONF sets no SPECIES" >&2; exit 2; }
if [ -d "data/basins/$TAG" ]; then
    echo "openQHA: REFUSING. data/basins/$TAG exists, so '$TAG' is a production tag." >&2
    echo "  A test run under it would write frames the production driver later RESUMES" >&2
    echo "  from. Test confs write under their own TAG and read basins via BASIN_TAG:" >&2
    echo "      TAG=${TAG}_t30" >&2
    echo "      BASIN_TAG=$TAG" >&2
    exit 2
fi
if [ -n "$BASIN_TAG" ] && [ ! -d "data/basins/$BASIN_TAG" ]; then
    echo "openQHA: BASIN_TAG=$BASIN_TAG has no basins under data/basins/. Branch A cannot" >&2
    echo "  run on a GPU node (no crest/xtb), so there is nothing to start from." >&2
    exit 2
fi

# ---- 3. the environment -----------------------------------------------------------------
case "${CONDA_PREFIX:-}" in
    *openqha*) ;;
    *) echo "openQHA: no openqha conda environment is active (CONDA_PREFIX=${CONDA_PREFIX:-unset})." >&2
       echo "  conda activate openqha-gpu" >&2
       exit 2 ;;
esac

# ---- run ------------------------------------------------------------------------------
export OPENQHA_PARTITION="${OPENQHA_PARTITION:-ai}"
export OPENQHA_KIND=gpu
export PYTHONNOUSERSITE=1
mkdir -p logs
LOG="logs/test30_${TAG}_${SLURM_JOB_ID}.log"

echo "test30    $CONF"
echo "  species $SPECIES   tag $TAG   basins from ${BASIN_TAG:-$TAG}"
echo "  node    $(hostname)   job $SLURM_JOB_ID   card(s) $(nvidia-smi -L | wc -l)"
echo "  log     $LOG   (also on this terminal)"
echo "  when done, this test's outputs are under analysis/qha/$TAG/ and can be deleted"
echo

bash examples/chain_body.sh "$CONF" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
echo
if [ "$rc" -eq 0 ]; then
    echo "test30    PASSED  ($CONF)   log $LOG"
else
    echo "test30    FAILED  exit $rc  ($CONF)"
    echo "  the last 'openQHA: step failed' line above names the step; the log is $LOG"
fi
exit "$rc"
