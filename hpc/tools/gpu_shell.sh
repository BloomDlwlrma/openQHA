#!/bin/bash
# =======================================================================================
# An interactive shell on a GPU card, for testing branch B by hand.
#
#     source /APP/u22/ai_x86/toolshs/set-XY-I.sh     # once per login shell (TianheXY-A)
#     bash hpc/tools/gpu_shell.sh                    # 1 card, 12 CPUs, 2 hours
#     bash hpc/tools/gpu_shell.sh 2 04:00:00         # 2 cards, 4 hours
#
#     OPENQHA_GPU_PARTITION=h100x bash hpc/tools/gpu_shell.sh 1 02:00:00
#                                                    # TianheXY-AI: no set-XY-I.sh there,
#                                                    # one Slurm, standard gres, 14 CPUs
#                                                    # per H100 (128 CPUs / 8 cards) as
#                                                    # examples/slurm/h100x.slurm asks
#
# WHY `yhrun -p temp --gres=gpu:1` IS REFUSED (measured 2026-09-11/12)
# --------------------------------------------------------------------
# TianheXY-A's login node speaks to TWO Slurm controllers:
#
#     default (fresh login)   partitions ai (an[9..43]) and temp.  Gres=(null) on every
#                             node -- the cards are not scheduled, so ANY --gres/-G is
#                             "Invalid generic resource (gres) specification".
#     fine-grained            after `source /APP/u22/ai_x86/toolshs/set-XY-I.sh`
#                             (which is only `PATH=/usr/local/slurm.24051/bin:$PATH`).
#                             ONE partition, `ai`, an[45-47,49-51,53,65]. Cards ARE
#                             scheduled: 1 GPU = 12 CPUs = 120 GB, -G mandatory,
#                             --mem forbidden.
#
# `temp` exists only on the DEFAULT controller, where no card can be requested. So
# `-p temp --gres=gpu:1` is refused whichever shell you are in: in the default one
# because there is no gres, in the fine-grained one because there is no `temp`.
#
# The working form is `-p ai` from the fine-grained environment, which is what this
# script runs -- after checking that you are actually in it.
# =======================================================================================
# `set -eo pipefail` removed 2026-09-13 (user ruling: a failing step must not end the job; .mem/notes/notes_2026-09-13_no-errexit-anywhere.md)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

GPUS="${1:-1}"
WALLTIME="${2:-02:00:00}"
PART="${OPENQHA_GPU_PARTITION:-ai}"
# CPUs per card differ by cluster: 12 is TianheXY-A's fine-grained policy (1 card = 12
# CPUs = 120 GB, billed as such); h100x on TianheXY-AI is 128 CPUs / 8 cards, and the
# repo's h100x.slurm asks 14 (open item 41: 16 would be the full share).
case "$PART" in
    h100x) PER_CARD="${OPENQHA_CPUS_PER_GPU:-14}" ;;
    *)     PER_CARD="${OPENQHA_CPUS_PER_GPU:-12}" ;;
esac
CPUS=$(( GPUS * PER_CARD ))

if ! command -v yhrun >/dev/null 2>&1; then
    echo "openQHA: no yhrun on PATH. This script is for TianheXY-A's login node." >&2
    exit 2
fi
if [ "${OPENQHA_SKIP_ENV_CHECK:-}" != "1" ]; then
    if ! sinfo -h -p "$PART" -o %G 2>/dev/null | grep -qi gpu; then
        echo "openQHA: partition '$PART' shows no gres in this shell, so a card cannot be" >&2
        echo "  requested." >&2
        if [ "$PART" = "ai" ]; then
            echo "  You are in TianheXY-A's DEFAULT Slurm environment. Run:" >&2
            echo "      source /APP/u22/ai_x86/toolshs/set-XY-I.sh" >&2
            echo "  then this script again." >&2
        else
            echo "  Is '$PART' a partition on THIS login node?  sinfo -o '%P %G %N' | head" >&2
        fi
        echo "  (OPENQHA_SKIP_ENV_CHECK=1 to override.)" >&2
        exit 2
    fi
fi

echo "interactive  yhrun -N 1 -n 1 -p $PART --gpus=$GPUS --cpus-per-task=$CPUS -t $WALLTIME --pty /bin/bash"
if [ "$PART" = "ai" ]; then
    echo "  cards      $GPUS   ($CPUS CPUs, $(( GPUS * 120 )) GB -- the site's per-card policy)"
    echo "  NOT passed --mem (forbidden here)"
else
    echo "  cards      $GPUS   ($CPUS CPUs at $PER_CARD per card)"
fi
echo
echo "  NOTE: a compute node has NO outbound network, not even through the site proxy"
echo "  (measured 2026-09-12: 'Failed to connect to 172.16.31.200 port 3138'). Anything"
echo "  that installs packages must be done from THIS login node first, or with"
echo "  'mamba install --offline' from the package cache."
echo
echo "  Once you are on the node, the smoke test is:"
echo "      cd $ROOT"
# `conda activate` needs the shell hook, and a compute node's shell has not run
# `conda init`. Sourcing the hook directly is what works there; ~/init_conda.sh does the
# same thing if it exists, but it also runs whatever else is in it.
echo "      source \$(conda info --base)/etc/profile.d/conda.sh"
echo "      conda activate openqha-gpu"
echo "      PYTHONNOUSERSITE=1 python scripts/tooling/s0_probe_openmm_cuda.py"
echo "  Load a CUDA module ONLY if section 1 says the environment's nvrtc is newer than"
echo "  the driver -- and check afterwards that it did not put a stubs/ directory on"
echo "  LD_LIBRARY_PATH (TianheXY-AI's CUDA/12.4 does; that is CUDA error 34)."
echo
exec yhrun -N 1 -n 1 -p "$PART" --gpus="$GPUS" --cpus-per-task="$CPUS" \
     -t "$WALLTIME" --pty /bin/bash
