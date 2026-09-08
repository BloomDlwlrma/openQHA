#!/bin/bash
# debug_realmole -- the whole openQHA chain on one real molecule, over a grid.
#
#     bash examples/02_qha_openmm_acetone/run_debug_realmole.sh              # smoke, local
#     bash examples/02_qha_openmm_acetone/run_debug_realmole.sh --production # full grid
#     bash examples/02_qha_openmm_acetone/run_debug_realmole.sh --tianhe     # submit it
#
# The grid lives in a `.conf`, in the convention of
# 00_QM9_reaction_eng/hkuhpc/REPT-dNN/search/*.conf -- `<NAME>_LIST=(...)` is swept,
# `export NAME="${NAME:-x}"` is fixed, every cell appends a row to $RESULT_LOG.
#
# WHAT IT COSTS, BEFORE YOU START IT
# ----------------------------------
# The full grid is 5 lengths x 3 intervals x 5 thermostats x 2 atom sets = 150 cells, but
# length is a truncation, interval is a stride and the atom set is a mask -- all read off
# ONE trajectory. So the real cost is
#
#     n_basins x SEEDS x n_thermostats  trajectories, each at the LONGEST length
#
# For acetone that is (basins) x 3 x 5 at 1500 ps. At the only figure this repository
# has -- 96.1 s/ps, one CPU thread, uncontended -- one trajectory is ~40 hours. **Run the
# smoke first, then --tianhe.** Do not start the full grid on a workstation.
set -eo pipefail

MODE="smoke"
case "${1:-}" in
    --production) MODE="production" ;;
    --tianhe)     MODE="tianhe" ;;
    --smoke|"")   MODE="smoke" ;;
    *) echo "usage: $0 [--smoke|--production|--tianhe]" >&2; exit 2 ;;
esac

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$HERE"
EX="examples/02_qha_openmm_acetone"

if [ "$MODE" = "smoke" ]; then
    CONF="$EX/debug_realmole_smoke.conf"
else
    CONF="$EX/debug_realmole.conf"
fi
echo "conf   $CONF"

# Show what was understood BEFORE spending anything. The parser ignores lines it does not
# recognise, so a typo in the .conf is otherwise invisible until the grid comes back the
# wrong shape.
python "$EX/s0_debug_realmole.py" --conf "$CONF" --print-conf

if [ "$MODE" = "tianhe" ]; then
    LOGDIR="${S0_RUNS_ROOT:-$HOME/HDD_POOL/runs/openQHA}/logs"
    mkdir -p "$LOGDIR"
    echo
    echo "submitting the full grid to TianheXY-A (temp for the smoke, ai for production)"
    yhbatch --partition=ai --time=7-00:00:00 --gpus=8 \
        --export=ALL,CONF="$CONF" \
        --output="$LOGDIR/debug_realmole_%j.out" \
        --error="$LOGDIR/debug_realmole_%j.err" \
        "$EX/debug_realmole_tianhe.slurm"
    exit 0
fi

echo
python -u "$EX/s0_debug_realmole.py" --conf "$CONF"
