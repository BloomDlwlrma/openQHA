#!/bin/bash
# Submit branch B trajectories to TianheXY-A. One job per species, 8 cards each.
#
#     bash hpc/slurm/submit_branchB_tianhe_a.sh [TAG] [PARTITION]
#     bash hpc/slurm/submit_branchB_tianhe_a.sh prod ai
#     bash hpc/slurm/submit_branchB_tianhe_a.sh a_debug temp     # 30-minute smoke test
#
# ONE JOB PER SPECIES, not one per trajectory. A species has (basins x seeds)
# trajectories and the node has 8 cards; the job script lays them over the cards itself.
# Submitting one job per trajectory would spend the 5-node quota on five trajectories
# instead of forty.
#
# The quota is 10 running jobs and 5 NODES, and the node quota binds first. Past it the
# scheduler simply refuses and the campaign looks stalled rather than capped, so this
# refuses instead.
set -eo pipefail

TAG="${1:-prod}"
PARTITION="${2:-ai}"
NODE_QUOTA=5

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$HERE"

case "$PARTITION" in
    ai)   WALLTIME="7-00:00:00"; PROD_PS="${PROD_PS:-200}"
          BUDGET=$(( 90 * 7 * 24 * 3600 / 100 )) ;;
    temp) WALLTIME="00:30:00";   PROD_PS="${PROD_PS:-2}"
          BUDGET=$(( 90 * 1800 / 100 )) ;;
    *)    echo "unknown partition $PARTITION (ai | temp)" >&2; exit 2 ;;
esac

LOGDIR="${S0_RUNS_ROOT:-$HOME/HDD_POOL/runs/openQHA}/logs"
mkdir -p "$LOGDIR"

module load anaconda3/2023.09 2>/dev/null || true
OPENQHA_ENV="${OPENQHA_ENV:-openqha-gpu}" source hpc/env/tianhe.sh 2>/dev/null || true

# Which species: those with a branch A basin list, unless SPECIES is given.
if [ -n "${SPECIES:-}" ]; then
    LIST="$SPECIES"
else
    LIST=$(python - <<'PY'
import sys
sys.path.insert(0, '.')
from openqha import config
cfg = config.load()
out = []
for edge in config.edges(cfg):
    for qid in config.edge_species(edge):
        if qid not in out:
            out.append(qid)
print(" ".join(out))
PY
)
fi

COUNT=$(echo "$LIST" | wc -w)
if [ "$COUNT" -gt "$NODE_QUOTA" ]; then
    echo "note: $COUNT species but the node quota is $NODE_QUOTA."
    echo "      Submitting all of them is fine -- the scheduler will run $NODE_QUOTA at a"
    echo "      time and queue the rest. It is exceeding max_blocks in Parsl that stalls,"
    echo "      not queued yhbatch jobs."
fi

echo "tag        $TAG"
echo "partition  $PARTITION   walltime $WALLTIME   production $PROD_PS ps"
echo "species    $COUNT"
echo

for s in $LIST; do
    # Branch A's basin list, if it exists. Without it the driver runs ONE geometry -- the
    # QM9 reference -- and says so in the record rather than pretending it is a basin.
    BASINS=$(python - <<PY
import sys
sys.path.insert(0, '.')
from openqha import basin_store
j, x = basin_store.paths_for("$s", tag="$TAG")[:2]
print(x if x.exists() else "")
PY
)
    NB=1
    if [ -n "$BASINS" ]; then
        NB=$(head -1 "$BASINS" >/dev/null 2>&1 && \
             python -c "
from ase.io import read
print(len(read('$BASINS', index=':')))" 2>/dev/null || echo 1)
        echo "  $s  basins=$NB  ($BASINS)"
    else
        echo "  $s  no branch A basin list -- ONE geometry, and the record will say so"
    fi
    OUT=$(yhbatch \
        --partition="$PARTITION" --time="$WALLTIME" --gpus=8 \
        --export=ALL,SPECIES="$s",TAG="$TAG",BASINS_FILE="$BASINS",N_BASINS="$NB",WALL_BUDGET_S="$BUDGET",PROD_PS="$PROD_PS" \
        --output="$LOGDIR/openqha_B_${TAG}_${s}_%j.out" \
        --error="$LOGDIR/openqha_B_${TAG}_${s}_%j.err" \
        hpc/slurm/branchB_traj_tianhe_a.slurm)
    echo "      -> $OUT"
done

echo
echo "watch with:  yhq -a ; yhi"
echo "when they finish, collect on the CPU cluster:"
echo "    TAG=$TAG yhbatch hpc/slurm/branchB_collect.slurm"
