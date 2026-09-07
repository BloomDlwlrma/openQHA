#!/bin/bash
# One molecule, end to end, on TianheXY-AI h100x. The supported way in.
#
#     bash examples/02_qha_openmm_acetone/run_on_tianhe_ai.sh            # acetone, full
#     bash examples/02_qha_openmm_acetone/run_on_tianhe_ai.sh --smoke    # 20 ps, 30 min
#     SPECIES=dsgdb9nsd_000035 bash examples/02_qha_openmm_acetone/run_on_tianhe_ai.sh
#
# WHAT THIS DOES THAT `yhbatch <script>` DOES NOT
#   * makes the log directory FIRST. Slurm opens --output before the job script runs, so
#     a directory the script would create is created too late and the job dies with no
#     log saying why.
#   * checks the weights are in place and hash correctly, on the LOGIN node, before
#     spending an allocation on discovering they are not.
#   * records what was submitted next to the logs, so "which settings produced this" has
#     an answer later.
#   * offers --smoke, which is the thing to run first.
set -eo pipefail

SMOKE=0
[ "${1:-}" = "--smoke" ] && SMOKE=1

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$HERE"

SPECIES="${SPECIES:-dsgdb9nsd_000018}"
PARTITION="${PARTITION:-h100x}"

if [ "$SMOKE" = "1" ]; then
    # 20 ps of production is a SMOKE LENGTH and is labelled as one. Branch B's acceptance
    # criteria 1 and 5 are refused below 20 ps, because a 0.4 ps run once passed the
    # saturation criterion and the reason it passed was that it had not begun to rise.
    TAG="${TAG:-acetone_h100x_smoke}"
    WALLTIME="00:30:00"
    PROD_PS=20
    EQUIL_PS=5
    SEEDS=1
    BUDGET=$(( 90 * 1800 / 100 ))
else
    TAG="${TAG:-acetone_h100x}"
    WALLTIME="3-00:00:00"
    PROD_PS=""          # empty -> configs/branchB_protocol.yaml (1500 ps)
    EQUIL_PS=""         # empty -> configs/branchB_protocol.yaml (520 ps)
    SEEDS="${SEEDS:-3}"
    BUDGET=$(( 90 * 3 * 24 * 3600 / 100 ))
fi

LOGDIR="${S0_RUNS_ROOT:-$HOME/HDD_POOL/runs/openQHA}/logs"
mkdir -p "$LOGDIR"

module load anaconda3/2023.09 2>/dev/null || true
OPENQHA_ENV="${OPENQHA_ENV:-openqha-gpu}" source hpc/env/tianhe.sh 2>/dev/null || true

# Refuse on the login node rather than in the queue. A missing or truncated weight file
# is the commonest way to waste an allocation, and it costs a second to rule out.
python - <<'PY' || exit 1
import sys
try:
    from openqha import engine, qha
    p = engine.provenance()
except Exception as exc:                                          # noqa: BLE001
    sys.exit("openQHA: cannot load the potential here -- {}: {}\n"
             "  The MACE-OFF weights are NOT downloaded on a login node. Fetch them\n"
             "  where you have bandwidth and copy them in:\n"
             "      rsync -a data/potentials/ <tianhe>:$HOME/openQHA/data/potentials/\n"
             "  See docs/branchA_production.md section 4.".format(
                 type(exc).__name__, exc))
print("engine   {}  sha256 {}  pinned={}".format(
    p["engine"], p["sha256"][:16], p["sha256_pinned"]))
q = qha.protocol()
print("protocol {} ps + {} ps, frame every {} ps  ({})".format(
    q["equilibration_ps"], q["production_ps"], q["sampling_interval_ps"], q["_source"]))
PY

echo
echo "species    $SPECIES"
echo "tag        $TAG"
echo "partition  $PARTITION   walltime $WALLTIME"
if [ "$SMOKE" = "1" ]; then
    echo "mode       SMOKE -- ${PROD_PS} ps production, ${SEEDS} seed."
    echo "           NOT a result: branch B refuses criteria 1 and 5 below 20 ps."
else
    echo "mode       PRODUCTION -- the published protocol from"
    echo "           configs/branchB_protocol.yaml"
fi
echo

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT=$(yhbatch \
    --partition="$PARTITION" --time="$WALLTIME" --gpus=1 \
    --export=ALL,SPECIES="$SPECIES",TAG="$TAG",SEEDS="$SEEDS",PROD_PS="$PROD_PS",EQUIL_PS="$EQUIL_PS",WALL_BUDGET_S="$BUDGET" \
    --output="$LOGDIR/openqha_full_${TAG}_%j.out" \
    --error="$LOGDIR/openqha_full_${TAG}_%j.err" \
    examples/02_qha_openmm_acetone/tianhe_ai_h100x.slurm)
echo "$OUT"

JOBID="$(echo "$OUT" | grep -oE '[0-9]+' | tail -1)"
cat > "$LOGDIR/submit_full_${TAG}_${STAMP}.json" <<JSON
{
  "submitted_at": "$STAMP",
  "job_id": "$JOBID",
  "species": "$SPECIES",
  "tag": "$TAG",
  "partition": "$PARTITION",
  "walltime": "$WALLTIME",
  "seeds": "$SEEDS",
  "prod_ps": "${PROD_PS:-from configs/branchB_protocol.yaml}",
  "equil_ps": "${EQUIL_PS:-from configs/branchB_protocol.yaml}",
  "wall_budget_s": $BUDGET,
  "smoke": $([ "$SMOKE" = "1" ] && echo true || echo false),
  "script": "examples/02_qha_openmm_acetone/tianhe_ai_h100x.slurm",
  "layout": "1 GPU + 14 CPUs; branch A on the CPUs, 14 trajectories sharing the one card",
  "note": "Resumable: trajectories flush every 250 frames and stop at 90% of the walltime. Resubmitting continues from the frames on disk."
}
JSON
echo
echo "recorded   $LOGDIR/submit_full_${TAG}_${STAMP}.json"
echo "watch      yhq -a   |   tail -f $LOGDIR/openqha_full_${TAG}_${JOBID}.out"
