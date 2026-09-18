#!/bin/bash
# Workflow hessian_learning, steps 01 -> 02 -> [03] -> 04 in order, on this machine.
# PRODUCTION.
#
#   bash workflows/hessian_learning/run.sh --tag rings [--tag propanal] --name smoke [--limit N] [--stratify] [--with-labels]
#
# 03 (reference labels, ORCA) runs only with --with-labels: it is minutes per frame here
# and a tianhe Batch in production (README). Without it the Dataset is all pool, which
# is what steps 05/06 will refuse and what a plumbing test wants to see. 05 and 06 refuse
# until round 2 is ruled. No `set -e` (project rule, 2026-09-13): every step's status is
# checked explicitly.
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tags=(); name=""; limit=""; stratify=""; labels=0; level=""
while [ $# -gt 0 ]; do
    case "$1" in
        --tag) tags+=("$2"); shift 2 ;;
        --name) name="$2"; shift 2 ;;
        --limit) limit="$2"; shift 2 ;;
        --stratify) stratify="--stratify"; shift ;;
        --with-labels) labels=1; shift ;;
        --level) level="$2"; shift 2 ;;
        *) echo "run.sh: unknown argument $1" >&2; exit 2 ;;
    esac
done
[ ${#tags[@]} -gt 0 ] && [ -n "$name" ] || { echo "usage: run.sh --tag T [--tag T2] --name NAME [--limit N] [--stratify] [--with-labels] [--level L]" >&2; exit 2; }
tagargs=(); for t in "${tags[@]}"; do tagargs+=(--tag "$t"); done
lvl=(); [ -n "$level" ] && lvl=(--level "$level")

echo "== 01 select"
python -u "$here/01_select.py" "${tagargs[@]}" --name "$name" ${limit:+--limit "$limit"} $stratify || { echo "01 failed" >&2; exit 1; }
echo "== 02 frames"
for t in "${tags[@]}"; do
    python -u "$here/02_frames.py" --tag "$t" --all || { echo "02 failed for tag $t" >&2; exit 1; }
done
# select.dat snapshots has_frames / n_frames; refresh it now that 02 has run (04 re-reads
# the Frame-set Records from disk anyway, so this is for the record, not for correctness)
python -u "$here/01_select.py" "${tagargs[@]}" --name "$name" ${limit:+--limit "$limit"} $stratify > /dev/null || { echo "01 (refresh) failed" >&2; exit 1; }
if [ "$labels" = 1 ]; then
    echo "== 03 labels (local, in process)"
    for t in "${tags[@]}"; do
        python -u "$here/03_labels.py" --tag "$t" --name "$name" --local "${lvl[@]}" || echo "03: a frame failed under $t (see the Batch table); continuing" >&2
    done
else
    echo "== 03 labels: skipped (--with-labels to run here; tianhe Batch in production)"
fi
echo "== 04 dataset"
python -u "$here/04_dataset.py" "${tagargs[@]}" --name "$name" "${lvl[@]}" || { echo "04 failed" >&2; exit 1; }
echo "== 05 train / 06 judge: round 2 open -- not run"
