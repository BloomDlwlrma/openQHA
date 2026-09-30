#!/bin/bash
# =======================================================================================
# 02d-2 SUBMIT. Reads settings.tsv, writes one conf per row, does the array arithmetic,
# submits ONE job array whose every task holds G cards and runs G rows at once.
#
#     source /APP/u22/ai_x86/toolshs/set-XY-I.sh            # TianheXY-A only
#     bash examples/02d-2_qha_settings_array/submit_array.sh [--plan] [array.conf]
#     # TianheXY-AI: same line; the cluster is recognised from sinfo (a100x / h100x).
#
# THE ARITHMETIC (Slurm job arrays: slurm.schedmd.com/job_array.html)
#     N     rows in settings.tsv
#     W     workers one row needs = SEEDS x basins (one trajectory per worker)
#     R     = min(N, CPUS_PER_CARD / W_max)   rows per card: the CPUs a card brings,
#                                             one per worker (ROWS_PER_CARD overrides)
#     C     = ceil(N / R)                     cards needed
#     T     = ceil(C / GPUS_PER_JOB)          array tasks  (--array=0-(T-1))
#     G     = ceil(C / T)                     cards per task, balanced (--gpus=G)
#     row   = (task_id * G + k) * R + j       k = card in task, j = slot on that card
#
# Until 2026-09-13 a row WAS a card (SEEDS=3 -> 3 workers, 9 CPUs idle). With one
# trajectory per basin a 12-CPU card takes twelve single-basin rows,
# so the 9-row grid is ONE job on ONE card -- the user's question on an113, and the
# answer. Every array task is one Slurm job with `--gpus=G --cpus-per-task=<CPUs> x G`
# (per-card allocation on both GPU clusters; -G mandatory; --mem forbidden). Inside it,
# array.slurm launches G x R copies of examples/chain_body.sh, R per card, each fenced
# to its card via S0_CARD=k CUDA_VISIBLE_DEVICES=k and to its own scratch via
# S0_SCRATCH_TAG=card<k>_row<j>, and waits for all of them.
#
# WHAT IS A FILE AND WHAT IS NOT. Each row's settings are written to
# generated/<NAME>.conf BEFORE submission and the job sources that file, so what ran is
# something you can diff -- the same rule as every other conf here. generated/manifest
# lists them in row order and is the only thing the job reads to find its work.
# =======================================================================================
# `set -eo pipefail` is deliberately not used: a failing step must not end the job.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
HERE=examples/02d-2_qha_settings_array

PLAN=0
CONF="$HERE/array.conf"
for a in "$@"; do
    case "$a" in
        --plan) PLAN=1 ;;
        *) CONF="$a" ;;
    esac
done
[ -f "$CONF" ] || { echo "no such conf: $CONF" >&2; exit 2; }
# shellcheck disable=SC1090
source "$CONF"
SPECIES="${SPECIES:?array.conf must set SPECIES}"
BASIN_TAG="${BASIN_TAG:?array.conf must set BASIN_TAG}"
TAG_PREFIX="${TAG_PREFIX:-02d2}"
SETTINGS="${SETTINGS:-$HERE/settings.tsv}"
GPUS_PER_JOB="${GPUS_PER_JOB:-8}"
WALLTIME="${WALLTIME:-36:00:00}"     # see array.conf: p1500 needs ~25.8 h
SUBMIT="${OPENQHA_SUBMIT:-yhbatch}"
[ -f "$SETTINGS" ] || { echo "no settings file: $SETTINGS" >&2; exit 2; }

# ---- the basins must exist, once, before N jobs go looking for them --------------------
# Branch A's product is <root>/<BASIN_TAG>/<range>/<chunk>/<SPECIES>/mace/basinNN/
# basin.extxyz; hpc/env/root.sh FINDS it and counts the basins (a `find`,
# so the shard rule stays spelled only in openqha/store/layout.py). A login node can
# activate the environment and run python; the shell form was accepted by the user on
# 2026-09-14 as part of the work. The root is derived from the partition.
# shellcheck disable=SC1091
source "$ROOT/hpc/env/root.sh"
OPENQHA_PARTITION="${OPENQHA_GPU_PARTITION:-}" openqha_resolve_root || exit 2
MOL_DIR="$(openqha_find_molecule "$BASIN_TAG" "$SPECIES")" || {
    echo "openQHA: no branch A product for '$SPECIES' under BASIN_TAG '$BASIN_TAG'." >&2
    echo "  Looked under $S0_RUNS_ROOT/$BASIN_TAG/ for $SPECIES/mace/basin00/basin.extxyz." >&2
    echo "  Every row reads it. Make it once, on the CPU cluster:" >&2
    echo "    bash examples/run_chain.sh examples/02d_qha_frequency_identity/branchA.conf deimos            # acetone" >&2
    echo "    bash examples/run_chain.sh examples/02d_qha_frequency_identity/branchA-propanal.conf deimos   # propanal" >&2
    ls -1 "$S0_RUNS_ROOT" 2>/dev/null | sed 's/^/  tags present: /' >&2
    exit 2
}

# ---- how many basins: it sets the workers per row ------------------------------------------
N_BASINS="$(openqha_count_basins "$MOL_DIR")"
case "$N_BASINS" in ''|0|*[!0-9]*) echo "openQHA: could not count basins in $MOL_DIR/mace; assuming 1" >&2; N_BASINS=1 ;; esac
W_MAX=0

# ---- read the rows -----------------------------------------------------------------------
GEN="$HERE/generated"
mkdir -p "$GEN" logs
MANIFEST="$GEN/manifest"
: > "$MANIFEST"
N=0
while read -r name equil prod sample seeds nucut rest; do
    case "$name" in ''|'#'*) continue ;; esac
    [ -n "$nucut" ] || { echo "settings row '$name' is short: need NAME EQUIL PROD SAMPLE SEEDS NU_CUT" >&2; exit 2; }
    tag="${TAG_PREFIX}_${name}"
    conf="$GEN/${name}.conf"
    # One worker per trajectory; the driver would otherwise start a full card's worth
    # of idle worker processes for every row sharing the card.
    workers=$(( seeds * N_BASINS ))
    [ "$workers" -gt "${W_MAX:-0}" ] && W_MAX=$workers
    # One worker per trajectory; the driver would otherwise start a full card's worth
    # of idle worker processes for every row sharing the card.
    workers=$(( seeds * N_BASINS ))
    [ "$workers" -gt "${W_MAX:-0}" ] && W_MAX=$workers
    cat > "$conf" <<EOF
# generated by $HERE/submit_array.sh from $SETTINGS row '$name' on $(date -Is)
# DO NOT EDIT: change settings.tsv and resubmit. This file is what the job sourced.
CHAIN=identity
SPECIES=$SPECIES
TAG=$tag
BASIN_TAG=$BASIN_TAG
SETTING=$name          # the setting: in the file stems of md_openmm/basinNN/ and _records/ (no setting level)
EQUIL_PS=$equil
PROD_PS=$prod
SAMPLE_EVERY=$sample
SEEDS=$seeds
NU_CUT=$nucut
THREADS=1
MAX_WORKERS=$workers
EOF
    echo "$conf" >> "$MANIFEST"
    N=$((N + 1))
done < "$SETTINGS"
[ "$N" -gt 0 ] || { echo "no rows in $SETTINGS" >&2; exit 2; }

# ---- which GPU cluster, and what one card brings ------------------------------------------
# Decided by what sinfo shows, not by a variable. Until 2026-09-13 this file knew only
# TianheXY-A (`ai`) and, on an113 (TianheXY-AI, a100x), told the operator to source
# set-XY-I.sh -- a file that cluster does not have. OPENQHA_GPU_PARTITION overrides.
SITE_PARTITION="${OPENQHA_GPU_PARTITION:-}"
if [ -z "$SITE_PARTITION" ] && command -v sinfo >/dev/null 2>&1; then
    for p in ai a100x h100x; do
        if sinfo -h -p "$p" -o %G 2>/dev/null | grep -qi gpu; then SITE_PARTITION="$p"; break; fi
    done
fi
# The job quota is the cluster's: MaxSubmit=10 on TianheXY-A (measured 2026-09-11),
# 6 submitted / 6 running / 6 nodes on TianheXY-AI (the account page, 2026-09-13).
case "$SITE_PARTITION" in
    ai)       CPUS_PER_CARD=12; JOB_QUOTA=10 ;;
    a100x)    CPUS_PER_CARD=12; JOB_QUOTA=6 ;;
    h100x)    CPUS_PER_CARD=14; JOB_QUOTA=6 ;;
    *)        CPUS_PER_CARD=12; JOB_QUOTA=6 ;;   # unknown: planned conservatively, refused below unless --plan
esac

# ---- the arithmetic -----------------------------------------------------------------------
R="${ROWS_PER_CARD:-}"
if [ -z "$R" ]; then
    R=$(( CPUS_PER_CARD / W_MAX )); [ "$R" -ge 1 ] || R=1
fi
[ "$R" -le "$N" ] || R=$N
C=$(( (N + R - 1) / R ))                            # cards needed
T=$(( (C + GPUS_PER_JOB - 1) / GPUS_PER_JOB ))     # array tasks
G=$(( (C + T - 1) / T ))                            # cards per task, balanced
CPUS=$(( CPUS_PER_CARD * G ))
LAST=$(( N - (T - 1) * G * R ))                     # rows in the last task
JOB_NAME="openqha_${SPECIES}_${TAG_PREFIX}_array"

echo "02d-2 plan"
echo "  species    $SPECIES     basins from  $BASIN_TAG"
echo "  rows       $N   ($SETTINGS)"
echo "  cluster    ${SITE_PARTITION:-UNKNOWN (no GPU partition visible to sinfo)}   1 card = $CPUS_PER_CARD CPUs   job quota $JOB_QUOTA"
echo "  workers    $W_MAX per row (SEEDS x $N_BASINS basin(s))  ->  $R row(s) per card, $C card(s)"
echo "  array      $T task(s), --array=0-$((T - 1))"
echo "  per task   --gpus=$G --cpus-per-task=$CPUS   ($((G * R)) rows)"
if [ "$LAST" -lt $(( G * R )) ]; then
    echo "  last task  $LAST row(s) on $G card(s)"
fi
echo "  confs      $GEN/<NAME>.conf   manifest $MANIFEST"
i=0
while read -r c; do
    t=$(( i / (G * R) )); k=$(( (i % (G * R)) / R )); j=$(( i % R ))
    printf '    row %2d  task %d card %d slot %d  %s\n' "$i" "$t" "$k" "$j" "$(basename "$c" .conf)"
    i=$((i + 1))
done < "$MANIFEST"
if [ "$T" -gt "$JOB_QUOTA" ]; then
    echo "openQHA: $T array tasks exceed this cluster's job quota of $JOB_QUOTA (every element" >&2
    echo "  counts). Raise ROWS_PER_CARD (now $R) or GPUS_PER_JOB (now $GPUS_PER_JOB), or trim the rows." >&2
    [ "$PLAN" = "1" ] || exit 2
fi

CMD=("$SUBMIT" --job-name="$JOB_NAME" --array="0-$((T - 1))" \
     --partition="${SITE_PARTITION:-ai}" \
     --nodes=1 --ntasks=1 --gpus="$G" --cpus-per-task="$CPUS" --time="$WALLTIME" \
     "$HERE/array.slurm" "$MANIFEST" "$G" "$R")
echo
echo "submit    ${CMD[*]}"
echo "  logs      logs/${JOB_NAME}_<arrayjob>_<task>.{out,err}  and  logs/${TAG_PREFIX}_<NAME>_<arrayjob>_<task>.log"
# --plan ends here, BEFORE any check of the shell: the confs are written and the plan is
# printed whatever machine this is (a compute node, a laptop). The generated confs are
# what hpc/tools/test30.sh runs.
[ "$PLAN" = "1" ] && { echo "  (--plan: not submitted; test one row with  bash hpc/tools/test30.sh $GEN/<NAME>.conf)"; exit 0; }

# ---- the environment check, for a real submission only ------------------------------------
if [ "${OPENQHA_SKIP_ENV_CHECK:-}" != "1" ] && [ -z "$SITE_PARTITION" ]; then
    echo "openQHA: no GPU partition is visible to sinfo (tried ai, a100x, h100x), so a card" >&2
    echo "  request here would be refused. On TianheXY-A this means the DEFAULT Slurm" >&2
    echo "  environment; enter the fine-grained one first:" >&2
    echo "      source /APP/u22/ai_x86/toolshs/set-XY-I.sh" >&2
    echo "  On TianheXY-AI, name the partition: OPENQHA_GPU_PARTITION=a100x" >&2
    echo "  (OPENQHA_SKIP_ENV_CHECK=1 to override.)" >&2
    exit 2
fi
exec "${CMD[@]}"
