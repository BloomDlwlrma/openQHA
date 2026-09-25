#!/bin/bash
# =============================================================================
# openQHA: verify ONE failed frame's ORCA on TianheXY-CN debug.
#
# The site gate of orca-slurm ticket 01 (the ORCA child environment is
# Slurm-blind; ADR 0008): ONE known-failed draw300 frame, re-run once by hand
# with the existing per-frame retry, must terminate normally. There is no ssh
# from the workstation, so this file is meant to be copied to the Tianhe
# checkout and SUBMITTED BY HAND:
#
#     cd ~/openQHA-main                      # the checkout that carries the fix
#     sbatch verify-orca-one.sh              # debug, 30 min, one exclusive 64-core node
#
# Submit from the checkout root: `logs/slurm/` (the header's log directory) is
# tracked there. Another checkout / frame, via the environment:
#
#     CHECKOUT=~/openQHA-main TAG=draw300 QID=dsgdb9nsd_006885 GEN=displaced \
#         BASIN=1 K=3 sbatch --export=ALL verify-orca-one.sh
#
# It also runs under `bash verify-orca-one.sh` inside an interactive allocation.
#
# ORDER OF WORK: the hl_labels environment exactly (module purge, the NARROW
# unset, common.sh -> tianhe.sh -> orca.sh); refuse unless the checkout carries
# the seam; seam self-check INSIDE this job's armed Slurm environment (the
# narrow unset deliberately leaves SLURM_JOBID / SLURM_NODELIST in place --
# that half-armed environment is what the fix is tested against); refuse
# unless the frame reads FAILED on disk (nothing is burned otherwise); show
# the previous failure signature; ONE manual `--retry`, 4 ranks pinned to
# cores 0-3; the verdict read from the FRESH `.out`.
#
# A *displaced* frame is energy+forces only and fits the 30-minute window; a
# Hessian frame (basin/merged/saddle generators) may not -- give those a
# longer `--time` on deimos. If the job is cut at the walltime the retry simply
# did not complete: resubmit (longer --time), nothing else was burned.
#
# Overridable: TAG QID GEN BASIN K LEVEL NPROCS MAXCORE TIMEOUT_S FRAME_CORES
#              CHECKOUT.
# =============================================================================
#SBATCH --job-name=openqha_verify_one
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

# no `set -e` (project rule 2026-09-13); no `set -u` (the conda hooks, common.sh)

# ---- 0. what to verify ---------------------------------------------------------------
TAG="${TAG:-draw300}"
QID="${QID:-dsgdb9nsd_006885}"
GEN="${GEN:-displaced}"
BASIN="${BASIN:-1}"
K="${K:-3}"
LEVEL="${LEVEL:-wb97m-d3bj_def2-tzvppd}"
NPROCS="${NPROCS:-4}"
MAXCORE="${MAXCORE:-6000}"
TIMEOUT_S="${TIMEOUT_S:-28800}"
FRAME_CORES="${FRAME_CORES:-0-3}"
CHECKOUT="${CHECKOUT:-}"

case "$BASIN" in *[!0-9]*) echo "OPENQHA-VERIFY: BASIN must be an integer: $BASIN"; exit 2;; esac
case "$K" in *[!0-9]*) echo "OPENQHA-VERIFY: K must be an integer: $K"; exit 2;; esac

CORES="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-$(nproc --all)}}"

# ---- 1. site environment: the hl_labels / hl_pipeline_debug shape, exactly ------------
module purge 2>/dev/null || true
module load anaconda3/202309 2>/dev/null || module load anaconda3/2023.09 2>/dev/null || true
# The NARROW unset (hpc/slurm/README.md rule 2): scheduling variables for the
# non-ORCA payloads. It deliberately LEAVES SLURM_JOBID / SLURM_NODELIST armed.
for v in $(env | awk -F= '{print $1}' | grep -E '^(PMI|SLURM_(CPU|TASK|NTASKS|NPROCS|STEP))'); do
    unset "$v"
done
if [ -n "$CHECKOUT" ]; then
    cd "$CHECKOUT" || { echo "OPENQHA-VERIFY: no checkout at $CHECKOUT"; exit 2; }
else
    cd "${SLURM_SUBMIT_DIR:-$PWD}" || { echo "OPENQHA-VERIFY: no submit dir"; exit 2; }
fi
export OPENQHA_PARTITION="${SLURM_JOB_PARTITION:-debug}"
source hpc/env/common.sh
source hpc/env/tianhe.sh || { echo "OPENQHA-VERIFY: hpc/env/tianhe.sh refused"; exit 2; }
source hpc/env/orca.sh
openqha_find_orca || { echo "OPENQHA-VERIFY: no ORCA -- the seam cannot be tested"; exit 2; }
openqha_report_env
echo "OPENQHA-VERIFY: checkout  $PWD"
echo "OPENQHA-VERIFY: job       ${SLURM_JOB_ID:-?} on $(hostname) partition ${SLURM_JOB_PARTITION:-?}"
echo "OPENQHA-VERIFY: runs root $S0_RUNS_ROOT"
echo "OPENQHA-VERIFY: frame     $TAG/$QID $GEN b$BASIN k$K level $LEVEL"
echo "OPENQHA-VERIFY: layout    $NPROCS ranks, frame pinned to cores $FRAME_CORES of $CORES"

# ---- 2. does THIS checkout carry the fix? ---------------------------------------------
if ! grep -qF 'OMPI_MCA_hwloc_base_binding_policy' openqha/qm_interfaces/orca.py; then
    echo "OPENQHA-VERIFY: VERDICT: FIX-NOT-DEPLOYED -- no seam in openqha/qm_interfaces/orca.py"
    echo "  sync the checkout to the fix commit (994df36 or later) and resubmit."
    exit 2
fi
echo "OPENQHA-VERIFY: seam present in openqha/qm_interfaces/orca.py"

# ---- 3. the seam as DEPLOYED, inside this armed job ------------------------------------
echo
echo "== seam self-check (expect: none / none)"
python - <<'PY'
from openqha.qm_interfaces import orca
env = orca.subprocess_env()
bad = sorted(k for k in env if k.startswith(("SLURM", "PMI")))
knob = env.get("OMPI_MCA_hwloc_base_binding_policy")
print("OPENQHA-VERIFY: seam stripped:", bad or "none")
print("OPENQHA-VERIFY: seam binding knob:", knob)
raise SystemExit(0 if (not bad and knob == "none") else 3)
PY
SEAM=$?
if [ "$SEAM" -ne 0 ]; then
    echo "OPENQHA-VERIFY: VERDICT: SEAM-FAIL -- subprocess_env() is not the fixed one"
    echo "  (see the traceback above; this is the failure the job exists to catch)"
    exit 3
fi

# ---- 4. the frame must read FAILED, or nothing is attempted ----------------------------
MOL="$S0_RUNS_ROOT/$TAG/$QID"
STEM="orca.${LEVEL}.${GEN}_b$(printf '%02d' "$BASIN")_k${K}"
OUT="$MOL/frames/$STEM.out"
echo
echo "== frame state before any ORCA"
python - "$MOL" "$STEM" "$LEVEL" "$GEN" "$BASIN" "$K" <<'PY'
import sys
from openqha.store import layout
from openqha.data import frame_labels
mol, stem, level, gen, basin, k = sys.argv[1:7]
folder = layout.frames_dir(mol)
out = layout.orca_frame_file(mol, level, gen, int(basin), int(k), ".out")
finished = frame_labels.finished(folder, stem, hessian=frame_labels.wants_hessian(gen))
failed = frame_labels.failed(folder, stem)
running = frame_labels.running_elsewhere(folder, stem)
print("OPENQHA-VERIFY: frame path", out)
print("OPENQHA-VERIFY: state finished", finished, "failed", failed, "running", running)
raise SystemExit(0 if (failed and not finished and not running) else 4)
PY
PRE=$?
if [ "$PRE" -eq 4 ]; then
    echo "OPENQHA-VERIFY: VERDICT: NOT-A-FAILED-FRAME -- nothing was run, nothing burned"
    echo "  the frame reads finished, held, or never attempted; pick a failed frame."
    exit 4
elif [ "$PRE" -ne 0 ]; then
    echo "OPENQHA-VERIFY: VERDICT: PRECHECK-FAILED -- see the traceback above"
    exit 5
fi

# ---- 5. keep the old evidence, then ONE manual retry -----------------------------------
mkdir -p "$HOME/orca_verify_backup"
cp -p "$OUT" "$HOME/orca_verify_backup/${STEM}.old.out" 2>/dev/null \
    && echo "OPENQHA-VERIFY: old .out saved to $HOME/orca_verify_backup/${STEM}.old.out"
echo
echo "== the previous failure, read once more before the retry overwrites the file"
grep -m6 -E 'ras_base_allocate|SLURM_TASKS_PER_NODE|aborting the run' "$OUT" 2>/dev/null \
    || echo "(no ORTE signature line found)"
echo
echo "== the retry: $QID $GEN b$BASIN k$K --retry"
if command -v taskset >/dev/null 2>&1; then
    taskset -c "$FRAME_CORES" python -u -m openqha.data.frame_labels "$MOL" "$GEN" "$BASIN" "$K" \
        --level "$LEVEL" --nprocs "$NPROCS" --maxcore "$MAXCORE" --timeout "$TIMEOUT_S" --retry
else
    python -u -m openqha.data.frame_labels "$MOL" "$GEN" "$BASIN" "$K" \
        --level "$LEVEL" --nprocs "$NPROCS" --maxcore "$MAXCORE" --timeout "$TIMEOUT_S" --retry
fi
RC=$?

# ---- 6. the verdict, from the fresh .out itself ----------------------------------------
echo
echo "== verdict"
VERDICT=FAILED
if [ -f "$OUT" ] && grep -qF '****ORCA TERMINATED NORMALLY****' "$OUT"; then
    VERDICT=NORMALLY
    tail -n 5 "$OUT"
else
    if [ -f "$OUT" ]; then
        echo "(the .out carries no terminal line; tail follows)"
        tail -n 12 "$OUT"
    else
        echo "(no .out: the retry was cut or never wrote -- nothing burned; resubmit)"
    fi
fi
echo "OPENQHA-VERIFY: frame command exit code $RC"

# ---- 7. the paste-back block -----------------------------------------------------------
echo
echo "================= paste this back ================="
echo "job        ${SLURM_JOB_ID:-?} on $(hostname) partition ${SLURM_JOB_PARTITION:-?}"
echo "checkout   $PWD"
echo "log        $PWD/logs/slurm/openqha_verify_one_${SLURM_JOB_ID:-jobid}.out"
echo "frame      $MOL"
echo "stem       $STEM"
echo "out        $OUT"
echo "exit code  $RC"
echo "VERDICT:   $VERDICT"
if [ "$VERDICT" = NORMALLY ]; then
    echo "site gate: PASSED -- the fresh .out carries ****ORCA TERMINATED NORMALLY****"
    echo "next: count the failures (python scripts/tooling/s0_hl_progress.py --tag $TAG),"
    echo "then the retry round -- orca-slurm ticket 02, NOT implemented yet."
    exit 0
fi
echo "site gate: NOT PASSED -- paste the log's tail and the .out tail back."
exit 1
