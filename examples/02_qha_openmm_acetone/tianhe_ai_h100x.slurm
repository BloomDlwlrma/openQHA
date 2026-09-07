#!/bin/bash
# ONE MOLECULE, END TO END, ON ONE h100x ALLOCATION. TianheXY-AI.
#
#     bash examples/02_qha_openmm_acetone/run_on_tianhe_ai.sh
#
# ...which is the supported way in. Submitting this file by hand works too:
#
#     mkdir -p $HOME/HDD_POOL/runs/openQHA/logs
#     SPECIES=dsgdb9nsd_000018 TAG=acetone_h100x \
#         yhbatch -p h100x --gpus=1 -t 3-00:00:00 \
#         examples/02_qha_openmm_acetone/tianhe_ai_h100x.slurm
#
# =======================================================================================
# WHAT THIS RUNS
# =======================================================================================
# The whole chain for a single molecule, in one allocation, in order:
#
#   1. branch A   CREST iMTD-GC + MACE refine=sp  ->  a basin list
#   2. branch B   one trajectory per (basin, seed) ->  frames.npy + meta.json
#   3. collect    quasi-harmonic analysis          ->  T*S, the conformational correction
#
# The allocation is **1 GPU + 14 CPUs + 240 GB** (site manual 1.1: that is the h100x
# package, and asking for more of either is a rejected submission, not a slow job).
#
# THE POINT OF THE LAYOUT: USE THE CARD *AND* THE 14 CPUs
# -------------------------------------------------------
# A quasi-harmonic trajectory of a 10-atom molecule cannot fill an 80 GB H100 -- the cost
# is kernel-launch latency and Python, not arithmetic. So step 2 runs **14 trajectories at
# once, all sharing the one card**, one per CPU in the package. One trajectory per
# allocation would leave 13 CPUs and most of the card idle.
#
# Step 1 is the opposite: CREST is CPU-bound (GFN2-xTB has no GPU path at all) and
# internally parallel, so it gets the 14 CPUs and never touches the card. **For a
# campaign branch A belongs on TianheXY-C**; it is here because this example is one
# molecule end to end, and splitting one molecule across two clusters to save a card-hour
# would make the example about scheduling rather than about the method.
#
# =======================================================================================
# THE ARITHMETIC, BEFORE YOU SUBMIT
# =======================================================================================
# Production protocol (configs/branchB_protocol.yaml, Rinaldo & Field 2003):
#     520 ps equilibration + 1500 ps production, one frame per 0.5 ps -> 3000 frames
#
# Cost, on the only number this repository has -- 96.1 s/ps, 10 atoms, ONE CPU THREAD,
# uncontended, on a workstation:
#     (520 + 1500) ps x 96.1 s/ps = 194 000 s = 54 hours per trajectory
#
# **That is a CPU number and this is a GPU job.** It is quoted because it is the only
# measurement that exists, and it is an upper bound if the card helps at all -- and this
# repository's one GPU measurement of branch B (D0-C-5) is 3.5x SLOWER than CPU, on a
# T400. An H100 is not a T400 and that number does not transfer, but nothing has replaced
# it. **Run with PROD_PS small first and read seconds_per_ps_this_run out of meta.json.**
#
# The job is resumable either way: trajectories flush every 250 frames and stop
# themselves at 90% of the walltime, so if 3 days is not enough, resubmit and it
# continues from the frames already on disk.
# =======================================================================================
#SBATCH --job-name=openqha_full_molecule
#SBATCH --partition=h100x
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=3-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# --gpus is MANDATORY here (site manual 6.2.1.1, 6.2.2).
# NO --exclusive: the flag is DISABLED by this scheduler; a job carrying it is refused.
# NO memory request: specifying memory is banned; the system sizes it from the GPU count.

set -eo pipefail
# NOT `set -u` -- the conda GROMACS activation hook fails under it and leaves the
# environment half-built while the script carries on. Measured 2026-09-03.

SPECIES="${SPECIES:-dsgdb9nsd_000018}"     # acetone
TAG="${TAG:-acetone_h100x}"
SEEDS="${SEEDS:-3}"                        # 3 is the minimum that gives a blank control
CREST_THREADS="${CREST_THREADS:-14}"       # the whole package, for one molecule
WORKERS="${WORKERS:-14}"                   # one trajectory per CPU, all on the one card
# 90% of the walltime, so a trajectory stops and flushes rather than being killed
# between a write and a rename.
WALL_BUDGET_S="${WALL_BUDGET_S:-233280}"   # 0.90 x 3 days
PROD_PS="${PROD_PS:-}"                     # empty -> the config's 1500 ps
EQUIL_PS="${EQUIL_PS:-}"                   # empty -> the config's 520 ps

# ---- 1. site environment ----------------------------------------------------------------
module purge 2>/dev/null || true
module load anaconda3/2023.09 2>/dev/null || module load miniforge/24.7.1 2>/dev/null || true
# CUDA 12.3: the version BOTH GPU clusters have (measured from the site module tree
# 2026-09-07 -- there is no CUDA/12.4). No MPI: nothing here needs collectives.
module load CUDA/12.3 || {
    echo "openQHA: module load CUDA/12.3 FAILED -- this job would run on the CPU" >&2
    exit 1
}
# Inherited task-layout variables make a child process misread its allocation and try to
# relaunch itself through the scheduler. CREST forks its own workers and is exposed to it.
for v in $(env | awk -F= '{print $1}' | grep -E '^(PMI|SLURM_(CPU|TASK|NTASKS|NPROCS|STEP))'); do
    unset "$v"
done

cd "${SLURM_SUBMIT_DIR:-$PWD}"
OPENQHA_ENV="${OPENQHA_ENV:-openqha-gpu}"
export OPENQHA_ENV
source hpc/env/common.sh
source hpc/env/tianhe.sh
openqha_report_env

echo
echo "species        $SPECIES"
echo "tag            $TAG"
echo "allocation     1 GPU + 14 CPUs + 240 GB  (h100x package)"
echo "card           ${CUDA_VISIBLE_DEVICES:-<set by slurm>}"
echo

# ---- 2. refuse early, not one task at a time ---------------------------------------------
python - <<'PY' || exit 1
import sys
import torch
from openqha import engine, qha
print("torch          {}  cuda={}  devices={}".format(
    torch.__version__, torch.cuda.is_available(), torch.cuda.device_count()))
if not torch.cuda.is_available():
    sys.exit("openQHA: torch cannot see a CUDA device. Refusing to run a GPU job on the "
             "CPU -- it would give correct numbers at the wrong cost and nothing "
             "downstream would say so.")
p = engine.provenance()
print("engine         {}  sha256 {}".format(p["engine"], p["sha256"][:16]))
q = qha.protocol()
print("protocol       {} ps equil + {} ps production, frame every {} ps -> {} frames"
      .format(q["equilibration_ps"], q["production_ps"],
              q["sampling_interval_ps"], q["expected_frames"]))
print("               source: {}".format(q["_source"]))
PY

# ---- 3. branch A: the basin list ----------------------------------------------------------
# CPU work on a GPU allocation, deliberately: see the header. One molecule, 14 threads.
echo
echo "================ 1/3  branch A -- conformer search ================"
if [ -f "analysis/branchA/$TAG/$SPECIES/basins.json" ]; then
    echo "already done; skipping. Delete that file to redo it."
else
    python -u scripts/production/s0_A_pipeline.py \
        --species "$SPECIES" --tag "$TAG" \
        --threads "$CREST_THREADS" --timeout-s 14400 --hessian-mode analytic
fi

BASINS=$(python - <<PY
import sys
sys.path.insert(0, '.')
from openqha import basin_store
j, x = basin_store.paths_for("$SPECIES", tag="$TAG")[:2]
print(x if x.exists() else "")
PY
)
if [ -z "$BASINS" ]; then
    echo "branch A wrote no basin file -- refusing to run branch B on a geometry that" >&2
    echo "was never searched. Look at analysis/branchA/$TAG/$SPECIES/driver.log." >&2
    exit 1
fi
N_BASINS=$(python -c "
from ase.io import read
print(len(read('$BASINS', index=':')))")
echo "basins         $N_BASINS   ($BASINS)"

# ---- 4. branch B: one trajectory per (basin, seed), 14 at a time on one card ---------------
echo
echo "================ 2/3  branch B -- trajectories on the card ================"
N_TASKS=$((N_BASINS * SEEDS))
echo "tasks          $N_TASKS  ($N_BASINS basins x $SEEDS seeds), $WORKERS at a time"

EXTRA=()
[ -n "$PROD_PS" ]  && EXTRA+=(--prod-ps "$PROD_PS")
[ -n "$EQUIL_PS" ] && EXTRA+=(--equil-ps "$EQUIL_PS")

mkdir -p "$S0_SCRATCH/traj"
: > "$S0_SCRATCH/failures.txt"
i=0
running=0
HAVE_WAIT_N=1
help wait 2>/dev/null | grep -q -- "-n" || HAVE_WAIT_N=0
for b in $(seq 0 $((N_BASINS - 1))); do
    for s in $(seq 0 $((SEEDS - 1))); do
        (
            # NO CUDA_VISIBLE_DEVICES here. The allocation IS one card and Slurm has
            # already set it; every worker inherits that unchanged. Setting an index
            # ourselves on a per-card cluster hands the process a card the job does not
            # own -- which is the opposite of what TianheXY-A needs, and the reason those
            # two resource configs differ.
            python -u scripts/production/s0_B_qha_trajectory_openmm.py \
                --species "$SPECIES" --tag "$TAG" --basins "$BASINS" \
                --basin "$b" --seed-index "$s" \
                --platform CUDA --wall-budget-s "$WALL_BUDGET_S" "${EXTRA[@]}" \
                > "$S0_SCRATCH/traj/b${b}_s${s}.log" 2>&1 \
                || echo "FAILED basin $b seed $s" >> "$S0_SCRATCH/failures.txt"
        ) &
        i=$((i + 1)); running=$((running + 1))
        if [ "$running" -ge "$WORKERS" ]; then
            if [ "$HAVE_WAIT_N" = "1" ]; then
                wait -n || true; running=$((running - 1))
            else
                wait || true; running=0
            fi
        fi
    done
done
wait
[ -s "$S0_SCRATCH/failures.txt" ] && { echo "FAILURES:"; cat "$S0_SCRATCH/failures.txt"; }

# ---- 5. collect: the quasi-harmonic analysis ----------------------------------------------
echo
echo "================ 3/3  collect -- quasi-harmonic analysis ================"
# --no-gmx: GROMACS is an EXTENSION and no number depends on it. MDAnalysis is the
# independent implementation behind acceptance criterion 2 and is in this environment.
python -u scripts/production/s0_B_qha_analyse.py \
    --species "$SPECIES" --tag "$TAG" --no-gmx

# ---- 6. where it went ----------------------------------------------------------------------
echo
python - <<PY
import json, sys
from pathlib import Path
sys.path.insert(0, '.')
from openqha import config
root = Path(config.runs_dir("qha")) / "$TAG" / "$SPECIES"
print("{:>6} {:>8} {:>8} {:>9} {:>14}  {}".format(
    "basin", "seed", "frames", "complete", "s/ps this run", "platform"))
for meta in sorted(root.glob("basin*/seed*/meta.json")):
    m = json.loads(meta.read_text(encoding="utf-8"))
    p = m.get("production", {})
    sp = p.get("seconds_per_ps_this_run")
    print("{:>6} {:>8} {:>8} {:>9} {:>14}  {}".format(
        m.get("basin_index"), m.get("seed"), p.get("n_frames"), str(p.get("complete")),
        "-" if sp is None else round(sp, 2), m.get("platform")))
print()
print("trajectories   {}".format(root))
print("analysis       analysis/qha/$TAG/")
print("basins         analysis/branchA/$TAG/$SPECIES/basins.json")
PY
echo
echo "If any trajectory says complete=False it stopped on its wall budget."
echo "Resubmit the same command -- it continues from the frames already on disk."
