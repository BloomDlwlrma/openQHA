#!/bin/bash
# openQHA branch A -- ONE EDGE on the `debug` partition. TianheXY-C.
#
#     cd $HOME/openQHA
#     mkdir -p $HOME/HDD_POOL/runs/openQHA/logs        # BEFORE submitting. See below.
#     yhbatch hpc/slurm/branchA_debug.slurm
#
# This is the gate before production, and it is a REAL job, not a rendered plan
# (user ruling 2026-09-07). Thirty minutes on the short queue tests the five things a
# plan cannot: that the module loads, that conda activates on a compute node, that the
# MACE weights are there and hash correctly, that CREST runs, and that the product is
# written where the next step expects it.
#
# Two molecules, 4 threads each, on one node. That is one edge -- the smallest unit
# that means anything scientifically, since branch A's deliverable is a DIFFERENCE
# between two species.
# =======================================================================================
#SBATCH --job-name=openqha_A_debug
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# ^ %x_%j in the SUBMIT directory, deliberately, and not a path under HDD_POOL.
#   Slurm opens --output BEFORE the script runs, so a directory the script creates is
#   created too late and the job dies with no log saying why (rule 6 of this project's
#   HPC skill). The submit directory always exists. The production script writes its
#   logs under HDD_POOL because there you have already made the directory by hand.

# `set -eo pipefail` removed 2026-09-13 (user ruling: a failing step must not end the job; .mem/notes/notes_2026-09-13_no-errexit-anywhere.md)
# NOT `set -u`: the conda GROMACS activation hook fails under it and the environment is
# then half-built while the script carries on. Measured 2026-09-03.

# ---- 1. site environment ---------------------------------------------------------------
module purge 2>/dev/null || true
module load anaconda3/2023.09 || {
    echo "openQHA: module load anaconda3/2023.09 FAILED -- conda is not on PATH" >&2
    exit 1
}

# Inherited task-layout variables make a child process misread its allocation and try to
# relaunch itself through the scheduler. CREST forks its own parallel workers, so it is
# exposed to exactly the failure this prevents (rule 7).
for v in $(env | awk -F= '{print $1}' | grep -E '^(PMI|SLURM_(CPU|TASK|NTASKS|NPROCS|STEP))'); do
    unset "$v"
done

cd "${SLURM_SUBMIT_DIR:-$PWD}"
source hpc/env/common.sh
source hpc/env/tianhe.sh || { echo "openQHA: hpc/env/tianhe.sh refused (root unresolved); stopping." >&2; exit 1; }
openqha_report_env

# ---- 2. refuse early, not one task at a time -------------------------------------------
openqha_require crest xtb python || exit 1
python - <<'PY' || exit 1
from openqha import engine
p = engine.provenance()          # recomputes the SHA-256 and raises on a mismatch
print("engine      {}  sha256 {}  pinned={}".format(
    p["engine"], p["sha256"][:16], p["sha256_pinned"]))
PY

# ---- 3. the work -----------------------------------------------------------------------
TAG="${TAG:-tianhe_debug}"
EDGE="${EDGE:-C3H6O1N0_18_35}"          # the acetone edge; both species, 10 atoms each
SPECIES=$(python -c "
import sys; sys.path.insert(0, '.')
from openqha import config
print(' '.join(config.edge_species('$EDGE')))")

echo
echo "edge     $EDGE"
echo "species  $SPECIES"
echo "tag      $TAG"
echo

# 4 threads per molecule, 2 molecules at once. The node has 64 cores, so this uses 8 of
# them -- a debug job is a correctness test, not a throughput test, and asking for the
# whole node's worth of concurrency here only makes the failure harder to read.
printf '%s\n' $SPECIES | xargs -P 2 -I{} \
    python -u scripts/production/s0_A_pipeline.py \
        --species {} --tag "$TAG" --threads 4 --timeout-s 1500 \
        --hessian-mode analytic

# ---- 4. say where it went, and what to look at -----------------------------------------
echo
echo "products:"
for s in $SPECIES; do
    echo "  analysis/branchA/$TAG/$s/basins.json"
done
python - <<PY
import sys; sys.path.insert(0, '.')
from openqha import basin_store
c = basin_store.census(tag="$TAG")
print("basin store: {} molecules under {}".format(c["total"], c["root"]))
PY
echo
echo "If every criterion passed, drop --debug / switch to branchA_deimos.slurm."
