#!/bin/bash
# The RI-MP2 column of 02c. Hours, not seconds -- run it detached and come back.
#
# Detached with `setsid nohup ... -u`: a plain `&` from an interactive WSL shell dies
# with the shell and leaves a zero-byte log, which has happened here before.
# `set -euo pipefail` removed 2026-09-13 (user ruling: a failing step must not end the job; .mem/notes/notes_2026-09-13_no-errexit-anywhere.md)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${OPENQHA_PYTHON:-$HOME/anaconda3/envs/openqha/bin/python}"
NPROCS="${NPROCS:-4}"
LOG="${LOG:-$HOME/runs/openQHA/02c_reference.log}"
mkdir -p "$(dirname "$LOG")"

cd "$ROOT"
{
  # Tags are the ones branchA.conf wrote: 02c_prod for both molecules (propanal's step 1
  # is the same conf with SPECIES=dsgdb9nsd_000035). Until 2026-09-11 this said "prod"
  # and "multibasin", and neither directory exists.
  for spec in "dsgdb9nsd_000018 02c_prod" "dsgdb9nsd_000035 02c_prod"; do
    set -- $spec
    echo "=== $1  tag $2  $(date -Is) ==="
    "$PY" examples/02c_hessian_benchmark_levels/s0_level_benchmark.py \
        --species "$1" --tag "$2" --levels mace,gfn2,rimp2 --nprocs "$NPROCS" || \
        echo "FAILED $1"
  done
  echo "=== done $(date -Is) ==="
} >> "$LOG" 2>&1

echo "log: $LOG"
