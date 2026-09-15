#!/bin/bash
# Stage 2 of 02d needs a trajectory whose atom order is branch A's own, produced under
# the branch B protocol. This makes one per basin, DENSELY sampled.
#
# WHY 2 fs AND NOT THE PROTOCOL'S 1.0 ps
# --------------------------------------
# The production protocol samples every 1.0 ps, chosen to filter high-frequency noise
# out of the ENTROPY. Stage 1's harmonic-limit control says the ZPE needs of order
# 12 500 frames to settle -- and 12 500 frames at 1.0 ps would be 12.5 ns, which is not
# reachable. At 2 fs the same frame count is 25 ps.
#
# That is the measurement, not a shortcut: the two terms want opposite sampling, and
# stage 2 exists to show what each interval can and cannot deliver. The interval is
# therefore an argument here rather than a constant, and every product records it.
#
# Detached with `setsid nohup`: a plain `&` from an interactive WSL shell dies with the
# shell and leaves a zero-byte log, which has happened in this repository before.
# `set -euo pipefail` removed 2026-09-13 (user ruling: a failing step must not end the job; .mem/notes/notes_2026-09-13_no-errexit-anywhere.md)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${OPENQHA_PYTHON:-$HOME/anaconda3/envs/openqha/bin/python}"

SPECIES="${SPECIES:-dsgdb9nsd_000018}"
BASINS="${BASINS:-auto}"                   # `auto`: branch A's mace/basinNN/basin.extxyz under BASIN_TAG
BASIN_TAG="${BASIN_TAG:-02d_prod}"         # the tag branch A's product is under (the molecule directory)
TRAJ_TAG="${TRAJ_TAG:-ex02d}"              # the setting name: md_ase/basinNN/*_<TRAJ_TAG>.* (since 2026-09-14)
PROD_PS="${PROD_PS:-25}"
EQUIL_PS="${EQUIL_PS:-10}"
SAMPLE_EVERY="${SAMPLE_EVERY:-2}"          # steps; the timestep is 1 fs
SEEDS="${SEEDS:-1}"
LOG="${LOG:-$HOME/runs/openQHA/02d_${SPECIES}_${TRAJ_TAG}.log}"

if [[ -z "$BASINS" ]]; then
    echo "BASINS must be 'auto' (branch A's basins under BASIN_TAG) or a multi-frame xyz." >&2
    echo "  BASIN_TAG=02d_prod SPECIES=dsgdb9nsd_000018 bash $0" >&2
    exit 2
fi
mkdir -p "$(dirname "$LOG")"
cd "$ROOT"

# --basins overrides the geometry source, --tag names the OUTPUT run. Passing them
# separately is what lets branch A's tag and branch B's tag differ, which they do here:
# propanal's basins live under `multibasin` and this run is `ex02d`.
"$PY" scripts/production/s0_B_qha_trajectory.py \
    --species "$SPECIES" --basins "$BASINS" --basin-tag "$BASIN_TAG" --tag "$BASIN_TAG" \
    --setting "$TRAJ_TAG" --seeds "$SEEDS" \
    --equil-ps "$EQUIL_PS" --prod-ps "$PROD_PS" --sample-every "$SAMPLE_EVERY" \
    >> "$LOG" 2>&1

echo "log: $LOG"
