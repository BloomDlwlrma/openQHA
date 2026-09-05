#!/bin/bash
# Tianhe (TianheXY-AI). Sourced AFTER hpc/env/common.sh, never instead of it.
#
# Everything measured lives in common.sh. This file holds only what is true of this
# machine: where conda is, which environment, where scratch goes, and the proxy.
#
# **NOTHING HERE HAS BEEN RUN ON TIANHE.** It is assembled from the site information the
# user supplied on 2026-08-31, recorded in configs/cluster_tianhe.yaml. Lines that are
# assumptions say so. Read docs/tianhe_runbook.md first.

# ---- conda ---------------------------------------------------------------------------
# The site's own init script; miniforge3 + mamba, already set up by the user.
: "${CONDA_INIT:=$HOME/init_conda.sh}"
: "${OPENQHA_ENV:=openqha}"

if [ -f "$CONDA_INIT" ]; then
    # shellcheck disable=SC1090
    source "$CONDA_INIT"
else
    echo "hpc/env/tianhe.sh: no conda init at $CONDA_INIT" >&2
    echo "  set CONDA_INIT, or create it -- the workers cannot activate without it." >&2
fi
conda activate "$OPENQHA_ENV" 2>/dev/null || {
    echo "hpc/env/tianhe.sh: could not activate conda env '$OPENQHA_ENV'." >&2
    echo "  bash install_dependency.sh --tianhe   # builds it" >&2
}

# ---- the repository --------------------------------------------------------------------
# OPENQHA_ROOT is exported by env_openqha.sh; fall back to walking up from this file so a
# worker that was started without it still finds the package rather than failing on an
# import three steps later.
if [ -z "$OPENQHA_ROOT" ]; then
    _here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    OPENQHA_ROOT="$(cd "$_here/../.." && pwd)"
    unset _here
fi
export OPENQHA_ROOT
export PYTHONPATH="$OPENQHA_ROOT:$PYTHONPATH"

# CREST and xtb are in the single environment (one environment since 2026-09-04).
# S0_CREST_BIN still overrides, for a CREST built elsewhere.
if [ -z "$S0_CREST_BIN" ] && command -v crest >/dev/null 2>&1; then
    S0_CREST_BIN="$(command -v crest)"
fi
export S0_CREST_BIN

# ---- scratch ----------------------------------------------------------------------------
# The site's own convention, copied from the ORCA_SCR pattern the user supplied:
# prefer $TMPDIR (node-local NVMe/SSD), fall back to /tmp, and NEVER write temporary
# files on Lustre.
#
# The reason is measured, not stylistic: CREST writes many small files into parallel `_N`
# subdirectories (one molecule's working directory is 2.6-15 MB across dozens of files on
# this project's workstation). Doing that on a shared parallel filesystem is slow for the
# job and slow for everyone else on the machine.
export S0_SCRATCH="${TMPDIR:-/tmp}/$USER/openqha.${SLURM_JOB_ID:-$$}"
mkdir -p "$S0_SCRATCH"

# Everything this repository writes goes under one root. On a cluster that root must be
# node-local for the work and shared for the results -- runs go to scratch, products are
# written back into the repository by the pipeline itself.
export S0_RUNS_ROOT="${S0_RUNS_ROOT:-$S0_SCRATCH/runs}"
mkdir -p "$S0_RUNS_ROOT"

# ---- proxy ------------------------------------------------------------------------------
# Outbound traffic goes through a proxy. conda and pip HANG rather than fail without it,
# so this is worth setting even in a job: a worker that tries to reach the network and
# hangs is charged for the whole walltime.
#
# ASSUMED HOST AND PORT: taken verbatim from the site line the user supplied. If the site
# changes them this is where to edit.
if [ -z "$https_proxy" ] && [ -f /APP/u22/ai_x86/toolshs/setproxy.sh ]; then
    # shellcheck disable=SC1091
    source /APP/u22/ai_x86/toolshs/setproxy.sh 172.16.31.200 3138 || true
fi

# ---- what this file could not verify ------------------------------------------------------
# Printed by openqha_report_env (common.sh) so it lands in every job's log rather than
# living only in a comment nobody reads at 3 a.m.
export OPENQHA_SITE="tianhe"
export OPENQHA_SITE_UNVERIFIED="scheduler status/status_fallback/cancel command names; \
CPU partition name; cores per CPU node"
