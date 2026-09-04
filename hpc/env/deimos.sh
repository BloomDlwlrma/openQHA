#!/bin/bash
# deimos: 64-core CPU nodes. Sourced AFTER hpc/env/common.sh, never instead of it.
#
# Everything measured lives in common.sh. This file holds only what is true of this
# machine and nothing else: where conda is, which environment, where CREST is.

: "${CONDA_INIT:=$HOME/anaconda3/etc/profile.d/conda.sh}"
: "${OPENQHA_ENV:=qm9fe}"
: "${OPENQHA_CREST_ENV:=s0crest}"

# shellcheck disable=SC1090
source "$CONDA_INIT"
conda activate "$OPENQHA_ENV"

# CREST: on PATH if this machine has the single `openqha` environment, and in a separate
# environment on machines built before 2026-09-04, when the split was retired. Take
# whichever is there rather than asserting either -- openqha/crest.py reads S0_CREST_BIN.
if [ -z "$S0_CREST_BIN" ]; then
    _sep="$(dirname "$(dirname "$CONDA_PREFIX")")/$OPENQHA_CREST_ENV/bin/crest"
    if command -v crest >/dev/null 2>&1; then
        S0_CREST_BIN="$(command -v crest)"
    elif [ -x "$_sep" ]; then
        S0_CREST_BIN="$_sep"
    fi
    unset _sep
fi
export S0_CREST_BIN

# Scratch: CREST writes many small files into parallel `_N` subdirectories, so it
# must not run on a shared filesystem. Node-local scratch first, results copied
# back afterwards -- what is copied is the result, not the process.
export S0_SCRATCH="${TMPDIR:-/tmp}/openqha.$$"
mkdir -p "$S0_SCRATCH"

openqha_require python "$S0_CREST_BIN" || return 1
