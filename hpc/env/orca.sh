#!/bin/bash
# ORCA for the `labels` role (Hessian-learning ticket 03). Sourced AFTER common.sh and the
# machine file, in a worker that is already in the `openqha` conda environment.
#
# THE PROBLEM THIS SOLVES. On tianhe ORCA 6.1.1 and its OpenMPI live in their own conda
# environment, `orca611`, entered with `~/env_orca611.sh` (verified on the login node
# 2026-09-18: `source ~/init_conda.sh; conda activate orca611; export
# LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH`). A Parsl worker cannot live there:
# it needs parsl, ase and numpy from `openqha`, and `orca611`'s lib on LD_LIBRARY_PATH
# would shadow the worker's own libraries for every import that follows. So this file
# enters the ORCA environment once, RECORDS where its binary and its lib directory are,
# and leaves again with PATH and LD_LIBRARY_PATH exactly as they were. The recorded paths
# reach the ORCA subprocess alone, through `openqha.qm_interfaces.orca.subprocess_env()`.
#
#   S0_ORCA_BIN     the executable (full path -- ORCA insists on it for parallel runs)
#   S0_ORCA_PATH    its bin directory (mpirun lives there)
#   S0_ORCA_LIB     its lib directory (OpenMPI's shared libraries)
#   S0_ORCA_ENV_SH  the script that enters the ORCA environment (default ~/env_orca611.sh)
#
# An operator who exports S0_ORCA_BIN (and, if needed, S0_ORCA_PATH / S0_ORCA_LIB) before
# the job skips all of this: an ORCA installed anywhere else is a legitimate answer.
# DO NOT ADD `set -u` (see common.sh).

openqha_find_orca() {
    if [ -n "$S0_ORCA_BIN" ] && [ -x "$S0_ORCA_BIN" ]; then
        export S0_ORCA_BIN S0_ORCA_PATH S0_ORCA_LIB
        return 0
    fi
    local envsh="${S0_ORCA_ENV_SH:-$HOME/env_orca611.sh}"
    if [ ! -f "$envsh" ]; then
        echo "hpc/env/orca.sh: no $envsh -- set S0_ORCA_ENV_SH to the script that enters the" >&2
        echo "  ORCA conda environment, or S0_ORCA_BIN to the executable. Labels cannot run." >&2
        return 1
    fi
    local _path="$PATH" _ld="$LD_LIBRARY_PATH" _lvl="${CONDA_SHLVL:-0}"
    # shellcheck disable=SC1090
    source "$envsh" || { echo "hpc/env/orca.sh: sourcing $envsh failed" >&2; return 1; }
    S0_ORCA_BIN="$(command -v orca 2>/dev/null)"
    S0_ORCA_PATH="${CONDA_PREFIX:+$CONDA_PREFIX/bin}"
    S0_ORCA_LIB="${CONDA_PREFIX:+$CONDA_PREFIX/lib}"
    # back to the worker's own environment: pop what the script pushed, then restore the
    # two search paths verbatim -- `conda deactivate` alone leaves the exported
    # LD_LIBRARY_PATH of env_orca611.sh in place.
    while [ "${CONDA_SHLVL:-0}" -gt "$_lvl" ]; do conda deactivate 2>/dev/null || break; done
    export PATH="$_path"
    export LD_LIBRARY_PATH="$_ld"
    if [ -z "$S0_ORCA_BIN" ] || [ ! -x "$S0_ORCA_BIN" ]; then
        echo "hpc/env/orca.sh: $envsh entered an environment with no 'orca' on PATH" >&2
        return 1
    fi
    export S0_ORCA_BIN S0_ORCA_PATH S0_ORCA_LIB
    echo "openQHA: ORCA at $S0_ORCA_BIN (PATH+=$S0_ORCA_PATH, LD_LIBRARY_PATH+=$S0_ORCA_LIB, for the ORCA subprocess only)"
    return 0
}
