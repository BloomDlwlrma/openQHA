#!/bin/bash
# ORCA for the `labels` role (Hessian-learning ticket 03). Sourced AFTER common.sh and the
# machine file, in a worker that is already in the `openqha` conda environment.
#
# THE PROBLEM THIS SOLVES. On tianhe ORCA 6.1.1 is a SHARED build
# (`.../qjliang/apps/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg`, its own `lib/`)
# and its OpenMPI 4.1.8 lives in the conda env `orca611`; `~/env_orca611.sh` enters that
# env, exports `ORCA_PATH` and defines an ALIAS `orca='$ORCA_PATH/orca'` (read off the
# user's session 2026-09-19 -- not the template in configs/init_template/). A Parsl worker
# cannot live in that shell: it needs parsl, ase and numpy from `openqha`, and `orca611`'s
# lib on LD_LIBRARY_PATH shadows the worker's own libraries (the `libtinfo.so.6: no
# version information` line bash prints there is that shadowing). So this file enters the
# ORCA environment once, RECORDS the executable and the two search paths the script
# produced, and leaves again with PATH and LD_LIBRARY_PATH exactly as they were. The
# recorded paths reach the ORCA subprocess alone, through
# `openqha.qm_interfaces.orca.subprocess_env()`.
#
#   S0_ORCA_BIN     the executable, full path (ORCA insists on it for parallel runs):
#                   $ORCA_PATH/orca when the script sets ORCA_PATH, else the PATH lookup
#                   (`type -P`, which ignores aliases and functions)
#   S0_ORCA_PATH    the PATH the script produced (mpirun, the ORCA directory)
#   S0_ORCA_LIB     the LD_LIBRARY_PATH it produced (ORCA's shared lib/, OpenMPI's)
#   S0_ORCA_ENV_SH  the script that enters the ORCA environment (default ~/env_orca611.sh)
#
# An operator who exports S0_ORCA_BIN (and S0_ORCA_PATH / S0_ORCA_LIB) before the job
# skips all of this: an ORCA installed anywhere else is a legitimate answer.
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
    local _path="$PATH" _ld="$LD_LIBRARY_PATH" _lvl="${CONDA_SHLVL:-0}" _orca_path_before="$ORCA_PATH"
    # shellcheck disable=SC1090
    source "$envsh" || { echo "hpc/env/orca.sh: sourcing $envsh failed" >&2; return 1; }
    if [ -n "$ORCA_PATH" ] && [ -x "$ORCA_PATH/orca" ]; then
        S0_ORCA_BIN="$ORCA_PATH/orca"                 # the tianhe script's own variable
    else
        S0_ORCA_BIN="$(type -P orca 2>/dev/null)"     # an executable on PATH; never an alias
    fi
    S0_ORCA_PATH="$PATH"
    S0_ORCA_LIB="$LD_LIBRARY_PATH"
    # the ORCA directory itself, for a shared build: its lib/ must be findable, and the
    # script may rely on the alias rather than PATH for the binary
    if [ -n "$S0_ORCA_BIN" ]; then
        local _dir; _dir="$(dirname "$S0_ORCA_BIN")"
        case ":$S0_ORCA_PATH:" in *":$_dir:"*) ;; *) S0_ORCA_PATH="$_dir:$S0_ORCA_PATH" ;; esac
        [ -d "$_dir/lib" ] && case ":$S0_ORCA_LIB:" in *":$_dir/lib:"*) ;; *) S0_ORCA_LIB="$_dir/lib:$S0_ORCA_LIB" ;; esac
    fi
    # back to the worker's own environment: pop what the script pushed, then restore the
    # two search paths verbatim -- `conda deactivate` alone leaves the exported
    # LD_LIBRARY_PATH of the script in place
    while [ "${CONDA_SHLVL:-0}" -gt "$_lvl" ]; do conda deactivate 2>/dev/null || break; done
    export PATH="$_path"
    export LD_LIBRARY_PATH="$_ld"
    unalias orca 2>/dev/null
    [ -z "$_orca_path_before" ] && unset ORCA_PATH
    if [ -z "$S0_ORCA_BIN" ] || [ ! -x "$S0_ORCA_BIN" ]; then
        echo "hpc/env/orca.sh: $envsh gave no executable 'orca' (ORCA_PATH='$ORCA_PATH', PATH lookup empty)" >&2
        return 1
    fi
    export S0_ORCA_BIN S0_ORCA_PATH S0_ORCA_LIB
    echo "openQHA: ORCA at $S0_ORCA_BIN (its PATH and LD_LIBRARY_PATH go to the ORCA subprocess only)"
    return 0
}
