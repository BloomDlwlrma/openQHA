#!/bin/bash
# Tianhe (TianheXY-AI). Sourced AFTER hpc/env/common.sh, never instead of it.
#
# Everything measured lives in common.sh. This file holds only what is true of this
# machine: where conda is, WHICH OF THE TWO environments (OPENQHA_ROLE), where scratch
# goes, and the proxy.
#
# **NOTHING HERE HAS BEEN RUN ON TIANHE.** It is assembled from the site information the
# user supplied on 2026-08-31, recorded in configs/cluster_tianhe.yaml. Lines that are
# assumptions say so. Read docs/tianhe_runbook.md first.

# ---- conda ---------------------------------------------------------------------------
# The site's own init script; miniforge3 + mamba, already set up by the user. On this
# account it is NOT in $HOME -- it sits beside the checkout, so try that too rather than
# printing "no conda init" at a file the operator can see with their own eyes.
if [ -z "$CONDA_INIT" ]; then
    for _c in "$HOME/init_conda.sh" \
              "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/../init_conda.sh"; do
        [ -f "$_c" ] && { CONDA_INIT="$_c"; break; }
    done
    : "${CONDA_INIT:=$HOME/init_conda.sh}"
    unset _c
fi

# ---- WHICH OF THE TWO ENVIRONMENTS ----------------------------------------------------
# There are two on this site and they differ in their LINEAR ALGEBRA, which is the one
# difference you cannot see by looking at the name:
#
#   OPENQHA_ROLE=cpu  ->  openqha       OpenBLAS/OpenMP pinned, crest+xtb, CPU torch
#                         branch A, and the branch B/QHA collection pass.
#   OPENQHA_ROLE=gpu  ->  openqha-gpu   MKL (solver's choice), CUDA 12.3 torch, OpenMM
#                         branch B trajectories, branch C training. NO crest, NO xtb.
#
# The split is not stylistic. conda-forge's `crest` needs the OpenMP OpenBLAS build (the
# pthreads one costs 4164 warning lines and 20 s per molecule, measured 2026-09-04), and
# `openmmtools + a CUDA torch + cuda-version=12.3 + those same BLAS pins` has no solution
# at all -- each pair solves, the triple does not (measured 2026-09-07). One environment
# cannot hold both requirements, so there are two, and each job says which it wants.
#
# OPENQHA_ENV still wins if it is set: a hand-built environment is a legitimate answer.
if [ -z "$OPENQHA_ENV" ]; then
    case "${OPENQHA_ROLE:-cpu}" in
        gpu) OPENQHA_ENV="${OPENQHA_ENV_GPU:-openqha-gpu}" ;;
        cpu) OPENQHA_ENV="${OPENQHA_ENV_CPU:-openqha}" ;;
        *)   echo "hpc/env/tianhe.sh: OPENQHA_ROLE='$OPENQHA_ROLE' is not cpu or gpu" >&2
             OPENQHA_ENV="${OPENQHA_ENV_CPU:-openqha}" ;;
    esac
fi
export OPENQHA_ENV

if [ -f "$CONDA_INIT" ]; then
    # shellcheck disable=SC1090
    source "$CONDA_INIT"
else
    echo "hpc/env/tianhe.sh: no conda init at $CONDA_INIT" >&2
    echo "  set CONDA_INIT, or create it -- the workers cannot activate without it." >&2
fi

# ---- deactivate down to base BEFORE activating ----------------------------------------
# `conda activate A; conda activate B` does not replace A: conda keeps a stack, and both
# prefixes stay on PATH and on the loader's search path. With THESE two environments that
# is not cosmetic -- one ships libblas linked to OpenBLAS and the other to MKL, so a
# stacked shell gets whichever the loader reaches first. It imports cleanly and gives you
# numbers from a BLAS you did not choose, which is the worst shape a defect can take.
#
# A job script that sources this file twice, or an interactive shell that already had one
# active, is enough to reach that state, so unwind rather than assume.
# Bounded: if `conda` is not a shell function the level never drops, and an unbounded
# loop in a sourced file hangs the job rather than failing it.
_unwind=0
while [ -n "$CONDA_PREFIX" ] && [ "${CONDA_SHLVL:-0}" -gt 0 ] && [ "$_unwind" -lt 8 ]; do
    conda deactivate 2>/dev/null || break
    _unwind=$((_unwind + 1))
done
unset _unwind
conda activate "$OPENQHA_ENV" 2>/dev/null || {
    echo "hpc/env/tianhe.sh: could not activate conda env '$OPENQHA_ENV'." >&2
    echo "  bash install_dependency.sh --tianhe-cuda --both   # builds BOTH of them" >&2
    echo "  bash install_dependency.sh --tianhe               # the CPU one alone" >&2
}

# ---- did we get the BLAS this role is supposed to have? -------------------------------
# The split above is only worth having if landing in the wrong half is VISIBLE, so record
# which BLAS this environment actually carries. Reported, not enforced: a hand-built
# environment is allowed to differ, and a job that refused to start over a library name
# would be worse than one that says in its log which library it used.
#
# READ FROM conda-meta, and not from the two places you would reach for first:
#
#   * `numpy.show_config()` CANNOT ANSWER THIS. Under conda-forge numpy links the
#     `libblas` metapackage, so the field reads `"name": "blas"` whether the provider
#     underneath is OpenBLAS or MKL. Checked 2026-09-08 against an env pinned to
#     OpenBLAS: it reported "blas". A check built on it would never fire.
#   * THE .so SYMLINK IS MISLEADING. In that same OpenMP-pinned environment
#     `libblas.so.3` resolves to `libopenblasp-r0.3.34.so` -- the trailing `p` is
#     conda-forge's file naming, NOT "pthreads". Reading it as the threading model gets
#     the answer exactly backwards.
#
# The conda-meta filename carries the build string, which is the thing that was actually
# solved for: `libblas-3.11.0-10_*_openblas` vs `*_mkl`, `libopenblas-0.3.34-openmp_*`
# vs `pthreads_*`. No tooling, one glob, correct.
OPENQHA_BLAS="unknown"; OPENQHA_BLAS_THREADS="n/a"
if [ -n "$CONDA_PREFIX" ] && [ -d "$CONDA_PREFIX/conda-meta" ]; then
    for _m in "$CONDA_PREFIX"/conda-meta/libblas-*.json; do
        [ -e "$_m" ] || continue
        case "$_m" in *_openblas.json) OPENQHA_BLAS="openblas" ;;
                      *_mkl.json)      OPENQHA_BLAS="mkl" ;; esac
    done
    for _m in "$CONDA_PREFIX"/conda-meta/libopenblas-*.json; do
        [ -e "$_m" ] || continue
        case "$_m" in *-openmp_*)   OPENQHA_BLAS_THREADS="openmp" ;;
                      *-pthreads_*) OPENQHA_BLAS_THREADS="pthreads" ;; esac
    done
    unset _m
fi
export OPENQHA_BLAS OPENQHA_BLAS_THREADS

if [ "${OPENQHA_ROLE:-cpu}" = "cpu" ]; then
    # The QHA diagonalisation -- the one place a BLAS can move a published number -- runs
    # in this half. So say so when it is not the one the environment file asks for.
    [ "$OPENQHA_BLAS" = "mkl" ] && {
        echo "hpc/env/tianhe.sh: NOTE -- role cpu, but '$OPENQHA_ENV' links MKL, not" >&2
        echo "  OpenBLAS. The quasi-harmonic diagonalisation runs against this. Expected" >&2
        echo "  only if you set OPENQHA_ENV yourself; else: install_dependency.sh --tianhe" >&2
    }
    # The measured defect, not a preference: conda-forge's crest is OpenMP-parallel, and
    # a pthreads OpenBLAS underneath it cost 4164 warning lines and 38.0 s against 18.5 s
    # on one molecule (2026-09-04). OPENBLAS_NUM_THREADS=1 in common.sh masks most of it;
    # this says whether the pin that removes it at source actually took.
    [ "$OPENQHA_BLAS_THREADS" = "pthreads" ] && {
        echo "hpc/env/tianhe.sh: NOTE -- pthreads OpenBLAS under an OpenMP CREST." >&2
        echo "  That is the 2026-09-04 defect. common.sh's OPENBLAS_NUM_THREADS=1 keeps" >&2
        echo "  it survivable; the fix is the libopenblas=*=openmp* pin in the env file." >&2
    }
fi

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

# CREST and xtb are in the CPU environment only -- they are the reason it pins OpenBLAS,
# and there is no crest/xtb module on this site, so conda is the only source.
# S0_CREST_BIN still overrides, for a CREST built elsewhere.
if [ -z "$S0_CREST_BIN" ] && command -v crest >/dev/null 2>&1; then
    S0_CREST_BIN="$(command -v crest)"
fi
export S0_CREST_BIN
if [ "${OPENQHA_ROLE:-cpu}" = "cpu" ] && [ -z "$S0_CREST_BIN" ]; then
    echo "hpc/env/tianhe.sh: role cpu, but no crest on PATH in '$OPENQHA_ENV'." >&2
    echo "  Branch A cannot run. Most likely you activated the GPU environment: it has" >&2
    echo "  no crest by design. OPENQHA_ROLE=cpu, or bash install_dependency.sh --tianhe" >&2
fi

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
