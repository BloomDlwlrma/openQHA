#!/bin/bash
# Environment every openQHA job sources first, on every machine.
#
# **Nothing machine-specific goes in here.** Paths, conda roots, module loads and
# scheduler details belong in hpc/env/<machine>.sh.
#
# Rescued verbatim from _superseded/cluster/env.sh (branch E section 0): the three
# blocks below were bought with measurements, and they outlive the route that was
# retired around them.

# --------------------------------------------------------------------------------------
# DO NOT ADD `set -u` HERE.
# --------------------------------------------------------------------------------------
# Measured 2026-09-03: with `set -u` active, activating the conda environment fails
# on the GROMACS activation hook --
#
#     $CONDA_PREFIX/bin.AVX2_256/GMXRC: line 10: shell: unbound variable
#     $CONDA_PREFIX/bin.AVX2_256/GMXRC.bash: line 44: GMXLDLIB: unbound variable
#
# and the job then runs with a half-built environment instead of stopping. This is
# not hypothetical: it is what made a `gmx --version` check print an empty string
# earlier the same day and look like a missing install. `set -e` and `set -o
# pipefail` are fine and are set by the caller.

# --------------------------------------------------------------------------------------
# 1. Site requirements (Tianhe). Harmless elsewhere.
# --------------------------------------------------------------------------------------
# From the site's own job scripts (user, 2026-08-31; D0-C-24). Without these the
# parallel run fails on Tianhe. They cost nothing on a machine that does not need
# them, so they are unconditional rather than guarded by a hostname test -- a
# guard is another thing that can be wrong.
ulimit -l unlimited 2>/dev/null || true
export GLEX_USE_ZC_RNDV=0

# --------------------------------------------------------------------------------------
# 2. One thread per worker. This is a MEASUREMENT, not a preference.
# --------------------------------------------------------------------------------------
# MACE thread scaling on a 10-atom molecule (acetone), measured:
#
#     1 thread 111 ms | 2 threads 90 ms | 4 threads 72 ms | 8 threads 101 ms
#
# Eight threads is SLOWER than four, and 1 -> 8 buys only 1.1x. The molecules are
# too small for thread start-up and synchronisation to pay for themselves
# (D0-P1-27, D0-P1-32).
#
# **So parallelism lives between processes, never inside threads.** Every executor
# in hpc/resource_configs therefore runs many single-threaded workers rather than
# few multi-threaded ones.
#
# The one exception is CREST's own `threads` setting, which is a different axis:
# it parallelises independent metadynamics runs, not linear algebra. It is set per
# task in hpc/configs/crest.json.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# --------------------------------------------------------------------------------------
# 3. OpenBLAS. This one FIXES A DEFECT; it is not tuning.
# --------------------------------------------------------------------------------------
# conda-forge's CREST links the **pthreads** build of OpenBLAS while CREST itself is
# OpenMP-parallel. Calling a pthreads-threaded BLAS from inside an OpenMP region
# makes OpenBLAS print, on every step:
#
#     OpenBLAS Warning : Detect OpenMP Loop and this application may hang.
#
# Measured on acetone, same input, this variable the only change (D0-C-9):
#
#     unset:  4.4 s | 4406 output lines, 3626 of them that warning | crest.out 476 KB
#     = 1  :  2.7 s |  779 output lines, 0 warnings                | crest.out  34 KB
#
# Three benefits from one action: the warning goes away (and with it the "may hang"
# risk it is warning about), the output is 14x smaller, and it is 1.6x faster.
export OPENBLAS_NUM_THREADS=1
export OPENBLAS_MAIN_FREE=1

# --------------------------------------------------------------------------------------
# 4. Repository conventions
# --------------------------------------------------------------------------------------
# Everything this repo writes goes under one root (D0-C-8). No literal path appears
# in the code; the code asks openqha.config.runs_root().
# A DEFAULT, and it says so. `hpc/env/<site>.sh` is sourced after this file and moves the
# root onto node-local scratch; it must be able to tell "nobody chose one" from "the
# operator chose one". Without the marker its own `${S0_RUNS_ROOT:-...}` saw the value
# this line had just set and kept it -- so on Tianhe the node-local scratch never took
# effect and CREST wrote its many small files onto Lustre, which is exactly what
# tianhe.sh section "scratch" exists to prevent. Measured from a job banner 2026-09-09:
# `runs root  /HOME/hku2021_fos4/.../runs/openQHA`.
# ---- the per-user site directory is NOT part of this environment ------------------------
# Python puts ~/.local/lib/pythonX.Y/site-packages on sys.path BEFORE the active
# environment's own site-packages (PEP 370). Anything a stray `pip install --user` left
# there therefore SHADOWS the version this environment was solved with, silently, in
# every job.
#
# Measured 2026-09-12 on TianheXY-A: scipy 1.16.1 sat in ~/.local while the environment
# held numpy 1.26.4. scipy 1.16 is built against numpy 2, and the result was
# `RuntimeError: Numpy is not available` from torch and
# `ImportError: numpy._core.multiarray failed to import` -- an error that names numpy,
# whose cause was scipy, in a directory nobody had looked at.
#
# PYTHONNOUSERSITE=1 removes that directory from sys.path. It is set here rather than
# fixed once by hand because ~/.local is outside this repository's control and one
# `pip install --user` puts it back.
export PYTHONNOUSERSITE=1

if [ -z "$S0_RUNS_ROOT" ]; then
    export S0_RUNS_ROOT="$HOME/runs/openQHA"
    export S0_RUNS_ROOT_IS_DEFAULT=1
fi
# On Tianhe this default is replaced by $S0_SCRATCH/runs (hpc/env/tianhe.sh), which since
# the 2026-09-12 ruling is $HOME/runs/openQHA/<owner>/<JOBID>/runs -- the same root, one
# directory per job.
mkdir -p "$S0_RUNS_ROOT"

# torch >= 2.6 defaults to weights_only=True, which refuses the MACE-OFF
# checkpoints. Set here so a job never fails on it halfway through a queue.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

openqha_require() {
    # Fail loudly and immediately if a required executable is missing, rather than
    # letting the queue discover it one task at a time.
    for exe in "$@"; do
        command -v "$exe" >/dev/null 2>&1 || {
            echo "openQHA: required executable not found: $exe" >&2
            return 1
        }
    done
}

openqha_report_env() {
    # Printed at the top of every job log. A cost or a result without the
    # conditions it was taken under is not reportable (D0-P1-12, defect 34).
    echo "openQHA environment"
    echo "  host            $(hostname)"
    echo "  date            $(date -Is)"
    echo "  python          $(command -v python)"
    echo "  crest           ${S0_CREST_BIN:-$(command -v crest)}"
    echo "  runs root       $S0_RUNS_ROOT"
    echo "  OMP/MKL/BLAS    $OMP_NUM_THREADS/$MKL_NUM_THREADS/$OPENBLAS_NUM_THREADS"
    # WHICH BLAS, not just how many threads of it. Sites that keep two environments --
    # one OpenBLAS for CREST, one MKL because a CUDA torch drags MKL in anyway -- can
    # land a job in the wrong one and get numbers that look fine. The site file fills
    # these in (hpc/env/tianhe.sh); elsewhere they are empty and the line says so.
    echo "  conda env       ${CONDA_DEFAULT_ENV:-none}  [role ${OPENQHA_ROLE:-unset}]"
    echo "  BLAS provider   ${OPENQHA_BLAS:-not probed}/${OPENQHA_BLAS_THREADS:-n/a}"
    echo "  loadavg         $(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null || echo n/a)"
}
