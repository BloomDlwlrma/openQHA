#!/bin/bash
# =====================================================================================
# openQHA -- environment installer
#
#     bash install_dependency.sh              # local workstation (default)
#     bash install_dependency.sh --tianhe     # Tianhe, or any site like it
#     bash install_dependency.sh --minimal    # smallest set that runs branch A
#     bash install_dependency.sh --check      # install nothing, just report
#     bash install_dependency.sh --no-weights # skip the MACE-OFF download
#     bash install_dependency.sh --all-weights# fetch the whole MACE-OFF family, not one
#     bash install_dependency.sh --cuda       # CUDA build, without the Tianhe specifics
#     bash install_dependency.sh --tianhe-cuda# TianheXY-AI (GPU) -- CUDA 12.3 env
#     bash install_dependency.sh --tianhe-a   # TianheXY-A  (GPU) -- the same env
#     bash install_dependency.sh --tianhe-cuda --both   # BOTH envs on one cluster
#
# THERE ARE THREE TIANHE CLUSTERS. The two GPU ones share an environment file; the CPU
# one does not:
#   --tianhe       TianheXY-C  (CPU: debug/deimos)          -> branch A + result collection
#   --tianhe-cuda  TianheXY-AI (GPU: hx/h100x/a100x/...)    -> branch C training
#   --tianhe-a     TianheXY-A  (GPU: temp/ai, 8 cards/node) -> branch B trajectories, C
#
# The two GPU flags build ONE file, environment-tianhe-gpu.yml, pinned at CUDA 12.3 --
# the version both module trees have (CONFIRMED on TianheXY-AI 2026-09-08: `module avail`
# lists CUDA/12.3 outright, closing the "recorded listing was elided" caveat that
# environment-tianhe-gpu.yml and hpc/resource_configs/tianhe_ai.py both carry).
#
# =====================================================================================
# NETWORK ON TIANHE: THE PROXY IS THE ONLY THING THIS SCRIPT SETS
# =====================================================================================
# Conda goes through the TUNA mirror, and ~/.condarc is not this
# script's to rewrite. The same file is in place on all three clusters (TianheXY-CN, -A,
# -AI) and maps `conda-forge` to <TUNA>/anaconda/cloud. With `nodefaults` in the
# environment files, that TUNA path is the only channel a solve here touches.
#
# The 2026-09-08 install failure looked like a TUNA outage and was not one:
#
#     Failed to resolve 'mirrors.tuna.tsinghua.edu.cn'
#     ([Errno -3] Temporary failure in name resolution)
#
# **Name resolution.** Not a refused connection, not a 404. A Tianhe login node has no
# direct DNS, so until setproxy.sh has run NOTHING resolves -- TUNA, anaconda.org or
# anything else. One cause, one fix. This script now:
#
#   * sources the site proxy itself (host/port from configs/cluster_tianhe.yaml) and
#     exports all four spellings plus no_proxy, so pip and curl see it too;
#   * PROBES the mirror your condarc actually names, once, with a 15 s timeout, and
#     stops there rather than letting conda retry for minutes and blame the mirror;
#   * silences the notice/retry storm through CONDA_* env vars -- no file is touched.
#
# It does NOT edit, read or override ~/.condarc. `OPENQHA_FORCE_ANACONDA=1` exists for
# exactly one case -- TUNA stale or incomplete for a pin, which surfaces as
# PackagesNotFoundError rather than a network error -- and is off by default.
#
# =====================================================================================
# TWO ENVIRONMENTS PER SITE, SPLIT ALONG THE BLAS. READ THIS BEFORE CHANGING EITHER.
# =====================================================================================
# `openqha` (CPU) and `openqha-gpu` (GPU) are not a tidy-up and not two ways of doing the
# same thing. They exist because THREE REQUIREMENTS THIS PROJECT HAS CANNOT ALL HOLD IN
# ONE SOLVE, and the measurements that establish that are these:
#
#   (a) conda-forge's `crest` links the PTHREADS OpenBLAS while CREST itself is
#       OpenMP-parallel. Measured 2026-09-04, dsgdb9nsd_000018, -T 4, same 2 conformers:
#           pthreads OpenBLAS   38.0 s   4910 output lines, 4164 of them warnings
#           openmp   OpenBLAS   18.5 s    745 output lines, 0 warnings
#       So the CPU side pins `libopenblas=*=openmp*` + `libblas/liblapack=*=*openblas`
#       + `nomkl`. That pin is worth 2x on branch A's hot loop.
#
#   (b) A CUDA build of pytorch DEPENDS ON MKL. `nomkl` and a CUDA torch are therefore
#       not both satisfiable -- not a preference, an unsatisfiable constraint.
#
#   (c) Dry runs 2026-09-07, one variable at a time:
#           openmmtools + BLAS pins, no CUDA                     SOLVES
#           openmmtools + CUDA torch, no BLAS pins               SOLVES
#           openmmtools + CUDA torch + cuda-version=12.3 + pins  FAILS
#       Each PAIR solves; the TRIPLE has no solution.
#
# One environment can satisfy (a) or (b), never both. So: the BLAS pins live in the CPU
# environment, where crest is and where the quasi-harmonic diagonalisation runs, and the
# GPU environment takes the solver's MKL, where neither of those things happens.
#
# WHAT THE SPLIT COSTS, stated rather than assumed: numpy links OpenBLAS on one side and
# MKL on the other. The only place that could move a published number is the
# quasi-harmonic diagonalisation -- and that runs in the COLLECTION pass, in the CPU
# environment, against OpenBLAS, always. See environment-tianhe-gpu.yml for the longer
# version of this argument.
#
# ---- IF YOU ONLY HAVE ONE CLUSTER: --both -------------------------------------------
# The three-cluster layout above assumes branch A goes to TianheXY-C. An account with
# only a GPU cluster still needs branch A somewhere, and the answer is BOTH environments
# on that cluster -- branch A then runs on the CPU cores of a GPU allocation:
#
#     bash install_dependency.sh --tianhe-cuda --both     # TianheXY-AI
#     bash install_dependency.sh --tianhe-a    --both     # TianheXY-A
#
# That is TWO SEPARATE SOLVES, which is the entire point: it is not a bigger environment,
# it is two smaller ones, each satisfiable. Pick one per job with OPENQHA_ROLE=cpu|gpu
# and let hpc/env/tianhe.sh do the activation -- it also unwinds a stacked activation,
# which is the one way to get both BLAS libraries onto a single loader path.
#
# NOT A CONTRADICTION OF THE "one environment" SETUP: it was about
# the WORKSTATION, where there is no CUDA torch and (b) never bites, and it stands:
# environment.yml still holds CREST, xtb, MACE and the OpenMM route together, and the
# bit-identical fingerprint that justified it is still in
# scripts/calibration/s0_B_stack_fingerprint.py.
#
# `--cuda` builds environment-cuda.yml into `openqha-cuda` -- a workstation CUDA build,
# without the Tianhe proxy/channel handling below.
#
# ---- THE WORKSTATION IS STILL ONE ENVIRONMENT, AND THAT IS NOT THE SAME QUESTION -----
# `openqha` on a workstation holds the Python stack, CREST, xtb and MACE together. It
# used to be two, to dodge OpenBLAS printing "Detect OpenMP Loop and this application may
# hang" on every CREST step -- and environment.yml now pins the OpenMP build of OpenBLAS,
# which removes the mismatch at its source instead. Measured 2026-09-04,
# dsgdb9nsd_000018, -T 4, same 2 conformers from every row:
#
#     one env (openmp_*), nothing set        18.5 s   745 lines      0 warnings
#     one env (openmp_*), OPENBLAS=1         18.4 s   749 lines      0 warnings
#     split env (pthreads_*), nothing set    38.0 s  4910 lines   4164 warnings
#     split env (pthreads_*), OPENBLAS=1     25.1 s   747 lines      0 warnings
#
# The bottom two rows are one binary and one variable, and they are why the split went:
# the measurement that justified it changed the environments AND the variable at once,
# then credited the separation. env_openqha.sh still exports the variable -- free here,
# and it is what protects a CREST installed from anywhere else.
#
# That is a DIFFERENT question from the Tianhe split above. There the second environment
# is forced by a CUDA torch that drags MKL in; on a workstation with a CPU torch nothing
# forces it, so one environment is right there and two are right on Tianhe. Both answers
# come from the same rule: pin the BLAS where CREST is.
#
# WHAT THIS SCRIPT WILL NOT DO
#   * install ORCA. Registration required; branch C only.
#   * download QM9 or curatedQM9. Both are large and neither belongs in version control.
#     Whether you use curatedQM9 at all is a choice you make with `f7_mode` -- see README.
# =====================================================================================

# NOTE: no `set -u`. The conda GROMACS activation hook fails under it
# ("GMXRC: line 10: shell: unbound variable") and the environment is then only half
# built while the script carries on. Measured 2026-09-03.
# `set -eo pipefail` is deliberately not used: a failing step must not end the job.

MODE="local"
PY_ENV="${OPENQHA_ENV:-openqha}"
ENV_FILE="environment.yml"
REQ="requirements.txt"
DO_INSTALL=1
WEIGHTS="default"          # default | all | none
#: Extra modules for the GPU modes. CUDA only: openQHA loads no MPI on a GPU cluster,
#: because nothing it runs there needs collectives. That is what removed the one real
#: difference between the two GPU environment files -- an openmpi built against a
#: different CUDA fails at the first collective rather than at import, so the safest
#: version of that dependency is not having it.
TIANHE_MODULES=""
#: Verbatim argument list, so error messages can tell you how to re-run THIS invocation
#: rather than a generic one.
ORIG_ARGS="$*"
#: --both: build the CPU environment as well as the GPU one, on the same cluster.
BOTH=0

for arg in "$@"; do
    case "$arg" in
        --tianhe|--hpc)  MODE="tianhe"; ENV_FILE="environment-tianhe.yml"
                         PY_ENV="${OPENQHA_ENV:-openqha}" ;;
        --cuda)          ENV_FILE="environment-cuda.yml"
                         PY_ENV="${OPENQHA_ENV:-openqha-cuda}" ;;
        # BOTH GPU clusters build the SAME environment file.
        # 12.3 is the version both module trees carry, and no MPI is loaded because
        # nothing openQHA runs on a card needs collectives -- which is exactly what let
        # the two files become one. See environment-tianhe-gpu.yml.
        --tianhe-cuda)   MODE="tianhe"; ENV_FILE="environment-tianhe-gpu.yml"
                         PY_ENV="${OPENQHA_ENV:-openqha-gpu}"
                         GPU_FLAG="--tianhe-cuda"
                         TIANHE_MODULES="CUDA/12.3" ;;
        --tianhe-a)      MODE="tianhe"; ENV_FILE="environment-tianhe-gpu.yml"
                         PY_ENV="${OPENQHA_ENV:-openqha-gpu}"
                         GPU_FLAG="--tianhe-a"
                         TIANHE_MODULES="CUDA/12.3" ;;
        # TWO ENVIRONMENTS ON ONE CLUSTER. Add this to a GPU flag when the cluster you
        # have is the only cluster you have -- see the BLAS section of the header.
        --both)          BOTH=1 ;;
        --local)         MODE="local" ;;
        --minimal)       REQ="requirements-minimal.txt" ;;
        --check)         DO_INSTALL=0 ;;
        --no-weights)    WEIGHTS="none" ;;
        --all-weights)   WEIGHTS="all" ;;
        -h|--help)       awk 'NR>1 && /^#/ {print; next} NR>1 {exit}' "$0"; exit 0 ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
warn() { printf '\033[33m!! %s\033[0m\n' "$*"; }

#: Set by the Tianhe preparation below to a scratch directory holding our own .condarc.
#: Empty everywhere else, which is what makes conda_solve a no-op off Tianhe.
CONDA_HOME_SHIM=""

# EVERY conda call that talks to a channel goes through this; activation and `conda info`
# do NOT, and must not, because they need the real $HOME.
#
# A subshell rather than a `VAR=value conda ...` prefix, deliberately: `conda` is a shell
# FUNCTION after conda.sh is sourced, and bash keeps a variable assignment that prefixes
# a function call in the calling shell afterwards. Silently moving $HOME for the rest of
# the script is not a small bug.
#: Which binary performs a solve: `conda`, or the mamba chosen in section 0. Set there;
#: `conda` until then, so a pre-section-0 call still works.
SOLVE_TOOL="conda"

conda_solve() {
    if [ -n "$CONDA_HOME_SHIM" ]; then
        ( HOME="$CONDA_HOME_SHIM"; export HOME; "$SOLVE_TOOL" "$@" )
    else
        "$SOLVE_TOOL" "$@"
    fi
}

#: Read-only queries -- `config --show`, and anything else that must reflect the real
#: conda regardless of which solver runs. Never mamba: mamba's `config` is not conda's.
conda_query() {
    if [ -n "$CONDA_HOME_SHIM" ]; then
        ( HOME="$CONDA_HOME_SHIM"; export HOME; conda "$@" )
    else
        conda "$@"
    fi
}


# -------------------------------------------------------------------------------------
# 0a. --both: the CPU environment AND the GPU one, on this cluster
# -------------------------------------------------------------------------------------
# Two SEPARATE solves, run one after the other, not one solve with more in it. That is
# the whole point -- see the BLAS section of the header. Each half gets the linear
# algebra it needs and neither has to satisfy the other's constraint.
#
# Re-executing this script twice rather than looping inside it is deliberate: each solve
# then gets its own clean `conda activate`, its own pip pass and its own capability
# check, and a failure in one is unambiguous about which environment it happened in.
if [ "$BOTH" -eq 1 ]; then
    if [ -z "$GPU_FLAG" ]; then
        echo "--both needs a GPU flag beside it: --tianhe-cuda --both, or --tianhe-a --both" >&2
        exit 2
    fi
    if [ -n "$OPENQHA_ENV" ]; then
        echo "--both builds two environments and cannot use a single OPENQHA_ENV=$OPENQHA_ENV." >&2
        echo "  Set OPENQHA_ENV_CPU / OPENQHA_ENV_GPU instead, or unset it." >&2
        exit 2
    fi
    # Pass through everything that is not a mode selector; the modes are supplied below.
    PASS=""
    for a in "$@"; do
        case "$a" in
            --both|--tianhe|--hpc|--tianhe-cuda|--tianhe-a|--cuda|--local|--minimal) ;;
            *) PASS="$PASS $a" ;;
        esac
    done
    say "--both: two environments, two solves"
    echo "  1/2  ${OPENQHA_ENV_CPU:-openqha}      environment-tianhe.yml      OpenBLAS/OpenMP, crest+xtb, CPU torch"
    echo "  2/2  ${OPENQHA_ENV_GPU:-openqha-gpu}  environment-tianhe-gpu.yml  MKL, CUDA 12.3 torch, OpenMM"
    # shellcheck disable=SC2086
    OPENQHA_ENV="${OPENQHA_ENV_CPU:-openqha}"     bash "$0" --tianhe   $PASS || exit 1
    # shellcheck disable=SC2086
    OPENQHA_ENV="${OPENQHA_ENV_GPU:-openqha-gpu}" bash "$0" "$GPU_FLAG" $PASS || exit 1
    say "--both: done. Pick one per job, never both at once"
    echo "  branch A / QHA collection   OPENQHA_ROLE=cpu  source hpc/env/tianhe.sh"
    echo "  branch B / branch C         OPENQHA_ROLE=gpu  source hpc/env/tianhe.sh"
    echo
    echo "  Do NOT 'conda activate openqha' and then 'conda activate openqha-gpu'."
    echo "  Stacked activation puts an OpenBLAS lib/ and an MKL lib/ on one loader path;"
    echo "  which BLAS you get is then decided by link order, not by the environment you"
    echo "  named. hpc/env/tianhe.sh deactivates down to base before activating."
    exit 0
fi
say "openQHA installer -- mode: $MODE"
echo "repository  : $HERE"
# Read the three facts that actually distinguish these environments off the file being
# built, rather than asserting them. The old line was a flat "one environment; CREST and
# xtb are in it" -- printed unchanged over `openqha-gpu`, which contains neither. That
# was the most misleading line the installer produced: it tells an operator on a GPU
# cluster that branch A is installed when it is not.
#
# BLAS is first because it is the axis the two Tianhe environments are split ALONG, and
# the one whose value you cannot see from the environment's name.
_yml_dep() { grep -qE "^[[:space:]]*-[[:space:]]*$1" "$ENV_FILE" 2>/dev/null; }
if _yml_dep 'nomkl' || _yml_dep 'libopenblas'; then ENV_BLAS="OpenBLAS/OpenMP (pinned)"
else                                                ENV_BLAS="solver's choice (MKL)"; fi
if _yml_dep 'crest'; then ENV_CONF="crest+xtb"; else ENV_CONF="no crest/xtb"; fi
if grep -qE '^[[:space:]]*-[[:space:]]*(cuda-version|pytorch.*cuda)' "$ENV_FILE" 2>/dev/null
then ENV_ACC="CUDA torch"; else ENV_ACC="CPU torch"; fi
echo "environment : $PY_ENV"
echo "role        : BLAS $ENV_BLAS | $ENV_CONF | $ENV_ACC"
echo "from        : $ENV_FILE"
echo "requirements: $REQ"
echo "weights     : $WEIGHTS"

# -------------------------------------------------------------------------------------
# 0. conda
# -------------------------------------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
    # Module-based sites first: on Tianhe conda does not exist until the module is
    # loaded, and the directory search below would just report "conda not found".
    # `module avail` there lists anaconda3/2019.10, 2020.11 and 2023.09; the newest is
    # the one this repository is built against (python 3.11 from conda-forge anyway).
    if command -v module >/dev/null 2>&1; then
        module load anaconda3/2023.09 >/dev/null 2>&1 || \
            module load anaconda3 >/dev/null 2>&1 || true
        for m in $TIANHE_MODULES; do
            module load "$m" >/dev/null 2>&1 || \
                echo "!! module not available: $m" >&2
        done
    fi
fi
if ! command -v conda >/dev/null 2>&1; then
    for c in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" /opt/conda; do
        [ -f "$c/etc/profile.d/conda.sh" ] && { . "$c/etc/profile.d/conda.sh"; break; }
    done
fi
command -v conda >/dev/null 2>&1 || {
    echo "conda not found. Install miniforge or miniconda first:" >&2
    echo "  https://github.com/conda-forge/miniforge" >&2
    exit 1
}
CONDA_BASE="$(conda info --base)"
. "$CONDA_BASE/etc/profile.d/conda.sh"
echo "conda       : $CONDA_BASE"

# The classic solver takes tens of minutes on this dependency set; libmamba takes a
# minute or two. Use it if it is there, and say so rather than silently doing either.
# WHY THIS IS MORE THAN A SPEED SETTING, measured on TianheXY-AI 2026-09-08:
# a `--tianhe-cuda` solve ran **1 h 37 min at 100% CPU and 3.5 GB RSS** and was still
# going. That is the classic solver's signature on a hard set; libmamba answers this
# file, or refuses it, in minutes. A classic solve does not merely take longer -- it goes
# silent behind a spinner for over an hour on a shared login node, which is both
# indistinguishable from a hang and antisocial.
echo "conda ver   : $(conda --version 2>/dev/null || echo unknown)"
SOLVER=()
SOLVE_TOOL="conda"
# `if command -v`, NOT `_mamba="$(command -v mamba)"`. `command -v` exits non-zero when
# the name is not found, and under `set -e` that assignment ABORTS THE SCRIPT -- silently,
# right after the "conda ver" line. Caught 2026-09-08 by the mock run, which is exactly
# the shape of bug a mock is for: on a machine that HAS mamba it never fires.
# A condition in `if`/`elif` is exempt from `set -e`; a bare assignment is not.
_mamba=""
if [ "${OPENQHA_SOLVE_WITH:-auto}" = "conda" ]; then
    :                                        # explicit opt-out: leave _mamba empty
elif command -v mamba >/dev/null 2>&1; then
    _mamba="$(command -v mamba)"
elif [ -x "$CONDA_BASE/bin/mamba" ]; then
    # miniforge3 ships mamba here even when the shell hook has not been run, which is the
    # case inside this script -- it sources conda.sh only.
    _mamba="$CONDA_BASE/bin/mamba"
fi

if [ -n "$_mamba" ]; then
    # PREFERRED ON THIS SITE. The conda here is miniforge3, which always ships mamba, and
    # the user's own init_conda.sh already exports MAMBA_EXE. mamba IS libmamba with no
    # question of whether a flag reached the right subcommand -- see the next branch for
    # why that question is not academic.
    SOLVE_TOOL="$_mamba"
    echo "solver      : mamba -- $_mamba  (OPENQHA_SOLVE_WITH=conda to use conda)"
elif conda env create --help 2>&1 | grep -q -- '--solver' && \
     "$CONDA_BASE/bin/python" -c "import conda_libmamba_solver" >/dev/null 2>&1; then
    # NOTE THE SUBCOMMAND. This used to probe `conda create --help` while the call the
    # script actually makes is `conda env create`. Those are different parsers and they
    # have not always carried the same flags -- so the check could pass, the flag be
    # dropped or rejected, and the solve fall back to classic with nothing said. Probe
    # the subcommand you are going to run.
    SOLVER=(--solver=libmamba)
    # Belt and braces: the flag is per subcommand, the variable is not.
    export CONDA_SOLVER=libmamba
    echo "solver      : conda + libmamba (flag and CONDA_SOLVER, not left to default)"
else
    warn "NO libmamba AND NO mamba. The classic solver will be used, and on this"
    warn "  dependency set that is not 'slower' -- it was still running after 1 h 37 min"
    warn "  at 100% CPU on TianheXY-AI (2026-09-08). Fix it before starting:"
    warn "    conda install -n base conda-libmamba-solver     # or"
    warn "    conda install -n base mamba"
    warn "  Set OPENQHA_ALLOW_CLASSIC=1 to proceed anyway."
    if [ -z "$OPENQHA_ALLOW_CLASSIC" ]; then
        echo "refusing to start a classic solve; see above." >&2
        exit 1
    fi
fi
unset _mamba

# -------------------------------------------------------------------------------------
# 1. Site-specific preparation
# -------------------------------------------------------------------------------------
if [ "$MODE" = "tianhe" ]; then
    say "Tianhe preparation"
    # Required by the site and harmless anywhere else. Without them the
    # parallel run fails.
    ulimit -l unlimited 2>/dev/null || warn "could not raise the locked-memory limit"
    export GLEX_USE_ZC_RNDV=0

    # ---- the proxy -------------------------------------------------------------------
    # A Tianhe login node has NO direct DNS. Until setproxy.sh has run, every outbound
    # name fails to resolve -- which is exactly what produced the 2026-09-08 install
    # failure: forty urllib3 retry lines for `mirrors.tuna.tsinghua.edu.cn`, then a
    # CondaHTTPError that reads like a mirror outage and is not one.
    #
    # This used to be a warning telling the operator to export it themselves. Host and
    # port have been recorded in configs/cluster_tianhe.yaml since 2026-08-31 and
    # hpc/env/tianhe.sh already sources the same script, so asking for them to be retyped
    # was one more thing to get wrong. Set it here; leave an already-set proxy alone.
    TIANHE_SETPROXY="${TIANHE_SETPROXY:-/APP/u22/ai_x86/toolshs/setproxy.sh}"
    TIANHE_PROXY_HOST="${TIANHE_PROXY_HOST:-172.16.31.200}"
    TIANHE_PROXY_PORT="${TIANHE_PROXY_PORT:-3138}"
    if [ -z "$https_proxy" ] && [ -z "$HTTPS_PROXY" ]; then
        if [ -f "$TIANHE_SETPROXY" ]; then
            # shellcheck disable=SC1090
            . "$TIANHE_SETPROXY" "$TIANHE_PROXY_HOST" "$TIANHE_PROXY_PORT" || true
            echo "proxy       : $TIANHE_SETPROXY $TIANHE_PROXY_HOST $TIANHE_PROXY_PORT"
        else
            warn "no https_proxy, and no $TIANHE_SETPROXY to source."
            warn "  export https_proxy=http://<host>:<port>   # then re-run"
        fi
    else
        echo "proxy       : ${https_proxy:-$HTTPS_PROXY}  (already set; left alone)"
    fi
    # pip, curl and the weight transfer must see the same proxy conda does, and they read
    # different spellings of the name. Set all four rather than guessing which.
    if [ -n "$https_proxy" ] || [ -n "$HTTPS_PROXY" ]; then
        export https_proxy="${https_proxy:-$HTTPS_PROXY}"
        export HTTPS_PROXY="$https_proxy"
        export http_proxy="${http_proxy:-$https_proxy}"
        export HTTP_PROXY="$http_proxy"
        # Never proxy the site itself -- that is how a module server or a licence check
        # turns into a hang with no error.
        export no_proxy="${no_proxy:-localhost,127.0.0.1,::1,.tianhe,.local}"
        export NO_PROXY="$no_proxy"
    fi

    # ---- channels: THE TUNA MIRROR IN YOUR ~/.condarc, USED AS IT STANDS --------------
    # On Tianhe, conda goes through the TUNA mirror, and
    # ~/.condarc is not this script's to rewrite. The same file is in place on all three
    # clusters -- TianheXY-CN, -A and -AI -- and it reads:
    #
    #     channels: [defaults]
    #     default_channels: <TUNA>/anaconda/pkgs/{main,r,msys2}
    #     custom_channels:  conda-forge: <TUNA>/anaconda/cloud
    #                       pytorch:     <TUNA>/anaconda/cloud
    #
    # So `conda-forge` in environment-tianhe*.yml resolves to <TUNA>/anaconda/cloud/
    # conda-forge, and `nodefaults` in those files drops `defaults` -- meaning the TUNA
    # conda-forge path is the ONLY channel a solve here touches. That is the intended
    # arrangement; nothing below changes it.
    #
    # AND THAT IS WHY THE 2026-09-08 FAILURE WAS NEVER A MIRROR PROBLEM. Re-read it:
    #
    #     Failed to resolve 'mirrors.tuna.tsinghua.edu.cn'
    #     ([Errno -3] Temporary failure in name resolution)
    #
    # Name resolution, not a refused connection or a 404. TUNA was not unreachable, it
    # was UNRESOLVABLE -- because no proxy had been set, which the block above now
    # handles. One cause, one fix. The forty retry lines that buried it are handled by
    # the scalar settings below.

    # ---- make conda fail in seconds rather than in retry storms ----------------------
    # Environment variables, so NO FILE IS TOUCHED -- and unlike `channels`, these are all
    # SCALAR parameters, which conda replaces rather than merges. (The merge behaviour is
    # specific to sequences; see the note on OPENQHA_FORCE_ANACONDA below.) Verified
    # against conda 24.1.2 on 2026-09-08: all five read back changed.
    #
    # `number_channel_notices=0` alone removes the ~40 "Retrieving notices" retry lines
    # that opened the failing log -- notices are chatter, and they are fetched for every
    # configured channel before the .yml is even parsed.
    export CONDA_NUMBER_CHANNEL_NOTICES="${CONDA_NUMBER_CHANNEL_NOTICES:-0}"
    export CONDA_REMOTE_CONNECT_TIMEOUT_SECS="${CONDA_REMOTE_CONNECT_TIMEOUT_SECS:-10.0}"
    export CONDA_REMOTE_READ_TIMEOUT_SECS="${CONDA_REMOTE_READ_TIMEOUT_SECS:-60.0}"
    export CONDA_REMOTE_MAX_RETRIES="${CONDA_REMOTE_MAX_RETRIES:-2}"
    export CONDA_REMOTE_BACKOFF_FACTOR="${CONDA_REMOTE_BACKOFF_FACTOR:-1}"

    # ---- pip: the TUNA PyPI index, to match ------------------------------------------
    # conda covers everything except mace (the fork, by git URL), pymsym and parsl,
    # which have no conda-forge package and come from pip. Left alone they cross the
    # proxy one wheel at a time; mace's dependency set makes that the slowest part of
    # the install.
    # Exported for this run only -- ~/.pip/pip.conf is not written, same principle as
    # ~/.condarc. Set OPENQHA_PIP_INDEX to override, or to "" to keep pypi.org.
    if [ -z "${PIP_INDEX_URL+x}" ]; then
        export PIP_INDEX_URL="${OPENQHA_PIP_INDEX-https://pypi.tuna.tsinghua.edu.cn/simple}"
        [ -n "$PIP_INDEX_URL" ] || unset PIP_INDEX_URL
    fi

    # ---- ESCAPE HATCH: OPENQHA_FORCE_ANACONDA=1 --------------------------------------
    # OFF BY DEFAULT, and only for one failure this cannot otherwise fix: TUNA mirrors
    # conda-forge's CONTENT, so if it is stale or incomplete for a pin this project needs
    # -- `openmm-torch=*cuda*`, `cuda-version=12.3` -- the solve fails with
    # PackagesNotFoundError. That is a CONTENT error, not a network one, and no proxy
    # setting helps. This routes that one run straight at conda.anaconda.org instead.
    #
    # The mechanism is a scratch HOME holding hpc/condarc.tianhe, which is not the
    # obvious choice because both obvious ones were measured and do not work
    # (conda 24.1.2, 2026-09-08):
    #   CONDARC=<file>       IGNORED OUTRIGHT. `conda config --show-sources` with it set
    #                        lists only ~/.condarc; the named file never appears.
    #   CONDA_CHANNELS=<url> MERGES, does not replace -- `channels` is a sequence and
    #                        conda concatenates sequences across sources, so the result
    #                        was [<our url>, conda-forge, defaults] with both
    #                        TUNA-mapped names still in the list and still fetched.
    #                        (`CONDA_CUSTOM_CHANNELS=''` is a hard type error: map
    #                        parameters cannot be emptied from the environment at all.)
    # A .condarc inside a HOME we control is read as the ONLY user config, so there is
    # nothing left to merge with. Your real ~/.condarc is never read or written either
    # way -- this substitutes for it for the duration of one command.
    if [ -n "$OPENQHA_FORCE_ANACONDA" ] && [ -f "$HERE/hpc/condarc.tianhe" ]; then
        # `|| true`: `set -e` aborts on a failing command substitution, and an
        # unreadable config is a thing to warn about, not to die on.
        REAL_ENVS_DIR="$(conda config --show envs_dirs 2>/dev/null \
                         | sed -n '2p' | sed 's/^[[:space:]]*-[[:space:]]*//')" || true
        REAL_PKGS_DIR="$(conda config --show pkgs_dirs 2>/dev/null \
                         | sed -n '2p' | sed 's/^[[:space:]]*-[[:space:]]*//')" || true
        CONDA_HOME_SHIM="${TMPDIR:-/tmp}/openqha-condarc.$$"
        mkdir -p "$CONDA_HOME_SHIM"
        cp "$HERE/hpc/condarc.tianhe" "$CONDA_HOME_SHIM/.condarc"
        # Read the REAL envs/pkgs dirs before moving HOME and pin them, or the
        # environment is built under the shim and vanishes with it. Only if non-empty:
        # CONDA_ENVS_DIRS="" tells conda the list is empty, which is worse than unset.
        [ -n "$REAL_ENVS_DIR" ] && export CONDA_ENVS_DIRS="${CONDA_ENVS_DIRS:-$REAL_ENVS_DIR}"
        [ -n "$REAL_PKGS_DIR" ] && export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$REAL_PKGS_DIR}"
        # `trap ... EXIT` rather than an rm at the end: the probe below can exit 1.
        trap 'rm -rf "$CONDA_HOME_SHIM" 2>/dev/null' EXIT
        warn "OPENQHA_FORCE_ANACONDA=1: bypassing the TUNA mirror for this run."
        echo "channels    : conda.anaconda.org direct ($HERE/hpc/condarc.tianhe)"
        echo "envs dir    : ${CONDA_ENVS_DIRS:-<conda default>}"
    fi

    # Record what the solve will actually use. Not a warning in either mode: on Tianhe a
    # TUNA URL here is CORRECT, and the earlier version of this check flagged it, which
    # would have fired on every properly configured run.
    EFFECTIVE_CHANNELS="$(conda_query config --show channels 2>/dev/null \
                          | tail -n +2 | sed 's/^[[:space:]]*-[[:space:]]*//' | tr '\n' ' ')" || true
    echo "channels    : ${EFFECTIVE_CHANNELS:-<none reported>}"
    if [ -z "$CONDA_HOME_SHIM" ]; then
        # `defaults` here is not the whole story and printing it alone would mislead:
        # the .yml adds conda-forge, and ~/.condarc's custom_channels is what decides
        # the URL it resolves to. Name where to look rather than guess it for you.
        echo "              names above resolve through custom_channels in ~/.condarc"
        echo "              (conda config --show-sources shows the mapping)"
    fi

    # ---- probe BEFORE handing conda the network --------------------------------------
    # Without this the failure mode is ~40 retry lines per channel and a final message
    # naming a mirror rather than the missing proxy. One 15 s request says it plainly,
    # and stopping here costs nothing: everything after this point needs the network.
    if [ -z "$OPENQHA_SKIP_NET_CHECK" ]; then
        # Probe THE HOST THIS ACCOUNT WILL ACTUALLY USE, read out of the effective conda
        # config rather than hard-coded. On Tianhe that is TUNA, via `custom_channels:
        # conda-forge: <TUNA>/anaconda/cloud`; under OPENQHA_FORCE_ANACONDA the shim has
        # no custom_channels and it falls through to conda.anaconda.org. Deriving it
        # means the check follows the site if the mirror ever moves, instead of testing
        # a host no solve here would contact.
        _cf_base="$(conda_query config --show custom_channels 2>/dev/null \
                    | awk '/^[[:space:]]*conda-forge:/ {print $2}')" || true
        if [ -n "$_cf_base" ]; then
            _probe="$_cf_base/conda-forge/noarch/repodata.json"
        else
            _probe="https://conda.anaconda.org/conda-forge/noarch/repodata.json"
        fi
        _code=000
        if command -v curl >/dev/null 2>&1; then
            # No `|| echo 000` here: on failure curl ALSO writes its own "000" to stdout,
            # and the two concatenate into "000000" -- which matches no case below and
            # reads like a corrupted status in the message. Normalise after instead.
            _code="$(curl -s -o /dev/null -w '%{http_code}' -I --max-time 15 "$_probe" \
                     2>/dev/null)" || true
            [ -n "$_code" ] || _code=000
        elif command -v python >/dev/null 2>&1; then
            _code="$(OPENQHA_PROBE_URL="$_probe" python -c 'import os,urllib.request as u
r=u.urlopen(u.Request(os.environ["OPENQHA_PROBE_URL"],method="HEAD"),timeout=15)
print(r.status)' 2>/dev/null || echo 000)"
        else
            warn "neither curl nor python available to probe the network; skipping check"
            _code=200
        fi
        case "$_code" in
            200|301|302)
                echo "network     : reachable, HTTP $_code -- $_probe" ;;
            *)
                warn "cannot reach $_probe  (HTTP $_code)"
                warn "  STOPPING HERE ON PURPOSE. Left to itself conda spends minutes"
                warn "  retrying and then reports this as a mirror outage -- which on"
                warn "  2026-09-08 it was not. That log said 'Failed to resolve', i.e."
                warn "  no DNS, i.e. NO PROXY. Check that first, in this order:"
                warn "    source $TIANHE_SETPROXY $TIANHE_PROXY_HOST $TIANHE_PROXY_PORT"
                warn "    env | grep -i proxy"
                warn "    curl -sI --max-time 15 $_probe | head -1"
                warn "  Site proxy moved?  TIANHE_PROXY_HOST=... TIANHE_PROXY_PORT=..."
                warn "  TUNA really down?  OPENQHA_FORCE_ANACONDA=1 routes this one run"
                warn "                     at conda.anaconda.org instead. It does not"
                warn "                     edit ~/.condarc."
                warn "  Go ahead anyway?   OPENQHA_SKIP_NET_CHECK=1 bash $0 $ORIG_ARGS"
                exit 1 ;;
        esac
        unset _probe _code _cf_base
    fi

    # Node-local scratch. CREST writes many small files into parallel _N subdirectories,
    # and doing that on a shared parallel filesystem is slow for everyone, not only you.
    export S0_RUNS_ROOT="${S0_RUNS_ROOT:-${TMPDIR:-$HOME}/runs/openQHA}"
    echo "runs root   : $S0_RUNS_ROOT"
    echo
    echo "NOT DONE BY THIS SCRIPT. Read docs/branchA_production.md, then:"
    echo "  1. copy the MACE-OFF weights in -- this script does NOT download them on a"
    echo "     login node (section 4 of that document says exactly why, and how):"
    echo "       rsync -a data/potentials/ <tianhe>:$HERE/data/potentials/"
    echo "  2. confirm the scheduler commands are still where they were on 2026-09-05:"
    echo "       python -c \"import sys; sys.path.insert(0,'hpc'); import providers, json;"
    echo "                   print(json.dumps(providers.preflight('tianhe'), indent=1))\""
    echo "     A wrong SUBMIT command fails loudly. A wrong STATUS command does not:"
    echo "     Parsl believes every job is pending and the queue silently stops."
    echo "  3. run the 30-minute smoke job BEFORE production. That -- not a dry run --"
    echo "     is the gate:"
    echo "       CPU  yhbatch hpc/slurm/branchA_debug.slurm     # debug partition, 00:30:00"
    echo "       GPU  bash hpc/slurm/submit_branchB_tianhe_a.sh a_debug temp"
    echo "  4. THREE CLUSTERS, three sets of rules:"
    echo "       TianheXY-C   CPU, whole-node --exclusive, partitions debug + deimos"
    echo "       TianheXY-AI  GPU per card, --gpus MANDATORY, --exclusive BANNED"
    echo "       TianheXY-A   GPU, 8 cards/node, -AI rules"
    echo "     parsl defaults exclusive=True, so a stock SlurmProvider is refused by both"
    echo "     GPU clusters. hpc/resource_configs/ already handles this."
fi

if [ "$DO_INSTALL" -eq 0 ]; then
    say "check only, installing nothing"
    conda env list
    exit 0
fi

# -------------------------------------------------------------------------------------
# 2. The environment
# -------------------------------------------------------------------------------------
say "environment: $PY_ENV"

# ---- __cuda: a LOGIN NODE HAS NO GPU DRIVER, so the solver thinks CUDA is impossible ---
# conda-forge's CUDA builds depend on the `__cuda` VIRTUAL package, which conda/mamba
# synthesise from the NVIDIA driver they can see. Login nodes have no card and no driver,
# so `__cuda` is absent and every `*cuda*` build is unsatisfiable. Measured on
# ln302%TianheXY-AI, 2026-09-08, with mamba:
#
#     pytorch =2.5.1 cuda120* is not installable because it requires
#     └─ __cuda =* *, which is missing on the system.
#
# CONDA_OVERRIDE_CUDA tells the solver to assume a driver of that version. It affects
# SOLVING ONLY -- nothing is faked at runtime -- and it is honest here because the
# compute nodes this environment will run on do have one (driver 550.54.15, a CUDA 12.4
# driver; minor-version compatibility covers a 12.3 runtime).
#
# **Read from $ENV_FILE, not hard-coded**, so the override and the `cuda-version=` pin
# cannot drift apart -- setting them independently is how you get an environment that
# solves against one CUDA and links against another.
#
# NOTE FOR ANYONE RE-TESTING THIS: `conda` on a workstation WITH a driver solves
# `*cuda*` happily and proves nothing. Check `conda info | grep -A6 'virtual packages'`
# and confirm `__cuda` is genuinely absent before concluding the override is unnecessary.
# A truncated read of that output is what got this wrong once already.
_cuda_pin="$(grep -oE '^[[:space:]]*-[[:space:]]*cuda-version=[0-9]+\.[0-9]+' "$ENV_FILE" 2>/dev/null \
             | grep -oE '[0-9]+\.[0-9]+' | head -1)" || true
if [ -n "$_cuda_pin" ] && [ -z "$CONDA_OVERRIDE_CUDA" ]; then
    export CONDA_OVERRIDE_CUDA="$_cuda_pin"
    echo "  __cuda      : assumed $_cuda_pin for the solve (login nodes carry no driver)"
fi
unset _cuda_pin
# SAY HOW LONG THIS TAKES, BEFORE IT GOES QUIET.
# `Solving environment: \` is a spinner and nothing else; the script then prints nothing
# for minutes. Every operator who has not seen it before reads that as a hang and kills
# it -- which on this dependency set throws away the most expensive step. So state the
# expectation, and state how to check it is alive, in the log itself.
cat <<'EXPECT'
  The solve prints a spinner and nothing else. That is normal, and it is not quick:
    libmamba   ~1-5 min here      classic   tens of minutes
  This file pins cuda-version, a *cuda* pytorch, a *cuda* openmm-torch and openmmtools
  at once; it is the hardest solve in this repository. To check it is working rather
  than hung, from a SECOND shell -- ~100% CPU means solving, ~0% means blocked:
    ps -o pid,etime,pcpu,rss,comm -u "$USER" | grep -i -E 'conda|mamba'
EXPECT
if conda env list | awk '{print $1}' | grep -qx "$PY_ENV"; then
    # DEFECT, fixed 2026-09-08: this said `-f environment.yml` unconditionally, so a
    # re-run of `--tianhe-cuda` after a failed solve updated `openqha-gpu` from the
    # WORKSTATION file -- notebook stack, GROMACS hooks, and a CPU torch over the CUDA
    # one. The failed-then-re-run path is the common one, which is what makes this worth
    # more than a tidy-up: the 2026-09-08 network failure puts every operator on it.
    echo "exists; updating in place from $ENV_FILE"
    conda_solve env update "${SOLVER[@]}" -n "$PY_ENV" -f "$ENV_FILE" --prune
elif [ "$REQ" = "requirements-minimal.txt" ]; then
    # --minimal does not use environment.yml: it is deliberately a smaller set. The BLAS
    # pins are NOT optional here either -- see the header of environment.yml.
    conda_solve create -y "${SOLVER[@]}" -n "$PY_ENV" -c conda-forge \
        python=3.11 nomkl "libopenblas=*=openmp*" "libblas=*=*openblas" \
        "liblapack=*=*openblas" \
        numpy scipy pyyaml ase rdkit "pytorch=*=*cpu*" "crest>=3.0.2" "xtb>=6.6"
else
    # ONE solve that knows about CUDA from the start. The alternative -- build the CPU
    # environment, then `conda install` a CUDA torch over the top -- is how you end up
    # with a half-swapped library set that imports fine and dies at the first kernel
    # launch. (Whether that solve is the CPU or the GPU file is $ENV_FILE's business.)
    conda_solve env create -y "${SOLVER[@]}" -n "$PY_ENV" -f "$ENV_FILE"
fi
# NOT conda_solve: activation needs the REAL $HOME, and touches no channel.
conda activate "$PY_ENV"

# -------------------------------------------------------------------------------------
# 3. pip-only packages, MACE among them
# -------------------------------------------------------------------------------------
say "pip packages (mace from the fork URL in $REQ; pymsym, parsl)"
# `--no-user` on every pip call: without an activated environment pip falls back to
# ~/.local, and a package there shadows the environment in every later job (PEP 370 puts
# the user site first on sys.path). Measured 2026-09-12: a scipy in ~/.local, built
# against numpy 2, broke torch's numpy support in an environment pinned to numpy 1.26.4.
python -m pip install --no-user --upgrade pip
python -m pip install --no-user -r "$REQ"

# Named explicitly rather than left to the requirements file, because these two are the
# ones with no conda-forge package and the ones whose absence is least obvious. MACE is
# deliberately NOT on this line: $REQ carries the fork by git URL now, and re-installing
# the wheel here would overwrite it.
python -m pip install --no-user "pymsym>=0.3.5"
[ "$REQ" = "requirements-minimal.txt" ] || python -m pip install --no-user "parsl>=2024.01"

python - <<'PY'
import mace, torch
print("mace-torch", mace.__version__, "| torch", torch.__version__)
PY

# -------------------------------------------------------------------------------------
# 3b. Branch B's OpenMM route
# -------------------------------------------------------------------------------------
# Conda-forge only. `pip install openmm` does not give a working build, which is why this
# is here and not in requirements.txt.
#
# What it buys: the Nose-Hoover chain thermostat branch B now runs on, at openmmtools'
# own documented defaults (collision_frequency 50/ps = a 20 fs coupling time, chain_length
# 5, num_mts 5, num_yoshidasuzuki 5), plus two independent superposition implementations
# to check ours against.
#
# What it costs to skip: nothing that stops branch B. The ASE + Langevin route is the
# other production path and needs none of this; scripts/production/s0_B_qha_analyse.py
# reads either. Skipping it does remove the OpenMM half of the implementation pair.
#
# NNPOps and openmm-ml are deliberately absent: openqha/openmm_mace.py builds the graph
# as the complete graph, which is exact for 10-19 atoms and needs no neighbour search.
# The OpenMM stack comes from the environment file now, not from a second conda call:
# it is CORE and belongs in the solve, so that conda resolves it together with pytorch
# rather than against it. What is left here is the CHECK -- and it is a real one, because
# an environment file that lists a package is not evidence that the package imports.
if [ "$REQ" = "requirements-minimal.txt" ]; then
    say "--minimal: branch B's production route is not installed"
    warn "openqha/capabilities.py will report openmm, openmm_torch, openmmtools and"
    warn "mdanalysis as missing, and s0_B_qha_trajectory_openmm.py will refuse to run"
else
    say "verifying branch B's core capabilities"
    python - <<'PY'
import sys
from pathlib import Path
for p in Path(__file__ if "__file__" in dir() else ".").resolve().parents:
    if (p / "openqha" / "__init__.py").is_file():
        sys.path.insert(0, str(p))
        break
else:
    sys.path.insert(0, ".")
try:
    from openqha import capabilities
    print(capabilities.summary())
    capabilities.require_core()
    print("\nbranch B: READY")
except Exception as exc:
    print("\nbranch B is NOT ready: {}: {}".format(type(exc).__name__, exc))
    raise SystemExit(1)
PY
    if [ $? -ne 0 ]; then
        warn "branch B's core capabilities are incomplete -- see the message above."
        warn "Branch A is unaffected."
    fi
fi

# A stack change must be measured, not assumed. This is the fingerprint to take BEFORE
# and AFTER any change to the environment files, on identical input:
#
#     python scripts/calibration/s0_B_stack_fingerprint.py --out analysis/qha/fp_before
#     ... change the environment ...
#     python scripts/calibration/s0_B_stack_fingerprint.py --out analysis/qha/fp_after
#     python scripts/calibration/s0_B_stack_fingerprint.py --compare \
#         analysis/qha/fp_before.json analysis/qha/fp_after.json

CREST_BIN="$(command -v crest || true)"
[ -x "$CREST_BIN" ] || warn "crest not on PATH inside $PY_ENV"

# -------------------------------------------------------------------------------------
# 4. MACE-OFF weights
# -------------------------------------------------------------------------------------
# The weights are NOT in this repository -- they are fetched here, from upstream, by you.
# A download that does not look like a model (an HTML error page under a 200, a truncated
# file) is caught by the size check below; nothing hashes the file, at fetch or at load.
MACE_ROOT="${S0_MACE_ROOT:-$HERE/data/potentials}"
MACE_URL_BASE="https://github.com/ACEsuit/mace-off/raw/main"

fetch_model () {          # fetch_model <registry-name> <upstream-subdir> <file>
    # <upstream-subdir> is where the file lives ON THE SERVER. The LOCAL destination is
    # flat: base weights live at <root>/<filename>, the loader's one directory.
    # Downloading into a mace_off23/ subdirectory here is what made the installer put
    # weights somewhere the loader never looked.
    local name="$1" fam="$2" file="$3"
    local dest="$MACE_ROOT/$file"

    if [ -s "$dest" ]; then
        echo "  $file: already present ($(du -h "$dest" | cut -f1))"
        return 0
    fi

    mkdir -p "$MACE_ROOT"
    echo "  $file: downloading"
    if ! curl -fL --retry 3 --progress-bar -o "$dest.part" "$MACE_URL_BASE/$fam/$file"; then
        rm -f "$dest.part"
        warn "  $file: download failed. Fetch it by hand from $MACE_URL_BASE/$fam/$file"
        return 1
    fi

    # A model file is tens of megabytes. Anything much smaller is not one -- the usual
    # cause is a proxy answering with an HTML error page under a 200 status, which is a
    # perfectly valid file and not a potential. This is a size check, not an identity
    # check: it catches the empty and the truncated, nothing subtler.
    local bytes
    bytes="$(stat -c%s "$dest.part" 2>/dev/null || echo 0)"
    if [ "$bytes" -lt 1000000 ]; then
        rm -f "$dest.part"
        warn "  $file: only $bytes bytes -- that is not a model file. NOT installed."
        warn "    Usually a proxy error page. Fetch it by hand from"
        warn "    $MACE_URL_BASE/$fam/$file"
        return 1
    fi
    mv "$dest.part" "$dest"
    echo "  $file: installed ($(du -h "$dest" | cut -f1))"
}

if [ "$WEIGHTS" = "none" ]; then
    say "MACE-OFF weights: skipped (--no-weights)"
    echo "  Branch A cannot run without them. When you have them, drop them in"
    echo "    $MACE_ROOT/MACE-OFF23_medium.model"
    echo "  FLAT -- one directory, no subdirectories. That is the only layout."
elif [ "$MODE" = "tianhe" ]; then
    say "MACE-OFF weights: not downloaded on a login node"
    echo "  Outbound traffic goes through a proxy here and a 100 MB pull from a login"
    echo "  node is antisocial. Fetch them where you have bandwidth and copy them in:"
    echo "    rsync -av data/potentials/ <tianhe>:$HERE/data/potentials/"
    echo "  FLAT in that directory -- no mace_off23/ subdirectory. That is the only"
    echo "  layout openqha/potentials/engine.py resolves."
    echo "  Then: python -c 'from openqha import engine; print(engine.provenance())'"
else
    say "MACE-OFF weights -> $MACE_ROOT"
    export PYTHONPATH="$HERE:$PYTHONPATH"
    fetch_model MACE-OFF23_medium mace_off23 MACE-OFF23_medium.model || true
    if [ "$WEIGHTS" = "all" ]; then
        fetch_model MACE-OFF23_small   mace_off23 MACE-OFF23_small.model   || true
        fetch_model MACE-OFF23_large   mace_off23 MACE-OFF23_large.model   || true
        fetch_model MACE-OFF23b_medium mace_off23 MACE-OFF23b_medium.model || true
        fetch_model MACE-OFF24_medium  mace_off24 MACE-OFF24_medium.model  || true
    else
        echo "  (only the production default. --all-weights fetches the committee too.)"
    fi
fi

# -------------------------------------------------------------------------------------
# 5. GROMACS -- branch B's independent cross-check only
# -------------------------------------------------------------------------------------
if [ "$MODE" = "local" ] && [ "$REQ" != "requirements-minimal.txt" ]; then
    say "GROMACS (optional: branch B cross-check only)"
    conda install -y "${SOLVER[@]}" -c conda-forge gromacs || \
        warn "GROMACS not installed. Branch B still runs; only its independent check of S_QH is unavailable."
fi

# -------------------------------------------------------------------------------------
# 6. The environment file to source
# -------------------------------------------------------------------------------------
say "writing env_openqha.sh"
cat > env_openqha.sh <<SH
# Source this before running openQHA.  source env_openqha.sh
#
# Do NOT 'set -u' before sourcing: the conda GROMACS activation hook fails under it and
# leaves the environment half built without stopping.
. "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate $PY_ENV

export OPENQHA_ROOT="$HERE"
export PYTHONPATH="\$OPENQHA_ROOT:\$PYTHONPATH"

# One environment: crest and xtb are on PATH once it is active. S0_CREST_BIN still
# overrides, for a CREST built elsewhere.
export S0_CREST_BIN="\${S0_CREST_BIN:-\$(command -v crest)}"

# One thread per worker. MEASURED, not preferred: MACE on a 10-atom molecule runs
# 111/90/72/101 ms at 1/2/4/8 threads -- eight is SLOWER than four. Parallelism belongs
# between processes, never inside threads.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# This one FIXES A DEFECT rather than tuning anything. conda-forge's CREST can link the
# pthreads OpenBLAS while CREST is OpenMP-parallel; environment.yml pins the OpenMP build
# so it should not arise, but this is the setting that was actually measured to remove
# 3992 warning lines and 11 seconds, and it also holds for a CREST from anywhere else.
export OPENBLAS_NUM_THREADS=1
export OPENBLAS_MAIN_FREE=1

# torch >= 2.6 defaults to weights_only=True, which refuses the MACE-OFF checkpoints.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# Everything this repository writes goes under one root.
export S0_RUNS_ROOT="\${S0_RUNS_ROOT:-\$HOME/runs/openQHA}"
mkdir -p "\$S0_RUNS_ROOT"

# The potential's weights. Fetched by install_dependency.sh; resolved by name at load.
export S0_MACE_ROOT="\${S0_MACE_ROOT:-$MACE_ROOT}"

# Optional: curatedQM9, only if you set f7_mode: "curated". See README.
# export S0_CURATED_QM9=/path/to/133660_curatedQM9_outof_133885
SH
chmod +x env_openqha.sh

# -------------------------------------------------------------------------------------
# 7. Report
# -------------------------------------------------------------------------------------
say "checking"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export S0_MACE_ROOT="$MACE_ROOT"
python check_dependency.py || true

say "next"
cat <<'MSG'
  source env_openqha.sh
  python check_dependency.py

Then:
  1. python scripts/production/s0_A_pipeline.py --species dsgdb9nsd_000018
  2. read docs/branchA_workflow.md before running anything at scale
  3. to TRAIN (Hessian Labels): run ../openQHA-Hessian/install.sh in the active environment
MSG
