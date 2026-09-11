#!/bin/bash
# openQHA -- build a conda environment ON A COMPUTE NODE instead of a login node.
#
#     # TianheXY-AI, GPU environment (the slow one). --gpus is MANDATORY here.
#     yhbatch -p h100x --gpus=1 -t 01:00:00 hpc/slurm/install_env_tianhe.slurm
#
#     # both environments in one job
#     ROLE=both yhbatch -p h100x --gpus=1 -t 02:00:00 hpc/slurm/install_env_tianhe.slurm
#
#     # TianheXY-A, fine-grained environment: enter it first, then -G is mandatory
#     # (1 card = 12 CPUs = 120 GB; --mem forbidden). Without the script the same
#     # prompt talks to a controller with Gres=(null) that refuses --gpus.
#     source /APP/u22/ai_x86/toolshs/set-XY-I.sh
#     yhbatch -p ai -G 1 -c 12 -t 01:00:00 hpc/slurm/install_env_tianhe.slurm
#
#     # TianheXY-CN (CPU cluster) -- but READ SECTION 3 FIRST, $HOME may not be the same
#     ROLE=cpu yhbatch -p debug -t 00:30:00 hpc/slurm/install_env_tianhe.slurm
#
# ROLE=gpu (default) | cpu | both      ENV_NAME= to override the environment name
#
# =======================================================================================
# 1. WHY THIS FILE EXISTS, AND WHAT IT DOES *NOT* FIX
# =======================================================================================
# Measured on ln302%TianheXY-AI, 2026-09-08: `conda env create -f
# environment-tianhe-gpu.yml` ran **1 h 37 min at 100% CPU and 3.5 GB RSS** and was still
# going. The same file solved in **5 min 30 s** with libmamba elsewhere. Two separate
# causes were in play and only one of them is the login node:
#
#   (a) THE SOLVER. It was the classic solver, because the installer probed
#       `conda create --help` for `--solver` while running `conda env create`. Fixed in
#       install_dependency.sh; this file simply REQUIRES mamba and refuses otherwise.
#       **If you have not re-run since that fix, fix the solver first -- a compute node
#       will not rescue a classic solve, it will only give it more cores to not use.**
#
#   (b) THE LOGIN NODE. Solving is one core, several GB of RSS, for minutes. Login nodes
#       are shared and commonly cgroup-limited on CPU and memory, so the same solve is
#       slower there and can be killed outright. That part IS what this file fixes.
#
# So: this is worth doing, but it is (b), not (a). Do not reach for it as a way around a
# slow solver.
#
# =======================================================================================
# 2. THE ONE THING THAT DECIDES WHETHER THIS WORKS: OUTBOUND NETWORK
# =======================================================================================
# **UNVERIFIED, and it is the whole ballgame.** On many HPC sites compute nodes have NO
# route off-cluster even when login nodes do -- the proxy is reachable from a login node
# and simply absent from a compute node. If that is true here, this job cannot download
# anything and section 5 will stop it in ~15 seconds rather than burning the allocation.
#
# That 15-second answer is worth having on its own. Submit this once with a short
# walltime purely to learn which world you are in.
#
# IF COMPUTE NODES HAVE NO NETWORK, the split that still works is:
#
#     # on the LOGIN node -- solve and DOWNLOAD only; no linking, no environment created
#     mamba env create -n openqha-gpu -f environment-tianhe-gpu.yml --download-only
#
#     # then submit this job with OFFLINE=1: it links from the package cache
#     OFFLINE=1 yhbatch -p h100x --gpus=1 -t 01:00:00 hpc/slurm/install_env_tianhe.slurm
#
# Note what that does and does not buy: **the SOLVE still happens on the login node**,
# because a download needs a solved package list. It moves the extract-and-link phase --
# tens of thousands of small files -- onto a compute node, which is the part that is
# slow on Lustre. If the solve itself is your problem, this arrangement does not help
# and the answer is the solver, i.e. (a) above.
#
# =======================================================================================
# 3. $HOME IS NOT OBVIOUSLY THE SAME FILESYSTEM ON EVERY CLUSTER
# =======================================================================================
# `conda env list` on this account shows one environment under BOTH
# `/HOME/hku2021_fos4/hku2021_fos4xy_2/miniforge3/envs` and
# `/XYAIFS00/HOME/hku2021_fos4/hku2021_fos4xy_2/miniforge3/envs`, while the account's own
# `init_conda.sh` sets `MAMBA_ROOT_PREFIX=/XYFS01/HOME/...`. Three mount names, one
# logical path. **XYFS01 and XYAIFS00 may be different filesystems**, in which case an
# environment built from TianheXY-CN is NOT the environment TianheXY-AI sees.
#
# This job prints the resolved path and the filesystem it landed on (section 8). Read
# that line before assuming an environment built on one cluster exists on another.
# =======================================================================================
#SBATCH --job-name=openqha_install
#SBATCH --nodes=1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# NO --exclusive: banned on TianheXY-AI (manual 6.2.2) and not passed on TianheXY-A
# either, by the 2026-09-05 ruling.
# NO --mem: the system sizes memory from the GPU count and specifying it is refused.
# NO --partition / --gpus / --time here ON PURPOSE: they differ per cluster and per run,
# so they are `yhbatch` arguments (see the header). A `#SBATCH` line cannot be
# parameterised, and a wrong one baked in here is a rejected submission on some other
# cluster (S0-G-62).

set -eo pipefail

ROLE="${ROLE:-gpu}"
OFFLINE="${OFFLINE:-}"

# ---- 4. site environment ---------------------------------------------------------------
module purge 2>/dev/null || true
module load anaconda3/2023.09 2>/dev/null || module load anaconda3 2>/dev/null || true

# Required by the site; without them the parallel run fails (D0-C-24). Harmless here.
ulimit -l unlimited 2>/dev/null || true
export GLEX_USE_ZC_RNDV=0

REPO="${OPENQHA_ROOT:-$SLURM_SUBMIT_DIR}"
[ -n "$REPO" ] || REPO="$(pwd)"
cd "$REPO"
echo "repository  : $REPO"
echo "host        : $(hostname)"
echo "job         : ${SLURM_JOB_ID:-<none>}  partition ${SLURM_JOB_PARTITION:-<none>}"
echo "cores       : ${SLURM_CPUS_ON_NODE:-$(nproc 2>/dev/null || echo ?)}"

# The account's own conda init. It is NOT in $HOME on this account -- it sits beside the
# checkout -- so look in both places rather than reporting "not found" at a file the
# operator can see with their own eyes.
CONDA_INIT="${CONDA_INIT:-}"
if [ -z "$CONDA_INIT" ]; then
    for c in "$HOME/init_conda.sh" "$REPO/../init_conda.sh"; do
        [ -f "$c" ] && { CONDA_INIT="$c"; break; }
    done
fi
if [ -f "$CONDA_INIT" ]; then
    # shellcheck disable=SC1090
    . "$CONDA_INIT"
    echo "conda init  : $CONDA_INIT"
else
    echo "no conda init found; set CONDA_INIT=<path to init_conda.sh>" >&2
    exit 1
fi
CONDA_BASE="$(conda info --base)"
echo "conda       : $CONDA_BASE  ($(conda --version 2>/dev/null || echo '?'))"

# ---- 5. proxy, then PROVE the network before spending the allocation --------------------
# This is section 2's question, answered in 15 seconds. A job that discovers a missing
# route halfway through a download has already cost the allocation; this one has not.
if [ -z "$https_proxy" ] && [ -z "$HTTPS_PROXY" ]; then
    SETPROXY="${TIANHE_SETPROXY:-/APP/u22/ai_x86/toolshs/setproxy.sh}"
    if [ -f "$SETPROXY" ]; then
        # shellcheck disable=SC1090
        . "$SETPROXY" "${TIANHE_PROXY_HOST:-172.16.31.200}" "${TIANHE_PROXY_PORT:-3138}" || true
        echo "proxy       : $SETPROXY"
    else
        echo "!! no proxy script at $SETPROXY -- compute nodes may not have one" >&2
    fi
fi
if [ -n "$https_proxy" ] || [ -n "$HTTPS_PROXY" ]; then
    export https_proxy="${https_proxy:-$HTTPS_PROXY}"; export HTTPS_PROXY="$https_proxy"
    export http_proxy="${http_proxy:-$https_proxy}";   export HTTP_PROXY="$http_proxy"
    export no_proxy="${no_proxy:-localhost,127.0.0.1,::1,.tianhe,.local}"
    export NO_PROXY="$no_proxy"
    echo "proxy       : ${https_proxy}"
fi

if [ -z "$OFFLINE" ]; then
    # The mirror THIS ACCOUNT uses, read out of its own condarc rather than hard-coded --
    # on Tianhe that is TUNA, via `custom_channels: conda-forge: <TUNA>/anaconda/cloud`.
    CF_BASE="$(conda config --show custom_channels 2>/dev/null \
               | awk '/^[[:space:]]*conda-forge:/ {print $2}')" || true
    PROBE="${CF_BASE:+$CF_BASE/conda-forge}/noarch/repodata.json"
    [ -n "$CF_BASE" ] || PROBE="https://conda.anaconda.org/conda-forge/noarch/repodata.json"
    CODE="$(curl -s -o /dev/null -w '%{http_code}' -I --max-time 15 "$PROBE" 2>/dev/null)" || true
    [ -n "$CODE" ] || CODE=000
    case "$CODE" in
        200|301|302) echo "network     : reachable, HTTP $CODE -- $PROBE" ;;
        *)
            echo "!! COMPUTE NODE CANNOT REACH $PROBE (HTTP $CODE)" >&2
            echo "!! This is the answer to section 2: compute nodes here have no route" >&2
            echo "!! off-cluster, or no proxy. Nothing below can work. Use the OFFLINE" >&2
            echo "!! split described in section 2:" >&2
            echo "!!   mamba env create -n <name> -f <file> --download-only   # login node" >&2
            echo "!!   OFFLINE=1 yhbatch ... hpc/slurm/install_env_tianhe.slurm" >&2
            exit 1 ;;
    esac
else
    echo "network     : OFFLINE=1, linking from the package cache only"
fi

# ---- 6. mamba, and nothing else ---------------------------------------------------------
# Not a preference. Measured 2026-09-08: classic 1 h 37 min and still running, versus
# 5 min 30 s. A compute node does not make a classic solve acceptable, it just hides it
# behind an allocation, so refuse rather than start one.
MAMBA=""
if command -v mamba >/dev/null 2>&1;   then MAMBA="$(command -v mamba)"
elif [ -x "$CONDA_BASE/bin/mamba" ];   then MAMBA="$CONDA_BASE/bin/mamba"
elif [ -n "$MAMBA_EXE" ] && [ -x "$MAMBA_EXE" ]; then MAMBA="$MAMBA_EXE"
fi
if [ -z "$MAMBA" ]; then
    echo "!! no mamba. miniforge3 ships it; this job will not start a classic solve." >&2
    echo "!!   conda install -n base mamba     # on the login node, then resubmit" >&2
    exit 1
fi
echo "solver      : $MAMBA"

# ---- __cuda -----------------------------------------------------------------------------
# A card-less allocation has no driver, so `__cuda` is absent and every conda-forge
# *cuda* build is unsatisfiable ("__cuda =* *, which is missing on the system" --
# measured on a login node 2026-09-08). An allocation WITH `--gpus` does have one, so
# this is usually redundant here; set it anyway, because a job submitted without --gpus
# would otherwise fail deep in the solve instead of building the environment it was
# asked for. Read from the yml so it cannot drift from the `cuda-version=` pin.
if [ "$ROLE" != "cpu" ] && [ -z "$CONDA_OVERRIDE_CUDA" ]; then
    _pin="$(grep -oE '^[[:space:]]*-[[:space:]]*cuda-version=[0-9]+\.[0-9]+' \
            environment-tianhe-gpu.yml 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' | head -1)" || true
    if [ -n "$_pin" ]; then
        export CONDA_OVERRIDE_CUDA="$_pin"
        echo "__cuda      : assumed $_pin for the solve"
    fi
    unset _pin
fi

export CONDA_NUMBER_CHANNEL_NOTICES=0
export CONDA_REMOTE_CONNECT_TIMEOUT_SECS=10.0
export CONDA_REMOTE_READ_TIMEOUT_SECS=60.0
export CONDA_REMOTE_MAX_RETRIES=2

# ---- 7. build --------------------------------------------------------------------------
build_one() {
    _file="$1"; _name="$2"
    echo
    echo "=== $_name  <-  $_file"
    [ -f "$_file" ] || { echo "!! missing $_file" >&2; return 1; }
    _t0=$(date +%s)
    if conda env list | awk '{print $1}' | grep -qx "$_name"; then
        echo "exists; updating in place from $_file"
        "$MAMBA" env update -n "$_name" -f "$_file" --prune ${OFFLINE:+--offline}
    else
        "$MAMBA" env create -y -n "$_name" -f "$_file" ${OFFLINE:+--offline}
    fi
    _t1=$(date +%s)
    # Print the cost. A solve time nobody recorded is a solve time nobody can compare
    # against next time -- and comparing is the only way the login-node question above
    # ever gets a number instead of an opinion.
    echo "=== $_name built in $((_t1 - _t0)) s on $(hostname)"
}

case "$ROLE" in
    gpu)  build_one environment-tianhe-gpu.yml "${ENV_NAME:-openqha-gpu}" ;;
    cpu)  build_one environment-tianhe.yml     "${ENV_NAME:-openqha}" ;;
    both) build_one environment-tianhe.yml     "${ENV_NAME_CPU:-openqha}"
          build_one environment-tianhe-gpu.yml "${ENV_NAME_GPU:-openqha-gpu}" ;;
    *)    echo "ROLE must be gpu, cpu or both (got '$ROLE')" >&2; exit 2 ;;
esac

# ---- 8. where did it land, and does it import? -----------------------------------------
# Section 3's question, answered with a path rather than an assumption.
for _n in ${ENV_NAME:-} ${ENV_NAME_CPU:-} ${ENV_NAME_GPU:-} openqha openqha-gpu; do
    _p="$(conda env list | awk -v n="$_n" '$1==n {print $NF}')" || true
    [ -n "$_p" ] || continue
    echo
    echo "$_n -> $_p"
    df -h "$_p" 2>/dev/null | tail -1 | sed 's/^/  filesystem  /'
done

# An environment file that LISTS a package is not evidence the package imports. The
# `=*cuda*` pin in particular is there so that a silently-cpu openmm-torch becomes a
# solver error -- but only an import proves what was actually linked.
if [ "$ROLE" != "cpu" ]; then
    echo
    echo "=== import check in ${ENV_NAME_GPU:-${ENV_NAME:-openqha-gpu}}"
    conda run -n "${ENV_NAME_GPU:-${ENV_NAME:-openqha-gpu}}" python - <<'PY' || true
import importlib
for m in ("torch", "openmm", "openmmtorch", "openmmtools", "MDAnalysis"):
    try:
        mod = importlib.import_module(m)
        print("  {:<14} {}".format(m, getattr(mod, "__version__", "ok")))
    except Exception as exc:
        print("  {:<14} FAILED: {}: {}".format(m, type(exc).__name__, exc))
try:
    import torch
    print("  torch.version.cuda", torch.version.cuda)
    # No GPU is required to BUILD the environment, and on a card-less allocation this is
    # expected to be False. It is printed, not asserted, for exactly that reason.
    print("  cuda available    ", torch.cuda.is_available())
except Exception:
    pass
PY
fi

echo
echo "done: $(date -Is)"
