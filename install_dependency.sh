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
#     bash install_dependency.sh --tianhe-cuda# TianheXY-AI (GPU) -- one CUDA 12.3 env
#     bash install_dependency.sh --tianhe-a   # TianheXY-A  (GPU) -- the same env
#
# THERE ARE THREE TIANHE CLUSTERS. The two GPU ones share an environment; the CPU
# one does not:
#   --tianhe       TianheXY-C  (CPU: debug/deimos)          -> branch A + result collection
#   --tianhe-cuda  TianheXY-AI (GPU: hx/h100x/a100x/...)     -> branch C training
#   --tianhe-a     TianheXY-A  (GPU: temp/ai, 8 cards/node)  -> branch B trajectories, C
#
# The two GPU flags build ONE file, environment-tianhe-gpu.yml, pinned at CUDA 12.3 --
# the version both module trees have. They differ only in which config submits the work.
#     bash install_dependency.sh --lean       # branch B production only, `openqha-openmm`
#
# ONE environment runs everything, including branch B's OpenMM route. That is a ruling of
# 2026-09-05 which accepted the downgrades the solver requires (pytorch 2.13.0 -> 2.12.1,
# numpy 2.4.6 -> 1.26.4) -- and the cost was MEASURED rather than assumed: energy, forces,
# every Hessian frequency, T*S on 2000 fixed frames and every quasi-harmonic frequency all
# came back BIT-IDENTICAL. See scripts/calibration/s0_B_stack_fingerprint.py.
#
# `--lean` builds `openqha-openmm` from environment-openmm.yml instead: the same branch B
# capability without CREST, xtb, rdkit or the notebook stack. Choose it for size, not for
# capability.
#
# --tianhe and --cuda build from environment-cuda.yml into `openqha-cuda`. That is for
# BRANCH C TRAINING ONLY: branches A and B are measured SLOWER on this project's GPU than
# on its CPU (D0-48, D0-C-5), and GFN2-xTB never touches a GPU at all.
#
# ONE environment, `openqha`, holding the Python stack, CREST, xtb and MACE.
#
# It used to be two, to dodge OpenBLAS printing "Detect OpenMP Loop and this application
# may hang" on every CREST step. environment.yml pins the OpenMP build of OpenBLAS, which
# removes the mismatch at its source. Measured 2026-09-04, dsgdb9nsd_000018, -T 4, same
# 2 conformers from every row:
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
# WHAT THIS SCRIPT WILL NOT DO
#   * install ORCA. Registration required; branch C only.
#   * download QM9 or curatedQM9. Both are large and neither belongs in version control.
#     Whether you use curatedQM9 at all is a choice you make with `f7_mode` -- see README.
# =====================================================================================

# NOTE: no `set -u`. The conda GROMACS activation hook fails under it
# ("GMXRC: line 10: shell: unbound variable") and the environment is then only half
# built while the script carries on. Measured 2026-09-03.
set -eo pipefail

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

for arg in "$@"; do
    case "$arg" in
        --tianhe|--hpc)  MODE="tianhe"; ENV_FILE="environment-tianhe.yml"
                         PY_ENV="${OPENQHA_ENV:-openqha}" ;;
        --cuda)          ENV_FILE="environment-cuda.yml"
                         PY_ENV="${OPENQHA_ENV:-openqha-cuda}" ;;
        # BOTH GPU clusters build the SAME environment file (user ruling 2026-09-07).
        # 12.3 is the version both module trees carry, and no MPI is loaded because
        # nothing openQHA runs on a card needs collectives -- which is exactly what let
        # the two files become one. See environment-tianhe-gpu.yml.
        --tianhe-cuda)   MODE="tianhe"; ENV_FILE="environment-tianhe-gpu.yml"
                         PY_ENV="${OPENQHA_ENV:-openqha-gpu}"
                         TIANHE_MODULES="CUDA/12.3" ;;
        --tianhe-a)      MODE="tianhe"; ENV_FILE="environment-tianhe-gpu.yml"
                         PY_ENV="${OPENQHA_ENV:-openqha-gpu}"
                         TIANHE_MODULES="CUDA/12.3" ;;
        --local)         MODE="local" ;;
        --minimal)       REQ="requirements-minimal.txt" ;;
        --lean)          ENV_FILE="environment-openmm.yml"; PY_ENV="openqha-openmm";
                         LEAN=1 ;;
        --check)         DO_INSTALL=0 ;;
        --no-weights)    WEIGHTS="none" ;;
        --all-weights)   WEIGHTS="all" ;;
        -h|--help)       sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
warn() { printf '\033[33m!! %s\033[0m\n' "$*"; }

say "openQHA installer -- mode: $MODE"
echo "repository  : $HERE"
echo "environment : $PY_ENV  (one environment; CREST and xtb are in it)"
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
SOLVER=()
if conda create --help 2>&1 | grep -q -- '--solver' && \
   "$CONDA_BASE/bin/python" -c "import conda_libmamba_solver" >/dev/null 2>&1; then
    SOLVER=(--solver=libmamba)
    echo "solver      : libmamba (asked for explicitly, not left to the default)"
else
    warn "libmamba solver not available; the classic solver will take much longer."
    warn "  Measured on this dependency set: libmamba minutes, classic tens of minutes."
    warn "  conda install -n base conda-libmamba-solver   # then re-run"
fi

# -------------------------------------------------------------------------------------
# 1. Site-specific preparation
# -------------------------------------------------------------------------------------
if [ "$MODE" = "tianhe" ]; then
    say "Tianhe preparation"
    # Required by the site and harmless anywhere else (D0-C-24). Without them the
    # parallel run fails.
    ulimit -l unlimited 2>/dev/null || warn "could not raise the locked-memory limit"
    export GLEX_USE_ZC_RNDV=0

    # Outbound traffic goes through a proxy. Set http_proxy/https_proxy in your shell
    # before running this, or conda will hang rather than fail.
    if [ -z "$https_proxy" ] && [ -z "$HTTPS_PROXY" ]; then
        warn "no https_proxy set. On Tianhe conda will HANG rather than report an error."
        warn "  export https_proxy=http://<host>:<port>   # then re-run"
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
    echo "     is the gate (user ruling 2026-09-07):"
    echo "       CPU  yhbatch hpc/slurm/branchA_debug.slurm     # debug partition, 00:30:00"
    echo "       GPU  bash hpc/slurm/submit_branchB_tianhe_a.sh a_debug temp"
    echo "  4. THREE CLUSTERS, three sets of rules:"
    echo "       TianheXY-C   CPU, whole-node --exclusive, partitions debug + deimos"
    echo "       TianheXY-AI  GPU per card, --gpus MANDATORY, --exclusive BANNED"
    echo "       TianheXY-A   GPU, 8 cards/node, -AI rules by the 2026-09-05 ruling"
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
if conda env list | awk '{print $1}' | grep -qx "$PY_ENV"; then
    echo "exists; updating in place from environment.yml"
    conda env update "${SOLVER[@]}" -n "$PY_ENV" -f environment.yml --prune
elif [ "$REQ" = "requirements-minimal.txt" ]; then
    # --minimal does not use environment.yml: it is deliberately a smaller set. The BLAS
    # pins are NOT optional here either -- see the header of environment.yml.
    conda create -y "${SOLVER[@]}" -n "$PY_ENV" -c conda-forge \
        python=3.11 nomkl "libopenblas=*=openmp*" "libblas=*=*openblas" \
        "liblapack=*=*openblas" \
        numpy scipy pyyaml ase rdkit "pytorch=*=*cpu*" "crest>=3.0.2" "xtb>=6.6"
else
    # --tianhe builds from environment-cuda.yml: ONE solve that knows about CUDA from the
    # start. The alternative -- build the CPU environment, then `conda install` a CUDA
    # torch over the top -- is how you end up with a half-swapped library set that
    # imports fine and dies at the first kernel launch.
    conda env create -y "${SOLVER[@]}" -n "$PY_ENV" -f "$ENV_FILE"
fi
conda activate "$PY_ENV"

# -------------------------------------------------------------------------------------
# 3. pip-only packages, MACE among them
# -------------------------------------------------------------------------------------
say "pip packages (mace-torch, pymsym, parsl)"
python -m pip install --upgrade pip
python -m pip install -r "$REQ"

# Named explicitly rather than left to the requirements file, because these three are
# the ones with no conda-forge package and the ones whose absence is least obvious:
# without mace-torch there is no potential at all.
python -m pip install "mace-torch>=0.3.6" "pymsym>=0.3.5"
[ "$REQ" = "requirements-minimal.txt" ] || python -m pip install "parsl>=2024.01"

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
# Every file is checked against the SHA-256 pinned in openqha/engine.py before it counts
# as installed. That check is what makes the level of theory reproducible: a truncated
# download, a proxy that served an HTML error page, and a silently swapped potential all
# look identical until you hash them.
MACE_ROOT="${S0_MACE_ROOT:-$HERE/data/potentials}"
MACE_URL_BASE="https://github.com/ACEsuit/mace-off/raw/main"

fetch_model () {          # fetch_model <registry-name> <family-dir> <file>
    local name="$1" fam="$2" file="$3"
    local dest="$MACE_ROOT/$fam/$file"
    local want
    want="$(python -c "from openqha import engine; print(engine.ENGINES['$name']['sha256'] or '')" 2>/dev/null)"

    if [ -f "$dest" ] && [ -n "$want" ] && \
       [ "$(sha256sum "$dest" | cut -d' ' -f1)" = "$want" ]; then
        echo "  $file: already present and matches its pinned digest"
        return 0
    fi

    mkdir -p "$MACE_ROOT/$fam"
    echo "  $file: downloading"
    if ! curl -fL --retry 3 --progress-bar -o "$dest.part" "$MACE_URL_BASE/$fam/$file"; then
        rm -f "$dest.part"
        warn "  $file: download failed. Fetch it by hand from $MACE_URL_BASE/$fam/$file"
        return 1
    fi

    local got
    got="$(sha256sum "$dest.part" | cut -d' ' -f1)"
    if [ -n "$want" ] && [ "$got" != "$want" ]; then
        # Do NOT install it. The commonest cause is a proxy returning an HTML error page
        # under a 200, which is a perfectly valid file and a completely wrong potential.
        rm -f "$dest.part"
        warn "  $file: SHA-256 MISMATCH -- NOT installed."
        warn "    expected $want"
        warn "    got      $got"
        return 1
    fi
    mv "$dest.part" "$dest"          # rename only after the hash agrees
    echo "  $file: installed, digest matches"
}

if [ "$WEIGHTS" = "none" ]; then
    say "MACE-OFF weights: skipped (--no-weights)"
    echo "  Branch A cannot run without them. When you have them, put them at"
    echo "    \$S0_MACE_ROOT/mace_off23/MACE-OFF23_medium.model"
elif [ "$MODE" = "tianhe" ]; then
    say "MACE-OFF weights: not downloaded on a login node"
    echo "  Outbound traffic goes through a proxy here and a 100 MB pull from a login"
    echo "  node is antisocial. Fetch them where you have bandwidth and copy them in:"
    echo "    rsync -a data/potentials/ <tianhe>:$HERE/data/potentials/"
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

# The potential's weights. Fetched by install_dependency.sh, hash-checked on every load.
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
MSG
