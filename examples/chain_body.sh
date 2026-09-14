#!/bin/bash
# THE WORK. Runs on a compute node (under one of examples/slurm/*.slurm) or locally.
#
#     bash examples/chain_body.sh <conf>
#
# It never submits anything. Submission is `examples/run_chain.sh`, which picks the right
# `.slurm` for the partition and hands it to sbatch/yhbatch. Splitting the two is what
# makes the job's stdout land in the job's own `--output` file instead of on the login
# node's terminal, and what lets each partition carry its own `#SBATCH` directives
# (`--cpus-per-task` for CPU, `--gpus` for GPU) which a single self-submitting file
# cannot.
#
# WHERE THE SETTINGS COME FROM
# ----------------------------
# One conf, sourced here, on the compute node. Not `--export`: that carries the
# submitting shell's whole environment into the job, so what the job saw depended on who
# submitted it, and the settings survived only in a scheduler record nobody reads.
#
# PARTITION and KIND come from the `.slurm` that launched this (they are properties of
# the job, not of the science), and fall back to the conf when run locally.
# =======================================================================================
# `set -eo pipefail` removed 2026-09-13 (user ruling: a failing step must not end the job; .mem/notes/notes_2026-09-13_no-errexit-anywhere.md)
# NOT `set -u`: the conda GROMACS activation hook fails under it and leaves the
# environment half-built while the script carries on. Measured 2026-09-03.

if [ -n "$SLURM_SUBMIT_DIR" ]; then
    ROOT="$SLURM_SUBMIT_DIR"
else
    ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$ROOT" || { echo "openQHA: cannot cd to $ROOT" >&2; exit 1; }

# ---------------------------------------------------------------------------------------
# `set -e` was removed project-wide on 2026-09-13 (user ruling, after an optional lookup
# ended a job silently). The other half of that ruling is this: a step whose PRODUCT the
# next step consumes must still stop the chain when it fails -- explicitly, by name, with
# its exit code -- or `collect` runs on a half-written store and the job ends "successfully".
# Everything that is not such a step (a diagnostic, a lookup, a copy-back) is allowed to
# fail and is written to say so.
must() {
    "$@"
    local rc=$?
    if [ "$rc" -ne 0 ]; then
        echo >&2
        echo "openQHA: step failed with exit $rc:" >&2
        echo "    $*" | cut -c1-200 >&2
        echo "  Its product is what the next step reads, so the chain stops here." >&2
        exit "$rc"
    fi
}

CONF="${1:-}"
[ -n "$CONF" ] || { echo "usage: bash examples/chain_body.sh <conf>" >&2; exit 2; }
[ -f "$CONF" ] || { echo "no such conf: $CONF" >&2; exit 2; }
# shellcheck disable=SC1090
source "$CONF"

SPECIES="${SPECIES:?the conf must set SPECIES}"
TAG="${TAG:-chain}"
SEEDS="${SEEDS:-1}"          # one trajectory per (molecule, basin): ruling 2026-09-13
THREADS="${THREADS:-4}"
CHAIN="${CHAIN:-qha}"

# ---------------------------------------------------------------------------------------
# Branch A diagnostics. All three OFF unless the conf asks for them.
# ---------------------------------------------------------------------------------------
# They exist because of job 7347197 (2026-09-09): CREST finished cleanly and handed back
# 1814 conformers with NaN for every energy, and the run left **no evidence whatsoever**
# about why -- the external client's stderr goes into CREST's calcspace, and CREST
# deletes the calcspace.
#
#   MAX_CONFORMERS   a ceiling for THIS molecule. Not a physical law and not a default:
#                    acetone at 10 atoms cannot have hundreds of conformers, and that is
#                    a statement about acetone, so it lives in the example's conf.
#   KEEP_CALCSPACE   name CREST's external-calculator directory so it persists.
#                    Keeps files for every gradient call -- a run you are watching, not
#                    a campaign.
#   MACE_TRACE       one line per gradient from BOTH ends of the socket: the client
#                    (which writes outside the calcspace, so it survives) and the server.
#                    Comparing the server's total against `Total number of energy+grad
#                    calls` in crest.out is what separates "the client never arrived"
#                    from "the client arrived and its answer never got back to CREST".
BRANCH_A_ARGS=""
[ -n "${MAX_CONFORMERS:-}" ] && BRANCH_A_ARGS="$BRANCH_A_ARGS --max-conformers $MAX_CONFORMERS"
[ -n "${KEEP_CALCSPACE:-}" ] && BRANCH_A_ARGS="$BRANCH_A_ARGS --keep-calcspace $KEEP_CALCSPACE"

# Set by the .slurm; on a local run there is no .slurm, so the conf decides.
PARTITION="${OPENQHA_PARTITION:-${PARTITION:-local}}"
KIND="${OPENQHA_KIND:-cpu}"

if [ "$KIND" = "gpu" ]; then
    # TianheXY-AI has several GPU partitions and hands out what it has: `h100x` gave an
    # A100-SXM4-80GB on an104 (2026-09-13), and the allocation reported itself as `a100x`.
    # Any of them is that cluster's resource config; only `ai` is TianheXY-A.
    case "$PARTITION" in
        h100x|a100x|a800x|v100x|hx) _res_default=tianhe_ai ;;
        *)                          _res_default=tianhe_a ;;
    esac
    RESOURCE="${RESOURCE:-$_res_default}"
    ROUTE="${ROUTE:-openmm}"; PLATFORM="CUDA"
elif [ -n "$SLURM_JOB_ID" ]; then
    RESOURCE="${RESOURCE:-tianhe_cpu}"; ROUTE="${ROUTE:-ase}"; PLATFORM="CPU"
else
    RESOURCE="${RESOURCE:-local}"; ROUTE="${ROUTE:-ase}"; PLATFORM="CPU"
fi

# ---------------------------------------------------------------------------------------
# Environment. Only a job needs the site modules; a local run uses whatever is active.
# ---------------------------------------------------------------------------------------
openqha_load_conda_module() {
    local m
    for m in "${OPENQHA_CONDA_MODULE:-}" anaconda3/202309 anaconda3/20250601 \
             miniconda3/202409 miniforge/24.7.1 anaconda3 miniforge; do
        [ -n "$m" ] || continue
        if module load "$m" >/dev/null 2>&1; then echo "module    $m"; return 0; fi
    done
    if command -v conda >/dev/null 2>&1; then
        echo "module    none loaded; conda already on PATH ($(command -v conda))"
        return 0
    fi
    echo "openQHA: no conda module could be loaded and conda is not on PATH." >&2
    echo "  Tried: \$OPENQHA_CONDA_MODULE, anaconda3/202309, anaconda3/20250601," >&2
    echo "  miniconda3/202409, miniforge/24.7.1. Run 'module avail' and set" >&2
    echo "  OPENQHA_CONDA_MODULE to the one this site actually has." >&2
    return 1
}

# ---------------------------------------------------------------------------------------
# THE CUDA TOOLKIT: measured, not assumed (2026-09-13)
# ---------------------------------------------------------------------------------------
# This used to `module load CUDA/12.2` for every GPU job, before the environment was even
# activated. Two clusters, two ways that was wrong:
#
#   * TianheXY-A (an45): the environment's nvrtc was 12.3, the driver 12.2; the module's
#     12.2 nvrtc, placed ahead on LD_LIBRARY_PATH, was the fix. There it was NEEDED.
#   * TianheXY-AI (an104): the environment's nvrtc is now 12.4 and so is the driver --
#     nothing to fix -- and the site's CUDA/12.4 module puts its lib64/stubs/ on
#     LD_LIBRARY_PATH, so loading it hands every CUDA program the link-time STUB:
#     "You are running using the stub version of nvrtc", then CUDA error 34
#     (CUDA_ERROR_STUB_LIBRARY) at openmm.Context(). There it was HARMFUL.
#
# So the decision is made after activation, from what openqha/gpu_preflight.py measures
# in this process: if the environment's toolkit already fits the driver, load nothing; if
# it is too new, try site modules in order and keep the first one that fits; and after
# any load, strip stubs/ directories from LD_LIBRARY_PATH before judging, because a stub
# is never an answer. None of this ends the job by itself -- `must` and the driver's own
# preflight do that -- except the one case with no way forward: a toolkit too new for
# the driver and no module that fits.
openqha_strip_stubs() {
    case ":${LD_LIBRARY_PATH:-}:" in
        *stubs*)
            export LD_LIBRARY_PATH="$(printf '%s' "${LD_LIBRARY_PATH}" | tr : '\n' | grep -v stubs | paste -sd: -)"
            echo "cuda      removed a stubs/ directory from LD_LIBRARY_PATH" ;;
    esac
}
openqha_cuda_verdict() {          # line 1: ok | stub | newer | unknown;  line 2: details
    python - <<'PY' 2>/dev/null || printf 'unknown\n(gpu_preflight not importable)\n'
import sys
sys.path.insert(0, ".")
from openqha import gpu_preflight as g
d = g.describe()
if d.get("stub_loaded") or (d.get("ld_stubs") and not d["nvrtc"]):
    print("stub")
elif d["nvrtc"] and d["driver_cuda"]:
    print("ok" if d["ok"] else "newer")
else:
    print("unknown")
print("nvrtc {} from {}  driver {}".format(
    ".".join(map(str, d["nvrtc"])) if d["nvrtc"] else "?", d.get("nvrtc_path") or "?",
    ".".join(map(str, d["driver_cuda"])) if d["driver_cuda"] else "?"))
PY
}
openqha_cuda_fit() {
    local out v info drv m
    openqha_strip_stubs
    out="$(openqha_cuda_verdict)"; v="${out%%$'\n'*}"; info="${out#*$'\n'}"
    echo "cuda      $info"
    case "$v" in
        ok)      echo "cuda      the environment's toolkit fits the driver; no site module loaded"
                 unset OPENQHA_CUDA_MODULE_CHOSEN          # workers must load none either
                 return 0 ;;
        unknown) echo "cuda      nvrtc or driver not readable here; the driver's preflight decides"
                 return 0 ;;
    esac
    drv="$(printf '%s' "$info" | sed -nE 's/.*driver ([0-9]+\.[0-9]+).*/\1/p')"
    for m in "${OPENQHA_CUDA_MODULE:-}" ${drv:+CUDA/$drv} CUDA/12.2; do
        [ -n "$m" ] || continue
        module load "$m" >/dev/null 2>&1 || continue
        openqha_strip_stubs
        out="$(openqha_cuda_verdict)"; v="${out%%$'\n'*}"; info="${out#*$'\n'}"
        if [ "$v" = ok ]; then
            echo "module    $m"
            echo "cuda      $info"
            # Parsl worker blocks start in a fresh shell; they load this and nothing else.
            export OPENQHA_CUDA_MODULE_CHOSEN="$m"
            return 0
        fi
        echo "cuda      $m loaded but $v ($info); unloading"
        module unload "$m" >/dev/null 2>&1
    done
    echo "openQHA: this driver cannot JIT the environment's CUDA toolkit ($info), and no" >&2
    echo "  site module fixed that. On a LOGIN node (compute nodes have no network):" >&2
    echo "      mamba install -n openqha-gpu cuda-version=${drv:-<driver version>}" >&2
    return 1
}

if [ -n "$SLURM_JOB_ID" ]; then
    module purge 2>/dev/null || true
    openqha_load_conda_module || exit 1
    # Inherited task-layout variables make a child process misread its allocation and try
    # to relaunch itself through the scheduler (rule 7). CREST forks its own workers.
    for v in $(env | awk -F= '{print $1}' \
               | grep -E '^(PMI|SLURM_(CPU|TASK|NTASKS|NPROCS|STEP))'); do
        unset "$v"
    done
    # cpu -> `openqha` (crest, xtb, pinned OpenBLAS); gpu -> `openqha-gpu` (CUDA torch,
    # OpenMM, MKL). hpc/env/tianhe.sh does the mapping; OPENQHA_ENV overrides both.
    export OPENQHA_ROLE="${OPENQHA_ROLE:-$KIND}"
    source hpc/env/common.sh
    source hpc/env/tianhe.sh || { echo "openQHA: hpc/env/tianhe.sh refused (root unresolved); stopping." >&2; exit 1; }
    openqha_report_env
    if [ "$KIND" = "gpu" ]; then
        openqha_cuda_fit || exit 1      # the one environment fault with no way forward
    fi
fi

# Every chain ends in a step that writes parquet tables (collect, report, the 02c
# benchmark, branch A's census). Those imports happen LAST, after the hours of compute
# they summarise, so the environment is asked for them FIRST. require.sh has no side
# effects (common.sh does: threads, runs root), so it is sourced here on every site.
source hpc/env/require.sh

# parsl's run directory: a record about the job, not about a molecule, so it goes with
# the tag's other per-tag records, <root>/<tag>/_records/parsl/<job>.<pid>/ (user ruling
# 2026-09-14 on per-tag leftovers), one per process. NOT under the checkout: parsl chmods
# its certificates directory to 0700, which a Windows drive under WSL refuses (measured
# 2026-09-15), and the root is on a filesystem that honours modes on every site.
export S0_PARSL_RUN_DIR="${S0_PARSL_RUN_DIR:-${S0_RUNS_ROOT:-$HOME/runs/openQHA}/$TAG/_records/parsl/${SLURM_JOB_ID:-pid$$}.$$}"

openqha_require_modules pandas pyarrow || {
    echo "  Every chain's collect/report step writes parquet and fails at its LAST line" >&2
    echo "  without them -- after the trajectories (an113, 2026-09-13). On a login node:" >&2
    echo "      pip install pyarrow     # the conda solve was refused on ln301, 2026-09-13" >&2
    echo "  then confirm numpy was left alone:" >&2
    echo "      python scripts/tooling/s0_probe_openmm_cuda.py --quiet --no-accuracy --steps 10" >&2
    exit 2
}

# ---------------------------------------------------------------------------------------
# NO COPY-BACK (ADR 0002, 2026-09-14). The molecule tree is written once, on the shared
# filesystem, at $S0_RUNS_ROOT (hpc/env/root.sh derives it from the partition). Until
# 2026-09-14 this block installed an EXIT/TERM/INT trap that copied the whole per-job
# scratch into logs/node_local/<jobid>/ with a MANIFEST.txt; the scratch existed for
# sockets, which now live node-local in $S0_SCRATCH and are not worth carrying.
# ---------------------------------------------------------------------------------------
if [ -n "$SLURM_JOB_ID" ]; then
    echo "scratch   ${S0_SCRATCH:-unset}   (node-local: sockets; nothing is copied back)"
fi

# The trace directory has to be somewhere that OUTLIVES the calcspace -- not in the
# calcspace CREST removes, and not in the node-local scratch nothing carries back any
# more. It is a diagnostic log, so it goes beside the job logs.
if [ -n "${MACE_TRACE:-}" ]; then
    # A value containing a slash is taken as the directory itself. Anything else
    # (conventionally `1`) goes under logs/trace/, one directory per job or process.
    case "$MACE_TRACE" in
        */*) S0_MACE_TRACE="$MACE_TRACE" ;;
        *)   S0_MACE_TRACE="$ROOT/logs/trace/${SLURM_JOB_ID:-pid$$}" ;;
    esac
    mkdir -p "$S0_MACE_TRACE"
    export S0_MACE_TRACE
    echo "trace     $S0_MACE_TRACE   (one line per gradient, both ends of the socket)"
fi

echo
echo "======================================================================"
echo "openQHA   chain $CHAIN   species $SPECIES   tag $TAG"
echo "  partition $PARTITION${SLURM_JOB_ID:+   job $SLURM_JOB_ID}   kind $KIND"
echo "  resource  $RESOURCE   route $ROUTE   platform $PLATFORM"

# ---------------------------------------------------------------------------------------
# THE TASK BUDGET MUST MATCH THIS JOB, NOT THE SITE DEFAULT
# ---------------------------------------------------------------------------------------
# A branch B task stops itself at a budget and flushes, rather than being killed between a
# write and a rename. The driver defaults that budget to the RESOURCE CONFIG's
# QHA_WALL_BUDGET_S -- 90% of `tianhe_a.WALLTIME`, which is 24 h since 2026-09-12.
#
# That default is wrong for any job whose --time differs from the site default, and 02d-2
# is exactly that: its array asks for a longer job precisely because a p1500 row needs
# ~25.8 h. With the budget left at 21.6 h the task would stop at 21.6 h INSIDE a 3-day
# job -- the longer walltime buying nothing, silently. (It was invisible until 2026-09-12
# because the site default was 7 days and never bound.)
#
# So ask the queue. `squeue -h -j <id> -o %L` is the time this job has LEFT, which is the
# only number that is true regardless of which .slurm, conf or array wrote --time. Fall
# back to the conf's WALLTIME, then to the driver's own default.
#
# **NOTHING IN THIS BLOCK MAY BE ABLE TO END THE JOB.** This file used to run under
# `set -eo pipefail` (removed project-wide 2026-09-13; the fallbacks below are kept
# because they also make the log say WHY it fell back), and the first version of the
# block did not respect that: when
# `squeue` exited non-zero (a compute node whose squeue talks to the other controller,
# say), pipefail marked the pipeline failed, the assignment failed, and errexit ended the
# job on this line -- with stderr already sent to /dev/null. Measured 2026-09-13 with a
# squeue stub that exits 1: the script died before printing anything about it. That is a
# job that ends in seconds with a header and no explanation, over a number whose worst
# case is "use the default". Every command below therefore carries its own fallback.
_wall_seconds() {           # D-HH:MM:SS | HH:MM:SS | MM:SS | SS  ->  seconds, or 0
    local spec="$1" d=0 rest a b c
    case "$spec" in *-*) d="${spec%%-*}"; rest="${spec#*-}" ;; *) rest="$spec" ;; esac
    IFS=: read -r a b c <<<"$rest" || true
    if [ -n "$c" ]; then :
    elif [ -n "$b" ]; then c="$b"; b="$a"; a=0
    else c="$a"; b=0; a=0; fi
    # A non-numeric field is an arithmetic error, which errexit would treat as fatal.
    case "${d}${a}${b}${c}" in *[!0-9]*) echo 0; return 0 ;; esac
    echo $(( 10#${d:-0} * 86400 + 10#${a:-0} * 3600 + 10#${b:-0} * 60 + 10#${c:-0} ))
}

WALL_BUDGET_ARG=""
_budget_src=""
_budget_note=""
if [ -n "${SLURM_JOB_ID:-}" ] && command -v squeue >/dev/null 2>&1; then
    _left="$(squeue -h -j "$SLURM_JOB_ID" -o "%L" 2>/dev/null | tr -d ' ' || true)"
    case "$_left" in
        "")                           _budget_note="squeue gave nothing for job $SLURM_JOB_ID" ;;
        UNLIMITED|NOT_SET|INVALID)    _budget_note="squeue time left is $_left" ;;
        *) _secs="$(_wall_seconds "$_left" 2>/dev/null || echo 0)"
           if [ "${_secs:-0}" -gt 0 ]; then
               WALL_BUDGET_ARG="--wall-budget-s $(( _secs * 9 / 10 ))"
               _budget_src="squeue, time left $_left"
           else
               _budget_note="could not parse squeue time left '$_left'"
           fi ;;
    esac
fi
if [ -z "$WALL_BUDGET_ARG" ] && [ -n "${WALLTIME:-}" ]; then
    _secs="$(_wall_seconds "$WALLTIME" 2>/dev/null || echo 0)"
    if [ "${_secs:-0}" -gt 0 ]; then
        WALL_BUDGET_ARG="--wall-budget-s $(( _secs * 9 / 10 ))"
        _budget_src="conf WALLTIME=$WALLTIME"
    fi
fi
if [ -n "$WALL_BUDGET_ARG" ]; then
    echo "  budget    ${WALL_BUDGET_ARG#--wall-budget-s } s  (90% of $_budget_src)${_budget_note:+  [$_budget_note]}"
else
    echo "  budget    from the resource config${_budget_note:+  [$_budget_note]}"
fi
echo "  seeds     $SEEDS   threads $THREADS   conf $CONF"
echo "======================================================================"

# =======================================================================================
# BRANCH A: MADE ONCE, ON THE CPU, AND CONSUMED BY EVERYTHING ELSE
# =======================================================================================
# Every chain starts from the basins; only one of them makes them. That split is what
# lets a molecule be run as two submissions -- a CPU job that establishes the basins, then
# GPU jobs that consume them -- instead of every GPU job repeating a CREST search it
# cannot even run. `openqha-gpu` has NO crest and NO xtb.
#
# Keyed by (SPECIES, TAG): the two confs of an example share a TAG for this reason.
# Basins are READ from BASIN_TAG (default: TAG) and everything this run makes is WRITTEN
# under TAG. The identity chain always worked this way; the qha chain does too since
# 2026-09-13, so a 30-minute test can reuse production basins without touching the
# production trajectory store -- which the driver would otherwise resume from.
BASINS_PRESENT=$(python - "$SPECIES" "${BASIN_TAG:-$TAG}" <<'PY'
import sys
sys.path.insert(0, ".")
try:
    from openqha.store import basins
    print("yes" if basins.exists(sys.argv[1], tag=sys.argv[2]) else "no")
except Exception:                                                 # noqa: BLE001
    print("unknown")
PY
)

if [ "$CHAIN" = "conformers_pair" ]; then
    # =================================================================================
    # TWO MOLECULES, CONCURRENTLY, IN **ONE** ALLOCATION
    # =================================================================================
    # Not two `sbatch` submissions. Branch A is embarrassingly parallel over molecules
    # and useless to parallelise within one (CREST's own `threads` covers the
    # metadynamics; MACE on ten atoms is 111/90/72/101 ms at 1/2/4/8 threads, so eight is
    # slower than four). One job holding a whole deimos node therefore runs several
    # molecules side by side rather than one molecule and 60 idle cores -- job 7347197
    # used about 1.8% of the node it held exclusively.
    #
    # Each entry is:   SPECIES  TAG  THREADS  [extra arguments for this molecule]
    # The extras are per molecule on purpose: a conformer ceiling is a claim about one
    # specific molecule and must not be inherited by its neighbour.
    #
    # Each pipeline starts its OWN MACE server, and the socket name carries the pid
    # (openqha/conformer_search/crest.py), so the two do not collide inside the one
    # scratch directory this job has.
    echo
    echo "---- branch A x N, concurrently, in ONE job ----------------------------"
    pair_pids=""; pair_tags=""; pair_n=0
    LOGBASE="$ROOT/logs/openqha_pair${SLURM_JOB_ID:+_$SLURM_JOB_ID}"
    mkdir -p "$ROOT/logs"
    for _entry in "${PAIR_1:-}" "${PAIR_2:-}" "${PAIR_3:-}" "${PAIR_4:-}"; do
        [ -n "$_entry" ] || continue
        # shellcheck disable=SC2086  -- the entry is a deliberate word list
        set -- $_entry
        p_species="$1"; p_tag="$2"; p_threads="${3:-4}"; shift 3 2>/dev/null || shift $#
        p_extra="$*"
        p_log="${LOGBASE}_${p_tag}.log"
        echo "     $p_species   tag $p_tag   threads $p_threads   ${p_extra:-(no extra args)}"
        echo "       -> $p_log"
        # shellcheck disable=SC2086
        python -u scripts/production/s0_A_pipeline.py \
            --species "$p_species" --tag "$p_tag" --threads "$p_threads" \
            --hessian-mode analytic $BRANCH_A_ARGS $p_extra > "$p_log" 2>&1 &
        pair_pids="$pair_pids $!"
        pair_tags="$pair_tags $p_tag"
        pair_n=$((pair_n + 1))
    done
    [ "$pair_n" -gt 0 ] || { echo "conformers_pair: the conf set no PAIR_1..PAIR_4" >&2; exit 2; }
    echo "     $pair_n pipeline(s) running; waiting for all of them"

    # **Wait for every one and report every one.** `wait` without arguments returns the
    # status of the last job only, which would let one molecule fail invisibly beside a
    # neighbour that succeeded.
    pair_rc=0
    set -- $pair_tags
    for _pid in $pair_pids; do
        _tag="$1"; shift
        if wait "$_pid"; then
            echo "     $_tag  OK"
        else
            _rc=$?
            echo "     $_tag  FAILED (exit $_rc) -- see ${LOGBASE}_${_tag}.log" >&2
            pair_rc=1
        fi
    done
    echo
    echo "---- the two logs, tail ------------------------------------------------"
    for _tag in $pair_tags; do
        echo "===== $_tag ====="
        tail -25 "${LOGBASE}_${_tag}.log" 2>/dev/null || echo "(no log)"
        echo
    done
    exit "$pair_rc"
fi

if [ "$CHAIN" = "conformers" ]; then
    echo
    echo "---- branch A: conformer search -> every basin ------------------------"
    [ "$BASINS_PRESENT" = "yes" ] && \
        echo "     (a record already exists for tag '$TAG'; re-running it -- this
     chain's product IS branch A)"
    # shellcheck disable=SC2086  -- BRANCH_A_ARGS is a deliberate word list, empty unless
    # the conf turned a diagnostic on
    must python -u scripts/production/s0_A_pipeline.py \
        --species "$SPECIES" --tag "$TAG" --threads "$THREADS" \
        --hessian-mode analytic $BRANCH_A_ARGS
elif [ "$BASINS_PRESENT" = "yes" ]; then
    echo
    echo "---- branch A: already done for tag '${BASIN_TAG:-$TAG}' -- reusing those basins ----"
elif [ "$KIND" = "gpu" ]; then
    echo "openQHA: no branch A product for '$SPECIES' under tag '$TAG', and branch A" >&2
    echo "  cannot run here (openqha-gpu has neither crest nor xtb). Run step 1 on the" >&2
    echo "  CPU cluster first, with the SAME tag:" >&2
    echo "    bash examples/run_chain.sh $(dirname "$CONF")/branchA.conf deimos" >&2
    exit 2
else
    echo
    echo "---- branch A: conformer search -> every basin ------------------------"
    echo "     (no record for tag '$TAG' yet, and this is a CPU run, so making it here)"
    # shellcheck disable=SC2086  -- BRANCH_A_ARGS is a deliberate word list, empty unless
    # the conf turned a diagnostic on
    must python -u scripts/production/s0_A_pipeline.py \
        --species "$SPECIES" --tag "$TAG" --threads "$THREADS" \
        --hessian-mode analytic $BRANCH_A_ARGS
fi

# =======================================================================================
# The chains. Every step is a production or example driver -- this file adds no science,
# and every science setting comes from the conf or from configs/branchB_protocol.yaml.
# =======================================================================================
case "$CHAIN" in

conformers)
    echo
    echo "branch A only. The basins are the product; nothing else runs."
    ;;

qha)    # the conformational free energy: A -> B -> collect -> F_conf
    echo
    echo "---- branch B: one trajectory per (basin, seed) -----------------------"
    # --basins auto: the count comes from branch A's own record, so the number of basins
    # and the geometries they start from cannot disagree.
    must python -u scripts/production/s0_E_branchB_parsl.py \
        --species "$SPECIES" --tag "$TAG" --basin-tag "${BASIN_TAG:-$TAG}" \
        --resource "$RESOURCE" \
        --route "$ROUTE" --basins auto --seeds "$SEEDS" ${SETTING:+--setting "$SETTING"} \
        ${MAX_WORKERS:+--max-workers "$MAX_WORKERS"} ${BLOCK_GPUS:+--gpus "$BLOCK_GPUS"} \
        ${PROD_PS:+--prod-ps "$PROD_PS"} ${EQUIL_PS:+--equil-ps "$EQUIL_PS"} \
        $WALL_BUDGET_ARG

    echo
    echo "---- collect: quasi-harmonic analysis per molecule --------------------"
    must python -u scripts/production/s0_E_branchB_collect_parsl.py \
        --species "$SPECIES" --tag "$TAG" --basin-tag "${BASIN_TAG:-$TAG}" \
        ${SETTING:+--setting "$SETTING"} --resource "${COLLECT_RESOURCE:-$RESOURCE}"

    echo
    echo "---- the answer: F_conf over the ensemble -----------------------------"
    must python -u scripts/production/s0_B_report_ensemble.py \
        --species "$SPECIES" --tag "$TAG" --basin-tag "${BASIN_TAG:-$TAG}" \
        ${SETTING:+--setting "$SETTING"}
    ;;

levels) # 02c: MACE vs GFN2-xTB vs RI-MP2, energies and forces through to G - E_el
    # CPU only. The cost is entirely the RI-MP2 `TightOpt NumFreq`: 1613 s per 10-atom
    # structure at 4 processes, one ORCA job per basin, in sequence.
    echo
    echo "---- 02c: three levels on every basin ---------------------------------"
    must python -u examples/02c_hessian_benchmark_levels/s0_level_benchmark.py \
        --species "$SPECIES" --tag "$TAG" \
        --levels "${LEVELS:-mace,gfn2,rimp2}" --nprocs "${ORCA_NPROCS:-$THREADS}" \
        ${ORCA_MAXCORE:+--maxcore "$ORCA_MAXCORE"}
    ;;

identity) # 02d: may nu_k replace omega_i in ZPE, enthalpy and entropy?
    # The trajectories are DENSELY sampled on purpose: the protocol's 1.0 ps interval is
    # chosen for the ENTROPY, while the zero-point energy needs of order 12 500 frames,
    # which at 1.0 ps would be 12.5 ns. Both numbers are measured in 02d stage 1.
    echo
    # BASIN_TAG: where branch A's basins are read from, when this run writes its
    # trajectories under a different TAG. examples/02d-2 runs many settings off ONE branch
    # A product; each row is its own TAG and they all read BASIN_TAG. Default: TAG.
    echo "---- branch B: dense trajectories, ${PROD_PS:-protocol} ps ------------"
    must python -u scripts/production/s0_E_branchB_parsl.py \
        --species "$SPECIES" --tag "$TAG" --basin-tag "${BASIN_TAG:-$TAG}" \
        --resource "$RESOURCE" \
        --route "$ROUTE" --basins auto --seeds "$SEEDS" ${SETTING:+--setting "$SETTING"} \
        ${MAX_WORKERS:+--max-workers "$MAX_WORKERS"} ${BLOCK_GPUS:+--gpus "$BLOCK_GPUS"} \
        ${PROD_PS:+--prod-ps "$PROD_PS"} ${EQUIL_PS:+--equil-ps "$EQUIL_PS"} \
        ${SAMPLE_EVERY:+--sample-every "$SAMPLE_EVERY"} \
        $WALL_BUDGET_ARG

    echo
    echo "---- 02d: G_total from omega and from nu, term by term ----------------"
    # --tag reads the basins (BASIN_TAG); --traj-tag reads this run's trajectories; --out
    # files the report under this run's tag, so two settings never overwrite each other.
    # --tag names the molecule directory (branch A's tag); --setting the openmm/<setting>/
    # the trajectories went to; the report lands in _records/openmm/<setting>/ (ADR 0001).
    must python -u examples/02d_qha_frequency_identity/s0_frequency_identity.py \
        --species "$SPECIES" --tag "${BASIN_TAG:-$TAG}" --stage all \
        ${SETTING:+--setting "$SETTING"} \
        ${NU_CUT:+--nu-cut "$NU_CUT"}
    ;;

*)  echo "CHAIN must be conformers, qha, levels or identity, got '$CHAIN'" >&2
    exit 2 ;;
esac

echo
echo "chain '$CHAIN' finished for $SPECIES (tag $TAG)."
