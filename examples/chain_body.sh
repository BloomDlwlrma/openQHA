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
set -eo pipefail
# NOT `set -u`: the conda GROMACS activation hook fails under it and leaves the
# environment half-built while the script carries on. Measured 2026-09-03.

if [ -n "$SLURM_SUBMIT_DIR" ]; then
    ROOT="$SLURM_SUBMIT_DIR"
else
    ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$ROOT"

CONF="${1:-}"
[ -n "$CONF" ] || { echo "usage: bash examples/chain_body.sh <conf>" >&2; exit 2; }
[ -f "$CONF" ] || { echo "no such conf: $CONF" >&2; exit 2; }
# shellcheck disable=SC1090
source "$CONF"

SPECIES="${SPECIES:?the conf must set SPECIES}"
TAG="${TAG:-chain}"
SEEDS="${SEEDS:-3}"
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
    RESOURCE="${RESOURCE:-$([ "$PARTITION" = "h100x" ] && echo tianhe_ai || echo tianhe_a)}"
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

if [ -n "$SLURM_JOB_ID" ]; then
    module purge 2>/dev/null || true
    openqha_load_conda_module || exit 1
    if [ "$KIND" = "gpu" ]; then
        module load CUDA/12.3 || {
            echo "openQHA: module load CUDA/12.3 FAILED -- this would run on the CPU" >&2
            exit 1; }
    fi
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
    source hpc/env/tianhe.sh
    openqha_report_env
fi

# ---------------------------------------------------------------------------------------
# NODE-LOCAL SCRATCH: run there, carry the end state back, then remove it.
# ---------------------------------------------------------------------------------------
# `hpc/env/tianhe.sh` puts the whole job under `<TMPDIR or /tmp>/<owner>/<job id>/` --
# sockets, runs, CREST's working directories, all of it. Two reasons, both measured:
#
#   * CREST writes many small files into parallel `_N` subdirectories, and doing that on
#     Lustre is slow for this job and for everyone else on the machine;
#   * two branch A jobs on two compute nodes both opened the same
#     `runs_root/sockets/s0_mace_pool_0.sock` on the SHARED filesystem (2026-09-09), and
#     whichever bound second unlinked the first one's socket.
#
# The obligation that comes with node-local storage: **anything written there is gone
# when the job ends unless the job carries it back**. So the whole tree is copied to
# `logs/node_local/<jobid>/` first -- `cp -a`, which recreates socket nodes too, so the
# end state is what you see -- and only then removed.
if [ -n "$SLURM_JOB_ID" ] && [ -n "$S0_SCRATCH" ]; then
    KEEP_DIR="$ROOT/logs/node_local/$SLURM_JOB_ID"
    echo "scratch   $S0_SCRATCH"
    echo "          -> end state kept at logs/node_local/$SLURM_JOB_ID/"
    keep_scratch() {
        local rc=$? src dst
        mkdir -p "$KEEP_DIR" 2>/dev/null || true

        # THE MANIFEST IS WRITTEN FIRST, and it is the part that always works.
        # `cp -a` carries socket nodes on a filesystem that supports them and drops them
        # SILENTLY on one that does not (measured: a DrvFs destination takes the logs and
        # not the `.sock`). So the listing of what was there is recorded before any copy
        # is attempted, and the copy is then checked against it.
        { echo "# end state of $S0_SCRATCH"
          echo "# job $SLURM_JOB_ID on $(hostname), exit $rc, $(date -Is)"
          echo
          ls -laR "$S0_SCRATCH" 2>/dev/null
        } > "$KEEP_DIR/MANIFEST.txt" 2>/dev/null || true

        # `cp -a .` copies the CONTENTS, dotfiles included, preserving modes and times.
        ( cd "$S0_SCRATCH" && cp -a . "$KEEP_DIR/" ) 2>>"$KEEP_DIR/MANIFEST.txt" || true

        # Say what did not make it, rather than leaving a quietly shorter directory.
        src=$(cd "$S0_SCRATCH" && find . -mindepth 1 | wc -l)
        dst=$(cd "$KEEP_DIR" && find . -mindepth 1 ! -name MANIFEST.txt | wc -l)
        if [ "$dst" -lt "$src" ]; then
            echo "kept      $dst of $src entries -- $((src - dst)) could not be" \
                 "recreated on this filesystem (sockets, usually). MANIFEST.txt lists" \
                 "everything that was there." >&2
        fi
        echo "kept      $(du -sh "$KEEP_DIR" 2>/dev/null | cut -f1) in $KEEP_DIR"
        # `rm -rf` on a computed path gets a guard: only a path under a tmp base is ever
        # removed, never something that resolved somewhere unexpected.
        case "$S0_SCRATCH" in
            /tmp/*|"${TMPDIR:-/nonexistent}"/*)
                rm -rf "$S0_SCRATCH" 2>/dev/null || true ;;
            *) echo "NOT removing $S0_SCRATCH -- it is not under a tmp base" >&2 ;;
        esac
    }
    trap keep_scratch EXIT
fi

# The trace directory has to be somewhere that OUTLIVES the calcspace, and on a job that
# means inside the node-local tree that gets carried back -- not in the calcspace, and not
# on Lustre. Set after the environment files have run, because that is what defines
# S0_SCRATCH.
if [ -n "${MACE_TRACE:-}" ]; then
    # A value containing a slash is taken as the directory itself. That is for a run whose
    # work is on the shared filesystem (examples/02a.../branchA-fs.conf): there the trace
    # should be written where it already survives, rather than node-local and dependent on
    # the end-of-job copy-back. Anything else (conventionally `1`) goes node-local.
    case "$MACE_TRACE" in
        */*) S0_MACE_TRACE="$MACE_TRACE" ;;
        *)   S0_MACE_TRACE="${S0_SCRATCH:-$ROOT/logs}/trace" ;;
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
BASINS_PRESENT=$(python - "$SPECIES" "$TAG" <<'PY'
import sys
sys.path.insert(0, ".")
try:
    from openqha.store import basin_store
    print("yes" if basin_store.exists(sys.argv[1], tag=sys.argv[2]) else "no")
except Exception:                                                 # noqa: BLE001
    print("unknown")
PY
)

if [ "$CHAIN" = "conformers" ]; then
    echo
    echo "---- branch A: conformer search -> every basin ------------------------"
    [ "$BASINS_PRESENT" = "yes" ] && \
        echo "     (a record already exists for tag '$TAG'; re-running it -- this
     chain's product IS branch A)"
    # shellcheck disable=SC2086  -- BRANCH_A_ARGS is a deliberate word list, empty unless
    # the conf turned a diagnostic on
    python -u scripts/production/s0_A_pipeline.py \
        --species "$SPECIES" --tag "$TAG" --threads "$THREADS" \
        --hessian-mode analytic $BRANCH_A_ARGS
elif [ "$BASINS_PRESENT" = "yes" ]; then
    echo
    echo "---- branch A: already done for tag '$TAG' -- reusing those basins ----"
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
    python -u scripts/production/s0_A_pipeline.py \
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
    python -u scripts/production/s0_E_branchB_parsl.py \
        --species "$SPECIES" --tag "$TAG" --resource "$RESOURCE" \
        --route "$ROUTE" --basins auto --seeds "$SEEDS" \
        ${PROD_PS:+--prod-ps "$PROD_PS"} ${EQUIL_PS:+--equil-ps "$EQUIL_PS"}

    echo
    echo "---- collect: quasi-harmonic analysis per molecule --------------------"
    python -u scripts/production/s0_E_branchB_collect_parsl.py \
        --species "$SPECIES" --tag "$TAG" --resource "${COLLECT_RESOURCE:-$RESOURCE}"

    echo
    echo "---- the answer: F_conf over the ensemble -----------------------------"
    python -u scripts/production/s0_B_report_ensemble.py --species "$SPECIES" --tag "$TAG"
    ;;

levels) # 02c: MACE vs GFN2-xTB vs RI-MP2, energies and forces through to G - E_el
    # CPU only. The cost is entirely the RI-MP2 `TightOpt NumFreq`: 1613 s per 10-atom
    # structure at 4 processes, one ORCA job per basin, in sequence.
    echo
    echo "---- 02c: three levels on every basin ---------------------------------"
    python -u examples/02c_hessian_benchmark_levels/s0_level_benchmark.py \
        --species "$SPECIES" --tag "$TAG" \
        --levels "${LEVELS:-mace,gfn2,rimp2}" --nprocs "${ORCA_NPROCS:-$THREADS}" \
        ${ORCA_MAXCORE:+--maxcore "$ORCA_MAXCORE"}
    ;;

identity) # 02d: may nu_k replace omega_i in ZPE, enthalpy and entropy?
    # The trajectories are DENSELY sampled on purpose: the protocol's 1.0 ps interval is
    # chosen for the ENTROPY, while the zero-point energy needs of order 12 500 frames,
    # which at 1.0 ps would be 12.5 ns. Both numbers are measured in 02d stage 1.
    echo
    echo "---- branch B: dense trajectories, ${PROD_PS:-protocol} ps ------------"
    python -u scripts/production/s0_E_branchB_parsl.py \
        --species "$SPECIES" --tag "$TAG" --resource "$RESOURCE" \
        --route "$ROUTE" --basins auto --seeds "$SEEDS" \
        ${PROD_PS:+--prod-ps "$PROD_PS"} ${EQUIL_PS:+--equil-ps "$EQUIL_PS"} \
        ${SAMPLE_EVERY:+--sample-every "$SAMPLE_EVERY"}

    echo
    echo "---- 02d: G_total from omega and from nu, term by term ----------------"
    python -u examples/02d_qha_frequency_identity/s0_frequency_identity.py \
        --species "$SPECIES" --tag "$TAG" --stage all --traj-tag "$TAG" \
        ${NU_CUT:+--nu-cut "$NU_CUT"}
    ;;

*)  echo "CHAIN must be conformers, qha, levels or identity, got '$CHAIN'" >&2
    exit 2 ;;
esac

echo
echo "chain '$CHAIN' finished for $SPECIES (tag $TAG)."
