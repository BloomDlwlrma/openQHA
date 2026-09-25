#!/bin/bash
# =====================================================================================
# openQHA: rerun the draw300 molecules PARKED by the old imaginary-frequency filter.
#
# AUTHORITY  Ticket 41, "Rulings of 2026-09-25", bullet Q5 (the imaginary-mode regime,
#            tickets 34-41). The draw300 branch-A runs that died with "no basin
#            survives the tightening and the imaginary-frequency filter" are rerun
#            under the frequency-floor census screen (tickets 37/38, ADR 0007). A
#            molecule whose candidates carry inversion windows now FINISHES -- basins
#            with N_INVERSION_WINDOW / LOWEST_FREQ per row in branchA.toml, the
#            marker cleared (s0_A_pipeline.py does that on success). A molecule with
#            no admissible basin REFUSES with the new message, which names ithr and
#            lists every condemned candidate's below-floor count and lowest frequency
#            -- that alone is a result to record in ticket 41.
#
# HOW IT RUNS  There is no ssh from the workstation. This file is copied to the Tianhe
#            checkout and used BY HAND there. One file, THREE ROLES:
#
#   1. LAUNCHER -- run on the login node, from the checkout root:
#        cd ~/openQHA-main
#        bash .scratch/msrrho-gtotal/q5-rerun-parked.sh
#      Checks the checkout carries the fix, finds the parked set (the Q5 grep:
#      "no basin survives" under $S0_RUNS_ROOT/draw300), shows it, and submits ONE
#      'deimos' job per molecule (24 h, exclusive node -- hl_branchA.slurm's shape).
#      Nothing heavy runs on the login node. YES=1 skips the prompt; DRY_RUN=1 prints
#      without doing (launcher: the submissions; worker: the resolved command).
#
#   2. WORKER -- the per-molecule job the launcher submits (or by hand:
#        sbatch --export=HOME="$HOME",QID=<qid> .scratch/msrrho-gtotal/q5-rerun-parked.sh,
#      and inside an allocation: QID=<qid> bash .scratch/msrrho-gtotal/q5-rerun-parked.sh).
#      Reruns ONE molecule with the same command hl_branchA_worker.sh runs, under the
#      campaign environment (module purge, the NARROW unset, common.sh -> tianhe.sh,
#      crest/xtb/python required), then reads the outcome FROM DISK and prints a
#      paste-back block.
#
#   3. STATUS -- one line per molecule, now:
#        bash .scratch/msrrho-gtotal/q5-rerun-parked.sh --status
#
# THE REUSE RULE (REUSE=1, the default)
#      The parked runs died AT THE SCREEN, after CREST, so their draw300 ensembles
#      are on disk. The worker then runs --skip-crest on THAT ensemble: the same
#      candidates go through the new screen -- the decisive verification ("whose
#      candidates carried window modes", Q5's words) -- and the cost is tighten +
#      Hessian, not a fresh search. When the on-disk ensemble is ambiguous (a crest/
#      and a crest_shake*/ both hold one) or absent, the worker says so and reruns
#      CREST IN FULL. REUSE=0 forces the full, campaign-exact rerun (the driver is
#      given --force-crest; without it `run_crest` would silently reuse the matching
#      `crest/` directory) -- note CREST is stochastic, so a fresh sample need not
#      reproduce the old candidates.
#
# THE MARKER  The pre-rerun `_records/branchA.failed` is MOVED to $BACKUP_DIR before
#      the run, so a fresh refusal writes its own marker (mark_crashed never clobbers)
#      and a success clears what is there. The old text stays readable in the backup.
#      An attempt that leaves NO fresh record and NO marker -- the WALL_S soft ceiling
#      (default 23 h of the 24 h job), a CREST timeout (which writes no marker by
#      design, ticket 26), anything else dying early -- restores the old marker:
#      nothing was burned, the molecule stays parked, resubmit is the whole retry.
#
# SUBMISSION  The launcher submits with an explicit `--export` list, never
#      `--export=ALL` -- and since sbatch's default IS all of the submitting shell's
#      environment, the explicit list must be given. Only the job's own variables
#      propagate (HOME included); the worker builds everything else from the checkout.
#      See openQHA/AGENTS.md, "Submitting jobs" (user ruling, 2026-09-25).
#
# EXPECTED SET  Typically five molecules; dsgdb9nsd_052993 is named in the repo as one
#            (ticket 38's -6.84 cm^-1 case; t_branch_a_crash section D). A different
#            count is not an error -- read the launcher's table before confirming.
#            After the jobs: paste each worker block into ticket 41 and record the
#            outcome; the five going forward into the frames/labels stages is the
#            user's call.
#
# OVERRIDABLE (environment): TAG (draw300) QIDS PARTITION (deimos) TIME_LIMIT
#            THREADS (4) TIMEOUT_S (3600) WALL_S (82800; 0 disables) REUSE (1)
#            CHECKOUT BACKUP_DIR ($HOME/openqha_q5_backup) YES DRY_RUN FORCE
#            BIND_CORES ("none" disables the pin; default 0..THREADS-1), and QID for
#            the worker.
#
# EXIT CODES (worker): 0 FINISHED with all criteria passed, or already done;
#            1 FINISHED with a criterion FAILED, or REFUSED (a result to record);
#            2 CRASHED; 3 TIMEOUT (the soft ceiling); 4 refused before any work
#            (environment / checkout / missing QID / molecule not found).
# =====================================================================================
#SBATCH --job-name=openqha_q5
#SBATCH --partition=deimos
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

# no `set -e` (project rule 2026-09-13); no `set -u` (the conda hooks, common.sh)

# ---- 0. knobs ------------------------------------------------------------------------
TAG="${TAG:-draw300}"
PARTITION="${PARTITION:-deimos}"
TIME_LIMIT="${TIME_LIMIT:-1-00:00:00}"
THREADS="${THREADS:-4}"
TIMEOUT_S="${TIMEOUT_S:-3600}"
WALL_S="${WALL_S:-82800}"
REUSE="${REUSE:-1}"
CHECKOUT="${CHECKOUT:-}"
BACKUP_DIR="${BACKUP_DIR:-$HOME/openqha_q5_backup}"
QIDS="${QIDS:-}"
YES="${YES:-0}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
FIX_COMMIT="ea21bb1bd595f3afd2be4eeeb218995215dfab18"     # Ticket 41, 2026-09-25

SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"

say()  { printf '%s\n' "$*"; }
warn() { printf 'OPENQHA-Q5: WARNING: %s\n' "$*" >&2; }
die()  { printf 'OPENQHA-Q5: %s\n' "$1" >&2; exit "${2:-2}"; }

usage() {
    cat <<'EOF'
q5-rerun-parked.sh -- ticket 41 Q5: rerun the draw300 molecules parked by the old
imaginary-frequency filter, under the frequency-floor census screen.

  bash q5-rerun-parked.sh            launcher: find, show, submit (run from the checkout)
  bash q5-rerun-parked.sh --status   one line per molecule, now
  sbatch --export=HOME="$HOME",QID=<qid> <this file>   worker: rerun that one molecule

Knobs (environment): TAG QIDS PARTITION TIME_LIMIT THREADS TIMEOUT_S WALL_S REUSE
CHECKOUT BACKUP_DIR YES DRY_RUN FORCE BIND_CORES, and QID for the worker.
Read the header of this file for the marker rules and the exit codes.
EOF
}

# ---- 1. which role is this -----------------------------------------------------------
ROLE=launcher
case "${1:-}" in
    --status) ROLE=status ;;
    -h|--help) usage; exit 0 ;;
    "") ;;
    *) usage >&2; die "unknown argument: $1" 2 ;;
esac
if [ "$ROLE" = launcher ] && [ -n "${SLURM_JOB_ID:-}" ]; then ROLE=worker; fi

# ---- 2. shared helpers ---------------------------------------------------------------
check_checkout() {
    # Ticket 41's precondition: nothing is submitted before the change is on the
    # checkout. The commit check is exact; the file check is for a clone without that
    # object (a shallow or re-created tree) and pins the census screen by content.
    if [ -d .git ] && command -v git >/dev/null 2>&1; then
        if git cat-file -e "$FIX_COMMIT^{commit}" 2>/dev/null; then
            if git merge-base --is-ancestor "$FIX_COMMIT" HEAD 2>/dev/null; then
                say "OPENQHA-Q5: checkout carries ticket 41 (HEAD $(git rev-parse --short HEAD 2>/dev/null))"
                return 0
            fi
            say "OPENQHA-Q5: REFUSED -- this checkout is BEHIND $FIX_COMMIT (ticket 41)."
            say "  sync it (git pull; origin/main carries 'Ticket 41: the package-1 batch screen') and resubmit."
            return 2
        fi
        say "OPENQHA-Q5: note: commit $FIX_COMMIT is not in this clone; checking file contents instead"
    fi
    if grep -qF 'the frequency-floor screen' openqha/conformer_search/crest_census.py 2>/dev/null \
       && grep -qF 'converged_orca_default' openqha/conformer_search/crest_census.py 2>/dev/null \
       && grep -qF 'n_inversion_window' openqha/conformer_search/crest_census.py 2>/dev/null; then
        say "OPENQHA-Q5: checkout carries the frequency-floor census screen (content check)"
        return 0
    fi
    say "OPENQHA-Q5: REFUSED -- openqha/conformer_search/crest_census.py is not the floor-screen version."
    say "  sync the checkout (git pull) and resubmit."
    return 2
}

resolve_root() {
    # root.sh only defines functions (openqha_resolve_root, openqha_count_basins), so
    # it is sourced even when S0_RUNS_ROOT is already exported -- is_done needs its
    # counter. An explicit S0_RUNS_ROOT wins; otherwise the root is derived from the
    # partition / the mounted prefix (ADR 0002). Safe to source anywhere.
    [ -f hpc/env/root.sh ] || { say "OPENQHA-Q5: no hpc/env/root.sh -- run this from the checkout root"; return 3; }
    # shellcheck source=/dev/null
    source hpc/env/root.sh
    [ -n "$S0_RUNS_ROOT" ] && return 0
    openqha_resolve_root || {
        say "OPENQHA-Q5: the root cannot be derived here."
        say "  export S0_RUNS_ROOT=/XYFS02/HDD_POOL/<acct>/<user>/sherwin/runs (or set OPENQHA_PARTITION)"
        return 3
    }
    return 0
}

list_markers() {
    # The Q5 grep, authoritative when the draw300 tree is flat; a find-based fallback
    # covers a nested layout without changing the rule. Prints marker paths, one per line.
    local found=""
    found="$(grep -rl -- "no basin survives" "$S0_RUNS_ROOT/$TAG"/*/_records/branchA.failed 2>/dev/null || true)"
    if [ -z "$found" ]; then
        found="$(find "$S0_RUNS_ROOT/$TAG" -path '*/_records/branchA.failed' -type f \
                 -exec grep -l -- "no basin survives" {} + 2>/dev/null || true)"
    fi
    [ -n "$found" ] && printf '%s\n' "$found"
    return 0
}

qid_of_marker() { basename "$(dirname "$(dirname "$1")")"; }
mol_of_marker() { dirname "$(dirname "$1")"; }

record_fields() {
    # The branchA.toml fields both the worker's verdict and --status print, read with
    # the campaign worker's own greps (uppercase Property-file keys).
    local rec="$1"
    RB_ALL="$(grep -m1 '^ALL_PASSED' "$rec" | awk '{print $3}')"
    RB_NBASINS="$(grep -m1 '^N_BASINS' "$rec" | awk '{print $3}')"
    RB_LOWS="$(grep '^LOWEST_FREQ' "$rec" | awk '{printf "%.2f ", $3}')"
    RB_WINS="$(grep '^N_INVERSION_WINDOW' "$rec" | awk '{printf "%s ", $3}')"
    RB_WINTOT="$(grep '^N_INVERSION_WINDOW' "$rec" | awk '{s += $3} END {print s + 0}')"
}

find_molecule_dir() {
    # <qid> -> molecule directory on stdout; rc 1 when not found. The marker path is
    # tried first (it is where the parked molecule's evidence is), then the directory
    # by name (a molecule whose marker was moved away outside this script).
    local qid="$1" hit=""
    hit="$(find "$S0_RUNS_ROOT/$TAG" -path "*/$qid/_records/branchA.failed" -type f -print -quit 2>/dev/null)"
    if [ -n "$hit" ]; then mol_of_marker "$hit"; return 0; fi
    hit="$(find "$S0_RUNS_ROOT/$TAG" -type d -name "$qid" -print -quit 2>/dev/null)"
    if [ -n "$hit" ] && { [ -d "$hit/_records" ] || [ -d "$hit/crest" ] || [ -d "$hit/mace" ]; }; then
        printf '%s\n' "$hit"; return 0
    fi
    return 1
}

is_done() {
    # Branch A's own rule (openqha.store.basins.done): STATUS NORMAL TERMINATION and
    # at least one basinNN/basin.extxyz; openqha_count_basins comes from hpc/env/root.sh.
    local mol="$1" rec="$1/_records/branchA.toml"
    [ -f "$rec" ] || return 1
    grep -q 'NORMAL TERMINATION' "$rec" || return 1
    [ "$(openqha_count_basins "$mol")" -gt 0 ]
}

confirm() {
    local r=""
    printf '  ? %s [y/N] ' "$1"
    read -r r || true
    case "$r" in [Yy]*) return 0 ;; *) return 1 ;; esac
}

# ---- 3. the roles --------------------------------------------------------------------
case "$ROLE" in

launcher)
    if [ -n "$CHECKOUT" ]; then cd "$CHECKOUT" || die "no checkout at $CHECKOUT" 2; fi
    say "OPENQHA-Q5: launcher -- the draw300 parked molecules (ticket 41, Q5)"
    say "OPENQHA-Q5: checkout  $PWD"
    check_checkout || exit 2
    resolve_root || exit 2
    [ -d "$S0_RUNS_ROOT/$TAG" ] || die "no tag tree at $S0_RUNS_ROOT/$TAG -- wrong TAG or root?" 2
    say "OPENQHA-Q5: root $S0_RUNS_ROOT   tag $TAG"

    PARKED=(); STALE=()
    say ""
    say "the parked set:"
    if [ -n "$QIDS" ]; then
        # QIDS="a b c" names the set by hand; word splitting here is intended.
        # shellcheck disable=SC2206
        PARKED=($QIDS)
        say "  (from QIDS, not from the markers)"
    else
        while IFS= read -r m; do
            [ -n "$m" ] || continue
            qid="$(qid_of_marker "$m")"
            mol="$(mol_of_marker "$m")"
            if is_done "$mol"; then STALE+=("$qid"); continue; fi
            PARKED+=("$qid")
            say "  $qid"
            say "      dir    $mol"
            say "      marker $(grep -m1 'no basin survives' "$m" 2>/dev/null | cut -c1-110)"
        done < <(list_markers)
        if [ "${#PARKED[@]}" -eq 0 ] && [ "${#STALE[@]}" -eq 0 ]; then
            say "  (none)"
            say ""
            say "OPENQHA-Q5: nothing parked under $S0_RUNS_ROOT/$TAG."
            say "  Check TAG and the root; the Q5 grep is:"
            say "    grep -rl \"no basin survives\" \"\$S0_RUNS_ROOT/$TAG\"/*/_records/branchA.failed"
            exit 1
        fi
        if [ "${#STALE[@]}" -gt 0 ]; then
            say ""
            say "already done (stale marker, left alone): ${STALE[*]}"
        fi
    fi
    [ "${#PARKED[@]}" -gt 0 ] || { say "nothing to submit."; exit 1; }

    mkdir -p "$BACKUP_DIR" || die "cannot create $BACKUP_DIR" 2
    LIST_FILE="$BACKUP_DIR/$TAG.parked.txt"
    printf '%s\n' "${PARKED[@]}" > "$LIST_FILE"
    say ""
    say "OPENQHA-Q5: ${#PARKED[@]} molecule(s); list written to $LIST_FILE"

    say ""
    say "submission plan: one '$PARTITION' job per molecule, $TIME_LIMIT, exclusive node;"
    say "  python -u scripts/production/s0_A_pipeline.py --species <qid> --tag $TAG --threads $THREADS --timeout-s $TIMEOUT_S --hessian-mode analytic"
    if [ "$REUSE" = 1 ]; then
        say "  plus --skip-crest on the on-disk draw300 ensemble when it is unambiguous (REUSE=1)"
    else
        say "  a full CREST rerun (REUSE=0, with --force-crest; CREST is stochastic, so this need not reproduce the old candidates)"
    fi

    if [ "$DRY_RUN" = 1 ]; then
        say ""
        say "DRY_RUN=1 -- nothing was submitted; the launcher would run, per qid:"
        for qid in "${PARKED[@]}"; do
            say "  sbatch --job-name=q5_$qid --partition=$PARTITION --time=$TIME_LIMIT \\"
            say "      --export=HOME=$HOME,QID=$qid,TAG=$TAG,REUSE=$REUSE,THREADS=$THREADS,TIMEOUT_S=$TIMEOUT_S,WALL_S=$WALL_S,BACKUP_DIR=$BACKUP_DIR,CHECKOUT=$PWD \\"
            say "      $SELF"
        done
        exit 0
    fi

    if [ ! -t 0 ] && [ "$YES" != 1 ]; then
        die "stdin is not a terminal; rerun interactively, or set YES=1" 2
    fi
    if [ "$YES" != 1 ]; then
        confirm "submit ${#PARKED[@]} job(s) to $PARTITION?" || { say "nothing submitted."; exit 0; }
    fi

    mkdir -p logs/slurm || die "cannot create logs/slurm -- run from the checkout root" 2
    say ""
    # Submit with an EXPLICIT --export list, never --export=ALL (AGENTS.md, "Submitting
    # jobs"): only the job's own variables travel; the worker builds its environment
    # from the checkout. sbatch's default would export the whole submitting environment.
    for qid in "${PARKED[@]}"; do
        if [ "$FORCE" != 1 ] && command -v squeue >/dev/null 2>&1 \
           && squeue -h -u "$USER" -o '%j' 2>/dev/null | grep -qx "q5_$qid"; then
            warn "q5_$qid is already queued or running; skipped (FORCE=1 submits anyway)"
            continue
        fi
        out="$(sbatch --job-name="q5_$qid" --partition="$PARTITION" --time="$TIME_LIMIT" \
              --export=HOME="$HOME",QID="$qid",TAG="$TAG",REUSE="$REUSE",THREADS="$THREADS",TIMEOUT_S="$TIMEOUT_S",WALL_S="$WALL_S",BACKUP_DIR="$BACKUP_DIR",CHECKOUT="$PWD" \
              "$SELF" 2>&1)"
        if [ "$?" != 0 ]; then warn "sbatch failed for $qid: $out"; continue; fi
        jid="${out##* }"
        say "  $qid  ->  job $jid   (log logs/slurm/q5_${qid}_${jid}.out)"
    done
    say ""
    say "when the jobs finish:  bash $SELF --status"
    say "then paste each worker block (the 'paste this back' section of its log) into ticket 41."
    exit 0
;;

worker)
    [ -n "${QID:-}" ] || die "the worker role needs QID; submit through the launcher, or: sbatch --export=HOME=\"\$HOME\",QID=<qid> $SELF" 4
    cd "${CHECKOUT:-${SLURM_SUBMIT_DIR:-$PWD}}" || die "no checkout directory" 4
    say "OPENQHA-Q5: worker -- $QID   tag $TAG   job ${SLURM_JOB_ID:-interactive} on $(hostname) partition ${SLURM_JOB_PARTITION:-?}"
    say "OPENQHA-Q5: checkout  $PWD"
    check_checkout || exit 4

    # ---- the campaign environment, exactly (hl_branchA.slurm / verify-orca-one.sh) --
    module purge 2>/dev/null || true
    module load anaconda3/202309 2>/dev/null || module load anaconda3/2023.09 2>/dev/null || true
    # The NARROW unset (hpc/slurm/README.md rule 2): scheduling variables for the
    # non-ORCA payloads. CREST forks its own workers and is exposed to exactly this.
    for v in $(env | awk -F= '{print $1}' | grep -E '^(PMI|SLURM_(CPU|TASK|NTASKS|NPROCS|STEP))'); do
        unset "$v"
    done
    export OPENQHA_PARTITION="${SLURM_JOB_PARTITION:-$PARTITION}"
    # shellcheck source=/dev/null
    source hpc/env/common.sh
    # shellcheck source=/dev/null
    source hpc/env/tianhe.sh || die "hpc/env/tianhe.sh refused (conda env / root unresolved)" 4
    openqha_report_env
    openqha_require crest xtb python || die "the environment is missing crest, xtb or python" 4
    python scripts/tooling/s0_check_weights.py 2>/dev/null | grep -E "registry pin" \
        || warn "the weights check did not print a registry pin"
    resolve_root || exit 4

    MOL="$(find_molecule_dir "$QID")" || true
    [ -n "$MOL" ] || die "no molecule directory for $QID under $S0_RUNS_ROOT/$TAG" 4
    REC="$MOL/_records/branchA.toml"
    MARKER="$MOL/_records/branchA.failed"
    say "OPENQHA-Q5: molecule  $MOL"

    if is_done "$MOL"; then
        say "OPENQHA-Q5: branch A already reads done for this molecule -- nothing to rerun."
        say "           basins $(openqha_count_basins "$MOL"); record $REC"
        exit 0
    fi
    if [ -f "$MARKER" ]; then
        say "OPENQHA-Q5: parked marker present: $MARKER"
    else
        warn "no branchA.failed on disk; rerunning anyway (the launcher's list said parked)"
    fi

    # ---- move the old marker aside: mark_crashed never clobbers an existing marker,
    # so a fresh refusal must find the name free, and a success clears what it finds.
    if [ -f "$MARKER" ]; then
        mkdir -p "$BACKUP_DIR" || die "cannot create $BACKUP_DIR" 4
        mv "$MARKER" "$BACKUP_DIR/${QID}.branchA.failed.old" \
            && say "OPENQHA-Q5: old marker -> $BACKUP_DIR/${QID}.branchA.failed.old"
    fi

    # ---- which ensemble did the draw300 run actually finish with? -------------------
    ENS_DIR=""
    if [ "$REUSE" != 1 ]; then
        ENS_NOTE="REUSE=0 -- full CREST rerun"
    elif [ -f "$MOL/crest/crest_conformers.xyz" ]; then
        if ls "$MOL"/crest_shake*/crest_conformers.xyz >/dev/null 2>&1; then
            ENS_NOTE="crest/ and a crest_shake*/ both hold an ensemble; ambiguous -- full rerun"
        else
            ENS_DIR="$MOL/crest"
            ENS_NOTE="the draw300 CREST work directory"
        fi
    else
        SHAKES=()
        for f in "$MOL"/crest_shake*/crest_conformers.xyz; do
            [ -f "$f" ] || continue
            SHAKES+=("$(dirname "$f")")
        done
        if [ "${#SHAKES[@]}" -gt 1 ]; then
            ENS_NOTE="several crest_shake*/ directories hold an ensemble; ambiguous -- full rerun"
        elif [ "${#SHAKES[@]}" -eq 1 ]; then
            ENS_DIR="${SHAKES[0]}"
            ENS_NOTE="a CREST SHAKE-fallback work directory"
        else
            ENS_NOTE="no crest_conformers.xyz on disk; full rerun is the only option"
        fi
    fi
    say "OPENQHA-Q5: reuse: $ENS_NOTE"
    [ -n "$ENS_DIR" ] && say "OPENQHA-Q5: ensemble $ENS_DIR/crest_conformers.xyz (mtime $(stat -c '%y' "$ENS_DIR/crest_conformers.xyz" 2>/dev/null | cut -c1-19))"

    # ---- the driver command: the same one hl_branchA_worker.sh runs -----------------
    cmd=(python -u scripts/production/s0_A_pipeline.py --species "$QID" --tag "$TAG"
         --threads "$THREADS" --timeout-s "$TIMEOUT_S" --hessian-mode analytic)
    if [ -n "$ENS_DIR" ]; then
        cmd+=(--skip-crest --reuse "$ENS_DIR")
    else
        # A full rerun must really rerun CREST: with the parked molecule's matching
        # `crest/` on disk, `run_crest` would otherwise silently reuse it (its
        # matching-scratch branch) and "full" would be a lie. --force-crest is the
        # driver's own switch for exactly this.
        cmd+=(--force-crest)
    fi
    say "OPENQHA-Q5: command ${cmd[*]}"

    if [ "$DRY_RUN" = 1 ]; then
        say "OPENQHA-Q5: DRY_RUN=1 -- everything resolved, nothing was run."
        exit 0
    fi

    BIND="${BIND_CORES:-0-$(( THREADS - 1 ))}"
    [ "$BIND" = none ] && BIND=""
    runner=()
    if [ -n "$WALL_S" ] && [ "$WALL_S" != 0 ] && command -v timeout >/dev/null 2>&1; then
        runner=(timeout -k 60 "$WALL_S")
    fi
    LOG="${S0_SCRATCH:-/tmp}/branchA_${QID}.log"
    T0=$(date +%s)
    say "OPENQHA-Q5: running (soft ceiling ${WALL_S}s, cores ${BIND:-unpinned}); driver log $LOG"
    if command -v taskset >/dev/null 2>&1 && [ -n "$BIND" ]; then
        "${runner[@]}" taskset -c "$BIND" "${cmd[@]}" > "$LOG" 2>&1
    else
        "${runner[@]}" "${cmd[@]}" > "$LOG" 2>&1
    fi
    rc=$?

    # ---- the verdict, read from the disk (the campaign worker's own greps) ----------
    newer() { [ -f "$1" ] && [ "$(stat -c '%Y' "$1" 2>/dev/null || echo 0)" -ge "$T0" ]; }
    EXIT=0
    if [ -f "$REC" ] && newer "$REC"; then
        STATUS_LINE="$(grep -m1 '^STATUS' "$REC" | cut -d'"' -f2)"
        record_fields "$REC"
        VERDICT="FINISHED ($STATUS_LINE)"
        [ "$RB_ALL" = true ] || EXIT=1
        say "OPENQHA-Q5: basins $RB_NBASINS   lowest cm^-1 [ $RB_LOWS]   window modes [ $RB_WINS] (total $RB_WINTOT)"
        say "OPENQHA-Q5: criteria ALL_PASSED=$RB_ALL   ($(grep -m1 'all criteria passed' "$LOG" 2>/dev/null || echo 'no criteria line in the log'))"
    elif [ -f "$MARKER" ]; then
        if grep -q 'no basin survives' "$MARKER"; then
            VERDICT="REFUSED -- the floor screen condemned every candidate"; EXIT=1
        else
            VERDICT="CRASHED -- the pipeline raised"; EXIT=2
        fi
        say ""
        say "== the marker this attempt left =="
        sed -n '1,10p' "$MARKER"
        say "== from the driver log =="
        blk="$(grep -m1 -B1 -A30 'no basin survives' "$LOG" 2>/dev/null)"
        if [ -n "$blk" ]; then printf '%s\n' "$blk" | sed -n '1,34p'; else tail -n 25 "$LOG" 2>/dev/null; fi
    elif [ "$rc" = 124 ]; then
        VERDICT="TIMEOUT -- the ${WALL_S}s soft ceiling cut the molecule loose"; EXIT=3
        tail -n 15 "$LOG" 2>/dev/null
    elif [ "$rc" != 0 ]; then
        VERDICT="CRASHED -- exit $rc with no marker written"; EXIT=2
        tail -n 25 "$LOG" 2>/dev/null
    else
        VERDICT="UNCLEAR -- exit 0 but no fresh record; read the log"; EXIT=2
        tail -n 15 "$LOG" 2>/dev/null
    fi

    # An attempt that left neither a fresh record nor a marker (the soft ceiling at
    # rc=124; a CREST timeout, which writes no marker by design; anything else dying
    # early) RESTORES the old marker: the molecule stays parked and discoverable, so
    # a resubmission is the whole retry.
    if ! newer "$REC" && [ ! -f "$MARKER" ] \
       && [ -f "$BACKUP_DIR/${QID}.branchA.failed.old" ]; then
        cp -p "$BACKUP_DIR/${QID}.branchA.failed.old" "$MARKER" \
            && say "OPENQHA-Q5: old marker RESTORED -- nothing was burned, the molecule stays parked"
    fi
    if [ -f "$LOG" ]; then
        mkdir -p "$BACKUP_DIR" && cp -f "$LOG" "$BACKUP_DIR/${QID}.branchA.last.log" 2>/dev/null
    fi

    say ""
    say "================= paste this back ================="
    say "job        ${SLURM_JOB_ID:-?} on $(hostname) partition ${SLURM_JOB_PARTITION:-?}"
    say "checkout   $PWD"
    say "molecule   $QID   tag $TAG   dir $MOL"
    say "reuse      $ENS_NOTE"
    say "backup     $BACKUP_DIR/${QID}.branchA.failed.old"
    say "driver log $BACKUP_DIR/${QID}.branchA.last.log  (also $LOG)"
    say "record     $REC"
    say "exit code  $rc"
    say "VERDICT:   $VERDICT"
    exit "$EXIT"
;;

status)
    if [ -n "$CHECKOUT" ]; then cd "$CHECKOUT" || die "no checkout at $CHECKOUT" 2; fi
    check_checkout || exit 2
    resolve_root || exit 2
    LIST_FILE="$BACKUP_DIR/$TAG.parked.txt"
    QLIST=()
    SRC=""
    if [ -n "$QIDS" ]; then
        # shellcheck disable=SC2206
        QLIST=($QIDS); SRC="QIDS"
    elif [ -f "$LIST_FILE" ]; then
        while IFS= read -r l; do [ -n "$l" ] && QLIST+=("$l"); done < "$LIST_FILE"
        SRC="$LIST_FILE"
    else
        while IFS= read -r m; do [ -n "$m" ] || continue; QLIST+=("$(qid_of_marker "$m")"); done < <(list_markers)
        SRC="the markers under $S0_RUNS_ROOT/$TAG"
    fi
    say "OPENQHA-Q5: status of ${#QLIST[@]} molecule(s)   (list: $SRC)"
    ndone=0; npark=0; nunk=0
    for qid in "${QLIST[@]}"; do
        MOL="$(find_molecule_dir "$qid")" || true
        if [ -z "$MOL" ]; then
            nunk=$((nunk + 1))
            say "  $qid  UNKNOWN  no directory under $S0_RUNS_ROOT/$TAG"
            continue
        fi
        REC="$MOL/_records/branchA.toml"
        MARKER="$MOL/_records/branchA.failed"
        if is_done "$MOL"; then
            record_fields "$REC"
            say "  $qid  DONE     basins=$RB_NBASINS  criteria ALL_PASSED=$RB_ALL"
            say "             lowest cm^-1 [ $RB_LOWS]  window modes [ $RB_WINS]"
            ndone=$((ndone + 1))
        elif [ -f "$MARKER" ]; then
            if grep -q 'frequency-floor screen' "$MARKER"; then
                kind="refused under the NEW floor screen (this rerun)"
            else
                kind="pre-fix wording (the rerun has not finished yet)"
            fi
            say "  $qid  PARKED   [$kind]"
            say "             $(sed -n '1p' "$MARKER" | cut -c1-110)"
            npark=$((npark + 1))
        else
            sq=""
            if command -v squeue >/dev/null 2>&1; then
                sq="$(squeue -h -u "$USER" -o '%j %T %M' 2>/dev/null | awk -v n="q5_$qid" '$1 == n {print "squeue: " $2 " " $3}')"
            fi
            say "  $qid  UNKNOWN  no marker, no record   ${sq}"
            nunk=$((nunk + 1))
        fi
    done
    say "OPENQHA-Q5: $ndone done, $npark parked, $nunk unknown"
    exit 0
;;

esac
