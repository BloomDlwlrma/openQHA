# hpc/env/root.sh -- the root of the molecule tree on Tianhe (ADR 0002, user 2026-09-14).
#
# Sourced by hpc/env/tianhe.sh; safe to source anywhere because it only DEFINES a
# function. `openqha_resolve_root` exports S0_RUNS_ROOT or returns 1 with the reason on
# stderr; it never falls back to $HOME (the 2026-09-12 defect was a default nobody chose).
#
#     root = <prefix>/HDD_POOL/<acct>/<user>/<S0_RUNS_TAIL>
#     prefix   /XYFS02   for partitions ai, temp (TianheXY-A) and cn, deimos, debug
#                        (TianheXY-CN; deimos measured 2026-09-13: branch A on cnode5200
#                        wrote under /XYFS02/HDD_POOL, the 02d_prod basins record)
#              /XYAIFS00 for partitions a100x h100x hx a800x v100x  (TianheXY-AI)
#     acct, user  from HOME = /HOME/<acct>/<user>  (or /XYAIFS00/HOME/<acct>/<user>)
#     tail     S0_RUNS_TAIL, default sherwin/runs -- ONE variable, so another account
#              changes one line.
#
# Partition: OPENQHA_PARTITION, then SLURM_JOB_PARTITION. On a login node with neither,
# the prefix that is mounted decides; when both or neither are mounted that is an error,
# and the message says which variable to set. An explicit S0_RUNS_ROOT always wins.
#
# S0_MOUNT_BASE (tests only) is prepended to the mount probe so a test can fake a mounted
# filesystem inside a temporary directory.

openqha_resolve_root() {
    local part prefix acct user tail have_a have_ai

    if [ -n "$S0_RUNS_ROOT" ] && [ -z "$S0_RUNS_ROOT_IS_DEFAULT" ]; then
        export S0_RUNS_ROOT
        return 0
    fi

    part="${OPENQHA_PARTITION:-${SLURM_JOB_PARTITION:-}}"
    case "$part" in
        ai|temp|cn|deimos|debug)   prefix=/XYFS02 ;;
        a100x|h100x|hx|a800x|v100x) prefix=/XYAIFS00 ;;
        "")
            have_a=0;  [ -d "${S0_MOUNT_BASE:-}/XYFS02/HDD_POOL" ]   && have_a=1
            have_ai=0; [ -d "${S0_MOUNT_BASE:-}/XYAIFS00/HDD_POOL" ] && have_ai=1
            if [ "$have_a" = 1 ] && [ "$have_ai" = 0 ]; then
                prefix=/XYFS02
            elif [ "$have_ai" = 1 ] && [ "$have_a" = 0 ]; then
                prefix=/XYAIFS00
            elif [ "$have_a" = 1 ] && [ "$have_ai" = 1 ]; then
                echo "openQHA: both /XYFS02 and /XYAIFS00 are mounted here and no partition is" >&2
                echo "  known, so the root cannot be chosen. Set OPENQHA_PARTITION (ai, temp, cn," >&2
                echo "  deimos, debug, a100x, h100x, hx, a800x, v100x) or S0_RUNS_ROOT." >&2
                return 1
            else
                echo "openQHA: neither /XYFS02/HDD_POOL nor /XYAIFS00/HDD_POOL is mounted and" >&2
                echo "  no partition is known, so this is not a Tianhe node this file" >&2
                echo "  recognises. Set S0_RUNS_ROOT explicitly." >&2
                return 1
            fi ;;
        *)
            echo "openQHA: partition '$part' is not one this file knows (ai, temp, cn, deimos," >&2
            echo "  debug -> /XYFS02; a100x, h100x, hx, a800x, v100x -> /XYAIFS00). Add it here, or set" >&2
            echo "  S0_RUNS_ROOT explicitly." >&2
            return 1 ;;
    esac

    user="$(basename "$HOME")"
    acct="$(basename "$(dirname "$HOME")")"
    if [ -z "$user" ] || [ -z "$acct" ] || [ "$acct" = "/" ]; then
        echo "openQHA: cannot read <acct>/<user> out of HOME=$HOME; set S0_RUNS_ROOT." >&2
        return 1
    fi
    tail="${S0_RUNS_TAIL:-sherwin/runs}"

    export S0_RUNS_ROOT="$prefix/HDD_POOL/$acct/$user/$tail"
    unset S0_RUNS_ROOT_IS_DEFAULT
    return 0
}

# The molecule directory of <qid> under <tag>, FOUND rather than composed: the shard rule
# lives in openqha/store/layout.py and nothing else spells it. Prints the directory, or
# nothing when branch A has not written `mace/basin00/basin.extxyz` there. Needs
# S0_RUNS_ROOT (openqha_resolve_root, or an explicit export).
#
# Not a login-node necessity: a login node can `conda activate openqha` and ask
# openqha.store.basins the same question. These two functions exist because the submit
# scripts are shell, and were accepted by the user on 2026-09-14 as part of the work
# (they were not asked for; see .scratch/native-engine-files/spec.md).
openqha_find_molecule() {
    local tag="$1" qid="$2" hit
    [ -n "$S0_RUNS_ROOT" ] || return 1
    hit="$(find "$S0_RUNS_ROOT/$tag" -path "*/$qid/mace/basin00/basin.extxyz" -print -quit 2>/dev/null)"
    [ -n "$hit" ] || return 1
    hit="${hit%/mace/basin00/basin.extxyz}"
    printf '%s\n' "$hit"
}

# How many basins a molecule directory holds: the basinNN folders under mace/ that hold
# a basin.extxyz. Prints the count (0 when none).
openqha_count_basins() {
    local mol="$1" n=0 d
    for d in "$mol"/mace/basin[0-9][0-9]; do
        [ -f "$d/basin.extxyz" ] && n=$((n + 1))
    done
    printf '%s\n' "$n"
}
