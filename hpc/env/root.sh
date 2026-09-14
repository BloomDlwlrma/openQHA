# hpc/env/root.sh -- the root of the molecule tree on Tianhe (ADR 0002, user 2026-09-14).
#
# Sourced by hpc/env/tianhe.sh; safe to source anywhere because it only DEFINES a
# function. `openqha_resolve_root` exports S0_RUNS_ROOT or returns 1 with the reason on
# stderr; it never falls back to $HOME (the 2026-09-12 defect was a default nobody chose).
#
#     root = <prefix>/HDD_POOL/<acct>/<user>/<S0_RUNS_TAIL>
#     prefix   /XYFS02   for partitions ai, cn                (TianheXY-A, TianheXY-CN)
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
        ai|cn)                     prefix=/XYFS02 ;;
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
                echo "  known, so the root cannot be chosen. Set OPENQHA_PARTITION (ai, cn," >&2
                echo "  a100x, h100x, hx, a800x, v100x) or S0_RUNS_ROOT." >&2
                return 1
            else
                echo "openQHA: neither /XYFS02/HDD_POOL nor /XYAIFS00/HDD_POOL is mounted and" >&2
                echo "  no partition is known, so this is not a Tianhe node this file" >&2
                echo "  recognises. Set S0_RUNS_ROOT explicitly." >&2
                return 1
            fi ;;
        *)
            echo "openQHA: partition '$part' is not one this file knows (ai, cn -> /XYFS02;" >&2
            echo "  a100x, h100x, hx, a800x, v100x -> /XYAIFS00). Add it here, or set" >&2
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
