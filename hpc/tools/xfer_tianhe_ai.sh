#!/bin/bash
# =======================================================================================
# Move openQHA files between the two Tianhe storage systems.
#
#     bash hpc/tools/xfer_tianhe_ai.sh check                       # can I reach it?
#     bash hpc/tools/xfer_tianhe_ai.sh push-repo                   # code + checkouts + basins + weights
#     bash hpc/tools/xfer_tianhe_ai.sh push data/basins/acetone data/potentials
#     bash hpc/tools/xfer_tianhe_ai.sh pull analysis/qha/02d_t30
#     DRY_RUN=1 bash hpc/tools/xfer_tianhe_ai.sh push-repo         # show, do not copy
#
# WHY THIS EXISTS (site manual 3.2.2; storage tags read 2026-09-11)
# ----------------------------------------------------------------
#     XYFS02     tianhexy-cn, tianhexy-a, k8s_xingyi, k8s_xingyiAI
#     XYAIFS00   tianhexy-ai, k8s_xingyiAI_2
#
# TianheXY-CN (deimos, branch A) and TianheXY-A (ai, branch B) SHARE a filesystem, so a
# basin list written by a CPU job is simply there for the GPU job: nothing to copy.
# TianheXY-AI (h100x) is on a different one. Anything h100x needs -- the branch A
# products, the weights, the repository itself -- must be copied, and anything it
# produces must be copied back. The manual's recipe is scp with the OTHER account's key:
#
#     scp -i accountB.id -r <path on XYFS02>  accountB@XYAIFS00:<path on XYAIFS00>
#     scp -i accountB.id -r accountB@XYAIFS00:<path on XYAIFS00>  <path on XYFS02>
#
# This script is that recipe with the paths filled in from where it is run.
#
# SETTINGS (environment; every one has a default)
#     XYAI_KEY       the TianheXY-AI account's private key, uploaded here. Default
#                    ~/accountB.id. Must be mode 400 -- checked, because ssh refuses a
#                    looser key with a message that does not say so.
#     XYAI_ACCOUNT   the TianheXY-AI account name. Default: $USER.
#     XYAI_HOST      default XYAIFS00, as in the manual.
#     XYAI_ROOT      the repository root ON XYAIFS00. Default: this repository's own
#                    path with /XYFS02/ (or /XYFS01/, /GLOBALFS/) replaced by /XYAIFS00/,
#                    which is how the two HDD_POOL trees are laid out for this project.
#
# Paths are RELATIVE TO THE REPOSITORY ROOT on both sides, so the tree stays congruent
# and every script's relative path (data/basins/<tag>/...) resolves on both clusters.
#
# RSYNC FIRST, SCP AS THE FALLBACK -- and why (2026-09-13). The manual's `scp -r DIR
# dest/` has cp's semantics: when dest/DIR already exists it copies INTO it, producing
# dest/DIR/DIR. The AI side already holds an older copy of this repository from the
# September runs, so a plain scp of `openqha` would have nested a second package inside
# the first and Python would have imported whichever came first. rsync has no such mode,
# sends only what changed (the weights are 100+ MB and never change), and takes an
# exclude list, so `logs/`, `analysis/`, `__pycache__` and the frozen `_backup/` stay
# where they are. When rsync is missing on either end, scp is used with `DIR/.` -- the
# form that copies contents, not the directory -- which avoids the nesting too.
#
# `push-repo` is the curated list the h100x tests need and nothing else: code, confs,
# the branch A basins, the weights, the environment files, the workflows and docs, and
# the two training checkouts -- `mace` and `openQHA-Hessian`, SIBLINGS of this repository
# (install_env_tianhe.slurm section 9 installs the training stack from them). Not
# analysis, not logs.
# =======================================================================================
# `set -eo pipefail` removed 2026-09-13 (user ruling: a failing step must not end the job; .mem/notes/notes_2026-09-13_no-errexit-anywhere.md)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

MODE="${1:-}"
shift || true
case "$MODE" in
    push|pull) [ $# -gt 0 ] || MODE="" ;;
    push-repo|check) ;;
    *) MODE="" ;;
esac
if [ -z "$MODE" ]; then
    echo "usage: bash hpc/tools/xfer_tianhe_ai.sh check" >&2
    echo "       bash hpc/tools/xfer_tianhe_ai.sh push-repo" >&2
    echo "       bash hpc/tools/xfer_tianhe_ai.sh push|pull <path relative to repo> [...]" >&2
    echo "   push: XYFS02 (here) -> XYAIFS00      pull: XYAIFS00 -> XYFS02 (here)" >&2
    echo "   DRY_RUN=1 shows what would move without moving it" >&2
    exit 2
fi

KEY="${XYAI_KEY:-$HOME/accountB.id}"
ACCT="${XYAI_ACCOUNT:-$USER}"
HOST="${XYAI_HOST:-XYAIFS00}"
if [ -z "${XYAI_ROOT:-}" ]; then
    XYAI_ROOT="$(printf '%s' "$ROOT" | sed -E 's#^/(XYFS01|XYFS02|GLOBALFS)/#/XYAIFS00/#')"
    if [ "$XYAI_ROOT" = "$ROOT" ]; then
        echo "openQHA: this repository is not under /XYFS01, /XYFS02 or /GLOBALFS, so the" >&2
        echo "  XYAIFS00 path cannot be derived. Set XYAI_ROOT=<repo root on XYAIFS00>." >&2
        exit 2
    fi
fi

[ -f "$KEY" ] || { echo "openQHA: no key at $KEY (set XYAI_KEY). Manual 3.2.2: upload the" >&2
                   echo "  TianheXY-AI account's key here first, then chmod 400 it." >&2; exit 2; }
perm="$(stat -c %a "$KEY" 2>/dev/null || stat -f %Lp "$KEY")"
if [ "$perm" != "400" ]; then
    echo "openQHA: $KEY is mode $perm; ssh requires 400 (manual: chmod 400 accountB.id)." >&2
    exit 2
fi

SSH=(ssh -i "$KEY" -o BatchMode=yes -o ConnectTimeout=20)

echo "xfer      $MODE${DRY_RUN:+   (DRY RUN)}"
echo "  here    $ROOT            (XYFS02: tianhexy-cn + tianhexy-a)"
echo "  there   $ACCT@$HOST:$XYAI_ROOT   (XYAIFS00: tianhexy-ai)"

# ---- can the other side be reached at all, and does it have rsync? ----------------------
remote="$("${SSH[@]}" "$ACCT@$HOST" "echo \"host=\$(hostname) rsync=\$(command -v rsync || echo none) root_exists=\$([ -d '$XYAI_ROOT' ] && echo yes || echo no)\"" 2>&1)"
rc=$?
if [ "$rc" -ne 0 ]; then
    echo "openQHA: cannot reach $ACCT@$HOST with $KEY (ssh exit $rc):" >&2
    echo "  $remote" >&2
    echo "  Check: is XYAI_ACCOUNT the TianheXY-AI account name? is this the key for THAT" >&2
    echo "  account (the one you log into ln301 with)? does XYAIFS00 resolve from here?" >&2
    exit 2
fi
echo "  remote  $remote"
if [ "$MODE" = "check" ]; then
    echo "reachable"
    exit 0
fi

have_rsync=no
command -v rsync >/dev/null 2>&1 && case "$remote" in *rsync=none*) ;; *) have_rsync=yes ;; esac
echo "  method  $([ "$have_rsync" = yes ] && echo rsync || echo "scp (rsync missing on one side)")"

# What never travels in EITHER direction: caches and sockets. What stays home on a PUSH:
# results, logs, the frozen baseline, the notes -- a pull is usually FOR the results, so
# `pull analysis/qha/<tag>` must not exclude `analysis` (a dry run caught exactly that).
# `/logs` is ANCHORED on purpose: an unanchored `logs` would also match `.git/logs` inside
# a pushed checkout, and the checkouts must arrive with `.git` COMPLETE -- `mace_fork_info()`
# reads the fork's commit from it, and training refuses anything else.
COMMON_EXCLUDES=(--exclude '__pycache__' --exclude '*.pyc' --exclude '*.sock')
PUSH_EXCLUDES=("${COMMON_EXCLUDES[@]}" --exclude '/logs' --exclude 'analysis'
               --exclude '_backup' --exclude '.mem' --exclude 'runs'
               --exclude 'generated/manifest')

if [ "$MODE" = "push-repo" ]; then
    # Code, confs, tests, environment files, the branch A products and the weights --
    # plus the TRAINING STACK: the workflows and docs that point at the checkouts, and
    # the checkouts themselves. `../mace` and `../openQHA-Hessian` are SIBLINGS of this
    # repository (the layout install_env_tianhe.slurm section 9 defaults to) and travel
    # `.git` complete -- `mace_fork_info()` reads the fork's commit from it.
    set -- openqha scripts workflows docs hpc examples configs tests patches \
           data/basins data/potentials \
           environment-tianhe-gpu.yml environment-tianhe.yml environment.yml \
           install_dependency.sh check_dependency.py requirements.txt requirements-minimal.txt README.md \
           ../mace ../openQHA-Hessian
    MODE=push
fi

fail=0
for rel in "$@"; do
    rel="${rel%/}"
    parent="$(dirname "$rel")"
    if [ "$MODE" = "push" ]; then
        [ -e "$ROOT/$rel" ] || { echo "  skip    $rel (not here)" >&2; continue; }
        echo "  push    $rel"
        [ -n "${DRY_RUN:-}" ] && [ "$have_rsync" != yes ] && continue
        "${SSH[@]}" "$ACCT@$HOST" "mkdir -p '$XYAI_ROOT/$parent'" || { fail=1; continue; }
        if [ "$have_rsync" = yes ]; then
            rsync -az ${DRY_RUN:+-n -v} --info=stats1 "${PUSH_EXCLUDES[@]}" -e "${SSH[*]}" \
                "$ROOT/$rel" "$ACCT@$HOST:$XYAI_ROOT/$parent/" || fail=1
        elif [ -d "$ROOT/$rel" ]; then
            "${SSH[@]}" "$ACCT@$HOST" "mkdir -p '$XYAI_ROOT/$rel'" || { fail=1; continue; }
            scp -i "$KEY" -r "$ROOT/$rel/." "$ACCT@$HOST:$XYAI_ROOT/$rel/" || fail=1
        else
            scp -i "$KEY" "$ROOT/$rel" "$ACCT@$HOST:$XYAI_ROOT/$parent/" || fail=1
        fi
    else
        echo "  pull    $rel"
        [ -n "${DRY_RUN:-}" ] && [ "$have_rsync" != yes ] && continue
        mkdir -p "$ROOT/$parent"
        if [ "$have_rsync" = yes ]; then
            rsync -az ${DRY_RUN:+-n -v} --info=stats1 "${COMMON_EXCLUDES[@]}" -e "${SSH[*]}" \
                "$ACCT@$HOST:$XYAI_ROOT/$rel" "$ROOT/$parent/" || fail=1
        else
            scp -i "$KEY" -r "$ACCT@$HOST:$XYAI_ROOT/$rel" "$ROOT/$parent/" || fail=1
        fi
    fi
done
if [ "$fail" = 0 ]; then
    echo "done"
else
    echo "done, WITH FAILURES above" >&2
    exit 1
fi
