#!/bin/bash
# =======================================================================================
# Move openQHA files between the two Tianhe storage systems.
#
#     bash hpc/tools/xfer_tianhe_ai.sh push data/basins/acetone data/potentials
#     bash hpc/tools/xfer_tianhe_ai.sh pull data/trajectories/acetone
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
# =======================================================================================
# `set -eo pipefail` removed 2026-09-13 (user ruling: a failing step must not end the job; .mem/notes/notes_2026-09-13_no-errexit-anywhere.md)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-}"
shift || true

if [ "$MODE" != "push" ] && [ "$MODE" != "pull" ] || [ $# -eq 0 ]; then
    echo "usage: bash hpc/tools/xfer_tianhe_ai.sh push|pull <path relative to repo> [...]" >&2
    echo "   push: XYFS02 (here) -> XYAIFS00      pull: XYAIFS00 -> XYFS02 (here)" >&2
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
                   echo "  TianheXY-AI account's key here first." >&2; exit 2; }
perm="$(stat -c %a "$KEY" 2>/dev/null || stat -f %Lp "$KEY")"
if [ "$perm" != "400" ]; then
    echo "openQHA: $KEY is mode $perm; ssh requires 400 (manual: chmod 400 accountB.id)." >&2
    exit 2
fi

echo "xfer      $MODE"
echo "  here    $ROOT            (XYFS02: tianhexy-cn + tianhexy-a)"
echo "  there   $ACCT@$HOST:$XYAI_ROOT   (XYAIFS00: tianhexy-ai)"
for rel in "$@"; do
    rel="${rel%/}"
    parent="$(dirname "$rel")"
    if [ "$MODE" = "push" ]; then
        [ -e "$ROOT/$rel" ] || { echo "  skip    $rel (not here)" >&2; continue; }
        echo "  push    $rel"
        # The parent must exist on the far side; scp does not create it.
        ssh -i "$KEY" "$ACCT@$HOST" "mkdir -p '$XYAI_ROOT/$parent'"
        scp -i "$KEY" -r "$ROOT/$rel" "$ACCT@$HOST:$XYAI_ROOT/$parent/"
    else
        echo "  pull    $rel"
        mkdir -p "$ROOT/$parent"
        scp -i "$KEY" -r "$ACCT@$HOST:$XYAI_ROOT/$rel" "$ROOT/$parent/"
    fi
done
echo "done"
