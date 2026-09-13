#!/bin/bash
# One function, no side effects, sourced by hpc/env/common.sh on the clusters and by
# examples/chain_body.sh everywhere.
#
# The python counterpart of openqha_require (executables). A missing executable fails
# at t=0; a python module that only the LAST step imports fails after the compute --
# an113, 2026-09-13: six trajectories, a complete analysis, then
# "Missing optional dependency 'fastparquet'" on collect's final line, because the
# openqha-gpu environment there had no pyarrow. Prints one line of versions on
# success so the job log records what wrote its products; exits 1 naming what is
# missing and from which prefix.
openqha_require_modules() {
    python - "$@" <<'EOF'
import importlib
import sys
missing, found = [], []
for name in sys.argv[1:]:
    try:
        mod = importlib.import_module(name)
    except ImportError:
        missing.append(name)
        continue
    found.append("{} {}".format(name, getattr(mod, "__version__", "?")))
print("modules   " + "  ".join(found))
if missing:
    print("openQHA: python module(s) missing from {}: {}".format(
        sys.prefix, ", ".join(missing)), file=sys.stderr)
    sys.exit(1)
EOF
}
