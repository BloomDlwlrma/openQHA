"""Whether nbconvert succeeded can be judged only by the notebook itself.

TOOLING. Renders a notebook and judges the run by the notebook's own counters.
Produces no science.

Three counters decide it -- errors / figures / code-cells-without-output -- and never the
shell exit code, which has already failed three times: `| tail`; `| grep` under
`set -o pipefail`; capturing the exit code inside `wsl bash -lc`. Each time it reported a failure as 0.

Usage:  python scripts/tooling/dump_notebook.py [notebook.ipynb]
"""
import json
import sys
from pathlib import Path

def _repo_root():
    """Directory holding the openQHA package, found by walking up from this file.

    Depth-independent on purpose: this file keeps working wherever under the
    repository it is moved to. The earlier move into `scripts/_superseded/` broke
    every `parents[1]` in the moved files silently, which is what this replaces.
    """
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


DEFAULT_NB = _repo_root() / "docs" / "s0-1_conformer-to-free-energy.ipynb"
p = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_NB
nb = json.loads(p.read_text(encoding="utf-8"))

n_md = n_code = n_err = n_img = n_noout = 0
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "markdown":
        n_md += 1
        continue
    n_code += 1
    outs = c.get("outputs", [])
    if not outs:
        n_noout += 1
    texts = []
    for o in outs:
        if o["output_type"] == "error":
            n_err += 1
            texts.append("!!! ERROR: " + o.get("ename", "") + ": " + str(o.get("evalue", "")))
        elif o["output_type"] == "stream":
            texts.append("".join(o.get("text", [])))
        elif o["output_type"] in ("execute_result", "display_data"):
            d = o.get("data", {})
            if "image/png" in d:
                n_img += 1
                texts.append("[FIGURE]")
            if "text/plain" in d and "image/png" not in d:
                texts.append("".join(d["text/plain"]))
    if texts:
        print("=" * 84)
        print("CODE CELL {}".format(i))
        print("=" * 84)
        print("".join(texts).rstrip())

print()
print("#" * 84)
print("markdown {}  code {}  figures {}  errors {}  code-cells-without-output {}".format(
    n_md, n_code, n_img, n_err, n_noout))
