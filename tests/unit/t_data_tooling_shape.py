"""The data tooling's shape after the checksum retirement: no hash machinery anywhere.

UNIT. Seconds; reads files, runs the prep tool's --check mode (which touches nothing).

Ticket 16b: the reference-data tools write/print no hashes. The curated pack's shape is
asserted in `t_curated_qm9_h5.py`; this file pins the other two -- the analysis/ INDEX
generator (`openqha_index_artifacts`) keeps no digest helper and writes no sha256 column,
and the QM9 prep tool (`s0_prepare_data`) keeps no head-digest helper and prints no digest
in its report.

Run::  python tests/unit/t_data_tooling_shape.py
"""
import contextlib
import io
import sys
import tempfile
from pathlib import Path


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "tooling"))
import openqha_index_artifacts as idx                       # noqa: E402
import s0_prepare_data as prep                              # noqa: E402

FAIL = []


def check(label, ok, detail=""):
    print("  {:80s} {}".format(label, "ok" if ok else "FAIL " + str(detail)[:300]))
    if not ok:
        FAIL.append(label)


def main():
    print("Data tooling shape -- no hash machinery")
    print("-" * 88)

    check("the artifact index module keeps no digest helper and imports no hashlib",
          not hasattr(idx, "sha256_of")
          and "hashlib" not in Path(idx.__file__).read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory(prefix="tooling_shape_") as tmp:
        tmp = Path(tmp)
        (tmp / "thing.json").write_text('{"a": 1}', encoding="utf-8")
        (tmp / "sub").mkdir()
        (tmp / "sub" / "x.txt").write_text("x", encoding="utf-8")
        old = idx.S0_ROOT
        idx.S0_ROOT = tmp          # scan() records paths relative to S0_ROOT
        try:
            rows = idx.scan(tmp)
            idx.write_index(rows, tmp)
        finally:
            idx.S0_ROOT = old
        row_keys = set(rows[0].keys())
        header = (tmp / "INDEX.csv").read_text(encoding="utf-8").splitlines()[0]
        check("the index scan rows and the written CSV carry path/name/sizes/mtime -- "
              "no sha256 column",
              bool(rows) and "sha256" not in row_keys and "name" in row_keys
              and "bytes" in row_keys and "sha256" not in header
              and header == ",".join(rows[0].keys()),
              (sorted(row_keys), header))

    check("the prep tool keeps no head-digest helper", not hasattr(prep, "sha256_head"))

    old_argv = sys.argv
    try:
        sys.argv = ["s0_prepare_data", "--check"]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            prep.main()
    finally:
        sys.argv = old_argv
    out = buf.getvalue()
    check("the prep tool's --check report prints no digest and no sha256 (sizes and "
          "counts are fine)",
          bool(out) and "digest" not in out.lower() and "sha256" not in out.lower(),
          out[:200])

    print()
    print("PASS" if not FAIL else "FAIL: " + "; ".join(FAIL))
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
