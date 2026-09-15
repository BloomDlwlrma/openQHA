"""Whitespace tables, the way CREST writes `crest.energies` (step 2, user ruling 2026-09-15).

    # species basin seed n_frames TS_QH_kcal passed
    dsgdb9nsd_000018 basin00 seed00 3000 3.215 true
    ...

One header line naming the columns, then one row per line, whitespace-separated. A value
with whitespace or quotes is double-quoted and backslash-escaped (`shlex` reads it back);
None is `NA`; booleans are `true`/`false`; floats are `repr`, so `nan` and `inf` survive
and so does every digit. `read_table` gives the rows back as dicts, numbers as numbers.
No library beyond the standard one, which is the point: the ensemble report reads these
where it used to need pandas and a parquet engine.
"""
import math
import shlex
from pathlib import Path


def _cell(v):
    if v is None:
        return "NA"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return repr(v)
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):     # numpy scalars
        return _cell(v.item())
    s = str(v)
    if s == "" or any(ch.isspace() for ch in s) or '"' in s or "'" in s or "\\" in s or s in ("NA",):
        return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return s


def _value(tok):
    if tok == "NA":
        return None
    if tok == "true":
        return True
    if tok == "false":
        return False
    try:
        return int(tok)
    except ValueError:
        pass
    try:
        return float(tok)
    except ValueError:
        return tok


def write_table(path, rows, columns=None):
    """Write `rows` (dicts) with the columns of the first row (or `columns`). Atomic."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(columns) if columns is not None else (list(rows[0].keys()) if rows else [])
    lines = ["# " + " ".join(cols)]
    for r in rows:
        lines.append(" ".join(_cell(r.get(c)) for c in cols))
    tmp = path.with_name(path.name + ".part")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def read_table(path):
    """The rows back as dicts. Refuses a missing file by name."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("no table at {}".format(path))
    cols, rows = None, []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                if cols is None:
                    cols = s[1:].split()
                continue
            toks = shlex.split(s, posix=True)
            if cols is None:
                raise ValueError("{} has no header line".format(path))
            rows.append({c: _value(tok) for c, tok in zip(cols, toks)})
    return rows
