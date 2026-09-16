"""The Table: whitespace `.dat`, the way CREST writes `crest.energies` (ADR 0003).

One file holds one or more named sections (user ruling 2026-09-16, ADR 0003 amendment):

    [trajectories]
    #   species       String: the molecule (QM9 index)
    #   n_frames      Integer: frames the covariance was formed from
    #   TS_QH_kcal    Double, kcal/mol: T*S from the quasi-harmonic entropy
    # species n_frames TS_QH_kcal
    dsgdb9nsd_000018 3000 3.215

    [blank]
    ...

A bare `[name]` line opens a section. Then one comment line per column, in the Property
file's form (`Type, unit: doc`, from a schema `{column: (Type, unit or None, doc)}`),
then the header naming the columns, then one row per line, whitespace-separated, then a
blank line. The header is the LAST `#` line before the rows, which the writer guarantees
by putting the comments first; a `#` line after the rows is a comment. A value with
whitespace or quotes is double-quoted and backslash-escaped (`shlex` reads it back);
None is `NA`; booleans are `true`/`false`; floats are `repr`, so `nan` and `inf` survive
and so does every digit. A string beginning with `[` is quoted, so no row can be taken
for a section line.

A file without any `[name]` line is one unnamed section (the form before 2026-09-16, and
what a table written by hand looks like): `read_table` reads it, `read_tables` returns it
under the key "". `read_table` on a sectioned file refuses, naming the sections, rather
than silently returning the first.

    write_tables(path, sections, schema=None) -> [(section, column)] the schema did not know
    write_table(path, rows, columns=None, schema=None) -> path        one unnamed section
    read_tables(path) -> {section: rows}
    read_table(path) -> rows

No library beyond the standard one, which is the point: the ensemble report reads these
where it used to need pandas and a parquet engine.
"""
import math
import re
import shlex
from pathlib import Path

_SECTION = re.compile(r"^\[([A-Za-z0-9_.-]+)\]$")


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
    if (s == "" or any(ch.isspace() for ch in s) or '"' in s or "'" in s or "\\" in s
            or s.startswith("[") or s.startswith("#") or s in ("NA",)):
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


def _comment_lines(cols, schema):
    """One `#   column   Type, unit: doc` line per column the schema knows; the unknown
    columns are returned so the caller's test can catch a column added without its doc."""
    if not schema:
        return [], list(cols)
    width = max((len(c) for c in cols), default=0)
    lines, unknown = [], []
    for c in cols:
        spec = schema.get(c)
        if spec is None:
            unknown.append(c)
            continue
        typ, unit, doc = spec
        head = typ if not unit else "{}, {}".format(typ, unit)
        lines.append("#   {}  {}: {}".format(c.ljust(width), head, doc).rstrip())
    return lines, unknown


def _section_lines(rows, columns, schema):
    cols = list(columns) if columns is not None else (list(rows[0].keys()) if rows else [])
    comments, unknown = _comment_lines(cols, schema)
    lines = comments + ["# " + " ".join(cols)]
    for r in rows:
        lines.append(" ".join(_cell(r.get(c)) for c in cols))
    return lines, unknown


def _write(path, lines):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def write_tables(path, sections, schema=None):
    """Write named sections. `sections` is an ordered mapping or a list of pairs,
    `name -> rows` or `name -> (rows, columns)`; every section is written, empty or not
    (its comments and header still say what it would hold). `schema` is
    `{section: {column: (Type, unit or None, doc)}}`. Atomic. Returns the
    (section, column) pairs the schema did not know."""
    items = list(sections.items()) if hasattr(sections, "items") else list(sections)
    lines, unknown = [], []
    for name, body in items:
        if not _SECTION.match("[{}]".format(name)):
            raise ValueError("section name {!r} is not a plain word (letters, digits, _ . -)".format(name))
        rows, columns = (body if isinstance(body, tuple) else (body, None))
        sec, miss = _section_lines(rows, columns, (schema or {}).get(name))
        lines += ["[{}]".format(name)] + sec + [""]
        unknown += [(name, c) for c in miss]
    _write(path, lines[:-1] if lines else lines)
    return unknown


def write_table(path, rows, columns=None, schema=None):
    """One unnamed section: the columns of the first row (or `columns`), with the
    column comments when `schema` (`{column: (Type, unit, doc)}`) is given. Atomic."""
    lines, _unknown = _section_lines(rows, columns, schema)
    return _write(path, lines)


def read_tables(path):
    """`{section: rows}` in file order; a file without `[name]` lines is `{"": rows}`.
    Refuses a missing file by name."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("no table at {}".format(path))
    out = {}
    cur = None                                   # [name, comment lines, columns, rows]

    def flush(sec):
        if sec is None:
            return
        name, comments, cols, rows = sec
        if cols is None and not comments and name == "":
            return                               # an empty file: no table at all
        out[name] = rows

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            m = _SECTION.match(s)
            if m:
                flush(cur)
                cur = [m.group(1), [], None, []]
                continue
            if cur is None:
                cur = ["", [], None, []]
            if s.startswith("#"):
                if cur[2] is None:
                    cur[1].append(s)             # the last of these is the header
                continue                         # a comment after the rows
            if cur[2] is None:
                if not cur[1]:
                    raise ValueError("{} has rows before any header line".format(path))
                cur[2] = cur[1][-1][1:].split()
            toks = shlex.split(s, posix=True)
            cur[3].append({c: _value(tok) for c, tok in zip(cur[2], toks)})
    flush(cur)
    return out


def read_table(path):
    """The rows of an unnamed (single) table. A sectioned file is refused by naming its
    sections: ask `read_tables` for the one you mean."""
    tables = read_tables(path)
    if list(tables) == [""] or not tables:
        return tables.get("", [])
    raise ValueError("{} holds sections {}; read_tables(path)[name] picks one"
                     .format(path, sorted(k for k in tables if k)))
