"""Write a record as TOML.

The five records this repository writes about a run take the format CREST uses for its
own settings: TOML, readable by a person and by the standard library (`tomllib`, Python
3.11). The standard library has no writer, so this is it -- one rule, held by
`tests/unit/t_toml_record_roundtrip.py`: what `tomllib.loads` reads back equals what was
given, minus the Nones.

    dumps(record) -> str            a dict of scalars, lists and dicts
    dump(record, path)              written to `.part` and renamed into place

TOML has no null. A None value is simply absent (readers already use `.get`); a None
inside a list, where dropping it would shift the other entries, becomes the string
"None". Tuples are lists. A key that is not a bare key (`A-Za-z0-9_-`) is quoted.
Dicts go out as tables and sub-tables, lists of dicts as arrays of tables, everything
else inline, so a record with nested sections stays readable.
"""
import math
import re
from pathlib import Path

_BARE = re.compile(r"^[A-Za-z0-9_-]+$")


def _key(k):
    k = str(k)
    return k if _BARE.match(k) else _string(k)


def _string(s):
    s = str(s)
    out = []
    for ch in s:
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif ord(ch) < 0x20 or ch == "\x7f":
            out.append("\\u{:04X}".format(ord(ch)))
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def _scalar(v):
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
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):      # numpy scalars
        return _scalar(v.item())
    if v is None:
        return '"None"'
    return _string(v)


def _is_table(v):
    return isinstance(v, dict)


def _is_array_of_tables(v):
    return isinstance(v, (list, tuple)) and len(v) > 0 and all(isinstance(x, dict) for x in v)


def _inline(v):
    """An inline value: scalar or (nested) array. Lists of dicts inside arrays are inline tables."""
    if isinstance(v, dict):
        items = ["{} = {}".format(_key(k), _inline(x)) for k, x in v.items() if x is not None]
        return "{ " + ", ".join(items) + " }" if items else "{ }"
    if hasattr(v, "tolist"):                                        # numpy arrays
        v = v.tolist()
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_inline(x) for x in v) + "]"
    return _scalar(v)


def _emit(lines, obj, prefix):
    """Emit `obj` (a dict) as the table `prefix` (a list of key parts): scalars first,
    then sub-tables and arrays of tables, so every value sits under the right header."""
    plain, tables, arrays = [], [], []
    for k, v in obj.items():
        if v is None:
            continue
        if _is_table(v):
            tables.append((k, v))
        elif _is_array_of_tables(v):
            arrays.append((k, v))
        else:
            plain.append((k, v))
    for k, v in plain:
        lines.append("{} = {}".format(_key(k), _inline(v)))
    for k, v in tables:
        path = prefix + [k]
        lines += ["", "[{}]".format(".".join(_key(p) for p in path))]
        _emit(lines, v, path)
    for k, v in arrays:
        path = prefix + [k]
        header = "[[{}]]".format(".".join(_key(p) for p in path))
        for item in v:
            lines += ["", header]
            _emit(lines, item, path)


def dumps(record):
    if not isinstance(record, dict):
        raise TypeError("a TOML document is a table: give a dict, not {}".format(type(record).__name__))
    lines = []
    _emit(lines, record, [])
    return "\n".join(lines).lstrip("\n") + "\n"


def dump(record, path):
    """Write atomically: `.part`, then rename, so a killed job leaves no half record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(dumps(record), encoding="utf-8")
    tmp.replace(path)
    return path


def load(path):
    import tomllib
    with open(path, "rb") as fh:
        return tomllib.load(fh)
