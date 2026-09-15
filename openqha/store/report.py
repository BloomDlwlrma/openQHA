"""The **readable form** of a product -- ORCA/CREST-style `.log` text plus parquet tables.

--------------------------------------------------------------------------------------
Why this changed, and what it changed to
--------------------------------------------------------------------------------------
The user, 2026-08-31: "`analysis/` is all JSON and unreadable; change it to `.log` text
like ORCA and CREST produce, or to parquet".

**The conclusion is "both, with a division of labour", not one or the other** -- because
products contain two kinds of thing with entirely different natures:

| Content | Example | The form that suits it |
|---|---|---|
| **Narrative and provenance**: settings, criteria, what happened, verdicts | the engine and its SHA-256, the CREST version, "anything with a non-zero imaginary frequency is thrown out" | **`.log`** -- banner, sections, aligned tables, units in the header |
| **Large amounts of homogeneous numbers** | 4000 molecules x several basins each x 3N-6 frequencies per basin | **parquet** -- one line to read into pandas, no scrolling in a text editor |

**JSON is not deleted; it is demoted to an archive.** Three reasons:

1. It is the **complete record**. A `.log` is for people to read and therefore
   necessarily selective; parquet is a flat table and cannot hold nested provenance. The
   checkpoint standard requires the raw record to be kept -- that is the JSON.
2. Scripts already in the repository read JSON (`s0_package1_summarise.py` and others).
3. **Both the `.log` and the parquet are GENERATED from the JSON**, so the three cannot
   drift: if the JSON changes, rerun the conversion.

**So each product has three files**::

    xxx.json      the complete record (archival, machine-read, **not for people**)
    xxx.log       the ORCA/CREST-style report (**the one for people**)
    xxx.parquet   the flat table (generated only when there are many homogeneous numbers)

--------------------------------------------------------------------------------------
What a `.log` looks like
--------------------------------------------------------------------------------------
Following the typographic habits of ORCA and CREST: a centred banner, `----` section
rules, key-value lines with leader dots and left-aligned keys, numbers right-aligned with
their unit in the header, and a `PASS`/`FAIL` column for verdicts. **No Markdown tables**
(`|` is harder to read in a monospace terminal than in a renderer); alignment is by
spaces.
"""
import datetime as _dt
import json
import math
import unicodedata
from pathlib import Path

WIDTH = 96


# --------------------------------------------------------------------------------------
# **Display width** in a monospace terminal, which is not the number of characters
# --------------------------------------------------------------------------------------
# A CJK character occupies **two cells** in a monospace font. Aligning with `len()` would
# skew every table containing one -- and "unreadable" is the very problem being fixed
# here; a skewed table is barely better than JSON.
def dwidth(s):
    """How many cells a string occupies in a monospace terminal."""
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1
               for c in str(s))


def rjust(s, w):
    s = str(s)
    return " " * max(0, w - dwidth(s)) + s


def ljust(s, w):
    s = str(s)
    return s + " " * max(0, w - dwidth(s))


def center(s, w):
    s = str(s)
    pad = max(0, w - dwidth(s))
    left = pad // 2
    return " " * left + s + " " * (pad - left)

#: Key suffix -> unit. Product keys have always had the form "quantity_unit"
#: (`energy_eV`, `relative_kcal`, `lowest_frequency_cm_inv`), so the unit can be **read
#: off the key** and no hand-written table has to be maintained -- a hand-written table
#: would certainly drift away from the code.
UNIT_SUFFIX = (
    ("_eV_A2", "eV/A^2"), ("_eV_A", "eV/A"), ("_kcal_per_cm", "kcal/(mol*cm^-1)"),
    ("_cm_inv", "cm^-1"), ("_kcal", "kcal/mol"), ("_eV", "eV"), ("_Eh", "Eh"),
    ("_A2", "A^2"), ("_A", "A"), ("_K", "K"), ("_Pa", "Pa"), ("_kT", "kT"),
    ("_s", "s"), ("_seconds", "s"), ("_hours", "h"), ("_days", "d"),
    ("_amu", "amu"), ("_MB", "MB"), ("_GB", "GB"),
)


def unit_of(key):
    """Read the unit off the key suffix; return an empty string when there is none."""
    for suf, u in UNIT_SUFFIX:
        if key.endswith(suf):
            return u
    return ""


def strip_unit(key):
    for suf, _ in UNIT_SUFFIX:
        if key.endswith(suf):
            return key[: -len(suf)]
    return key


class Report:
    """Builder for one `.log`. Written section by section, then `write()`."""

    #: The "where the source of truth is" sentence in the footer. The default describes
    #: **the real situation for a per-molecule product**: the `.log` is laid out directly
    #: from the in-memory record, and the machine-readable copy is the `.parquet` of the
    #: same name (raw float64). The older conversion path through `json_to_log()` replaces
    #: it with the JSON wording.
    SOURCE_NOTE = ("this file is a **laid-out** copy for people (numbers truncated to a "
                   "fixed number of places); the machine-readable copy carrying full "
                   "float64 precision is the .parquet of the same name.")

    def __init__(self, title, subtitle=None, width=WIDTH, source_note=None):
        self.w = int(width)
        self.lines = []
        self.source_note = source_note or self.SOURCE_NOTE
        self._banner(title, subtitle)

    # ---------------------------------------------------------------- layout pieces
    def _banner(self, title, subtitle):
        self.lines += ["=" * self.w, center(title, self.w).rstrip()]
        if subtitle:
            self.lines.append(center(subtitle, self.w).rstrip())
        self.lines += ["=" * self.w, ""]

    def section(self, title):
        self.lines += ["", "-" * self.w, title, "-" * self.w]
        return self

    def blank(self):
        self.lines.append("")
        return self

    def text(self, s):
        for line in str(s).splitlines() or [""]:
            self.lines.append(line)
        return self

    def note(self, s):
        for i, line in enumerate(_wrap(str(s), self.w - 4)):
            self.lines.append(("  * " if i == 0 else "    ") + line)
        return self

    def warn(self, s):
        for i, line in enumerate(_wrap(str(s), self.w - 4)):
            self.lines.append(("  ! " if i == 0 else "    ") + line)
        return self

    def kv(self, key, value, unit=None, note=None):
        """`key ....... value unit` -- ORCA's leader-dot style."""
        u = unit if unit is not None else unit_of(key)
        label = strip_unit(key).replace("_", " ")
        v = _fmt(value)
        if u:
            v = "{} {}".format(v, u)
        pad = max(2, 40 - dwidth(label))
        self.lines.append("  {}{} {}".format(label, "." * pad, v))
        if note:
            self.note(note)
        return self

    def table(self, headers, rows, units=None, title=None):
        """A space-aligned table. `units` is the same length as `headers` and goes on the
        second header line."""
        if title:
            self.lines += ["", "  " + title]
        headers = [str(h) for h in headers]
        cells = [[_fmt(c) for c in row] for row in rows]
        wid = [dwidth(h) for h in headers]
        if units:
            wid = [max(w, dwidth(u or "")) for w, u in zip(wid, units)]
        for row in cells:
            for i, c in enumerate(row):
                if i < len(wid):
                    wid[i] = max(wid[i], dwidth(c))
        # the last column is left-aligned: it is usually explanatory text, and
        # right-aligning it would expose a ragged left edge
        def line(vals):
            out = []
            for i, v in enumerate(vals):
                if i >= len(wid):
                    break
                out.append(ljust(v, wid[i]) if i == len(wid) - 1 else rjust(v, wid[i]))
            return "  " + "  ".join(out).rstrip()
        self.lines.append(line(headers))
        if units:
            self.lines.append(line([u or "" for u in units]))
        self.lines.append("  " + "  ".join("-" * w for w in wid))
        for row in cells:
            self.lines.append(line(row))
        return self

    def verdict(self, criterion, measured, passed):
        """criterion -> measured -> PASS/FAIL. **The criterion is written as a sentence**,
        not referred to by an identifier."""
        tag = "PASS" if passed else "FAIL"
        self.lines.append("  [{}] {}".format(tag, criterion))
        self.lines.append("         measured: {}".format(_fmt(measured)))
        return self

    def json_dump(self, obj, title="complete record (expanded automatically)"):
        """Render any JSON structure verbatim as indented text -- **the fallback**, so
        that no information is left outside the `.log`."""
        self.section(title)
        self.lines += _render(obj, 1, self.w)
        return self

    def write(self, path, step=None):
        """Write the report. With `step`, the LAST line is the terminal line
        `openQHA <step> terminated normally` -- CREST's own convention, and since
        2026-09-15 the completion marker of every step that writes a `.out`: a report
        killed before its last line is not finished, and `terminated_normally()` says so.
        Written to `.part` and renamed, so the last line is either there or the file is not.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        foot = ["", "-" * self.w,
                "generated {} by openqha.report  (stage 0)".format(
                    _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                self.source_note,
                "-" * self.w]
        if step:
            foot += ["", terminal_line(step)]
        tmp = path.with_name(path.name + ".part")
        tmp.write_text("\n".join(self.lines + foot) + "\n", encoding="utf-8")
        tmp.replace(path)
        return path


def terminal_line(step):
    return "openQHA {} terminated normally".format(step)


def terminated_normally(path, step):
    """Does the `.out` at `path` end with this step's terminal line? False when the file
    is absent, empty, cut short, or another step's."""
    path = Path(path)
    if not path.is_file():
        return False
    try:
        text = path.read_text(encoding="utf-8", errors="replace").rstrip("\n")
    except OSError:
        return False
    return bool(text) and text.splitlines()[-1].strip() == terminal_line(step)


# ======================================================================================
# General rendering: any JSON -> indented text
# ======================================================================================
def _fmt(v):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        if v != v or math.isinf(v):
            return str(v)
        a = abs(v)
        if a and (a < 1e-4 or a >= 1e7):
            return "{:.6e}".format(v)
        return "{:.6f}".format(v).rstrip("0").rstrip(".") or "0"
    return str(v)


def _wrap(s, w):
    out, cur = [], ""
    for word in str(s).split():
        if cur and dwidth(cur) + 1 + dwidth(word) > w:
            out.append(cur)
            cur = word
        else:
            cur = (cur + " " + word).strip()
    if cur:
        out.append(cur)
    return out or [""]


def _is_number_list(v):
    return (isinstance(v, list) and v
            and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v))


def _is_record_list(v):
    return (isinstance(v, list) and v and all(isinstance(x, dict) for x in v)
            and len(v) >= 2)


def _render(obj, depth, width, key=None):
    ind = "  " * depth
    out = []
    if isinstance(obj, dict):
        if key is not None:
            out.append("{}{}:".format(ind[:-2], key))
        for k, v in obj.items():
            out += _render(v, depth + 1, width, key=str(k))
        return out
    if _is_number_list(obj):
        u = unit_of(key or "")
        head = "{}{}{}: [{} entries]".format(ind, strip_unit(key or "").replace("_", " "),
                                            " / " + u if u else "", len(obj))
        out.append(head)
        vals = [_fmt(x) for x in obj]
        line = ind + "  "
        for v in vals:
            if dwidth(line) + len(v) + 1 > width:
                out.append(line.rstrip())
                line = ind + "  "
            line += v + " "
        if line.strip():
            out.append(line.rstrip())
        return out
    if _is_record_list(obj):
        keys = []
        for r in obj:
            for k in r:
                if k not in keys and not isinstance(r[k], (dict, list)):
                    keys.append(k)
        if keys and len(keys) <= 10:
            out.append("{}{}: [{} rows]".format(ind, key, len(obj)))
            wid = [max(dwidth(strip_unit(k)), *(dwidth(_fmt(r.get(k))) for r in obj))
                   for k in keys]
            out.append(ind + "  " + "  ".join(
                rjust(strip_unit(k), wid[i]) for i, k in enumerate(keys)))
            out.append(ind + "  " + "  ".join(
                rjust(unit_of(k) or "", wid[i]) for i, k in enumerate(keys)))
            for r in obj:
                out.append(ind + "  " + "  ".join(
                    rjust(_fmt(r.get(k)), wid[i]) for i, k in enumerate(keys)))
            return out
    if isinstance(obj, list):
        out.append("{}{}: [{} items]".format(ind, key, len(obj)))
        for i, v in enumerate(obj[:50]):
            out += _render(v, depth + 1, width, key="[{}]".format(i))
        if len(obj) > 50:
            out.append("{}  ... the remaining {} items are in the JSON".format(
                ind, len(obj) - 50))
        return out
    # scalar
    u = unit_of(key or "")
    label = strip_unit(key or "").replace("_", " ")
    txt = _fmt(obj)
    if isinstance(obj, str) and dwidth(obj) > width - len(ind) - dwidth(label) - 6:
        out.append("{}{}:".format(ind, label))
        for line in _wrap(obj, width - len(ind) - 4):
            out.append(ind + "    " + line)
        return out
    pad = max(2, 46 - len(ind) - dwidth(label))
    out.append("{}{}{} {}{}".format(ind, label, "." * pad, txt,
                                    " " + u if u else ""))
    return out


def json_to_log(json_path, log_path=None, title=None):
    """Render any product JSON as a `.log`. **The general fallback path** -- a product
    with no report written specially for it is still readable."""
    json_path = Path(json_path)
    obj = json.loads(json_path.read_text(encoding="utf-8"))
    r = Report(title or json_path.stem,
               subtitle="stage 0 product report (generated from {})".format(json_path.name),
               source_note="this file is generated from the JSON record and is **not** "
                           "the source of truth; to change a number, change {} and rerun "
                           "the conversion.".format(json_path.name))
    r.json_dump(obj, title="record contents")
    return r.write(log_path or json_path.with_suffix(".log"))


# ======================================================================================
# parquet
# ======================================================================================
#: What worked on ln301, 2026-09-13. `mamba install --freeze-installed pyarrow=18.1.0`
#: was refused by the solver (aws-crt-cpp / libarrow-acero against the frozen set); the
#: pip wheel (25.0.1) installed in seconds and left numpy alone (probe: tree matches).
INSTALL_PARQUET = "pip install pyarrow      # on a login node; the conda solve was refused 2026-09-13"


def parquet_engine():
    """(engine name, version) pandas will write parquet with, or ImportError.

    Call it at the START of any driver whose last act is `write_parquet`. On an113
    (2026-09-13) the first branch B analysis on a card ran to completion -- every
    criterion judged, 10 s of work -- and then died on its final line because the
    `openqha-gpu` environment there had no pyarrow, although environment-tianhe-gpu.yml
    lists it. The check costs a millisecond here and the whole step at the other end.
    """
    import importlib
    import sys
    for name in ("pyarrow", "fastparquet"):
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        return name, getattr(mod, "__version__", "?")
    raise ImportError(
        "no parquet engine (pyarrow or fastparquet) in {}. This driver's product is a "
        "set of parquet tables; nothing it computed would be written. On a login node:"
        "\n    {}".format(sys.prefix, INSTALL_PARQUET))


def write_parquet(tables, stem):
    """`tables` is {table name: list of row dicts}. Each table is written to one
    `<stem>__<table name>.parquet`.

    **Why one file per table rather than one multi-table file**: parquet is a single-table
    format; several tables mean either a directory or hive partitioning. One table per
    file is the simplest, and the easiest to read straight into `pandas.read_parquet`.
    """
    import pandas as pd
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for name, rows in tables.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        p = stem.parent / "{}__{}.parquet".format(stem.name, name)
        df.to_parquet(p, index=False)
        out.append((p, len(df), list(df.columns)))
    return out
