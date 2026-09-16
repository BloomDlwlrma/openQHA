"""A Calculation's Property file: ORCA `.property.txt` carried into TOML (user ruling 2026-09-15).

ORCA leaves `base.out` for a person and `base.property.txt` for a program: `$Block`s of
`&KEY [&Type "Double", &Units "u"] value "doc"`, `$Calculation_Status` with
`&STATUS "NORMAL TERMINATION"`. This module writes the same thing as TOML, which the
standard library reads:

    [Calculation_Status]
    PROGNAME = "openQHA branchA"        # String: the step that wrote this file
    VERSION  = "0.3.0"                  # String: openQHA version
    STATUS   = "NORMAL TERMINATION"     # String: the completion marker

    [Calculation_Info]
    TSTEP    = 5.0                      # Double, fs: CREST MD time step

    [[Basin]]
    INDEX    = 0
    ENERGY   = -5259.18                 # Double, eV: electronic energy after tightening

The content rule (CONTEXT.md, "Property file"): the status block, the inputs, and the
result blocks a later step reads; nothing else. Prose, provenance and diagnostics belong
in the Report (`openqha.store.report`). The comments come from a SCHEMA the writer is
given, `{block: {KEY: (type, unit or None, doc)}}`, so every file of a step carries the
same comments; a key the schema does not know is written without a comment and returned
by `write()`, so the writer's test can catch it.

    write(path, blocks, schema, status, progname) -> [keys outside the schema]
    dumps(blocks, schema) -> (text, missing)
    load(path) -> dict                     (tomllib)
    status_of(path) -> STATUS or None      (None: no file, or no status block)

STATUS values: NORMAL_TERMINATION when the step finished; RUNNING when a trajectory
driver has equilibrated and may be resumed from the frames on disk; FAILED is what a
Batch prints for a Calculation whose file is absent and whose return code is not zero
(no step writes it).
"""
from pathlib import Path

from openqha import __version__ as VERSION
from openqha.store.toml_out import _inline, _key

NORMAL_TERMINATION = "NORMAL TERMINATION"
RUNNING = "RUNNING"
FAILED = "FAILED"

STATUS_BLOCK = "Calculation_Status"
INFO_BLOCK = "Calculation_Info"

#: The status block's own schema (every Property file carries it).
STATUS_SCHEMA = {
    "PROGNAME": ("String", None, "the step that wrote this file"),
    "VERSION": ("String", None, "openQHA version"),
    "STATUS": ("String", None, "the completion marker"),
}

_INDEX = ("Integer", None, "")


def describe(entry):
    """`Type, unit: doc` from a schema entry `(Type, unit or None, doc)`; `Type` alone when
    the doc is empty. The one spelling of a key's explanation, in a Property file's
    trailing comment and in a Table's column comment (`openqha.store.dat`)."""
    kind, unit, doc = (tuple(entry) + (None, ""))[:3]
    head = kind if not unit else "{}, {}".format(kind, unit)
    return "{}: {}".format(head, doc) if doc else head


def _comment(entry):
    return "# " + describe(entry)


def _emit_table(lines, block, table, schema, missing):
    """The key = value lines of one table, keys padded to one column, comments after."""
    items = [(k, v) for k, v in table.items() if v is not None]
    if not items:
        return
    width = max(len(_key(k)) for k, _ in items)
    known = schema.get(block, {}) if schema else {}
    rows = []
    for k, v in items:
        if isinstance(v, dict):
            raise TypeError("Property file block {!r}: key {!r} is a table; blocks are flat "
                            "(a repeated thing is an array of tables)".format(block, k))
        entry = known.get(k) or (_INDEX if k == "INDEX" else None)
        if entry is None and k != "INDEX":
            missing.append("{}.{}".format(block, k))
        rows.append(("{:<{w}} = {}".format(_key(k), _inline(v), w=width), entry))
    # The comment column: two spaces past the longest `key = value` line of the table,
    # capped so a long array does not push every comment off the screen.
    col = min(max(len(r) for r, _ in rows) + 2, 48)
    for row, entry in rows:
        if entry is None:
            lines.append(row)
        elif len(row) + 2 <= col:
            lines.append("{:<{c}}{}".format(row, _comment(entry), c=col))
        else:
            lines.append("{}  {}".format(row, _comment(entry)))


def dumps(blocks, schema=None):
    """`blocks` is an ordered dict `{block: table or [table, ...]}`. Returns (text, missing)."""
    if not isinstance(blocks, dict):
        raise TypeError("Property file blocks: give a dict of blocks, not {}".format(type(blocks).__name__))
    lines, missing = [], []
    for block, body in blocks.items():
        if isinstance(body, dict):
            lines += ["", "[{}]".format(_key(block))]
            _emit_table(lines, block, body, schema, missing)
        elif isinstance(body, (list, tuple)) and all(isinstance(x, dict) for x in body):
            for item in body:
                lines += ["", "[[{}]]".format(_key(block))]
                _emit_table(lines, block, item, schema, missing)
        else:
            raise TypeError("Property file block {!r} must be a table or a list of tables".format(block))
    return "\n".join(lines).lstrip("\n") + "\n", missing


def write(path, blocks, schema, status, progname):
    """Write `[Calculation_Status]` first, then `blocks`, atomically (`.part`, rename).
    Returns the keys the schema did not know (empty when every key has its comment)."""
    if status not in (NORMAL_TERMINATION, RUNNING):
        raise ValueError("STATUS must be {!r} or {!r}, not {!r}".format(NORMAL_TERMINATION, RUNNING, status))
    ordered = {STATUS_BLOCK: {"PROGNAME": str(progname), "VERSION": VERSION, "STATUS": status}}
    for block, body in blocks.items():
        if block == STATUS_BLOCK:
            raise ValueError("the status block is written by write(); do not pass one")
        ordered[block] = body
    full_schema = dict(schema or {})
    full_schema[STATUS_BLOCK] = STATUS_SCHEMA
    text, missing = dumps(ordered, full_schema)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return missing


def load(path):
    import tomllib
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def status_of(path):
    """STATUS in the file's `[Calculation_Status]`, or None (no file, unreadable, no block)."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        doc = load(path)
    except Exception:
        return None
    block = doc.get(STATUS_BLOCK)
    if not isinstance(block, dict):
        return None
    return block.get("STATUS")
