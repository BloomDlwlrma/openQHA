"""Filing of analysis artifacts: identity is a REQUIRED argument at write time.

Why this module exists
----------------------
The problem in `analysis/` was never "nobody tidied the folder". It was that
**what a product IS was never forced to be written down, and nothing ever checked
it.** A survey on 2026-09-03 found 78 top-level entries, 12 388 files, 188 MB, with
retired routes lying flat beside live ones, two generations of on-disk format
(4439 JSON against 783 parquet), and metadata present in about half the files and
read by nothing:

    committee_calibration.json    generated_by yes, status no
    symmetry_backtest.json        generated_by yes, status no
    frequency_benchmark.json      generated_by yes, status YES
    engine_provenance.json        neither
    package1_summary_1_4000.json  neither

The repository's own contrast settles the remedy. `record.py` has a four-directory
convention (out/log/mol/xyz) and it has never drifted; `analysis/` had no convention
and grew to 78 top-level entries. **Tidying a directory is a one-off; a write-time
rule applies on every run.** So identity becomes a mandatory keyword argument here,
and the caller never chooses a path.

The discipline is not new in this repository -- it is the missing-symmetry-number
refusal applied to filing, and the no-literal-paths rule applied to destinations.

What is enforced, and how each rule can fail
--------------------------------------------
1. `category`, `status`, `produced_by` are keyword-only and have no defaults.
   Omitting one is a TypeError from Python itself, before any file is touched.
2. `category` and `status` are checked against closed vocabularies. An unknown
   value raises; there is no silent pass-through.
3. `produced_by` must name a file that EXISTS under the repository root. A product
   whose producing script cannot be found is a product nobody can reproduce, so it
   is refused. This is the rule that keeps `analysis/` tied to `scripts/`.
4. `decision`, when given, must look like `S0-*` or `D0-*`.
5. A DataFrame is never written as JSON.
6. Writing over an existing artifact requires `exist_ok=True`. Silent overwrite is
   how measurements disappear.

`status="unknown"` is deliberately allowed. It is for artifacts inherited from
before this module existed, and the INDEX lists them separately with a count --
per the design's criterion 3, unknown is not the same as absent (write down what
you cannot source, do not invent it).
"""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from .. import S0_ROOT, __version__

#: Mirrors the `scripts/` taxonomy exactly. A product is filed by the category of
#: the script that produced it -- one taxonomy, not two.
CATEGORIES = ("production", "calibration", "diagnostics", "raw")

#: What the artifact IS, which is a different axis from where it lives.
STATUSES = ("production", "calibration", "test", "superseded", "invalid", "unknown")

#: Directory for each category. `raw` and the superseded override use leading
#: underscores so they sort away from the live categories in a plain listing.
_DIR_OF_CATEGORY = {
    "production": "production",
    "calibration": "calibration",
    "diagnostics": "diagnostics",
    "raw": "_raw",
}
_SUPERSEDED_DIR = "_superseded"

_DECISION_RE = re.compile(r"^(?:S0|D0)-[A-Z0-9]+(?:-[A-Z0-9]+)*$")

SCHEMA_VERSION = 1

ANALYSIS = S0_ROOT / "analysis"


class ArtifactError(ValueError):
    """Refusal to write an artifact whose identity is incomplete or inconsistent."""


# ======================================================================================
# 1. Validation -- every branch here is a way for a write to fail loudly
# ======================================================================================
def _check_category(category):
    if category not in CATEGORIES:
        raise ArtifactError(
            "category={!r} is not one of {}. The categories mirror scripts/; if this "
            "product does not fit one of them, the producing script does not fit "
            "either.".format(category, CATEGORIES))
    return category


def _check_status(status):
    if status not in STATUSES:
        raise ArtifactError(
            "status={!r} is not one of {}. Use the value unknown only for artifacts "
            "inherited from before this module existed.".format(status, STATUSES))
    return status


def _check_produced_by(produced_by):
    """Return `produced_by` normalised to a repo-relative POSIX path, or raise.

    A product whose producing script is not on disk cannot be reproduced, so it is
    refused rather than filed. This is what ties `analysis/` back to `scripts/`.
    """
    if not produced_by:
        raise ArtifactError("produced_by is required and must name a file in this repo")
    p = Path(produced_by)
    candidate = p if p.is_absolute() else (S0_ROOT / p)
    if not candidate.is_file():
        raise ArtifactError(
            "produced_by={!r} does not exist (looked at {}). Pass the path of the "
            "script that is writing this artifact, relative to the repository root, "
            "for example scripts/calibration/s0_engine_benchmark.py".format(
                produced_by, candidate))
    try:
        return candidate.resolve().relative_to(S0_ROOT.resolve()).as_posix()
    except ValueError:
        raise ArtifactError(
            "produced_by={!r} resolves outside the repository ({})".format(
                produced_by, candidate))


def _check_decision(decision):
    if decision is None:
        return None
    if not _DECISION_RE.match(decision):
        raise ArtifactError(
            "decision={!r} does not look like a decision identifier. Expected the "
            "shape of S0-D-10 or D0-C-21.".format(decision))
    return decision


def _check_name(name):
    if not name or Path(name).name != name:
        raise ArtifactError(
            "name={!r} must be a bare file name without directories -- this module "
            "decides the directory, the caller does not.".format(name))
    return name


# ======================================================================================
# 2. Destination -- chosen here, never by the caller
# ======================================================================================
def destination(name, *, category, status, subdir=None, suffix=""):
    """Where an artifact with this identity belongs. Pure; touches no disk."""
    _check_name(name)
    _check_category(category)
    _check_status(status)
    top = _SUPERSEDED_DIR if status == "superseded" else _DIR_OF_CATEGORY[category]
    out = ANALYSIS / top
    if subdir:
        sub = Path(subdir)
        if sub.is_absolute() or ".." in sub.parts:
            raise ArtifactError("subdir={!r} must be a relative path inside the "
                                "category directory".format(subdir))
        out = out / sub
    return out / (name + suffix)


def _meta(name, category, status, produced_by, decision):
    return {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "category": category,
        "status": status,
        "generated_by": produced_by,
        "decision": decision,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "openqha_version": __version__,
    }


# ======================================================================================
# 3. The write
# ======================================================================================
_SUFFIX_OF_KIND = {"json": ".json", "parquet": ".parquet", "text": ".log"}


def _shown(path):
    """Repo-relative path for messages, falling back to the absolute one.

    `ANALYSIS` is a rebindable module constant, so a destination can legitimately
    sit outside the repository (a test redirects it to a temporary directory). A
    bare `relative_to` would then raise ValueError while BUILDING an error message,
    which replaces a clear refusal with a confusing traceback -- this is the
    lesson in miniature: an error path is only verified under the caller's
    real conditions.
    """
    try:
        return str(Path(path).relative_to(S0_ROOT))
    except ValueError:
        return str(path)


def write_artifact(payload, name, *, category, status, produced_by,
                   decision=None, subdir=None, exist_ok=False):
    """File one artifact and return its path. All identity fields are required.

    `payload` decides the format, and the mapping is not negotiable:

        dict / list           -> `<name>.json`, metadata at the top level,
                                 content under the "data" key
        pandas.DataFrame      -> `<name>.parquet`, metadata in the parquet schema
        str                   -> `<name>.log`, metadata in a leading comment block

    A DataFrame is refused as JSON: per-molecule products are written as `.log`
    (human) plus `.parquet` (machine), never JSON. Nesting the content under
    "data" keeps the metadata keys from ever colliding with it.
    """
    _check_name(name)
    category = _check_category(category)
    status = _check_status(status)
    produced_by = _check_produced_by(produced_by)
    decision = _check_decision(decision)
    meta = _meta(name, category, status, produced_by, decision)

    kind = _kind_of(payload)
    path = destination(name, category=category, status=status, subdir=subdir,
                       suffix=_SUFFIX_OF_KIND[kind])
    if path.exists() and not exist_ok:
        raise ArtifactError(
            "{} already exists. Pass exist_ok=True to replace it deliberately; a "
            "silent overwrite is how a measurement disappears.".format(_shown(path)))
    path.parent.mkdir(parents=True, exist_ok=True)

    if kind == "json":
        doc = dict(meta)
        doc["data"] = payload
        path.write_text(json.dumps(doc, indent=2, ensure_ascii=False,
                                   default=_json_default) + "\n", encoding="utf-8")
    elif kind == "parquet":
        _write_parquet(payload, path, meta)
    else:
        header = "".join("# {}: {}\n".format(k, v) for k, v in meta.items())
        path.write_text(header + "#\n" + payload, encoding="utf-8")
    return path


def _kind_of(payload):
    if isinstance(payload, str):
        return "text"
    if _is_dataframe(payload):
        return "parquet"
    if isinstance(payload, (dict, list, tuple)):
        return "json"
    raise ArtifactError(
        "payload of type {} cannot be filed. Pass a dict/list (JSON), a "
        "pandas.DataFrame (parquet), or a str (log).".format(type(payload).__name__))


def _is_dataframe(obj):
    return type(obj).__name__ == "DataFrame" and hasattr(obj, "to_parquet")


def _json_default(obj):
    """Numpy scalars and arrays are common in this repository; keep full precision."""
    if hasattr(obj, "item") and getattr(obj, "ndim", None) == 0:
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError("not JSON serialisable: {!r}".format(type(obj)))


def _write_parquet(df, path, meta):
    """Parquet plus metadata carried in the file's own schema, not a sidecar."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.Table.from_pandas(df, preserve_index=False)
    merged = dict(table.schema.metadata or {})
    merged[b"openqha"] = json.dumps(meta).encode("utf-8")
    pq.write_table(table.replace_schema_metadata(merged), path)


# ======================================================================================
# 4. Reading identity back -- what the INDEX generator uses
# ======================================================================================
def read_meta(path):
    """Identity block of an artifact, or None if it carries none.

    Returns None rather than raising, because most of `analysis/` predates this
    module; the INDEX reports those as status unknown instead of guessing.
    """
    path = Path(path)
    try:
        if path.suffix == ".json":
            doc = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(doc, dict) and doc.get("schema_version") and "generated_by" in doc:
                return {k: v for k, v in doc.items() if k != "data"}
            return None
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq
            raw = (pq.read_schema(path).metadata or {}).get(b"openqha")
            return json.loads(raw.decode("utf-8")) if raw else None
        if path.suffix == ".log":
            meta = {}
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.startswith("#"):
                    break
                if ": " in line:
                    k, v = line[1:].split(": ", 1)
                    meta[k.strip()] = v.strip()
            return meta if meta.get("schema_version") else None
    except Exception:
        return None
    return None


def read_artifact(path):
    """Payload of a JSON artifact written by `write_artifact` (metadata stripped)."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(doc, dict) and "data" in doc and doc.get("schema_version"):
        return doc["data"]
    return doc
