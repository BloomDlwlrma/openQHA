"""stage 0 configuration loading -- **this repository's own source of truth, reading no other stage**.

Since 2026-08-28 stage 0 is an independent open framework. The edge set, species
table, symmetry numbers, parameters and data paths all come from
`configs/openqha.yaml`; anything copied from elsewhere records its source,
the date it was copied, and a checksum, so that drift is discovered rather than
inherited.

Path resolution order (first match wins):
    1. environment variables (`S0_CONFIG`, `S0_QM9_ROOT`, `S0_MACE_MODEL`)
    2. relative paths in the configuration file, taken relative to **this repository's root**
"""
import hashlib
import os
from pathlib import Path

from . import S0_ROOT

DEFAULT_CONFIG = S0_ROOT / "configs" / "openqha.yaml"

_CACHE = {}


def load(path=None):
    """Read the configuration. Each path is read once."""
    p = Path(path or os.environ.get("S0_CONFIG", str(DEFAULT_CONFIG)))
    if not p.exists():
        raise FileNotFoundError(
            "stage 0 configuration not found: {}\n"
            "stage 0 does not borrow configuration from another stage -- fix the path "
            "or set the environment variable S0_CONFIG.".format(p))
    key = str(p.resolve())
    if key not in _CACHE:
        import yaml
        cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
        cfg["_path"] = str(p)
        _CACHE[key] = _resolve_includes(cfg, p)
    return _CACHE[key]


def _resolve_includes(cfg, path):
    """Merge the files named in `include:` into `cfg`.

    Sections are MOVED into their own files, never copied, so an included file is the
    sole definition of the keys it carries. A key defined in both is therefore a real
    mistake and raises here rather than being resolved by precedence -- silent
    precedence is how a run ends up using settings nobody chose.

    Includes are resolved relative to the including file and are not recursive: one
    level keeps "which file defines this key" answerable by looking, not by tracing.
    """
    import yaml
    names = cfg.pop("include", None)
    if not names:
        return cfg
    cfg.setdefault("_includes", [])
    for name in names:
        sub_path = (Path(path).parent / name).resolve()
        if not sub_path.exists():
            raise FileNotFoundError(
                "{} includes {!r}, which does not exist at {}".format(
                    path, name, sub_path))
        sub = yaml.safe_load(sub_path.read_text(encoding="utf-8")) or {}
        clash = sorted(set(sub) & set(cfg) - {"_includes"})
        if clash:
            raise ValueError(
                "{} and {} both define {}. Included sections are MOVED, not copied -- "
                "delete one of the two definitions rather than relying on which file "
                "is read last.".format(path.name, name, ", ".join(clash)))
        cfg.update(sub)
        cfg["_includes"].append(str(sub_path))
    return cfg


def edges(cfg=None):
    cfg = cfg or load()
    e = cfg["edges"]
    if not e:
        raise ValueError("edges in the configuration is empty -- refusing to run")
    if len(set(e)) != len(e):
        raise ValueError("edges in the configuration has duplicates: {}".format(e))
    return list(e)


def edge_list_sha256(cfg=None):
    """Checksum of the edge set.

    Once copied from stage 2, this file IS the source of truth; this number is what
    reveals any later drift.
    """
    return hashlib.sha256("\n".join(edges(cfg)).encode("utf-8")).hexdigest()


def edge_species(edge):
    """'C3H6O1N0_18_35' -> ('dsgdb9nsd_000018', 'dsgdb9nsd_000035')."""
    parts = edge.split("_")
    if len(parts) < 3:
        raise ValueError("cannot parse species out of edge name: {!r}".format(edge))
    return tuple("dsgdb9nsd_{:06d}".format(int(p)) for p in parts[1:])


def species(qm9_index, cfg=None):
    """The full declaration of one species.

    **A missing symmetry number or electronic degeneracy raises; there is no default**
    (D0-9).
    """
    cfg = cfg or load()
    s = cfg["species"].get(qm9_index)
    if s is None:
        raise KeyError("species not declared in the configuration: {} -- refusing to "
                       "run".format(qm9_index))
    for k in ("symmetry_number", "electronic_degeneracy"):
        if s.get(k) is None:
            raise KeyError("{} has no {} declared -- refusing to run".format(qm9_index, k))
    return s


def qm9_root(cfg=None):
    cfg = cfg or load()
    p = Path(os.environ.get("S0_QM9_ROOT", "")) if os.environ.get("S0_QM9_ROOT") \
        else S0_ROOT / cfg["data"]["qm9_root"]
    return p


def qm9_index_csv(cfg=None):
    cfg = cfg or load()
    p = qm9_root(cfg) / cfg["data"]["qm9_index_csv"]
    if not p.exists():
        raise FileNotFoundError(
            "QM9 index table not found: {}\n"
            "What stage 0 carries itself is only the reference geometries of the 7 "
            "target species ({}).\n"
            "For the full data set run:  python scripts/tooling/s0_prepare_data.py "
            "--from <some QM9 directory>\n"
            "or set the environment variable S0_QM9_ROOT to point at an existing "
            "copy.".format(p, cfg["data"]["vendored_reference_geometries"]))
    return p


def qm9_xyz(qm9_index, cfg=None):
    """A reference geometry.

    **The 7 geometries shipped with the repository are looked at first, the full data
    set second** -- which is what lets package 2 run with no external data at all.
    """
    cfg = cfg or load()
    vend = S0_ROOT / cfg["data"]["vendored_reference_geometries"] / (qm9_index + ".xyz")
    if vend.exists():
        return vend
    p = qm9_root(cfg) / cfg["data"]["qm9_xyz_dir"] / (qm9_index + ".xyz")
    if p.exists():
        return p

    # curatedQM9. Under f7_mode = "curated" a repaired geometry is the one we WANT for
    # a molecule on the uncharacterized list; for everything else the archive simply
    # carries QM9's own file, which is equally usable. Either way `qm9_xyz_source`
    # below records which of the three naming patterns matched.
    from . import curated_qm9
    cur, kind = curated_qm9.find(qm9_index, cfg)
    if cur is not None:
        return cur

    raise FileNotFoundError(
        "reference geometry not found for {}.\n"
        "  looked in: {}\n"
        "             {}\n"
        "             curatedQM9 at {}\n"
        "The repository ships only the 7 stage-0 species (D0-41). For anything else "
        "place QM9 with scripts/tooling/s0_prepare_data.py, or unpack curatedQM9 "
        "under data/qm9/{}.".format(
            qm9_index, vend, p, curated_qm9.root(cfg),
            curated_qm9.DEFAULT_DIRNAME))


def qm9_smiles_from_xyz(path):
    """Read the SMILES out of a QM9 extended-xyz file.

    QM9's format puts, after the N atom lines: a harmonic-frequency line, then a line
    with TWO SMILES (the GDB-17 one and the one parsed back from the relaxed geometry),
    then a line with two InChI. curatedQM9 keeps that layout in both its naming
    schemes, so this works on repaired and original files alike.

    Returns (gdb17_smiles, relaxed_smiles). **They are not always the same molecule** --
    that disagreement is precisely what the uncharacterized list is about, and F7 is
    the gate that acts on it. Returning both keeps the disagreement visible instead of
    silently picking one.

    Why this exists: it removes the 119 MB index CSV from the dependency chain for
    anything that only needs a SMILES. A molecule's own geometry file already carries
    it, and one fewer required download is one fewer way for a run to be impossible.
    """
    lines = [l for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    n = int(lines[0].split()[0])
    # atoms occupy lines[2 : 2+n]; then frequencies; then SMILES; then InChI
    idx = 2 + n + 1
    if idx >= len(lines):
        raise ValueError("{} has no SMILES line (only {} non-empty lines for {} "
                         "atoms)".format(path, len(lines), n))
    parts = lines[idx].split()
    if not parts:
        raise ValueError("{}: the SMILES line is empty".format(path))
    gdb17 = parts[0]
    relaxed = parts[1] if len(parts) > 1 else parts[0]
    return gdb17, relaxed


def read_qm9_xyz(path):
    """Read the FIRST frame of a QM9 extended-xyz file as an ase.Atoms.

    Why not `ase.io.read`: QM9 puts three property lines AFTER the atoms (harmonic
    frequencies, two SMILES, two InChI). ASE reads the first frame, then tries to parse
    the frequency line as the next frame's atom count and raises XYZError. The seven
    geometries shipped with this repository had those lines stripped, so the failure
    only appears once a geometry comes from QM9 or curatedQM9 -- that is, for every
    molecule the repository does not ship.

    Also handles QM9's Fortran-style exponents (`-0.535689*^-6`), which float() rejects.
    """
    from ase import Atoms

    lines = Path(path).read_text(encoding="utf-8").splitlines()
    n = int(lines[0].split()[0])
    symbols, positions = [], []
    for row in lines[2:2 + n]:
        f = row.split()
        symbols.append(f[0])
        positions.append([float(x.replace("*^", "e")) for x in f[1:4]])
    return Atoms(symbols=symbols, positions=positions)


def qm9_smiles(qm9_index, cfg=None, prefer="relaxed"):
    """SMILES for a QM9 index, from the index table if present, else from the geometry.

    `prefer` selects which of the two the file carries. `relaxed` is the default
    because it is the SMILES parsed back from the deposited geometry, and the geometry
    is what we actually compute on.
    """
    cfg = cfg or load()
    try:
        return qm9_row(qm9_index, cfg)["qm9_smiles"]
    except (KeyError, FileNotFoundError):
        pass
    gdb17, relaxed = qm9_smiles_from_xyz(qm9_xyz(qm9_index, cfg))
    return relaxed if prefer == "relaxed" else gdb17


def qm9_xyz_source(qm9_index, cfg=None):
    """Where a geometry came from, as a string for the product record.

    One of: `vendored` (the 7 shipped species), `qm9_native`, `curatedQM9:original`,
    `curatedQM9:repaired_qm9_tag`, `curatedQM9:repaired_short_tag`.

    **This is not decoration.** A repaired geometry is a different structure from the
    deposited one, and a product that does not say which it used cannot be compared
    with one that used the other.
    """
    cfg = cfg or load()
    vend = (S0_ROOT / cfg["data"]["vendored_reference_geometries"]
            / (qm9_index + ".xyz"))
    if vend.exists():
        return "vendored"
    p = qm9_root(cfg) / cfg["data"]["qm9_xyz_dir"] / (qm9_index + ".xyz")
    if p.exists():
        return "qm9_native"
    from . import curated_qm9
    _cur, kind = curated_qm9.find(qm9_index, cfg)
    if kind is not None:
        return "curatedQM9:" + kind
    return "not_found"


def qm9_row(qm9_index, cfg=None):
    """The row for one species in the QM9 index table.

    **The 7 rows shipped with the repository are searched first, the full index
    second** -- which is what lets package 2 run with no external data at all. The
    excerpt was taken verbatim, so its columns match the full table exactly.
    """
    import csv
    cfg = cfg or load()
    vend = (S0_ROOT / cfg["data"]["vendored_reference_geometries"]
            / "qm9_reference_rows.csv")
    for src in (vend, None):
        if src is None:
            src = qm9_index_csv(cfg)   # raises on its own when absent, with the fix in it
        elif not src.exists():
            continue
        with src.open(encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                if r["qm9_index"] == qm9_index:
                    return r
    raise KeyError("{} is not in the QM9 index -- neither in the excerpt shipped with "
                   "the repository nor in the full table".format(qm9_index))


def package(n, cfg=None):
    cfg = cfg or load()
    key = "package{}".format(n)
    if key not in cfg:
        raise KeyError("the configuration has no {}".format(key))
    return cfg[key]


def temperature(cfg=None):
    return float((cfg or load())["thermodynamics"]["temperature_K"])


def pressure(cfg=None):
    return float((cfg or load())["thermodynamics"]["pressure_Pa"])

def runs_root(cfg=None):
    """The **single root** under which this repository writes on a machine.

    User ruling 2026-08-31: do not write straight into the home directory. The
    environment variable `S0_RUNS_ROOT` overrides the configuration. The directory is
    created if it does not exist, so a caller always receives a writable directory and
    none of them has to mkdir for itself.
    """
    cfg = cfg or load()
    raw = os.environ.get("S0_RUNS_ROOT") or cfg.get("runtime", {}).get(
        "runs_root", "~/runs/openQHA")
    p = Path(raw).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def runs_dir(name, cfg=None):
    """One subdirectory under `runs_root`, created and returned."""
    p = runs_root(cfg) / name
    p.mkdir(parents=True, exist_ok=True)
    return p
