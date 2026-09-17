"""Ticket 30: is this molecule in MACE-OFF23's training set?

THE QUESTION
------------
The released MACE-OFF23 data (Moore et al., Apollo doi:10.17863/CAM.107498) is the
SPICE-derived pair `train_large_neut_no_bad_clean.xyz` (951,005 frames, 17,132
molecules, ~55 conformers each) and `test_large_neut_all.xyz` (50,195 frames; the split
is by FRAME, so 15,542 of the molecules appear in both). Energies and forces only. If a
molecule is in it, MACE has seen its conformers, and every "model error" tier of
`level_compare` is an in-distribution number. That has to be on the record next to the
tier, or a reader takes interpolation for generalisation.

IDENTITY, NOT STRING MATCH
--------------------------
The files' `smiles` are atom-mapped with explicit hydrogens
(`[H:5][C:1](=[C:2](...)...)`); QM9's are plain. Both sides are canonicalised with RDKit
(atom maps cleared, explicit H removed) and compared at three declared strictnesses:

    isomeric       canonical SMILES with stereo            the strictest
    no_stereo      canonical SMILES, stereo removed        2-methyloxirane R == S
    connectivity   InChIKey connectivity block             tautomer / charge insensitive

The strictest level at which a match exists is the answer (`MATCH_LEVEL`); a molecule
absent at all three is `none`. A SPICE frame may hold two fragments (`DES370K Dimers`):
a fragment match counts separately (`N_*_DIMER_FRAMES`) from a monomer frame
(`N_*_FRAMES`), because a dimer frame teaches the interaction, not the conformer.

THE INDEX
---------
The two files are read once (5.2 GB; the 17k distinct SMILES strings are parsed once
each) into a table `<cache>/mace-off23_spice_index.dat` keyed by canonical identity, with
frame counts per file and per `config_type`, and a `[Source]` record of the files' sizes,
mtimes and frame counts; a changed source invalidates the cache. A query is then a
dictionary lookup. The table is the repository's `.dat` form (ADR 0003), not parquet:
parquet is git-ignored here and needs an engine, and the index is what ticket 31 runs on.

Configuration (`configs/openqha.yaml`, `data.training_sets.mace_off23`): the root that
holds the two files (overridable by S0_TRAINING_SETS_ROOT), their names, the DOI, and
the cache directory (`data/training_sets`, in the repository).
"""
import os
import re
import time
from pathlib import Path

from .. import config
from ..store import dat

SOURCE_NAME = "mace-off23_spice"
MATCH_LEVELS = ("isomeric", "no_stereo", "connectivity")
INDEX_STEM = "mace-off23_spice_index"
_SMILES_RE = re.compile(r'smiles="([^"]*)"')
_CONFIG_RE = re.compile(r'config_type="([^"]*)"')

INDEX_SCHEMA = {
    "Index": {
        "KEY": ("String", None, "canonical identity at LEVEL"),
        "LEVEL": ("String", None, "isomeric, no_stereo or connectivity"),
        "FILE": ("String", None, "train or test"),
        "CONFIG_TYPE": ("String", None, "the frame's config_type ('' when absent)"),
        "N_FRAMES": ("Integer", None, "monomer frames whose whole molecule is this identity"),
        "N_DIMER_FRAMES": ("Integer", None, "multi-fragment frames holding this identity as one fragment"),
    },
    "Source": {
        "FILE": ("String", None, "train or test"),
        "PATH": ("String", None, "the file read"),
        "SIZE": ("Integer", "bytes", "size when indexed"),
        "MTIME": ("Double", "s", "modification time when indexed"),
        "N_FRAMES_TOTAL": ("Integer", None, "frames in the file"),
        "N_DISTINCT_SMILES": ("Integer", None, "distinct smiles strings in the file"),
        "N_UNPARSED": ("Integer", None, "smiles strings RDKit could not parse"),
        "SECONDS": ("Double", "s", "wall time of the pass"),
        "DOI": ("String", None, "the data set's DOI"),
    },
}


# ====================================================================== identity
def canonical_keys(smiles):
    """{level: key} for a SMILES at the three strictnesses, or None when RDKit cannot
    parse it. Atom-map numbers are cleared and explicit hydrogens removed first, so the
    SPICE form and the plain form of one molecule give one key."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    for a in mol.GetAtoms():
        a.SetAtomMapNum(0)
    mol = Chem.RemoveHs(mol)
    iso = Chem.MolToSmiles(mol, isomericSmiles=True)
    flat = Chem.Mol(mol)
    Chem.RemoveStereochemistry(flat)
    no_stereo = Chem.MolToSmiles(flat, isomericSmiles=False)
    try:
        conn = Chem.MolToInchiKey(mol).split("-")[0]
    except Exception:                                    # noqa: BLE001 -- InChI refuses some inputs
        conn = "INCHI_FAILED:" + no_stereo
    return {"isomeric": iso, "no_stereo": no_stereo, "connectivity": conn}


def fragment_keys(smiles):
    """The per-fragment keys of a (possibly multi-fragment) SMILES: a list with one
    `canonical_keys` dict per fragment (None for an unparsable fragment)."""
    return [canonical_keys(f) for f in smiles.split(".")]


# ====================================================================== configuration
def settings(cfg=None):
    cfg = cfg or config.load()
    ts = (cfg.get("data") or {}).get("training_sets") or {}
    s = dict(ts.get("mace_off23") or {})
    if os.environ.get("S0_TRAINING_SETS_ROOT"):
        s["root"] = os.environ["S0_TRAINING_SETS_ROOT"]
    root = Path(s.get("root", "data/training_sets/mace-off23_spice")).expanduser()
    if not root.is_absolute():
        root = config.S0_ROOT / root
    cache = Path(s.get("cache_dir", "data/training_sets"))
    if not cache.is_absolute():
        cache = config.S0_ROOT / cache
    return dict(root=root, cache=cache,
                train=root / s.get("train_file", "train_large_neut_no_bad_clean.xyz"),
                test=root / s.get("test_file", "test_large_neut_all.xyz"),
                doi=s.get("doi", "10.17863/CAM.107498"))


def index_path(cfg=None):
    return settings(cfg)["cache"] / (INDEX_STEM + ".dat")


# ====================================================================== the index
def _stream_headers(path):
    """Yield (smiles, config_type) per frame of an extxyz file, reading only headers."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        while True:
            line = fh.readline()
            if not line:
                return
            s = line.strip()
            if not s:
                continue
            try:
                nat = int(s.split()[0])
            except ValueError:
                raise ValueError("{}: expected an atom count, read {!r}".format(path, s[:40]))
            hdr = fh.readline()
            m = _SMILES_RE.search(hdr)
            c = _CONFIG_RE.search(hdr)
            yield (m.group(1) if m else ""), (c.group(1) if c else "")
            for _ in range(nat):
                fh.readline()


def build_index(cfg=None, files=None, cache_path=None, doi=None):
    """Read the training and test files once and write the index table. `files` may
    override `{"train": path, "test": path}` (the unit test's tiny fixture). Returns the
    path written."""
    s = settings(cfg)
    files = files or {"train": s["train"], "test": s["test"]}
    cache_path = Path(cache_path or index_path(cfg))
    doi = doi or s["doi"]
    counts = {}            # (key, level, file, config_type) -> [monomer, dimer]
    sources = []
    for tag in ("train", "test"):
        path = Path(files[tag])
        if not path.is_file():
            raise FileNotFoundError("training-set file not found: {} (set data.training_sets."
                                    "mace_off23.root or S0_TRAINING_SETS_ROOT)".format(path))
        t0 = time.time()
        keys_of = {}
        n_frames = n_unparsed = 0
        for smiles, ctype in _stream_headers(path):
            n_frames += 1
            if smiles not in keys_of:
                frags = fragment_keys(smiles) if smiles else []
                if any(f is None for f in frags):
                    n_unparsed += 1
                keys_of[smiles] = [f for f in frags if f is not None]
            frags = keys_of[smiles]
            dimer = len(frags) > 1
            for fk in frags:
                for level, key in fk.items():
                    slot = counts.setdefault((key, level, tag, ctype), [0, 0])
                    slot[1 if dimer else 0] += 1
        st = path.stat()
        sources.append({"FILE": tag, "PATH": str(path), "SIZE": int(st.st_size),
                        "MTIME": float(st.st_mtime), "N_FRAMES_TOTAL": n_frames,
                        "N_DISTINCT_SMILES": len(keys_of), "N_UNPARSED": n_unparsed,
                        "SECONDS": float(time.time() - t0), "DOI": doi})
    rows = [{"KEY": k, "LEVEL": lvl, "FILE": f, "CONFIG_TYPE": c, "N_FRAMES": v[0], "N_DIMER_FRAMES": v[1]}
            for (k, lvl, f, c), v in sorted(counts.items())]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    dat.write_tables(cache_path, {"Source": sources, "Index": rows}, INDEX_SCHEMA)
    return cache_path


def index_is_current(cfg=None, files=None, cache_path=None):
    """True when the cached index exists and its recorded sources match the files on
    disk (size and mtime). False when absent or stale; None when the sources themselves
    are absent (the check cannot be run here)."""
    s = settings(cfg)
    files = files or {"train": s["train"], "test": s["test"]}
    cache_path = Path(cache_path or index_path(cfg))
    if not cache_path.is_file():
        return False if all(Path(p).is_file() for p in files.values()) else None
    try:
        src = {r["FILE"]: r for r in dat.read_tables(cache_path).get("Source", [])}
    except (ValueError, KeyError):
        return False
    for tag, p in files.items():
        p = Path(p)
        if not p.is_file():
            # the sources moved away; the cached index still answers, and says so
            continue
        r = src.get(tag)
        st = p.stat()
        if r is None or int(r["SIZE"]) != st.st_size or abs(float(r["MTIME"]) - st.st_mtime) > 1.0:
            return False
    return True


_INDEX_CACHE = {}


def load_index(cfg=None, cache_path=None):
    """{level: {key: {file: {config_type: (n, n_dimer)}}}} plus the [Source] rows, from
    the cached table; built first when absent."""
    cache_path = Path(cache_path or index_path(cfg))
    key = str(cache_path)
    if key in _INDEX_CACHE and _INDEX_CACHE[key][0] == cache_path.stat().st_mtime:
        return _INDEX_CACHE[key][1]
    if not cache_path.is_file():
        build_index(cfg, cache_path=cache_path)
    tables = dat.read_tables(cache_path)
    idx = {lvl: {} for lvl in MATCH_LEVELS}
    for r in tables.get("Index", []):
        idx[r["LEVEL"]].setdefault(r["KEY"], {}).setdefault(r["FILE"], {})[r["CONFIG_TYPE"]] = (
            int(r["N_FRAMES"]), int(r["N_DIMER_FRAMES"]))
    out = dict(index=idx, sources=tables.get("Source", []), path=cache_path)
    _INDEX_CACHE[key] = (cache_path.stat().st_mtime, out)
    return out


# ====================================================================== the answer
def membership(smiles, cfg=None, cache_path=None):
    """The `[Training_Set]` answer for one SMILES: SOURCE, IN_TRAINING, MATCH_LEVEL (the
    strictest level with a match, or 'none'), frame counts per file (monomer and dimer),
    CONFIG_TYPES, N_HEAVY_ATOMS, and the per-level presence for the record."""
    keys = canonical_keys(smiles)
    if keys is None:
        raise ValueError("RDKit cannot parse the SMILES {!r}".format(smiles))
    ix = load_index(cfg, cache_path)["index"]
    out = {"SOURCE": SOURCE_NAME, "SMILES_CANONICAL": keys["isomeric"],
           "N_HEAVY_ATOMS": _n_heavy(keys["isomeric"]),
           "IN_TRAINING": False, "MATCH_LEVEL": "none",
           "N_TRAIN_FRAMES": 0, "N_TEST_FRAMES": 0,
           "N_TRAIN_DIMER_FRAMES": 0, "N_TEST_DIMER_FRAMES": 0, "CONFIG_TYPES": []}
    present = {}
    for level in MATCH_LEVELS:
        hit = ix[level].get(keys[level])
        present["MATCH_" + level.upper()] = hit is not None
        if hit is not None and out["MATCH_LEVEL"] == "none":
            out["MATCH_LEVEL"] = level
            ctypes = set()
            for tag, by_type in hit.items():
                for ctype, (n, nd) in by_type.items():
                    out["N_{}_FRAMES".format(tag.upper())] += n
                    out["N_{}_DIMER_FRAMES".format(tag.upper())] += nd
                    ctypes.add(ctype or "untagged")
            out["CONFIG_TYPES"] = sorted(ctypes)
    out.update(present)
    out["IN_TRAINING"] = bool(out["N_TRAIN_FRAMES"] > 0 or out["N_TRAIN_DIMER_FRAMES"] > 0)
    out["IN_TEST_ONLY"] = bool(not out["IN_TRAINING"] and (out["N_TEST_FRAMES"] + out["N_TEST_DIMER_FRAMES"]) > 0)
    return out


def _n_heavy(canonical_smiles):
    from rdkit import Chem
    m = Chem.MolFromSmiles(canonical_smiles)
    return int(m.GetNumHeavyAtoms()) if m is not None else -1


def sentence(rec):
    """The one line a Report prints beside the tiers."""
    if rec is None:
        return ("training-set membership not checked: no MACE-OFF23 (SPICE) index is available "
                "here (data.training_sets.mace_off23 / S0_TRAINING_SETS_ROOT)")
    if rec["IN_TRAINING"]:
        return ("this molecule IS in MACE-OFF23's training data ({} match: {} train frames, {} test "
                "frames, {} dimer frames; {}): the model-error tiers are in-distribution numbers"
                .format(rec["MATCH_LEVEL"], rec["N_TRAIN_FRAMES"], rec["N_TEST_FRAMES"],
                        rec["N_TRAIN_DIMER_FRAMES"] + rec["N_TEST_DIMER_FRAMES"],
                        ", ".join(rec["CONFIG_TYPES"])))
    if rec["IN_TEST_ONLY"]:
        return ("this molecule is in MACE-OFF23's TEST file only ({} test frames, {} match): held "
                "out from training, but drawn from the same distribution".format(rec["N_TEST_FRAMES"], rec["MATCH_LEVEL"]))
    return ("this molecule is NOT in MACE-OFF23's training or test data at any strictness: the "
            "model-error tiers are out-of-distribution numbers")
