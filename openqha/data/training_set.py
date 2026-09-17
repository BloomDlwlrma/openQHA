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


# ====================================================================== ticket 31: every target
TARGETS_STEM = "qm9_targets_membership"
TARGETS_PROGNAME = "openQHA training_set_membership"

TARGETS_SCHEMA = {
    "Calculation_Info": {
        "SOURCE": ("String", None, "the training data checked"),
        "DOI": ("String", None, "its DOI"),
        "INDEX_PATH": ("String", None, "the cached index the answers came from"),
        "TRAIN_FILE": ("String", None, "training file as recorded in the index"),
        "TRAIN_SIZE": ("Integer", "bytes", "its size when indexed"),
        "TRAIN_MTIME": ("Double", "s", "its mtime when indexed"),
        "TRAIN_FRAMES": ("Integer", None, "frames in the training file"),
        "TEST_FILE": ("String", None, "test file as recorded in the index"),
        "TEST_SIZE": ("Integer", "bytes", "its size when indexed"),
        "TEST_MTIME": ("Double", "s", "its mtime when indexed"),
        "TEST_FRAMES": ("Integer", None, "frames in the test file"),
        "QM9_ROOT": ("String", None, "the curated QM9 directory screened"),
        "SMILES_SOURCE": ("String", None, "which SMILES of the QM9 file was used: relaxed (parsed back from the geometry)"),
        "GATES": ("ArrayOfStrings", None, "the branch A gates applied, in order (configs/filters.yaml)"),
        "F7_MODE": ("String", None, "the F7 mode of the configuration"),
        "LIMIT": ("Integer", None, "molecules considered (0: all)"),
        "SECONDS": ("Double", "s", "wall time of the run"),
    },
    "Summary": {
        "N_CONSIDERED": ("Integer", None, "QM9 molecules read"),
        "N_TARGETS": ("Integer", None, "molecules passing every gate"),
        "N_FAILED_GATE": ("Integer", None, "molecules failing a gate"),
        "N_UNREADABLE": ("Integer", None, "QM9 files without a usable SMILES line"),
        "N_IN_TRAINING": ("Integer", None, "targets with frames in the training file, at the strictest matching level"),
        "N_IN_TRAINING_ISOMERIC": ("Integer", None, "targets matched at canonical isomeric SMILES"),
        "N_IN_TRAINING_NO_STEREO": ("Integer", None, "targets matched with stereo removed (isomeric or looser)"),
        "N_IN_TRAINING_CONNECTIVITY": ("Integer", None, "targets matched at the InChIKey connectivity block (any level)"),
        "N_IN_TEST_ONLY": ("Integer", None, "targets with frames in the test file only"),
        "N_MONOMER_ONLY": ("Integer", None, "targets in training whose frames are monomers only"),
        "N_DIMER_ONLY": ("Integer", None, "targets in training whose frames are dimers only"),
        "FRACTION_IN_TRAINING": ("Double", None, "N_IN_TRAINING / N_TARGETS"),
        "TARGETS_BY_HEAVY": ("ArrayOfIntegers", None, "targets with 1..9 heavy atoms"),
        "IN_TRAINING_BY_HEAVY": ("ArrayOfIntegers", None, "of those, in the training file"),
    },
    "Failed_Gate": {
        "GATE": ("String", None, "the gate"),
        "N": ("Integer", None, "molecules it refused"),
    },
    "Config_Type": {
        "CONFIG_TYPE": ("String", None, "a config_type tag of the matched frames"),
        "N_TARGETS": ("Integer", None, "targets with at least one frame under it"),
    },
}

TARGET_ROW_SCHEMA = {
    "qm9_index": ("String", None, "the molecule"),
    "smiles": ("String", None, "relaxed SMILES of the QM9 file"),
    "n_heavy": ("Integer", None, "heavy atoms"),
    "in_training": ("Boolean", None, "frames in the training file"),
    "in_test_only": ("Boolean", None, "frames in the test file only"),
    "match_level": ("String", None, "isomeric, no_stereo, connectivity or none"),
    "n_train_frames": ("Integer", None, "monomer frames, training file"),
    "n_test_frames": ("Integer", None, "monomer frames, test file"),
    "n_train_dimer_frames": ("Integer", None, "dimer frames, training file"),
    "n_test_dimer_frames": ("Integer", None, "dimer frames, test file"),
    "config_types": ("String", None, "config_type tags, ';'-joined"),
}


def applied_gates(cfg=None):
    """The gates `filters.screen` actually applies: the enabled ones, minus F7 when the
    F7 mode is off (screen drops it silently; the record must not list it)."""
    from ..conformer_search import filters
    cfg = cfg or config.load()
    gates = tuple(filters.enabled_gates(cfg))
    if "F7" in gates and filters.f7_mode(cfg) == "off":
        gates = tuple(g for g in gates if g != "F7")
    return gates


def qm9_targets(cfg=None, limit=None, progress=None):
    """Every curated QM9 molecule through the branch A gate. Yields
    (qm9_index, smiles, passed, failed_gate); a file without a usable SMILES line is
    yielded with smiles None. The SMILES is the file's own relaxed one (parsed back
    from the deposited geometry), which is what branch A computes on."""
    from ..conformer_search import filters
    from . import curated_qm9
    cfg = cfg or config.load()
    gates = applied_gates(cfg)
    index = curated_qm9._index(cfg)
    for k, n in enumerate(sorted(index)):
        if limit and k >= limit:
            return
        qid = "dsgdb9nsd_{:06d}".format(n)
        path = index[n][1]
        try:
            _gdb, relaxed = config.qm9_smiles_from_xyz(path)
        except (ValueError, IndexError):
            yield qid, None, False, "unreadable"
            continue
        passed, failed, _reason = filters.screen(relaxed, cfg=cfg, gates=gates, identifier=qid)
        if progress and k % 5000 == 0:
            progress(k, qid)
        yield qid, relaxed, bool(passed), failed


def qm9_targets_membership(cfg=None, limit=None, out_dir=None, progress=None):
    """Ticket 31: the Dataset `qm9_targets_membership.{dat,toml}` -- one row per target
    (every curated QM9 molecule passing the gate), the summary for the SI, the SPICE
    provenance and the gate configuration. Returns the record dict."""
    from ..conformer_search import filters
    from ..store import property as prop
    from . import curated_qm9
    cfg = cfg or config.load()
    s = settings(cfg)
    out_dir = Path(out_dir or s["cache"])
    ix = load_index(cfg)
    src = {r["FILE"]: r for r in ix["sources"]}
    t0 = time.time()
    rows, failed, n_unreadable, n_considered = [], {}, 0, 0
    by_heavy = [0] * 10
    in_by_heavy = [0] * 10
    ctype_count = {}
    for qid, smiles, passed, gate in qm9_targets(cfg, limit, progress):
        n_considered += 1
        if smiles is None:
            n_unreadable += 1
            continue
        if not passed:
            failed[gate] = failed.get(gate, 0) + 1
            continue
        r = membership(smiles, cfg)
        row = {"qm9_index": qid, "smiles": smiles, "n_heavy": r["N_HEAVY_ATOMS"],
               "in_training": r["IN_TRAINING"], "in_test_only": r["IN_TEST_ONLY"],
               "match_level": r["MATCH_LEVEL"], "n_train_frames": r["N_TRAIN_FRAMES"],
               "n_test_frames": r["N_TEST_FRAMES"], "n_train_dimer_frames": r["N_TRAIN_DIMER_FRAMES"],
               "n_test_dimer_frames": r["N_TEST_DIMER_FRAMES"], "config_types": ";".join(r["CONFIG_TYPES"]),
               "_match": {k: r[k] for k in ("MATCH_ISOMERIC", "MATCH_NO_STEREO", "MATCH_CONNECTIVITY")}}
        rows.append(row)
        h = min(max(row["n_heavy"], 0), 9)
        by_heavy[h] += 1
        if row["in_training"]:
            in_by_heavy[h] += 1
            for c in r["CONFIG_TYPES"]:
                ctype_count[c] = ctype_count.get(c, 0) + 1
    n_targets = len(rows)
    n_in = sum(1 for r in rows if r["in_training"])
    summary = {
        "N_CONSIDERED": n_considered, "N_TARGETS": n_targets,
        "N_FAILED_GATE": int(sum(failed.values())), "N_UNREADABLE": n_unreadable,
        "N_IN_TRAINING": n_in,
        "N_IN_TRAINING_ISOMERIC": sum(1 for r in rows if r["in_training"] and r["_match"]["MATCH_ISOMERIC"]),
        "N_IN_TRAINING_NO_STEREO": sum(1 for r in rows if r["in_training"] and r["_match"]["MATCH_NO_STEREO"]),
        "N_IN_TRAINING_CONNECTIVITY": sum(1 for r in rows if r["in_training"] and r["_match"]["MATCH_CONNECTIVITY"]),
        "N_IN_TEST_ONLY": sum(1 for r in rows if r["in_test_only"]),
        "N_MONOMER_ONLY": sum(1 for r in rows if r["in_training"] and r["n_train_frames"] > 0 and r["n_train_dimer_frames"] == 0),
        "N_DIMER_ONLY": sum(1 for r in rows if r["in_training"] and r["n_train_frames"] == 0 and r["n_train_dimer_frames"] > 0),
        "FRACTION_IN_TRAINING": (n_in / n_targets) if n_targets else 0.0,
        "TARGETS_BY_HEAVY": by_heavy[1:], "IN_TRAINING_BY_HEAVY": in_by_heavy[1:],
    }
    info = {"SOURCE": SOURCE_NAME, "DOI": s["doi"], "INDEX_PATH": str(ix["path"]),
            "TRAIN_FILE": src.get("train", {}).get("PATH"), "TRAIN_SIZE": src.get("train", {}).get("SIZE"),
            "TRAIN_MTIME": src.get("train", {}).get("MTIME"), "TRAIN_FRAMES": src.get("train", {}).get("N_FRAMES_TOTAL"),
            "TEST_FILE": src.get("test", {}).get("PATH"), "TEST_SIZE": src.get("test", {}).get("SIZE"),
            "TEST_MTIME": src.get("test", {}).get("MTIME"), "TEST_FRAMES": src.get("test", {}).get("N_FRAMES_TOTAL"),
            "QM9_ROOT": str(curated_qm9.root(cfg)), "SMILES_SOURCE": "relaxed",
            "GATES": list(applied_gates(cfg)), "F7_MODE": str(filters.f7_mode(cfg)),
            "LIMIT": int(limit or 0), "SECONDS": float(time.time() - t0)}
    blocks = {"Calculation_Info": info, "Summary": summary,
              "Failed_Gate": [{"GATE": g, "N": n} for g, n in sorted(failed.items())],
              "Config_Type": [{"CONFIG_TYPE": c, "N_TARGETS": n} for c, n in sorted(ctype_count.items())]}
    out_dir.mkdir(parents=True, exist_ok=True)
    missing = prop.write(out_dir / (TARGETS_STEM + ".toml"), blocks, TARGETS_SCHEMA,
                         prop.NORMAL_TERMINATION, TARGETS_PROGNAME)
    if missing:
        raise RuntimeError("{}.toml keys outside the schema: {}".format(TARGETS_STEM, missing))
    dat.write_table(out_dir / (TARGETS_STEM + ".dat"),
                    [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
                    columns=list(TARGET_ROW_SCHEMA), schema=TARGET_ROW_SCHEMA)
    return dict(info=info, summary=summary, failed=failed, config_types=ctype_count, rows=rows,
                record=out_dir / (TARGETS_STEM + ".toml"), table=out_dir / (TARGETS_STEM + ".dat"))
