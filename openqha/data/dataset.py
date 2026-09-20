"""The Dataset of the Hessian-learning set: selection, split, files, index (CONTEXT.md
"Dataset"; ticket 04).

SELECTION (`select`, driver `01_select.py`)
------------------------------------------
Every molecule under the given tags whose branch A finished (`basins.done`), with what
the split needs to know about it: the SMILES (branch A's Record), heavy atoms, ring
count and heteroatom pattern (the stratification keys, RDKit on the SMILES), its
membership in MACE-OFF23's training set (`data/training_sets/qm9_targets_membership.dat`,
`training_set.qm9_targets_membership`), whether it is one of the PINNED molecules, and
whether a Frame set exists. Written as `select.{out,toml,dat}` under
`<root>/<tag>/_datasets/<name>/` (the first tag's).

SPLIT (`build`, driver `04_dataset.py`) -- rounds 3-4, Q3/Q6 (b), ruled 2026-09-18
--------------------------------------------------------------------------------
    test    WHOLE molecules the model never sees: the pinned 7 (the msRRHO study's
            molecules: acetone, acetamide, propanal, N-methylformamide, 2-methyloxirane,
            cyclopropanol, oxetane) plus TEST_FRACTION of the others, drawn per stratum
            (ring count x heteroatom pattern) with `seed`.
    valid   VALID_FRACTION of the labelled frames of the training molecules, drawn BY
            FRAME with `seed` -- early stopping sees interpolation, as SPICE's split does.
    train   the rest of the training molecules' labelled frames.
    pool    every frame that exists at the engine level but carries no label at `level`
            yet, whatever its molecule's split (`molecule_split` in the index says where
            it will go once labelled).
The split is set here, written into `index.dat`, and never recomputed: the same
selection, level and seed give the same split (sorted inputs, one seeded generator).

FILES
-----
    <root>/<tag>/_datasets/<name>/{train,valid,test}.<level>.extxyz
        the labelled frames with the REFERENCE energy (eV), forces (eV/A) and Cartesian
        Hessian (3N x 3N flattened, eV/A^2) under the keys a MACE loader reads --
        `energy`, `forces` (through the ASE calculator) and `hessian` (info) -- plus the
        frame's identity (`qm9_index`, `basin`, `generator`, `k`, `seed`, `level`).
        The engine's E-F-H at the same frame stays in the molecule's own
        `<generator>.<engine level>.extxyz`; `index.dat` names both files.
    <root>/<tag>/_datasets/<name>/pool.<level>.extxyz
        the unlabelled frames with the ENGINE's E-F-H (their `level` key says so).
    index.dat        one row per frame: molecule, tag, basin, generator, k, split,
                     molecule_split, levels present, seed, engine fingerprint, ORCA
                     version, SPICE membership, stratum, the file and row it sits in.
    dataset.{out,toml}   the Record.
    molecules-<name>.h5  (`--export openreact`) OpenREACT's layout: one group per molecule
                     with `coordinates` (A), `energies` (Eh), `forces` (Eh/A), `hessian`
                     (Eh/A^2) and `species`, attrs crg/mult/nat/molec -- the units READ
                     OFF molecules-RTP.h5 (2026-09-18: with Eh/A^2 the C-H stretches
                     project to 3156-3183 cm^-1; Eh/bohr^2 would give ~6000), which are
                     torchani's, the convention PHL's loader assumes.
"""
import time
from pathlib import Path

import numpy as np

from ..qm_interfaces import orca
from ..store import basins as basins_mod, dat, layout, property as prop, report
from . import frame_labels, frames as frames_mod

STEP_SELECT = "select"
STEP = "dataset"
PROGNAME_SELECT = "openQHA dataset select"
PROGNAME = "openQHA dataset"
SPLITS = ("train", "valid", "test", "pool")
#: the 7 known molecules of the msRRHO study, always `test` (rounds 3-4, Q6/Q3 rulings)
PINNED = ("dsgdb9nsd_000018", "dsgdb9nsd_000019", "dsgdb9nsd_000035", "dsgdb9nsd_000036",
          "dsgdb9nsd_000044", "dsgdb9nsd_000046", "dsgdb9nsd_000048")
VALID_FRACTION = 0.1
TEST_FRACTION = 0.1
SEED = 0
#: a label file whose positions differ from the engine file's by more than this is stale
STALE_TOL_A = 1e-6
DATASETS = "_datasets"
MEMBERSHIP_FILE = Path(__file__).resolve().parents[2] / "data" / "training_sets" / "qm9_targets_membership.dat"

SELECT_ROW_SCHEMA = {
    "qm9_index": ("String", None, "the molecule"),
    "tag": ("String", None, "the tag its branch A product is under"),
    "molecule_dir": ("String", None, "the molecule directory"),
    "smiles": ("String", None, "branch A's declared SMILES"),
    "n_heavy": ("Integer", None, "heavy atoms"),
    "n_rings": ("Integer", None, "rings (RDKit, SSSR)"),
    "hetero": ("String", None, "heteroatom pattern, element counts other than C and H (CH when none)"),
    "stratum": ("String", None, "r<n_rings>_<hetero>: the stratification key"),
    "in_training": ("String", None, "true / false: frames of this molecule in MACE-OFF23's training file; unknown when the membership table has no row"),
    "match_level": ("String", None, "how it was matched there (isomeric, no_stereo, connectivity, none, unknown)"),
    "pinned": ("Boolean", None, "one of the 7 known molecules, always test"),
    "n_basins": ("Integer", None, "MACE basins"),
    "has_frames": ("Boolean", None, "a Frame set Record exists"),
    "n_frames": ("Integer", None, "kept frames of the Frame set (0 without one)"),
}

SELECT_SCHEMA = {
    "Calculation_Info": {
        "NAME": ("String", None, "the Dataset name"),
        "TAGS": ("ArrayOfStrings", None, "the tags scanned, in order"),
        "ROOT": ("String", None, "the runs root"),
        "MEMBERSHIP_FILE": ("String", None, "the SPICE membership table consulted"),
        "PINNED": ("ArrayOfStrings", None, "the molecules always in test"),
        "LIMIT": ("Integer", None, "at most this many molecules (0: all)"),
        "STRATIFY": ("Boolean", None, "with a limit: spread over the sorted index list rather than the first N"),
        "N_MOLECULES": ("Integer", None, "molecules selected"),
        "N_WITH_FRAMES": ("Integer", None, "of those, with a Frame set"),
        "N_IN_TRAINING": ("Integer", None, "of those, in MACE-OFF23's training file"),
        "N_PINNED_PRESENT": ("Integer", None, "pinned molecules found under the tags"),
        "SECONDS": ("Double", "s", "wall time"),
    },
    "Stratum": {
        "STRATUM": ("String", None, "r<n_rings>_<hetero>"),
        "N_MOLECULES": ("Integer", None, "molecules in it"),
        "N_PINNED": ("Integer", None, "of those, pinned"),
    },
}

INDEX_SCHEMA = {
    "qm9_index": ("String", None, "the molecule"),
    "tag": ("String", None, "the tag its molecule directory is under"),
    "basin": ("Integer", None, "the basin the frame was born from"),
    "generator": ("String", None, "basin / displaced / merged / saddle"),
    "k": ("Integer", None, "index within (generator, basin)"),
    "split": ("String", None, "train / valid / test / pool -- set at write time, never recomputed"),
    "molecule_split": ("String", None, "train / test: where the molecule's frames go once labelled"),
    "levels": ("String", None, "levels present at this frame, ';'-joined"),
    "seed": ("Integer", None, "the frame's own draw seed (displaced), 0 otherwise"),
    "engine_params_sha256": ("String", None, "parameter fingerprint of the engine that made the frame"),
    "orca_version": ("String", None, "ORCA version of the reference label, or - when unlabelled"),
    "in_training": ("String", None, "the molecule's MACE-OFF23 membership (true / false / unknown)"),
    "stratum": ("String", None, "the molecule's stratification key"),
    "file": ("String", None, "the split file holding the frame (name, under the Dataset folder)"),
    "row": ("Integer", None, "its 0-based index in that file"),
    "engine_file": ("String", None, "the molecule's engine-level file holding the same frame"),
}

SCHEMA = {
    "Calculation_Info": {
        "NAME": ("String", None, "the Dataset name"),
        "TAGS": ("ArrayOfStrings", None, "the tags of the molecules"),
        "LEVEL": ("String", None, "the reference level of the labelled splits"),
        "MACE_LEVEL": ("String", None, "the engine level of the frames (and of the pool's values)"),
        "SEED": ("Integer", None, "the split's random seed"),
        "VALID_FRACTION": ("Double", None, "fraction of the training molecules' labelled frames drawn as valid"),
        "TEST_FRACTION": ("Double", None, "fraction of the non-pinned molecules drawn as test, per stratum"),
        "PINNED": ("ArrayOfStrings", None, "the molecules always in test"),
        "N_MOLECULES": ("Integer", None, "molecules with a Frame set"),
        "N_TEST_MOLECULES": ("Integer", None, "molecules in test (pinned + drawn)"),
        "N_TRAIN_MOLECULES": ("Integer", None, "molecules whose frames feed train and valid"),
        "N_FRAMES": ("Integer", None, "frames, all splits"),
        "N_LABELLED": ("Integer", None, "frames with the reference label"),
        "N_TRAIN": ("Integer", None, "frames in train"),
        "N_VALID": ("Integer", None, "frames in valid"),
        "N_TEST": ("Integer", None, "frames in test"),
        "N_POOL": ("Integer", None, "frames in pool (unlabelled)"),
        "N_STALE": ("Integer", None, "label frames ignored because their geometry differs from the engine file's (a rerun of branch A / 02 after 03)"),
        "KEPT_PREVIOUS": ("Boolean", None, "a previous index.dat existed and its splits were kept"),
        "ENGINE_PARAMS_SHA256": ("ArrayOfStrings", None, "engine fingerprints seen (one, unless Frame sets were built by different weights)"),
        "ORCA_VERSIONS": ("ArrayOfStrings", None, "ORCA versions of the labels seen"),
        "SECONDS": ("Double", "s", "wall time"),
    },
    "Split": {
        "SPLIT": ("String", None, "train / valid / test / pool"),
        "N_FRAMES": ("Integer", None, "frames"),
        "N_MOLECULES": ("Integer", None, "molecules contributing"),
        "FILE": ("String", None, "the extxyz (absent when empty)"),
    },
    "Molecule": {
        "QM9_INDEX": ("String", None, "the molecule"),
        "TAG": ("String", None, "its tag"),
        "STRATUM": ("String", None, "stratification key"),
        "MOLECULE_SPLIT": ("String", None, "train / test"),
        "PINNED": ("Boolean", None, "always test"),
        "N_FRAMES": ("Integer", None, "frames of its Frame set"),
        "N_LABELLED": ("Integer", None, "of those, labelled at LEVEL"),
        "N_STALE": ("Integer", None, "label files at other coordinates than the engine file (ignored, treated as unlabelled)"),
        "N_TRAIN": ("Integer", None, "frames in train"),
        "N_VALID": ("Integer", None, "frames in valid"),
        "N_TEST": ("Integer", None, "frames in test"),
        "N_POOL": ("Integer", None, "frames in pool"),
    },
}


# ====================================================================== helpers
def datasets_dir(root, tag, name):
    """`<root>/<tag>/_datasets/<name>/`."""
    return Path(root) / str(tag) / DATASETS / str(name)


def stratum_keys(smiles):
    """(n_heavy, n_rings, hetero pattern, stratum) from a SMILES; RDKit. Unparseable ->
    (0, 0, '?', 'r?_?')."""
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
        m = Chem.MolFromSmiles(smiles)
    except Exception:
        m = None
    if m is None:
        return 0, 0, "?", "r?_?"
    counts = {}
    for a in m.GetAtoms():
        s = a.GetSymbol()
        if s not in ("C", "H"):
            counts[s] = counts.get(s, 0) + 1
    hetero = "".join("{}{}".format(s, counts[s]) for s in sorted(counts)) or "CH"
    n_rings = int(rdMolDescriptors.CalcNumRings(m))
    return int(m.GetNumHeavyAtoms()), n_rings, hetero, "r{}_{}".format(n_rings, hetero)


_MEMBERSHIP = {}


def membership(path=MEMBERSHIP_FILE):
    """`{qm9_index: row}` of the SPICE membership table; empty when the file is absent."""
    key = str(path)
    if key not in _MEMBERSHIP:
        p = Path(path)
        _MEMBERSHIP[key] = {r["qm9_index"]: r for r in dat.read_table(p)} if p.is_file() else {}
    return _MEMBERSHIP[key]


def molecules_with_branch_a(root, tag):
    """(qid, molecule dir) for every molecule under `tag` whose branch A finished."""
    base = Path(root) / str(tag)
    out = []
    for rec in sorted(base.glob("*/*/*/{}/{}".format(layout.RECORDS, "branchA.toml"))):
        mol = rec.parent.parent
        if basins_mod.done(mol.name, tag, root=root):
            out.append((mol.name, mol))
    return out


def _smiles_of(mol, qid, tag, root):
    rec = basins_mod.read_record(qid, tag, root=root) or {}
    s = (rec.get("Calculation_Info") or {}).get("SMILES")
    if not s:
        s = (membership().get(qid) or {}).get("smiles", "")
    return s or ""


# ====================================================================== selection
def apply_limit(rows, limit, stratify=False):
    """At most `limit` rows (dicts with `qm9_index` and `pinned`): the pinned ones always,
    the others the first N or, with `stratify`, spread evenly over the sorted index list
    (QM9 is ordered by heavy-atom count). One function for 01_select and 03_labels, so
    the two draw the same molecules."""
    rows = sorted(rows, key=lambda r: r["qm9_index"])
    if not limit or limit >= len(rows):
        return rows
    keep = [r for r in rows if r.get("pinned")]
    rest = [r for r in rows if not r.get("pinned")]
    n = max(0, int(limit) - len(keep))
    if stratify and n:
        idx = sorted({round(i * (len(rest) - 1) / max(1, n - 1)) for i in range(n)})
        keep += [rest[i] for i in idx]
    else:
        keep += rest[:n]
    return sorted(keep, key=lambda r: r["qm9_index"])


def select(root, tags, name, limit=None, stratify=False, membership_file=MEMBERSHIP_FILE, pinned=PINNED):
    """The molecule list: every branch-A-finished molecule under `tags`, with the
    stratification keys, SPICE membership, pin status and Frame-set status. Writes
    `select.{out,toml,dat}` under the first tag's `_datasets/<name>/` and returns the rows."""
    t0 = time.time()
    root = Path(root)
    tags = [tags] if isinstance(tags, str) else list(tags)
    mem = membership(membership_file)
    rows, seen = [], set()
    for tag in tags:
        for qid, mol in molecules_with_branch_a(root, tag):
            if qid in seen:
                continue                                   # the first tag wins
            seen.add(qid)
            smiles = _smiles_of(mol, qid, tag, root)
            n_heavy, n_rings, hetero, stratum = stratum_keys(smiles)
            m = mem.get(qid)
            fr = layout.frames_dir(mol) / (frames_mod.STEP + ".toml")
            n_frames = 0
            if fr.is_file():
                n_frames = int(prop.load(fr)["Calculation_Info"]["N_FRAMES"])
            rows.append(dict(qm9_index=qid, tag=tag, molecule_dir=str(mol), smiles=smiles, n_heavy=n_heavy,
                             n_rings=n_rings, hetero=hetero, stratum=stratum,
                             in_training=("true" if m["in_training"] else "false") if m else "unknown",
                             match_level=m["match_level"] if m else "unknown",
                             pinned=qid in pinned, n_basins=len(basins_mod.basin_files(mol)),
                             has_frames=fr.is_file(), n_frames=n_frames))
    rows = apply_limit(rows, limit, stratify)
    strata = {}
    for r in rows:
        s = strata.setdefault(r["stratum"], dict(STRATUM=r["stratum"], N_MOLECULES=0, N_PINNED=0))
        s["N_MOLECULES"] += 1
        s["N_PINNED"] += int(r["pinned"])
    info = dict(NAME=str(name), TAGS=tags, ROOT=str(root), MEMBERSHIP_FILE=str(membership_file), PINNED=list(pinned),
                LIMIT=int(limit or 0), STRATIFY=bool(stratify), N_MOLECULES=len(rows),
                N_WITH_FRAMES=sum(1 for r in rows if r["has_frames"]),
                N_IN_TRAINING=sum(1 for r in rows if r["in_training"] == "true"),
                N_PINNED_PRESENT=sum(1 for r in rows if r["pinned"]), SECONDS=time.time() - t0)
    d = datasets_dir(root, tags[0], name)
    d.mkdir(parents=True, exist_ok=True)
    strata_rows = [strata[k] for k in sorted(strata)]
    missing = prop.write(d / (STEP_SELECT + ".toml"), {"Calculation_Info": info, "Stratum": strata_rows},
                         SELECT_SCHEMA, prop.NORMAL_TERMINATION, PROGNAME_SELECT)
    if missing:
        raise RuntimeError("select.toml keys outside the schema: {}".format(missing))
    dat.write_table(d / (STEP_SELECT + ".dat"), rows, list(SELECT_ROW_SCHEMA), SELECT_ROW_SCHEMA)
    rep = report.Report(PROGNAME_SELECT, "the molecule list of Dataset {!r}".format(name))
    rep.section("conventions")
    for k in ("TAGS", "LIMIT", "STRATIFY", "N_MOLECULES", "N_WITH_FRAMES", "N_IN_TRAINING", "N_PINNED_PRESENT"):
        rep.kv(k, info[k])
    rep.section("per stratum")
    rep.table(["stratum", "molecules", "pinned"], [[s["STRATUM"], s["N_MOLECULES"], s["N_PINNED"]] for s in strata_rows])
    rep.section("per molecule")
    rep.table(["qm9_index", "tag", "smiles", "heavy", "stratum", "in_training", "pinned", "basins", "frames"],
              [[r["qm9_index"], r["tag"], r["smiles"], r["n_heavy"], r["stratum"], r["in_training"],
                "yes" if r["pinned"] else "-", r["n_basins"], r["n_frames"] if r["has_frames"] else "-"] for r in rows])
    rep.note("pinned molecules are the msRRHO study's seven and are always test; the stratum (ring count x "
             "heteroatom pattern) is what the test draw is balanced over; in_training is membership in "
             "MACE-OFF23's SPICE training file (data/training_sets/qm9_targets_membership.dat).")
    rep.write(d / (STEP_SELECT + ".out"), step=STEP_SELECT)
    return rows


def read_selection(root, tag, name):
    p = datasets_dir(root, tag, name) / (STEP_SELECT + ".dat")
    if not p.is_file():
        raise FileNotFoundError("no selection at {} (run 01_select first)".format(p))
    return dat.read_table(p)


# ====================================================================== the split
def _frames_of(mol, level, mace_level):
    """Every kept frame of a molecule: (key, engine atoms, label atoms or None, engine file,
    stale). A label is paired by (generator, basin, k) AND by geometry: a label file left
    behind by an earlier branch A / 02_frames run sits at other coordinates and is not a
    label of this frame -- it counts as `stale` (unlabelled, and named in the Record)."""
    out = []
    for g in frames_mod.GENERATORS:
        ef = layout.frames_file(mol, g, mace_level)
        if not ef.is_file():
            continue
        lf = layout.frames_file(mol, g, level)
        labels = {}
        if lf.is_file():
            for a in frames_mod.read_frames(lf):
                labels[(int(a.info["basin"]), int(a.info["k"]))] = a
        for a in frames_mod.read_frames(ef):
            key = (g, int(a.info["basin"]), int(a.info["k"]))
            lab = labels.get((key[1], key[2]))
            stale = False
            if lab is not None and (len(lab) != len(a) or np.abs(lab.get_positions() - a.get_positions()).max() > STALE_TOL_A):
                lab, stale = None, True
            out.append((key, a, lab, ef, stale))
    return out


def _previous_split(dataset_dir):
    """`{(qm9_index, generator, basin, k): split}` of the labelled frames of the last
    `index.dat`, and `{qm9_index: molecule_split}` -- what a rebuild keeps."""
    p = Path(dataset_dir) / "index.dat"
    frames_prev, mols_prev = {}, {}
    if p.is_file():
        for r in dat.read_table(p):
            mols_prev[r["qm9_index"]] = r["molecule_split"]
            if r["split"] != "pool":
                frames_prev[(r["qm9_index"], r["generator"], int(r["basin"]), int(r["k"]))] = r["split"]
    return frames_prev, mols_prev


def _rng(seed, *parts):
    """A generator seeded from `seed` and the names of what it draws for -- so a draw for
    one molecule (or stratum) does not move when another molecule is added or labelled."""
    import hashlib
    h = hashlib.sha256("|".join([str(seed)] + [str(x) for x in parts]).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def build(root, tags, name, level=frame_labels.DEFAULT_LEVEL, valid_fraction=VALID_FRACTION,
          test_fraction=TEST_FRACTION, seed=SEED, mace_level=None, selection=None, pinned=PINNED,
          keep_previous=True):
    """The Dataset: split every Frame set of the selected molecules, write the four
    files, `index.dat` and the Record. `selection`: rows as `select` returns them
    (default: read `select.dat` under the first tag; run `select` first).

    A rebuild KEEPS the previous `index.dat`'s decisions (`keep_previous`): a molecule's
    train/test side and every already-labelled frame's split stay what they were; only
    new molecules and newly labelled frames are drawn, each with a generator seeded from
    `seed` and its own name, so nothing moves because something else was added. That is
    what "set at write time, never recomputed" means across the label Batches that fill
    the pool."""
    t0 = time.time()
    root = Path(root)
    tags = [tags] if isinstance(tags, str) else list(tags)
    rows_sel = selection if selection is not None else read_selection(root, tags[0], name)
    # has_frames is re-read from disk, not taken from select.dat: 01 may have run before 02
    mols = [r for r in sorted(rows_sel, key=lambda r: r["qm9_index"])
            if (layout.frames_dir(r["molecule_dir"]) / (frames_mod.STEP + ".toml")).is_file()]
    if not mols:
        raise RuntimeError("no selected molecule has a Frame set (run 02_frames first)")
    if mace_level is None:
        mace_level = frame_labels.mace_level(mols[0]["molecule_dir"])
    d = datasets_dir(root, tags[0], name)
    prev_frames, prev_mols = _previous_split(d) if keep_previous else ({}, {})

    # ---- test molecules: pinned + the previous index's + a per-stratum draw of the NEW ones
    test_mols = {r["qm9_index"] for r in mols if r["qm9_index"] in pinned}
    test_mols.update(q for q, s in prev_mols.items() if s == "test")
    by_stratum = {}
    for r in mols:
        if r["qm9_index"] not in test_mols and r["qm9_index"] not in prev_mols:
            by_stratum.setdefault(r["stratum"], []).append(r["qm9_index"])
    for s in sorted(by_stratum):
        ids = sorted(by_stratum[s])
        n = int(round(float(test_fraction) * len(ids)))
        if n:
            test_mols.update(_rng(seed, "test", s, *ids).choice(ids, size=n, replace=False).tolist())

    # ---- frames ------------------------------------------------------------------------
    index, per_mol, split_frames = [], [], {s: [] for s in SPLITS}
    fingerprints, versions = set(), set()
    for r in mols:
        qid, mol = r["qm9_index"], Path(r["molecule_dir"])
        msplit = "test" if qid in test_mols else "train"
        frames_here = _frames_of(mol, level, mace_level)
        labelled = [(key, a, lab, ef) for key, a, lab, ef, _st in frames_here if lab is not None]
        n_stale = sum(1 for _k, _a, _l, _e, st in frames_here if st)
        # valid: frames the previous index already placed keep their split; the NEW labelled
        # frames are drawn (per-molecule generator) so the molecule reaches the fraction
        valid_keys = {key for key, _a, _l, _e in labelled if prev_frames.get((qid,) + key) == "valid"}
        if msplit == "train" and labelled:
            new = [key for key, _a, _l, _e in labelled if (qid,) + key not in prev_frames]
            want = int(round(float(valid_fraction) * len(labelled))) - len(valid_keys)
            if new and want > 0:
                pick = _rng(seed, "valid", qid).choice(len(new), size=min(want, len(new)), replace=False)
                valid_keys.update(new[i] for i in pick)
        counts = dict(train=0, valid=0, test=0, pool=0)
        for key, a, lab, ef, _st in frames_here:
            if lab is None:
                split, atoms, levels, ver = "pool", a, [mace_level], "-"
            else:
                split = msplit if key not in valid_keys else "valid"
                atoms, levels, ver = lab, [mace_level, level], str(lab.info.get("orca_version", "-"))
                versions.add(ver)
            fp = str(a.info.get("engine_params_sha256", "-"))
            fingerprints.add(fp)
            counts[split] += 1
            split_frames[split].append(atoms)
            index.append(dict(qm9_index=qid, tag=r["tag"], basin=key[1], generator=key[0], k=key[2], split=split,
                              molecule_split=msplit, levels=";".join(levels), seed=int(a.info.get("seed", 0)),
                              engine_params_sha256=fp, orca_version=ver, in_training=r["in_training"],
                              stratum=r["stratum"], file="{}.{}.extxyz".format(split, level),
                              row=len(split_frames[split]) - 1, engine_file=str(ef)))
        per_mol.append(dict(QM9_INDEX=qid, TAG=r["tag"], STRATUM=r["stratum"], MOLECULE_SPLIT=msplit,
                            PINNED=qid in pinned, N_FRAMES=len(frames_here), N_LABELLED=len(labelled), N_STALE=n_stale,
                            N_TRAIN=counts["train"], N_VALID=counts["valid"], N_TEST=counts["test"], N_POOL=counts["pool"]))

    # ---- files -------------------------------------------------------------------------
    d.mkdir(parents=True, exist_ok=True)
    split_rows = []
    for s in SPLITS:
        path = d / "{}.{}.extxyz".format(s, level)
        n_mol = len({row["qm9_index"] for row in index if row["split"] == s})
        if split_frames[s]:
            _write_split(path, split_frames[s])
        elif path.is_file():
            path.unlink()
        split_rows.append(dict(SPLIT=s, N_FRAMES=len(split_frames[s]), N_MOLECULES=n_mol, FILE=str(path) if split_frames[s] else None))
    dat.write_table(d / "index.dat", index, list(INDEX_SCHEMA), INDEX_SCHEMA)
    info = dict(NAME=str(name), TAGS=tags, LEVEL=level, MACE_LEVEL=mace_level, SEED=int(seed),
                VALID_FRACTION=float(valid_fraction), TEST_FRACTION=float(test_fraction), PINNED=list(pinned),
                N_MOLECULES=len(mols), N_TEST_MOLECULES=len(test_mols), N_TRAIN_MOLECULES=len(mols) - len(test_mols),
                N_FRAMES=len(index), N_LABELLED=sum(1 for row in index if row["split"] != "pool"),
                N_TRAIN=len(split_frames["train"]), N_VALID=len(split_frames["valid"]),
                N_TEST=len(split_frames["test"]), N_POOL=len(split_frames["pool"]),
                N_STALE=sum(m["N_STALE"] for m in per_mol), KEPT_PREVIOUS=bool(prev_frames or prev_mols),
                ENGINE_PARAMS_SHA256=sorted(fingerprints), ORCA_VERSIONS=sorted(versions), SECONDS=time.time() - t0)
    missing = prop.write(d / (STEP + ".toml"), {"Calculation_Info": info, "Split": split_rows, "Molecule": per_mol},
                         SCHEMA, prop.NORMAL_TERMINATION, PROGNAME)
    if missing:
        raise RuntimeError("dataset.toml keys outside the schema: {}".format(missing))
    _write_report(d / (STEP + ".out"), info, split_rows, per_mol)
    return dict(info=info, splits=split_rows, molecules=per_mol, index=index, dir=d)


def _write_split(path, atoms_list):
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io import write
    out = []
    for a in atoms_list:
        b = a.copy()
        b.calc = SinglePointCalculator(b, energy=float(a.get_potential_energy()), forces=np.asarray(a.get_forces(), dtype=float))
        b.info = dict(a.info)
        if a.info.get("hessian") is not None:
            b.info["hessian"] = np.asarray(a.info["hessian"], dtype=float).reshape(-1)
            b.info["has_hessian"] = True
        else:
            b.info.pop("hessian", None)
            b.info["has_hessian"] = False
        out.append(b)
    write(str(path), out, format="extxyz")


def read_index(dataset_dir):
    return dat.read_table(Path(dataset_dir) / "index.dat")


def _write_report(path, info, split_rows, per_mol):
    rep = report.Report(PROGNAME, "Dataset {!r} at {}".format(info["NAME"], info["LEVEL"]))
    rep.section("conventions")
    for k in ("TAGS", "LEVEL", "MACE_LEVEL", "SEED", "VALID_FRACTION", "TEST_FRACTION", "N_MOLECULES", "N_TEST_MOLECULES",
              "N_TRAIN_MOLECULES", "N_FRAMES", "N_LABELLED", "N_STALE", "KEPT_PREVIOUS", "ENGINE_PARAMS_SHA256", "ORCA_VERSIONS"):
        rep.kv(k, info[k])
    rep.section("per split")
    rep.table(["split", "frames", "molecules", "file"],
              [[r["SPLIT"], r["N_FRAMES"], r["N_MOLECULES"], Path(r["FILE"]).name if r["FILE"] else "-"] for r in split_rows])
    rep.section("per molecule")
    rep.table(["qm9_index", "tag", "stratum", "split", "pinned", "frames", "labelled", "stale", "train", "valid", "test", "pool"],
              [[m["QM9_INDEX"], m["TAG"], m["STRATUM"], m["MOLECULE_SPLIT"], "yes" if m["PINNED"] else "-", m["N_FRAMES"],
                m["N_LABELLED"], m["N_STALE"], m["N_TRAIN"], m["N_VALID"], m["N_TEST"], m["N_POOL"]] for m in per_mol])
    rep.note("test = whole molecules (the pinned seven + a per-stratum draw of TEST_FRACTION); valid = VALID_FRACTION of "
             "the training molecules' labelled frames, drawn by frame; train = the rest; pool = frames without a label "
             "at LEVEL yet, whatever their molecule's split. The split is set here and never recomputed: a rebuild keeps "
             "every decision of the previous index.dat and draws only the new molecules and newly labelled frames, each "
             "with its own seeded generator. A label file whose geometry differs from the engine file's (stale) is "
             "ignored. train/valid/test files carry the REFERENCE E-F-H; pool carries the engine's.")
    rep.write(path, step=STEP)


# ====================================================================== OpenREACT export
def export_openreact(dataset_dir, level=frame_labels.DEFAULT_LEVEL, name=None, splits=("train", "valid", "test")):
    """`molecules-<name>.h5` in OpenREACT's layout (one group per molecule: coordinates
    A, energies Eh, forces Eh/A, hessian (n, 3N, 3N) Eh/A^2, species; attrs crg, mult,
    nat, molec) plus, per frame, `split`, `generator`, `basin`, `k` datasets of our own
    -- extra keys a reader by name ignores. Only labelled frames; returns the path."""
    import h5py
    d = Path(dataset_dir)
    name = name or d.name
    index = read_index(d)
    files = {}
    by_mol = {}
    for row in index:
        if row["split"] not in splits:
            continue
        if row["file"] not in files:
            files[row["file"]] = frames_mod.read_frames(d / row["file"])
        by_mol.setdefault(row["qm9_index"], []).append((row, files[row["file"]][int(row["row"])]))
    out = d / "molecules-{}.h5".format(name)
    ev2eh = 1.0 / orca.EV_PER_HARTREE
    with h5py.File(out, "w") as f:
        f.attrs["readme"] = ("openQHA Hessian-learning Dataset {!r}: frames of QM9 molecules at the engine's basins and "
                             "harmonic displacements, labelled at {} with ORCA (single point + EnGrad + analytic Hessian "
                             "at the fixed geometry). Units as OpenREACT's molecules-*.h5: A, Eh, Eh/A, Eh/A^2.").format(name, level)
        f.attrs["level"] = level
        for qid in sorted(by_mol):
            items = by_mol[qid]
            a0 = items[0][1]
            g = f.create_group(qid)
            g.attrs.update(crg=0, mult=1, nat=len(a0), molec=len(items), smiles=str(a0.info.get("smiles", "")))
            g.create_dataset("species", data=np.array([s.encode() for s in a0.get_chemical_symbols()], dtype="S1"))
            g.create_dataset("coordinates", data=np.array([a.get_positions() for _r, a in items]))
            g.create_dataset("energies", data=np.array([float(a.get_potential_energy()) * ev2eh for _r, a in items]))
            g.create_dataset("forces", data=np.array([np.asarray(a.get_forces()) * ev2eh for _r, a in items]))
            nat3 = 3 * len(a0)
            g.create_dataset("hessian", data=np.array([np.asarray(a.info["hessian"]) * ev2eh if a.info.get("hessian") is not None
                                                       else np.full((nat3, nat3), np.nan) for _r, a in items]))
            g.create_dataset("has_hessian", data=np.array([a.info.get("hessian") is not None for _r, a in items]))
            g.create_dataset("split", data=np.array([r["split"].encode() for r, _a in items], dtype="S5"))
            g.create_dataset("generator", data=np.array([r["generator"].encode() for r, _a in items], dtype="S9"))
            g.create_dataset("basin", data=np.array([int(r["basin"]) for r, _a in items]))
            g.create_dataset("k", data=np.array([int(r["k"]) for r, _a in items]))
    return out
