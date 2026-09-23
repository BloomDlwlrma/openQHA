"""The Dataset of the Hessian-learning set: selection, split, files, index (CONTEXT.md
"Dataset"; ticket 04).

SELECTION (`select`, driver `01_select.py`)
------------------------------------------
Every molecule under the given tags whose branch A finished (`basins.done`), with what
the split needs to know about it: the SMILES (branch A's Record), heavy atoms, ring
count and heteroatom pattern (the stratification keys, RDKit on the SMILES), its
structure classes (`structure_classes.classify`; ticket 07), its membership in
MACE-OFF23's training set (`data/training_sets/qm9_targets_membership.dat`,
`training_set.qm9_targets_membership`), whether it is one of the PINNED molecules, and
whether a Frame set exists. When a DRAW (`draw.dat` of `00_draw`, ticket 05) exists
under the Dataset folder, EVERY drawn molecule is a row -- with `has_basins` /
`has_frames` false until branch A / 02 have run for it -- so the campaign's progress is
one table and a Batch over `--name` always sees the whole draw. Written as
`select.{out,toml,dat}` under `<root>/<tag>/_datasets/<name>/` (the first tag's).

SPLIT (`build`, driver `04_dataset.py`)
---------------------------------------
Two modes (`split_by`). BY MOLECULE is the PRODUCTION split since S0-C-65 (2026-09-23), the
granularity MACE-OFF was trained with (Kovacs et al., JACS 147, 17598 (2025), Sec. 2.2: 95 % of
SPICE for training and validation, 5 % for testing, "splitting was performed at the molecule
level, ensuring that conformers of the same molecule do not appear in both train/validation and
test sets"; the validation set is then drawn by configuration inside the 95 %, as mace's
`--valid_fraction` does it). It supersedes the by-frame production split of round 5 Q4
(S0-C-49) so that a test frame is always a molecule the fine-tune never saw: the judge's
`interpolation` distribution is empty by construction and its gate row is a generalisation
reading. BY FRAME stays for the smoke and fit Datasets, where whole-molecule test frames would
eat a large share of a small label budget.

BY FRAME (the smoke / fit split; the production split until S0-C-65):
    test    the pinned 7 (the msRRHO study's molecules: acetone, acetamide, propanal,
            N-methylformamide, 2-methyloxirane, cyclopropanol, oxetane) as WHOLE
            molecules, plus FRAME_TEST_FRACTION of every other molecule's labelled frames
    valid   FRAME_VALID_FRACTION of those frames
    train   the rest -- 90 / 5 / 5 of the non-pinned labelled frames
Every frame is drawn on its own: one number from a generator seeded from (seed, "frame",
molecule, generator, basin, k), so a frame's split never depends on what else is
labelled, and the fractions are EXPECTATIONS (exact to +-0.1 % over 100,000 frames, +-1
frame over 15). BY MOLECULE (the production split, S0-C-65; the smoke split of rounds 3-4, Q3/Q6 (b)):
    test    whole molecules: the pinned 7 plus TEST_FRACTION of the others, drawn per
            stratum (ring count x heteroatom pattern) with `seed`, so every stratum has
            test molecules
    valid   VALID_FRACTION of the training molecules' labelled frames, drawn by frame
    train   the rest of the training molecules' labelled frames
Both draws are per-identity (`_rng(seed, "test", stratum, ...)` for a molecule,
`_rng(seed, "valid", qid)` for its frames), so a rebuild as labels arrive keeps every earlier
decision: round t's splits are a prefix of round t+1's.
In both modes `pool` is every frame that exists at the engine level but carries no label
at `level` yet, whatever its molecule's side (`molecule_split` in the index says where
it will go once labelled). The split is set here, written into `index.dat`, and never
recomputed: a rebuild keeps every earlier decision and draws only what is new.

THE HELD-OUT GENERATORS (`train_generators`, S0-C-54, ADR 0005; ticket 20). Only the
frames of the generators in `train_generators` -- default `("basin",)` -- can enter
train or valid. EVERY labelled frame of any other generator (displaced, merged, saddle)
goes to test, whatever the frame draw or the previous index said, with
`held_out_generator = yes` in `index.dat`: the judge reads them (the RMS-displacement
reference rows), training never sees them. The label rule (`frame_labels.HESSIAN_
GENERATORS`) is untouched -- what changes is who trains on what. The Record names
`TRAIN_GENERATORS`, `HELD_OUT_GENERATORS`, `N_TRAIN_BASIN` (train frames of the training
generators) and `N_TEST_HELD_OUT`.

FILES
-----
    <root>/<tag>/_datasets/<name>/{train,valid,test}.<level>.extxyz
        the labelled frames with the REFERENCE energy (eV), forces (eV/A) and Cartesian
        Hessian (3N x 3N flattened, eV/A^2) under the keys a MACE loader reads --
        `energy`, `forces` (through the ASE calculator) and `hessian` (info) -- AND under
        MACE-torch's default training keys `REF_energy` (info), `REF_forces` (arrays),
        `REF_hessian` (info, flattened; only where the frame has one, `has_hessian`
        says so; round 5 Q7 (b)) so `--energy_key` / `--forces_key` need no override --
        plus `split` and the frame's identity (`qm9_index`, `basin`, `generator`, `k`,
        `seed`, `level`). The engine's E-F-H at the same frame stays in the molecule's
        own `<generator>.<engine level>.extxyz`; `index.dat` names both files.
    <root>/<tag>/_datasets/<name>/mace_<name>.<level>.extxyz
        the three labelled splits in ONE file (the user's "single xyz" for MACE Hessian
        learning), every frame with its `split` key; the same frames, same order, as the
        per-split files concatenated train, valid, test.
    <root>/<tag>/_datasets/<name>/pool.<level>.extxyz
        the unlabelled frames with the ENGINE's E-F-H (their `level` key says so).
    index.dat        one row per frame: molecule, tag, basin, generator, k, split,
                     molecule_split, levels present, seed, engine fingerprint, ORCA
                     version, SPICE membership, stratum, classes, the file and row it
                     sits in.
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
#: by-molecule split (production since S0-C-65, MACE-OFF's granularity): 5 % of the molecules
#: are test (whole), 5 % of the training molecules' labelled frames are valid
VALID_FRACTION = 0.05
TEST_FRACTION = 0.05
#: The fixed validation probes (S0-C-67, PHL's fixed-vector protocol): every labelled frame
#: carrying a Hessian gets its own [VALID_PROBE_KMAX, 3N] Rademacher set, drawn here from
#: the frame's IDENTITY and written into the file. The loss takes the first k rows of it, so
#: the sets are nested: a K scan is a flag, not a rebuild, and the K = 4 and K = 8 readings
#: are comparable by construction. Nothing is derived from the Label's bytes.
#: Every labelled split, not only valid: mace evaluates the loss on the TRAINING split too
#: (its final error table), and in eval mode the loss reads stored probes and draws none.
#: What is particular to valid is the USE -- the reading that drives the schedule.
VALID_PROBE_KMAX = 16
VALID_PROBE_MODE = "rademacher"
#: by-frame split (the smoke / fit mode; production until S0-C-65): 90 / 5 / 5 of the frames
FRAME_VALID_FRACTION = 0.05
FRAME_TEST_FRACTION = 0.05
SPLIT_MODES = ("frame", "molecule")
#: what `04_dataset.py` uses unless told otherwise (S0-C-65)
DEFAULT_SPLIT_MODE = "molecule"
SEED = 0
#: the generators whose frames may train (S0-C-54: basin Hessians only); every other
#: generator is held out -- its labelled frames go to test for the judge
TRAIN_GENERATORS = ("basin",)
#: the production row R4 (S0-C-60): Replay frames = this x the train frames with a Hessian
REPLAY_PER_HESSIAN_FRAME_R4 = 4
REPLAY_CONFIG_WEIGHT_R4 = 10.0
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
    "classes": ("String", None, "structure classes (configs/structure_classes.yaml), ';'-joined; - when none"),
    "in_training": ("String", None, "true / false: frames of this molecule in MACE-OFF23's training file; unknown when the membership table has no row"),
    "match_level": ("String", None, "how it was matched there (isomeric, no_stereo, connectivity, none, unknown)"),
    "pinned": ("Boolean", None, "one of the 7 known molecules, always test"),
    "drawn": ("Boolean", None, "a row of the draw (draw.dat); false for a molecule found under a tag only"),
    "has_basins": ("Boolean", None, "branch A finished (basins.done)"),
    "n_basins": ("Integer", None, "MACE basins (0 without branch A)"),
    "has_frames": ("Boolean", None, "a Frame set Record exists"),
    "n_frames": ("Integer", None, "kept frames of the Frame set (0 without one)"),
}

SELECT_SCHEMA = {
    "Calculation_Info": {
        "NAME": ("String", None, "the Dataset name"),
        "TAGS": ("ArrayOfStrings", None, "the tags scanned, in order"),
        "ROOT": ("String", None, "the runs root"),
        "MEMBERSHIP_FILE": ("String", None, "the SPICE membership table consulted"),
        "DRAW_FILE": ("String", None, "the draw.dat the rows come from, or - when the tags were scanned instead"),
        "PINNED": ("ArrayOfStrings", None, "the molecules always in test"),
        "LIMIT": ("Integer", None, "at most this many molecules (0: all)"),
        "STRATIFY": ("Boolean", None, "with a limit: spread over the sorted index list rather than the first N"),
        "N_MOLECULES": ("Integer", None, "molecules selected"),
        "N_DRAWN": ("Integer", None, "of those, rows of the draw"),
        "N_WITH_BASINS": ("Integer", None, "of those, with branch A finished"),
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
    "Class": {
        "CLASS": ("String", None, "a structure class"),
        "N_MOLECULES": ("Integer", None, "selected molecules in it"),
        "N_WITH_BASINS": ("Integer", None, "of those, with branch A finished"),
        "N_WITH_FRAMES": ("Integer", None, "of those, with a Frame set"),
    },
}

INDEX_SCHEMA = {
    "qm9_index": ("String", None, "the molecule"),
    "tag": ("String", None, "the tag its molecule directory is under"),
    "basin": ("Integer", None, "the basin the frame was born from"),
    "generator": ("String", None, "basin / displaced / merged / saddle"),
    "k": ("Integer", None, "index within (generator, basin)"),
    "split": ("String", None, "train / valid / test / pool -- set at write time, never recomputed"),
    "molecule_split": ("String", None, "test: a whole-molecule test molecule (pinned, or drawn by molecule); train: its labelled frames go to train / valid (by molecule) or train / valid / test (by frame)"),
    "held_out_generator": ("String", None, "yes: the frame's generator is not in TRAIN_GENERATORS, so it is in test whatever the draw (S0-C-54); no otherwise"),
    "levels": ("String", None, "levels present at this frame, ';'-joined"),
    "seed": ("Integer", None, "the frame's own draw seed (displaced), 0 otherwise"),
    "engine_params_sha256": ("String", None, "parameter fingerprint of the engine that made the frame"),
    "orca_version": ("String", None, "ORCA version of the reference label, or - when unlabelled"),
    "in_training": ("String", None, "the molecule's MACE-OFF23 membership (true / false / unknown)"),
    "stratum": ("String", None, "the molecule's stratification key"),
    "classes": ("String", None, "the molecule's structure classes, ';'-joined (- when none)"),
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
        "SPLIT_BY": ("String", None, "molecule (production since S0-C-65, MACE-OFF's granularity: whole test molecules, valid drawn by frame from the training molecules) or frame (the smoke / fit mode: 90/5/5 of the non-pinned labelled frames, each drawn on its own)"),
        "RESPLIT": ("Boolean", None, "this build discarded the previous index's decisions and drew the whole Dataset again (a split-mode change)"),
        "VALID_FRACTION": ("Double", None, "fraction of the labelled frames drawn as valid (by frame: of every non-pinned molecule's; by molecule: of the training molecules')"),
        "TEST_FRACTION": ("Double", None, "by frame: fraction of the non-pinned labelled frames drawn as test; by molecule: fraction of the non-pinned molecules drawn as test, per stratum"),
        "PINNED": ("ArrayOfStrings", None, "the molecules always in test"),
        "PURPOSE": ("String", None, "judge (the production Dataset: its test split is a claim about generalisation) or fit (a Dataset built with the pinned rule OFF, for measuring cost and weights only -- every number from it is interpolation within the same molecules and must be reported as such; ticket 15)"),
        "TRAIN_GENERATORS": ("ArrayOfStrings", None, "the Frame generators whose frames may enter train and valid (S0-C-54: basin)"),
        "HELD_OUT_GENERATORS": ("ArrayOfStrings", None, "the generators whose labelled frames all go to test (the judge's reference rows), never train or valid"),
        "N_TRAIN_BASIN": ("Integer", None, "train frames of the training generators (every train frame, by construction)"),
        "N_TEST_HELD_OUT": ("Integer", None, "test frames that are there because their generator is held out"),
        "N_MOLECULES": ("Integer", None, "molecules with a Frame set"),
        "N_TEST_MOLECULES": ("Integer", None, "molecules in test (pinned + drawn)"),
        "N_TRAIN_MOLECULES": ("Integer", None, "molecules whose frames feed train and valid"),
        "N_FRAMES": ("Integer", None, "frames, all splits"),
        "N_LABELLED": ("Integer", None, "frames with the reference label"),
        "N_TRAIN": ("Integer", None, "frames in train"),
        "N_VALID": ("Integer", None, "frames in valid"),
        "N_TEST": ("Integer", None, "frames in test"),
        "N_POOL": ("Integer", None, "frames in pool (unlabelled)"),
        "N_VALID_PROBE_FRAMES": ("Integer", None, "labelled frames carrying their fixed probe set: every frame with a Hessian, because mace evaluates the loss on the training split too (S0-C-67)"),
        "VALID_PROBE_SOURCE": ("String", None, "where the validation probes come from: file (drawn here, stored in the valid split) -- the loss draws none"),
        "VALID_PROBE_KMAX": ("Integer", None, "rows stored per valid frame; the loss takes the first k of them (nested, so a K scan needs no rebuild)"),
        "VALID_PROBE_MODE": ("String", None, "the stored probes' distribution (rademacher: the smallest variance of the unit-variance draws)"),
        "N_HESSIAN_FRAMES": ("Integer", None, "labelled frames carrying a reference Hessian (basin / merged / saddle)"),
        "N_TRAIN_HESSIAN": ("Integer", None, "train frames carrying a reference Hessian: the number the production Replay is 4x of (S0-C-60)"),
        "REPLAY_R4_FRAMES": ("Integer", None, "4 x N_TRAIN_HESSIAN: the Replay size of the production row R4 (s0_spice_pt_draw.py --n)"),
        "N_STALE": ("Integer", None, "label frames ignored because their geometry differs from the engine file's (a rerun of branch A / 02 after 03)"),
        "MERGED_FILE": ("String", None, "the single xyz of the three labelled splits (mace_<name>.<level>.extxyz), or - when nothing is labelled"),
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
        "N_TEST_HELD_OUT": ("Integer", None, "of the test frames, those of a held-out generator"),
        "N_POOL": ("Integer", None, "frames in pool"),
    },
    "Class": {
        "CLASS": ("String", None, "a structure class"),
        "N_MOLECULES": ("Integer", None, "molecules with a Frame set in it"),
        "N_FRAMES": ("Integer", None, "their frames"),
        "N_LABELLED": ("Integer", None, "of those, labelled at LEVEL"),
        "N_TRAIN": ("Integer", None, "frames in train (basin frames)"),
        "N_VALID": ("Integer", None, "frames in valid"),
        "N_TEST": ("Integer", None, "frames in test"),
        "N_TEST_HELD_OUT": ("Integer", None, "of the test frames, those of a held-out generator"),
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
    for rec in sorted(base.glob("*/{}/{}".format(layout.RECORDS, "branchA.toml"))):
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


_CLASSIFIER = []


def classes_of(smiles):
    """The structure classes of a SMILES, ';'-joined (`-` when none or unparseable):
    `structure_classes.Classifier` on `configs/structure_classes.yaml`, built once."""
    if not smiles:
        return "-"
    if not _CLASSIFIER:
        from . import structure_classes                    # here: structure_classes imports this module
        _CLASSIFIER.append(structure_classes.Classifier())
    return ";".join(sorted(_CLASSIFIER[0].classify(smiles))) or "-"


def class_rows(rows, count_keys):
    """`[[Class]]` rows: per class over `rows` (dicts with `classes`), each counter of
    `count_keys` = (column, predicate or key) summed over the molecules in the class."""
    out = {}
    for r in rows:
        for c in [c for c in str(r.get("classes", "-")).split(";") if c and c != "-"]:
            row = out.setdefault(c, dict(CLASS=c))
            for col, fn in count_keys:
                row[col] = row.get(col, 0) + int(fn(r))
    return [out[c] for c in sorted(out)]


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


def _select_row(root, tag, qid, mol, smiles, mem, pinned, drawn, classes=None):
    n_heavy, n_rings, hetero, stratum = stratum_keys(smiles)
    m = mem.get(qid)
    has_basins = basins_mod.done(qid, tag, root=root)
    fr = layout.frames_dir(mol) / (frames_mod.STEP + ".toml")
    n_frames = int(prop.load(fr)["Calculation_Info"]["N_FRAMES"]) if fr.is_file() else 0
    return dict(qm9_index=qid, tag=tag, molecule_dir=str(mol), smiles=smiles, n_heavy=n_heavy,
                n_rings=n_rings, hetero=hetero, stratum=stratum,
                classes=classes if classes is not None else classes_of(smiles),
                in_training=("true" if m["in_training"] else "false") if m else "unknown",
                match_level=m["match_level"] if m else "unknown",
                pinned=qid in pinned, drawn=bool(drawn), has_basins=bool(has_basins),
                n_basins=len(basins_mod.basin_files(mol)) if has_basins else 0,
                has_frames=fr.is_file(), n_frames=n_frames)


def select(root, tags, name, limit=None, stratify=False, membership_file=MEMBERSHIP_FILE, pinned=PINNED):
    """The molecule list: every molecule of the draw (`draw.dat` under the Dataset
    folder, when there is one) and every branch-A-finished molecule under `tags`, with
    the stratification keys, structure classes, SPICE membership, pin status, branch A
    and Frame-set status. Writes `select.{out,toml,dat}` under the first tag's
    `_datasets/<name>/` and returns the rows."""
    t0 = time.time()
    root = Path(root)
    tags = [tags] if isinstance(tags, str) else list(tags)
    mem = membership(membership_file)
    d = datasets_dir(root, tags[0], name)
    rows, seen = [], set()
    # the draw first: a drawn molecule is a row whether or not branch A has run for it, at
    # the molecule directory it has (any tag) or will have (the first tag)
    draw_file = d / "draw.dat"
    if draw_file.is_file():
        for r in dat.read_table(draw_file):
            qid = r["qm9_index"]
            tag = next((t for t in tags if basins_mod.done(qid, t, root=root)), tags[0])
            mol = basins_mod.molecule_for(qid, tag, root=root)
            smiles = _smiles_of(mol, qid, tag, root) or str(r.get("smiles", ""))
            seen.add(qid)
            rows.append(_select_row(root, tag, qid, mol, smiles, mem, pinned, True, classes=str(r.get("classes", "-"))))
    for tag in tags:
        for qid, mol in molecules_with_branch_a(root, tag):
            if qid in seen:
                continue                                   # the draw, or the first tag, wins
            seen.add(qid)
            rows.append(_select_row(root, tag, qid, mol, _smiles_of(mol, qid, tag, root), mem, pinned, False))
    rows = apply_limit(rows, limit, stratify)
    strata = {}
    for r in rows:
        s = strata.setdefault(r["stratum"], dict(STRATUM=r["stratum"], N_MOLECULES=0, N_PINNED=0))
        s["N_MOLECULES"] += 1
        s["N_PINNED"] += int(r["pinned"])
    cls_rows = class_rows(rows, (("N_MOLECULES", lambda r: 1), ("N_WITH_BASINS", lambda r: r["has_basins"]),
                                 ("N_WITH_FRAMES", lambda r: r["has_frames"])))
    info = dict(NAME=str(name), TAGS=tags, ROOT=str(root), MEMBERSHIP_FILE=str(membership_file),
                DRAW_FILE=str(draw_file) if draw_file.is_file() else "-", PINNED=list(pinned),
                LIMIT=int(limit or 0), STRATIFY=bool(stratify), N_MOLECULES=len(rows),
                N_DRAWN=sum(1 for r in rows if r["drawn"]),
                N_WITH_BASINS=sum(1 for r in rows if r["has_basins"]),
                N_WITH_FRAMES=sum(1 for r in rows if r["has_frames"]),
                N_IN_TRAINING=sum(1 for r in rows if r["in_training"] == "true"),
                N_PINNED_PRESENT=sum(1 for r in rows if r["pinned"]), SECONDS=time.time() - t0)
    d.mkdir(parents=True, exist_ok=True)
    strata_rows = [strata[k] for k in sorted(strata)]
    missing = prop.write(d / (STEP_SELECT + ".toml"), {"Calculation_Info": info, "Stratum": strata_rows, "Class": cls_rows},
                         SELECT_SCHEMA, prop.NORMAL_TERMINATION, PROGNAME_SELECT)
    if missing:
        raise RuntimeError("select.toml keys outside the schema: {}".format(missing))
    dat.write_table(d / (STEP_SELECT + ".dat"), rows, list(SELECT_ROW_SCHEMA), SELECT_ROW_SCHEMA)
    rep = report.Report(PROGNAME_SELECT, "the molecule list of Dataset {!r}".format(name))
    rep.section("conventions")
    for k in ("TAGS", "DRAW_FILE", "LIMIT", "STRATIFY", "N_MOLECULES", "N_DRAWN", "N_WITH_BASINS", "N_WITH_FRAMES",
              "N_IN_TRAINING", "N_PINNED_PRESENT"):
        rep.kv(k, info[k])
    rep.section("per stratum")
    rep.table(["stratum", "molecules", "pinned"], [[s["STRATUM"], s["N_MOLECULES"], s["N_PINNED"]] for s in strata_rows])
    rep.section("per class (a molecule counts for every class it is in)")
    rep.table(["class", "molecules", "with basins", "with frames"],
              [[c["CLASS"], c["N_MOLECULES"], c["N_WITH_BASINS"], c["N_WITH_FRAMES"]] for c in cls_rows])
    rep.section("per molecule")
    rep.table(["qm9_index", "tag", "smiles", "heavy", "stratum", "classes", "in_training", "pinned", "basins", "frames"],
              [[r["qm9_index"], r["tag"], r["smiles"], r["n_heavy"], r["stratum"], r["classes"], r["in_training"],
                "yes" if r["pinned"] else "-", r["n_basins"] if r["has_basins"] else "-",
                r["n_frames"] if r["has_frames"] else "-"] for r in rows])
    rep.note("pinned molecules are the msRRHO study's seven and are always test; the stratum (ring count x "
             "heteroatom pattern) is what the by-molecule test draw is balanced over; classes are the structure "
             "classes of configs/structure_classes.yaml (from draw.dat for a drawn molecule, else classified here); "
             "in_training is membership in MACE-OFF23's SPICE training file (data/training_sets/"
             "qm9_targets_membership.dat). A drawn molecule without basins / frames is listed with '-': the "
             "campaign's progress is this table.")
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


def _previous_mode(dataset_dir):
    """The `SPLIT_BY` of the Dataset already in `dataset_dir`, or None when there is none.
    Read from the Record, which is written with every build."""
    p = Path(dataset_dir) / (STEP + ".toml")
    if not p.is_file():
        return None
    try:
        rec = prop.load(p)
    except Exception:                                                # noqa: BLE001
        return None
    return (rec.get("Calculation_Info") or {}).get("SPLIT_BY") or None


def _previous_split(dataset_dir):
    """`{(qm9_index, generator, basin, k): split}` of the labelled frames of the last
    `index.dat`, and `{qm9_index: molecule_split}` -- what a rebuild keeps."""
    p = Path(dataset_dir) / "index.dat"
    frames_prev, mols_prev = {}, {}
    if p.is_file():
        for r in dat.read_table(p):
            mols_prev[r["qm9_index"]] = r["molecule_split"]
            # a held-out frame's split was the rule, not a draw: nothing to keep (a later build
            # that trains its generator draws it fresh)
            if r["split"] != "pool" and r.get("held_out_generator", "no") != "yes":
                frames_prev[(r["qm9_index"], r["generator"], int(r["basin"]), int(r["k"]))] = r["split"]
    return frames_prev, mols_prev


def _rng(seed, *parts):
    """A generator seeded from `seed` and the names of what it draws for -- so a draw for
    one molecule (or stratum) does not move when another molecule is added or labelled."""
    import hashlib
    h = hashlib.sha256("|".join([str(seed)] + [str(x) for x in parts]).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def valid_probes(seed, qid, key, n_atoms, k_max=VALID_PROBE_KMAX):
    """The frame's own fixed probe set, [k_max, 3N] of +-1 (S0-C-67).

    The generator is seeded by the frame's IDENTITY -- the same rule the split itself uses
    (`_rng(seed, "frame", ...)`) -- so the set is reproducible from the Record's SEED alone,
    is independent between frames, and does NOT move when the Label is recomputed at the
    same level. Nothing here reads `H_r`.
    """
    rng = _rng(seed, "probe", qid, key[0], key[1], key[2])
    return rng.choice([-1, 1], size=(int(k_max), 3 * int(n_atoms))).astype(np.int8)


def frame_draw(seed, qid, key):
    """One number in [0, 1) for a frame, from its own generator: the by-frame split reads
    it against the fractions, so the frame's split depends on nothing else."""
    return float(_rng(seed, "frame", qid, key[0], key[1], key[2]).random())


def merged_file(dataset_dir, name, level):
    """`mace_<name>.<level>.extxyz`: the three labelled splits in one file."""
    return Path(dataset_dir) / "mace_{}.{}.extxyz".format(name, level)


def build(root, tags, name, level=frame_labels.DEFAULT_LEVEL, split_by=DEFAULT_SPLIT_MODE, valid_fraction=None,
          test_fraction=None, seed=SEED, mace_level=None, selection=None, pinned=PINNED,
          keep_previous=True, purpose=None, train_generators=TRAIN_GENERATORS, resplit=False):
    """The Dataset: split every Frame set of the selected molecules (`split_by`: "molecule",
    the production split since S0-C-65 -- whole test molecules, MACE-OFF's granularity -- or
    "frame", the smoke / fit split; module docstring), write the four split files, the merged `mace_<name>.<level>.extxyz`,
    `index.dat` and the Record. `selection`: rows as `select` returns them (default: read
    `select.dat` under the first tag; run `select` first). The fractions default to the
    mode's (FRAME_* or the by-molecule ones). `train_generators`: the generators whose
    frames may train (default basin only, S0-C-54); every labelled frame of another
    generator is routed to test as a held-out frame, before and above the draw.

    A rebuild KEEPS the previous `index.dat`'s decisions (`keep_previous`): a molecule's
    train/test side and every already-labelled frame's split stay what they were; only
    new molecules and newly labelled frames are drawn, each with a generator seeded from
    `seed` and its own name, so nothing moves because something else was added. That is
    what "set at write time, never recomputed" means across the label Batches that fill
    the pool.

    A rebuild that CHANGES `split_by` is refused (`resplit=True` overrides): keeping the old
    decisions under a new rule would leave one index with two split schemes in it -- some
    molecules split by frame, the newly labelled ones by molecule -- and no reader could say
    what the test split means. `resplit=True` discards the previous decisions and draws the
    whole Dataset again under the new mode."""
    t0 = time.time()
    if split_by not in SPLIT_MODES:
        raise ValueError("split_by must be one of {}, not {!r}".format(SPLIT_MODES, split_by))
    train_generators = tuple(train_generators)
    unknown = [g for g in train_generators if g not in frames_mod.GENERATORS]
    if unknown or not train_generators:
        raise ValueError("train_generators must be a non-empty subset of {}; got {!r}".format(
            frames_mod.GENERATORS, train_generators))
    held_out_generators = tuple(g for g in frames_mod.GENERATORS if g not in train_generators)
    by_frame = split_by == "frame"
    if valid_fraction is None:
        valid_fraction = FRAME_VALID_FRACTION if by_frame else VALID_FRACTION
    if test_fraction is None:
        test_fraction = FRAME_TEST_FRACTION if by_frame else TEST_FRACTION
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
    previous_mode = _previous_mode(d)
    if previous_mode and previous_mode != split_by and not resplit:
        raise ValueError(
            "this Dataset was split by {0} and the build asks for {1}: a rebuild would mix two split schemes in one "
            "index (the old molecules by {0}, the new frames by {1}) and the test split would mean neither. Pass "
            "resplit=True (04_dataset.py --resplit) to discard the previous decisions and draw it all again by {1}, "
            "or keep --split-by {0}.".format(previous_mode, split_by))
    if resplit:
        keep_previous = False
    prev_frames, prev_mols = _previous_split(d) if keep_previous else ({}, {})

    # ---- test molecules: pinned + the previous index's (+ by molecule: a per-stratum draw of the NEW ones)
    test_mols = {r["qm9_index"] for r in mols if r["qm9_index"] in pinned}
    test_mols.update(q for q, s in prev_mols.items() if s == "test")
    if not by_frame:
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
    n_valid_probe_frames = 0
    for r in mols:
        qid, mol = r["qm9_index"], Path(r["molecule_dir"])
        classes = str(r.get("classes") or "") or classes_of(r.get("smiles", ""))
        msplit = "test" if qid in test_mols else "train"
        frames_here = _frames_of(mol, level, mace_level)
        labelled = [(key, a, lab, ef) for key, a, lab, ef, _st in frames_here if lab is not None]
        n_stale = sum(1 for _k, _a, _l, _e, st in frames_here if st)
        # the split of every labelled frame: the previous index's where it has one; else, by
        # frame, its own draw against the fractions; by molecule, the molecule's side with the
        # NEW frames drawn (per-molecule generator) so the molecule reaches the valid fraction
        frame_split = {key: prev_frames[(qid,) + key] for key, _a, _l, _e in labelled if (qid,) + key in prev_frames}
        new = [key for key, _a, _l, _e in labelled if key not in frame_split]
        if msplit == "test":
            frame_split.update((key, "test") for key in new)
        elif by_frame:
            for key in new:
                u = frame_draw(seed, qid, key)
                frame_split[key] = "test" if u < test_fraction else ("valid" if u < test_fraction + valid_fraction else "train")
        else:
            n_valid_have = sum(1 for s in frame_split.values() if s == "valid")
            want = int(round(float(valid_fraction) * len(labelled))) - n_valid_have
            picked = set()
            if new and want > 0:
                pick = _rng(seed, "valid", qid).choice(len(new), size=min(want, len(new)), replace=False)
                picked = {new[i] for i in pick}
            frame_split.update((key, "valid" if key in picked else "train") for key in new)
        counts = dict(train=0, valid=0, test=0, pool=0, held_out=0)
        for key, a, lab, ef, _st in frames_here:
            held_out = False
            if lab is None:
                split, atoms, levels, ver = "pool", a, [mace_level], "-"
            else:
                split = frame_split[key]
                # the held-out generators (S0-C-54): to test, above the draw and the previous index
                if key[0] not in train_generators:
                    split, held_out = "test", True
                atoms, levels, ver = lab, [mace_level, level], str(lab.info.get("orca_version", "-"))
                versions.add(ver)
                if lab.info.get("hessian") is not None:
                    atoms.info["valid_probes"] = valid_probes(seed, qid, key, len(atoms)).reshape(-1)
                    atoms.info["has_valid_probes"] = True
                    n_valid_probe_frames += 1
            fp = str(a.info.get("engine_params_sha256", "-"))
            fingerprints.add(fp)
            counts[split] += 1
            counts["held_out"] += int(held_out)
            split_frames[split].append((atoms, split))
            index.append(dict(qm9_index=qid, tag=r["tag"], basin=key[1], generator=key[0], k=key[2], split=split,
                              molecule_split=msplit, held_out_generator="yes" if held_out else "no",
                              levels=";".join(levels), seed=int(a.info.get("seed", 0)),
                              engine_params_sha256=fp, orca_version=ver, in_training=r["in_training"],
                              stratum=r["stratum"], classes=classes, file="{}.{}.extxyz".format(split, level),
                              row=len(split_frames[split]) - 1, engine_file=str(ef)))
        per_mol.append(dict(QM9_INDEX=qid, TAG=r["tag"], STRATUM=r["stratum"], MOLECULE_SPLIT=msplit,
                            PINNED=qid in pinned, N_FRAMES=len(frames_here), N_LABELLED=len(labelled), N_STALE=n_stale,
                            N_TRAIN=counts["train"], N_VALID=counts["valid"], N_TEST=counts["test"],
                            N_TEST_HELD_OUT=counts["held_out"], N_POOL=counts["pool"], classes=classes))

    # ---- files -------------------------------------------------------------------------
    d.mkdir(parents=True, exist_ok=True)
    split_rows = []
    for s in SPLITS:
        path = d / "{}.{}.extxyz".format(s, level)
        n_mol = len({row["qm9_index"] for row in index if row["split"] == s})
        if split_frames[s]:
            _write_split(path, split_frames[s], reference=(s != "pool"))
        elif path.is_file():
            path.unlink()
        split_rows.append(dict(SPLIT=s, N_FRAMES=len(split_frames[s]), N_MOLECULES=n_mol, FILE=str(path) if split_frames[s] else None))
    merged = merged_file(d, name, level)
    labelled_all = split_frames["train"] + split_frames["valid"] + split_frames["test"]
    if labelled_all:
        _write_split(merged, labelled_all, reference=True)
    elif merged.is_file():
        merged.unlink()
    dat.write_table(d / "index.dat", index, list(INDEX_SCHEMA), INDEX_SCHEMA)
    cls_rows = class_rows(per_mol, (("N_MOLECULES", lambda m: 1), ("N_FRAMES", lambda m: m["N_FRAMES"]),
                                    ("N_LABELLED", lambda m: m["N_LABELLED"]), ("N_TRAIN", lambda m: m["N_TRAIN"]),
                                    ("N_VALID", lambda m: m["N_VALID"]), ("N_TEST", lambda m: m["N_TEST"]),
                                    ("N_TEST_HELD_OUT", lambda m: m["N_TEST_HELD_OUT"]),
                                    ("N_POOL", lambda m: m["N_POOL"])))
    for m in per_mol:
        m.pop("classes")
    info = dict(NAME=str(name), TAGS=tags, LEVEL=level, MACE_LEVEL=mace_level, SEED=int(seed), SPLIT_BY=split_by,
                RESPLIT=bool(resplit),
                N_VALID_PROBE_FRAMES=int(n_valid_probe_frames), VALID_PROBE_SOURCE="file",
                VALID_PROBE_KMAX=int(VALID_PROBE_KMAX), VALID_PROBE_MODE=VALID_PROBE_MODE,
                VALID_FRACTION=float(valid_fraction), TEST_FRACTION=float(test_fraction), PINNED=list(pinned),
                PURPOSE=str(purpose or ("judge" if pinned else "fit")),
                TRAIN_GENERATORS=list(train_generators), HELD_OUT_GENERATORS=list(held_out_generators),
                N_TRAIN_BASIN=len(split_frames["train"]),
                N_TEST_HELD_OUT=sum(m["N_TEST_HELD_OUT"] for m in per_mol),
                N_MOLECULES=len(mols), N_TEST_MOLECULES=len(test_mols), N_TRAIN_MOLECULES=len(mols) - len(test_mols),
                N_FRAMES=len(index), N_LABELLED=sum(1 for row in index if row["split"] != "pool"),
                N_TRAIN=len(split_frames["train"]), N_VALID=len(split_frames["valid"]),
                N_TEST=len(split_frames["test"]), N_POOL=len(split_frames["pool"]),
                N_HESSIAN_FRAMES=sum(1 for a, _s in labelled_all if a.info.get("hessian") is not None),
                N_TRAIN_HESSIAN=sum(1 for a, _s in split_frames["train"] if a.info.get("hessian") is not None),
                REPLAY_R4_FRAMES=REPLAY_PER_HESSIAN_FRAME_R4 * sum(1 for a, _s in split_frames["train"] if a.info.get("hessian") is not None),
                N_STALE=sum(m["N_STALE"] for m in per_mol), KEPT_PREVIOUS=bool(prev_frames or prev_mols),
                MERGED_FILE=str(merged) if labelled_all else "-",
                ENGINE_PARAMS_SHA256=sorted(fingerprints), ORCA_VERSIONS=sorted(versions), SECONDS=time.time() - t0)
    missing = prop.write(d / (STEP + ".toml"), {"Calculation_Info": info, "Split": split_rows, "Molecule": per_mol,
                                                 "Class": cls_rows},
                         SCHEMA, prop.NORMAL_TERMINATION, PROGNAME)
    if missing:
        raise RuntimeError("dataset.toml keys outside the schema: {}".format(missing))
    _write_report(d / (STEP + ".out"), info, split_rows, per_mol, cls_rows)
    return dict(info=info, splits=split_rows, molecules=per_mol, classes=cls_rows, index=index, dir=d)


#: MACE-torch's default training keys (mace/tools/arg_parser.py: --energy_key REF_energy,
#: --forces_key REF_forces); the Hessian key is this repository's, read by 05_train.
REF_ENERGY_KEY = "REF_energy"
REF_FORCES_KEY = "REF_forces"
REF_HESSIAN_KEY = "REF_hessian"
REF_PROBES_KEY = "REF_valid_probes"                        # the fork's --valid_probes_key


def _write_split(path, atoms_list, reference=True):
    """Write frames `(atoms, split)`: energy / forces on the calculator, `hessian` flat in
    info with `has_hessian`, the `split` key, the valid split's fixed probes (`valid_probes`
    flat with `has_valid_probes`, S0-C-67) and -- for the labelled splits (`reference`) --
    the same values again under REF_energy / REF_forces / REF_hessian / REF_valid_probes."""
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io import write
    out = []
    for a, split in atoms_list:
        b = a.copy()
        e, f = float(a.get_potential_energy()), np.asarray(a.get_forces(), dtype=float)
        b.calc = SinglePointCalculator(b, energy=e, forces=f)
        b.info = dict(a.info)
        b.info["split"] = split
        if a.info.get("hessian") is not None:
            b.info["hessian"] = np.asarray(a.info["hessian"], dtype=float).reshape(-1)
            b.info["has_hessian"] = True
        else:
            b.info.pop("hessian", None)
            b.info["has_hessian"] = False
        if reference:
            b.info[REF_ENERGY_KEY] = e
            b.arrays[REF_FORCES_KEY] = f
            if b.info["has_hessian"]:
                b.info[REF_HESSIAN_KEY] = b.info["hessian"]
            if b.info.get("has_valid_probes"):
                b.info[REF_PROBES_KEY] = np.asarray(b.info["valid_probes"], dtype=int).reshape(-1)
        out.append(b)
    write(str(path), out, format="extxyz")


def read_index(dataset_dir):
    return dat.read_table(Path(dataset_dir) / "index.dat")


def _write_report(path, info, split_rows, per_mol, cls_rows=()):
    rep = report.Report(PROGNAME, "Dataset {!r} at {}".format(info["NAME"], info["LEVEL"]))
    rep.section("conventions")
    for k in ("TAGS", "LEVEL", "MACE_LEVEL", "SEED", "SPLIT_BY", "VALID_FRACTION", "TEST_FRACTION", "TRAIN_GENERATORS",
              "HELD_OUT_GENERATORS", "N_MOLECULES", "N_TEST_MOLECULES", "N_TRAIN_MOLECULES", "N_FRAMES", "N_LABELLED",
              "N_HESSIAN_FRAMES", "N_TRAIN_HESSIAN", "REPLAY_R4_FRAMES", "N_TRAIN_BASIN", "N_TEST_HELD_OUT", "N_STALE",
              "KEPT_PREVIOUS", "MERGED_FILE", "ENGINE_PARAMS_SHA256", "ORCA_VERSIONS",
              "VALID_PROBE_SOURCE", "VALID_PROBE_KMAX", "VALID_PROBE_MODE", "N_VALID_PROBE_FRAMES"):
        rep.kv(k, info[k])
    rep.section("per split")
    rep.table(["split", "frames", "molecules", "file"],
              [[r["SPLIT"], r["N_FRAMES"], r["N_MOLECULES"], Path(r["FILE"]).name if r["FILE"] else "-"] for r in split_rows])
    rep.section("per class (a molecule counts for every class it is in; test = held-out generator frames + the drawn test frames)")
    rep.table(["class", "molecules", "frames", "labelled", "train", "valid", "test", "held-out", "pool"],
              [[c["CLASS"], c["N_MOLECULES"], c["N_FRAMES"], c["N_LABELLED"], c["N_TRAIN"], c["N_VALID"], c["N_TEST"],
                c["N_TEST_HELD_OUT"], c["N_POOL"]] for c in cls_rows])
    rep.section("per molecule")
    rep.table(["qm9_index", "tag", "stratum", "split", "pinned", "frames", "labelled", "stale", "train", "valid", "test",
               "held-out", "pool"],
              [[m["QM9_INDEX"], m["TAG"], m["STRATUM"], m["MOLECULE_SPLIT"], "yes" if m["PINNED"] else "-", m["N_FRAMES"],
                m["N_LABELLED"], m["N_STALE"], m["N_TRAIN"], m["N_VALID"], m["N_TEST"], m["N_TEST_HELD_OUT"], m["N_POOL"]]
               for m in per_mol])
    rep.note("TRAIN_GENERATORS (S0-C-54, ADR 0005): only frames of these generators may enter train or valid; every "
             "labelled frame of a held-out generator ({}) is in test, whatever the draw or the previous index said "
             "(`held_out_generator = yes` in index.dat). The judge reads them as reference rows; training never "
             "sees them.".format(", ".join(info["HELD_OUT_GENERATORS"]) or "none"))
    if info["SPLIT_BY"] == "frame":
        rep.note("SPLIT_BY frame (the smoke / fit mode; production until S0-C-65): test = the pinned seven as whole molecules + TEST_FRACTION of "
                 "every other molecule's labelled frames; valid = VALID_FRACTION of those frames; train = the rest. Every "
                 "frame is drawn on its own (a generator seeded from the seed and the frame's name), so its split never "
                 "depends on what else is labelled and the fractions are expectations. pool = frames without a label at "
                 "LEVEL yet.")
    else:
        rep.note("SPLIT_BY molecule (production, S0-C-65 -- MACE-OFF's granularity): test = whole molecules (the "
                 "pinned seven + a per-stratum draw of TEST_FRACTION), so conformers of one molecule never sit on both "
                 "sides; valid = VALID_FRACTION of the TRAINING molecules' labelled frames, drawn by frame (MACE-OFF "
                 "splits its 95 % pool by configuration too); train = the rest; pool = frames without a label at LEVEL "
                 "yet, whatever their molecule's split. Every test frame is a molecule the fine-tune never saw, so the "
                 "judge's interpolation distribution is empty and its gate row is a generalisation reading.")
    if info.get("RESPLIT"):
        rep.note("RESPLIT: the previous index.dat's decisions were discarded and the whole Dataset drawn again (the "
                 "split mode changed). Numbers from a run against the earlier split are not comparable with these.")
    rep.note("The split is set here and never recomputed: a rebuild keeps every decision of the previous index.dat and "
             "draws only the new molecules and newly labelled frames, each with its own seeded generator. A label file "
             "whose geometry differs from the engine file's (stale) is ignored. train/valid/test files carry the "
             "REFERENCE E-F-H (also as REF_energy / REF_forces / REF_hessian, MACE-torch's keys; has_hessian says which "
             "frames carry one) and are repeated in MERGED_FILE, the single xyz for training; pool carries the engine's.")
    rep.note("VALID_PROBE_* (S0-C-67, PHL's fixed-vector protocol): every labelled frame with a Hessian carries its "
             "own [VALID_PROBE_KMAX, 3N] Rademacher set, drawn HERE from the frame's identity (qm9_index | generator "
             "| basin | k) and this Record's SEED, and written as valid_probes / REF_valid_probes -- every labelled "
             "split, because mace evaluates the loss on the training split too and the loss draws nothing in eval "
             "mode. What is particular to valid is the USE. The training loss "
             "draws fresh probes every step; the VALIDATION loss reads the first k rows of this set and draws nothing, "
             "so the epoch-to-epoch difference is free of draw noise and nothing depends on the Label's bytes -- a "
             "label recomputed at the same level leaves the probes untouched. The rows are nested, so k may be changed "
             "on the command line up to VALID_PROBE_KMAX without rebuilding; changing VALID_PROBE_KMAX itself, or the "
             "SEED, needs a rebuild of the valid file and makes the readings incomparable with the earlier runs'.")
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
