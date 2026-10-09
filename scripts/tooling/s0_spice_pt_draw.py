"""The Replay draw: one seed, one draw, from SPICE's TRAIN split.

TOOLING. Produces no scientific number of its own; it selects frames, once, by seed, and
writes the list of ids beside them so that the same draw can be made again anywhere.

    python scripts/tooling/s0_spice_pt_draw.py --n 5000 --seed 0
    python scripts/tooling/s0_spice_pt_draw.py --n 30000 --seed 0 --weight 1  --out $S0_RUNS_ROOT/spice/spice_pt_replay30k_w1.extxyz
    python scripts/tooling/s0_spice_pt_draw.py --n 30000 --seed 0 --weight 10 --out $S0_RUNS_ROOT/spice/spice_pt_replay30k_w10.extxyz
    python scripts/tooling/s0_spice_pt_draw.py --n 3 --seed 0 --source tests/data/spice_tiny/train_large_neut_no_bad_clean.xyz \\
        --forgetting-ids tests/data/.../spice_test_x.extxyz.ids.dat --out /tmp/x.extxyz

WHAT IT WRITES. `<out>` (extxyz): the first N ELIGIBLE frames of ONE seeded permutation
of the source file (`default_rng(seed).permutation(n_frames)`), so a smaller `--n` is a
prefix of a larger one with the same seed (the S0 scan rows R0-R4 were nested); every
frame carries `config_weight = --weight` (1.0 by default; round 1's production draw is the
30,000-frame size written twice, at weights 1 and 10),
`REF_energy` / `REF_forces` (MACE-torch's keys), `spice_index` (its row in the source)
and no Hessian. `<out>.ids.dat`: one row per drawn frame (source index, SMILES, config
type, atoms, the frame's molecule key and that molecule's frame count in the draw, so
the coverage of a by-frame draw is a number). `<out>.valid.extxyz`: `--n-valid` frames
(200) from the END of the same permutation -- the pretraining head's own validation
set, the same for every draw (mace would otherwise take `--valid_fraction` of the Replay
file itself). `<out>.toml`: the Record (`[Replay]`: DOI, SPLIT, SEED, N, WEIGHT,
N_MOLECULES, FRAMES_PER_MOLECULE_MAX, the exclusions).

THE OPTIONAL ELEMENT FILTER. With `--elements` (a comma list, e.g. `H,C,N,O,F` -- the
element set of the fine-tune's Dataset, made explicit), a frame carrying any element
outside the set is skipped like the molecule exclusions, in the same single pass; the draw
and its validation companion are taken from the frames that remain. The frame's own atoms
are the judge (not its SMILES), and the declared set is recorded (`ELEMENTS` /
`N_SKIPPED_ELEMENTS`).

THE OPTIONAL COVERAGE RULE. With `--coverage min1`, every eligible molecule contributes at
least one frame: the coverage tier takes each molecule's first frame the permutation meets
(in permutation order), and the fill tier completes the draw to `--n` in permutation order;
a smaller `--n` stays the prefix. `--n` must be at least the number of eligible molecules
(else the tool refuses and writes nothing). The rule is recorded (`COVERAGE`) with the
eligible-molecule count beside it (`N_ELIGIBLE_MOLECULES`); the default `by-frame` keeps the
plain first-`N` behaviour.

WHAT IT EXCLUDES, AND THE TWO ASSERTIONS. The Replay must share no molecule
with the forgetting draw (`s0_spice_test_draw.py`'s `<file>.ids.dat`, `--forgetting-ids`)
and none of the in_distribution molecules (`judge.IN_DISTRIBUTION`, the four shipped
molecules MACE-OFF23 was trained on). SPICE's test split is BY FRAME (15,542 of the
train file's 17,132 molecules also have test frames; `data/training_sets/mace-off23_
spice_index.dat`), so a uniform draw that merely refused on overlap could never write a
file: the walk over the permutation SKIPS every frame whose molecule (any fragment of
its SMILES, at the connectivity level -- the loosest, so the exclusion is the widest)
is in either set, and the two assertions are then checked on what is about to be
written. Either failing exits 2 and writes nothing; so does an absent forgetting ids
file (the tool cannot promise (a) without it), a source file that does not exist (the
DOI is named), or a permutation that runs out of eligible frames before N.

The source is the train file of `data.training_set.settings()` (`S0_TRAINING_SETS_ROOT`
moves it). The frames are read by BYTE OFFSET: one header pass over the 5 GB file finds
every frame's position and SMILES, RDKit sees each distinct SMILES string once (17,132),
and only the drawn frames are parsed as atoms.
"""
import argparse
import io
import re
import sys
import time
from pathlib import Path


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))
from openqha.data import dataset as dataset_mod, training_set    # noqa: E402
from openqha.store import dat, property as prop                  # noqa: E402
from openqha_hessian import judge                                # noqa: E402

PROGNAME = "openQHA spice_pt_draw"
DEFAULT_OUT = ROOT / "data" / "training_sets" / "spice_pt_{n}.extxyz"
DEFAULT_FORGETTING_IDS = ROOT / "data" / "training_sets" / "spice_test_5000.extxyz.ids.dat"
#: the molecule identity level of the exclusions: connectivity (InChIKey's first block), the widest
KEY_LEVEL = "connectivity"
DEFAULT_N_VALID = 200
COVERAGE_BY_FRAME = "by-frame"
COVERAGE_MIN1 = "min1"

ID_ROW = {
    "index": ("Integer", None, "the frame's position in the source file, 0-based"),
    "smiles": ("String", None, "the frame's SMILES, as the source gives it"),
    "config_type": ("String", None, "SPICE's config_type, or - "),
    "n_atoms": ("Integer", None, "atoms in the frame"),
    "molecule_key": ("String", None, "the molecule's identity at the connectivity level (fragments ';'-joined)"),
    "frames_of_molecule": ("Integer", None, "how many frames of this molecule the draw holds (a by-frame draw's coverage)"),
}

SCHEMA = {
    "Replay": {
        "DOI": ("String", None, "the SPICE release's DOI"),
        "SPLIT": ("String", None, "train: the file the frames come from"),
        "SOURCE": ("String", None, "the source file"),
        "N_SOURCE": ("Integer", None, "frames in the source file"),
        "SEED": ("Integer", None, "the permutation's seed"),
        "N": ("Integer", None, "frames written (the first N eligible of the permutation)"),
        "WEIGHT": ("Double", None, "config_weight on every frame"),
        "N_MOLECULES": ("Integer", None, "distinct molecules (connectivity keys) in the draw"),
        "FRAMES_PER_MOLECULE_MAX": ("Integer", None, "the most frames one molecule contributes"),
        "N_VALID": ("Integer", None, "frames in the companion validation file (from the end of the permutation)"),
        "KEY_LEVEL": ("String", None, "the identity level of the exclusions"),
        "COVERAGE": ("String", None, "the draw's coverage rule: by-frame (the permutation's first N) or min1 (every eligible molecule once, then fill)"),
        "ELEMENTS": ("String", None, "the declared element set the frames were filtered against ('-' when unfiltered)"),
        "FORGETTING_IDS": ("String", None, "the forgetting draw's ids file the molecules were excluded against"),
        "N_FORGETTING_MOLECULES": ("Integer", None, "molecules of the forgetting draw"),
        "N_IN_DISTRIBUTION": ("Integer", None, "in_distribution molecules excluded"),
        "N_SKIPPED_FORGETTING": ("Integer", None, "frames of the permutation skipped for a forgetting-draw molecule"),
        "N_SKIPPED_IN_DISTRIBUTION": ("Integer", None, "frames skipped for an in_distribution molecule"),
        "N_SKIPPED_ELEMENTS": ("Integer", None, "frames skipped because an atom is outside the --elements set"),
        "N_SKIPPED_UNPARSED": ("Integer", None, "frames skipped because RDKit could not read their SMILES (no identity, no promise)"),
        "N_ELIGIBLE": ("Integer", None, "frames of the source that could have been drawn"),
        "N_ELIGIBLE_MOLECULES": ("Integer", None, "distinct molecules among the eligible frames"),
        "FILE": ("String", None, "the Replay file"),
        "IDS_FILE": ("String", None, "the ids file"),
        "VALID_FILE": ("String", None, "the companion validation file"),
        "SECONDS": ("Double", "s", "wall time"),
    },
}

_SMILES_RE = re.compile(r'smiles="([^"]*)"|smiles=(\S+)')
_CONFIG_RE = re.compile(r'config_type="([^"]*)"|config_type=(\S+)')


_SYMBOL_RE = re.compile(r'[A-Z][a-z]?$')


def scan_frames(path, elements=None):
    """One pass over an extxyz file: per frame (byte offset, byte length, n_atoms, smiles,
    config_type, elements_ok), reading headers only. Byte offsets, so a frame can be re-read
    alone. With `elements` (a set of symbols), every atom line's first token is validated and
    tested against it -- elements_ok is False for a frame holding an element outside the set,
    None when no set was asked for; the SMILES is not consulted for this, the atoms are."""
    out = []
    with open(path, "rb") as fh:
        while True:
            start = fh.tell()
            line = fh.readline()
            if not line:
                break
            s = line.strip()
            if not s:
                continue
            try:
                nat = int(s.split()[0])
            except ValueError:
                raise ValueError("{}: expected an atom count at byte {}, read {!r}".format(path, start, s[:40]))
            header = fh.readline().decode("utf-8", errors="replace")
            elements_ok = None
            if elements is None:
                for _ in range(nat):
                    fh.readline()
            else:
                elements_ok = True
                for _ in range(nat):
                    tok = fh.readline().split(None, 1)
                    if not tok:
                        raise ValueError("{}: expected an atom line at byte {}".format(path, fh.tell()))
                    sym = tok[0].decode("utf-8", errors="replace")
                    if not _SYMBOL_RE.match(sym):
                        raise ValueError("{}: expected an element symbol at byte {}, read {!r}".format(path, fh.tell(), sym[:12]))
                    if sym not in elements:
                        elements_ok = False
            m = _SMILES_RE.search(header)
            c = _CONFIG_RE.search(header)
            smiles = (m.group(1) if m and m.group(1) is not None else (m.group(2) if m else "")) or ""
            ctype = (c.group(1) if c and c.group(1) is not None else (c.group(2) if c else "")) or ""
            out.append((start, fh.tell() - start, nat, smiles, ctype, elements_ok))
    return out


def molecule_key(smiles, cache):
    """The frame's molecule identity: the connectivity keys of its fragments, sorted and
    ';'-joined (a dimer names both partners); None when RDKit cannot read any fragment.
    `cache` maps a SMILES string to its key, so each distinct string is parsed once."""
    if smiles in cache:
        return cache[smiles]
    keys = training_set.fragment_keys(smiles) if smiles else []
    if not keys or any(k is None for k in keys):
        cache[smiles] = None
        return None
    key = ";".join(sorted(k[KEY_LEVEL] for k in keys))
    cache[smiles] = key
    return key


def fragments_of(key):
    return set(key.split(";")) if key else set()


def forgetting_molecules(ids_file, cache):
    """The connectivity keys (per fragment) of every molecule in the forgetting draw's ids file."""
    keys = set()
    for r in dat.read_table(ids_file):
        k = molecule_key(str(r.get("smiles", "")), cache)
        keys |= fragments_of(k)
    return keys


def in_distribution_molecules(cache, membership=None, ids=judge.IN_DISTRIBUTION):
    """The connectivity keys of the in_distribution molecules, from the membership table's SMILES."""
    membership = dataset_mod.membership() if membership is None else membership
    keys = set()
    missing = []
    for qid in ids:
        row = membership.get(qid)
        if row is None:
            missing.append(qid)
            continue
        keys |= fragments_of(molecule_key(str(row["smiles"]), cache))
    if missing:
        raise RuntimeError("no SMILES for the in_distribution molecules {} in {}".format(missing, dataset_mod.MEMBERSHIP_FILE))
    return keys


def check_disjoint(drawn_keys, forgetting_keys, in_distribution_keys):
    """The two assertions on the molecule keys about to be written: (a) none
    shared with the forgetting draw, (b) none in_distribution. Raises AssertionError
    naming the offending molecules."""
    frags = set()
    for k in drawn_keys:
        frags |= fragments_of(k)
    shared = sorted(frags & set(forgetting_keys))
    if shared:
        raise AssertionError("(a) {} molecule(s) of the draw are in the forgetting draw: {}".format(
            len(shared), ", ".join(shared[:10]) + (" ..." if len(shared) > 10 else "")))
    ind = sorted(frags & set(in_distribution_keys))
    if ind:
        raise AssertionError("(b) {} in_distribution molecule(s) in the draw: {}".format(len(ind), ", ".join(ind)))


def classify(frames, forgetting_keys, in_distribution_keys, cache):
    """Every frame's class, once: `eligible`, `forgetting` (a fragment in the forgetting
    draw), `in_distribution`, `element` (an atom outside the `--elements` set) or `unparsed`
    (RDKit could not read the SMILES: no identity, so no promise, so excluded). Returns
    (classes list, counts dict)."""
    forgetting_keys, in_distribution_keys = set(forgetting_keys), set(in_distribution_keys)
    classes = []
    for _off, _len, _nat, smiles, _ct, elements_ok in frames:
        key = molecule_key(smiles, cache)
        if key is None:
            classes.append("unparsed")
            continue
        fr = fragments_of(key)
        if fr & forgetting_keys:
            classes.append("forgetting")
        elif fr & in_distribution_keys:
            classes.append("in_distribution")
        elif elements_ok is False:
            classes.append("element")
        else:
            classes.append("eligible")
    counts = {c: classes.count(c) for c in ("eligible", "forgetting", "in_distribution", "element", "unparsed")}
    return classes, counts


def draw(frames, n, seed, forgetting_keys, in_distribution_keys, cache, n_valid=DEFAULT_N_VALID,
         coverage=COVERAGE_BY_FRAME):
    """The draw, as `coverage` says. `by-frame`: the first `n` eligible frames of the seeded
    permutation. `min1`: the coverage tier -- every eligible molecule's first frame the
    permutation meets, in permutation order -- then the fill tier to `n`, also in
    permutation order. In both rules `n_valid` eligible frames come from the end of the
    permutation (never one of the drawn). Returns (selected indices, valid indices, the
    counts -- `classify`'s plus the eligible molecules')."""
    import numpy as np
    classes, counts = classify(frames, forgetting_keys, in_distribution_keys, cache)
    counts["molecules_eligible"] = len({molecule_key(frames[i][3], cache)
                                        for i, c in enumerate(classes) if c == "eligible"})
    perm = np.random.default_rng(int(seed)).permutation(len(frames))
    selected = []
    if coverage == COVERAGE_MIN1:
        first = {}
        for i in perm:
            i = int(i)
            if classes[i] != "eligible":
                continue
            k = molecule_key(frames[i][3], cache)
            if k not in first:
                first[k] = i
        coverage_frames = set(first.values())
        selected = [int(i) for i in perm if int(i) in coverage_frames]
        taken = set(selected)
        for i in perm:
            if len(selected) >= n:
                break
            i = int(i)
            if classes[i] == "eligible" and i not in taken:
                selected.append(i)
                taken.add(i)
    else:
        for i in perm:
            if len(selected) >= n:
                break
            if classes[int(i)] == "eligible":
                selected.append(int(i))
    selected_set = set(selected)
    valid = []
    for i in perm[::-1]:
        if len(valid) >= n_valid or int(i) in selected_set:
            break                               # enough, or the draw reached the end of the permutation
        if classes[int(i)] == "eligible":
            valid.append(int(i))
    return selected, valid, counts


def read_frame(path, offset, length):
    from ase.io import read
    with open(path, "rb") as fh:
        fh.seek(offset)
        chunk = fh.read(length).decode("utf-8", errors="replace")
    return read(io.StringIO(chunk), index=0, format="extxyz")


def prepare(atoms, weight, index, energy_key="REF_energy", forces_key="REF_forces"):
    if energy_key not in atoms.info:
        atoms.info[energy_key] = float(atoms.get_potential_energy())
    if forces_key not in atoms.arrays:
        atoms.arrays[forces_key] = atoms.get_forces()
    atoms.info.pop("REF_hessian", None)
    atoms.info.pop("hessian", None)
    atoms.info["config_weight"] = float(weight)
    atoms.info["spice_index"] = int(index)
    return atoms


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, required=True, help="frames to draw (round 1's production draw: 30000; the S0 ladder used 5000 / 17132 / 68528)")
    ap.add_argument("--seed", type=int, default=0, help="the permutation's seed; one seed, one draw, the whole campaign")
    ap.add_argument("--weight", type=float, default=1.0, help="config_weight on every frame (round 1: 1 and 10)")
    ap.add_argument("--elements", default=None,
                    help="comma-separated element symbols (e.g. H,C,N,O,F): a frame carrying any other "
                         "element is skipped -- the element set of the fine-tune's Dataset, made explicit")
    ap.add_argument("--coverage", choices=(COVERAGE_BY_FRAME, COVERAGE_MIN1), default=COVERAGE_BY_FRAME,
                    help="by-frame: the permutation's first N; min1: every eligible molecule at least once, "
                         "then filled in permutation order (needs --n >= the eligible molecules)")
    ap.add_argument("--source", default=None, help="the SPICE train file (default: data.training_set's)")
    ap.add_argument("--forgetting-ids", default=None,
                    help="the forgetting draw's ids file (default: data/training_sets/spice_test_5000.extxyz.ids.dat)")
    ap.add_argument("--out", default=None, help="the extxyz to write (default: data/training_sets/spice_pt_<n>.extxyz)")
    ap.add_argument("--n-valid", type=int, default=DEFAULT_N_VALID,
                    help="frames of the companion validation file, from the end of the permutation")
    ap.add_argument("--membership-file", default=None, help="the QM9 membership table with the in_distribution SMILES")
    args = ap.parse_args()
    t0 = time.time()

    elements = None
    elements_label = "-"
    if args.elements is not None:
        tokens = [tok.strip() for tok in args.elements.split(",")]
        bad = [tok for tok in tokens if not _SYMBOL_RE.match(tok)]
        if not tokens or bad:
            print("openQHA: --elements wants comma-separated element symbols (e.g. H,C,N,O,F); got {!r}{}"
                  .format(args.elements, " (not a symbol: {})".format(", ".join(map(repr, bad))) if bad else ""),
                  file=sys.stderr)
            return 2
        elements = frozenset(tokens)
        elements_label = ",".join(dict.fromkeys(tokens))

    from ase.io import write
    src = Path(args.source) if args.source else Path(training_set.settings()["train"])
    if not src.is_file():
        print("openQHA: no SPICE train file at {}.\n"
              "  The frames are not in this repository (the index is). Point S0_TRAINING_SETS_ROOT at the\n"
              "  MACE-OFF23 SPICE release ({}), or pass --source.".format(src, training_set.settings()["doi"]),
              file=sys.stderr)
        return 2
    ids_file = Path(args.forgetting_ids) if args.forgetting_ids else DEFAULT_FORGETTING_IDS
    if not ids_file.is_file():
        print("openQHA: no forgetting ids file at {}.\n"
              "  The Replay must share no molecule with the forgetting draw, which this tool cannot promise\n"
              "  without its ids file: run scripts/tooling/s0_spice_test_draw.py first, or pass --forgetting-ids."
              .format(ids_file), file=sys.stderr)
        return 2

    cache = {}
    frames = scan_frames(src, elements=elements)
    forgetting_keys = forgetting_molecules(ids_file, cache)
    membership = dataset_mod.membership(args.membership_file) if args.membership_file else None
    ind_keys = in_distribution_molecules(cache, membership=membership)
    selected, valid, counts = draw(frames, int(args.n), args.seed, forgetting_keys, ind_keys, cache,
                                   n_valid=args.n_valid, coverage=args.coverage)
    if args.coverage == COVERAGE_MIN1 and counts["molecules_eligible"] > int(args.n):
        print("openQHA: --coverage min1 needs --n >= the {} molecules among the eligible frames; got --n {}. "
              "Nothing written.".format(counts["molecules_eligible"], args.n), file=sys.stderr)
        return 2
    if len(selected) < int(args.n):
        skipped = "{} skipped for the forgetting draw, {} in_distribution".format(
            counts["forgetting"], counts["in_distribution"])
        if elements is not None:
            skipped += ", {} outside the declared elements {}".format(counts["element"], elements_label)
        skipped += ", {} unparsed".format(counts["unparsed"])
        print("openQHA: only {} eligible frames in {} after the exclusions ({}); --n {} cannot be drawn. "
              "Nothing written.".format(len(selected), src, skipped, args.n), file=sys.stderr)
        return 2
    drawn_keys = [molecule_key(frames[i][3], cache) for i in selected]
    try:
        check_disjoint(drawn_keys, forgetting_keys, ind_keys)
    except AssertionError as exc:
        print("openQHA: the draw fails the assertion {} -- nothing written.".format(exc), file=sys.stderr)
        return 2

    out = Path(args.out) if args.out else Path(str(DEFAULT_OUT).format(n=len(selected)))
    out.parent.mkdir(parents=True, exist_ok=True)
    per_mol = {}
    for k in drawn_keys:
        per_mol[k] = per_mol.get(k, 0) + 1
    atoms_list, rows = [], []
    for i, key in zip(selected, drawn_keys):
        off, length, nat, smiles, ctype, _elements_ok = frames[i]
        atoms_list.append(prepare(read_frame(src, off, length), args.weight, i))
        rows.append(dict(index=int(i), smiles=smiles or "-", config_type=ctype or "-", n_atoms=int(nat),
                         molecule_key=key, frames_of_molecule=per_mol[key]))
    write(str(out), atoms_list, format="extxyz")
    ids = out.with_suffix(out.suffix + ".ids.dat")
    dat.write_table(ids, rows, list(ID_ROW), ID_ROW)
    header = "# source {}\n# seed {}\n# n_source {}\n# n_drawn {}\n# weight {}\n".format(
        src, args.seed, len(frames), len(selected), args.weight)
    if elements is not None:
        header += "# elements {}\n".format(elements_label)
    ids.write_text(header + ids.read_text(encoding="utf-8"), encoding="utf-8")
    valid_file = out.with_suffix(".valid" + out.suffix)
    if valid:
        write(str(valid_file), [prepare(read_frame(src, frames[i][0], frames[i][1]), args.weight, i) for i in valid],
              format="extxyz")
    info = dict(DOI=training_set.settings()["doi"], SPLIT="train", SOURCE=str(src), N_SOURCE=len(frames),
                SEED=int(args.seed), N=len(selected), WEIGHT=float(args.weight), N_MOLECULES=len(per_mol),
                FRAMES_PER_MOLECULE_MAX=max(per_mol.values()) if per_mol else 0, N_VALID=len(valid),
                KEY_LEVEL=KEY_LEVEL, COVERAGE=args.coverage, ELEMENTS=elements_label, FORGETTING_IDS=str(ids_file),
                N_FORGETTING_MOLECULES=len(forgetting_keys),
                N_IN_DISTRIBUTION=len(ind_keys), N_SKIPPED_FORGETTING=counts["forgetting"],
                N_SKIPPED_IN_DISTRIBUTION=counts["in_distribution"], N_SKIPPED_ELEMENTS=counts["element"],
                N_SKIPPED_UNPARSED=counts["unparsed"],
                N_ELIGIBLE=counts["eligible"], N_ELIGIBLE_MOLECULES=counts["molecules_eligible"],
                FILE=str(out), IDS_FILE=str(ids),
                VALID_FILE=str(valid_file) if valid else "-", SECONDS=time.time() - t0)
    missing = prop.write(out.with_suffix(".toml"), {"Replay": info}, SCHEMA,
                         prop.NORMAL_TERMINATION, PROGNAME)
    if missing:
        raise RuntimeError("keys outside the schema: {}".format(missing))
    print("source     {} ({} frames; {} eligible after the exclusions)".format(src, len(frames), counts["eligible"]))
    element_note = "" if elements is None else ", {} outside the declared elements {}".format(counts["element"], elements_label)
    print("excluded   {} forgetting-draw molecules, {} in_distribution; skipped {} + {} frames of the permutation{}, {} unparsed".format(
        len(forgetting_keys), len(ind_keys), counts["forgetting"], counts["in_distribution"], element_note, counts["unparsed"]))
    if args.coverage == COVERAGE_MIN1:
        print("coverage   min1: all {} eligible molecules covered, then filled in permutation order".format(
            counts["molecules_eligible"]))
    print("drawn      {} frames with seed {}, config_weight {}: {} molecules, at most {} frames of one".format(
        len(selected), args.seed, args.weight, len(per_mol), info["FRAMES_PER_MOLECULE_MAX"]))
    print("written    {}\n           {}\n           {} ({} frames)\n           {}".format(
        out, ids, valid_file if valid else "-", len(valid), out.with_suffix(".toml")))
    print("\nthe fine-tune: 05_train.py --multiheads --pt-train-file {} --pt-valid-file {}".format(out, valid_file))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
