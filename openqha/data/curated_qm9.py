"""curatedQM9 — repaired geometries for the molecules QM9 could not characterise.

Senthil, Chakraborty & Ramakrishnan, *Chem. Sci.* **2021**, 12, 5566,
DOI [10.1039/D0SC05591C](https://doi.org/10.1039/D0SC05591C). Produced by the ConnGO
workflow: a connectivity-preserving four-layer re-optimisation whose top layer is
**B3LYP/6-31G(2df,p) — the same level as QM9 itself**, so a repaired geometry is
comparable with the rest of QM9 rather than being a different level of theory.

--------------------------------------------------------------------------------------
Why this module exists
--------------------------------------------------------------------------------------
QM9 ships 3054 molecules (2.3%) whose deposited B3LYP geometry and deposited SMILES are
**not the same molecule** — the Corina starting structure rearranged or dissociated
during optimisation while the index kept the original SMILES. Upstream lists them in
`uncharacterized.txt`.

Gate F7 removed all 3054. That is **over-kill**: the paper repaired 2988 of them and
found only **66 genuinely unstable** (52 of those containing –NNO–). Dropping ~3000
molecules to avoid 66 costs 2.2% of the dataset for no reason, provided the repaired
geometry is actually used.

`f7_mode` is the switch:

    drop_all   remove every molecule on the list          (the old behaviour, kept
                                                           so existing products stay
                                                           comparable)
    curated    use the repaired geometry; remove only
               those curatedQM9 does not contain          (the recommended default)
    off        do not apply F7 at all                     (debugging only)

--------------------------------------------------------------------------------------
The naming, measured rather than assumed
--------------------------------------------------------------------------------------
The archive does NOT use one naming scheme. Counted on the 133 661 files present:

    dsgdb9nsd_XXXXXX.xyz          130 831   untouched QM9 originals
    dsgdb9nsd_XXXXXX_new.xyz        1 257   repaired, QM9-style tag
    dsgdb9_XXXXXX_new.xyz           1 573   repaired, short tag

So **2830 files carry a `_new` suffix**, and a lookup that assumed a single pattern
would silently miss most of the repairs. `find()` tries all three and reports which one
matched, so `geometry_source` in a product says what was actually read.

Every file is in QM9's own extended-xyz format (line 2 carries the property row), so
this repository's QM9 parser needs no change.

--------------------------------------------------------------------------------------
The single-file form (ticket 17, 2026-09-21)
--------------------------------------------------------------------------------------
A cluster does not want 133 661 small files on its shared filesystem, and `data/qm9/`
is not in git. `scripts/tooling/s0_pack_curated_qm9.py` writes the whole archive into
ONE HDF5, `data/qm9/curated_qm9.h5`, ONE GROUP PER MOLECULE named `dsgdb9nsd_%06d`
whatever the file was called (user ruling 2026-09-21: the naming pattern is metadata,
not a key):

    /dsgdb9nsd_000058
      attrs   qm9_index  natoms  repaired (a `_new` file)  source_file (the archive's name)
              smiles_gdb17  smiles_relaxed  inchi_gdb17  inchi_relaxed
      species      (N,)  |S2       positions (N, 3) float64 A       -- lines 3 .. 2+N of the file
      charges      (N,)  float64   Mulliken e; NaN for a repaired file (they carry none)
      frequencies  (3N-6,) float64 cm^-1                            -- line 3+N

Not stored (rulings 2026-09-21): the property row (line 2: A B C mu ... Cv -- nothing in
the repository reads it) and the file's text (it doubled the size; the parsed form holds
everything a reader uses). The file's attributes carry the source, the census per naming
pattern, a sha256 of all the source text and the date. `find()` prefers the directory and,
when it is absent and the archive is present, RENDERS the one molecule asked for as a QM9
file (`render_qm9_text`: N, a `gdb <n>` header, the atom lines with the charge column when
there is one, the frequency line, the SMILES line, the InChI line -- the lines
`config.read_qm9_xyz` and `qm9_smiles_from_xyz` read) into `<archive dir>/_h5cache/
<source_file>` (once) and returns that path -- a campaign touches its 6 458 molecules, not
133 661. `atoms(n)`, `smiles(n)`, `frequencies(n)` read the parsed form directly. Iterating
the whole of QM9 (`training_set.qm9_targets`, the structure census) still needs the
directory. `S0_CURATED_QM9_H5` overrides the archive's location.
"""
import os
import re
from pathlib import Path

from .. import S0_ROOT

#: Directory name as distributed. Users drop the unpacked archive under `data/qm9/`.
DEFAULT_DIRNAME = "133660_curatedQM9_outof_133885"

#: Filename patterns, in the order they are tried. The order is not arbitrary: a
#: repaired geometry must win over the original, because the whole point of `curated`
#: mode is to use the repair.
PATTERNS = (
    ("repaired_qm9_tag", "dsgdb9nsd_{n:06d}_new.xyz"),
    ("repaired_short_tag", "dsgdb9_{n:06d}_new.xyz"),
    ("original", "dsgdb9nsd_{n:06d}.xyz"),
)

_INDEX_CACHE = {}

#: The single-file form: `data/qm9/curated_qm9.h5` (S0_CURATED_QM9_H5 overrides; a
#: `data.curated_qm9_h5` path in the configuration is tried before the default).
H5_NAME = "curated_qm9.h5"
#: Where `find()` extracts a molecule from the archive: beside it, one file per molecule
#: asked for, named as the archive named it.
CACHE_DIRNAME = "_h5cache"
_ARCHIVE_CACHE = {}


def root(cfg=None):
    """Where the archive is. `S0_CURATED_QM9` overrides everything.

    Searched rather than hard-coded, because this repository has broken four times on
    a directory that moved while a literal path stayed behind (S0-G-30, S0-G-31).
    """
    env = os.environ.get("S0_CURATED_QM9")
    if env:
        p = Path(env).expanduser()
        if not p.is_dir():
            raise FileNotFoundError(
                "S0_CURATED_QM9 points at {}, which is not a directory".format(p))
        return p

    name = DEFAULT_DIRNAME
    if cfg is not None:
        name = cfg.get("data", {}).get("curated_qm9_dir", name)

    for cand in (S0_ROOT / "data" / "qm9" / name,
                 S0_ROOT / "data" / name,
                 S0_ROOT.parent / "source-code" / name,
                 S0_ROOT.parent / name):
        if cand.is_dir():
            return cand
    return None


def available(cfg=None):
    """The directory or the archive."""
    return root(cfg) is not None or archive(cfg) is not None


# ====================================================================== the archive
def archive(cfg=None):
    """The HDF5 form's path, or None. `S0_CURATED_QM9_H5` overrides everything."""
    env = os.environ.get("S0_CURATED_QM9_H5")
    if env:
        p = Path(env).expanduser()
        if not p.is_file():
            raise FileNotFoundError("S0_CURATED_QM9_H5 points at {}, which is not a file".format(p))
        return p
    cands = []
    if cfg is not None:
        h = cfg.get("data", {}).get("curated_qm9_h5")
        if h:
            cands.append(Path(h) if Path(h).is_absolute() else S0_ROOT / "data" / "qm9" / h)
    cands += [S0_ROOT / "data" / "qm9" / H5_NAME, S0_ROOT / "data" / H5_NAME]
    for c in cands:
        if c.is_file():
            return c
    return None


def group_name(n):
    """The group of a QM9 index: `dsgdb9nsd_%06d`, whatever the file was called."""
    return "dsgdb9nsd_{:06d}".format(int(n))


def pattern_of(filename):
    """The PATTERNS name an archive file name belongs to (`original` when none matches)."""
    for name, rx in (("repaired_qm9_tag", re.compile(r"^dsgdb9nsd_\d{6}_new\.xyz$")),
                     ("repaired_short_tag", re.compile(r"^dsgdb9_\d{6}_new\.xyz$")),
                     ("original", re.compile(r"^dsgdb9nsd_\d{6}\.xyz$"))):
        if rx.match(str(filename)):
            return name
    return "original"


def qm9_float(s):
    """A number as QM9's files write it. Three exponent spellings occur: Python's, the
    Mathematica-style `-0.535689*^-6` (documented in `config.read_qm9_xyz`) and
    `-8.7796E^-6` -- 151 files of the curated archive carry the third (found by the
    packing of 2026-09-21; `float()` rejects it and so did the reader until then)."""
    return float(str(s).replace("*^", "e").replace("E^", "e").replace("e^", "e"))


def parse_qm9_text(text):
    """A QM9 file's text -> dict(natoms, species, positions (A), charges (NaN when the
    file carries none: the repaired ones), frequencies, smiles_gdb17, smiles_relaxed,
    inchi_gdb17, inchi_relaxed). Every QM9 exponent spelling is accepted (`qm9_float`)."""
    import numpy as np
    f = qm9_float
    lines = text.splitlines()
    n = int(lines[0].split()[0])
    species, pos, chg = [], [], []
    for row in lines[2:2 + n]:
        c = row.split()
        species.append(c[0])
        pos.append([f(v) for v in c[1:4]])
        chg.append(f(c[4]) if len(c) > 4 else float("nan"))
    freqs = [f(v) for v in lines[2 + n].split()] if len(lines) > 2 + n else []
    smi = lines[3 + n].split() if len(lines) > 3 + n else [""]
    inchi = lines[4 + n].split() if len(lines) > 4 + n else [""]
    return dict(natoms=n, species=species, positions=np.asarray(pos, dtype=float), charges=np.asarray(chg, dtype=float),
                frequencies=np.asarray(freqs, dtype=float), smiles_gdb17=smi[0], smiles_relaxed=smi[1] if len(smi) > 1 else smi[0],
                inchi_gdb17=inchi[0], inchi_relaxed=inchi[1] if len(inchi) > 1 else inchi[0])


def render_qm9_text(n, d):
    """A QM9-format file from a parsed record `d` (`parse_qm9_text`'s keys): the lines the
    repository's readers use, the property row replaced by `gdb <n>`."""
    import numpy as np
    chg = np.asarray(d["charges"], dtype=float)
    with_charge = len(chg) == len(d["species"]) and not np.isnan(chg).any()
    lines = [str(int(d["natoms"])), "gdb {}".format(int(n))]
    for i, (s, r) in enumerate(zip(d["species"], np.asarray(d["positions"], dtype=float))):
        row = "{}\t{: .10f}\t{: .10f}\t{: .10f}".format(s, r[0], r[1], r[2])
        if with_charge:
            row += "\t{: .6f}".format(chg[i])
        lines.append(row)
    lines.append("\t".join("{:.4f}".format(v) for v in np.asarray(d["frequencies"], dtype=float)))
    lines.append("{}\t{}".format(d["smiles_gdb17"], d["smiles_relaxed"]))
    lines.append("{}\t{}".format(d["inchi_gdb17"], d["inchi_relaxed"]))
    return "\n".join(lines) + "\n"


class Archive:
    """One opened `curated_qm9.h5` (module docstring: one group per molecule). Read-only;
    many workers may hold it open at once; a molecule is `f[group_name(n)]`, no listing."""

    def __init__(self, path):
        import h5py
        self.path = Path(path)
        self.h5 = h5py.File(str(self.path), "r")
        self.attrs = dict(self.h5.attrs)

    def __contains__(self, n):
        return group_name(n) in self.h5

    def __len__(self):
        return int(self.attrs.get("n_molecules", len(self.h5)))

    def group(self, n):
        return self.h5[group_name(n)]

    def filename(self, n):
        """The archive's own file name (`source_file`): `dsgdb9_000058_new.xyz` for a repair."""
        return str(self.group(n).attrs["source_file"])

    def pattern_name(self, n):
        return pattern_of(self.filename(n))

    def repaired(self, n):
        return bool(self.group(n).attrs["repaired"])

    def record(self, n):
        """The parsed record, `parse_qm9_text`'s keys."""
        g = self.group(n)
        return dict(natoms=int(g.attrs["natoms"]),
                    species=[s.decode() if isinstance(s, bytes) else str(s) for s in g["species"][:]],
                    positions=g["positions"][:], charges=g["charges"][:], frequencies=g["frequencies"][:],
                    smiles_gdb17=str(g.attrs["smiles_gdb17"]), smiles_relaxed=str(g.attrs["smiles_relaxed"]),
                    inchi_gdb17=str(g.attrs["inchi_gdb17"]), inchi_relaxed=str(g.attrs["inchi_relaxed"]))

    def text(self, n):
        """The molecule as a QM9 file (`render_qm9_text`): not the source's bytes -- the
        property row is `gdb <n>` -- but every line a reader of this repository uses."""
        return render_qm9_text(n, self.record(n))

    def smiles(self, n):
        """(gdb17, relaxed), as `config.qm9_smiles_from_xyz` returns them."""
        a = self.group(n).attrs
        return str(a["smiles_gdb17"]), str(a["smiles_relaxed"])

    def frequencies(self, n):
        return self.group(n)["frequencies"][:]

    def atoms(self, n):
        """An ase.Atoms with the Mulliken charges as initial charges (absent for a repair)."""
        import numpy as np
        from ase import Atoms
        g = self.group(n)
        d = self.record(n)
        a = Atoms(symbols=d["species"], positions=d["positions"])
        chg = d["charges"]
        if not np.isnan(chg).any():
            a.set_initial_charges(chg)
        a.info.update(qm9_index=int(g.attrs["qm9_index"]), repaired=bool(g.attrs["repaired"]),
                      smiles_gdb17=str(g.attrs["smiles_gdb17"]), smiles_relaxed=str(g.attrs["smiles_relaxed"]))
        return a

    def extract(self, n, cache_dir=None):
        """Write the molecule as a QM9 file into the cache (once, atomically) and return its path."""
        d = Path(cache_dir) if cache_dir else self.path.parent / CACHE_DIRNAME
        out = d / self.filename(n)
        if out.is_file():
            return out
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / (out.name + ".tmp{}".format(os.getpid()))
        tmp.write_text(self.text(n), encoding="utf-8")
        os.replace(tmp, out)
        return out


def open_archive(cfg=None):
    """The Archive at `archive(cfg)`, opened once per process; None without one."""
    p = archive(cfg)
    if p is None:
        return None
    key = str(p)
    if key not in _ARCHIVE_CACHE:
        try:
            _ARCHIVE_CACHE[key] = Archive(p)
        except Exception as e:                                              # noqa: BLE001
            raise RuntimeError(_unreadable(p, None, e)) from e
    return _ARCHIVE_CACHE[key]


def _index(cfg=None):
    """Map QM9 integer index -> (pattern name, path). Built once, from a listing.

    One directory listing of 133k entries beats 133k stat() calls, and it also gives
    the pattern census that the module docstring quotes.
    """
    r = root(cfg)
    if r is None:
        return {}
    key = str(r)
    if key in _INDEX_CACHE:
        return _INDEX_CACHE[key]

    rank = {name: i for i, (name, _p) in enumerate(PATTERNS)}
    rx = (("repaired_qm9_tag", re.compile(r"^dsgdb9nsd_(\d{6})_new\.xyz$")),
          ("repaired_short_tag", re.compile(r"^dsgdb9_(\d{6})_new\.xyz$")),
          ("original", re.compile(r"^dsgdb9nsd_(\d{6})\.xyz$")))

    out = {}
    for entry in os.scandir(r):
        if not entry.is_file():
            continue
        for name, pat in rx:
            m = pat.match(entry.name)
            if not m:
                continue
            n = int(m.group(1))
            prev = out.get(n)
            if prev is None or rank[name] < rank[prev[0]]:
                out[n] = (name, Path(entry.path))
            break
    _INDEX_CACHE[key] = out
    return out


def census(cfg=None):
    """How many files of each naming pattern. A measurement, reported in products."""
    r = root(cfg)
    if r is None:
        arc = open_archive(cfg)
        if arc is None:
            return dict(available=False, root=None)
        counts = {name: int(arc.attrs.get("n_" + name, 0)) for name, _ in PATTERNS}
        return dict(available=True, root=str(arc.path), form="hdf5", counts=counts, unmatched=0,
                    distinct_indices=len(arc), n_repaired=counts["repaired_qm9_tag"] + counts["repaired_short_tag"],
                    source="Senthil, Chakraborty & Ramakrishnan, Chem. Sci. 2021, 12, 5566")
    counts = {name: 0 for name, _ in PATTERNS}
    other = 0
    rx = (("repaired_qm9_tag", re.compile(r"^dsgdb9nsd_\d{6}_new\.xyz$")),
          ("repaired_short_tag", re.compile(r"^dsgdb9_\d{6}_new\.xyz$")),
          ("original", re.compile(r"^dsgdb9nsd_\d{6}\.xyz$")))
    for entry in os.scandir(r):
        if not entry.is_file():
            continue
        for name, pat in rx:
            if pat.match(entry.name):
                counts[name] += 1
                break
        else:
            other += 1
    idx = _index(cfg)
    return dict(available=True, root=str(r), form="directory", counts=counts, unmatched=other,
                distinct_indices=len(idx),
                n_repaired=counts["repaired_qm9_tag"] + counts["repaired_short_tag"],
                source="Senthil, Chakraborty & Ramakrishnan, Chem. Sci. 2021, 12, 5566")


def _to_int(qm9_index):
    if isinstance(qm9_index, int):
        return qm9_index
    s = str(qm9_index).strip()
    m = re.search(r"(\d+)\s*$", s)
    if not m:
        raise ValueError("cannot read a QM9 index out of {!r}".format(qm9_index))
    return int(m.group(1))


def _unreadable(path, n, e):
    """The message for an archive that answers with an HDF5 error. A damaged copy and a
    copy being overwritten while it is read both answer with these errors; a healthy one
    cannot (measured 2026-09-24: three Tianhe molecules died on `incorrect metadata
    checksum after all read attempts` while 6 455 read fine). The sentence carries the
    file, the group, and the command that decides which it is."""
    return ("the archive {} {}. ({}: {})\n"
            "  A damaged copy answers exactly like this -- a truncated one says "
            "'truncated file', a damaged one 'incorrect metadata checksum after all read "
            "attempts' -- and a healthy one does not. Check with:\n"
            "      python scripts/tooling/s0_verify_curated_qm9.py{}\n"
            "  and, if the same group fails there, re-copy the archive the packer wrote "
            "(write a temporary name, move it into place; never overwrite the file a "
            "campaign is reading).".format(
                path,
                "could not be opened" if n is None else
                "could not be read for {}".format(group_name(n)),
                type(e).__name__, e,
                "" if n is None else " --only {}".format(n)))


def find(qm9_index, cfg=None):
    """Return (path, pattern_name) for a QM9 index, or (None, None).

    `pattern_name` is `"original"` when the archive kept QM9's own geometry, and one of
    the two `repaired_*` names when ConnGO produced a new one. **A product records this
    string**: which geometry a number was computed on is part of the number, and this
    is the one place that distinction is visible.
    """
    n = _to_int(qm9_index)
    hit = _index(cfg).get(n)
    if hit is not None:
        return hit[1], hit[0]
    arc = open_archive(cfg)
    if arc is not None:
        try:
            inside = n in arc
        except Exception as e:                                              # noqa: BLE001
            raise RuntimeError(_unreadable(arc.path, n, e)) from e
        if inside:
            try:
                return arc.extract(n), arc.pattern_name(n)
            except Exception as e:                                          # noqa: BLE001
                raise RuntimeError(_unreadable(arc.path, n, e)) from e
    return None, None


def is_repaired(qm9_index, cfg=None):
    _p, kind = find(qm9_index, cfg)
    return kind is not None and kind.startswith("repaired")


def reconcile(cfg=None, uncharacterized=None):
    """Cross-check the archive against QM9's uncharacterized list. Three counts.

    plan_A section 2.3.2 item 3 asks for exactly these, and for the third to be looked
    at rather than summarised:

        rescuable   on the list AND present here      -> `curated` mode keeps them
        truly_lost  on the list AND absent here       -> removed even in `curated` mode
        unexpected  NOT on the list AND absent here   -> **look at these individually**

    The third is the interesting one. A molecule that upstream considered fine but the
    repair workflow did not emit is a disagreement between two sources, and neither
    "keep" nor "drop" is obviously right for it.
    """
    from . import qm9_uncharacterized
    listed = set(int(i) for i in (uncharacterized or qm9_uncharacterized.indices()))
    present = set(_index(cfg))
    rescuable = sorted(listed & present)
    truly_lost = sorted(listed - present)
    # "unexpected" needs the full QM9 index range; the archive's own maximum is the
    # best available bound without the full dataset on disk.
    hi = max(present) if present else 0
    unexpected = sorted(set(range(1, hi + 1)) - present - listed)
    return dict(
        n_listed=len(listed), n_present=len(present),
        n_rescuable=len(rescuable), n_truly_lost=len(truly_lost),
        n_unexpected=len(unexpected),
        truly_lost=truly_lost[:200],
        unexpected=unexpected[:200],
        note=("`unexpected` = not on the uncharacterized list and not in curatedQM9. "
              "Two sources disagree about these; neither keeping nor dropping them is "
              "obviously right, so they are listed rather than counted away."),
        paper_claim=dict(n_uncharacterized=3054, n_repaired=2988, n_unstable=66,
                         n_unstable_with_NNO=52))
