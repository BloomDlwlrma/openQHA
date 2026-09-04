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
"""
import os
import re
from pathlib import Path

from . import S0_ROOT

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
    return root(cfg) is not None


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
        return dict(available=False, root=None)
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
    return dict(available=True, root=str(r), counts=counts, unmatched=other,
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


def find(qm9_index, cfg=None):
    """Return (path, pattern_name) for a QM9 index, or (None, None).

    `pattern_name` is `"original"` when the archive kept QM9's own geometry, and one of
    the two `repaired_*` names when ConnGO produced a new one. **A product records this
    string**: which geometry a number was computed on is part of the number, and this
    is the one place that distinction is visible.
    """
    hit = _index(cfg).get(_to_int(qm9_index))
    if hit is None:
        return None, None
    return hit[1], hit[0]


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
