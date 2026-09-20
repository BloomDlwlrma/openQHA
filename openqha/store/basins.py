"""Branch A's product, read from the molecule directory (ADR 0001, 2026-09-14).

    <molecule>/mace/basinNN/basin.extxyz     the basins, one file each, branch A's atom order
    <molecule>/mace/basinNN/hessian.npy      the raw analytic Hessian at that basin
    <molecule>/_records/branchA.toml         branch A's Property file (records redesign, 2026-09-15)
    <molecule>/_records/branchA.out          branch A's Report; its last line is the marker

Every reader of the basins -- branch B (`--basins auto`), the ensemble report, 02c, 02d,
the chain's "already done" check, the campaign worklist -- asks here. The existence of a
product is the existence of `mace/basin00/basin.extxyz`, an engine file; the record is
read when a reader needs what branch A had to say about it, never to locate a geometry.

History: until 2026-09-14 the same record and a multi-frame xyz were written into a
sharded store, `data/basins/<tag>/<range>/<chunk>/<qid>.basins.{json,xyz}`, and read
back from there by a module of the same shape as this one (`basin_store`). The store was
a byte-identical second copy of the record; the layout rule it introduced (range 16 000,
chunk -- then 4 000, now 1 000) lives on in `layout.py`.
"""
import json
import re
from pathlib import Path

from .. import config
from . import layout

_BASIN_DIR = re.compile(r"^basin(\d+)$")


def molecule_for(qid, tag, cfg=None, root=None):
    """The molecule directory of `qid` under `tag`: `root` (default: the runs root)."""
    return layout.molecule_dir(root if root is not None else config.runs_root(cfg), tag, qid)


def basin_files(molecule):
    """`mace/basinNN/basin.extxyz` for every basin folder that holds one, in basin order."""
    mace = Path(molecule) / "mace"
    if not mace.is_dir():
        return []
    out = []
    for d in mace.iterdir():
        m = _BASIN_DIR.match(d.name)
        if m and (d / "basin.extxyz").is_file():
            out.append((int(m.group(1)), d / "basin.extxyz"))
    return [p for _i, p in sorted(out)]


def exists(qid, tag, cfg=None, root=None):
    return bool(basin_files(molecule_for(qid, tag, cfg, root)))


def read_basins(qid, tag, cfg=None, root=None):
    """The basins as `ase.Atoms`, in basin order, header info in `atoms.info`."""
    from ase.io import read
    files = basin_files(molecule_for(qid, tag, cfg, root))
    return [read(str(p), format="extxyz") for p in files]


def read_record(qid, tag, cfg=None, root=None):
    """`_records/branchA.toml` as a dict of blocks (`Calculation_Status`, `Calculation_Info`,
    `CREST_Run`, `Census`, `Basin` (a list), `Criteria`), or None when branch A has not
    written one under this tag. `openqha.store.branch_a_property` spells the keys and
    answers the usual questions (`basin_rows`, `relative_kcal`)."""
    from . import property as prop
    p = record_path(qid, tag, cfg, root)
    if not p.is_file():
        return None
    return prop.load(p)


def record_path(qid, tag, cfg=None, root=None):
    from . import branch_a_property
    return layout.records_dir(molecule_for(qid, tag, cfg, root)) / branch_a_property.FILE


def report_path(qid, tag, cfg=None, root=None):
    from . import branch_a_property
    return layout.records_dir(molecule_for(qid, tag, cfg, root)) / branch_a_property.REPORT


def status(qid, tag, cfg=None, root=None):
    """STATUS of branch A's Property file: NORMAL TERMINATION when branch A finished,
    None when there is no Property file (or it has no status block)."""
    from . import property as prop
    return prop.status_of(record_path(qid, tag, cfg, root))


def done(qid, tag, cfg=None, root=None):
    """Did branch A finish for this molecule under this tag: STATUS is NORMAL TERMINATION
    and the basins exist as engine files."""
    from . import property as prop
    return status(qid, tag, cfg, root) == prop.NORMAL_TERMINATION and exists(qid, tag, cfg, root)


def tags_for(qid, cfg=None, root=None):
    """The tags under which this molecule HAS basins, sorted -- for the message a reader
    gets when it asks for the wrong tag (on 2026-09-11 that one `ls` was the whole fix)."""
    base = Path(root if root is not None else config.runs_root(cfg))
    if not base.is_dir():
        return []
    out = []
    for t in base.iterdir():
        if t.is_dir() and basin_files(layout.molecule_dir(base, t.name, qid)):
            out.append(t.name)
    return sorted(out)


def missing_message(qid, tag, cfg=None, root=None):
    """One sentence naming the folder that was looked for, and what exists instead."""
    mol = molecule_for(qid, tag, cfg, root)
    msg = ("no branch A product for {} under tag {!r}: {} holds no basinNN/basin.extxyz"
           .format(qid, tag, mol / "mace"))
    have = tags_for(qid, cfg, root)
    if have:
        msg += (".\n  This molecule HAS basins under: {}.\n  The tag is the one the "
                "branchA.conf that produced them set; pass --tag {}."
                .format(", ".join(have), have[0]))
    else:
        msg += ".\n  No tag under {} holds it. Run branch A first.".format(
            root if root is not None else config.runs_root(cfg))
    return msg


def completed(tag, cfg=None, root=None):
    """QM9 indices whose basins exist under `tag`: what a resume subtracts.

    Done means both: `mace/basin00/basin.extxyz` (the geometries are the product) AND
    `_records/branchA.toml` (the Property file, written last, only ever with STATUS
    NORMAL TERMINATION by branch A). A job killed between the census and the record
    leaves the first without the second and is redone. Two globs, no per-molecule stat
    (the Lustre lesson in `worklist.py`); the same rule as `done()`, one molecule at a time.
    """
    from . import branch_a_property
    base = Path(root if root is not None else config.runs_root(cfg)) / str(tag)
    if not base.is_dir():
        return set()
    leaf = "*"                       # the tag directory is flat (ADR 0001, amendment 3)

    def _numbers(pattern, up):
        out = set()
        for p in base.glob(pattern):
            try:
                out.add(layout.qid_number(p.parents[up].name))
            except ValueError:
                continue
        return out

    return (_numbers(leaf + "/mace/basin00/basin.extxyz", 2)
            & _numbers(leaf + "/" + layout.RECORDS + "/" + branch_a_property.FILE, 1))


def census(tag, cfg=None, root=None):
    """How many molecules have basins under `tag`. Login-node cheap (one glob)."""
    base = Path(root if root is not None else config.runs_root(cfg)) / str(tag)
    if not base.is_dir():
        return dict(root=str(base), exists=False, total=0)
    n = sum(1 for _ in base.glob("*/mace/basin00/basin.extxyz"))
    return dict(root=str(base), exists=True, total=n)
