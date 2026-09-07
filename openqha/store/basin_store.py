"""Where basin lists are written, and how to find one again.

--------------------------------------------------------------------------------------
The problem
--------------------------------------------------------------------------------------
A full QM9 campaign is 133 885 molecules. One directory per molecule means 133 885
directories in one parent, which is a bad idea on every filesystem this project touches:

  * `ls` on the parent becomes unusable, and so does tab completion;
  * ext4 without dir_index degrades badly past ~10k entries per directory;
  * a Lustre MDT (Tianhe) serialises metadata on the parent directory's stripe, so a
    fan-out of 16 workers all creating siblings contends on one server;
  * rsync and tar spend their time in `readdir` rather than moving bytes.

--------------------------------------------------------------------------------------
The layout
--------------------------------------------------------------------------------------
Two levels of shard, taken from the campaign structure that already exists (the QM9
range is processed in chunks of 4000, `package1.chunk_size`):

    data/basins/<range>/<chunk>/<qm9_index>.basins.json
                               /<qm9_index>.basins.xyz

    data/basins/1_16000/1_4000/dsgdb9nsd_000018.basins.json
    data/basins/1_16000/1_4000/dsgdb9nsd_000018.basins.xyz
    data/basins/1_16000/4001_8000/dsgdb9nsd_005271.basins.json
    ...

**Files, not directories, at the leaf.** Two files per molecule in a directory of at
most 4000 molecules gives at most 8000 entries per directory -- comfortable everywhere,
and `ls data/basins/1_16000/1_4000/` is still something a person can read.

The chunk that owns a molecule is arithmetic on its index, so `path_for()` never needs
to search and a reader never needs an index file to find one result.

--------------------------------------------------------------------------------------
Why not one big file, or a database
--------------------------------------------------------------------------------------
Because a run is interrupted. Per-molecule files mean an interrupted campaign has
exactly the molecules it finished, each complete, and a resume is a set difference. One
append-only file would need locking across the 16 workers of a node, and a partial
write at the end of a killed job would corrupt the tail -- which is the case that
actually happens when a queue walltime expires (D0-C-25).

The parquet summaries under `analysis/` are built FROM these files, not instead of them.
"""
import json
import re
from pathlib import Path

from .. import S0_ROOT

#: Molecules per leaf directory. Matches `package1.chunk_size`, so a chunk of the
#: campaign is exactly one leaf directory and "which chunk am I resuming" has one
#: answer. modifiable_convention.
CHUNK = 4000

#: Molecules per first-level directory.
RANGE = 16000


def _to_int(qm9_index):
    m = re.search(r"(\d+)\s*$", str(qm9_index).strip())
    if not m:
        raise ValueError("cannot read a QM9 index out of {!r}".format(qm9_index))
    return int(m.group(1))


def shard(qm9_index, chunk=CHUNK, rng=RANGE):
    """(range_dir, chunk_dir) for a QM9 index. Pure arithmetic, no filesystem access."""
    n = _to_int(qm9_index)
    r0 = ((n - 1) // rng) * rng + 1
    c0 = ((n - 1) // chunk) * chunk + 1
    return ("{}_{}".format(r0, r0 + rng - 1), "{}_{}".format(c0, c0 + chunk - 1))


def root(cfg=None):
    """Where basin lists live. `S0_BASIN_ROOT` overrides it."""
    import os
    env = os.environ.get("S0_BASIN_ROOT")
    if env:
        return Path(env).expanduser()
    return S0_ROOT / "data" / "basins"


def dir_for(qm9_index, cfg=None, tag=None, chunk=CHUNK, rng=RANGE):
    """The directory that holds one molecule's basin list.

    `tag` separates campaigns (a re-run under different settings does not overwrite
    the first). Without a tag the layout is exactly the one in the module docstring.
    """
    r, c = shard(qm9_index, chunk, rng)
    base = root(cfg)
    return (base / tag / r / c) if tag else (base / r / c)


def paths_for(qm9_index, cfg=None, tag=None, **kw):
    """(json_path, xyz_path) for one molecule. Neither is required to exist."""
    d = dir_for(qm9_index, cfg, tag, **kw)
    stem = str(qm9_index)
    return d / (stem + ".basins.json"), d / (stem + ".basins.xyz")


def write(qm9_index, record, xyz_text, cfg=None, tag=None, **kw):
    """Write one molecule's basin list. Returns (json_path, xyz_path).

    Written to a temporary name and renamed, because a job killed at its walltime
    otherwise leaves a truncated JSON that later reads as a corrupt result rather than
    as a missing one. A missing result is resumable; a corrupt one has to be found first.
    """
    j, x = paths_for(qm9_index, cfg, tag, **kw)
    j.parent.mkdir(parents=True, exist_ok=True)
    tmp_j = j.with_suffix(j.suffix + ".part")
    tmp_x = x.with_suffix(x.suffix + ".part")
    tmp_j.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_x.write_text(xyz_text, encoding="utf-8")
    tmp_j.replace(j)
    tmp_x.replace(x)
    return j, x


def read(qm9_index, cfg=None, tag=None, **kw):
    """The basin record for one molecule, or None if it has not been computed."""
    j, _x = paths_for(qm9_index, cfg, tag, **kw)
    if not j.exists():
        return None
    return json.loads(j.read_text(encoding="utf-8"))


def exists(qm9_index, cfg=None, tag=None, **kw):
    return paths_for(qm9_index, cfg, tag, **kw)[0].exists()


def completed(cfg=None, tag=None, chunk_dir=None):
    """QM9 indices already computed. The set a resume subtracts from its worklist.

    `chunk_dir` restricts the scan to one leaf ("1_4000"), which is what a per-chunk
    job wants -- scanning the whole tree to resume one chunk is the kind of thing that
    makes a 133k campaign slow at the wrong end.
    """
    base = (root(cfg) / tag) if tag else root(cfg)
    if not base.is_dir():
        return set()
    pattern = "*/{}/*.basins.json".format(chunk_dir) if chunk_dir else "*/*/*.basins.json"
    out = set()
    for p in base.glob(pattern):
        try:
            out.add(_to_int(p.name.split(".")[0]))
        except ValueError:
            continue
    return out


def census(cfg=None, tag=None):
    """How many results exist, per chunk. Cheap enough to run on a login node."""
    base = (root(cfg) / tag) if tag else root(cfg)
    if not base.is_dir():
        return dict(root=str(base), exists=False, total=0, chunks={})
    chunks = {}
    for p in sorted(base.glob("*/*")):
        if p.is_dir():
            chunks["{}/{}".format(p.parent.name, p.name)] = len(
                list(p.glob("*.basins.json")))
    return dict(root=str(base), exists=True, total=sum(chunks.values()), chunks=chunks)
