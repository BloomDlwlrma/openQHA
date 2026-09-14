"""The molecule tree: where every file of one molecule goes (ADR 0001, 2026-09-14).

One molecule directory per (root, tag, qid), one folder per engine inside it, engine
files only in those folders, and this repository's records in `_records/`:

    <root>/<tag>/<range>/<chunk>/<qid>/
      crest/                      CREST's working directory, verbatim
      crest_shake1/               the SHAKE fallback, only when it ran
      mace/confNN/                opt.traj  opt.log  conf.extxyz    every tightened conformer
      mace/basinNN/               basin.extxyz  hessian.npy         every surviving basin
      openmm/<setting>/basinNN/   start.pdb system.xml integrator.xml traj.dcd
                                  state.csv state.xml state.chk
      xtb/basinNN/  orca/basinNN/ 02c only
      _records/                   everything this repository writes about the run

The shard is arithmetic on the QM9 index: range 16 000, chunk 1 000 (user ruling
2026-09-14; the retired basin store used 4 000), so a directory holds at most 1 000
molecules and a reader never needs an index file to find one.

**This module composes paths and nothing else.** It reads no configuration and touches
no filesystem, so a test can hold it to the documented tree with literals. Every writer
and every reader asks here; a path built anywhere else is a defect.
"""
import re
from pathlib import Path

#: Molecules per leaf directory. modifiable_convention.
CHUNK = 1000

#: Molecules per first-level directory. modifiable_convention.
RANGE = 16000

#: The engine files of one OpenMM trajectory, in the order they are created.
OPENMM_FILES = ("start.pdb", "system.xml", "integrator.xml", "traj.dcd",
                "state.csv", "state.xml", "state.chk")

#: The engine files of one tightened conformer (ASE's own forms).
MACE_CONFORMER_FILES = ("opt.traj", "opt.log", "conf.extxyz")

#: The engine files of one surviving basin.
MACE_BASIN_FILES = ("basin.extxyz", "hessian.npy")

#: The records folder inside a molecule directory.
RECORDS = "_records"


def qid_number(qid):
    """The integer a QM9 index ends in: 'dsgdb9nsd_000018' -> 18."""
    m = re.search(r"(\d+)\s*$", str(qid).strip())
    if not m:
        raise ValueError(
            "cannot read a QM9 index out of {!r}; the molecule tree is sharded on the "
            "number a qid ends in, and this one has none".format(qid))
    return int(m.group(1))


def shard(qid, chunk=CHUNK, rng=RANGE):
    """(range_dir, chunk_dir) for a QM9 index. Pure arithmetic."""
    n = qid_number(qid)
    r0 = ((n - 1) // rng) * rng + 1
    c0 = ((n - 1) // chunk) * chunk + 1
    return ("{}_{}".format(r0, r0 + rng - 1), "{}_{}".format(c0, c0 + chunk - 1))


def molecule_dir(root, tag, qid):
    """`<root>/<tag>/<range>/<chunk>/<qid>`. `root` may be any path-like.

    A molecule identified only by a label (a SMILES run, `s0_A_pipeline.py --smiles`)
    has no index to shard on and goes to `<root>/<tag>/_label/<label>`: an ad-hoc
    molecule is a handful of runs, and a made-up index would put a fiction into the path.
    """
    try:
        r, c = shard(qid)
    except ValueError:
        return Path(root) / str(tag) / "_label" / str(qid)
    return Path(root) / str(tag) / r / c / str(qid)


def crest_dir(molecule, fallback_shake=None):
    """`crest/`, or `crest_shake<N>/` for the SHAKE fallback attempt."""
    name = "crest" if fallback_shake is None else "crest_shake{}".format(int(fallback_shake))
    return Path(molecule) / name


def mace_conformer_dir(molecule, index):
    return Path(molecule) / "mace" / "conf{:02d}".format(int(index))


def mace_basin_dir(molecule, basin):
    return Path(molecule) / "mace" / "basin{:02d}".format(int(basin))


def openmm_dir(molecule, setting, basin):
    """One trajectory: `openmm/<setting>/basinNN/`. No seed level (ruling S0-B-59)."""
    return Path(molecule) / "openmm" / str(setting) / "basin{:02d}".format(int(basin))


def xtb_dir(molecule, basin):
    return Path(molecule) / "xtb" / "basin{:02d}".format(int(basin))


def orca_dir(molecule, basin):
    return Path(molecule) / "orca" / "basin{:02d}".format(int(basin))


def records_dir(molecule):
    return Path(molecule) / RECORDS
