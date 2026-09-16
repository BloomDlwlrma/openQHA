"""The molecule tree: where every file of one molecule goes (ADR 0001, 2026-09-14).

One molecule directory per (root, tag, qid), one folder per engine inside it, engine
files only in those folders, and this repository's records in `_records/`:

    <root>/<tag>/<range>/<chunk>/<qid>/
      crest/                      CREST's working directory, verbatim
      crest_shake1/               the SHAKE fallback, only when it ran
      mace/confNN/                opt.traj  opt.log  conf.extxyz    every tightened conformer
      mace/basinNN/               basin.extxyz  hessian.npy         every surviving basin
      md_openmm/basinNN/          start.pdb system.xml integrator.xml traj.dcd
                                  state.csv state.xml state.chk        (the default setting)
                                  start_<setting>.pdb ... traj_<setting>.dcd ...
                                  (every other setting, same folder, its name in the file)
      md_ase/basinNN/             start.extxyz  md.traj  md.log  (the ASE route, the CPU
                                  cross-check; same setting-in-the-name rule)
      xtb/basinNN/  orca/basinNN/ 02c only
      _records/                   everything this repository writes about the run

The MD folders are named by ROLE (user ruling 2026-09-15): `mace/` is the potential's
relax + Hessian, `md_openmm/` and `md_ase/` are the two implementations of the same
sampling step. The route identifiers a driver takes (`--route openmm|ase`) are unchanged;
`md_folder(route)` maps them to the folder and nothing else spells the folder name.

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

#: The engine files of one ASE-route trajectory (ASE's own forms): the post-relax start,
#: the Trajectory (positions, momenta, energy, forces per frame), the MDLogger table.
ASE_FILES = ("start.extxyz", "md.traj", "md.log")

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


#: The setting whose files keep the bare names.
DEFAULT_SETTING = "default"

#: Route identifier -> the MD folder inside a molecule directory (named by role).
MD_FOLDER = {"openmm": "md_openmm", "ase": "md_ase"}


def md_folder(route):
    """`md_openmm` for route `openmm`, `md_ase` for `ase`."""
    try:
        return MD_FOLDER[str(route)]
    except KeyError:
        raise ValueError("unknown MD route {!r}; known: {}".format(route, sorted(MD_FOLDER)))


def route_of_folder(name):
    """The inverse of `md_folder`, or None for a folder that is not an MD folder."""
    for r, f in MD_FOLDER.items():
        if f == str(name):
            return r
    return None


def openmm_dir(molecule, setting, basin):
    """The basin's one OpenMM folder, `md_openmm/basinNN/`, whatever the setting.

    No seed level (ruling S0-B-59) and, since the user's ruling of 2026-09-14, no setting
    level either: every setting's trajectory of a basin sits in the same folder and the
    setting is in the file name (`openmm_file_name`). `setting` is accepted so a caller
    states which run it means, and ignored here.
    """
    return Path(molecule) / md_folder("openmm") / "basin{:02d}".format(int(basin))


def ase_dir(molecule, setting, basin):
    """The basin's one ASE-route folder, `md_ase/basinNN/`; the setting is in the file name."""
    return Path(molecule) / md_folder("ase") / "basin{:02d}".format(int(basin))


def engine_file_name(name, setting=DEFAULT_SETTING):
    """`traj.dcd` / `md.traj` for the default setting; `traj_<setting>.dcd` for any other.
    The one naming rule for every trajectory engine (ruling 2026-09-14)."""
    if str(setting) == DEFAULT_SETTING:
        return name
    stem, dot, ext = name.partition(".")
    return "{}_{}{}{}".format(stem, setting, dot, ext)


def md_records_dir(molecule, route):
    """`_records/<md folder>/` (`_records/md_openmm/`): where the trajectory, collect, ensemble
    and 02d Records of one route live. No setting level (user ruling 2026-09-15, records
    redesign Q7 of round 1): the setting is in the file stem, `record_file_name`, exactly
    as for engine files. Named like the engine folder it describes."""
    return Path(molecule) / RECORDS / md_folder(route)


def basin_records_dir(molecule, route, basin):
    """`_records/<md folder>/basinNN/`: one trajectory's Record (`md.out`, `md.toml`,
    `driver.log`, and the `md_<setting>.*` of every other setting)."""
    return md_records_dir(molecule, route) / "basin{:02d}".format(int(basin))


def record_file_name(name, setting=DEFAULT_SETTING):
    """`md.toml` for the default setting, `md_s2.toml` for setting `s2`;
    `collect.dat` -> `collect_s2.dat`. The engine-file rule,
    applied to Records: a Report and its Property file share a stem."""
    return engine_file_name(name, setting)


def openmm_file_name(name, setting=DEFAULT_SETTING):
    """`traj.dcd` for the default setting; `traj_<setting>.dcd` for any other.

    The suffix is the SETTING, not the job id: a resumed trajectory is continued by a
    later job and must find its own files by name; the job id belongs in the record.
    """
    return engine_file_name(name, setting)


def openmm_file(molecule, setting, basin, name):
    """The on-disk path of one engine file of one trajectory."""
    return openmm_dir(molecule, setting, basin) / openmm_file_name(name, setting)


def tag_records_dir(root, tag):
    """`<root>/<tag>/_records/`: the one per-tag folder, holding parsl's own run
    directories (`parsl/<job>.<pid>/`). Since 2026-09-15 no Batch record lives here: the
    Slurm log is the Batch's report (CONTEXT.md, Batch)."""
    return Path(root) / str(tag) / RECORDS


def xtb_dir(molecule, basin):
    return Path(molecule) / "xtb" / "basin{:02d}".format(int(basin))


def orca_dir(molecule, basin):
    return Path(molecule) / "orca" / "basin{:02d}".format(int(basin))


def records_dir(molecule):
    return Path(molecule) / RECORDS


#: The level folder (CONTEXT.md; ADR 0004): every level's results for one molecule, one
#: sub-folder per level named by the level. Engine files never go here.
LEVELS = "levels"

_LEVEL_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def crest_entropy_dir(molecule, run):
    """`crest_entropy/runNN/`: one CREST `--entropy` run (ticket 25); a separate engine
    folder because CREST's entropy mode writes its own tree beside the branch A run."""
    return Path(molecule) / "crest_entropy" / "run{:02d}".format(int(run))


def xtb_entropy_dir(molecule, run, conformer):
    """`xtb/entropy_runNN/confKK/`: the xtb Hessian of one CREST entropy-run conformer."""
    return Path(molecule) / "xtb" / "entropy_run{:02d}".format(int(run)) / "conf{:02d}".format(int(conformer))


def orca_level_dir(molecule, level, basin):
    """`orca/<level>/basinNN/`: ORCA's engine files for one basin at one level (ticket 26)."""
    return Path(molecule) / "orca" / str(level) / "basin{:02d}".format(int(basin))


def level_dir(molecule, level):
    """`<molecule>/levels/<level>/`. The level name must already be in the CONTEXT.md
    spelling (lower-case, `wb97m-d3bj_def2-tzvppd`, `gfn2`, `mace-off23_medium`); an
    upper-case or slash-bearing name is refused here rather than spelled two ways."""
    if not _LEVEL_NAME.match(str(level)):
        raise ValueError("level name {!r} is not in the CONTEXT.md spelling (lower-case, "
                         "method first, basis second, joined by '_')".format(level))
    return Path(molecule) / LEVELS / str(level)

