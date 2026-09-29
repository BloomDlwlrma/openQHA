"""The molecule tree: where every file of one molecule goes.

One molecule directory per (root, tag, qid), one folder per engine inside it, engine
files only in those folders, and this repository's records in `_records/`:

    <root>/<tag>/<qid>/
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
      frames/                     the Hessian-learning set:
        <gen>.<level>.extxyz  frames.{out,toml}  labels.<level>.{out,toml}
        orca.<level>.<gen>_bBB_kK.{inp,out,hess,engrad}   one frame's ORCA job, a file group
      msrrho/                     the msRRHO study:
        orca.<level>.basinNN.{inp,out,hess,xyz}           one basin's Opt+Freq job at a level
        crest_entropy/runNN/  xtb/entropy_runNN/confKK/   the study's engines, verbatim
        thermo/<level>.<step>.{out,toml,dat}  thermo/level_compare.*  hessian_compare.*
      _records/                   everything this repository writes about the run

The MD folders are named by ROLE (since 2026-09-15): `mace/` is the potential's
relax + Hessian, `md_openmm/` and `md_ase/` are the two implementations of the same
sampling step. The route identifiers a driver takes (`--route openmm|ase`) are unchanged;
`md_folder(route)` maps them to the folder and nothing else spells the folder name.

The tag directory is FLAT (since 2026-09-20): the molecule directory is
`<root>/<tag>/<qid>/`, the two shard layers of 2026-09-14 (`1_16000/1001_2000/`) are
gone -- a campaign's tag holds thousands of molecules, not 133 885, and the path was
too long. A molecule without a QM9 index (a SMILES run) goes under its label the
same way. The Dataset files of a tag live beside the molecules in `_datasets/<name>/`,
tag-wide records in `_records/`.

**This module composes paths and nothing else.** It reads no configuration and touches
no filesystem, so a test can hold it to the documented tree with literals. Every writer
and every reader asks here; a path built anywhere else is a defect.
"""
import re
from pathlib import Path

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
            "cannot read a QM9 index out of {!r}: the number a qid ends in, and this one "
            "has none".format(qid))
    return int(m.group(1))


def molecule_dir(root, tag, qid):
    """`<root>/<tag>/<qid>`. `root` may be any path-like. A molecule identified only by a
    label (a SMILES run, `s0_A_pipeline.py --smiles`) goes under that label the same way;
    the tag directory is flat."""
    return Path(root) / str(tag) / str(qid)


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

    No seed level, and since 2026-09-14 no setting level either: every setting's
    trajectory of a basin sits in the same folder and the setting is in the file name
    (`openmm_file_name`). `setting` is accepted so a caller states which run it means,
    and ignored here.
    """
    return Path(molecule) / md_folder("openmm") / "basin{:02d}".format(int(basin))


def ase_dir(molecule, setting, basin):
    """The basin's one ASE-route folder, `md_ase/basinNN/`; the setting is in the file name."""
    return Path(molecule) / md_folder("ase") / "basin{:02d}".format(int(basin))


def engine_file_name(name, setting=DEFAULT_SETTING):
    """`traj.dcd` / `md.traj` for the default setting; `traj_<setting>.dcd` for any other.
    The one naming rule for every trajectory engine."""
    if str(setting) == DEFAULT_SETTING:
        return name
    stem, dot, ext = name.partition(".")
    return "{}_{}{}{}".format(stem, setting, dot, ext)


def md_records_dir(molecule, route):
    """`_records/<md folder>/` (`_records/md_openmm/`): where the trajectory, collect, ensemble
    and 02d Records of one route live. No setting level (records redesign, 2026-09-15):
    the setting is in the file stem, `record_file_name`, exactly as for engine files.
    Named like the engine folder it describes."""
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
    Slurm log is the Batch's report."""
    return Path(root) / str(tag) / RECORDS


def xtb_dir(molecule, basin):
    return Path(molecule) / "xtb" / "basin{:02d}".format(int(basin))


def records_dir(molecule):
    return Path(molecule) / RECORDS


# ====================================================================== the two studies
#: Two sub-folders of the molecule directory separate the two studies that share its
#: branch A (since 2026-09-20): `msrrho/` for the msRRHO study
#: (the basin-level ORCA jobs, CREST's entropy runs, the xtb Hessians of those, and the
#: study's Records under `msrrho/thermo/`), `frames/` for the Hessian-learning set (the
#: Frame sets, their labels and the per-frame ORCA jobs). ORCA jobs are FILE GROUPS,
#: `orca.<level>.<job>.{inp,out,hess,engrad,xyz}`, never a directory per job.
MSRRHO = "msrrho"
THERMO = "thermo"
FRAMES = "frames"

_LEVEL_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def _check_level(level):
    if not _LEVEL_NAME.match(str(level)):
        raise ValueError("level name {!r} is not in the standard spelling (lower-case, "
                         "method first, basis second, joined by '_')".format(level))
    return str(level)


def msrrho_dir(molecule):
    """`<molecule>/msrrho/`."""
    return Path(molecule) / MSRRHO


def crest_entropy_dir(molecule, run):
    """`msrrho/crest_entropy/runNN/`: one CREST `--entropy` run, CREST's own tree
    verbatim, as `crest/` is for branch A."""
    return msrrho_dir(molecule) / "crest_entropy" / "run{:02d}".format(int(run))


def xtb_entropy_dir(molecule, run, conformer):
    """`msrrho/xtb/entropy_runNN/confKK/`: the xtb Hessian of one CREST entropy-run conformer."""
    return msrrho_dir(molecule) / "xtb" / "entropy_run{:02d}".format(int(run)) / "conf{:02d}".format(int(conformer))


def orca_level_stem(level, basin):
    """`orca.<level>.basinNN`: the file group of one basin's job at one level (optimisation
    to the level's own minimum + Hessian), files of `msrrho/`."""
    return "orca.{}.basin{:02d}".format(_check_level(level), int(basin))


def orca_level_file(molecule, level, basin, ext):
    """`msrrho/orca.<level>.basinNN<ext>` (`ext` with its dot)."""
    return msrrho_dir(molecule) / (orca_level_stem(level, basin) + str(ext))


def thermo_dir(molecule):
    """`msrrho/thermo/`: the msRRHO study's Records, flat -- `<level>.<step>.<ext>` per level
    (`wb97m-d3bj_def2-tzvppd.thermo_msrrho.toml`, `mace-off23_medium.merge_map.dat`) and the
    cross-level ones bare (`level_compare.toml`, `hessian_compare.toml`). Replaces the
    earlier per-level folders."""
    return msrrho_dir(molecule) / THERMO


def level_file(molecule, level, name):
    """`msrrho/thermo/<level>.<name>`: a Record of one level (`name` = `thermo_msrrho.toml`,
    `merge_map.dat`, `degeneracy.out`, ...)."""
    return thermo_dir(molecule) / "{}.{}".format(_check_level(level), name)


def thermo_file(molecule, name):
    """`msrrho/thermo/<name>`: a Record across levels (`level_compare.toml`, `hessian_compare.out`)."""
    return thermo_dir(molecule) / str(name)


def levels_present(molecule, name="thermo_msrrho.toml"):
    """The levels whose `<level>.<name>` Record exists, from the file names."""
    d = thermo_dir(molecule)
    if not d.is_dir():
        return []
    suffix = "." + name
    return sorted(p.name[:-len(suffix)] for p in d.iterdir() if p.name.endswith(suffix) and p.is_file())


#: The Frame set folder: one molecule's frames, one extxyz per
#: generator and per level, engine-independent; its Records and the per-frame ORCA file
#: groups beside them.


def frames_dir(molecule):
    """`<molecule>/frames/`."""
    return Path(molecule) / FRAMES


def frames_file(molecule, generator, level):
    """`<molecule>/frames/<generator>.<level>.extxyz` (level in the standard
    lower-case spelling)."""
    return frames_dir(molecule) / "{}.{}.extxyz".format(generator, _check_level(level))


def orca_frame_stem(level, generator, basin, k):
    """`orca.<level>.<generator>_bBB_kK`: the file group of one frame's ORCA label -- FILES
    of `frames/`, `<stem>.{inp,out,hess,engrad}` (a frame's
    job is not a directory two levels down). The fields are joined by `.`, the repository's
    `<thing>.<level>.<ext>` convention (`frames/<generator>.<level>.extxyz`), because the
    level and the frame tag both contain `_`; `name.split(".")` gives engine, level (the
    middle parts), frame tag, extension."""
    return "orca.{}.{}_b{:02d}_k{}".format(_check_level(level), generator, int(basin), int(k))


def orca_frame_file(molecule, level, generator, basin, k, ext):
    """`frames/orca.<level>.<generator>_bBB_kK<ext>` (`ext` with its dot: ".out")."""
    return frames_dir(molecule) / (orca_frame_stem(level, generator, basin, k) + str(ext))
