"""Read an OpenMM engine folder back: positions, symbols, masses, the per-frame table.

The one reader every analysis goes through (ticket 04 of the 2026-09-14 layout change,
ADR 0001): collect (`s0_B_qha_analyse.py`), the ensemble report and the 02d identity
check ask here and never open `frames.npy`. The trajectory IS `traj.dcd`; `start.pdb` is
its topology; `state.csv` is the per-frame energy and temperature at the same instants.

    read_trajectory(engine_dir, records_dir=None) -> dict
        positions_A       (n_frames, N, 3) float64, angstrom, straight from the DCD
        symbols           element symbols in the engine's atom order (start.pdb)
        masses_amu        the elements' masses (start.pdb); the record's masses, when a
                          record is given, are what the driver actually integrated with
        table             dict of arrays: step, time_ps, potential_kJ, kinetic_kJ,
                          temperature_K -- one entry per frame
        frame_spacing_ps  from the CSV's time column (the DCD header carries it too)
        n_frames
        meta              records_dir/meta.json as a dict, or None
        engine_dir

A folder without `traj.dcd` or `start.pdb` is refused with a FileNotFoundError that
names the folder and the file; a CSV shorter than the DCD (a job killed between the two
writes) is truncated to the frames both hold, and the count is what the DCD holds.

DCD stores positions as float32: about 1e-7 A at these coordinates, against
quasi-harmonic fluctuations of 0.1 A. mdtraj is the reader, a different code from the
`app.DCDFile` writer, which is the point of reading the engine's file back.
"""
import csv
import json
from pathlib import Path

import numpy as np

NM_TO_A = 10.0

CSV_COLUMNS = ("step", "time_ps", "potential_kJ", "kinetic_kJ", "temperature_K")


def _need(folder, name, setting="default"):
    from ..store import layout
    p = Path(folder) / layout.openmm_file_name(name, setting)
    if not p.is_file():
        raise FileNotFoundError(
            "no {} in {}: not an OpenMM engine folder, or the trajectory never started"
            .format(p.name, folder))
    return p


def read_table(path):
    """`state.csv` as a dict of arrays. The header is StateDataReporter's (`#"Step",...`)."""
    cols = {k: [] for k in CSV_COLUMNS}
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.reader(fh):
            if not row or row[0].startswith("#"):
                continue
            if len(row) < 5:
                continue                      # a line cut by a kill: the frame after it is absent too
            cols["step"].append(int(float(row[0])))
            cols["time_ps"].append(float(row[1]))
            cols["potential_kJ"].append(float(row[2]))
            cols["kinetic_kJ"].append(float(row[3]))
            cols["temperature_K"].append(float(row[4]))
    return {k: np.asarray(v) for k, v in cols.items()}


EV_TO_KJ = 96.48533212


def route_of(engine_dir):
    """`openmm` or `ase`, from the folder's place in the tree
    (`<molecule>/md_openmm/basinNN` or `<molecule>/md_ase/basinNN`)."""
    from ..store import layout
    r = layout.route_of_folder(Path(engine_dir).parent.name)
    if r is None:
        raise ValueError("{} is not under an md_openmm/ or md_ase/ folder of a molecule directory"
                         .format(engine_dir))
    return r


def read_md_log(path):
    """ASE's MDLogger table as the same dict of arrays the CSV gives.

    MDLogger writes `Time[ps] Etot[eV] Epot[eV] Ekin[eV] T[K]` (no step count: `step`
    is the row index); energies are converted to kJ/mol so the keys mean the same thing
    for both routes.
    """
    cols = {k: [] for k in CSV_COLUMNS}
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(l for l in fh if l.strip() and not l.lstrip().startswith("Time")):
            f = line.split()
            if len(f) < 5:
                continue
            cols["step"].append(len(cols["step"]))
            cols["time_ps"].append(float(f[0]))
            cols["potential_kJ"].append(float(f[2]) * EV_TO_KJ)
            cols["kinetic_kJ"].append(float(f[3]) * EV_TO_KJ)
            cols["temperature_K"].append(float(f[4]))
    return {k: np.asarray(v) for k, v in cols.items()}


def read_trajectory(engine_dir, records_dir=None, setting="default"):
    """`setting` picks the file names inside the folder (layout.engine_file_name); the
    route (openmm: traj.dcd + start.pdb + state.csv; ase: md.traj + md.log) comes from the
    folder's parent name."""
    from ..store import layout
    engine_dir = Path(engine_dir)
    route = route_of(engine_dir)
    if route == "ase":
        from ase.io import read as ase_read
        traj = _need(engine_dir, "md.traj", setting)
        frames = ase_read(str(traj), index=":")
        if not frames:
            raise FileNotFoundError("{} holds no frame yet".format(traj))
        positions = np.asarray([a.get_positions() for a in frames], dtype=np.float64)
        symbols = list(frames[0].get_chemical_symbols())
        masses = np.asarray(frames[0].get_masses(), dtype=float)
        log_p = engine_dir / layout.engine_file_name("md.log", setting)
        table = read_md_log(log_p) if log_p.is_file() else {k: np.zeros(0) for k in CSV_COLUMNS}
    else:
        import mdtraj as md
        dcd = _need(engine_dir, "traj.dcd", setting)
        pdb = _need(engine_dir, "start.pdb", setting)
        t = md.load(str(dcd), top=str(pdb))
        positions = np.asarray(t.xyz, dtype=np.float64) * NM_TO_A
        symbols = [a.element.symbol for a in t.topology.atoms]
        masses = np.asarray([a.element.mass for a in t.topology.atoms], dtype=float)
        csv_p = engine_dir / layout.openmm_file_name("state.csv", setting)
        table = read_table(csv_p) if csv_p.is_file() else {k: np.zeros(0) for k in CSV_COLUMNS}
    n = int(positions.shape[0])
    if len(table["step"]) > n:
        table = {k: v[:n] for k, v in table.items()}
    spacing = None
    if len(table["time_ps"]) >= 2:
        spacing = float(np.median(np.diff(table["time_ps"])))

    meta = None
    if records_dir is not None:
        mp = Path(records_dir) / "meta.json"
        if mp.is_file():
            meta = json.loads(mp.read_text(encoding="utf-8"))

    return dict(positions_A=positions, symbols=symbols, masses_amu=masses, table=table,
                frame_spacing_ps=spacing, n_frames=n, meta=meta, engine_dir=str(engine_dir),
                setting=str(setting), route=route)


#: The trajectory file each route is recognised by.
ROUTE_TRAJECTORY = {"openmm": "traj.dcd", "ase": "md.traj"}


def trajectory_dirs(molecule, setting="default", route="auto"):
    """[(basin, engine_dir, records_dir)] for every `<route>/basinNN/` that holds this
    setting's trajectory file, in basin order. `route="auto"` takes openmm when the
    molecule directory has any openmm trajectory of this setting, else ase. The records
    folder is named whether or not it exists."""
    from ..store import layout
    routes = [route] if route in ROUTE_TRAJECTORY else ["openmm", "ase"]
    for r in routes:
        root = Path(molecule) / layout.md_folder(r)
        fname = layout.engine_file_name(ROUTE_TRAJECTORY[r], setting)
        out = []
        if root.is_dir():
            for d in sorted(root.iterdir()):
                if d.name.startswith("basin") and d.name[5:].isdigit() and (d / fname).is_file():
                    out.append((int(d.name[5:]), d, layout.records_for(molecule, r, setting) / d.name))
        if out or route in ROUTE_TRAJECTORY:
            return out
    return []


def route_found(molecule, setting="default", route="auto"):
    """The route `trajectory_dirs` would read, or None."""
    dirs = trajectory_dirs(molecule, setting, route)
    return route_of(dirs[0][1]) if dirs else None
