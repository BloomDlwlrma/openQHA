"""OpenMM's own files for one trajectory: what an engine folder holds (ADR 0001).

    openmm/<setting>/basinNN/
      start.pdb        the structure the dynamics started from (post-relax), one residue MOL
      system.xml       XmlSerializer of the System, TorchForce included
      integrator.xml   XmlSerializer of the integrator
      traj.dcd         one frame per sampling interval, production only
      state.csv        one row per frame, StateDataReporter's own header and columns
      state.xml        XmlSerializer of the last flushed State (portable resume)
      state.chk        createCheckpoint() at the same moment (fast resume, same hardware)

OpenMM writes nothing by itself: a Reporter has to be attached, and this repository's
driver runs a bare `openmm.Context`, not an `app.Simulation`. So the same classes the
reporters use are driven here directly -- `app.DCDFile`, `app.PDBFile`,
`XmlSerializer` -- and the CSV is written with `StateDataReporter`'s header and column
names, so a reader that knows OpenMM's files knows these.

Why both `state.xml` and `state.chk` (user, 2026-09-14): a checkpoint is bound to the
platform and hardware it was written on and may not load on another card; the XML state
is portable and larger. Resume tries the checkpoint first and says which one loaded.

Every write that a killed job could leave half done goes to a `.part` sibling and is
renamed into place; the DCD and CSV are appended to and flushed, and their frame count
is read back from the files themselves, never from a record.
"""
import os
import struct
from pathlib import Path

NM_TO_A = 10.0

#: StateDataReporter's header, verbatim, for the five columns written here.
CSV_HEADER = ('#"Step","Time (ps)","Potential Energy (kJ/mole)",'
              '"Kinetic Energy (kJ/mole)","Temperature (K)"')


def topology_for(atomic_numbers):
    """One chain, one residue `MOL`, atoms named by element and index, no bonds.

    Enough for a PDB and for every DCD reader to place the atoms; bonds are not a
    property of a machine-learned potential and are not invented here.
    """
    from openmm import app
    top = app.Topology()
    chain = top.addChain()
    res = top.addResidue("MOL", chain)
    for i, z in enumerate(atomic_numbers):
        el = app.Element.getByAtomicNumber(int(z))
        top.addAtom("{}{}".format(el.symbol, i + 1), el, res)
    return top


def _atomic_write_text(path, text):
    path = Path(path)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _atomic_write_bytes(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)


def write_start_pdb(path, topology, positions_A):
    from openmm import app, unit
    import io
    buf = io.StringIO()
    app.PDBFile.writeFile(topology, (positions_A / NM_TO_A) * unit.nanometer, buf)
    _atomic_write_text(path, buf.getvalue())


def write_system_xml(path, system):
    from openmm import XmlSerializer
    _atomic_write_text(path, XmlSerializer.serialize(system))


def write_integrator_xml(path, integrator):
    from openmm import XmlSerializer
    _atomic_write_text(path, XmlSerializer.serialize(integrator))


def dcd_frame_count(path):
    """Frames in a DCD, from its header; 0 when the file is absent or too short.

    The header is one 84-byte record whose first field is the magic `CORD` and whose
    second int is the frame count, kept current by every `writeModel`. Native byte
    order, which is what `app.DCDFile` writes.
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size < 12:
        return 0
    with open(path, "rb") as fh:
        head = fh.read(12)
    if head[4:8] != b"CORD":
        return 0
    return int(struct.unpack("i", head[8:12])[0])


def csv_row_count(path):
    path = Path(path)
    if not path.exists():
        return 0
    n = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip() and not line.startswith("#"):
                n += 1
    return n


class EngineFolder:
    """The seven files of one trajectory, opened for writing or appending."""

    def __init__(self, folder, topology, frame_spacing_ps):
        self.folder = Path(folder)
        self.topology = topology
        self.frame_spacing_ps = float(frame_spacing_ps)
        self.folder.mkdir(parents=True, exist_ok=True)
        self._dcd_fh = None
        self._dcd = None
        self._csv_fh = None

    # ---- what is on disk -------------------------------------------------------------
    @property
    def paths(self):
        return {n: self.folder / n for n in ("start.pdb", "system.xml", "integrator.xml",
                                             "traj.dcd", "state.csv", "state.xml", "state.chk")}

    def frames_on_disk(self):
        """The trajectory's own count: the DCD header, cross-checked against the CSV.

        A killed job can leave the CSV one row ahead of the DCD (the row is written after
        the frame); the smaller of the two is the number of frames both files hold.
        """
        return min(dcd_frame_count(self.paths["traj.dcd"]),
                   csv_row_count(self.paths["state.csv"])) \
            if self.paths["state.csv"].exists() else dcd_frame_count(self.paths["traj.dcd"])

    def has_state(self):
        p = self.paths
        return p["state.xml"].exists() or p["state.chk"].exists()

    # ---- the start of a trajectory -----------------------------------------------------
    def write_start(self, positions_A, system, integrator):
        write_start_pdb(self.paths["start.pdb"], self.topology, positions_A)
        write_system_xml(self.paths["system.xml"], system)
        write_integrator_xml(self.paths["integrator.xml"], integrator)

    # ---- frames ------------------------------------------------------------------------
    def open_frames(self, append):
        """Open traj.dcd and state.csv; `append` continues the files already there."""
        from openmm import app, unit
        p = self.paths
        append = bool(append and p["traj.dcd"].exists() and dcd_frame_count(p["traj.dcd"]) > 0)
        self._dcd_fh = open(p["traj.dcd"], "r+b" if append else "wb")
        self._dcd = app.DCDFile(self._dcd_fh, self.topology,
                                self.frame_spacing_ps * unit.picosecond,
                                firstStep=0, interval=1, append=append)
        if append and p["state.csv"].exists():
            self._csv_fh = open(p["state.csv"], "a", encoding="utf-8")
        else:
            self._csv_fh = open(p["state.csv"], "w", encoding="utf-8")
            self._csv_fh.write(CSV_HEADER + "\n")
        return self

    def write_frame(self, positions_A, step, time_ps, potential_kJ, kinetic_kJ, temperature_K):
        from openmm import unit
        self._dcd.writeModel((positions_A / NM_TO_A) * unit.nanometer)
        self._csv_fh.write("{},{:.4f},{:.10g},{:.10g},{:.10g}\n".format(
            int(step), float(time_ps), float(potential_kJ), float(kinetic_kJ),
            float(temperature_K)))

    def flush_frames(self):
        for fh in (self._dcd_fh, self._csv_fh):
            if fh is not None:
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except OSError:
                    pass

    def close_frames(self):
        self.flush_frames()
        for fh in (self._dcd_fh, self._csv_fh):
            if fh is not None:
                fh.close()
        self._dcd_fh = self._csv_fh = self._dcd = None

    # ---- state -------------------------------------------------------------------------
    def flush_state(self, context):
        """state.xml and state.chk from the same instant, each written atomically."""
        from openmm import XmlSerializer
        st = context.getState(getPositions=True, getVelocities=True, getEnergy=True,
                              getParameters=True, getIntegratorParameters=True)
        _atomic_write_text(self.paths["state.xml"], XmlSerializer.serialize(st))
        _atomic_write_bytes(self.paths["state.chk"], context.createCheckpoint())

    def load_state(self, context):
        """Restore a context: the checkpoint when it loads here, else the XML state.

        Returns the name of the file that was used, or None when neither exists. A
        checkpoint that does not load (other hardware, other platform) is not an error:
        that is exactly what the XML state is for.
        """
        from openmm import XmlSerializer
        p = self.paths
        if p["state.chk"].exists():
            try:
                context.loadCheckpoint(p["state.chk"].read_bytes())
                return "state.chk"
            except Exception:                                       # noqa: BLE001
                pass
        if p["state.xml"].exists():
            st = XmlSerializer.deserialize(p["state.xml"].read_text(encoding="utf-8"))
            context.setState(st)
            return "state.xml"
        return None
