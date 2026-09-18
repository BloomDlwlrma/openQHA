"""One trajectory's Record: `md.out` (the Report) and `md.toml` (the Property file).

    <molecule>/_records/md_<route>/basinNN/md.toml        default setting
    <molecule>/_records/md_<route>/basinNN/md_<setting>.toml   any other setting, same folder
    <molecule>/_records/md_<route>/basinNN/md.out         the Report; last line = terminal line

Records redesign (user ruling 2026-09-15, tickets 16-18): no setting LEVEL under
`_records` (the setting is in the stem, `layout.record_file_name`), and the Property
file takes ORCA's `.property.txt` shape, only what a later step reads:

    [Calculation_Status]   PROGNAME VERSION STATUS   (RUNNING after equilibration, NORMAL TERMINATION at the end)
    [Calculation_Info]     molecule, basin, route, setting, seed, the identity inputs (thermostat, timestep,
                           hydrogen mass, ...), temperature, frame spacing, platform
    [Masses]               SYMBOLS MASSES          what the driver integrated with (the analysis weights by them)
    [Force_Check]          AGREES DELTA_ENERGY MAX_DELTA_FORCE    OpenMM's force against ASE's
    [Equilibration]        DONE PS WALL T_SECOND_HALF DRIFT
    [Production]           N_FRAMES N_TARGET COMPLETE PS WALL SECONDS_PER_PS T_MEAN T_DEVIATION COM_DRIFT ...
    [[Segment]]            INDEX SEED FRAMES_FROM FRAMES_TO      the resume history

Engine internals, the integrator's parameters, the relaxation, the neighbour-list probe,
the file lists and the job id are printed in `md.out` only.

Both drivers build one flat `meta` dict (the identity assertion's vocabulary, unchanged
since 2026-09-07); `blocks_from_meta` maps it onto the blocks, and `read` maps the
blocks back onto the same flat vocabulary, so the resume, the analysis, 02d and the
Batch driver keep asking the questions they asked (`meta["production"]["complete"]`,
`meta["symbols"]`, `meta.get("hydrogen_mass_amu")`).

STATUS RUNNING: the OpenMM driver writes the Property file the moment equilibration
ends, so a job killed during production resumes from the frames in the DCD without
re-equilibrating; `read` returns that partial record with `status == "RUNNING"`.
"""
from pathlib import Path

from ..store import layout, property as prop, report as _report

PROGNAME = "openQHA md"
STEP = "md"

SCHEMA = {
    prop.INFO_BLOCK: {
        "QM9_INDEX": ("String", None, "the molecule"),
        "BASIN": ("Integer", None, "the basin this trajectory started from"),
        "ROUTE": ("String", None, "openmm or ase"),
        "SETTING": ("String", None, "the setting name (default, or a row of a settings array)"),
        "ENGINE": ("String", None, "the potential"),
        "SEED": ("Integer", None, "velocity seed of the last segment"),
        "SEED_FORMULA": ("String", None, "how the seed was chosen"),
        "THERMOSTAT": ("String", None, "the integrator (identity input)"),
        "THERMOSTAT_TDAMP": ("Double", "fs", "Nose-Hoover coupling time (identity input)"),
        "THERMOSTAT_CHAIN_LENGTH": ("Integer", None, "Nose-Hoover chain length (identity input)"),
        "THERMOSTAT_FIXCM": ("Boolean", None, "ASE Langevin fixcm (identity input; must be false)"),
        "TIMESTEP": ("Double", "fs", "integration step (identity input)"),
        "HMASS": ("Double", "amu", "hydrogen mass (identity input; the real one)"),
        "BIAS_POTENTIAL": ("String", None, "identity input; absent means none"),
        "CONSTRAINTS": ("String", None, "identity input; absent means none"),
        "SOURCE": ("String", None, "which driver produced the trajectory (identity input)"),
        "ENSEMBLE": ("String", None, "the ensemble sampled"),
        "TEMPERATURE": ("Double", "K", "thermostat target"),
        "SAMPLE_EVERY": ("Integer", "steps", "frames are written every this many steps"),
        "FRAME_SPACING": ("Double", "fs", "time between frames"),
        "PLATFORM": ("String", None, "OpenMM platform, or the ASE thermostat class"),
        "COM_PINNED": ("Boolean", None, "centre of mass pinned to the origin"),
        "GEOMETRY_SOURCE": ("String", None, "where the starting geometry came from"),
        "ENGINE_FOLDER": ("String", None, "the engine folder of this trajectory"),
    },
    "Masses": {
        "SYMBOLS": ("ArrayOfStrings", None, "atom order of the trajectory (branch A's)"),
        "MASSES": ("ArrayOfDoubles", "amu", "the masses the driver integrated with"),
    },
    "Force_Check": {
        "AGREES": ("Boolean", None, "the engine's force equals ASE's at the start geometry"),
        "DELTA_ENERGY": ("Double", "eV", "engine minus ASE"),
        "MAX_DELTA_FORCE": ("Double", "eV/A", "largest force component difference"),
    },
    "Equilibration": {
        "DONE": ("Boolean", None, "equilibration finished (the resume marker)"),
        "PS": ("Double", "ps", "length"),
        "WALL": ("Double", "s", "wall time"),
        "T_SECOND_HALF": ("Double", "K", "mean temperature over the second half"),
        "DRIFT": ("Double", "kJ/mol", "potential energy, second half minus first"),
    },
    "Production": {
        "N_FRAMES": ("Integer", None, "frames in the engine file"),
        "N_TARGET": ("Integer", None, "frames asked for"),
        "COMPLETE": ("Boolean", None, "N_FRAMES reached N_TARGET"),
        "PS": ("Double", "ps", "length asked for"),
        "WALL": ("Double", "s", "wall time of this run's production"),
        "SECONDS_PER_PS": ("Double", "s/ps", "cost of this run (its own frames only)"),
        "T_MEAN": ("Double", "K", "mean temperature over production"),
        "T_DEVIATION": ("Double", "K", "T_MEAN minus TEMPERATURE"),
        "COM_DRIFT": ("Double", "A", "centre-of-mass drift over production"),
        "STOPPED_ON_WALL_BUDGET": ("Boolean", None, "cut by the wall budget"),
        "RESUMED": ("Boolean", None, "this run appended to frames already on disk"),
        "N_FRAMES_ALREADY_ON_DISK": ("Integer", None, "frames found at the start of this run"),
    },
    "Segment": {
        "SEED": ("Integer", None, "velocity seed of this segment"),
        "FRAMES_FROM": ("Integer", None, "first frame index of this segment"),
        "FRAMES_TO": ("Integer", None, "one past its last frame"),
    },
}


def _f(v):
    return None if v is None else float(v)


def _i(v):
    return None if v is None else int(v)


def _b(v):
    return None if v is None else bool(v)


def paths(records_dir, setting=layout.DEFAULT_SETTING):
    """(md.toml, md.out, driver.log) for this setting, in `records_dir`."""
    d = Path(records_dir)
    return (d / layout.record_file_name("md.toml", setting),
            d / layout.record_file_name("md.out", setting),
            d / layout.record_file_name("driver.log", setting))


# ---- meta -> blocks ---------------------------------------------------------------------
def blocks_from_meta(meta):
    fc = meta.get("force_check_against_ase") or {}
    eq = meta.get("equilibration") or {}
    pr = meta.get("production") or {}
    info = {
        "QM9_INDEX": meta.get("qm9_index"),
        "BASIN": _i(meta.get("basin_index")),
        "ROUTE": meta.get("route"),
        "SETTING": meta.get("setting"),
        "ENGINE": meta.get("engine_name") or (meta.get("engine") or {}).get("engine"),
        "SEED": _i(meta.get("seed")),
        "SEED_FORMULA": meta.get("seed_formula"),
        "THERMOSTAT": meta.get("thermostat"),
        "THERMOSTAT_TDAMP": _f(meta.get("thermostat_tdamp_fs")),
        "THERMOSTAT_CHAIN_LENGTH": _i(meta.get("thermostat_chain_length")),
        "THERMOSTAT_FIXCM": _b(meta.get("thermostat_fixcm")),
        "TIMESTEP": _f(meta.get("timestep_fs")),
        "HMASS": _f(meta.get("hydrogen_mass_amu")),
        "BIAS_POTENTIAL": meta.get("bias_potential"),
        "CONSTRAINTS": meta.get("constraints"),
        "SOURCE": meta.get("source"),
        "ENSEMBLE": meta.get("ensemble"),
        "TEMPERATURE": _f(meta.get("temperature_K")),
        "SAMPLE_EVERY": _i(meta.get("sample_every_steps")),
        "FRAME_SPACING": _f(meta.get("frame_spacing_fs")),
        "PLATFORM": meta.get("platform"),
        "COM_PINNED": _b(meta.get("centre_of_mass_pinned_to_origin")),
        "GEOMETRY_SOURCE": meta.get("geometry_source"),
        "ENGINE_FOLDER": meta.get("engine_folder"),
    }
    blocks = {prop.INFO_BLOCK: info}
    if meta.get("symbols") is not None:
        blocks["Masses"] = {"SYMBOLS": [str(s) for s in meta["symbols"]],
                            "MASSES": [float(x) for x in (meta.get("masses_amu") or [])]}
    if fc:
        blocks["Force_Check"] = {"AGREES": _b(fc.get("agrees")),
                                 "DELTA_ENERGY": _f(fc.get("delta_energy_eV")),
                                 "MAX_DELTA_FORCE": _f(fc.get("max_delta_force_eV_per_A"))}
    blocks["Equilibration"] = {
        "DONE": bool(meta.get("equilibration_done", bool(eq))),
        "PS": _f(eq.get("ps")), "WALL": _f(eq.get("wall_seconds")),
        "T_SECOND_HALF": _f(eq.get("temperature_second_half_K")),
        "DRIFT": _f(eq.get("drift_kJ")),
    }
    blocks["Production"] = {
        "N_FRAMES": _i(pr.get("n_frames")), "N_TARGET": _i(pr.get("n_target")),
        "COMPLETE": _b(pr.get("complete")), "PS": _f(pr.get("ps")),
        "WALL": _f(pr.get("wall_seconds")),
        "SECONDS_PER_PS": _f(pr.get("seconds_per_ps_this_run", pr.get("seconds_per_ps"))),
        "T_MEAN": _f(pr.get("temperature_mean_K")),
        "T_DEVIATION": _f(pr.get("temperature_deviation_K")),
        "COM_DRIFT": _f(pr.get("centre_of_mass_drift_A")),
        "STOPPED_ON_WALL_BUDGET": _b(pr.get("stopped_on_wall_budget")),
        "RESUMED": _b(pr.get("resumed")),
        "N_FRAMES_ALREADY_ON_DISK": _i(pr.get("n_frames_already_on_disk")),
    }
    segs = meta.get("segments") or []
    if segs:
        blocks["Segment"] = [{"INDEX": i, "SEED": _i(s.get("seed")),
                              "FRAMES_FROM": _i(s.get("frames_from")), "FRAMES_TO": _i(s.get("frames_to"))}
                             for i, s in enumerate(segs)]
    return blocks


# ---- blocks -> meta ---------------------------------------------------------------------
def meta_from_doc(doc):
    """The flat vocabulary the readers use, from a loaded Property file."""
    info = doc.get(prop.INFO_BLOCK) or {}
    ms = doc.get("Masses") or {}
    fc = doc.get("Force_Check") or {}
    eq = doc.get("Equilibration") or {}
    pr = doc.get("Production") or {}
    st = doc.get(prop.STATUS_BLOCK) or {}
    meta = dict(
        status=st.get("STATUS"),
        qm9_index=info.get("QM9_INDEX"), basin_index=info.get("BASIN"), route=info.get("ROUTE"),
        setting=info.get("SETTING"), engine_name=info.get("ENGINE"),
        seed=info.get("SEED"), seed_formula=info.get("SEED_FORMULA"),
        thermostat=info.get("THERMOSTAT"), thermostat_tdamp_fs=info.get("THERMOSTAT_TDAMP"),
        thermostat_chain_length=info.get("THERMOSTAT_CHAIN_LENGTH"),
        thermostat_fixcm=info.get("THERMOSTAT_FIXCM"),
        timestep_fs=info.get("TIMESTEP"), hydrogen_mass_amu=info.get("HMASS"),
        bias_potential=info.get("BIAS_POTENTIAL"), constraints=info.get("CONSTRAINTS"),
        source=info.get("SOURCE"), ensemble=info.get("ENSEMBLE"),
        temperature_K=info.get("TEMPERATURE"), sample_every_steps=info.get("SAMPLE_EVERY"),
        frame_spacing_fs=info.get("FRAME_SPACING"), platform=info.get("PLATFORM"),
        centre_of_mass_pinned_to_origin=info.get("COM_PINNED"),
        geometry_source=info.get("GEOMETRY_SOURCE"), engine_folder=info.get("ENGINE_FOLDER"),
        equilibration_done=bool(eq.get("DONE")),
        equilibration=dict(ps=eq.get("PS"), wall_seconds=eq.get("WALL"),
                           temperature_second_half_K=eq.get("T_SECOND_HALF"), drift_kJ=eq.get("DRIFT")),
        production=dict(n_frames=pr.get("N_FRAMES"), n_target=pr.get("N_TARGET"),
                        complete=pr.get("COMPLETE"), ps=pr.get("PS"), wall_seconds=pr.get("WALL"),
                        seconds_per_ps=pr.get("SECONDS_PER_PS"),
                        seconds_per_ps_this_run=pr.get("SECONDS_PER_PS"),
                        temperature_mean_K=pr.get("T_MEAN"), temperature_deviation_K=pr.get("T_DEVIATION"),
                        centre_of_mass_drift_A=pr.get("COM_DRIFT"),
                        stopped_on_wall_budget=pr.get("STOPPED_ON_WALL_BUDGET"),
                        resumed=pr.get("RESUMED"), n_frames_already_on_disk=pr.get("N_FRAMES_ALREADY_ON_DISK")),
        segments=[dict(seed=s.get("SEED"), frames_from=s.get("FRAMES_FROM"), frames_to=s.get("FRAMES_TO"))
                  for s in sorted(doc.get("Segment") or [], key=lambda s: int(s.get("INDEX", 0)))],
    )
    if ms:
        meta["symbols"] = list(ms.get("SYMBOLS") or [])
        meta["masses_amu"] = list(ms.get("MASSES") or [])
    if fc:
        meta["force_check_against_ase"] = dict(agrees=fc.get("AGREES"), delta_energy_eV=fc.get("DELTA_ENERGY"),
                                               max_delta_force_eV_per_A=fc.get("MAX_DELTA_FORCE"))
    return {k: v for k, v in meta.items() if v is not None}


def read_doc(records_dir, setting=layout.DEFAULT_SETTING):
    """The Property file as blocks, or None when absent or unreadable."""
    p = paths(records_dir, setting)[0]
    if not p.is_file():
        return None
    try:
        return prop.load(p)
    except Exception:                                                 # noqa: BLE001
        return None


def read(records_dir, setting=layout.DEFAULT_SETTING):
    """The flat record (`meta_from_doc`) with `status`, or None (a resume then starts over)."""
    doc = read_doc(records_dir, setting)
    return None if doc is None else meta_from_doc(doc)


def status(records_dir, setting=layout.DEFAULT_SETTING):
    return prop.status_of(paths(records_dir, setting)[0])


# ---- writers ----------------------------------------------------------------------------
def write_running(meta, records_dir, setting=layout.DEFAULT_SETTING):
    """The Property file alone, STATUS RUNNING: written the moment equilibration ends so a
    kill during production resumes from the DCD without re-equilibrating."""
    toml_path = paths(records_dir, setting)[0]
    return prop.write(toml_path, blocks_from_meta(meta), SCHEMA, status=prop.RUNNING, progname=PROGNAME)


def write(meta, records_dir, setting=layout.DEFAULT_SETTING):
    """The Report first, then the Property file with STATUS NORMAL TERMINATION (the marker,
    so it is the last thing written). Returns the keys the schema did not know."""
    toml_path, out_path, _ = paths(records_dir, setting)
    _write_report(meta, out_path)
    return prop.write(toml_path, blocks_from_meta(meta), SCHEMA,
                      status=prop.NORMAL_TERMINATION, progname=PROGNAME)


def _write_report(meta, out_path):
    r = _report.Report("openQHA branch B -- one trajectory",
                       subtitle="{}  basin {}  setting {}  route {}".format(
                           meta.get("qm9_index"), meta.get("basin_index"), meta.get("setting"),
                           meta.get("route")))
    eng = meta.get("engine") or {}
    r.section("Provenance")
    r.kv("engine", eng.get("engine") or meta.get("engine_name"))
    for k in ("weights_path", "bytes", "params_sha256", "n_tensors", "params_pin_status", "interface",
              "mace_torch_version", "torch_version", "dtype"):
        if k in eng:
            r.kv(k, eng[k])
    for k in ("composite_notation", "protocol_source", "protocol_status", "slurm_job_id", "platform"):
        if k in meta:
            r.kv(k, meta[k])
    r.section("Identity (what the analysis refuses without)")
    for k in ("bias_potential", "constraints", "hydrogen_mass_amu", "timestep_fs", "thermostat",
              "thermostat_tdamp_fs", "thermostat_chain_length", "thermostat_fixcm", "source", "ensemble"):
        if k in meta:
            r.kv(k, meta[k])
    r.section("Run")
    for k in ("seed", "seed_formula", "temperature_K", "sample_every_steps", "frame_spacing_fs",
              "geometry_source", "engine_folder", "centre_of_mass_pinned_to_origin"):
        if k in meta:
            r.kv(k, meta[k])
    for name in ("integrator", "force", "force_check_against_ase", "relaxation", "equilibration", "production"):
        block = meta.get(name)
        if isinstance(block, dict):
            r.section(name)
            for k, v in block.items():
                if not isinstance(v, (dict, list)):
                    r.kv(k, v)
    files = (meta.get("engine_files") or (meta.get("production") or {}).get("files") or {})
    if files:
        r.table(["engine file", "path"], [[k, v] for k, v in files.items()], title="engine files")
    segs = meta.get("segments") or []
    if segs:
        r.table(["seed", "frames_from", "frames_to"],
                [[s.get("seed"), s.get("frames_from"), s.get("frames_to")] for s in segs],
                title="segments (one velocity draw each)")
    r.json_dump(meta, title="complete record (expanded)")
    return r.write(out_path, step=STEP)
