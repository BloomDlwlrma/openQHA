"""Reference E-F-H labels per frame: the Calculation of ticket 03 of the Hessian-learning set
(CONTEXT.md "Frame", "Frame set", "Dataset").

For every kept frame of a molecule's Frame set (`frames.py`, the MACE level) ORCA computes
the energy, the gradient and the Cartesian Hessian AT THAT FIXED GEOMETRY at one reference
level -- a single point with `EnGrad` and the level's Hessian route (`Freq` analytic, or
`NumFreq` where the level has no analytic Hessian), never an optimisation: the label of a
displaced frame is the raw Hessian there, gradient term included, exactly what the loss
compares to the engine's (note 3 section 2: Rodriguez, HIP, PHL all label the fixed
geometry and leave projection to the loss).

WHAT IS WRITTEN
---------------
    <molecule>/orca.<level>.<generator>_bBB_kK.{inp,out,hess,engrad}
        ORCA's engine files as a FILE GROUP of the molecule directory (ticket 09, ruling
        2026-09-20: no per-frame directory; `layout.orca_frame_stem`), the FULL `.out`
        kept as ORCA wrote it (user ruling 2026-09-17). Basin / merged / saddle frames
        are HESSIAN jobs (`EnGrad Freq`, a `.hess`); displaced frames are GRADIENT jobs
        (`EnGrad`, an `.engrad`, no Hessian -- round 5, Q7 (b)). ORCA itself runs as
        `job.*` in a run directory -- the node-local `scratch`, or `<molecule>/.<stem>/`
        without one -- and the KEEP kinds are copied back under the stem, the run
        directory removed. A finished frame (terminal line in the `.out` and its product
        file) is skipped on a rerun, an unfinished one is rerun -- that is how a Batch
        resumes.
    <molecule>/frames/<generator>.<level>.extxyz
        the labelled frames: positions VERBATIM from the MACE file, `energy` (eV),
        `forces` (eV/A), `hessian` (3N x 3N flattened, eV/A^2) from ORCA. A frame whose
        `.hess` geometry differs in shape from the MACE file by more than POSITION_TOL_A
        is refused (ORCA writes the `.hess` in the centre-of-mass frame; that translation
        is removed and reported): the two levels of a frame must be at the same
        coordinates or the Dataset compares different geometries.
    <molecule>/frames/labels.<level>.{out,toml}
        the Record: what was labelled, reused, refused or failed; wall time and ORCA's
        peak memory per frame; the Hessian route and the rigid-block noise floor of every
        Hessian; ORCA's version.

UNITS: ORCA's Eh, Eh/bohr, Eh/bohr^2 are converted here, once, to eV, eV/A, eV/A^2 --
the engine's units, so the two files of a frame read alike (`orca.EV_PER_HARTREE`,
`orca.BOHR_PER_ANGSTROM`, `orca.hessian_to_ev_per_angstrom2`). Force = -gradient.

HOW IT RUNS
-----------
`label_one` is one frame (one ORCA job, `%pal nprocs 4`, `%maxcore` per core); `assemble`
reads every finished frame of a molecule and writes the file and the Record; `run` is
the two in sequence for one molecule. The Batch driver (`workflows/hessian_learning/
03_labels.py`) fans `label_one` out over Parsl -- 16 frames per node, 4 cores each on
tianhe (role `labels`) -- and assembles per molecule afterwards. ORCA's scratch files
(integrals, response vectors: GB per analytic Hessian) go to node-local `scratch` when one
is given (`S0_SCRATCH` on a cluster); only KEEP is copied back to the molecule tree.
"""
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from ..qm_interfaces import orca
from ..store import layout, property as prop, report
from ..thermochem import hessian as hessian_mod
from . import frames as frames_mod

STEP = "labels"
PROGNAME = "openQHA frame labels"
#: the reference level of the Hessian-learning set (spec, ruling 2026-09-18)
DEFAULT_LEVEL = "wb97m-d3bj_def2-tzvppd"
#: ORCA ranks per frame: 16 frames x 4 = one 64-core tianhe node (ticket 03)
NPROCS = 4
#: %maxcore (MB per rank) on tianhe deimos: 512 GB x 0.75 / 64 cores
MAXCORE_MB = 6000
#: the `.hess` geometry must agree in shape with the MACE file to this (A). Measured
#: 2026-09-18 on oxetane: after removing ORCA's centre-of-mass translation the residual is
#: 1.0e-8 A -- the print precision of the .hess / .engrad coordinates (bohr, 9 decimals ~
#: 3e-8 A), not a geometry difference. 1e-7 is ten times that and ten times below the
#: 1e-6 A distortion the unit test refuses.
POSITION_TOL_A = 1e-7
#: ORCA files published from scratch into the molecule tree (everything else is scratch).
#: Round 5 (2026-09-19): no `.gbw`, no `.loc`, and no `property.txt` (it duplicates the
#: Hessian: 85 KB of 231 per 10-atom frame) -- the full `.out`, the `.hess`, the `.engrad`.
KEEP = (".inp", ".out", ".hess", ".engrad")
#: generators whose frames get a reference HESSIAN (single point + EnGrad + Freq): the
#: stationary conformers. The displaced frames get energy + gradient only (round 5, Q7 (b):
#: the literature trains Hessians at stationary points; an analytic Hessian of a 19-atom
#: molecule is 40-80 min, its gradient 3-5 min).
HESSIAN_GENERATORS = ("basin", "merged", "saddle")
STEM = "job"
TERMINAL = "****ORCA TERMINATED NORMALLY****"
#: a frame is claimed with `<stem>.running` beside its files while ORCA runs; a claim
#: older than LOCK_MAX_AGE_S (a killed job) is ignored. Two Batches over one selection then
#: partition the frames instead of labelling the same ones (two debug jobs did, 2026-09-19).
LOCK = ".running"
LOCK_MAX_AGE_S = 2 * 3600

SCHEMA = {
    "Calculation_Info": {
        "MOLECULE_DIR": ("String", None, "the molecule directory"),
        "QM9_INDEX": ("String", None, "the molecule"),
        "LEVEL": ("String", None, "the reference level (CONTEXT.md spelling; the file suffix)"),
        "KEYWORDS": ("String", None, "ORCA's ! line of a Hessian job: the level's single point + EnGrad + its Hessian route; a gradient job (displaced frames) drops the Freq"),
        "N_HESSIAN_FRAMES": ("Integer", None, "frames labelled with a Hessian (basin / merged / saddle)"),
        "N_GRADIENT_FRAMES": ("Integer", None, "frames labelled with energy + forces only (displaced)"),
        "HESSIAN_ROUTE": ("String", None, "analytic / numerical, as the level declares it (orca.LEVELS)"),
        "ORCA_VERSION": ("String", None, "Program Version of the .out files (the set of versions, if a rerun mixed them)"),
        "NPROCS": ("Integer", None, "ORCA ranks per frame"),
        "MAXCORE_MB": ("Integer", "MB", "%maxcore per rank"),
        "MACE_LEVEL": ("String", None, "the Frame set's engine level whose files supplied the geometries"),
        "POSITION_TOL_A": ("Double", "A", "a .hess geometry further from the MACE file (translation removed) is refused"),
        "N_FRAMES": ("Integer", None, "kept frames of the Frame set, the generators considered"),
        "N_LABELLED": ("Integer", None, "frames with a label in the extxyz (computed now or reused)"),
        "N_COMPUTED": ("Integer", None, "frames ORCA ran in this call"),
        "N_REUSED": ("Integer", None, "frames whose finished ORCA job was on disk before this call"),
        "N_REFUSED": ("Integer", None, "finished jobs whose geometry did not match the MACE file"),
        "N_UNLABELLED": ("Integer", None, "frames with no finished ORCA job (failed or not yet run)"),
        "SECONDS_PER_FRAME": ("Double", "s", "mean ORCA wall time of the labelled frames (TOTAL RUN TIME)"),
        "MAX_MEMORY_MB": ("Double", "MB", "largest 'Maximum memory used' ORCA reported over all frames, per rank"),
        "NOISE_FLOOR_MAX_CM": ("Double", "cm^-1", "largest rigid-block value over the labelled BASIN frames (the noise floor); displaced frames are excluded, their block holds the gradient term"),
        "SECONDS": ("Double", "s", "wall time of this call"),
    },
    "Generator": {
        "GENERATOR": ("String", None, "basin / displaced / merged / saddle"),
        "N_FRAMES": ("Integer", None, "kept frames at the MACE level"),
        "N_LABELLED": ("Integer", None, "frames written to the level's extxyz"),
        "FILE": ("String", None, "the extxyz written, or absent when no frame was labelled"),
    },
    "Frame": {
        "GENERATOR": ("String", None, "the generator"),
        "BASIN": ("Integer", None, "the basin the frame was born from"),
        "K": ("Integer", None, "index within (generator, basin)"),
        "HAS_HESSIAN": ("Boolean", None, "a Hessian job (basin / merged / saddle: EnGrad + Freq) rather than a gradient job (displaced: EnGrad only; round 5, Q7 (b))"),
        "ENERGY": ("Double", "eV", "reference energy (FINAL SINGLE POINT ENERGY, dispersion included)"),
        "ENERGY_ABOVE_BASIN": ("Double", "kcal/mol", "reference energy above the frame's basin frame (nan when the basin frame is unlabelled)"),
        "MAX_FORCE": ("Double", "eV/A", "reference |F|max"),
        "LOWEST_FREQ": ("Double", "cm^-1", "lowest projected eigenvalue of the reference Hessian as a wavenumber (a curvature at a displaced frame)"),
        "NOISE_FLOOR_CM": ("Double", "cm^-1", "rigid-block value of the unprojected Hessian (hessian.rigid_block_floor_cm): the noise floor at a basin frame; at a displaced frame the rotational block carries the gradient term and is physics, not noise (oxetane: 24 at the basin, 178 at a displaced frame)"),
        "MAX_POSITION_DEV_A": ("Double", "A", "largest |.hess geometry - MACE file geometry| after removing the common translation (ORCA writes the .hess in the centre-of-mass frame)"),
        "COM_SHIFT_A": ("Double", "A", "the translation removed (|centre-of-mass shift|)"),
        "HESSIAN_ROUTE": ("String", None, "analytic / numerical / unknown, read from the .out"),
        "SECONDS": ("Double", "s", "ORCA's TOTAL RUN TIME of the job"),
        "MEMORY_MB": ("Double", "MB", "largest 'Maximum memory used' line of the .out, per rank"),
        "ORCA_VERSION": ("String", None, "Program Version of the .out"),
        "STATUS": ("String", None, "labelled / reused / refused / unlabelled"),
        "REASON": ("String", None, "why refused or unlabelled, or -"),
        "OUT": ("String", None, "the ORCA .out"),
    },
}


# ====================================================================== helpers
def frame_tag(generator, basin, k):
    return "{}_b{:02d}_k{}".format(generator, int(basin), int(k))


def keyword_line(level, hessian=True):
    """ORCA's `!` line for a label: the level's single point, `EnGrad`, and -- with
    `hessian` -- `Freq` or `NumFreq` by the level's Hessian route. Returns (keywords,
    blocks, route); route is "gradient" for a gradient-only job."""
    spec = orca.level_spec(level)
    if not hessian:
        return "{} EnGrad".format(spec["single_point"]), spec.get("blocks", ""), "gradient"
    freq = "Freq" if spec["route"] == "analytic" else "NumFreq"
    return "{} EnGrad {}".format(spec["single_point"], freq), spec.get("blocks", ""), spec["route"]


def wants_hessian(generator):
    return generator in HESSIAN_GENERATORS


def frames_record(molecule):
    """The Frame set's Record (`frames/frames.toml`), or a FileNotFoundError naming it."""
    p = layout.frames_dir(molecule) / (frames_mod.STEP + ".toml")
    if not p.is_file():
        raise FileNotFoundError("no Frame set at {} (run 02_frames first)".format(p))
    return prop.load(p)


def mace_level(molecule):
    return frames_record(molecule)["Calculation_Info"]["LEVEL"]


def frame_list(molecule, generators=None):
    """Every kept frame of the Frame set as (generator, basin, k), in file order."""
    rec = frames_record(molecule)
    gens = tuple(generators) if generators else frames_mod.GENERATORS
    return [(r["GENERATOR"], int(r["BASIN"]), int(r["K"])) for r in rec.get("Frame", [])
            if r["STATUS"] == "kept" and r["GENERATOR"] in gens]


def load_frames(molecule, generator, level=None):
    """The MACE-level frames of one generator, as `frames.read_frames` gives them."""
    level = level or mace_level(molecule)
    p = layout.frames_file(molecule, generator, level)
    return frames_mod.read_frames(p) if p.is_file() else []


def load_frame(molecule, generator, basin, k, level=None):
    for a in load_frames(molecule, generator, level):
        if int(a.info["basin"]) == int(basin) and int(a.info["k"]) == int(k):
            return a
    raise KeyError("no frame {} in the {} file of {}".format(frame_tag(generator, basin, k), generator, molecule))


def finished(molecule, stem, hessian=True):
    """True when ORCA terminated normally for the frame `<molecule>/<stem>.*` and left a
    `.hess` (a Hessian job) or an `.engrad` (a gradient job)."""
    out = Path(molecule) / (stem + ".out")
    product = Path(molecule) / (stem + (".hess" if hessian else ".engrad"))
    return product.is_file() and out.is_file() and TERMINAL in out.read_text(encoding="utf-8", errors="replace")


def engrad_positions(path, natoms):
    """The atomic numbers and coordinates (bohr) at the end of an ORCA `.engrad`."""
    vals = [l.strip() for l in Path(path).read_text(encoding="utf-8").split("\n") if l.strip() and not l.strip().startswith("#")]
    rows = vals[2 + 3 * natoms:2 + 4 * natoms]
    if len(rows) != natoms:
        raise ValueError("{} has no coordinate block of {} atoms".format(path, natoms))
    z = [int(float(r.split()[0])) for r in rows]
    xyz = np.array([[float(v) for v in r.split()[1:4]] for r in rows])
    return z, xyz


def lock_file(molecule, stem):
    """`<molecule>/<stem>.running`."""
    return Path(molecule) / (stem + LOCK)


def running_elsewhere(molecule, stem, max_age_s=LOCK_MAX_AGE_S):
    """True when another process holds a fresh claim on this frame."""
    lock = lock_file(molecule, stem)
    try:
        return lock.is_file() and (time.time() - lock.stat().st_mtime) < max_age_s
    except OSError:
        return False


def _claim(molecule, stem):
    """Claim the frame: the lock file names the job (Slurm id or pid). Returns False when
    a fresh claim by someone else is there."""
    Path(molecule).mkdir(parents=True, exist_ok=True)
    if running_elsewhere(molecule, stem):
        return False
    lock_file(molecule, stem).write_text("{} {}\n".format(os.environ.get("SLURM_JOB_ID", "pid{}".format(os.getpid())),
                                                          time.strftime("%Y-%m-%dT%H:%M:%S")), encoding="utf-8")
    return True


def _release(molecule, stem):
    try:
        lock_file(molecule, stem).unlink()
    except OSError:
        pass


def total_run_seconds(out_text):
    """ORCA's `TOTAL RUN TIME: d days h hours m minutes s seconds ms msec`, or None."""
    m = re.search(r"TOTAL RUN TIME:\s+(\d+) days (\d+) hours (\d+) minutes (\d+) seconds (\d+) msec", out_text)
    if not m:
        return None
    d, h, mi, s, ms = (int(x) for x in m.groups())
    return d * 86400 + h * 3600 + mi * 60 + s + ms / 1000.0


def max_memory_mb(out_text):
    """The largest `Maximum memory used throughout the entire X-calculation: N MB`, or None."""
    vals = [float(v) for v in re.findall(r"Maximum memory used throughout the entire \S+-calculation:\s+([\d.]+) MB", out_text)]
    return max(vals) if vals else None


def first_energy_from_out(out_text):
    """The FIRST `FINAL SINGLE POINT ENERGY` of a label job (Eh): the energy at the frame's
    geometry. There is no optimisation here, so the first single point is the frame;
    a NumFreq level prints one more per displacement afterwards."""
    vals = re.findall(r"FINAL SINGLE POINT ENERGY\s+([-+]?\d+\.\d+)", out_text)
    if not vals:
        raise ValueError("no FINAL SINGLE POINT ENERGY in the ORCA output")
    return float(vals[0])


def gradient_from_out(out_text, natoms):
    """The last CARTESIAN GRADIENT block of an ORCA output (Eh/bohr), the fallback when no
    `.engrad` was written."""
    blocks = out_text.split("CARTESIAN GRADIENT")
    if len(blocks) < 2:
        raise ValueError("no CARTESIAN GRADIENT block in the ORCA output")
    grad = []
    for line in blocks[-1].split("\n")[3:]:
        parts = line.split()
        if len(parts) >= 6 and parts[2] == ":":
            grad.append([float(x) for x in parts[3:6]])
        elif grad:
            break
    if len(grad) != natoms:
        raise ValueError("CARTESIAN GRADIENT block has {} atoms, expected {}".format(len(grad), natoms))
    return np.asarray(grad, dtype=float)


def _run_orca(inp, out, cwd, timeout_s=None):
    """The default runner: the ORCA binary on `inp`, stdout to `out`, in `cwd`."""
    with open(out, "w") as fh:
        return subprocess.call([orca.orca_binary(), str(inp)], stdout=fh, stderr=subprocess.STDOUT,
                               cwd=str(cwd), env=orca.subprocess_env(), timeout=timeout_s)


def parse_label(molecule, atoms, stem):
    """Read a finished frame job `<molecule>/<stem>.*`: energy (eV), forces (eV/A), the
    Hessian (eV/A^2) when the job computed one (`hessian` None otherwise), the geometry
    check against `atoms`, wall time, memory, route, version, noise floor (nan without a
    Hessian)."""
    molecule = Path(molecule)
    text = (molecule / (stem + ".out")).read_text(encoding="utf-8", errors="replace")
    symbols = list(atoms.get_chemical_symbols())
    n = len(symbols)
    hess_path, engrad = molecule / (stem + ".hess"), molecule / (stem + ".engrad")
    parsed = orca.parse_hess(hess_path) if hess_path.is_file() else None
    if parsed is not None:
        if parsed["symbols"] != symbols:
            raise ValueError("ORCA reordered the atoms in {}: {} -> {}".format(hess_path, symbols, parsed["symbols"]))
        # ORCA writes the .hess $atoms in the centre-of-mass frame (measured 2026-09-18: every
        # atom of oxetane shifted by the same 0.0764 A). A translation leaves the Cartesian
        # Hessian unchanged, so it is removed before the comparison and reported apart; what
        # must agree is the shape. ORCA does not rotate (no symmetry handling here), and a
        # rotation would show up as a residual after the shift.
        pos_job = np.asarray(parsed["positions_bohr"]) / orca.BOHR_PER_ANGSTROM
    elif engrad.is_file():
        from ase.data import atomic_numbers
        z, pos_bohr = engrad_positions(engrad, n)
        if z != [atomic_numbers[s] for s in symbols]:
            raise ValueError("ORCA reordered the atoms in {}".format(engrad))
        pos_job = pos_bohr / orca.BOHR_PER_ANGSTROM
    else:
        raise FileNotFoundError("neither {} nor {}".format(hess_path.name, engrad.name))
    diff = pos_job - atoms.get_positions()
    shift = diff.mean(axis=0)
    dev = float(np.abs(diff - shift).max())
    if engrad.is_file():
        e_eh, grad = orca._parse_engrad(engrad, n)
    else:
        e_eh, grad = first_energy_from_out(text), gradient_from_out(text, n)
    # the FIRST single point is the frame's: a NumFreq level prints one more per displaced
    # geometry, and the last of those is not the label (review 2026-09-18)
    e_out = first_energy_from_out(text)
    forces = -np.asarray(grad, dtype=float) * orca.EV_PER_HARTREE * orca.BOHR_PER_ANGSTROM
    masses = atoms.get_masses()
    m = re.search(r"Program Version\s+(\S+)", text)
    out = dict(energy=float(e_out) * orca.EV_PER_HARTREE, energy_engrad=float(e_eh) * orca.EV_PER_HARTREE,
               forces=forces, hessian=None, max_position_dev_A=dev, com_shift_A=float(np.linalg.norm(shift)),
               seconds=total_run_seconds(text), memory_mb=max_memory_mb(text),
               hessian_route="gradient", orca_version=m.group(1) if m else "unknown",
               noise_floor_cm=float("nan"), lowest_freq=float("nan"),
               max_force=float(np.abs(forces).max()), out=str(molecule / (stem + ".out")))
    if parsed is not None:
        h = orca.hessian_to_ev_per_angstrom2(parsed["hessian_eh_bohr2"])
        out.update(hessian=np.asarray(h, dtype=float), hessian_route=orca.hessian_route(text),
                   noise_floor_cm=float(hessian_mod.rigid_block_floor_cm(h, masses, atoms.get_positions())),
                   lowest_freq=frames_mod.lowest_projected_cm(h, masses, atoms.get_positions()))
    return out


# ====================================================================== one frame
def label_one(molecule, level, generator, basin, k, nprocs=NPROCS, maxcore=MAXCORE_MB,
              scratch=None, timeout_s=None, runner=None, mace_level_name=None, charge=0, mult=1):
    """ORCA on one frame. Returns the parsed label plus `status` ("labelled" when ORCA ran
    now, "reused" when its finished job was on disk, "refused" on a geometry mismatch,
    "running" when another Batch holds a fresh claim on the frame -- nothing is parsed
    then) and the wall seconds of this call. Raises when ORCA does not terminate normally --
    the caller (a Batch task) records that; the `.out` stays for reading.

    ORCA runs as `job.*` in a run directory -- `scratch/<molecule>/<frame tag>/` (node-
    local) when `scratch` is given, `<molecule>/.<stem>/` otherwise; the KEEP kinds are
    copied back as `<molecule>/<stem>.<ext>` and the run directory removed (the `.out`
    comes back on a failure too, for reading). `runner`: a callable
    (inp, out, cwd, timeout_s) -> rc replacing the ORCA binary (tests)."""
    molecule = Path(molecule)
    t0 = time.time()
    atoms = load_frame(molecule, generator, basin, k, mace_level_name)
    stem = layout.orca_frame_stem(level, generator, basin, k)
    hessian = wants_hessian(generator)
    keywords, blocks, _route = keyword_line(level, hessian=hessian)
    status = "reused"
    if not finished(molecule, stem, hessian=hessian):
        if not _claim(molecule, stem):
            return dict(status="running", generator=generator, basin=int(basin), k=int(k), keywords=keywords,
                        wall_seconds=time.time() - t0, workdir=str(molecule), stem=stem,
                        out=str(molecule / (stem + ".out")),
                        seconds=None, memory_mb=None, hessian_route="-", noise_floor_cm=float("nan"), orca_version="-")
        status = "labelled"
        rundir = Path(scratch) / molecule.name / frame_tag(generator, basin, k) if scratch else molecule / ("." + stem)
        rundir.mkdir(parents=True, exist_ok=True)
        inp, out = rundir / (STEM + ".inp"), rundir / (STEM + ".out")
        inp.write_text(orca.input_text(atoms.get_chemical_symbols(), atoms.get_positions(), keywords,
                                       nprocs, maxcore, charge, mult, blocks), encoding="utf-8")
        try:
            rc = (runner or _run_orca)(inp, out, rundir, timeout_s)
            text = out.read_text(encoding="utf-8", errors="replace") if out.is_file() else ""
            for ext in KEEP:
                f = rundir / (STEM + ext)
                if f.is_file():
                    shutil.copy2(f, molecule / (stem + ext))
            shutil.rmtree(rundir, ignore_errors=True)
        finally:
            _release(molecule, stem)
        if TERMINAL not in text:
            raise RuntimeError("ORCA did not finish normally for {} of {} (rc {}). Tail:\n{}".format(
                frame_tag(generator, basin, k), molecule.name, rc, "\n".join(text.split("\n")[-25:])))
    lab = parse_label(molecule, atoms, stem)
    if lab["max_position_dev_A"] > POSITION_TOL_A:
        status = "refused"
    lab.update(status=status, generator=generator, basin=int(basin), k=int(k), keywords=keywords,
               wall_seconds=time.time() - t0, workdir=str(molecule), stem=stem)
    return lab


# ====================================================================== the molecule
def assemble(molecule, level=DEFAULT_LEVEL, generators=None, nprocs=NPROCS, maxcore=MAXCORE_MB, computed=()):
    """Write `<generator>.<level>.extxyz` from every finished frame job of the molecule and
    the Record `frames/labels.<level>.{out,toml}`. `computed`: the frame tags ORCA ran in
    this call (for the N_COMPUTED / N_REUSED split; everything finished on disk counts
    as reused otherwise)."""
    molecule = Path(molecule)
    t0 = time.time()
    rec = frames_record(molecule)
    info0 = rec["Calculation_Info"]
    mlevel = info0["LEVEL"]
    keywords, _blocks, route = keyword_line(level)
    computed = set(computed)
    rows, kept, e_basin = [], {}, {}
    gens = tuple(generators) if generators else frames_mod.GENERATORS
    by_gen = {g: load_frames(molecule, g, mlevel) for g in gens}

    # the basin frames first, so ENERGY_ABOVE_BASIN of the others can be stated
    order = [(g, a) for g in gens for a in by_gen[g]]
    order.sort(key=lambda ga: 0 if ga[0] == "basin" else 1)
    for g, a in order:
        b, k = int(a.info["basin"]), int(a.info["k"])
        tag = frame_tag(g, b, k)
        stem = layout.orca_frame_stem(level, g, b, k)
        row = dict(GENERATOR=g, BASIN=b, K=k, HAS_HESSIAN=wants_hessian(g), ENERGY=float("nan"), ENERGY_ABOVE_BASIN=float("nan"),
                   MAX_FORCE=float("nan"), LOWEST_FREQ=float("nan"), NOISE_FLOOR_CM=float("nan"),
                   MAX_POSITION_DEV_A=float("nan"), COM_SHIFT_A=float("nan"), HESSIAN_ROUTE="-", SECONDS=float("nan"),
                   MEMORY_MB=float("nan"), ORCA_VERSION="-", STATUS="unlabelled", REASON="-",
                   OUT=str(molecule / (stem + ".out")))
        if not finished(molecule, stem, hessian=wants_hessian(g)):
            row["REASON"] = "no finished ORCA job {} in {}".format(stem, molecule)
            rows.append(row)
            continue
        try:
            lab = parse_label(molecule, a, stem)
        except Exception as exc:
            row["REASON"] = "{}: {}".format(type(exc).__name__, str(exc)[:160])
            rows.append(row)
            continue
        row.update(ENERGY=lab["energy"], MAX_FORCE=lab["max_force"], LOWEST_FREQ=lab["lowest_freq"],
                   NOISE_FLOOR_CM=lab["noise_floor_cm"], MAX_POSITION_DEV_A=lab["max_position_dev_A"],
                   COM_SHIFT_A=lab["com_shift_A"],
                   HESSIAN_ROUTE=lab["hessian_route"], SECONDS=lab["seconds"] if lab["seconds"] is not None else float("nan"),
                   MEMORY_MB=lab["memory_mb"] if lab["memory_mb"] is not None else float("nan"),
                   ORCA_VERSION=lab["orca_version"])
        if lab["max_position_dev_A"] > POSITION_TOL_A:
            row.update(STATUS="refused", REASON="the .hess geometry differs from the MACE file by {:.2e} A > {:.0e}".format(
                lab["max_position_dev_A"], POSITION_TOL_A))
            rows.append(row)
            continue
        row["STATUS"] = "labelled" if tag in computed else "reused"
        if g == "basin":
            e_basin[b] = lab["energy"]
        if b in e_basin:
            row["ENERGY_ABOVE_BASIN"] = (lab["energy"] - e_basin[b]) * frames_mod.EV_TO_KCAL
        rows.append(row)
        info = {key: a.info[key] for key in ("qm9_index", "basin", "generator", "k", "seed", "source_conformer",
                                             "rms_displacement_A", "smiles") if key in a.info}
        info.update(level=level, orca_version=lab["orca_version"], hessian_route=lab["hessian_route"],
                    noise_floor_cm=lab["noise_floor_cm"], keywords=keyword_line(level, hessian=wants_hessian(g))[0],
                    has_hessian=lab["hessian"] is not None)
        kept.setdefault(g, []).append(dict(atoms=a, energy=lab["energy"], forces=lab["forces"],
                                           hessian=lab["hessian"], info=info))

    gen_rows = []
    for g in gens:
        path = layout.frames_file(molecule, g, level)
        n_l = len(kept.get(g, []))
        if n_l:
            frames_mod._write_frames(path, kept[g])
        elif path.is_file():
            path.unlink()                     # a rerun that lost every label must not leave a stale file
        gen_rows.append(dict(GENERATOR=g, N_FRAMES=len(by_gen[g]), N_LABELLED=n_l, FILE=str(path) if n_l else None))
    lab_rows = [r for r in rows if r["STATUS"] in ("labelled", "reused")]
    secs = [r["SECONDS"] for r in lab_rows if np.isfinite(r["SECONDS"])]
    mems = [r["MEMORY_MB"] for r in lab_rows if np.isfinite(r["MEMORY_MB"])]
    versions = sorted({r["ORCA_VERSION"] for r in lab_rows})
    info = dict(MOLECULE_DIR=str(molecule), QM9_INDEX=str(info0["QM9_INDEX"]), LEVEL=level, KEYWORDS=keywords,
                HESSIAN_ROUTE=route, ORCA_VERSION=", ".join(versions) if versions else "-",
                NPROCS=int(nprocs), MAXCORE_MB=int(maxcore), MACE_LEVEL=mlevel, POSITION_TOL_A=POSITION_TOL_A,
                N_FRAMES=len(rows), N_LABELLED=len(lab_rows),
                N_HESSIAN_FRAMES=sum(1 for r in lab_rows if r["HAS_HESSIAN"]),
                N_GRADIENT_FRAMES=sum(1 for r in lab_rows if not r["HAS_HESSIAN"]),
                N_COMPUTED=sum(1 for r in rows if r["STATUS"] == "labelled"),
                N_REUSED=sum(1 for r in rows if r["STATUS"] == "reused"),
                N_REFUSED=sum(1 for r in rows if r["STATUS"] == "refused"),
                N_UNLABELLED=sum(1 for r in rows if r["STATUS"] == "unlabelled"),
                SECONDS_PER_FRAME=float(np.mean(secs)) if secs else float("nan"),
                MAX_MEMORY_MB=float(max(mems)) if mems else float("nan"),
                NOISE_FLOOR_MAX_CM=float(max([r["NOISE_FLOOR_CM"] for r in lab_rows if r["GENERATOR"] == "basin" and np.isfinite(r["NOISE_FLOOR_CM"])] or [float("nan")])),
                SECONDS=time.time() - t0)
    toml = record_path(molecule, level)
    missing = prop.write(toml, {"Calculation_Info": info, "Generator": gen_rows, "Frame": rows},
                         SCHEMA, prop.NORMAL_TERMINATION, PROGNAME)
    if missing:
        raise RuntimeError("labels toml keys outside the schema: {}".format(missing))
    _write_report(toml.with_name(toml.name[:-5] + ".out"), info, gen_rows, rows)
    return dict(info=info, generators=gen_rows, frames=rows, record=toml)


def record_path(molecule, level=DEFAULT_LEVEL):
    """`<molecule>/frames/labels.<level>.toml`, the Record of one molecule's labels at one level."""
    return layout.frames_dir(molecule) / "{}.{}.toml".format(STEP, level)


def run(molecule, level=DEFAULT_LEVEL, generators=None, nprocs=NPROCS, maxcore=MAXCORE_MB,
        scratch=None, timeout_s=None, runner=None, progress=None):
    """Every frame of one molecule in sequence, then `assemble`. A frame whose ORCA fails
    is reported (`unlabelled`), not fatal: the Record says which, and a rerun retries it."""
    molecule = Path(molecule)
    mlevel = mace_level(molecule)
    computed, failures = [], []
    for g, b, k in frame_list(molecule, generators):
        try:
            lab = label_one(molecule, level, g, b, k, nprocs=nprocs, maxcore=maxcore, scratch=scratch,
                            timeout_s=timeout_s, runner=runner, mace_level_name=mlevel)
        except Exception as exc:
            failures.append((frame_tag(g, b, k), "{}: {}".format(type(exc).__name__, str(exc)[:200])))
            if progress:
                progress("FAILED  {} {}: {}".format(molecule.name, frame_tag(g, b, k), str(exc)[:120]))
            continue
        if lab["status"] == "running":
            if progress:
                progress("{} {} running elsewhere, skipped".format(molecule.name, frame_tag(g, b, k)))
            continue
        if lab["status"] == "labelled":
            computed.append(frame_tag(g, b, k))
        if progress:
            progress("{} {} {} {:.0f} s ({})".format(molecule.name, frame_tag(g, b, k), lab["status"],
                                                   lab["seconds"] or lab["wall_seconds"], lab["hessian_route"]))
    out = assemble(molecule, level, generators, nprocs, maxcore, computed=computed)
    out["failures"] = failures
    return out


def _write_report(path, info, gen_rows, rows):
    rep = report.Report(PROGNAME, "reference E-F-H labels of {} at {}".format(info["QM9_INDEX"], info["LEVEL"]))
    rep.section("conventions")
    for k in ("LEVEL", "KEYWORDS", "HESSIAN_ROUTE", "ORCA_VERSION", "NPROCS", "MAXCORE_MB", "MACE_LEVEL",
              "POSITION_TOL_A", "N_FRAMES", "N_LABELLED", "N_HESSIAN_FRAMES", "N_GRADIENT_FRAMES", "N_COMPUTED", "N_REUSED", "N_REFUSED", "N_UNLABELLED",
              "SECONDS_PER_FRAME", "MAX_MEMORY_MB", "NOISE_FLOOR_MAX_CM"):
        rep.kv(k, info[k])
    rep.section("per generator")
    rep.table(["generator", "frames", "labelled", "file"],
              [[r["GENERATOR"], r["N_FRAMES"], r["N_LABELLED"], Path(r["FILE"]).name if r["FILE"] else "-"] for r in gen_rows])
    rep.section("per frame (eV; dE kcal/mol above the basin frame; eV/A; cm^-1; A; s; MB)")
    rep.table(["generator", "basin", "k", "H", "E", "dE", "|F|max", "lowest", "rigid", "|dx|max", "com", "route", "s", "MB", "status", "reason"],
              [[r["GENERATOR"], r["BASIN"], r["K"], "yes" if r["HAS_HESSIAN"] else "-", "%.6f" % r["ENERGY"], "%.2f" % r["ENERGY_ABOVE_BASIN"],
                "%.4f" % r["MAX_FORCE"], "%.1f" % r["LOWEST_FREQ"], "%.2f" % r["NOISE_FLOOR_CM"],
                "%.1e" % r["MAX_POSITION_DEV_A"], "%.4f" % r["COM_SHIFT_A"], r["HESSIAN_ROUTE"], "%.0f" % r["SECONDS"],
                "%.0f" % r["MEMORY_MB"], r["STATUS"], r["REASON"]] for r in rows])
    rep.note("a label is ORCA's energy, gradient and -- at basin / merged / saddle frames ('H' = yes) -- raw "
             "Cartesian Hessian at the FRAME'S FIXED GEOMETRY (single point + EnGrad [+ Freq/NumFreq]; no "
             "optimisation; displaced frames get energy + forces only, round 5 Q7 (b)), converted once to eV, eV/A, eV/A^2 "
             "and written beside the MACE file with the same positions. 'lowest' at a displaced frame is a "
             "curvature; 'rigid' is the rigid-body block of the unprojected Hessian -- the noise floor at a basin "
             "frame (a few cm^-1 to ~30 for an analytic Hessian), the gradient term at a displaced frame. 'com' is "
             "the centre-of-mass translation ORCA applied in the .hess, removed before the geometry check. A "
             "refused frame's .hess geometry differs in shape from the MACE file: the two levels "
             "of a frame must sit at one geometry or the Dataset compares different points. The full .out of "
             "every job is kept as <molecule>/orca.<level>.<frame>.out.")
    rep.write(path, step=STEP)


# ====================================================================== the worker
def main(argv=None):
    """One frame from the command line -- the `xargs` worker of hpc/slurm/hl_labels.slurm
    (the hkuhpc shape: a task list, one process per line, idempotent, judged by the
    terminal line). Prints one line `<tag> <status> <seconds> <MB>`; exit 0 when the frame
    is labelled, reused, refused or running elsewhere, 1 when ORCA failed.

        python -m openqha.data.frame_labels <molecule dir> <generator> <basin> <k> \
            [--level L] [--nprocs 4] [--maxcore 6000] [--scratch DIR]
    """
    import argparse
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("molecule")
    ap.add_argument("generator")
    ap.add_argument("basin", type=int)
    ap.add_argument("k", type=int)
    ap.add_argument("--level", default=DEFAULT_LEVEL)
    ap.add_argument("--nprocs", type=int, default=NPROCS)
    ap.add_argument("--maxcore", type=int, default=MAXCORE_MB)
    ap.add_argument("--scratch", default=os.environ.get("S0_SCRATCH") or None)
    ap.add_argument("--timeout", type=float, default=None)
    a = ap.parse_args(argv)
    tag = "{} {}".format(Path(a.molecule).name, frame_tag(a.generator, a.basin, a.k))
    try:
        lab = label_one(a.molecule, a.level, a.generator, a.basin, a.k, nprocs=a.nprocs, maxcore=a.maxcore,
                        scratch=a.scratch, timeout_s=a.timeout)
    except Exception as exc:
        print("{} FAILED {}: {}".format(tag, type(exc).__name__, str(exc).strip().splitlines()[-1][:200] if str(exc).strip() else ""), flush=True)
        return 1
    print("{} {} {} {}".format(tag, lab["status"],
                               "{:.0f}".format(lab["seconds"]) if lab.get("seconds") is not None else "-",
                               "{:.0f}".format(lab["memory_mb"]) if lab.get("memory_mb") is not None else "-"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
