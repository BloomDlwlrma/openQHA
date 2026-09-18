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
    <molecule>/orca/<level>/frames/<generator>_bBB_kK/job.{inp,out,hess,engrad,...}
        ORCA's engine files, the FULL `.out` kept as ORCA wrote it (user ruling
        2026-09-17). A finished frame (terminal line in the `.out` and a `.hess`) is
        skipped on a rerun, an unfinished one is rerun -- that is how a Batch resumes.
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
#: ORCA files published from scratch into the molecule tree (everything else is scratch)
KEEP = (".inp", ".out", ".hess", ".engrad", ".xyz", ".property.txt", "_property.txt")
STEM = "job"
TERMINAL = "****ORCA TERMINATED NORMALLY****"

SCHEMA = {
    "Calculation_Info": {
        "MOLECULE_DIR": ("String", None, "the molecule directory"),
        "QM9_INDEX": ("String", None, "the molecule"),
        "LEVEL": ("String", None, "the reference level (CONTEXT.md spelling; the file suffix)"),
        "KEYWORDS": ("String", None, "ORCA's ! line: the level's single point + EnGrad + its Hessian route"),
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


def keyword_line(level):
    """ORCA's `!` line for a label: the level's single point, `EnGrad`, and `Freq` or
    `NumFreq` by the level's Hessian route. Returns (keywords, blocks, route)."""
    spec = orca.level_spec(level)
    freq = "Freq" if spec["route"] == "analytic" else "NumFreq"
    return "{} EnGrad {}".format(spec["single_point"], freq), spec.get("blocks", ""), spec["route"]


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


def finished(workdir, stem=STEM):
    """True when ORCA terminated normally there and left a `.hess`."""
    out, hess = Path(workdir) / (stem + ".out"), Path(workdir) / (stem + ".hess")
    return hess.is_file() and out.is_file() and TERMINAL in out.read_text(encoding="utf-8", errors="replace")


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


def parse_label(workdir, atoms, stem=STEM):
    """Read a finished frame job: energy (eV), forces (eV/A), Hessian (eV/A^2), the
    geometry check against `atoms`, wall time, memory, route, version, noise floor."""
    workdir = Path(workdir)
    text = (workdir / (stem + ".out")).read_text(encoding="utf-8", errors="replace")
    parsed = orca.parse_hess(workdir / (stem + ".hess"))
    symbols = list(atoms.get_chemical_symbols())
    if parsed["symbols"] != symbols:
        raise ValueError("ORCA reordered the atoms in {}: {} -> {}".format(workdir, symbols, parsed["symbols"]))
    # ORCA writes the .hess $atoms in the centre-of-mass frame (measured 2026-09-18: every
    # atom of oxetane shifted by the same 0.0764 A). A translation leaves the Cartesian
    # Hessian unchanged, so it is removed before the comparison and reported apart; what
    # must agree is the shape. ORCA does not rotate (no symmetry handling here), and a
    # rotation would show up as a residual after the shift.
    pos_hess = np.asarray(parsed["positions_bohr"]) / orca.BOHR_PER_ANGSTROM
    diff = pos_hess - atoms.get_positions()
    shift = diff.mean(axis=0)
    dev = float(np.abs(diff - shift).max())
    engrad = workdir / (stem + ".engrad")
    if engrad.is_file():
        e_eh, grad = orca._parse_engrad(engrad, len(symbols))
    else:
        e_eh, grad = first_energy_from_out(text), gradient_from_out(text, len(symbols))
    # the FIRST single point is the frame's: a NumFreq level prints one more per displaced
    # geometry, and the last of those is not the label (review 2026-09-18)
    e_out = first_energy_from_out(text)
    forces = -np.asarray(grad, dtype=float) * orca.EV_PER_HARTREE * orca.BOHR_PER_ANGSTROM
    h = orca.hessian_to_ev_per_angstrom2(parsed["hessian_eh_bohr2"])
    masses = atoms.get_masses()
    m = re.search(r"Program Version\s+(\S+)", text)
    return dict(energy=float(e_out) * orca.EV_PER_HARTREE, energy_engrad=float(e_eh) * orca.EV_PER_HARTREE,
                forces=forces, hessian=np.asarray(h, dtype=float), max_position_dev_A=dev,
                com_shift_A=float(np.linalg.norm(shift)),
                seconds=total_run_seconds(text), memory_mb=max_memory_mb(text),
                hessian_route=orca.hessian_route(text), orca_version=m.group(1) if m else "unknown",
                noise_floor_cm=float(hessian_mod.rigid_block_floor_cm(h, masses, atoms.get_positions())),
                lowest_freq=frames_mod.lowest_projected_cm(h, masses, atoms.get_positions()),
                max_force=float(np.abs(forces).max()), out=str(workdir / (stem + ".out")))


# ====================================================================== one frame
def label_one(molecule, level, generator, basin, k, nprocs=NPROCS, maxcore=MAXCORE_MB,
              scratch=None, timeout_s=None, runner=None, mace_level_name=None, charge=0, mult=1):
    """ORCA on one frame. Returns the parsed label plus `status` ("labelled" when ORCA ran
    now, "reused" when its finished job was on disk, "refused" on a geometry mismatch)
    and the wall seconds of this call. Raises when ORCA does not terminate normally --
    the caller (a Batch task) records that; the `.out` stays for reading.

    `scratch`: a directory to run in (node-local); KEEP files are copied to the
    molecule tree on normal termination and the scratch copy removed. `runner`: a
    callable (inp, out, cwd, timeout_s) -> rc replacing the ORCA binary (tests)."""
    molecule = Path(molecule)
    t0 = time.time()
    atoms = load_frame(molecule, generator, basin, k, mace_level_name)
    workdir = layout.orca_frame_dir(molecule, level, generator, basin, k)
    keywords, blocks, _route = keyword_line(level)
    status = "reused"
    if not finished(workdir):
        status = "labelled"
        rundir = Path(scratch) / molecule.name / frame_tag(generator, basin, k) if scratch else workdir
        rundir.mkdir(parents=True, exist_ok=True)
        workdir.mkdir(parents=True, exist_ok=True)
        inp, out = rundir / (STEM + ".inp"), rundir / (STEM + ".out")
        inp.write_text(orca.input_text(atoms.get_chemical_symbols(), atoms.get_positions(), keywords,
                                       nprocs, maxcore, charge, mult, blocks), encoding="utf-8")
        rc = (runner or _run_orca)(inp, out, rundir, timeout_s)
        text = out.read_text(encoding="utf-8", errors="replace") if out.is_file() else ""
        if scratch:
            for f in sorted(rundir.iterdir()):
                if f.name.startswith(STEM) and any(f.name.endswith(s) for s in KEEP):
                    shutil.copy2(f, workdir / f.name)
            shutil.rmtree(rundir, ignore_errors=True)
        if TERMINAL not in text:
            raise RuntimeError("ORCA did not finish normally for {} of {} (rc {}). Tail:\n{}".format(
                frame_tag(generator, basin, k), molecule.name, rc, "\n".join(text.split("\n")[-25:])))
    lab = parse_label(workdir, atoms)
    if lab["max_position_dev_A"] > POSITION_TOL_A:
        status = "refused"
    lab.update(status=status, generator=generator, basin=int(basin), k=int(k), keywords=keywords,
               wall_seconds=time.time() - t0, workdir=str(workdir))
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
        wd = layout.orca_frame_dir(molecule, level, g, b, k)
        row = dict(GENERATOR=g, BASIN=b, K=k, ENERGY=float("nan"), ENERGY_ABOVE_BASIN=float("nan"),
                   MAX_FORCE=float("nan"), LOWEST_FREQ=float("nan"), NOISE_FLOOR_CM=float("nan"),
                   MAX_POSITION_DEV_A=float("nan"), COM_SHIFT_A=float("nan"), HESSIAN_ROUTE="-", SECONDS=float("nan"),
                   MEMORY_MB=float("nan"), ORCA_VERSION="-", STATUS="unlabelled", REASON="-",
                   OUT=str(wd / (STEM + ".out")))
        if not finished(wd):
            row["REASON"] = "no finished ORCA job at {}".format(wd)
            rows.append(row)
            continue
        try:
            lab = parse_label(wd, a)
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
                    noise_floor_cm=lab["noise_floor_cm"], keywords=keywords)
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
                N_COMPUTED=sum(1 for r in rows if r["STATUS"] == "labelled"),
                N_REUSED=sum(1 for r in rows if r["STATUS"] == "reused"),
                N_REFUSED=sum(1 for r in rows if r["STATUS"] == "refused"),
                N_UNLABELLED=sum(1 for r in rows if r["STATUS"] == "unlabelled"),
                SECONDS_PER_FRAME=float(np.mean(secs)) if secs else float("nan"),
                MAX_MEMORY_MB=float(max(mems)) if mems else float("nan"),
                NOISE_FLOOR_MAX_CM=float(max([r["NOISE_FLOOR_CM"] for r in lab_rows if r["GENERATOR"] == "basin"] or [float("nan")])),
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
              "POSITION_TOL_A", "N_FRAMES", "N_LABELLED", "N_COMPUTED", "N_REUSED", "N_REFUSED", "N_UNLABELLED",
              "SECONDS_PER_FRAME", "MAX_MEMORY_MB", "NOISE_FLOOR_MAX_CM"):
        rep.kv(k, info[k])
    rep.section("per generator")
    rep.table(["generator", "frames", "labelled", "file"],
              [[r["GENERATOR"], r["N_FRAMES"], r["N_LABELLED"], Path(r["FILE"]).name if r["FILE"] else "-"] for r in gen_rows])
    rep.section("per frame (eV; dE kcal/mol above the basin frame; eV/A; cm^-1; A; s; MB)")
    rep.table(["generator", "basin", "k", "E", "dE", "|F|max", "lowest", "rigid", "|dx|max", "com", "route", "s", "MB", "status", "reason"],
              [[r["GENERATOR"], r["BASIN"], r["K"], "%.6f" % r["ENERGY"], "%.2f" % r["ENERGY_ABOVE_BASIN"],
                "%.4f" % r["MAX_FORCE"], "%.1f" % r["LOWEST_FREQ"], "%.2f" % r["NOISE_FLOOR_CM"],
                "%.1e" % r["MAX_POSITION_DEV_A"], "%.4f" % r["COM_SHIFT_A"], r["HESSIAN_ROUTE"], "%.0f" % r["SECONDS"],
                "%.0f" % r["MEMORY_MB"], r["STATUS"], r["REASON"]] for r in rows])
    rep.note("a label is ORCA's energy, gradient and raw Cartesian Hessian at the FRAME'S FIXED GEOMETRY "
             "(single point + EnGrad + Freq/NumFreq; no optimisation), converted once to eV, eV/A, eV/A^2 "
             "and written beside the MACE file with the same positions. 'lowest' at a displaced frame is a "
             "curvature; 'rigid' is the rigid-body block of the unprojected Hessian -- the noise floor at a basin "
             "frame (a few cm^-1 to ~30 for an analytic Hessian), the gradient term at a displaced frame. 'com' is "
             "the centre-of-mass translation ORCA applied in the .hess, removed before the geometry check. A "
             "refused frame's .hess geometry differs in shape from the MACE file: the two levels "
             "of a frame must sit at one geometry or the Dataset compares different points. The full .out of "
             "every job is kept under orca/<level>/frames/.")
    rep.write(path, step=STEP)
