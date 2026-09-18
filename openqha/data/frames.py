"""The Frame set of one molecule at the engine level (ticket 02 of the Hessian-learning
set; CONTEXT.md "Frame", "Frame set").

WHAT A FRAME IS
---------------
One geometry of one molecule, born from one Basin by a named generator, carrying the
engine's energy, forces and raw Cartesian Hessian at that geometry. The Hessian at a
displaced frame is a fixed-geometry matrix with its gradient term included -- it is the
training target of Hessian learning, not a frequency (see the 2026-09-17 dry run).

GENERATORS (rounds 3-4, rulings 2026-09-18)
-------------------------------------------
    basin      the MACE basin itself (branch A: CREST on GFN2 with `refine = "sp"`,
               then MACE tightening to fmax 1e-4 and the MACE analytic Hessian). Its
               Hessian is the basin's stored `hessian.npy`, reused, not recomputed.
    displaced  n draws per basin from the harmonic quantum distribution along the
               basin's modes at the target temperature (`hessian.thermal_displacements`
               with `hessian=` the stored matrix), RMS displacement <= MAX_RMS_A; the
               engine's Hessian is computed at each.
    merged     every input conformer branch A's deduplication merged into a basin
               (`DUPLICATE_MAP` of the branch-A Property; geometry = the tightened
               `mace/confNN/conf.extxyz`), one frame each, no displacement.
    saddle     every input conformer that survived deduplication and was rejected for
               imaginary modes (`SADDLE_CONFORMER_IDS`), one frame each.
SPICE's hot-MD / cooled frames and OpenREACT's hot normal-mode frames were read and
are not built (rulings 2026-09-18): the three published Hessian-learning sets train
near minima, and that is where the entropy error was measured.

SEEDS
-----
Every displaced frame is drawn from its own generator seeded by
`frame_seed(qm9_index, basin, generator, k)` (SHA-256 of that tuple, 8 bytes), so one
frame is reproducible from its own row of the Record without the others.

FILTER
------
A frame whose engine |F|max exceeds FMAX_FILTER_EV_A, or whose bond graph differs from
its basin's, is dropped and counted with its reason. The bond test is asymmetric on
purpose: a basin bond is BROKEN when its length in the frame exceeds BOND_BREAK_MULT x
(r_i + r_j) (covalent radii), a bond is FORMED when a non-bonded pair comes within
BOND_FORM_MULT x (r_i + r_j). A single cutoff on both sides (ASE natural cutoffs at
1.2) flagged 2 of 12 propanal frames whose C-H bond was stretched by 0.2 A at 0.12 A RMS
-- the zero-point stretch, not a reaction (measured 2026-09-18). The MACE E-F-H of a
kept frame is written at generation time (Q8): it is the very quantity the loss
compares to the label, and with it on disk the judge needs no engine.

ENERGY OF A DISPLACED FRAME
---------------------------
The harmonic QUANTUM draw at 298 K puts ~hbar omega / 2 into every stretch (the
zero-point motion), so a displaced frame sits 0.5-1.8 eV (14-42 kcal/mol) above its
basin at only 0.09-0.15 A RMS -- the Wigner-like distribution of the vibrational
ground state, not a 298 K classical thermal energy (SPICE's 500 K MD is classical:
~0.03 eV per mode). Both are stated per frame (`ENERGY_ABOVE_BASIN`); which one the
training should see is round-2 territory, this module reports it.

FILES
-----
`<molecule>/frames/<generator>.<engine level>.extxyz`   one file per generator (ASE
extxyz: `energy`, `forces`; info keys `qm9_index`, `basin`, `generator`, `k`, `seed`,
`level`, `source_conformer`, `rms_displacement_A`, `smiles`, `hessian` = the 3N x 3N
matrix flattened row-major in eV/A^2, `engine_params_sha256`), and the Record
`<molecule>/frames/frames.{out,toml}`.
"""
import hashlib
import time
from pathlib import Path

import numpy as np

from ..potentials import engine
from ..store import basins as basins_mod, branch_a_property, layout, property as prop, report
from ..thermochem import hessian as hessian_mod

EV_TO_KCAL = 23.060547830619026

STEP = "frames"
PROGNAME = "openQHA frames"
GENERATORS = ("basin", "displaced", "merged", "saddle")
#: displaced frames per basin (round 4, Q2)
N_DISPLACED = 4
#: the target temperature of the harmonic quantum draw
TEMPERATURE_K = 298.15
#: RMS displacement ceiling of a displaced frame (hessian.MAX_RMS_DISPLACEMENT_A)
MAX_RMS_A = hessian_mod.MAX_RMS_DISPLACEMENT_A
#: a frame with a larger engine force is not a geometry 298 K visits; dropped
FMAX_FILTER_EV_A = 10.0
#: a basin bond is broken in a frame when its length exceeds this x (r_i + r_j)
BOND_BREAK_MULT = 1.35
#: a bond is formed in a frame when a non-bonded pair comes within this x (r_i + r_j)
BOND_FORM_MULT = 0.95
#: covalent-radius multiplier that defines the BASIN's bond graph (ASE natural cutoffs)
BOND_CUTOFF_MULT = 1.2

SCHEMA = {
    "Calculation_Info": {
        "MOLECULE_DIR": ("String", None, "the molecule directory"),
        "QM9_INDEX": ("String", None, "the molecule"),
        "SMILES": ("String", None, "as declared in the branch-A record"),
        "ENGINE": ("String", None, "the potential that labelled the frames"),
        "LEVEL": ("String", None, "the engine's level name (the file suffix)"),
        "ENGINE_PARAMS_SHA256": ("String", None, "parameter fingerprint of the weights (engine.parameter_fingerprint)"),
        "ENGINE_PIN_STATUS": ("String", None, "matches / differs / unpinned against the registry"),
        "TEMPERATURE": ("Double", "K", "temperature of the harmonic quantum draw"),
        "N_DISPLACED_PER_BASIN": ("Integer", None, "displaced frames drawn per basin"),
        "MAX_RMS_A": ("Double", "A", "RMS displacement ceiling of a displaced frame (over the 3N coordinates)"),
        "FMAX_FILTER_EV_A": ("Double", "eV/A", "a frame with a larger engine |F|max is dropped"),
        "BOND_CUTOFF_MULT": ("Double", None, "covalent-radius multiplier defining the basin's bond graph"),
        "BOND_BREAK_MULT": ("Double", None, "a basin bond longer than this x (r_i + r_j) in a frame is broken"),
        "BOND_FORM_MULT": ("Double", None, "a non-bonded pair closer than this x (r_i + r_j) in a frame is a formed bond"),
        "N_BASINS": ("Integer", None, "basins of the molecule"),
        "N_FRAMES": ("Integer", None, "frames kept, all generators"),
        "N_DROPPED": ("Integer", None, "frames dropped, all generators"),
        "SECONDS": ("Double", "s", "wall time"),
    },
    "Generator": {
        "GENERATOR": ("String", None, "basin / displaced / merged / saddle"),
        "N_FRAMES": ("Integer", None, "frames kept"),
        "N_DROPPED": ("Integer", None, "frames dropped"),
        "FILE": ("String", None, "the extxyz written, or absent when no frame was kept"),
    },
    "Frame": {
        "GENERATOR": ("String", None, "the generator"),
        "BASIN": ("Integer", None, "the basin the frame was born from"),
        "K": ("Integer", None, "index within (generator, basin)"),
        "SEED": ("Integer", None, "the frame's own seed (displaced), 0 otherwise"),
        "SOURCE_CONFORMER": ("Integer", None, "branch-A input frame id (merged / saddle / basin), -1 otherwise"),
        "RMS_DISPLACEMENT_A": ("Double", "A", "RMS displacement from the basin over the 3N Cartesian coordinates (hessian.thermal_displacements' definition; per-atom RMS is sqrt(3) larger)"),
        "ENERGY": ("Double", "eV", "engine energy"),
        "ENERGY_ABOVE_BASIN": ("Double", "kcal/mol", "engine energy above the frame's basin"),
        "MAX_FORCE": ("Double", "eV/A", "engine |F|max"),
        "LOWEST_FREQ": ("Double", "cm^-1", "lowest projected eigenvalue of the engine Hessian at the frame, as a wavenumber (negative: imaginary); at a displaced frame this is a curvature, not a mode"),
        "STATUS": ("String", None, "kept / dropped"),
        "REASON": ("String", None, "why dropped, or -"),
    },
}


# ====================================================================== helpers
def frame_seed(qm9_index, basin, generator, k):
    """A frame's own seed: the first 8 bytes of SHA-256 over 'qm9_index|basin|generator|k'."""
    s = "{}|{}|{}|{}".format(qm9_index, int(basin), generator, int(k)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(s).digest()[:8], "big") % (2 ** 62)


def connectivity(atoms, mult=BOND_CUTOFF_MULT):
    """Bonded pairs of a minimum by covalent-radius cutoffs (the same check branch A's
    census uses to count a changed graph), as a frozenset of (i, j)."""
    from ase.neighborlist import natural_cutoffs, neighbor_list
    i, j = neighbor_list("ij", atoms, natural_cutoffs(atoms, mult=mult), self_interaction=False)
    return frozenset((int(a), int(b)) for a, b in zip(i, j) if a < b)


def bond_change(atoms, basin_bonds, break_mult=BOND_BREAK_MULT, form_mult=BOND_FORM_MULT):
    """What changed in the frame's bond graph against the basin's: 'broken i-j' when a
    basin bond is longer than break_mult x (r_i + r_j), 'formed i-j' when a non-bonded
    pair is closer than form_mult x (r_i + r_j); None when nothing changed."""
    from ase.data import covalent_radii
    r = covalent_radii[atoms.numbers]
    d = atoms.get_all_distances()
    n = len(atoms)
    for i, j in sorted(basin_bonds):
        if d[i, j] > break_mult * (r[i] + r[j]):
            return "broken {}{}-{}{} ({:.2f} A)".format(atoms[i].symbol, i, atoms[j].symbol, j, d[i, j])
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in basin_bonds and d[i, j] < form_mult * (r[i] + r[j]):
                return "formed {}{}-{}{} ({:.2f} A)".format(atoms[i].symbol, i, atoms[j].symbol, j, d[i, j])
    return None


def engine_efh(atoms, calc):
    """Energy (eV), forces (eV/A), raw Cartesian Hessian (3N x 3N, eV/A^2) of the engine
    at `atoms`; the Hessian by the engine's analytic route (`hessian.hessian`)."""
    a = atoms.copy()
    a.calc = calc
    e = float(a.get_potential_energy())
    f = np.asarray(a.get_forces(), dtype=float)
    h, _asym = hessian_mod.hessian(a, calc, mode="analytic")
    return e, f, np.asarray(h, dtype=float)


def lowest_projected_cm(h, masses, positions):
    pr = hessian_mod.project_and_diagonalise(h, masses, positions)
    fr = pr["frequencies_cm_inv"]
    return float(fr[0]) if len(fr) else float("nan")


def _write_frames(path, frames):
    """Write kept frames of one generator as extxyz (ASE writer; energy and forces
    through a SinglePointCalculator so they land in the standard keys)."""
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io import write
    out = []
    for fr in frames:
        a = fr["atoms"].copy()
        a.calc = SinglePointCalculator(a, energy=fr["energy"], forces=fr["forces"])
        a.info.update(fr["info"])
        a.info["hessian"] = np.asarray(fr["hessian"], dtype=float).reshape(-1)
        out.append(a)
    path.parent.mkdir(parents=True, exist_ok=True)
    write(str(path), out, format="extxyz")


def read_frames(path):
    """Read one Frame-set file back: a list of ase.Atoms with `hessian` reshaped to
    (3N, 3N) in `info` and energy / forces on a SinglePointCalculator."""
    from ase.io import read
    out = []
    for a in read(str(path), index=":", format="extxyz"):
        n = 3 * len(a)
        a.info["hessian"] = np.asarray(a.info["hessian"], dtype=float).reshape(n, n)
        out.append(a)
    return out


# ====================================================================== the Calculation
def generate(molecule, n_displaced=N_DISPLACED, temperature_K=TEMPERATURE_K, max_rms_A=MAX_RMS_A,
             engine_name=None, fmax_filter=FMAX_FILTER_EV_A, qm9_index=None, calc=None):
    """Build the Frame set of one molecule directory at the engine level: one extxyz per
    generator under `frames/` and the Record `frames/frames.{out,toml}`. Returns a dict
    with the rows and paths. `calc=` injects a calculator (tests); otherwise the
    registered engine is loaded."""
    from ase.io import read
    t0 = time.time()
    molecule = Path(molecule)
    doc = prop.load(layout.records_dir(molecule) / branch_a_property.FILE)
    info_a = doc.get("Calculation_Info") or {}
    census = doc.get("Census") or {}
    qid = qm9_index if qm9_index is not None else info_a.get("QM9_INDEX")
    smiles = info_a.get("SMILES")
    if calc is None:
        calc, ename, prov = engine.calculator(name=engine_name)
    else:
        ename = engine_name or engine.engine_name()
        try:
            prov = engine.provenance(ename)
        except FileNotFoundError:                      # a test calculator, no weights on this machine
            prov = dict(params_sha256=None, params_pin_status="weights absent (calculator injected)")
    level = engine.level_name(ename)
    files = {int(p.parent.name[len("basin"):]): p for p in basins_mod.basin_files(molecule)}
    basin_cids = [int(x) for x in (census.get("BASIN_CONFORMER_IDS") or [])]
    dup_map = [int(x) for x in (census.get("DUPLICATE_MAP") or [])]
    saddle_cids = [int(x) for x in (census.get("SADDLE_CONFORMER_IDS") or [])]

    rows, kept = [], {g: [] for g in GENERATORS}
    graphs, masses_of, e_basin = {}, {}, {}

    def consider(generator, basin, k, atoms, seed, source, rms, hessian=None, e=None, f=None):
        if hessian is None:
            e, f, hessian = engine_efh(atoms, calc)
        fmax = float(np.abs(f).max())
        lowest = lowest_projected_cm(hessian, masses_of[basin], atoms.get_positions())
        row = dict(GENERATOR=generator, BASIN=int(basin), K=int(k), SEED=int(seed), SOURCE_CONFORMER=int(source),
                   RMS_DISPLACEMENT_A=float(rms), ENERGY=float(e),
                   ENERGY_ABOVE_BASIN=(float(e) - e_basin[basin]) * EV_TO_KCAL,
                   MAX_FORCE=fmax, LOWEST_FREQ=lowest, STATUS="kept", REASON="-")
        change = bond_change(atoms, graphs[basin])
        if fmax > fmax_filter:
            row.update(STATUS="dropped", REASON="engine |F|max {:.2f} eV/A > {:.1f}".format(fmax, fmax_filter))
        elif change:
            row.update(STATUS="dropped", REASON="bond graph: " + change)
        rows.append(row)
        if row["STATUS"] == "kept":
            kept[generator].append(dict(
                atoms=atoms, energy=float(e), forces=f, hessian=hessian,
                info=dict(qm9_index=str(qid), basin=int(basin), generator=generator, k=int(k), seed=int(seed),
                          level=level, source_conformer=int(source), rms_displacement_A=float(rms),
                          smiles=str(smiles), engine_params_sha256=prov.get("params_sha256") or "")))

    # ---- basin and displaced -------------------------------------------------------
    for b in sorted(files):
        atoms = read(str(files[b]), format="extxyz")
        h_path = files[b].parent / "hessian.npy"
        if not h_path.is_file():
            raise FileNotFoundError("basin {} has no hessian.npy: {}".format(b, h_path))
        h_b = np.load(h_path)
        masses_of[b] = atoms.get_masses()
        graphs[b] = connectivity(atoms)
        # the basin frame: the stored Hessian, the engine's energy and forces at the geometry
        a0 = atoms.copy(); a0.calc = calc
        e_b = float(a0.get_potential_energy()); f_b = np.asarray(a0.get_forces(), dtype=float)
        e_basin[b] = e_b
        src = basin_cids[b] if b < len(basin_cids) else -1
        consider("basin", b, 0, atoms.copy(), 0, src, 0.0, hessian=h_b, e=e_b, f=f_b)
        seeds = [frame_seed(qid, b, "displaced", k) for k in range(int(n_displaced))]
        if n_displaced > 0:
            disp, drec = hessian_mod.thermal_displacements(
                atoms, None, temperature_K=temperature_K, n_samples=int(n_displaced),
                max_rms_displacement_A=max_rms_A, hessian=h_b, seeds=seeds)
            for k, (a, rms) in enumerate(zip(disp, drec["rms_displacement_A"])):
                consider("displaced", b, k, a, seeds[k], -1, rms)

    # ---- merged and saddle: tightened input conformers from branch A's engine files --
    def basin_of(cid):
        home = dup_map[cid] if cid < len(dup_map) else cid
        return basin_cids.index(home) if home in basin_cids else None

    merged = [j for j, home in enumerate(dup_map) if home != j]
    for gen, cids in (("merged", merged), ("saddle", saddle_cids)):
        for k, cid in enumerate(cids):
            conf = layout.mace_conformer_dir(molecule, cid) / "conf.extxyz"
            if not conf.is_file():
                rows.append(dict(GENERATOR=gen, BASIN=-1, K=k, SEED=0, SOURCE_CONFORMER=cid, RMS_DISPLACEMENT_A=float("nan"),
                                 ENERGY=float("nan"), ENERGY_ABOVE_BASIN=float("nan"), MAX_FORCE=float("nan"), LOWEST_FREQ=float("nan"),
                                 STATUS="dropped", REASON="no conf.extxyz for input frame {}".format(cid)))
                continue
            atoms = read(str(conf), format="extxyz")
            b = basin_of(cid)
            if b is None:
                # a saddle has no home basin: measure against the nearest basin by energy order 0
                b = min(files) if files else 0
            ref = read(str(files[b]), format="extxyz")
            rms = float(np.sqrt(((atoms.get_positions() - ref.get_positions()) ** 2).mean()))   # over coordinates, as thermal_displacements
            consider(gen, b, k, atoms, 0, cid, rms)

    # ---- files and Record ----------------------------------------------------------
    gen_rows = []
    for g in GENERATORS:
        n_k = len(kept[g]); n_d = sum(1 for r in rows if r["GENERATOR"] == g and r["STATUS"] == "dropped")
        path = layout.frames_file(molecule, g, level)
        if n_k:
            _write_frames(path, kept[g])
        gen_rows.append(dict(GENERATOR=g, N_FRAMES=n_k, N_DROPPED=n_d, FILE=str(path) if n_k else None))
    info = dict(MOLECULE_DIR=str(molecule), QM9_INDEX=str(qid), SMILES=smiles, ENGINE=ename, LEVEL=level,
                ENGINE_PARAMS_SHA256=prov.get("params_sha256"), ENGINE_PIN_STATUS=prov.get("params_pin_status"),
                TEMPERATURE=float(temperature_K), N_DISPLACED_PER_BASIN=int(n_displaced), MAX_RMS_A=float(max_rms_A),
                FMAX_FILTER_EV_A=float(fmax_filter), BOND_CUTOFF_MULT=BOND_CUTOFF_MULT,
                BOND_BREAK_MULT=BOND_BREAK_MULT, BOND_FORM_MULT=BOND_FORM_MULT, N_BASINS=len(files),
                N_FRAMES=sum(r["N_FRAMES"] for r in gen_rows), N_DROPPED=sum(r["N_DROPPED"] for r in gen_rows),
                SECONDS=time.time() - t0)
    fdir = layout.frames_dir(molecule)
    fdir.mkdir(parents=True, exist_ok=True)
    missing = prop.write(fdir / (STEP + ".toml"), {"Calculation_Info": info, "Generator": gen_rows, "Frame": rows},
                         SCHEMA, prop.NORMAL_TERMINATION, PROGNAME)
    if missing:
        raise RuntimeError("frames.toml keys outside the schema: {}".format(missing))
    _write_report(fdir / (STEP + ".out"), info, gen_rows, rows)
    return dict(info=info, generators=gen_rows, frames=rows, record=fdir / (STEP + ".toml"))


def _write_report(path, info, gen_rows, rows):
    rep = report.Report("openQHA frames", "the Frame set of {} at {}".format(info["QM9_INDEX"], info["LEVEL"]))
    rep.section("conventions")
    for k in ("ENGINE", "LEVEL", "ENGINE_PARAMS_SHA256", "ENGINE_PIN_STATUS", "TEMPERATURE", "N_DISPLACED_PER_BASIN",
              "MAX_RMS_A", "FMAX_FILTER_EV_A", "BOND_CUTOFF_MULT", "BOND_BREAK_MULT", "BOND_FORM_MULT",
              "N_BASINS", "N_FRAMES", "N_DROPPED"):
        rep.kv(k, info[k])
    rep.section("per generator")
    rep.table(["generator", "kept", "dropped", "file"],
              [[r["GENERATOR"], r["N_FRAMES"], r["N_DROPPED"], Path(r["FILE"]).name if r["FILE"] else "-"] for r in gen_rows])
    rep.section("per frame (eV; dE in kcal/mol above the basin; eV/A; A; cm^-1)")
    rep.table(["generator", "basin", "k", "source", "rms", "E", "dE", "|F|max", "lowest", "status", "reason"],
              [[r["GENERATOR"], r["BASIN"], r["K"], r["SOURCE_CONFORMER"], "%.4f" % r["RMS_DISPLACEMENT_A"],
                "%.6f" % r["ENERGY"], "%.2f" % r["ENERGY_ABOVE_BASIN"], "%.4f" % r["MAX_FORCE"],
                "%.1f" % r["LOWEST_FREQ"], r["STATUS"], r["REASON"]]
               for r in rows])
    rep.note("a frame's Hessian is the engine's raw Cartesian matrix at that fixed geometry, gradient term "
             "included -- the Hessian-learning target, not a frequency; 'lowest' at a displaced frame is a "
             "curvature. The basin frame reuses the basin's stored hessian.npy. dE of a displaced frame is "
             "the harmonic QUANTUM draw's energy (zero-point motion in every stretch), tens of kcal/mol at "
             "0.1 A RMS -- not a classical 298 K energy. Hot-MD, cooled and hot normal-mode frames (SPICE, "
             "OpenREACT) were considered and not built (rulings 2026-09-18).")
    rep.write(path, step=STEP)
