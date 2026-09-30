"""The numbers this environment produces, so that changing it is a measurement.

CALIBRATION. It produces no scientific number; it produces the fingerprint against which
an environment change is judged.

Why
---
The OpenMM stack lives in the core environment, accepting the
downgrades the solver requires:

    pytorch   2.13.0 -> 2.12.1        numpy   2.4.6 -> 1.26.4
    libtorch  2.13.0 -> 2.12.1        pandas  3.0.5 -> 2.3.3

A different pytorch is a different set of last bits in every MACE force call, and this
repository's whole discipline is that such a claim is measured rather than asserted. So:
run this before the change, run it after, and diff. Whatever moves, moves on the record.

What it fingerprints, and why each
----------------------------------
  * ENERGY AND FORCES at a fixed geometry -- the potential itself. Everything downstream
    is a function of these.
  * HESSIAN FREQUENCIES -- second derivatives amplify what the forces only hint at, and
    they are what branch C's route is built on.
  * T*S from a FIXED set of frames loaded from disk -- the branch B estimator with no
    dynamics in it, so a difference here is arithmetic and not sampling.
  * The quasi-harmonic frequencies behind that T*S, because a T*S that agrees while its
    spectrum moves would be two errors cancelling.

Deliberately NOT fingerprinted: anything involving fresh dynamics. Molecular dynamics is
chaotic and two runs on two builds diverge exponentially by construction; comparing them
would measure the Lyapunov exponent, not the environment. Fixed frames are the point.

    python scripts/calibration/s0_B_stack_fingerprint.py --out analysis/qha/fingerprint_before
    # ... change the environment ...
    python scripts/calibration/s0_B_stack_fingerprint.py --out analysis/qha/fingerprint_after
    python scripts/calibration/s0_B_stack_fingerprint.py --compare \\
        analysis/qha/fingerprint_before.json analysis/qha/fingerprint_after.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))

SPECIES = "dsgdb9nsd_000018"
#: Frames from a real production run, loaded from disk. Fixed input, so any difference is
#: the software stack and nothing else.
FRAMES = (Path.home() / "runs" / "openQHA" / "qha" / "prod01" / SPECIES
          / "basin00" / "seed00")


def fingerprint(species=SPECIES, frames_dir=FRAMES, n_frames=2000):
    from ase.io import read
    from openqha import config, engine, hessian, qha

    import numpy
    import torch
    record = dict(
        stack=dict(numpy=numpy.__version__, torch=torch.__version__,
                   python=sys.version.split()[0]),
    )
    try:
        import openmm
        record["stack"]["openmm"] = openmm.version.version
    except Exception:                                          # noqa: BLE001
        record["stack"]["openmm"] = None
    try:
        import MDAnalysis
        record["stack"]["MDAnalysis"] = MDAnalysis.__version__
    except Exception:                                          # noqa: BLE001
        record["stack"]["MDAnalysis"] = None

    cfg = config.load()
    atoms = read(str(config.qm9_xyz(species, cfg)))
    calc, name, prov = engine.calculator(device="cpu")
    atoms.calc = calc
    record["engine"] = name
    record["mace_torch_version"] = prov.get("mace_torch_version")
    record["mace_module_path"] = prov.get("mace_module_path")
    record["weights_path"] = prov.get("weights_path")

    # ---- the potential, at the QM9 geometry exactly as it is on disk -------------------
    record["energy_eV"] = float(atoms.get_potential_energy())
    forces = atoms.get_forces()
    record["force_rms_eV_per_A"] = float(np.sqrt((forces ** 2).mean()))
    record["force_max_eV_per_A"] = float(np.abs(forces).max())
    record["forces_checksum"] = float(np.abs(forces).sum())

    # ---- second derivatives ------------------------------------------------------------
    H, asym = hessian.hessian(atoms, calc, mode="analytic")
    rec = hessian.project_and_diagonalise(H, atoms.get_masses(), atoms.get_positions())
    freqs = sorted(float(x) for x in rec["frequencies_cm_inv"])
    record["hessian_asymmetry_eV_A2"] = float(asym)
    record["hessian_frequencies_cm_inv"] = freqs
    record["hessian_lowest_cm_inv"] = freqs[0] if freqs else None

    # ---- the branch B estimator on FIXED frames ---------------------------------------
    fpath = Path(frames_dir) / "frames.npy"
    mpath = Path(frames_dir) / "meta.json"
    if fpath.is_file() and mpath.is_file():
        meta = json.loads(mpath.read_text(encoding="utf-8"))
        frames = np.load(fpath)[:n_frames]
        out = qha.analyse(frames, np.asarray(meta["masses_amu"], dtype=float),
                          float(meta["temperature_K"]))
        record["qha"] = dict(
            source=str(frames_dir), n_frames=int(len(frames)),
            TS_QH_kcal=out["entropy"]["TS_QH_kcal"],
            TS_Schlitter_kcal=out["entropy"]["TS_Schlitter_kcal"],
            lowest_frequency_cm_inv=out["entropy"]["lowest_frequency_cm_inv"],
            frequencies_cm_inv=sorted(
                float(x) for x in out["entropy"]["frequencies_cm_inv"]),
            n_nonzero=out["spectrum"]["n_nonzero_eigenvalues"])
    else:
        record["qha"] = dict(skipped="no frames at {}".format(frames_dir))
    return record


def compare(a, b):
    """Diff two fingerprints, loudest first."""
    out = []

    def add(key, x, y, unit=""):
        if x is None or y is None:
            out.append((key, x, y, None, unit))
            return
        out.append((key, x, y, float(y) - float(x), unit))

    out.append(("stack", json.dumps(a["stack"]), json.dumps(b["stack"]), None, ""))
    add("energy_eV", a["energy_eV"], b["energy_eV"], "eV")
    add("force_rms_eV_per_A", a["force_rms_eV_per_A"], b["force_rms_eV_per_A"], "eV/A")
    add("force_max_eV_per_A", a["force_max_eV_per_A"], b["force_max_eV_per_A"], "eV/A")
    add("hessian_lowest_cm_inv", a["hessian_lowest_cm_inv"], b["hessian_lowest_cm_inv"],
        "cm-1")
    fa = np.asarray(a["hessian_frequencies_cm_inv"], dtype=float)
    fb = np.asarray(b["hessian_frequencies_cm_inv"], dtype=float)
    if len(fa) == len(fb):
        out.append(("hessian_frequencies max |diff|", "", "",
                    float(np.abs(fb - fa).max()), "cm-1"))
    if "TS_QH_kcal" in (a.get("qha") or {}) and "TS_QH_kcal" in (b.get("qha") or {}):
        add("qha_TS_QH_kcal", a["qha"]["TS_QH_kcal"], b["qha"]["TS_QH_kcal"], "kcal/mol")
        qa = np.asarray(a["qha"]["frequencies_cm_inv"], dtype=float)
        qb = np.asarray(b["qha"]["frequencies_cm_inv"], dtype=float)
        if len(qa) == len(qb):
            out.append(("qha_frequencies max |diff|", "", "",
                        float(np.abs(qb - qa).max()), "cm-1"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species", default=SPECIES)
    ap.add_argument("--frames", default=str(FRAMES))
    ap.add_argument("--n-frames", type=int, default=2000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", nargs=2, default=None,
                    help="two fingerprint .json files, before and after")
    args = ap.parse_args()

    if args.compare:
        a = json.loads(Path(args.compare[0]).read_text(encoding="utf-8"))
        b = json.loads(Path(args.compare[1]).read_text(encoding="utf-8"))
        print("=" * 92)
        print("Stack fingerprint -- what the environment change moved")
        print("=" * 92)
        print("before: {}".format(json.dumps(a["stack"])))
        print("after : {}".format(json.dumps(b["stack"])))
        print()
        print("{:<34} {:>22} {:>22} {:>14}".format("quantity", "before", "after", "diff"))
        for key, x, y, d, unit in compare(a, b):
            if key == "stack":
                continue
            fx = "{:.9g}".format(float(x)) if x not in ("", None) else ""
            fy = "{:.9g}".format(float(y)) if y not in ("", None) else ""
            fd = "" if d is None else "{:+.3e}".format(d)
            print("{:<34} {:>22} {:>22} {:>14} {}".format(key, fx, fy, fd, unit))
        print()
        print("A difference here is the software stack: the geometry, the frames and the")
        print("weights were identical. Nothing that involves fresh dynamics is compared,")
        print("because chaotic trajectories diverge by construction and would measure the")
        print("Lyapunov exponent rather than the environment.")
        return 0

    rec = fingerprint(args.species, Path(args.frames), args.n_frames)
    print("=" * 92)
    print("Stack fingerprint")
    print("=" * 92)
    for k, v in rec["stack"].items():
        print("  {:<14} {}".format(k, v))
    print("  {:<14} {}".format("engine", rec["engine"]))
    print("  {:<14} {}".format("mace", rec["mace_torch_version"]))
    print()
    print("  energy                {:.9f} eV".format(rec["energy_eV"]))
    print("  force rms             {:.9f} eV/A".format(rec["force_rms_eV_per_A"]))
    print("  hessian lowest        {:.6f} cm-1".format(rec["hessian_lowest_cm_inv"]))
    if "TS_QH_kcal" in rec["qha"]:
        print("  T*S on {} fixed frames  {:.9f} kcal/mol".format(
            rec["qha"]["n_frames"], rec["qha"]["TS_QH_kcal"]))
    else:
        print("  qha: {}".format(rec["qha"]["skipped"]))

    out = Path(args.out or (ROOT / "analysis" / "qha" / "stack_fingerprint"))
    out.parent.mkdir(parents=True, exist_ok=True)
    path = out.with_suffix(".json")
    path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print("\nwritten: {}".format(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
