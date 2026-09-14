"""The census leaves MACE's relax and Hessian in `mace/` as ASE's own files (ticket 05).

INTEGRATION. Runs `census_from_frames` with the real MACE calculator on a prepared
two-frame ensemble of a shipped molecule (no CREST binary involved); tens of seconds.
Skips with a message when MACE is absent.

The decision (ADR 0001, user 2026-09-14):

    mace/confNN/   opt.traj  opt.log  conf.extxyz     every tightened conformer
    mace/basinNN/  basin.extxyz  hessian.npy          every surviving basin

`conf.extxyz` carries energy and forces as ASE writes them; `basin.extxyz` names the
conformer it came from (index and CREST's comment line); `hessian.npy` is the raw
analytic Hessian, 3N x 3N, eV/A^2, float64, symmetric, neither mass-weighted nor
projected. Nothing else is written under `mace/`.

The ensemble: the deposited geometry, and the same geometry with every atom displaced
by 0.05 A in a fixed pattern. Both tighten to the same minimum, so the deduplication
keeps one basin -- which is what makes "conformers are kept even when they lose the
deduplication" checkable: two conformer folders, one basin folder.
"""
import sys
import tempfile
from pathlib import Path


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))
from openqha.store import layout   # noqa: E402

QID = "dsgdb9nsd_000018"
FAIL = []


def check(label, ok, detail=""):
    print("  {:62s} {}".format(label, "ok" if ok else "FAIL " + str(detail)[:300]))
    if not ok:
        FAIL.append(label)


def main():
    try:
        import numpy as np
        from ase.io import read
        from openqha import config, engine
        from openqha.conformer_search import crest_census
        calc, _name, _prov = engine.calculator(device="cpu")
    except Exception as exc:                          # noqa: BLE001
        print("SKIP: {}".format(exc))
        return 0

    cfg = config.load()
    ref = config.read_qm9_xyz(config.qm9_xyz(QID, cfg))
    smiles = config.qm9_smiles(QID, cfg)
    shifted = ref.copy()
    rng = np.random.RandomState(7)
    shifted.positions = shifted.positions + 0.05 * rng.choice([-1.0, 1.0], size=shifted.positions.shape)
    frames = [ref, shifted]
    comments = ["deposited geometry", "deposited geometry, every atom moved 0.05 A"]

    tmp = Path(tempfile.mkdtemp(prefix="openqha_t05_"))
    try:
        mol = layout.molecule_dir(tmp, "t05", QID)
        rec, basins, _m = crest_census.census_from_frames(
            smiles, frames, calc, name=QID, fmax=1e-3, threshold_A=0.30,
            temperature_K=298.15, do_hessian=True, reject_imaginary=True,
            comments=comments, hessian_mode="analytic", molecule_dir=mol)

        print("A. one folder per tightened conformer, three files each")
        mace = mol / "mace"
        conf_dirs = sorted(p for p in mace.iterdir() if p.name.startswith("conf")) if mace.is_dir() else []
        check("two conformer folders (conf00, conf01)", [p.name for p in conf_dirs] == ["conf00", "conf01"],
              [p.name for p in conf_dirs])
        for d in conf_dirs:
            names = sorted(p.name for p in d.iterdir())
            check("{} holds exactly opt.traj opt.log conf.extxyz".format(d.name),
                  names == sorted(layout.MACE_CONFORMER_FILES), names)
        if conf_dirs:
            a = read(conf_dirs[0] / "conf.extxyz")
            check("conf.extxyz carries energy and forces",
                  a.calc is not None and "energy" in a.calc.results and "forces" in a.calc.results,
                  getattr(a.calc, "results", None))
            from ase.io import Trajectory
            n = len(Trajectory(str(conf_dirs[0] / "opt.traj")))
            check("opt.traj holds the optimisation steps ({} frames)".format(n), n >= 1, n)

        print("B. one folder per surviving basin, two files each")
        basin_dirs = sorted(p for p in mace.iterdir() if p.name.startswith("basin")) if mace.is_dir() else []
        check("one basin (the two conformers deduplicate)", [p.name for p in basin_dirs] == ["basin00"],
              [p.name for p in basin_dirs])
        check("record agrees: n_basins == 1", rec.get("n_basins") == 1 and len(basins) == 1, rec.get("n_basins"))
        if basin_dirs:
            d = basin_dirs[0]
            names = sorted(p.name for p in d.iterdir())
            check("basin00 holds exactly basin.extxyz hessian.npy", names == sorted(layout.MACE_BASIN_FILES), names)
            b = read(d / "basin.extxyz")
            check("basin.extxyz names its conformer index", "conformer" in b.info and int(b.info["conformer"]) in (0, 1),
                  b.info)
            check("basin.extxyz carries CREST's comment for that conformer",
                  str(b.info.get("crest_comment", "")).startswith("deposited geometry"), b.info)
            h = np.load(d / "hessian.npy")
            n3 = 3 * len(ref)
            check("hessian.npy is 3N x 3N float64", h.shape == (n3, n3) and h.dtype == np.float64, (h.shape, h.dtype))
            check("hessian.npy is symmetric (max asym {:.1e})".format(float(np.abs(h - h.T).max())),
                  float(np.abs(h - h.T).max()) < 1e-6)
            # raw, not mass-weighted: an H-H diagonal block and a C-C block differ by
            # the force constant alone, so a mass-weighted matrix would have divided the
            # hydrogen blocks by 1 amu and the carbon blocks by 12 -- an eV/A^2 Hessian of
            # this molecule has its LARGEST diagonal entries on the heavy atoms.
            diag = np.diag(h).reshape(-1, 3).sum(1)
            heavy = [i for i, z in enumerate(ref.numbers) if z > 1]
            hyd = [i for i, z in enumerate(ref.numbers) if z == 1]
            check("raw (heavy-atom diagonal blocks exceed hydrogen ones)",
                  diag[heavy].mean() > diag[hyd].mean(), (diag[heavy].mean(), diag[hyd].mean()))

        print("C. nothing else under mace/")
        extra = sorted(p.name for p in mace.iterdir() if not (p.name.startswith("conf") or p.name.startswith("basin")))
        check("no other entries", extra == [], extra)
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n{}".format("PASS" if not FAIL else "FAIL: " + "; ".join(FAIL)))
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
