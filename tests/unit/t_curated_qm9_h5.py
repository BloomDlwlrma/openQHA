"""The curated QM9 archive as one HDF5 (one group per molecule, parsed form
only), and the on-demand rendering of a molecule back into a QM9 file.

A tiny directory in the archive's three naming patterns (the seven shipped geometries
as originals, two of them ALSO as `_new` repairs so the repair must win) is packed with
`s0_pack_curated_qm9.pack`; then, with a configuration that names no directory but the
archive, `curated_qm9.find()` extracts a molecule into `_h5cache/` with the archive's
file name and the same bytes, the pattern is the repaired one where a repair exists,
`config.qm9_xyz()` reaches it for a molecule the repository does not ship,
`read_qm9_xyz` / `qm9_smiles_from_xyz` read the extracted file as they read the
original, a second `find()` writes nothing, `available()` and `census()` see the
archive, and `verify()` reports no mismatch.
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "tooling"))
from openqha import config                                   # noqa: E402
from openqha.data import curated_qm9                         # noqa: E402
import s0_pack_curated_qm9 as pk                             # noqa: E402

FAIL = []
GEOMS = ROOT / "data" / "reference-geometries"


def check(label, ok, detail=""):
    print("  {:80s} {}".format(label, "ok" if ok else "FAIL " + str(detail)[:300]))
    if not ok:
        FAIL.append(label)


def qm9_form(path, smiles, with_charge=True):
    """The shipped geometry in QM9's own layout: property line, atoms, frequencies, SMILES, InChI."""
    lines = path.read_text(encoding="utf-8").splitlines()
    n = int(lines[0].split()[0])
    # the shipped geometries have no charge column; give the originals one (0.1 per atom) so the
    # Mulliken column is exercised, and leave the repairs without one, as the archive has them
    atoms = [row + ("\t0.1" if with_charge else "") for row in lines[2:2 + n]]
    body = ["{}".format(n), "gdb 1\t0.0"] + atoms + ["100.0\t200.0\t300.0", "{}\t{}".format(smiles, smiles), "InChI=1S/x\tInChI=1S/x"]
    return "\n".join(body) + "\n"


def main():
    with tempfile.TemporaryDirectory(prefix="curated_h5_") as tmp:
        tmp = Path(tmp)
        src = tmp / "archive"
        src.mkdir()
        smiles = {"000018": "CC(=O)C", "000019": "CC(=O)N", "000035": "CCC=O", "000036": "CNC=O",
                  "000044": "C[C@H]1CO1", "000046": "OC1CC1", "000048": "C1COC1"}
        for k, s in smiles.items():
            (src / "dsgdb9nsd_{}.xyz".format(k)).write_text(qm9_form(GEOMS / "dsgdb9nsd_{}.xyz".format(k), s), encoding="utf-8")
        # two repairs, one per repaired pattern, with a marker line so the winner is visible
        (src / "dsgdb9nsd_000044_new.xyz").write_text(qm9_form(GEOMS / "dsgdb9nsd_000044.xyz", "C[C@H]1CO1", False) + "# repaired\n", encoding="utf-8")
        (src / "dsgdb9_000046_new.xyz").write_text(qm9_form(GEOMS / "dsgdb9nsd_000046.xyz", "OC1CC1", False) + "# repaired\n", encoding="utf-8")
        os.environ["S0_CURATED_QM9"] = str(src)
        curated_qm9._INDEX_CACHE.clear()
        idx = curated_qm9._index()
        out = tmp / "curated_qm9.h5"
        attrs = pk.pack(idx, out, source=str(src))
        bad = pk.verify(idx, out, n=7)
        check("pack: 7 molecules, 5 originals + 1 + 1 repaired (the repair wins over the original it shadows), no verify mismatch, sha256 attr",
              attrs["n_molecules"] == 7 and attrs["n_original"] == 5 and attrs["n_repaired_qm9_tag"] == 1
              and attrs["n_repaired_short_tag"] == 1 and not bad and len(attrs["sha256"]) == 64, (attrs, bad))
        del os.environ["S0_CURATED_QM9"]
        curated_qm9._INDEX_CACHE.clear()
        curated_qm9._ARCHIVE_CACHE.clear()

        # --- the directory is gone; the archive answers ---------------------------------
        cfg = config.load()
        cfg["data"]["curated_qm9_dir"] = "no_such_directory_" + os.urandom(3).hex()
        cfg["data"]["curated_qm9_h5"] = str(out)
        cfg["data"]["vendored_reference_geometries"] = "data/no_such_vendored"      # so qm9_xyz falls through
        cfg["data"]["qm9_xyz_dir"] = "no_such_xyz"
        check("without the directory: root() None, archive() the h5, available() true, census() from the attrs (form hdf5)",
              curated_qm9.root(cfg) is None and curated_qm9.archive(cfg) == out and curated_qm9.available(cfg)
              and curated_qm9.census(cfg)["form"] == "hdf5" and curated_qm9.census(cfg)["counts"]["original"] == 5,
              (curated_qm9.root(cfg), curated_qm9.archive(cfg), curated_qm9.census(cfg)))
        p46, kind46 = curated_qm9.find("dsgdb9nsd_000046", cfg)
        p18, kind18 = curated_qm9.find(18, cfg)
        cache = out.parent / curated_qm9.CACHE_DIRNAME
        r46 = config.read_qm9_xyz(p46); s46 = config.qm9_smiles_from_xyz(p46)
        r18 = config.read_qm9_xyz(p18); o18 = config.read_qm9_xyz(src / "dsgdb9nsd_000018.xyz")
        lines18 = p18.read_text(encoding="utf-8").splitlines()
        check("find() renders into _h5cache/ under the archive's file name: the repaired short-tag file wins for 046, the original for 018; read_qm9_xyz / qm9_smiles_from_xyz read the rendered file as the source (positions exact, SMILES, the charge column only where the source had one)",
              p46 == cache / "dsgdb9_000046_new.xyz" and kind46 == "repaired_short_tag"
              and p18 == cache / "dsgdb9nsd_000018.xyz" and kind18 == "original"
              and np.abs(r18.positions - o18.positions).max() == 0.0 and r18.get_chemical_symbols() == o18.get_chemical_symbols()
              and s46 == ("OC1CC1", "OC1CC1") and len(r46) == 10
              and lines18[1] == "gdb 18" and len(lines18[2].split()) == 5 and len(p46.read_text(encoding="utf-8").splitlines()[2].split()) == 4
              and len(lines18) == 10 + 5, (p46, kind46, p18, kind18, lines18[:3]))
        m0 = p46.stat().st_mtime_ns
        p46b, _k = curated_qm9.find("dsgdb9nsd_000046", cfg)
        missing = curated_qm9.find("dsgdb9nsd_000099", cfg)
        check("a second find() returns the cached file without rewriting it; an absent index gives (None, None)",
              p46b == p46 and p46.stat().st_mtime_ns == m0 and missing == (None, None), (p46b, missing))
        xyz = config.qm9_xyz("dsgdb9nsd_000035", cfg)
        atoms = config.read_qm9_xyz(xyz)
        gdb, relaxed = config.qm9_smiles_from_xyz(xyz)
        check("config.qm9_xyz() reaches the archive for a molecule the repository does not ship here; read_qm9_xyz gives 10 atoms, the SMILES line reads CCC=O",
              xyz == cache / "dsgdb9nsd_000035.xyz" and len(atoms) == 10 and (gdb, relaxed) == ("CCC=O", "CCC=O"),
              (xyz, len(atoms), gdb, relaxed))
        arc = curated_qm9.open_archive(cfg)
        a35, a44 = arc.atoms(35), arc.atoms(44)
        ref35 = config.read_qm9_xyz(GEOMS / "dsgdb9nsd_000035.xyz")
        check("Archive: 7 groups named dsgdb9nsd_%06d whatever the file was called, __contains__, no text dataset, text() is the rendered file (6 + N lines), filename() is the source name, repaired() per group",
              len(arc) == 7 and 44 in arc and 99 not in arc and "text" not in arc.group(44)
              and len(arc.text(44).splitlines()) == 10 + 5 and arc.text(44).splitlines()[1] == "gdb 44"
              and arc.filename(44) == "dsgdb9nsd_000044_new.xyz" and arc.filename(19) == "dsgdb9nsd_000019.xyz"
              and arc.repaired(44) and not arc.repaired(19) and "dsgdb9nsd_000044" in arc.h5 and "dsgdb9_000044_new" not in arc.h5)
        check("the parsed form: atoms(35) = the shipped geometry with initial charges 0.1 (the Mulliken column), atoms(44) (a repair) has NaN charges -> no initial charges; smiles(); frequencies() 3 values; no property row anywhere",
              a35.get_chemical_symbols() == ref35.get_chemical_symbols() and np.abs(a35.positions - ref35.positions).max() == 0.0
              and np.allclose(a35.get_initial_charges(), 0.1) and np.abs(a44.get_initial_charges()).max() == 0.0
              and np.isnan(arc.group(44)["charges"][:]).all() and arc.smiles(35) == ("CCC=O", "CCC=O")
              and list(arc.frequencies(18)) == [100.0, 200.0, 300.0] and "properties" not in arc.group(35)
              and a44.info["repaired"] is True and a35.info["qm9_index"] == 35,
              (a35.get_initial_charges()[:2], a44.get_initial_charges()[:2], arc.smiles(35), list(arc.frequencies(18))))
        sub7 = tmp / "sub.h5"
        # --only: a reference file of two molecules
        os.environ["S0_CURATED_QM9"] = str(src)
        curated_qm9._INDEX_CACHE.clear()
        idx2 = curated_qm9._index()
        at2 = pk.pack(idx2, sub7, source=str(src), only={18, 46})
        del os.environ["S0_CURATED_QM9"]
        curated_qm9._INDEX_CACHE.clear()
        check("pack(only={18, 46}): a two-group file, counts 1 original + 1 repaired, verify on the subset finds no mismatch",
              at2["n_molecules"] == 2 and at2["n_original"] == 1 and at2["n_repaired_short_tag"] == 1
              and sorted(curated_qm9.Archive(sub7).h5.keys()) == ["dsgdb9nsd_000018", "dsgdb9nsd_000046"]
              and not pk.verify(idx2, sub7, n=2, only={18, 46}), at2)
        curated_qm9._ARCHIVE_CACHE.clear()
        curated_qm9._INDEX_CACHE.clear()

    check("qm9_float reads all three exponent spellings: 1.5, -0.535689*^-6, -8.7796E^-6 (151 archive files), 2e-3",
          curated_qm9.qm9_float("1.5") == 1.5 and abs(curated_qm9.qm9_float("-0.535689*^-6") + 0.535689e-6) < 1e-18
          and abs(curated_qm9.qm9_float("-8.7796E^-6") + 8.7796e-6) < 1e-17 and curated_qm9.qm9_float("2e-3") == 0.002)
    print("PASS" if not FAIL else "FAIL: " + "; ".join(FAIL))
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
