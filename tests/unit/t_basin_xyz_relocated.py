"""A basin store copied to another cluster still finds its geometries.

UNIT. No dependencies beyond the package; under a second.

The defect (an113, 2026-09-13)
------------------------------
Branch A ran on TianheXY-CN and wrote, inside each record,
`basin_store.xyz = '/XYFS02/.../dsgdb9nsd_000018.basins.xyz'`. The store was copied to
TianheXY-AI (/XYAIFS00) for branch B. The trajectory driver found the xyz -- it derives
the path from the store root -- and the 02d identity step, which dereferenced the string
in the record, opened /XYFS02 on a filesystem that has no /XYFS02:

    FileNotFoundError: [Errno 2] No such file or directory: '/XYFS02/.../*.basins.xyz'

`basin_store.xyz_for` now locates the file the way `read()` located the json.

What is asserted
----------------
  A. With S0_BASIN_ROOT pointing at a fresh store whose record carries a foreign
     absolute path, xyz_for returns the sibling of the json, which exists.
  B. With the sibling removed it raises FileNotFoundError naming BOTH the path it
     looked at and the path the record carries, so a real gap in the copy is named.
"""
import json
import os
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

FAIL = []
SPECIES = "dsgdb9nsd_000018"
TAG = "t_relocated"
FOREIGN = "/XYFS02/HDD_POOL/somebody/openQHA-main/data/basins/{}/1_16000/1_4000/{}.basins.xyz".format(
    TAG, SPECIES)
XYZ = "3\nbasin 0\nC 0 0 0\nO 1.2 0 0\nH -1 0 0\n"


def main():
    saved = os.environ.get("S0_BASIN_ROOT")
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["S0_BASIN_ROOT"] = tmp
        try:
            from openqha.store import basin_store
            rec = dict(qm9_index=SPECIES, basins=[dict(relative_kcal=0.0)],
                       basin_store=dict(json="/XYFS02/whatever.json", xyz=FOREIGN))
            j, x = basin_store.write(SPECIES, rec, XYZ, tag=TAG)

            print("A. record carries a foreign absolute path; the store finds the sibling")
            got = basin_store.xyz_for(SPECIES, basin_store.read(SPECIES, tag=TAG), tag=TAG)
            ok = got == x and got.read_text(encoding="utf-8") == XYZ and str(got) != FOREIGN
            print("   {}  {}".format("ok  " if ok else "FAIL", got))
            if not ok:
                FAIL.append("xyz_for returned {} (wanted {})".format(got, x))

            print("\nB. sibling absent -> refusal that names both paths")
            x.unlink()
            try:
                basin_store.xyz_for(SPECIES, rec, tag=TAG)
                FAIL.append("xyz_for returned with the xyz absent")
                print("   FAIL  returned")
            except FileNotFoundError as exc:
                text = str(exc)
                ok = str(x) in text and FOREIGN in text
                print("   {}  {}".format("ok  " if ok else "FAIL", text.splitlines()[0]))
                if not ok:
                    FAIL.append("the refusal does not name both paths: {}".format(text))
        finally:
            if saved is None:
                os.environ.pop("S0_BASIN_ROOT", None)
            else:
                os.environ["S0_BASIN_ROOT"] = saved

    print()
    if FAIL:
        print("{} problem(s):".format(len(FAIL)))
        for f in FAIL:
            print("  - " + f)
        return 1
    print("the store locates its own geometries wherever it was copied to")
    return 0


if __name__ == "__main__":
    sys.exit(main())
