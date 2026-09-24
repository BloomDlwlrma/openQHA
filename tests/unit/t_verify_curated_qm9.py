"""The archive check: read every group of the packed curated QM9 back and report the
damaged ones (`scripts/tooling/s0_verify_curated_qm9.py`).

The failure this exists for (Tianhe, 2026-09-24): a copy of the 400 MB archive whose
metadata was damaged for three molecules -- the file OPENS, most groups read fine, and
the three die with "incorrect metadata checksum after all read attempts" (one at the
link existence check, one at the object open). A truncated copy fails at open instead,
so the tool must not confuse the two. Built here: a three-molecule archive from the
shipped geometries; the healthy file verifies with no failure; single bytes flipped
across the middle of a copy are caught (as an unreadable group, or -- when the flip
lands in a dataset's raw bytes, which HDF5 does not checksum -- as a changed per-group
digest against the healthy file); an index asked for with --only that the archive does
not hold is reported ABSENT rather than passed.
"""
import os
import shutil
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
sys.path.insert(0, str(ROOT / "scripts" / "tooling"))
import s0_pack_curated_qm9 as pk                             # noqa: E402
import s0_verify_curated_qm9 as vf                           # noqa: E402
from openqha import config                                   # noqa: E402
from openqha.data import curated_qm9                         # noqa: E402

FAIL = []
GEOMS = ROOT / "data" / "reference-geometries"


def check(label, ok, detail=""):
    print("  {:80s} {}".format(label, "ok" if ok else "FAIL " + str(detail)[:300]))
    if not ok:
        FAIL.append(label)


def qm9_form(path, smiles):
    lines = path.read_text(encoding="utf-8").splitlines()
    n = int(lines[0].split()[0])
    atoms = [row + "\t0.1" for row in lines[2:2 + n]]
    body = ["{}".format(n), "gdb 1\t0.0"] + atoms + ["100.0\t200.0\t300.0",
                                                     "{}\t{}".format(smiles, smiles),
                                                     "InChI=1S/x\tInChI=1S/x"]
    return "\n".join(body) + "\n"


def main():
    with tempfile.TemporaryDirectory(prefix="verify_h5_") as tmp:
        tmp = Path(tmp)
        src = tmp / "archive"
        src.mkdir()
        smiles = {"000018": "CC(=O)C", "000019": "CC(=O)NC", "000035": "CCC=O"}
        for k, s in smiles.items():
            (src / "dsgdb9nsd_{}.xyz".format(k)).write_text(
                qm9_form(GEOMS / "dsgdb9nsd_{}.xyz".format(k), s), encoding="utf-8")
        os.environ["S0_CURATED_QM9"] = str(src)
        curated_qm9._INDEX_CACHE.clear()
        out = tmp / "curated_qm9.h5"
        pk.pack(curated_qm9._index(), out, source=str(src))
        del os.environ["S0_CURATED_QM9"]
        curated_qm9._INDEX_CACHE.clear()

        n, bad, lines = vf.verify_archive(out)
        check("healthy three-group archive: 3 groups checked, no failure, one digest line "
              "per group (name + sha256)",
              n == 3 and not bad and len(lines) == 3
              and all(l.split()[0].startswith("dsgdb9nsd_") and len(l.split()) == 2
                      for l in lines), (n, bad, lines))

        n2, bad2, d2 = vf.verify_archive(out, only=[18, 999999])
        check("--only: a present index checks and passes, an index the archive does not hold "
              "is a failure ('ABSENT'), not a silent pass",
              n2 == 1 and list(bad2) == [999999] and "ABSENT" in bad2[999999], (n2, bad2))

        dg = tmp / "digests.txt"
        vf.verify_archive(out, digest_path=dg)
        check("--digest writes the same content the return value carries",
              dg.read_text(encoding="utf-8").splitlines() == lines, dg.read_text(encoding="utf-8")[:120])

        # --- a byte flipped in the middle: the file still opens, and the tool catches it --
        size = out.stat().st_size
        caught = None
        for frac in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.97]:
            cut = tmp / "flip.h5"
            shutil.copy(out, cut)
            with open(cut, "r+b") as fh:
                off = int(size * frac)
                fh.seek(off)
                b = fh.read(1)
                fh.seek(off)
                fh.write(bytes([b[0] ^ 0xFF]))
            nc, bc, dc = vf.verify_archive(cut)
            import h5py
            opens = True
            try:
                h5py.File(cut, "r").close()
            except Exception:                                               # noqa: BLE001
                opens = False
            if bc or dc != lines:
                caught = dict(frac=frac, errors=bc, digest_changed=dc != lines, opens=opens)
                break
        check("a single byte flipped mid-file (file still opens) is caught: as an unreadable "
              "group and/or as a digest that no longer matches the healthy file",
              caught is not None and caught["opens"] and (caught["errors"] or caught["digest_changed"]),
              caught)

        # --- the errors are the Tianhe shape, not "truncated file" ----------------------
        if caught and caught["errors"]:
            msg = " ".join(caught["errors"].values())
            check("the caught error is a checksum/membership failure, not 'truncated file' "
                  "(the two must not be confused: truncation fails at open)",
                  ("checksum" in msg or "open object" in msg or "link" in msg)
                  and "truncated file" not in msg, msg[:200])

        # --- main(): exit codes ---------------------------------------------------------
        rc_ok = vf.main(["--h5", str(out), "--quiet"])
        rc_bad = None
        if caught:
            shutil.copy(out, tmp / "bad2.h5")
            with open(tmp / "bad2.h5", "r+b") as fh:
                fh.seek(int(size * caught["frac"]))
                b = fh.read(1)
                fh.seek(int(size * caught["frac"]))
                fh.write(bytes([b[0] ^ 0xFF]))
            _ncc, bcc, _dcc = vf.verify_archive(tmp / "bad2.h5")
            if bcc:
                rc_bad = vf.main(["--h5", str(tmp / "bad2.h5"), "--quiet"])
        check("main(): 0 on the healthy archive, 1 when a group is unreadable",
              rc_ok == 0 and (rc_bad is None or rc_bad == 1), (rc_ok, rc_bad))

        # --- curated_qm9.find() on the damaged copy: the error names the file and the fix --
        old_env = {k: os.environ.pop(k, None) for k in ("S0_CURATED_QM9", "S0_CURATED_QM9_H5")}
        try:
            cfg = config.load()
            cfg["data"]["curated_qm9_dir"] = "no_such_directory_for_the_test"
            cfg["data"]["curated_qm9_h5"] = str(tmp / "find_cut.h5")
            curated_qm9._ARCHIVE_CACHE.clear()
            curated_qm9._INDEX_CACHE.clear()
            found_k, found_frac, msg = None, None, ""
            for frac in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.97]:
                cut = tmp / "find_cut.h5"
                shutil.copy(out, cut)
                with open(cut, "r+b") as fh:
                    off = int(size * frac)
                    fh.seek(off)
                    b = fh.read(1)
                    fh.seek(off)
                    fh.write(bytes([b[0] ^ 0xFF]))
                for k in (18, 19, 35):
                    curated_qm9._ARCHIVE_CACHE.clear()
                    try:
                        curated_qm9.find(k, cfg)
                    except RuntimeError as e:
                        if "s0_verify_curated_qm9.py" in str(e):
                            found_k, found_frac, msg = k, frac, str(e)
                            break
                if found_k is not None:
                    break
            check("curated_qm9.find() on a damaged copy raises with the file, the group, "
                  "the underlying error, the verify command and the re-copy advice (an "
                  "unreadable group must not come back as (None, None))",
                  found_k is not None and "could not be read for dsgdb9nsd_{:06d}".format(found_k) in msg
                  and str(cut) in msg and "incorrect metadata checksum" in msg
                  and "--only {}".format(found_k) in msg and "re-copy" in msg,
                  (found_k, found_frac, (msg or "")[:300]))

            # a truncated copy cannot be opened at all: the same remedy, and the two cases
            # stay named apart ('truncated file' vs 'metadata checksum')
            trunc = tmp / "truncated.h5"
            shutil.copy(out, trunc)
            with open(trunc, "r+b") as fh:
                fh.truncate(size - 40)
            curated_qm9._ARCHIVE_CACHE.clear()
            cfg["data"]["curated_qm9_h5"] = str(trunc)
            try:
                curated_qm9.find(18, cfg)
                err_t = ""
            except RuntimeError as e:
                err_t = str(e)
            check("find() on a truncated copy names it too ('could not be opened', "
                  "'truncated file'), with the verify command -- not a bare h5py OSError",
                  "could not be opened" in err_t and "truncated file" in err_t
                  and "s0_verify_curated_qm9.py" in err_t, err_t[:200])
        finally:
            for k, v in old_env.items():
                if v is not None:
                    os.environ[k] = v
            curated_qm9._ARCHIVE_CACHE.clear()
            curated_qm9._INDEX_CACHE.clear()

    print("PASS" if not FAIL else "FAIL: " + "; ".join(FAIL))
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
