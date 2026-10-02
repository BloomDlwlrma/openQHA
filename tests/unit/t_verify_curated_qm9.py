"""The archive check: read every group of the packed curated QM9 back and report the
damaged ones (`scripts/tooling/s0_verify_curated_qm9.py`).

The failure this exists for (Tianhe, 2026-09-24): a copy of the 400 MB archive whose
metadata was damaged for three molecules -- the file OPENS, most groups read fine, and
the three die with "incorrect metadata checksum after all read attempts" (one at the
link existence check, one at the object open). A truncated copy fails at open instead,
so the tool must not confuse the two. Built here: a three-molecule archive from the
shipped geometries; the healthy file verifies with no failure; a copy with one group's
object header damaged is caught as an unreadable group (the file still opens, and the
error carries the Tianhe shape); an index asked for with --only that the archive does
not hold is reported ABSENT rather than passed. The file hash and the per-group digest
modes are gone (identity without checksums, 2026-10-02): `--sha` and `--digest` are
asserted to no longer exist.
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


def flip_header_byte(src, dst, addr, delta=8):
    """Copy `src` to `dst` with one byte flipped inside the object header at `addr`."""
    shutil.copy(src, dst)
    with open(dst, "r+b") as fh:
        fh.seek(addr + delta)
        b = fh.read(1)
        fh.seek(addr + delta)
        fh.write(bytes([b[0] ^ 0xFF]))
    return Path(dst)


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

        n, bad = vf.verify_archive(out)
        check("healthy three-group archive: 3 groups checked, no failure",
              n == 3 and not bad, (n, bad))

        n2, bad2 = vf.verify_archive(out, only=[18, 999999])
        check("--only: a present index checks and passes, an index the archive does not hold "
              "is a failure ('ABSENT'), not a silent pass",
              n2 == 1 and list(bad2) == [999999] and "ABSENT" in bad2[999999], (n2, bad2))

        codes = []
        for argv in (["--h5", str(out), "--sha"],
                     ["--h5", str(out), "--digest", str(tmp / "digests.txt")]):
            try:
                vf.main(argv)
                codes.append(0)
            except SystemExit as e:
                codes.append(e.code)
        check("the hash modes are gone: --sha and --digest are unknown arguments (exit 2)",
              codes == [2, 2], codes)

        # --- one group's object header damaged: the file still opens, the tool catches it --
        import h5py
        with h5py.File(str(out), "r") as f:
            header_addr = int(h5py.h5o.get_info(f["dsgdb9nsd_000019"].id).addr)
        size = out.stat().st_size
        cut = flip_header_byte(out, tmp / "flip.h5", header_addr)
        nc, bc = vf.verify_archive(cut)
        opens = True
        try:
            h5py.File(cut, "r").close()
        except Exception:                                                   # noqa: BLE001
            opens = False
        check("a byte flipped inside group 000019's object header (the file still opens) is "
              "caught as an unreadable group",
              nc == 3 and list(bc) == [19] and opens, (nc, bc, opens))
        msg = (bc or {}).get(19, "")
        check("the caught error is the Tianhe shape -- a metadata checksum failure, not "
              "'truncated file' (the two must not be confused: truncation fails at open)",
              "checksum" in msg and "truncated file" not in msg, msg[:200])

        # --- main(): exit codes ---------------------------------------------------------
        rc_ok = vf.main(["--h5", str(out), "--quiet"])
        rc_bad = vf.main(["--h5", str(cut), "--quiet"])
        check("main(): 0 on the healthy archive, 1 when a group is unreadable",
              rc_ok == 0 and rc_bad == 1, (rc_ok, rc_bad))

        # --- curated_qm9.find() on the damaged copy: the error names the file and the fix --
        old_env = {k: os.environ.pop(k, None) for k in ("S0_CURATED_QM9", "S0_CURATED_QM9_H5")}
        try:
            cfg = config.load()
            cfg["data"]["curated_qm9_dir"] = "no_such_directory_for_the_test"
            find_cut = flip_header_byte(out, tmp / "find_cut.h5", header_addr)
            cfg["data"]["curated_qm9_h5"] = str(find_cut)
            curated_qm9._ARCHIVE_CACHE.clear()
            curated_qm9._INDEX_CACHE.clear()
            try:
                curated_qm9.find(19, cfg)
                msg = ""
            except RuntimeError as e:
                msg = str(e)
            check("curated_qm9.find() on a copy whose group 000019 header is damaged raises "
                  "with the file, the group, the underlying error, the verify command and "
                  "the re-copy advice (an unreadable group must not come back as (None, None))",
                  "could not be read for dsgdb9nsd_000019" in msg
                  and str(find_cut) in msg and "incorrect metadata checksum" in msg
                  and "--only 19" in msg and "re-copy" in msg, msg[:300])

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
