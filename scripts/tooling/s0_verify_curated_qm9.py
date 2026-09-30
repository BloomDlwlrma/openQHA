"""Read the packed curated QM9 archive back, group by group, and report the ones that
cannot be read.

TOOLING. The archive is ONE 400 MB file and a campaign asks it for
6 458 molecules a day from every node; a copy that lost a few blocks -- an
interrupted `scp`, a bad block on the target pool -- damages a FEW of the
133 661 groups and nothing says so until a molecule that lives in one of them
reaches branch A. Measured on Tianhe 2026-09-24: three molecules died in 21 s
each with

    RuntimeError: Unable to synchronously check link existence
                  (incorrect metadata checksum after all read attempts)
    KeyError: 'Unable to synchronously open object
              (incorrect metadata checksum after all read attempts)'

while everything else read fine. **A truncated copy is NOT this**: truncation
fails at `h5py.File(...)` with "truncated file", before any molecule is looked
up. These checksums are HDF5 detecting damaged metadata for specific groups.

    # the two numbers that decide "same copy?" -- run on both sides, compare
    python scripts/tooling/s0_verify_curated_qm9.py --sha
    # every group, read back: exit 0 only when nothing is damaged
    python scripts/tooling/s0_verify_curated_qm9.py
    # just the molecules a campaign is failing on
    python scripts/tooling/s0_verify_curated_qm9.py --only 14156,65586,88898
    # a content fingerprint per group, to diff a cluster copy against the workstation
    python scripts/tooling/s0_verify_curated_qm9.py --digest /tmp/h5_digests.txt

The digest is the sha256 of the molecule's rendered QM9 text (`Archive.text`),
which covers every stored attribute and every stored array. HDF5 checksums
metadata, not dataset bytes, so a `--digest` diff is the only way to catch a
corruption that leaves the structure valid but the numbers wrong. Copy the
archive in a way that cannot be watched mid-write (write a temporary name, move
it into place) -- a reader that opens the file while `scp` is overwriting it in
place sees exactly the errors above.
"""
import argparse
import hashlib
import sys
import time
from pathlib import Path


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


sys.path.insert(0, str(_repo_root()))
from openqha import config                                      # noqa: E402
from openqha.data import curated_qm9                            # noqa: E402


def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify_group(arc, n):
    """Read molecule `n` back the way a campaign reads it.

    Returns `(error, digest, n_chars)`: `error` is None or "Type: message"; `digest`
    is the sha256 of the rendered QM9 text (None when the group could not be read).
    The rendered text carries every stored attribute and every stored array, so the
    digest is the group's content fingerprint.
    """
    try:
        arc.filename(n)                    # the attributes the archive names it by
        arc.pattern_name(n)                # ... and the pattern the name decides
        arc.repaired(n)
        text = arc.text(n)                 # attrs + species + positions + charges +
        return None, hashlib.sha256(text.encode("utf-8")).hexdigest(), len(text)
        #   frequencies, rendered through render_qm9_text
    except Exception as e:                                                  # noqa: BLE001
        return "{}: {}".format(type(e).__name__, e), None, None


def verify_archive(h5_path, only=None, digest_path=None, progress_every=10000,
                   log=print):
    """Read the archive at `h5_path` back. Returns (n_checked, failures, digest_lines).

    `only`: QM9 indices to check (default: every group in the file). `digest_path`:
    where to write one line per group, "dsgdb9nsd_%06d <sha256|ERROR ...>", for a
    diff between two copies. A group listed by `only` but absent from the archive is
    a failure; an absent index is NOT an error when scanning the whole file.
    """
    arc = curated_qm9.Archive(h5_path)
    failures = {}
    digest_lines = []

    if only is None:
        try:
            names = sorted(arc.h5.keys())
        except Exception as e:                                              # noqa: BLE001
            arc.h5.close()
            raise RuntimeError(
                "the file listing itself could not be read ({}: {}) -- the root "
                "group's link structure is damaged; this copy cannot be trusted at "
                "all".format(type(e).__name__, e))
        numbers = [int(n.rsplit("_", 1)[1]) for n in names]
        wanted = set()
    else:
        numbers = sorted(int(n) for n in only)
        wanted = set(numbers)

    checked = 0
    t0 = time.time()
    for i, n in enumerate(numbers, start=1):
        present = True
        try:
            present = n in arc
        except Exception as e:                                              # noqa: BLE001
            failures[n] = "membership check failed: {}: {}".format(type(e).__name__, e)
            digest_lines.append("{} ERROR {}".format(curated_qm9.group_name(n),
                                                     failures[n]))
            continue
        if not present:
            if n in wanted:
                failures[n] = "ABSENT from the archive"
                digest_lines.append("{} ABSENT".format(curated_qm9.group_name(n)))
            continue
        err, digest, _nchars = verify_group(arc, n)
        checked += 1
        if err:
            failures[n] = err
            digest_lines.append("{} ERROR {}".format(curated_qm9.group_name(n), err))
        else:
            digest_lines.append("{} {}".format(curated_qm9.group_name(n), digest))
        if progress_every and i % progress_every == 0:
            log("  ... {} looked up, {} checked, {} bad ({:.0f} s)".format(
                i, checked, len(failures), time.time() - t0))
    arc.h5.close()

    if digest_path:
        Path(digest_path).write_text("\n".join(digest_lines) + "\n", encoding="utf-8")
        log("digest of {} groups -> {}".format(len(digest_lines), digest_path))
    return checked, failures, digest_lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", default=None,
                    help="the archive; default: the one this repository resolves "
                         "(S0_CURATED_QM9_H5, configuration, data/qm9/curated_qm9.h5)")
    ap.add_argument("--only", default=None,
                    help="comma-separated QM9 indices, `18` or `dsgdb9nsd_000018`; "
                         "default: every group")
    ap.add_argument("--digest", default=None, metavar="FILE",
                    help="write one line per group for a diff between two copies")
    ap.add_argument("--sha", action="store_true",
                    help="print the file's size and sha256, then stop (seconds)")
    ap.add_argument("--quiet", action="store_true", help="only the summary and the failures")
    args = ap.parse_args(argv)

    cfg = config.load()
    path = Path(args.h5) if args.h5 else curated_qm9.archive(cfg)
    if path is None or not Path(path).is_file():
        raise SystemExit("no archive: --h5, S0_CURATED_QM9_H5, data.curated_qm9_h5, or "
                         "data/qm9/{}".format(curated_qm9.H5_NAME))
    path = Path(path)
    size = path.stat().st_size
    log = (lambda *a: None) if args.quiet else print
    print("archive  {}".format(path))
    print("size     {} bytes".format(size))
    if args.sha:
        print("sha256   {}".format(file_sha256(path)))
        return 0

    only = sorted(curated_qm9._to_int(x) for x in args.only.split(",")) if args.only else None
    n, failures, _digest = verify_archive(path, only=only, digest_path=args.digest,
                                          log=log)
    print("checked  {} group{}".format(n, "" if n == 1 else "s"))
    if failures:
        print("DAMAGED  {} group{}:".format(
            len(failures), "" if len(failures) == 1 else "s"))
        for m in sorted(failures):
            print("  {}  {}".format(curated_qm9.group_name(m), failures[m]))
        print("This copy is damaged. Re-copy the archive written by "
              "scripts/tooling/s0_pack_curated_qm9.py (write a temporary name, move it "
              "into place, then run this tool again on the other side of the copy and "
              "diff the two --digest files).")
        return 1
    print("no damage in the groups checked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
