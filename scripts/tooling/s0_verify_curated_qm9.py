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

    # every group, read back: exit 0 only when nothing is damaged
    python scripts/tooling/s0_verify_curated_qm9.py
    # just the molecules a campaign is failing on
    python scripts/tooling/s0_verify_curated_qm9.py --only 14156,65586,88898

The file hash and the per-group content digests (`--sha`, `--digest`) were retired
2026-10-02 (identity without checksums: delete them, no size substitute); the tool
reports which groups cannot be read and nothing else. HDF5 checksums metadata, not
dataset bytes, so a corruption that leaves the structure valid but the values wrong
is NOT detected here. Copy the archive in a way that cannot be watched mid-write
(write a temporary name, move it into place) -- a reader that opens the file while
`scp` is overwriting it in place sees exactly the errors above.
"""
import argparse
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


def verify_group(arc, n):
    """Read molecule `n` back the way a campaign reads it.

    Returns `error`: None, or "Type: message" when the group cannot be read.
    """
    try:
        arc.filename(n)                    # the attributes the archive names it by
        arc.pattern_name(n)                # ... and the pattern the name decides
        arc.repaired(n)
        arc.text(n)                        # attrs + species + positions + charges +
        return None                        #   frequencies, rendered through render_qm9_text
    except Exception as e:                                                  # noqa: BLE001
        return "{}: {}".format(type(e).__name__, e)


def verify_archive(h5_path, only=None, progress_every=10000, log=print):
    """Read the archive at `h5_path` back. Returns (n_checked, failures).

    `only`: QM9 indices to check (default: every group in the file). A group listed
    by `only` but absent from the archive is a failure; an absent index is NOT an
    error when scanning the whole file.
    """
    arc = curated_qm9.Archive(h5_path)
    failures = {}

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
            continue
        if not present:
            if n in wanted:
                failures[n] = "ABSENT from the archive"
            continue
        err = verify_group(arc, n)
        checked += 1
        if err:
            failures[n] = err
        if progress_every and i % progress_every == 0:
            log("  ... {} looked up, {} checked, {} bad ({:.0f} s)".format(
                i, checked, len(failures), time.time() - t0))
    arc.h5.close()
    return checked, failures


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", default=None,
                    help="the archive; default: the one this repository resolves "
                         "(S0_CURATED_QM9_H5, configuration, data/qm9/curated_qm9.h5)")
    ap.add_argument("--only", default=None,
                    help="comma-separated QM9 indices, `18` or `dsgdb9nsd_000018`; "
                         "default: every group")
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

    only = sorted(curated_qm9._to_int(x) for x in args.only.split(",")) if args.only else None
    n, failures = verify_archive(path, only=only, log=log)
    print("checked  {} group{}".format(n, "" if n == 1 else "s"))
    if failures:
        print("DAMAGED  {} group{}:".format(
            len(failures), "" if len(failures) == 1 else "s"))
        for m in sorted(failures):
            print("  {}  {}".format(curated_qm9.group_name(m), failures[m]))
        print("This copy is damaged. Re-copy the archive written by "
              "scripts/tooling/s0_pack_curated_qm9.py (write a temporary name, move it "
              "into place, then run this tool again on the other side of the copy).")
        return 1
    print("no damage in the groups checked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
