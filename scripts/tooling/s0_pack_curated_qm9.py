"""Pack the curated QM9 directory (133 661 xyz files) into ONE HDF5, one group per
molecule (layout fixed 2026-09-21, `curated_qm9` module docstring).

TOOLING. For every file (the repairs winning over the originals exactly as the directory
lookup decides) the group `dsgdb9nsd_%06d` gets the parsed geometry (`species`,
`positions` in A, the Mulliken `charges` -- NaN for a repaired file, which carries none),
the `frequencies`, the two SMILES and two InChI as attributes, `repaired` and the
archive's own `source_file` name. Not stored: the property row
(line 2; nothing reads it) and the file's text (it doubled the size: the parsed form is
what `curated_qm9.find()` renders back into a QM9 file for the readers). File attributes:
the source, N, the census per naming pattern, the date.
After writing, N random molecules are read back through `curated_qm9.Archive` and their
species, positions, charges, frequencies and SMILES compared with the files.

    python scripts/tooling/s0_pack_curated_qm9.py                                  # data/qm9/curated_qm9.h5
    python scripts/tooling/s0_pack_curated_qm9.py --only 18,19,35,36,44,46,48 --out data/qm9/curated_qm9_sample7.h5
    scp data/qm9/curated_qm9.h5 tianhe:~/openQHA-main/data/qm9/                   # then the campaign runs there
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


sys.path.insert(0, str(_repo_root()))
from openqha import config                                      # noqa: E402
from openqha.data import curated_qm9                            # noqa: E402

LAYOUT = ("one group per molecule, dsgdb9nsd_%06d; attrs qm9_index natoms repaired source_file smiles_gdb17 "
          "smiles_relaxed inchi_gdb17 inchi_relaxed; datasets species positions(A) charges(e, NaN when the file "
          "has none) frequencies(cm^-1); the property row and the file's text are not stored")


def pack(src_index, out, source="", only=None):
    """`src_index`: {qm9 int: (pattern name, path)} as `curated_qm9._index` gives it;
    `only`: a subset of indices. Writes `out`; returns the attrs written."""
    import h5py
    numbers = sorted(n for n in src_index if only is None or n in only)
    counts = {name: 0 for name, _p in curated_qm9.PATTERNS}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".h5.tmp")
    # libver="latest": compact object headers -- 133 661 groups x 5 datasets are ~670 000 HDF5
    # objects, and their headers, not the data, are the file's size (measured 2026-09-21 on 2 000
    # molecules: 769 MB projected with the default format, 637 MB with this one; the text
    # alone is 180 MB). Needs HDF5 >= 1.10 to read, which every environment here has.
    with h5py.File(str(tmp), "w", libver="latest") as f:
        for n in numbers:
            name, path = src_index[n]
            counts[name] += 1
            text = Path(path).read_text(encoding="utf-8")
            d = curated_qm9.parse_qm9_text(text)
            g = f.create_group(curated_qm9.group_name(n))
            g.attrs["qm9_index"] = int(n)
            g.attrs["natoms"] = int(d["natoms"])
            g.attrs["repaired"] = bool(name != "original")
            g.attrs["source_file"] = Path(path).name
            for k in ("smiles_gdb17", "smiles_relaxed", "inchi_gdb17", "inchi_relaxed"):
                g.attrs[k] = d[k]
            g.create_dataset("species", data=np.array(d["species"], dtype="S2"))
            g.create_dataset("positions", data=d["positions"])
            g.create_dataset("charges", data=d["charges"])
            g.create_dataset("frequencies", data=d["frequencies"])
        attrs = dict(source=str(source), n_molecules=len(numbers),
                     packed=time.strftime("%Y-%m-%dT%H:%M:%S"), layout=LAYOUT,
                     citation="Senthil, Chakraborty & Ramakrishnan, Chem. Sci. 2021, 12, 5566")
        attrs.update({"n_" + k: v for k, v in counts.items()})
        for k, v in attrs.items():
            f.attrs[k] = v
    tmp.replace(out)
    return attrs


def verify(src_index, out, n=20, seed=0, only=None):
    """Read `n` random molecules back and compare with the files; returns the mismatches."""
    arc = curated_qm9.Archive(out)
    rng = np.random.default_rng(seed)
    numbers = sorted(m for m in src_index if only is None or m in only)
    pick = rng.choice(numbers, size=min(n, len(numbers)), replace=False)
    bad = []
    for m in pick:
        name, path = src_index[int(m)]
        text = Path(path).read_text(encoding="utf-8")
        d = curated_qm9.parse_qm9_text(text)
        a = arc.atoms(int(m))
        r = arc.record(int(m))
        same_charges = (np.isnan(r["charges"]).all() and np.isnan(d["charges"]).all()) or np.array_equal(r["charges"], d["charges"])
        if (arc.pattern_name(int(m)) != name or arc.filename(int(m)) != Path(path).name
                or a.get_chemical_symbols() != d["species"] or np.abs(a.positions - d["positions"]).max() > 0
                or not same_charges or not np.array_equal(r["frequencies"], d["frequencies"])
                or arc.smiles(int(m)) != (d["smiles_gdb17"], d["smiles_relaxed"])
                or (r["inchi_gdb17"], r["inchi_relaxed"]) != (d["inchi_gdb17"], d["inchi_relaxed"])):
            bad.append(int(m))
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="default: data/qm9/curated_qm9.h5")
    ap.add_argument("--only", default=None, help="comma-separated QM9 indices (a reference file)")
    ap.add_argument("--verify", type=int, default=20, help="molecules read back and compared")
    args = ap.parse_args()
    cfg = config.load()
    r = curated_qm9.root(cfg)
    if r is None:
        raise SystemExit("no curated QM9 directory (S0_CURATED_QM9 or data/qm9/{})".format(curated_qm9.DEFAULT_DIRNAME))
    out = Path(args.out) if args.out else _repo_root() / "data" / "qm9" / curated_qm9.H5_NAME
    only = {int(x) for x in args.only.split(",")} if args.only else None
    t0 = time.time()
    idx = curated_qm9._index(cfg)
    print("packing {} molecules from {}".format(len(only) if only else len(idx), r))
    attrs = pack(idx, out, source=str(r), only=only)
    print("wrote {}  {:.1f} MB ({:.1f} s)".format(out, out.stat().st_size / 1e6, time.time() - t0))
    for k in ("n_molecules", "n_original", "n_repaired_qm9_tag", "n_repaired_short_tag", "packed"):
        print("  {:22s} {}".format(k, attrs[k]))
    bad = verify(idx, out, args.verify, only=only)
    print("verify: {} molecules read back, {} mismatches".format(min(args.verify, attrs["n_molecules"]), len(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
