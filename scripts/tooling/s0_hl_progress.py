"""The progress of a Hessian-learning campaign, from disk, in one table (ticket 08).

TOOLING. For every molecule of the draw (`draw.dat`; else `select.dat`; else every
molecule under the tag with branch A) it reads what is on disk -- branch A done
(`basins.done`), branch A's failure marker (`_records/branchA.failed`, ticket 26: ran, no
ensemble, not rerun), a Frame set Record (`frames/frames.toml`, its kept frames), the frames
whose ORCA job at `--level` is finished (`frame_labels.finished` on the file group in
`frames/`), the frames whose ORCA ran and FAILED (a `.out` without the terminal line;
not rerun, a human's `--retry` -- ticket 24), the frames another job holds (`.running`
lock: its job alive by `squeue`, its heartbeat fresh) -- and prints one row per structure
class and the total: molecules drawn / branch A / Frame sets, frames total / labelled /
failed / unlabelled / running. `squeue` is asked only about the job named in a lock; what
is on disk is the question. Login-node cheap: a few `stat`s per molecule, no MACE, no
extxyz parsed.

    python scripts/tooling/s0_hl_progress.py --tag draw300
    python scripts/tooling/s0_hl_progress.py --tag rings --tag propanal --name smoke --level hf_cc-pvtz
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
from openqha.data import dataset, frame_labels, frames as frames_mod   # noqa: E402
from openqha.store import basins, dat, layout                   # noqa: E402

COLUMNS = ("drawn", "branchA", "A_failed", "frame_sets", "frames", "labelled", "failed", "unlabelled", "running")


def molecules(root, tags, name):
    """[(qid, tag, molecule_dir, classes)]: the draw's rows, else the selection's, else
    every branch-A molecule under the tags (classes then classified from the SMILES)."""
    d = dataset.datasets_dir(root, tags[0], name)
    for stem in ("draw", "select"):
        p = d / (stem + ".dat")
        if p.is_file():
            out = []
            for r in dat.read_table(p):
                qid = r["qm9_index"]
                tag = r.get("tag") or next((t for t in tags if basins.done(qid, t, root=root)), tags[0])
                out.append((qid, tag, basins.molecule_for(qid, tag, root=root), str(r.get("classes", "-"))))
            return out, str(p)
    out = []
    for tag in tags:
        for qid, mol in dataset.molecules_with_branch_a(root, tag):
            out.append((qid, tag, mol, dataset.classes_of(dataset._smiles_of(mol, qid, tag, root))))
    return out, "the tags"


def molecule_progress(qid, tag, mol, level, root):
    """One molecule's counters (COLUMNS without `drawn`), from disk."""
    row = dict(branchA=0, A_failed=0, frame_sets=0, frames=0, labelled=0, failed=0, unlabelled=0, running=0)
    if not basins.done(qid, tag, root=root):
        row["A_failed"] = int(basins.failed(qid, tag, root=root))      # branch A ran and left its marker (ticket 26)
        return row
    row["branchA"] = 1
    rec = layout.frames_dir(mol) / (frames_mod.STEP + ".toml")
    if not rec.is_file():
        return row
    row["frame_sets"] = 1
    folder = layout.frames_dir(mol)
    for g, b, k in frame_labels.frame_list(mol):
        row["frames"] += 1
        stem = layout.orca_frame_stem(level, g, b, k)
        if frame_labels.finished(folder, stem, hessian=frame_labels.wants_hessian(g)):
            row["labelled"] += 1
        elif frame_labels.failed(folder, stem):
            row["failed"] += 1
        else:
            row["unlabelled"] += 1
            if frame_labels.running_elsewhere(folder, stem):
                row["running"] += 1
    return row


def progress(root, tags, name, level=frame_labels.DEFAULT_LEVEL):
    """{'total': counters, 'classes': {class: counters}, 'molecules': n, 'source': path}."""
    mols, source = molecules(root, tags, name)
    total = {c: 0 for c in COLUMNS}
    per_class = {}
    for qid, tag, mol, classes in mols:
        row = molecule_progress(qid, tag, mol, level, root)
        row["drawn"] = 1
        for c in COLUMNS:
            total[c] += row[c]
        for cls in [c for c in classes.split(";") if c and c != "-"]:
            acc = per_class.setdefault(cls, {c: 0 for c in COLUMNS})
            for c in COLUMNS:
                acc[c] += row[c]
    return dict(total=total, classes=per_class, molecules=len(mols), source=source)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", action="append", required=True)
    ap.add_argument("--name", default=None, help="the Dataset (default: the first tag)")
    ap.add_argument("--level", default=frame_labels.DEFAULT_LEVEL)
    args = ap.parse_args()
    t0 = time.time()
    root = config.runs_root(config.load())
    name = args.name or args.tag[0]
    out = progress(root, args.tag, name, args.level)
    fmt = "{:22s} {:>6} {:>7} {:>8} {:>10} {:>7} {:>8} {:>6} {:>10} {:>7}"
    print("progress of {!r} at {} ({} molecules from {})".format(name, args.level, out["molecules"], out["source"]))
    print(fmt.format("class", "drawn", "branchA", "A failed", "frame sets", "frames", "labelled", "failed", "unlabelled", "running"))
    for cls in sorted(out["classes"]):
        r = out["classes"][cls]
        print(fmt.format(cls, *[r[c] for c in COLUMNS]))
    t = out["total"]
    print(fmt.format("TOTAL", *[t[c] for c in COLUMNS]))
    done = t["labelled"] / t["frames"] * 100 if t["frames"] else 0.0
    print("{:.1f} % of the frames labelled; {} failed (not rerun: read the .out, then --retry); {} unlabelled with no job on disk; "
          "branch A failed {} (not rerun: read _records/branchA.failed, then s0_A_pipeline --species by hand); {:.1f} s".format(
        done, t["failed"], t["unlabelled"], t["A_failed"], time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
