"""Move a tag's molecule tree to the flat form, in three stages that are each
idempotent.

TOOLING. Stage 1, the shard layers: `<tag>/<range>/<chunk>/<qid>/` -> `<tag>/<qid>/` and
`<tag>/_label/<label>/` -> `<tag>/<label>/`. Stage 2, the frame labels: every
`<molecule>/orca/<level>/frames/<generator>_bBB_kK/job.<ext>` (or the earlier interim
form `<molecule>/orca.<level>.<frame>.<ext>`) becomes `frames/orca.<level>.<generator>_bBB_kK<ext>`;
a `running` claim becomes `<stem>.running`. Stage 3, the msRRHO study into `msrrho/`:
`orca/<level>/basinNN/job.<ext>` -> `msrrho/orca.<level>.basinNN<ext>` (a probe sub-folder
`<tag>/job.<ext>` -> `msrrho/orca.<level>.basinNN.<tag><ext>`), `levels/<level>/<name>` ->
`msrrho/thermo/<level>.<name>`, `levels/<name>` -> `msrrho/thermo/<name>`, and `crest_entropy/`,
`xtb/` moved whole. Emptied folders are removed; a target that already exists is not
overwritten (reported as a conflict, the source left). Without `--apply` nothing moves: the
plan is printed. A second `--apply` finds nothing to do.

    python scripts/tooling/s0_flatten_tree.py --tag smoke            # the plan
    python scripts/tooling/s0_flatten_tree.py --tag smoke --apply
"""
import argparse
import re
import sys
from pathlib import Path


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


sys.path.insert(0, str(_repo_root()))
from openqha import config                                      # noqa: E402
from openqha.store import layout                                # noqa: E402

FRAME_DIR = re.compile(r"^(basin|displaced|merged|saddle)_b(\d{2})_k(\d+)$")
BASIN_DIR = re.compile(r"^basin(\d{2})$")
SHARD_DIR = re.compile(r"^\d+_\d+$")


def plan_shards(tag_dir):
    """[(molecule_dir, target)] for every molecule under a shard layer or `_label/`, and
    the folders to try to remove after (deepest first)."""
    tag_dir = Path(tag_dir)
    moves, folders = [], []
    ranges = sorted(p for p in tag_dir.iterdir() if p.is_dir() and SHARD_DIR.match(p.name))
    for r in ranges:
        chunks = sorted(p for p in r.iterdir() if p.is_dir() and SHARD_DIR.match(p.name))
        for c in chunks:
            for mol in sorted(p for p in c.iterdir() if p.is_dir()):
                moves.append((mol, tag_dir / mol.name))
            folders.append(c)
        folders.append(r)
    label = tag_dir / "_label"
    if label.is_dir():
        for mol in sorted(p for p in label.iterdir() if p.is_dir()):
            moves.append((mol, tag_dir / mol.name))
        folders.append(label)
    return moves, folders


def _job_files(job_dir, stem, target_dir):
    """The moves of one job folder's files to the file group `<target_dir>/<stem>.<ext>`."""
    out = []
    for f in sorted(p for p in job_dir.iterdir() if p.is_file()):
        if f.name == "running":
            out.append((f, target_dir / (stem + ".running")))
        elif f.name.startswith("job."):
            out.append((f, target_dir / (stem + f.name[len("job"):])))
        else:
            out.append((f, target_dir / (stem + "." + f.name)))
    return out


def plan_frames(molecule):
    """[(source, target)] for one molecule's frame jobs, and the folders to remove after."""
    molecule = Path(molecule)
    moves, folders = [], []
    orca = molecule / "orca"
    if orca.is_dir():
        for level_dir in sorted(p for p in orca.iterdir() if p.is_dir()):
            frames = level_dir / "frames"
            if not frames.is_dir():
                continue
            for job in sorted(p for p in frames.iterdir() if p.is_dir()):
                m = FRAME_DIR.match(job.name)
                if not m:
                    continue
                stem = layout.orca_frame_stem(level_dir.name, m.group(1), int(m.group(2)), int(m.group(3)))
                moves += _job_files(job, stem, layout.frames_dir(molecule))
                folders.append(job)
            folders.append(frames)
            folders.append(level_dir)
        folders.append(orca)
    # the earlier interim form: file groups directly in the molecule directory
    for f in sorted(molecule.glob("orca.*")):
        if f.is_file() and FRAME_DIR.match(f.name.split(".")[-2]):
            moves.append((f, layout.frames_dir(molecule) / f.name))
    return moves, folders


def plan_msrrho(molecule):
    """[(source, target)] for one molecule's msRRHO study, and the folders to remove after."""
    molecule = Path(molecule)
    moves, folders = [], []
    orca = molecule / "orca"
    if orca.is_dir():
        for level_dir in sorted(p for p in orca.iterdir() if p.is_dir()):
            for job in sorted(p for p in level_dir.iterdir() if p.is_dir()):
                m = BASIN_DIR.match(job.name)
                if not m:
                    continue
                stem = layout.orca_level_stem(level_dir.name, int(m.group(1)))
                moves += _job_files(job, stem, layout.msrrho_dir(molecule))
                for probe in sorted(p for p in job.iterdir() if p.is_dir()):
                    moves += _job_files(probe, stem + "." + probe.name, layout.msrrho_dir(molecule))
                    folders.append(probe)
                folders.append(job)
            folders.append(level_dir)
        folders.append(orca)
    levels = molecule / "levels"
    if levels.is_dir():
        for p in sorted(levels.iterdir()):
            if p.is_dir():
                for f in sorted(q for q in p.iterdir() if q.is_file()):
                    moves.append((f, layout.level_file(molecule, p.name, f.name)))
                folders.append(p)
            elif p.is_file():
                moves.append((p, layout.thermo_file(molecule, p.name)))
        folders.append(levels)
    for name in ("crest_entropy", "xtb"):
        d = molecule / name
        if d.is_dir():
            moves.append((d, layout.msrrho_dir(molecule) / name))
    return moves, folders


def apply(moves, folders):
    n_moved = n_conflict = 0
    for src, dst in moves:
        if dst.exists():
            n_conflict += 1
            print("conflict: {} exists; {} left".format(dst, src))
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
        n_moved += 1
    n_rm = 0
    for d in folders:                       # deepest first as listed
        try:
            d.rmdir()
            n_rm += 1
        except OSError:
            pass                            # not empty (a conflict) -> stays
    return n_moved, n_conflict, n_rm


def molecules(tag_dir):
    """Every molecule directory of the flat tag: a `_records/branchA.toml` one level down."""
    return sorted(p.parents[1] for p in Path(tag_dir).glob("*/{}/branchA.toml".format(layout.RECORDS)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    root = config.runs_root(config.load())
    tag_dir = Path(root) / args.tag
    if not tag_dir.is_dir():
        raise SystemExit("no tag directory {}".format(tag_dir))
    # stage 1: the shard layers
    moves, folders = plan_shards(tag_dir)
    if args.apply:
        n, c, r = apply(moves, folders)
        print("shards: {} molecules moved up, {} conflicts, {} folders removed".format(n, c, r))
    else:
        print("shards: {} molecules to move up{}".format(
            len(moves), ", e.g. {} -> {}".format(moves[0][0].relative_to(tag_dir), moves[0][1].name) if moves else ""))
    # stages 2 and 3, molecule by molecule (after stage 1 the molecules are one level down)
    mols = molecules(tag_dir) if args.apply else molecules(tag_dir) + [s for s, _ in moves]
    for stage, planner in (("frames", plan_frames), ("msrrho", plan_msrrho)):
        total_moves = total_conf = total_rm = 0
        for mol in mols:
            fm, ff = planner(mol)
            if not fm:
                continue
            if args.apply:
                n, c, r = apply(fm, ff)
                total_moves, total_conf, total_rm = total_moves + n, total_conf + c, total_rm + r
                print("{} {}: {} moved, {} conflicts, {} folders removed".format(stage, mol.name, n, c, r))
            else:
                total_moves += len(fm)
                print("{} {}: {} to move, e.g. {} -> {}".format(stage, mol.name, len(fm),
                                                              fm[0][0].relative_to(mol), fm[0][1].relative_to(mol)))
        print("{}: {} molecules; {} {}{}".format(
            stage, len(mols), total_moves, "moved" if args.apply else "to move",
            ", {} conflicts, {} folders removed".format(total_conf, total_rm) if args.apply else " (add --apply)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
