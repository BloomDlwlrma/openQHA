"""Ticket 08 of the Hessian-learning set: the campaign page and the progress table.

`s0_hl_progress.progress` on a fake tree -- a draw of three molecules: one with branch A,
a Frame set and some finished ORCA file groups plus a fresh `.running` claim, one with
branch A only, one branch A never ran for -- counts drawn / branch A / Frame sets /
frames / labelled / failed / unlabelled / running per class and in total from disk alone; a fresh
draw (nothing computed) gives the drawn count with zeros elsewhere. And the campaign page
(`docs/hessian_learning_campaign.md`) carries every `sbatch` command the stage scripts'
headers state, and every `sbatch` command it states is one of theirs -- as
`t_script_taxonomy` holds the taxonomy lines, so the page and the scripts cannot drift.
"""
import re
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
sys.path.insert(0, str(ROOT / "tests" / "unit"))
sys.path.insert(0, str(ROOT / "scripts" / "tooling"))
from t_frames import Harmonic, check, make_molecule, FAIL           # noqa: E402

from openqha.data import dataset, frame_labels, frames, structure_classes as sc   # noqa: E402
from openqha.store import dat, layout                                 # noqa: E402
import s0_hl_progress as hp                                           # noqa: E402

LEVEL = "wb97m-d3bj_def2-tzvppd"
TAG = "fake"
SCRIPTS = ("hl_branchA.slurm", "hl_frames.slurm", "hl_labels.slurm", "hl_pipeline_debug.slurm")
PAGE = ROOT / "docs" / "hessian_learning_campaign.md"


def place(tmp, qid, with_frames=True):
    root = Path(tmp) / "root"
    mol, basins = make_molecule(Path(tmp) / ("src_" + qid), with_merged=False)
    dest = layout.molecule_dir(root, TAG, qid)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(mol), str(dest))
    rec = dest / "_records" / "branchA.toml"
    rec.write_text(rec.read_text(encoding="utf-8").replace("dsgdb9nsd_000035", qid), encoding="utf-8")
    if with_frames:
        frames.generate(dest, calc=Harmonic(basins), engine_name="MACE-OFF23_medium", n_displaced=4)
    return root, dest


def finish(mol, g, b, k):
    """A finished ORCA file group for one frame: the terminal line and the product."""
    folder = layout.frames_dir(mol)
    stem = layout.orca_frame_stem(LEVEL, g, b, k)
    (folder / (stem + ".out")).write_text("x\n" + frame_labels.TERMINAL + "\n", encoding="utf-8")
    (folder / (stem + (".hess" if frame_labels.wants_hessian(g) else ".engrad"))).write_text("", encoding="utf-8")


def sbatch_lines(text, header=False):
    """The `sbatch` commands in a text, whitespace-normalised, trailing comments dropped;
    `header`: only the `#     ...` lines of a script's head, not its echo lines."""
    out = set()
    for line in text.splitlines():
        if header and not line.startswith("#     "):
            continue
        s = line.strip().lstrip("#").strip()
        if "sbatch" in s and "hpc/slurm/hl_" in s:
            out.add(re.sub(r"\s+", " ", s.split("#")[0].strip()))
    return out


def main():
    with tempfile.TemporaryDirectory(prefix="hl_campaign_") as tmp:
        root, mol_a = place(tmp, "dsgdb9nsd_000035")
        _r, mol_b = place(tmp, "dsgdb9nsd_000036", with_frames=False)
        d = dataset.datasets_dir(root, TAG, "p")
        d.mkdir(parents=True)
        rows = [dict(qm9_index="dsgdb9nsd_000035", smiles="CCC=O", n_heavy=4, classes="aldehyde", in_training=False, pinned=True),
                dict(qm9_index="dsgdb9nsd_000036", smiles="CCC=O", n_heavy=4, classes="aldehyde;ketone", in_training=False, pinned=False),
                dict(qm9_index="dsgdb9nsd_000100", smiles="C1COC1", n_heavy=4, classes="small_ring", in_training=False, pinned=False)]
        dat.write_table(d / "draw.dat", rows, list(sc.ROW_SCHEMA), sc.ROW_SCHEMA)
        fresh = hp.progress(root, [TAG], "p", LEVEL)
        t = fresh["total"]
        check("a fresh draw: 3 drawn, 2 with branch A, 1 Frame set of 15 frames, none labelled, none running; per class from the draw's classes",
              (t["drawn"], t["branchA"], t["frame_sets"], t["frames"], t["labelled"], t["unlabelled"], t["running"]) == (3, 2, 1, 15, 0, 15, 0)
              and fresh["classes"]["aldehyde"]["drawn"] == 2 and fresh["classes"]["small_ring"]["branchA"] == 0
              and fresh["classes"]["ketone"]["frame_sets"] == 0 and fresh["source"].endswith("draw.dat"), (t, fresh["classes"]))
        # four finished jobs (three basin Hessians, one displaced gradient) and one fresh claim
        for g, b, k in (("basin", 0, 0), ("basin", 1, 0), ("basin", 2, 0), ("displaced", 0, 1)):
            finish(mol_a, g, b, k)
        frame_labels._claim(layout.frames_dir(mol_a), layout.orca_frame_stem(LEVEL, "displaced", 1, 2))
        (layout.frames_dir(mol_a) / (layout.orca_frame_stem(LEVEL, "displaced", 2, 0) + ".out")).write_text("started\n", encoding="utf-8")
        later = hp.progress(root, [TAG], "p", LEVEL)
        t = later["total"]
        check("after four finished file groups, one fresh claim and one .out without the terminal line: labelled 4, failed 1 (that .out), unlabelled 10, running 1 (the claim only)",
              (t["frames"], t["labelled"], t["failed"], t["unlabelled"], t["running"]) == (15, 4, 1, 10, 1)
              and later["classes"]["aldehyde"]["labelled"] == 4 and later["classes"]["ketone"]["labelled"] == 0, t)
        # without a draw: the selection, then the tags
        (d / "draw.dat").unlink()
        dataset.select(root, [TAG], "p", pinned=("dsgdb9nsd_000035",))
        sel = hp.progress(root, [TAG], "p", LEVEL)
        shutil.rmtree(d)
        tags_only = hp.progress(root, [TAG], "p", LEVEL)
        check("without a draw the selection is the list (2 molecules, classes classified: aldehyde), without a selection the tags are",
              sel["molecules"] == 2 and sel["source"].endswith("select.dat") and sel["classes"]["aldehyde"]["drawn"] == 2
              and tags_only["molecules"] == 2 and tags_only["source"] == "the tags" and tags_only["total"]["labelled"] == 4,
              (sel["molecules"], sel["source"], tags_only["molecules"], tags_only["source"]))

    # --- the page and the scripts' headers -----------------------------------------------
    page = sbatch_lines(PAGE.read_text(encoding="utf-8"))
    heads = set()
    for s in SCRIPTS:
        heads |= sbatch_lines((ROOT / "hpc" / "slurm" / s).read_text(encoding="utf-8"), header=True)
    campaign = {l for l in heads if "smoke" not in l and "SPECIES=" not in l and "hl_pipeline_debug" not in l
                and "sbatch hpc/slurm/hl_labels.slurm" != l and "--partition=debug --time=00:30:00 hpc/slurm/hl_labels.slurm" not in l}
    missing = sorted(l for l in campaign if not any(l in p or p in l for p in page))
    foreign = sorted(l for l in page if not any(h in l or l in h for h in heads) and "--time=03:00:00" not in l
                     and "3-00:00:00" not in l)
    check("the campaign page carries every campaign sbatch command of the stage scripts' headers (gate + array), and states no sbatch the headers do not know (the 3-day rounds and the 3 h gate rerun excepted)",
          not missing and not foreign and len(page) >= 6, (missing, foreign, len(page)))
    text = PAGE.read_text(encoding="utf-8")
    check("the page names the progress script, the exit-code signal, the resubmit-as-is rule, TIMEOUT_S = 28800, "
          "the failed state and --retry (ticket 24)",
          all(s in text for s in ("s0_hl_progress.py --tag draw300", "assemble exits 0", "resubmitted **as it is**",
                                  "3-00:00:00", "TIMEOUT_S=28800", "**failed**", "--retry", "touched within 30 min")))
    sequence = ("TIMEOUT_S=14400 TAG=draw300 sbatch --array=0-11 --time=1-00:00:00 hpc/slurm/hl_branchA.slurm",
                "TAG=draw300 sbatch --array=0-1 --time=04:00:00 hpc/slurm/hl_frames.slurm",
                "python workflows/hessian_learning/01_select.py --tag draw300",
                "tmux new -s hl-labels",
                "03_labels.py --tag draw300 --resource tianhe_cpu",
                "--max-blocks 12 --walltime 3-00:00:00",
                "04_dataset.py --tag draw300 --split-by molecule")
    check("ticket 25: the page carries the six-command production sequence (A array, 02 on two nodes, 01, tmux, the parsl driver "
          "with 12 blocks of 3 days, 04), the quota (32 submissions, every array task counted), the tmux gate and its "
          "three outcomes, the sbatch rounds as the fallback",
          all(s in text for s in sequence) and "| tenant `hku2021_fos4`" in text and "every array task as a submission" in text
          and "--resource tianhe_cpu --debug --limit-frames 1" in text and "address_by_interface" in text
          and "AssocMaxSubmitJobLimit" in text and "fallback" in text and "command -v tmux" in text,
          [s for s in sequence if s not in text])

    print("PASS" if not FAIL else "FAIL: " + "; ".join(FAIL))
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
