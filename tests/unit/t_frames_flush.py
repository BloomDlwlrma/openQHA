"""The trajectory driver's atomic frame flush must actually produce frames.npy.

UNIT. NumPy and a temporary directory; well under a second.

The defect (an104, 2026-09-13)
------------------------------
Six of six branch B trajectories on the A100 died at the FIRST flush with

    FileNotFoundError: '.../frames.npy.part' -> '.../frames.npy'

leaving `frames.npy.part.npy` beside each. `np.save(path, arr)` appends `.npy` to a name
that lacks it, so the temp file was never at the name the rename looked for. The code had
existed for days and never run: every earlier GPU trajectory had died before reaching a
flush (PTX, then the device constant), and the CPU route has its own driver. A defect in
a line that has never executed is invisible to every test that does not execute it --
which is what this one is for.

Part A reproduces the old form and asserts that it FAILS, so the test cannot be satisfied
by a NumPy that changed its mind about suffixes. Part B runs the shipped `flush_frames`
on a fresh directory, then on a resume, and checks: `frames.npy` exists, round-trips, no
`.part` file of any spelling remains, and the stale `frames.npy.part.npy` a killed run of
the old code leaves is not mistaken for anything.

Part C (same day, next failure) is `already_complete`: re-running the chain over six
finished trajectories made run_one() skip its production loop, leave the temperature
None, and die formatting it -- after ~70 s per task of re-tracing and re-equilibrating
something it would not touch. A finished (basin, seed) must be recognised from its own
meta.json and frames.npy, before any model is loaded, and anything short of finished
(partial frames, an older protocol wanting more frames, a missing or unreadable record)
must NOT be.
"""
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
sys.path.insert(0, str(ROOT / "scripts" / "production"))
import numpy as np                                                    # noqa: E402

FAIL = []


def part_a_old_form_fails(tmp):
    print("A. the old form: np.save appends .npy, so the rename has nothing to rename")
    frames_path = tmp / "a" / "frames.npy"
    frames_path.parent.mkdir(parents=True)
    bad = frames_path.with_suffix(frames_path.suffix + ".part")     # frames.npy.part
    np.save(bad, np.zeros((2, 3, 3)))
    written = sorted(p.name for p in frames_path.parent.iterdir())
    print("   np.save({!r}) wrote        {}".format(bad.name, written))
    try:
        bad.replace(frames_path)
        FAIL.append("the old form did NOT fail; this test's premise is wrong for this numpy")
        print("   FAIL  rename succeeded -- premise broken")
    except FileNotFoundError:
        print("   ok    rename raised FileNotFoundError, as on an104")
    if "frames.npy.part.npy" not in written:
        FAIL.append("expected the .part.npy leftover, got {}".format(written))


def part_b_shipped_form_works(tmp):
    print("\nB. the shipped flush_frames: fresh write, then a resume, nothing left behind")
    from s0_B_qha_trajectory_openmm import flush_frames                # the real function
    d = tmp / "b"
    d.mkdir()
    frames_path = d / "frames.npy"
    # A killed run of the OLD code leaves this; the driver removes it on resume, and this
    # flush must at least never read or rename it.
    (d / "frames.npy.part.npy").write_bytes(b"not an array")

    first = np.arange(2 * 4 * 3, dtype=float).reshape(2, 4, 3)
    out = flush_frames(frames_path, list(first))
    back = np.load(frames_path)
    print("   fresh   -> {}  shape {}  round-trip {}".format(
        Path(out).name, back.shape, "ok" if np.array_equal(back, first) else "MISMATCH"))
    if not np.array_equal(back, first):
        FAIL.append("fresh flush did not round-trip")

    second = np.concatenate([first, first + 100.0])
    flush_frames(frames_path, list(second))
    back = np.load(frames_path)
    print("   resume  -> shape {}  round-trip {}".format(
        back.shape, "ok" if np.array_equal(back, second) else "MISMATCH"))
    if not np.array_equal(back, second):
        FAIL.append("resumed flush did not round-trip")

    leftovers = sorted(p.name for p in d.iterdir() if p.name != "frames.npy")
    print("   files beside frames.npy   {}".format(leftovers or "(none)"))
    ours = [n for n in leftovers if n.endswith(".part.npy") and n != "frames.npy.part.npy"]
    if ours:
        FAIL.append("flush_frames left its own temp file behind: {}".format(ours))
    if not (d / "frames.npy.part.npy").exists():
        FAIL.append("flush_frames touched the old-code leftover; that is run_one's job, on resume")


def part_c_already_complete(tmp):
    print("\nC. already_complete reads the record on disk, and only a finished one counts")
    import json
    from s0_B_qha_trajectory_openmm import already_complete

    def make(name, n_frames, complete, write_meta=True, meta_text=None):
        d = tmp / "c" / name
        d.mkdir(parents=True)
        np.save(d / "frames.npy", np.zeros((n_frames, 4, 3)))
        if write_meta:
            (d / "meta.json").write_text(
                meta_text if meta_text is not None else json.dumps(
                    dict(seed=7, production=dict(complete=complete, n_frames=n_frames,
                                                 temperature_mean_K=298.0))),
                encoding="utf-8")
        return d

    cases = [
        ("finished, exact frames",   make("ok",      5, True),                5, True),
        ("finished, extra frames",   make("more",    8, True),                5, True),
        ("partial: too few frames",  make("short",   3, True),                5, False),
        ("not marked complete",      make("running", 5, False),               5, False),
        ("no meta.json",             make("nometa",  5, True, write_meta=False), 5, False),
        ("meta.json unreadable",     make("broken",  5, True, meta_text="{not json"), 5, False),
        ("older protocol, needs more", make("old",   5, True),               10, False),
    ]
    for label, d, n_target, want in cases:
        got = already_complete(d, n_target) is not None
        ok = got == want
        print("   {}  {:28s} n_target {:>2} -> {}".format("ok  " if ok else "FAIL", label, n_target,
                                                         "complete" if got else "not complete"))
        if not ok:
            FAIL.append("already_complete({}, {}) = {}, expected {}".format(label, n_target, got, want))
    # and the record it returns is the file's own
    meta = already_complete(tmp / "c" / "ok", 5)
    if not (meta and meta["production"]["temperature_mean_K"] == 298.0 and meta["seed"] == 7):
        FAIL.append("already_complete did not return the meta.json content")


def main():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        part_a_old_form_fails(tmp)
        part_b_shipped_form_works(tmp)
        part_c_already_complete(tmp)
    print()
    if FAIL:
        print("{} problem(s):".format(len(FAIL)))
        for f in FAIL:
            print("  - " + f)
        return 1
    print("frame flush is atomic, and a finished trajectory is recognised from disk")
    return 0


if __name__ == "__main__":
    sys.exit(main())
