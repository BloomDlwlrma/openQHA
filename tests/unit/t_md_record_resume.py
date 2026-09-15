"""A finished trajectory is recognised from its record md.toml and its DCD (ticket 12).

UNIT. numpy, a temporary directory and the OpenMM driver's `already_complete`; under a
second. Replaces t_frames_flush.py (2026-09-13), whose subject -- the atomic flush of
`frames.npy` -- no longer exists: since 2026-09-15 the trajectory is `traj.dcd` and the
record is `md.toml`, and nothing is written twice.

The rule `already_complete` holds (an113, 2026-09-13, the resume path that had never
executed): a (basin, setting) is finished when md.toml is there, its production block says
complete, and the DCD holds at least n_target frames. Anything less -- a partial run, an
older protocol wanting more frames, a missing or unreadable record, a DCD shorter than
the record claims -- is not, and the driver resumes it.
"""
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "production"))
FAIL = []


def fake_dcd(path, n_frames):
    """Only the header matters to dcd_frame_count: 'CORD' then the frame count."""
    head = struct.pack("i", 84) + b"CORD" + struct.pack("i", int(n_frames)) + b"\0" * 80
    path.write_bytes(head)


def main():
    from s0_B_qha_trajectory_openmm import already_complete
    from openqha.store import toml_out

    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)

        def make(name, n_frames, complete, write_meta=True, meta_text=None):
            rec = tmp / name / "_records" / "md_openmm" / "default" / "basin00"
            eng = tmp / name / "md_openmm" / "basin00"
            rec.mkdir(parents=True)
            eng.mkdir(parents=True)
            fake_dcd(eng / "traj.dcd", n_frames)
            if write_meta:
                if meta_text is not None:
                    (rec / "md.toml").write_text(meta_text, encoding="utf-8")
                else:
                    toml_out.dump(dict(seed=7, production=dict(complete=complete, n_frames=n_frames,
                                                               temperature_mean_K=298.0)),
                                  rec / "md.toml")
            return rec, eng

        print("already_complete(records_dir, n_target, engine_dir) reads md.toml and the DCD header")
        cases = [
            ("finished, exact frames",     make("ok",      5, True),                    5, True),
            ("finished, extra frames",     make("more",    8, True),                    5, True),
            ("partial: DCD too short",     make("short",   3, True),                    5, False),
            ("not marked complete",        make("running", 5, False),                   5, False),
            ("no md.toml",                 make("nometa",  5, True, write_meta=False),  5, False),
            ("md.toml unreadable",         make("broken",  5, True, meta_text="= not toml"), 5, False),
            ("older protocol, needs more", make("old",     5, True),                    10, False),
        ]
        for label, (rec, eng), n_target, want in cases:
            got = already_complete(rec, n_target, engine_dir=eng, setting="default") is not None
            ok = got == want
            print("   {}  {:30s} n_target {:>2} -> {}".format("ok  " if ok else "FAIL", label, n_target,
                                                             "complete" if got else "not complete"))
            if not ok:
                FAIL.append("already_complete({}, {}) = {}, expected {}".format(label, n_target, got, want))
        rec, eng = tmp / "ok" / "_records" / "md_openmm" / "default" / "basin00", tmp / "ok" / "md_openmm" / "basin00"
        meta = already_complete(rec, 5, engine_dir=eng, setting="default")
        if not (meta and meta["production"]["temperature_mean_K"] == 298.0 and meta["seed"] == 7):
            FAIL.append("already_complete did not return the md.toml content")
        print("   {}  the record it returns is md.toml's own".format("ok  " if meta and meta["seed"] == 7 else "FAIL"))

    print()
    if FAIL:
        print("{} problem(s):".format(len(FAIL)))
        for f in FAIL:
            print("  - " + f)
        return 1
    print("a finished trajectory is recognised from md.toml and traj.dcd, and nothing less is")
    return 0


if __name__ == "__main__":
    sys.exit(main())
