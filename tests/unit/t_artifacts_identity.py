"""Unit test: `openqha/artifacts.py` refuses to file an artifact without an identity.

UNIT. Seconds; touches no engine and no network.

The point of `write_artifact` is that it can REFUSE. A filing helper that accepts
everything would leave `analysis/` exactly where it was, so most of the cases below
are cases that MUST raise -- a criterion nobody has built a
failing example for has not been shown to be able to fail.

The write cases are redirected into a temporary directory by rebinding
`artifacts.ANALYSIS`. That is the one seam this test needs, and it is deliberate:
the module chooses destinations from that constant precisely so that the choice is
in one place.

Run::  python tests/unit/t_artifacts_identity.py
"""
import sys
import tempfile
from pathlib import Path


def _repo_root():
    """Directory holding the openQHA package, found by walking up from this file."""
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


sys.path.insert(0, str(_repo_root()))
from openqha import artifacts  # noqa: E402

ROOT = _repo_root()
FAIL = []

#: Any file that really exists in the repository works as a stand-in producer.
REAL_PRODUCER = "tests/unit/t_artifacts_identity.py"


def check(label, ok, detail=""):
    print("  {:<62s} {}".format(label, "ok" if ok else "FAIL " + detail))
    if not ok:
        FAIL.append(label)


def raises(label, exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc:
        check(label, True)
        return
    except Exception as e:                       # wrong exception type is still a failure
        check(label, False, "raised {} instead of {}".format(type(e).__name__, exc.__name__))
        return
    check(label, False, "did not raise")


def main():
    print("A. identity is mandatory -- these MUST raise")
    # Omitting a keyword-only argument with no default is a TypeError from Python
    # itself, which means it cannot be forgotten by accident anywhere in the repo.
    raises("category omitted", TypeError, artifacts.write_artifact,
           {"x": 1}, "t_a", status="test", produced_by=REAL_PRODUCER)
    raises("status omitted", TypeError, artifacts.write_artifact,
           {"x": 1}, "t_b", category="calibration", produced_by=REAL_PRODUCER)
    raises("produced_by omitted", TypeError, artifacts.write_artifact,
           {"x": 1}, "t_c", category="calibration", status="test")

    print("B. closed vocabularies -- these MUST raise")
    raises("category not in CATEGORIES", artifacts.ArtifactError, artifacts.write_artifact,
           {"x": 1}, "t_d", category="misc", status="test", produced_by=REAL_PRODUCER)
    raises("status not in STATUSES", artifacts.ArtifactError, artifacts.write_artifact,
           {"x": 1}, "t_e", category="calibration", status="probably-fine",
           produced_by=REAL_PRODUCER)
    raises("decision malformed", artifacts.ArtifactError, artifacts.write_artifact,
           {"x": 1}, "t_f", category="calibration", status="test",
           produced_by=REAL_PRODUCER, decision="see the notes")

    print("C. the producing script must exist on disk -- MUST raise")
    raises("produced_by names a file that is not there", artifacts.ArtifactError,
           artifacts.write_artifact, {"x": 1}, "t_g", category="calibration",
           status="test", produced_by="scripts/production/s0_does_not_exist.py")
    raises("produced_by outside the repository", artifacts.ArtifactError,
           artifacts.write_artifact, {"x": 1}, "t_h", category="calibration",
           status="test", produced_by="/etc/hostname")

    print("D. the caller does not choose the path -- MUST raise")
    raises("name carries a directory", artifacts.ArtifactError, artifacts.write_artifact,
           {"x": 1}, "sub/t_i", category="calibration", status="test",
           produced_by=REAL_PRODUCER)
    raises("subdir escapes upward", artifacts.ArtifactError, artifacts.destination,
           "t_j", category="calibration", status="test", subdir="../../etc")

    print("E. format is decided by the payload, not by the caller")
    saved = artifacts.ANALYSIS
    tmp = tempfile.mkdtemp(prefix="openqha_artifacts_")
    artifacts.ANALYSIS = Path(tmp)
    try:
        p = artifacts.write_artifact({"nu_cm_inv": 13.94}, "json_case",
                                     category="calibration", status="test",
                                     produced_by=REAL_PRODUCER, decision="S0-D-10")
        check("dict -> .json", p.suffix == ".json", str(p))
        meta = artifacts.read_meta(p)
        check("json metadata round-trips",
              meta is not None and meta["status"] == "test"
              and meta["generated_by"] == REAL_PRODUCER and meta["decision"] == "S0-D-10",
              repr(meta))
        check("json payload survives untouched",
              artifacts.read_artifact(p) == {"nu_cm_inv": 13.94},
              repr(artifacts.read_artifact(p)))

        p = artifacts.write_artifact("first line\nsecond line\n", "text_case",
                                     category="diagnostics", status="test",
                                     produced_by=REAL_PRODUCER)
        check("str -> .log", p.suffix == ".log", str(p))
        check("log metadata round-trips",
              (artifacts.read_meta(p) or {}).get("category") == "diagnostics")

        try:
            import pandas as pd
            df = pd.DataFrame({"qm9_index": ["dsgdb9nsd_000506"], "freq_cm_inv": [13.94]})
            p = artifacts.write_artifact(df, "frame_case", category="production",
                                         status="production", produced_by=REAL_PRODUCER)
            check("DataFrame -> .parquet, never .json", p.suffix == ".parquet", str(p))
            check("parquet metadata rides in the schema",
                  (artifacts.read_meta(p) or {}).get("status") == "production")
        except ImportError:
            print("  {:<62s} skipped (pandas absent)".format("DataFrame -> .parquet"))

        raises("payload of an unsupported type", artifacts.ArtifactError,
               artifacts.write_artifact, 3.14, "float_case", category="calibration",
               status="test", produced_by=REAL_PRODUCER)

        print("F. overwrite is deliberate, never silent")
        raises("second write without exist_ok", artifacts.ArtifactError,
               artifacts.write_artifact, {"nu_cm_inv": 0.0}, "json_case",
               category="calibration", status="test", produced_by=REAL_PRODUCER)
        p = artifacts.write_artifact({"nu_cm_inv": 0.0}, "json_case",
                                     category="calibration", status="test",
                                     produced_by=REAL_PRODUCER, exist_ok=True)
        check("exist_ok=True replaces it",
              artifacts.read_artifact(p) == {"nu_cm_inv": 0.0})

        print("G. routing follows identity")
        d = artifacts.destination("x", category="production", status="superseded")
        check("status superseded overrides the category directory",
              d.parent.name == "_superseded", str(d))
        d = artifacts.destination("x", category="raw", status="test")
        check("category raw lands in _raw", d.parent.name == "_raw", str(d))
        d = artifacts.destination("x", category="calibration", status="calibration",
                                  subdir="branchA_workhorse")
        check("subdir nests under the category",
              d.parent.name == "branchA_workhorse"
              and d.parent.parent.name == "calibration", str(d))
    finally:
        artifacts.ANALYSIS = saved

    print()
    if FAIL:
        print("{} case(s) failed: {}".format(len(FAIL), ", ".join(FAIL)))
        return 1
    print("all cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
