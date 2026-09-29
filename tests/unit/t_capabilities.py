"""The capability contract: locally skip, but never skip what you declared you provide.

The arrangement is ACEsuit/mace's (`tests/conftest.py`, `tests/extensions/`,
`.github/workflows/ci-extensions.yaml`) and so is the rule that matters:

    locally, work whose capability is unavailable is SKIPPED;
    a run that DECLARES it via S0_REQUIRE_CAPS FAILS instead.

openQHA needs the second half for a specific, twice-repeated reason. Acceptance criterion
2 reported "no comparison produced" once because GROMACS was installed in a sibling conda
environment, and once because `--no-gmx` was passed. Both times a criterion that exists to
catch an error in the covariance spectrum went quiet instead of red, and nothing in the
product distinguished "checked and agreed" from "did not check".

This file tests the mechanism, not the capabilities. It must pass on a machine with none
of them installed and on a machine with all of them.
"""
import os
import sys
from pathlib import Path


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


sys.path.insert(0, str(_repo_root()))
from openqha import capabilities as cap                      # noqa: E402


def main():
    print("=" * 92)
    print("The capability contract")
    print("=" * 92)
    print(cap.summary())
    print()

    checks = []

    # ---- the probes are real, and they answer ----------------------------------------
    st = cap.state(refresh=True)
    checks.append((
        "every capability probes to a bool, with no exception escaping",
        "{} capabilities: {}".format(len(st), ", ".join(sorted(st))),
        all(isinstance(v, bool) for v in st.values()) and set(st) == set(cap.ALL)))

    # A probe must use a REAL import or a REAL execution. `find_spec` would call a broken
    # wheel available -- upstream's note is "a broken wheel (e.g. missing libmpi) must
    # read as unavailable" -- so a probe that merely looks for a file is a bug.
    # Look for a CALL, not for the word: the module docstring names find_spec in order
    # to say it is not used, and a substring search flagged that -- a check that fires on
    # its own explanation is a check nobody will keep.
    import ast
    tree = ast.parse((_repo_root() / "openqha" / "capabilities.py")
                     .read_text(encoding="utf-8"))
    called = sorted({
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Attribute, ast.Name))})
    banned = [n for n in ("find_spec", "exists", "is_file") if n in called]
    checks.append((
        "no probe CALLS find_spec or a bare path test: a present-but-broken install must "
        "read as UNAVAILABLE, and only using the thing can tell",
        "banned calls found = {}".format(banned or "none"),
        not banned))

    # ---- core and extensions are disjoint, and every extension says what is lost ------
    overlap = sorted(set(cap.CORE) & set(cap.EXTENSIONS))
    checks.append((
        "core and extension are disjoint: a capability cannot be both required and "
        "optional",
        "overlap = {}".format(overlap or "none"),
        not overlap))
    undocumented = sorted(n for n in cap.EXTENSIONS if not cap.EXTENSIONS[n].strip())
    checks.append((
        "every extension states what its absence costs, so 'optional' is never a shrug",
        "undocumented = {}".format(undocumented or "none"),
        not undocumented))

    # ---- S0_REQUIRE_CAPS ---------------------------------------------------------------
    saved = os.environ.get("S0_REQUIRE_CAPS")
    try:
        # An unknown name is an error. A typo that quietly guarantees nothing is exactly
        # the failure this variable exists to prevent.
        os.environ["S0_REQUIRE_CAPS"] = "gromacs,not_a_capability"
        raised = False
        try:
            cap.required_caps()
        except ValueError:
            raised = True
        checks.append((
            "S0_REQUIRE_CAPS rejects an unknown capability rather than guaranteeing "
            "nothing",
            "raised = {}".format(raised), raised))

        # Declaring something absent must FAIL. Pick a capability that is genuinely
        # unavailable here; if everything is installed, fabricate the condition by
        # declaring one and monkeypatching its probe to False, so the test still tests.
        absent = next((n for n in cap.ALL if not cap.available(n)), None)
        fabricated = False
        if absent is None:
            absent = "gromacs"
            fabricated = True
            cap._CACHE[absent] = False                        # noqa: SLF001
        os.environ["S0_REQUIRE_CAPS"] = absent
        failed = False
        try:
            cap.check_declared()
        except RuntimeError:
            failed = True
        checks.append((
            "declaring an ABSENT capability FAILS -- this is the half that stops a job "
            "going green while skipping the thing it exists to test",
            "declared {!r}{}: raised = {}".format(
                absent, " (fabricated: everything is installed here)" if fabricated
                else "", failed),
            failed))
        if fabricated:
            cap._CACHE.pop(absent, None)                      # noqa: SLF001

        # Declaring something present must pass.
        present = next((n for n in cap.ALL if cap.available(n)), None)
        if present:
            os.environ["S0_REQUIRE_CAPS"] = present
            ok = False
            try:
                ok = cap.check_declared() == [present]
            except RuntimeError:
                ok = False
            checks.append((
                "declaring a PRESENT capability passes and returns it",
                "declared {!r}: {}".format(present, ok), ok))

        # Nothing declared is the local default: skipping is allowed.
        os.environ.pop("S0_REQUIRE_CAPS", None)
        checks.append((
            "with nothing declared, nothing is required: locally a missing capability "
            "is a skip",
            "required = {}".format(cap.check_declared()),
            cap.check_declared() == []))
    finally:
        if saved is None:
            os.environ.pop("S0_REQUIRE_CAPS", None)
        else:
            os.environ["S0_REQUIRE_CAPS"] = saved

    # ---- require_core names the fix ---------------------------------------------------
    missing_core = [n for n in cap.CORE if not cap.available(n)]
    if missing_core:
        message = ""
        try:
            cap.require_core()
        except RuntimeError as exc:
            message = str(exc)
        checks.append((
            "a missing CORE capability names the environment file that provides it, "
            "rather than only the package",
            "mentions environment.yml = {}".format(
                "environment.yml" in message),
            "environment.yml" in message))
    else:
        checks.append((
            "every CORE capability is present here, so require_core passes",
            "core = {}".format(", ".join(cap.CORE)), cap.require_core()))

    bad = 0
    for text, measured, ok in checks:
        print("[{}] {}".format("PASS" if ok else "FAIL", text))
        print("       measured: {}".format(measured))
        bad += 0 if ok else 1
    print()
    print("{} of {} checks failed".format(bad, len(checks)))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
