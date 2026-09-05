"""Gate F7 -- QM9's official "uncharacterized" list.

UNIT. Seconds; reads files only.

What is checked, one line each, and a failure is reported rather than raised:

1. the list parses to **exactly 3054** entries, and the file's sha256 matches the pinned
   value;
2. `dsgdb9nsd_003838` is on it, and the SMILES the list records for the deposited B3LYP
   geometry contains a dot (two fragments);
3. that molecule is rejected by F7, and the reason names BOTH SMILES;
4. an ordinary molecule (`dsgdb9nsd_000010`, CC#N) passes F7;
5. **enabling F7 without an identifier raises ValueError** -- silently skipping a gate
   that was switched on is not allowed;
6. F7 is last in GATE_ORDER, so the existing F0-F6 attributions are unaffected;
7. `f7_scope` -- and this one depends on data the repository does not ship. See below.

WHY THE SCOPE CASES ARE CONDITIONAL, AND WHY THAT IS NOT A DODGE
---------------------------------------------------------------
`f7_scope="identity_from_geometry_only"` keeps a listed molecule when the index column
`qm9_identity_source` says `relaxed` -- meaning the row's SMILES was itself parsed from
the B3LYP geometry, so "geometry disagrees with SMILES" does not apply to us.

Reading that column needs the QM9 index, which is 119 MB and **not shipped**. Without it
`_identity_source` returns "" for every molecule and F7 drops them all -- the documented
conservative direction, and identical in behaviour to `f7_scope="all"`.

Until 2026-09-05 this file asserted the scope behaviour anyway. One case failed
(`000080` was expected to survive) and the other **passed for the wrong reason**:
`003838` was dropped because its row was missing, not because its identity source is
`gdb17`. Two assertions, neither testing what it named.

So the scope block now asks `filters.identity_source_available()` first. With the index
present it tests the classification. Without it, it tests the FALLBACK -- that an
unavailable index makes F7 drop rather than keep -- and reports the classification cases
as NOT TESTED, naming what would have to be present to test them. A test that quietly
passes when its subject is absent is worse than not having it.

Run::  python tests/unit/t_filters_f7.py
"""
import sys
from pathlib import Path


def _repo_root():
    """Directory holding the openQHA package, found by walking up from this file.

    Depth-independent on purpose: this file keeps working wherever under the repository
    it is moved to. An earlier move into `scripts/_superseded/` broke every `parents[1]`
    in the moved files silently, which is what this replaces.
    """
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


sys.path.insert(0, str(_repo_root()))
from openqha import filters, qm9_uncharacterized  # noqa: E402

FAIL = []
NOT_TESTED = []


def check(name, cond, detail=""):
    mark = "pass" if cond else "**FAIL**"
    print("  [{}] {}{}".format(mark, name, ("   " + detail) if detail else ""))
    if not cond:
        FAIL.append(name)


def not_tested(name, why):
    print("  [n/a ] {}   {}".format(name, why))
    NOT_TESTED.append("{} -- {}".format(name, why))


def main():
    print("Gate F7 -- QM9's official uncharacterized list")
    print("-" * 88)

    rows, prov = qm9_uncharacterized.load()
    check("the list has 3054 entries", len(rows) == 3054,
          "measured {}".format(len(rows)))
    check("sha256 matches the pinned value", prov["sha256_matches_recorded"],
          prov["sha256"][:16] + "...")
    check("how many genuinely disagree has been counted",
          prov["n_deposited_geometry_disagrees_with_smiles"] > 0,
          "{} disagree / {} differ only in Corina's guess".format(
              prov["n_deposited_geometry_disagrees_with_smiles"],
              prov["n_only_corina_disagrees"]))

    e = qm9_uncharacterized.entry("dsgdb9nsd_003838")
    check("dsgdb9nsd_003838 is on the list", e is not None)
    if e is not None:
        check("the SMILES parsed from its B3LYP geometry contains a dot "
              "(i.e. several fragments)",
              "." in e["smiles_b3lyp_xyz"], repr(e["smiles_b3lyp_xyz"]))
        check("an integer index and a string index find the same row",
              qm9_uncharacterized.entry(3838) == e)

    gates = ("F0", "F1", "F3", "F4", "F5", "F7")
    ok, gate, why = filters.screen("N=C1N=CON=N1", gates=gates,
                                   identifier="dsgdb9nsd_003838")
    check("003838 is rejected by F7", (not ok) and gate == "F7", why[:100])
    check("the reason names both SMILES",
          ("N=C1N=CON=N1" in why) and ("N#N" in why))

    ok2, gate2, _ = filters.screen("CC#N", gates=gates,
                                   identifier="dsgdb9nsd_000010")
    check("an ordinary molecule, dsgdb9nsd_000010, passes F7", ok2 and gate2 is None)

    try:
        filters.screen("CC#N", gates=("F7",), identifier=None)
        raised = False
    except ValueError:
        raised = True
    check("F7 enabled without an identifier raises ValueError "
          "(it does not skip silently)", raised)

    # ---- f7_scope -------------------------------------------------------------------
    import copy
    from openqha import config as _cfg
    base = _cfg.load()
    c_all = copy.deepcopy(base)
    c_all["species_filter"]["f7_scope"] = "all"
    c_geo = copy.deepcopy(base)
    c_geo["species_filter"]["f7_scope"] = "identity_from_geometry_only"

    have_index = filters.identity_source_available()
    print("  ---- f7_scope (QM9 index available: {}) ----".format(have_index))

    r_all = filters.screen("N=C1N=CON=N1", c_all, gates, identifier="dsgdb9nsd_003838")
    check("003838 is dropped under f7_scope=all", r_all[1] == "F7")

    if have_index:
        # 003838's index identity source is gdb17 (its relaxed SMILES is two fragments,
        # so the index fell back to the GDB17 string) -- BOTH scopes must drop it.
        r_geo = filters.screen("N=C1N=CON=N1", c_geo, gates,
                               identifier="dsgdb9nsd_003838")
        check("003838 is dropped under f7_scope=identity_from_geometry_only too",
              r_geo[1] == "F7", "its identity source is gdb17, not relaxed")

        # 000080: GDB17 has the glycine zwitterion, the B3LYP geometry is neutral
        # glycine, and the index adopted the relaxed (geometry-derived) SMILES -- so for
        # us it is self-consistent and this scope must KEEP it.
        a = filters.screen("NCC(=O)O", c_all, gates, identifier="dsgdb9nsd_000080")
        g = filters.screen("NCC(=O)O", c_geo, gates, identifier="dsgdb9nsd_000080")
        check("000080 is dropped under f7_scope=all", a[1] == "F7")
        check("000080 SURVIVES under f7_scope=identity_from_geometry_only",
              g[0] is True, "the index uses the relaxed SMILES, which its geometry matches")
    else:
        # The classification cannot be exercised. What CAN be exercised is the fallback,
        # and it is the more important of the two: not knowing must mean dropping.
        g = filters.screen("NCC(=O)O", c_geo, gates, identifier="dsgdb9nsd_000080")
        check("with no index, identity_from_geometry_only still DROPS a listed molecule "
              "(conservative fallback)", g[1] == "F7",
              "not knowing the identity source must never mean keeping it")
        check("_identity_source reports '' when the index is unavailable",
              filters._identity_source("dsgdb9nsd_000080") == "")
        not_tested("003838 / 000080 classified by qm9_identity_source",
                   "needs data/qm9/index_Chem_composition.csv (119 MB, not shipped); "
                   "run scripts/tooling/s0_prepare_data.py to test it")

    try:
        bad = copy.deepcopy(base)
        bad["species_filter"]["f7_scope"] = "whatever"
        filters.screen("NCC(=O)O", bad, gates, identifier="dsgdb9nsd_000080")
        raised2 = False
    except ValueError:
        raised2 = True
    check("a third value for f7_scope raises ValueError "
          "(it does not fall back to the default silently)", raised2)

    check("F7 is last in GATE_ORDER", filters.GATE_ORDER[-1] == "F7",
          str(filters.GATE_ORDER))
    check("F7 has a config key and a description",
          "F7" in filters.GATE_DESCRIPTION and "F7" in filters._CONFIG_KEY)

    print("-" * 88)
    if NOT_TESTED:
        print("NOT TESTED here, and why:")
        for n in NOT_TESTED:
            print("  - {}".format(n))
    if FAIL:
        print("**{} failure(s)**: {}".format(len(FAIL), "; ".join(FAIL)))
        return 1
    print("all checks passed"
          + ("  ({} case(s) not testable without the QM9 index)".format(len(NOT_TESTED))
             if NOT_TESTED else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
