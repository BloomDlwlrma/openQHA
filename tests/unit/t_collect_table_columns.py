"""Collect's Table explains every column it writes (ticket 02, collect-one-table, 2026-09-16).

UNIT. No engine; under a second.

The ruling (user, 2026-09-16, ADR 0003 amendment): collect leaves one `collect.dat` with
the sections `trajectories`, `blank`, `assembly`, always all three, and above each header
one comment per column from `chain_records.COLUMNS`. The seam: the rows collect builds
(`table_sections` in the analyse driver, fed the dict shapes `analyse_one`, the blank
control and the assembly produce) go through `chain_records.write_collect_table`, which
returns every column the schema does not explain and every schema column the rows lack.
That list must be empty; the file must carry the three sections in order with a comment
for each header column; a section with no rows still has its header and comments.
"""
import importlib.util
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
FAIL = []


def check(label, ok, detail=""):
    print("  {:60s} {}".format(label, "ok" if ok else "FAIL " + str(detail)[:300]))
    if not ok:
        FAIL.append(label)


def _load_driver():
    path = ROOT / "scripts" / "production" / "s0_B_qha_analyse.py"
    spec = importlib.util.spec_from_file_location("s0_B_qha_analyse", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_trajectory(basin):
    """The shape analyse_one returns, with only the keys table_sections reads."""
    return dict(
        basin=basin, seed="seed00", path="/x/_records/md_ase/" + basin,
        analysis=dict(
            rank_check=dict(n_frames=5),
            entropy=dict(TS_QH_kcal=1.5, TS_Schlitter_kcal=1.7, S_QH_kcal_per_K=0.005, A_vib_kcal=8.0,
                         lowest_frequency_cm_inv=110.0, highest_frequency_cm_inv=3000.0),
            spectrum=dict(n_nonzero_eigenvalues=4, expected_vibrational_modes=24,
                          rigid_to_first_vibrational_ratio=float("inf"))),
        saturation=dict(increment_over_last_doubling_kcal=0.0),
        meta=dict(production=dict(wall_seconds=533.0, seconds_per_ps_this_run=106.6)))


def main():
    from openqha.quasi_harmonic import chain_records
    from openqha.store import dat
    drv = _load_driver()
    per_traj = [_fake_trajectory("basin00"), _fake_trajectory("basin01")]
    blank = [dict(basin="basin00", n_seeds=1, TS_QH_mean_kcal=1.5, TS_QH_spread_kcal=0.0,
                  TS_QH_rms_about_mean_kcal=0.0, standard_error_kcal=None,
                  values_kcal=[1.5], note="one seed per basin")]
    assembly = [dict(basin="basin00", G_minus_Eel_kcal=-10.0, A_vib_kcal=8.0, S_vib_kcal_per_K=0.005,
                     consistency=dict(all_identical=True))]

    print("A. the rows collect builds against the schema")
    sections = drv.table_sections("dsgdb9nsd_000035", per_traj, blank, assembly)
    check("three sections, the schema's, in order", list(sections) == list(chain_records.SECTIONS), list(sections))
    for s in chain_records.SECTIONS:
        check("every column of {} is explained and every explained column is written".format(s),
              sections[s] and list(sections[s][0].keys()) == list(chain_records.COLUMNS[s]),
              (list(sections[s][0].keys()) if sections[s] else None, list(chain_records.COLUMNS[s])))
    check("values_kcal (a list) is not a column of blank", "values_kcal" not in sections["blank"][0])

    print("B. the file")
    with tempfile.TemporaryDirectory() as t:
        stem = Path(t) / "collect"
        drift = chain_records.write_collect_table(stem, sections)
        check("the writer reports no drift", drift == [], drift)
        paths = chain_records.collect_paths(stem)
        check("collect_paths gives out, toml, dat and nothing else", sorted(paths) == ["dat", "out", "toml"], paths)
        check("the Table is collect.dat", paths["dat"].name == "collect.dat" and paths["dat"].is_file(), paths["dat"])
        lines = paths["dat"].read_text(encoding="utf-8").splitlines()
        heads = [l for l in lines if l.startswith("[")]
        check("[trajectories] [blank] [assembly] in that order", heads == ["[trajectories]", "[blank]", "[assembly]"], heads)
        for s in chain_records.SECTIONS:
            i = lines.index("[{}]".format(s))
            block = []
            for l in lines[i + 1:]:
                if l.startswith("[") or not l.strip():
                    break
                block.append(l)
            comments = [l for l in block if l.startswith("#   ")]
            header = [l for l in block if l.startswith("# ") and not l.startswith("#   ")]
            check("{}: one comment per header column, header last".format(s),
                  len(header) == 1 and header[0].split()[1:] == list(chain_records.COLUMNS[s])
                  and [c.split()[1] for c in comments] == list(chain_records.COLUMNS[s])
                  and block.index(header[0]) == len(comments),
                  block[:len(comments) + 1])
            check("{}: the comments carry Type, unit: doc".format(s),
                  all(": " in c.split(None, 2)[2] for c in comments), comments[:2])
        back = chain_records.read_collect_table(stem)
        check("read back: two trajectories, one blank, one assembly row",
              [len(back[s]) for s in chain_records.SECTIONS] == [2, 1, 1], {k: len(v) for k, v in back.items()})
        check("TS_QH_kcal and inf survive", back["trajectories"][0]["TS_QH_kcal"] == 1.5
              and back["trajectories"][0]["rigid_ratio"] == float("inf"), back["trajectories"][0])

        print("C. a section with no rows")
        empty = drv.table_sections("x", per_traj, [], [])
        drift = chain_records.write_collect_table(Path(t) / "e", empty)
        check("no drift for empty blank and assembly", drift == [], drift)
        back = chain_records.read_collect_table(Path(t) / "e")
        check("all three sections present, the empty ones as []",
              list(back) == list(chain_records.SECTIONS) and back["blank"] == [] and back["assembly"] == [], back)
        text = (Path(t) / "e.dat").read_text(encoding="utf-8")
        check("the empty section still has its header and comments",
              "# species basin G_minus_Eel_kcal A_vib_kcal S_vib_kcal_per_K terms_match_hessian_route" in text
              and "#   TS_QH_rms_about_mean_kcal" in text, text)

        print("D. drift is reported")
        bad = dict(sections)
        bad["assembly"] = [dict(assembly[0], **{"species": "x", "basin": "basin00", "new_column": 1})]
        bad["assembly"][0].pop("consistency")
        drift = chain_records.write_collect_table(Path(t) / "d", bad)
        check("an unexplained column is reported by section and name", ("assembly", "new_column") in drift, drift)
        check("a missing explained column is reported too",
              any(c in ("G_minus_Eel_kcal", "terms_match_hessian_route") for s, c in drift), drift)
    print("\n{}".format("PASS" if not FAIL else "FAIL: " + "; ".join(FAIL)))
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
