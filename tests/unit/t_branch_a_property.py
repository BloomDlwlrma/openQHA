"""branchA.toml: branch A's Property file from the pipeline's record (records redesign, ticket 17).

UNIT. No engine; under a second. Needs Python 3.11 (tomllib).

The pipeline builds one record dict in memory (unchanged since step 2) and
`openqha.store.branch_a_property.blocks_from_record` maps it onto the blocks a later
step reads: `[Calculation_Info]`, `[CREST_Run]`, `[Census]`, `[[Basin]]`, `[Criteria]`.
This test feeds a record shaped like the real one (the 2026-09-15 propanal run) and holds:

    A. every key the mapping writes is in the schema (the writer reports none missing)
    B. the readers' questions: relative_kcal, basin_rows (sigma, g0, G - E_el), ALL_PASSED, FAILED
    C. what LEFT the Property file is not in it: no sigma sweep, no thermo breakdown,
       no engine provenance, no criterion detail line, no file list
    D. a basin without RELATIVE raises in relative_kcal (no silent zero)
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
FAIL = []


def check(label, ok, detail=""):
    print("  {:60s} {}".format(label, "ok" if ok else "FAIL " + str(detail)[:300]))
    if not ok:
        FAIL.append(label)


def _basin(i, e, rel, sigma, g_minus):
    return dict(basin_index=i, energy_eV=e, relative_kcal=rel, electronic_degeneracy=1,
                electronic_degeneracy_source="declared_in_config", lowest_frequency_cm_inv=100.0 + i,
                n_imaginary=0,
                symmetry=dict(sigma=sigma, sigma_source="declared_in_config", tolerance_A=0.1,
                              pymsym_point_group="C1", sigma_values_over_sweep=[1, 7],
                              tolerance_sweep={"0.01": dict(sigma=1, n_improper=0), "0.4": dict(sigma=7, n_improper=5)}),
                thermo=dict(temperature_K=298.15, pressure_Pa=1e5, A_minus_Eel_kcal=g_minus - 0.59,
                            pV_kcal=0.59, G_minus_Eel_kcal=g_minus,
                            vibrational=dict(A_vib_kcal=52.1, ZPE_kcal=53.1, n_modes=24),
                            rotational=dict(symmetry_number=sigma, q_rot=40391.5)))


RECORD = dict(
    generated_by="scripts/production/s0_A_pipeline.py", branch="A",
    qm9_index="dsgdb9nsd_000035", label="dsgdb9nsd_000035", name="propanal", smiles="CCC=O",
    tag="propanal", molecule_dir="/r/propanal/1_16000/1_1000/dsgdb9nsd_000035",
    composite_notation="RI-MP2/cc-pVTZ // MACE-OFF23_medium",
    engine=dict(engine="MACE-OFF23_medium", weights_path="/w/MACE-OFF23_medium.model", bytes=18350596,
                torch_version="2.12.1", neighbour_list_patch=dict(applied=True)),
    crest_version=dict(version="3.0.2", commit="6914a25"),
    settings=dict(workhorse="gfn2", refine="sp", runtype="imtd-gc", optlev="tight", shake=2, tstep_fs=5.0,
                  hydrogen_mass_amu=2.0, threads=4, fmax_eV_A=1e-4, dedup_rmsd_A=0.125,
                  dedup_criterion="cregen_three_fold", dedup_ethr_kcal=0.05, dedup_bthr_relative=0.01,
                  hessian_mode="analytic", temperature_K=298.15, symmetry_tolerance_A=0.1),
    gate=dict(smiles="CCC=O", passed=True, reason="passed F0/F1/F3/F4/F5/F7"),
    crest=dict(workdir="/r/.../crest", seconds=224.2, returncode=0, binary="/env/bin/crest",
               terminated_normally=True, n_terminated_early=0, n_conformers=4, energy_spread=0.00197,
               shake_used=2, used_shake_fallback=False, wall_seconds=224.21, reused_scratch=False,
               wall_is_valid_cost=True, products={"crest_conformers.xyz": 2704}),
    census=dict(n_frames_in=5, n_not_converged=0, n_graph_changed=0, max_residual_force_eV_A=8.88e-5,
                n_saddles_rejected=0, n_basins=3, basin_conformer_ids=[0, 1, 3],
                basin_energies_eV=[-5259.19, -5259.15, -5259.15], n_frames_from_crest=4,
                n_reference_geometries_pooled=1,
                hessian={"0": dict(n_imaginary=0, frequencies_cm_inv=[128.4, 231.7])},
                crest_vs_repo=dict(n_conformers_reported_by_crest=4, n_basins_by_repo_criteria=3)),
    machine=dict(cpu_count=16, node="Sherwin"),
    mace=dict(folder="/r/.../mace", basins=["/r/.../mace/basin00/basin.extxyz"]),
    basins=[_basin(0, -5259.1889563698605, 0.0, 1, 36.02523397086022),
            _basin(1, -5259.152703422143, 0.8360128348498219, 1, 35.77002194687515),
            _basin(2, -5259.152703421603, 0.8360128472870755, 3, 35.77002194687515)],
    criteria=[dict(number=1, criterion="workhorse identity", passed=True, detail="asked gfn2, input.toml records gfn2"),
              dict(number=2, criterion="cost claim is unambiguous", passed=True, detail="wall 224.2 s"),
              dict(number=9, criterion="sigma is a group order", passed=False, detail="basin 2 flips at 0.1 A")],
    all_criteria_passed=False, wall_seconds_total=247.77,
)


def main():
    if sys.version_info < (3, 11):
        print("  needs Python 3.11 (tomllib); skipped")
        return 0
    from openqha.store import branch_a_property as bap, property as prop

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "branchA.toml"
        missing = bap.write(path, RECORD)
        text = path.read_text(encoding="utf-8")
        doc = prop.load(path)

        print("A. every key is in the schema:")
        check("no key outside the schema", missing == [], missing)
        check("status block first and NORMAL TERMINATION", text.startswith("[Calculation_Status]")
              and prop.status_of(path) == prop.NORMAL_TERMINATION, text[:120])
        check("block order", list(doc) == ["Calculation_Status", "Calculation_Info", "CREST_Run", "Census", "Basin", "Criteria"],
              list(doc))
        info = doc["Calculation_Info"]
        check("the inputs: settings and molecule", info["WORKHORSE"] == "gfn2" and info["TSTEP"] == 5.0
              and info["HMASS"] == 2.0 and info["SHAKE"] == 2 and info["QM9_INDEX"] == "dsgdb9nsd_000035"
              and info["TAG"] == "propanal" and info["ENGINE"] == "MACE-OFF23_medium", info)
        check("the CREST run", doc["CREST_Run"]["N_CONFORMERS"] == 4 and doc["CREST_Run"]["SHAKE_USED"] == 2
              and doc["CREST_Run"]["USED_SHAKE_FALLBACK"] is False and doc["CREST_Run"]["WALL"] == 224.21, doc["CREST_Run"])
        check("the census", doc["Census"]["N_BASINS"] == 3 and doc["Census"]["BASIN_CONFORMER_IDS"] == [0, 1, 3]
              and doc["Census"]["N_POOLED"] == 1, doc["Census"])

        print("B. what the readers ask:")
        check("relative_kcal in basin order", bap.relative_kcal(doc) == [0.0, 0.8360128348498219, 0.8360128472870755])
        rows = bap.basin_rows(doc)
        check("three Basin blocks with INDEX", [r["INDEX"] for r in rows] == [0, 1, 2])
        check("SIGMA, G0, G_MINUS_EEL per basin", rows[2]["SIGMA"] == 3 and rows[2]["G0"] == 1
              and rows[0]["G_MINUS_EEL"] == 36.02523397086022 and rows[0]["ENERGY"] == -5259.1889563698605, rows[2])
        check("LOWEST_FREQ and N_IMAGINARY", rows[1]["LOWEST_FREQ"] == 101.0 and rows[1]["N_IMAGINARY"] == 0)
        crit = doc["Criteria"]
        check("criteria counts and the failed numbers", crit["N_PASSED"] == 2 and crit["N_TOTAL"] == 3
              and crit["ALL_PASSED"] is False and crit["FAILED"] == [9], crit)

        print("C. what left the Property file:")
        for word in ("tolerance_sweep", "pymsym", "weights_path", "torch_version", "ZPE", "q_rot",
                     "input.toml records", "crest_conformers.xyz", "loadavg", "basin.extxyz",
                     "protocol_source", "composite_notation"):
            check("not in branchA.toml: {}".format(word), word not in text)
        # 79 keys for 3 basins after the engine fingerprint (2) and the dedup map + saddle ids
        # (2) of the Hessian-learning tickets 01/02 (2026-09-18) joined; the bound guards
        # against the provenance sprawl of the old basins.toml, not against a named key
        check("key count is small (about 80 for 3 basins)", 45 <= text.count(" = ") <= 95, text.count(" = "))

        print("D. no silent zero:")
        bad = {"Basin": [{"INDEX": 0, "SIGMA": 1}]}
        try:
            bap.relative_kcal(bad)
            check("a basin without RELATIVE raises", False)
        except KeyError as exc:
            check("a basin without RELATIVE raises", "RELATIVE" in str(exc), exc)

    print()
    print("FAILED: {}".format(FAIL) if FAIL else "all checks passed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
