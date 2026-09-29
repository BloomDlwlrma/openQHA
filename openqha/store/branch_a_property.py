"""Branch A's Property file, `_records/branchA.toml` (records redesign, 2026-09-15).

What a later step reads about branch A, and nothing else:

    [Calculation_Status]   PROGNAME VERSION STATUS
    [Calculation_Info]     the molecule and every setting the run was given
    [CREST_Run]            how CREST ended, what it reported, which SHAKE actually ran
    [Census]               tightening, deduplication, the basin count
    [[Basin]]              INDEX ENERGY RELATIVE SIGMA G0 N_IMAGINARY LOWEST_FREQ N_INVERSION_WINDOW G_MINUS_EEL A_MINUS_EEL
    [Criteria]             N_PASSED N_TOTAL ALL_PASSED FAILED

The readers: the ensemble report (RELATIVE, SIGMA, G0, G_MINUS_EEL per basin), 02a, 02c,
02d (the same), the branch A parsl driver (STATUS, ALL_PASSED, FAILED, the CREST
counts). Provenance, the criterion detail lines, the sigma tolerance sweep and the
thermochemistry breakdown are in `branchA.out`, the Report.

`blocks_from_record(record)` takes the record the pipeline builds in memory (unchanged
since step 2; `tests/unit/t_branch_a_property.py` holds the mapping) so the pipeline's
physics code does not learn the file's shape.
"""
from . import property as prop

PROGNAME = "openQHA branchA"
FILE = "branchA.toml"
REPORT = "branchA.out"
STEP = "branch A"

SCHEMA = {
    prop.INFO_BLOCK: {
        "QM9_INDEX": ("String", None, "the molecule (QM9 index, or the label of a SMILES molecule)"),
        "NAME": ("String", None, "the molecule's name in the config"),
        "SMILES": ("String", None, "declared in config"),
        "TAG": ("String", None, "the tag this run was made under"),
        "ENGINE": ("String", None, "the potential"),
        "MOLECULE_DIR": ("String", None, "the molecule directory"),
        "WORKHORSE": ("String", None, "CREST sampling level"),
        "REFINE": ("String", None, "CREST refinement of the survivors"),
        "RUNTYPE": ("String", None, "CREST run type"),
        "OPTLEV": ("String", None, "CREST optimisation level"),
        "SHAKE": ("Integer", None, "SHAKE (2: all bonds) in CREST's MD"),
        "TSTEP": ("Double", "fs", "CREST MD time step"),
        "HMASS": ("Double", "amu", "hydrogen mass in CREST's MD"),
        "THREADS": ("Integer", None, "CREST threads"),
        "FMAX": ("Double", "eV/A", "tightening convergence on the max force"),
        "DEDUP_RMSD": ("Double", "A", "CREGEN three-fold deduplication: RMSD threshold"),
        "DEDUP_CRITERION": ("String", None, "the deduplication criterion"),
        "DEDUP_ETHR": ("Double", "kcal/mol", "CREGEN energy threshold"),
        "DEDUP_BTHR": ("Double", None, "CREGEN relative rotational-constant threshold"),
        "HESSIAN_MODE": ("String", None, "analytic or finite_difference"),
        "TEMPERATURE": ("Double", "K", "thermochemistry temperature"),
        "SYMMETRY_TOLERANCE": ("Double", "A", "sigma detection tolerance"),
    },
    "CREST_Run": {
        "RETURNCODE": ("Integer", None, "CREST's exit code"),
        "TERMINATED_NORMALLY": ("Boolean", None, "crest.out carries CREST's own terminal line"),
        "N_CONFORMERS": ("Integer", None, "what CREST reported; not a basin count"),
        "N_TERMINATED_EARLY": ("Integer", None, "MD runs CREST terminated early"),
        "SHAKE_USED": ("Integer", None, "the SHAKE that actually ran (read back from input.toml)"),
        "USED_SHAKE_FALLBACK": ("Boolean", None, "the declared fallback ran instead of the published protocol"),
        "WALL": ("Double", "s", "CREST wall time"),
        "WALL_IS_VALID_COST": ("Boolean", None, "false when a directory was reused"),
        "REUSED_SCRATCH": ("Boolean", None, "an existing CREST directory was reused"),
    },
    "Census": {
        "N_FRAMES_IN": ("Integer", None, "CREST conformers plus pooled reference geometries"),
        "N_FROM_CREST": ("Integer", None, "frames from the CREST ensemble"),
        "N_POOLED": ("Integer", None, "reference geometries pooled"),
        "N_NOT_CONVERGED": ("Integer", None, "frames whose tightening did not converge"),
        "N_GRAPH_CHANGED": ("Integer", None, "frames whose bond graph changed in tightening"),
        "N_SADDLES_REJECTED": ("Integer", None, "frames rejected below the frequency floor ithr"),
        "N_BASINS": ("Integer", None, "after tightening and deduplication"),
        "BASIN_CONFORMER_IDS": ("ArrayOfIntegers", None, "the input frame each basin came from"),
        "DUPLICATE_MAP": ("ArrayOfIntegers", None, "per input frame j (index = j): the input frame it was merged into by the deduplication, or j itself when it survived"),
        "SADDLE_CONFORMER_IDS": ("ArrayOfIntegers", None, "input frames that survived the deduplication and were then rejected below the frequency floor ithr"),
        "MAX_RESIDUAL_FORCE": ("Double", "eV/A", "largest residual force after tightening"),
    },
    "Basin": {
        "ENERGY": ("Double", "eV", "electronic energy after tightening"),
        "RELATIVE": ("Double", "kcal/mol", "above the lowest basin"),
        "SIGMA": ("Integer", None, "symmetry number"),
        "SIGMA_SOURCE": ("String", None, "declared_in_config, or detected"),
        "G0": ("Integer", None, "electronic degeneracy"),
        "G0_SOURCE": ("String", None, "declared_in_config, or assumed_singlet"),
        "N_IMAGINARY": ("Integer", None, "imaginary modes of the Hessian"),
        "LOWEST_FREQ": ("Double", "cm^-1", "lowest harmonic frequency"),
        "N_INVERSION_WINDOW": ("Integer", None, "modes in the inversion window [ithr, 0) at admission -- the thermochemistry inverts them"),
        "G_MINUS_EEL": ("Double", "kcal/mol", "harmonic G - E_el at TEMPERATURE"),
        "A_MINUS_EEL": ("Double", "kcal/mol", "harmonic A - E_el at TEMPERATURE"),
    },
    "Criteria": {
        "N_PASSED": ("Integer", None, "acceptance criteria passed"),
        "N_TOTAL": ("Integer", None, "acceptance criteria checked (each with its detail in branchA.out)"),
        "ALL_PASSED": ("Boolean", None, "what the batch driver reads"),
        "FAILED": ("ArrayOfIntegers", None, "numbers of the criteria that failed"),
    },
}


def _f(v):
    return None if v is None else float(v)


def _i(v):
    return None if v is None else int(v)


def blocks_from_record(record):
    """The blocks of `branchA.toml` from the pipeline's in-memory record."""
    s = record.get("settings") or {}
    c = record.get("crest") or {}
    cv = record.get("census") or {}
    crit = record.get("criteria") or []
    info = {
        "QM9_INDEX": record.get("qm9_index") or record.get("label"),
        "NAME": record.get("name"),
        "SMILES": record.get("smiles"),
        "TAG": record.get("tag"),
        "ENGINE": (record.get("engine") or {}).get("engine"),
        "MOLECULE_DIR": record.get("molecule_dir"),
        "WORKHORSE": s.get("workhorse"),
        "REFINE": s.get("refine"),
        "RUNTYPE": s.get("runtype"),
        "OPTLEV": s.get("optlev"),
        "SHAKE": _i(s.get("shake")),
        "TSTEP": _f(s.get("tstep_fs")),
        "HMASS": _f(s.get("hydrogen_mass_amu")),
        "THREADS": _i(s.get("threads")),
        "FMAX": _f(s.get("fmax_eV_A")),
        "DEDUP_RMSD": _f(s.get("dedup_rmsd_A")),
        "DEDUP_CRITERION": s.get("dedup_criterion"),
        "DEDUP_ETHR": _f(s.get("dedup_ethr_kcal")),
        "DEDUP_BTHR": _f(s.get("dedup_bthr_relative")),
        "HESSIAN_MODE": s.get("hessian_mode"),
        "TEMPERATURE": _f(s.get("temperature_K")),
        "SYMMETRY_TOLERANCE": _f(s.get("symmetry_tolerance_A")),
    }
    crest_run = {
        "RETURNCODE": _i(c.get("returncode")),
        "TERMINATED_NORMALLY": c.get("terminated_normally"),
        "N_CONFORMERS": _i(c.get("n_conformers")),
        "N_TERMINATED_EARLY": _i(c.get("n_terminated_early")),
        "SHAKE_USED": _i(c.get("shake_used")),
        "USED_SHAKE_FALLBACK": c.get("used_shake_fallback"),
        "WALL": _f(c.get("wall_seconds")),
        "WALL_IS_VALID_COST": c.get("wall_is_valid_cost"),
        "REUSED_SCRATCH": c.get("reused_scratch"),
    }
    census = {
        "N_FRAMES_IN": _i(cv.get("n_frames_in")),
        "N_FROM_CREST": _i(cv.get("n_frames_from_crest")),
        "N_POOLED": _i(cv.get("n_reference_geometries_pooled")),
        "N_NOT_CONVERGED": _i(cv.get("n_not_converged")),
        "N_GRAPH_CHANGED": _i(cv.get("n_graph_changed")),
        "N_SADDLES_REJECTED": _i(cv.get("n_saddles_rejected")),
        "N_BASINS": _i(cv.get("n_basins")),
        "BASIN_CONFORMER_IDS": [int(x) for x in (cv.get("basin_conformer_ids") or [])],
        "DUPLICATE_MAP": _duplicate_map(cv),
        "SADDLE_CONFORMER_IDS": [int(s["conformer_id"]) for s in (cv.get("saddles") or []) if "conformer_id" in s],
        "MAX_RESIDUAL_FORCE": _f(cv.get("max_residual_force_eV_A")),
    }
    basins = []
    for b in record.get("basins") or []:
        sym = b.get("symmetry") or {}
        th = b.get("thermo") or {}
        basins.append({
            "INDEX": int(b["basin_index"]),
            "ENERGY": _f(b.get("energy_eV")),
            "RELATIVE": _f(b.get("relative_kcal")),
            "SIGMA": _i(sym.get("sigma")),
            "SIGMA_SOURCE": sym.get("sigma_source"),
            "G0": _i(b.get("electronic_degeneracy")),
            "G0_SOURCE": b.get("electronic_degeneracy_source"),
            "N_IMAGINARY": _i(b.get("n_imaginary")),
            "LOWEST_FREQ": _f(b.get("lowest_frequency_cm_inv")),
            "N_INVERSION_WINDOW": _i(b.get("n_inversion_window")),
            "G_MINUS_EEL": _f(th.get("G_minus_Eel_kcal")),
            "A_MINUS_EEL": _f(th.get("A_minus_Eel_kcal")),
        })
    failed = [int(x["number"]) for x in crit if not x.get("passed")]
    criteria = {
        "N_PASSED": len(crit) - len(failed),
        "N_TOTAL": len(crit),
        "ALL_PASSED": bool(record.get("all_criteria_passed", not failed)),
        "FAILED": failed,
    }
    return {prop.INFO_BLOCK: info, "CREST_Run": crest_run, "Census": census,
            "Basin": basins, "Criteria": criteria}


def write(path, record):
    """Write `branchA.toml` with STATUS NORMAL TERMINATION. Returns the keys the schema
    did not know (the writer's test requires none)."""
    return prop.write(path, blocks_from_record(record), SCHEMA,
                      status=prop.NORMAL_TERMINATION, progname=PROGNAME)


# ---- what readers ask -------------------------------------------------------------------
def _duplicate_map(cv):
    """The census's `duplicate_map` {j: kept} as one array indexed by input frame: the
    frame j merged into, or j itself. Empty when the census did not record it (records
    written before 2026-09-18)."""
    dm = cv.get("duplicate_map")
    n = cv.get("n_frames_in") or cv.get("n_input_frames_total")
    if not isinstance(dm, dict) or not n:
        return []
    return [int(dm.get(j, dm.get(str(j), j))) for j in range(int(n))]


def basin_rows(doc):
    """The `[[Basin]]` tables of a loaded `branchA.toml`, in INDEX order."""
    rows = doc.get("Basin") or []
    return sorted(rows, key=lambda r: int(r.get("INDEX", 0)))


def relative_kcal(doc):
    """RELATIVE per basin, in basin order. A basin without it RAISES: a default of 0.0 is
    a valid relative energy, so a silent default would make the basins degenerate and no
    check would fire (the 2026-09-13 propanal defect)."""
    out = []
    for r in basin_rows(doc):
        if "RELATIVE" not in r:
            raise KeyError("basin {} of this branchA.toml has no RELATIVE; its keys are {}"
                           .format(r.get("INDEX"), sorted(r)))
        out.append(float(r["RELATIVE"]))
    return out
