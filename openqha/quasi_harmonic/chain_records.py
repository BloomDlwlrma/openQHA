"""The Records of collect, the ensemble report and 02d (records redesign ticket 19, 2026-09-15).

All three live in `_records/md_<route>/` with the setting in the stem
(`collect.out`, `collect_s2.toml`, `collect_s2.trajectories.dat`, `ensemble_s2.toml`):

    collect.out              the Report; written before the Property file
    collect.toml             [Calculation_Status] [Calculation_Info] [Criteria]; STATUS is what the
                             collect Batch reads to say a molecule is done
    collect.trajectories.dat .criteria.dat .assembly.dat .blank.dat   the tables (openqha.store.dat)
    ensemble.out / .toml     [Calculation_Info] [[Result]] (one per atom set) [[Basin]]
    02d_frequency_identity.out / .toml   [Calculation_Info] [[Basin]] [[Hybrid]]

The Property files hold only what a later step reads (collect's STATUS and verdict count;
the ensemble's F_conf, populations, per-basin T*S; 02d's per-basin numbers). Every
detail is in the `.out`.
"""
import re
from pathlib import Path

from ..store import layout, property as prop

COLLECT = "collect"
ENSEMBLE = "ensemble"
IDENTITY = "02d_frequency_identity"
TABLES = ("trajectories", "criteria", "assembly", "blank")

_LEADING_NUMBER = re.compile(r"^\s*(\d+)\b")


def stem(molecule, route, setting, name):
    """`_records/md_<route>/<name>` or `<name>_<setting>` (no setting level)."""
    return layout.md_records_dir(molecule, route) / layout.record_file_name(name, setting)


def collect_paths(stem_path):
    """The files of one collect Calculation from its stem (`.../collect_s2`)."""
    stem_path = Path(stem_path)
    d, n = stem_path.parent, stem_path.name
    out = {"out": d / (n + ".out"), "toml": d / (n + ".toml")}
    for t in TABLES:
        out[t] = d / "{}.{}.dat".format(n, t)
    return out


def _f(v):
    return None if v is None else float(v)


def _i(v):
    return None if v is None else int(v)


# ---- collect ----------------------------------------------------------------------------
COLLECT_PROGNAME = "openQHA collect"
COLLECT_SCHEMA = {
    prop.INFO_BLOCK: {
        "QM9_INDEX": ("String", None, "the molecule"),
        "TAG": ("String", None, "the tag of this run"),
        "BASIN_TAG": ("String", None, "the tag whose basins the trajectories started from"),
        "SETTING": ("String", None, "the setting name"),
        "ROUTE": ("String", None, "openmm or ase"),
        "ENGINE": ("String", None, "the potential"),
        "TEMPERATURE": ("Double", "K", "the temperature of the entropies"),
        "SIGMA": ("Integer", None, "symmetry number declared for the molecule"),
        "G0": ("Integer", None, "electronic degeneracy declared for the molecule"),
        "N_TRAJECTORIES": ("Integer", None, "trajectories analysed"),
        "N_BASINS": ("Integer", None, "basins with at least one trajectory"),
        "FRAME_SPACING": ("Double", "fs", "time between frames"),
        "TIMESTEP": ("Double", "fs", "integration step of the trajectories"),
    },
    "Criteria": {
        "N_PASSED": ("Integer", None, "collect's acceptance criteria passed"),
        "N_TOTAL": ("Integer", None, "criteria checked (each with its measure in collect.out and collect.criteria.dat)"),
        "ALL_PASSED": ("Boolean", None, "what the collect Batch and the ensemble report read"),
        "FAILED": ("ArrayOfIntegers", None, "numbers of the criteria that failed"),
    },
}


def criterion_number(text):
    m = _LEADING_NUMBER.match(str(text))
    return int(m.group(1)) if m else None


def write_collect(stem_path, info, verdicts):
    """`collect.toml` (STATUS NORMAL TERMINATION). `verdicts` is collect's list of
    (criterion text, measured, passed). Returns the keys the schema did not know."""
    failed = [criterion_number(c) for c, _m, p in verdicts if not p]
    failed = [n for n in failed if n is not None]
    blocks = {
        prop.INFO_BLOCK: {
            "QM9_INDEX": info.get("species"), "TAG": info.get("tag"), "BASIN_TAG": info.get("basin_tag"),
            "SETTING": info.get("setting"), "ROUTE": info.get("route"), "ENGINE": info.get("engine"),
            "TEMPERATURE": _f(info.get("temperature_K")), "SIGMA": _i(info.get("sigma")),
            "G0": _i(info.get("g0")), "N_TRAJECTORIES": _i(info.get("n_trajectories")),
            "N_BASINS": _i(info.get("n_basins")), "FRAME_SPACING": _f(info.get("frame_spacing_fs")),
            "TIMESTEP": _f(info.get("timestep_fs")),
        },
        "Criteria": {
            "N_PASSED": sum(1 for _c, _m, p in verdicts if p), "N_TOTAL": len(verdicts),
            "ALL_PASSED": bool(verdicts) and all(p for _c, _m, p in verdicts), "FAILED": failed,
        },
    }
    return prop.write(collect_paths(stem_path)["toml"], blocks, COLLECT_SCHEMA,
                      status=prop.NORMAL_TERMINATION, progname=COLLECT_PROGNAME)


def collect_status(stem_path):
    return prop.status_of(collect_paths(stem_path)["toml"])


def collect_done(stem_path):
    """The collect Batch's "done": STATUS NORMAL TERMINATION in collect.toml."""
    return collect_status(stem_path) == prop.NORMAL_TERMINATION


# ---- ensemble ---------------------------------------------------------------------------
ENSEMBLE_PROGNAME = "openQHA ensemble"
ENSEMBLE_SCHEMA = {
    prop.INFO_BLOCK: {
        "QM9_INDEX": ("String", None, "the molecule"),
        "TAG": ("String", None, "the tag of this run"),
        "BASIN_TAG": ("String", None, "the tag whose basins were summed"),
        "SETTING": ("String", None, "the setting name"),
        "ROUTE": ("String", None, "openmm or ase"),
        "TEMPERATURE": ("Double", "K", ""),
        "N_BASINS": ("Integer", None, "basins branch A found"),
        "COLLECT_PASSED": ("Integer", None, "collect's criteria passed"),
        "COLLECT_TOTAL": ("Integer", None, "collect's criteria checked"),
    },
    "Result": {
        "ATOMS": ("String", None, "the atom set the entropies were computed over (all or heavy)"),
        "F_CONF": ("Double", "kcal/mol", "conformational free energy relative to the lowest basin alone (<= 0)"),
        "PARTITION_FUNCTION_RELATIVE": ("Double", None, "sum over basins of exp(-dG/kT)"),
        "DELTA_G": ("ArrayOfDoubles", "kcal/mol", "G of each basin above the lowest"),
        "POPULATIONS": ("ArrayOfDoubles", None, "Boltzmann population of each basin"),
        "EFFECTIVE_BASINS": ("Double", None, "exp of the population entropy"),
        "N_WITH_ENTROPY": ("Integer", None, "basins with a T*S from collect"),
        "N_ELECTRONIC_ONLY": ("Integer", None, "basins summed on their electronic energy alone (a failure when > 0)"),
        "DISTINCT_CROSSINGS": ("Integer", None, "trajectories that left their basin"),
        "SYMMETRY_CROSSINGS": ("Integer", None, "symmetry-equivalent crossings"),
    },
    "Basin": {
        "ATOMS": ("String", None, "the atom set"),
        "TS": ("Double", "kcal/mol", "T*S over the seeds of this basin"),
        "SPREAD": ("Double", "kcal/mol", "spread of T*S over the seeds"),
        "N_SEEDS": ("Integer", None, ""),
        "N_FRAMES": ("Integer", None, ""),
        "DISTINCT_CROSSINGS": ("Integer", None, ""),
        "SYMMETRY_CROSSINGS": ("Integer", None, ""),
    },
}


def write_ensemble(stem_path, report):
    """`ensemble.toml` from the report dict the ensemble step builds (its `results` keyed
    by atom set, each with `per_basin_detail`)."""
    cc = report.get("collect_criteria") or {}
    info = {
        "QM9_INDEX": report.get("species"), "TAG": report.get("tag"), "BASIN_TAG": report.get("basin_tag"),
        "SETTING": report.get("setting"), "ROUTE": report.get("route"),
        "TEMPERATURE": _f(report.get("temperature_K")), "N_BASINS": _i(report.get("n_basins_branch_a")),
        "COLLECT_PASSED": _i(cc.get("passed")), "COLLECT_TOTAL": _i(cc.get("total")),
    }
    results, basins = [], []
    for atoms, rec in (report.get("results") or {}).items():
        results.append({
            "INDEX": len(results), "ATOMS": atoms,
            "F_CONF": _f(rec.get("F_conf_kcal")),
            "PARTITION_FUNCTION_RELATIVE": _f(rec.get("partition_function_relative")),
            "DELTA_G": [float(x) for x in (rec.get("delta_G_kcal") or [])],
            "POPULATIONS": [float(x) for x in (rec.get("populations") or [])],
            "EFFECTIVE_BASINS": _f(rec.get("effective_basins")),
            "N_WITH_ENTROPY": _i(rec.get("n_with_entropy")),
            "N_ELECTRONIC_ONLY": _i(rec.get("n_electronic_only")),
            "DISTINCT_CROSSINGS": _i(rec.get("distinct_crossings")),
            "SYMMETRY_CROSSINGS": _i(rec.get("symmetry_crossings")),
        })
        for b, d in sorted((rec.get("per_basin_detail") or {}).items(), key=lambda kv: int(kv[0])):
            d = d or {}
            basins.append({
                "INDEX": int(b), "ATOMS": atoms, "TS": _f(d.get("TS_kcal")), "SPREAD": _f(d.get("spread_kcal")),
                "N_SEEDS": _i(d.get("n_seeds")), "N_FRAMES": _i(d.get("n_frames")),
                "DISTINCT_CROSSINGS": _i(d.get("distinct_crossings")),
                "SYMMETRY_CROSSINGS": _i(d.get("symmetry_crossings")),
            })
    blocks = {prop.INFO_BLOCK: info, "Result": results, "Basin": basins}
    return prop.write(Path(str(stem_path) + ".toml"), blocks, ENSEMBLE_SCHEMA,
                      status=prop.NORMAL_TERMINATION, progname=ENSEMBLE_PROGNAME)


# ---- 02d --------------------------------------------------------------------------------
IDENTITY_PROGNAME = "openQHA 02d"
IDENTITY_SCHEMA = {
    prop.INFO_BLOCK: {
        "QM9_INDEX": ("String", None, "the molecule"),
        "TAG": ("String", None, ""),
        "SETTING": ("String", None, ""),
        "ROUTE": ("String", None, "openmm or ase"),
        "ENGINE": ("String", None, "the potential"),
        "TEMPERATURE": ("Double", "K", ""),
        "NU_CUTS": ("ArrayOfDoubles", "cm^-1", "the hybrid cut-offs tried in stage 4"),
        "MIN_BLOCK_OVERLAP": ("Double", None, "the mode-block overlap a splice needs"),
        "WALL": ("Double", "s", ""),
    },
    "Basin": {
        "SIGMA": ("Integer", None, "symmetry number"),
        "G0": ("Integer", None, "electronic degeneracy"),
        "HESSIAN_ASYMMETRY": ("Double", "eV/A^2", "stage 1: the Hessian's asymmetry"),
        "PROJECTION_AGREEMENT": ("Double", "cm^-1", "stage 1: projected against diagonalised modes"),
        "REAL_TRAJECTORY": ("String", None, "stage 2: the trajectory folder, or absent when skipped"),
        "G_MINUS_EEL_HESSIAN": ("Double", "kcal/mol", "stage 3: G - E_el from the Hessian's omega"),
        "G_MINUS_EEL_QHA": ("Double", "kcal/mol", "stage 3: G - E_el from the quasi-harmonic nu"),
        "VIBRATIONAL_ONLY_DELTA": ("Double", "kcal/mol", "stage 3: everything the spectrum can touch, QHA minus Hessian"),
        "ROTATIONAL_GEOMETRY_SHIFT": ("Double", "kcal/mol", "stage 3: G_rot moved by the geometry, not the spectrum"),
    },
    "Hybrid": {
        "BASIN": ("Integer", None, ""),
        "NU_CUT": ("Double", "cm^-1", "stage 4: modes below this come from QHA"),
        "G_MINUS_EEL": ("Double", "kcal/mol", "stage 4: the spliced G - E_el"),
        "N_FROM_QHA": ("Integer", None, "modes taken from QHA"),
        "N_REJECTED_BY_OVERLAP": ("Integer", None, "modes refused on block overlap"),
    },
}


def write_identity(stem_path, report):
    """`02d_frequency_identity.toml` from 02d's report dict (`basins` keyed by index)."""
    info = {
        "QM9_INDEX": report.get("species"), "TAG": report.get("tag"), "SETTING": report.get("setting"),
        "ROUTE": report.get("route"), "ENGINE": report.get("engine"),
        "TEMPERATURE": _f(report.get("temperature_K")),
        "NU_CUTS": [float(x) for x in (report.get("nu_cuts_cm") or [])],
        "MIN_BLOCK_OVERLAP": _f(report.get("min_block_overlap")), "WALL": _f(report.get("wall_seconds")),
    }
    basins, hybrids = [], []
    for b, e in sorted((report.get("basins") or {}).items(), key=lambda kv: int(kv[0])):
        harm = e.get("harmonic_limit") or {}
        real = e.get("real") or {}
        gt = e.get("gtotal") or {}
        cmp_ = gt.get("comparison") or {}
        basins.append({
            "INDEX": int(b), "SIGMA": _i(e.get("sigma")), "G0": _i(e.get("electronic_degeneracy")),
            "HESSIAN_ASYMMETRY": _f(harm.get("hessian_asymmetry_eV_A2")),
            "PROJECTION_AGREEMENT": _f(harm.get("projection_agreement_cm")),
            "REAL_TRAJECTORY": real.get("trajectory"),
            "G_MINUS_EEL_HESSIAN": _f((gt.get("hessian") or {}).get("G_minus_Eel_kcal")),
            "G_MINUS_EEL_QHA": _f((gt.get("qha") or {}).get("G_minus_Eel_kcal")),
            "VIBRATIONAL_ONLY_DELTA": _f(cmp_.get("vibrational_only_delta_kcal")),
            "ROTATIONAL_GEOMETRY_SHIFT": _f(cmp_.get("rotational_geometry_shift_kcal")),
        })
        for hy in ((e.get("hybrid") or {}).get("hybrid") or []):
            hybrids.append({
                "INDEX": len(hybrids), "BASIN": int(b), "NU_CUT": _f(hy.get("nu_cut_cm")),
                "G_MINUS_EEL": _f(hy.get("G_minus_Eel_kcal")),
                "N_FROM_QHA": _i((hy.get("spectrum_record") or {}).get("n_from_qha")),
                "N_REJECTED_BY_OVERLAP": _i((hy.get("spectrum_record") or {}).get("n_rejected_by_overlap")),
            })
    blocks = {prop.INFO_BLOCK: info, "Basin": basins}
    if hybrids:
        blocks["Hybrid"] = hybrids
    return prop.write(Path(str(stem_path) + ".toml"), blocks, IDENTITY_SCHEMA,
                      status=prop.NORMAL_TERMINATION, progname=IDENTITY_PROGNAME)
