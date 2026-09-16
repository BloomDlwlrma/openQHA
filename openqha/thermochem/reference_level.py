"""Ticket 26: the reference level, and the comparison of every level a molecule has.

REFERENCE LEVEL (ADR 0004)
--------------------------
`wb97m-d3bj_def2-tzvppd`, the level MACE-OFF23 was trained to. Every branch A basin is
re-optimised at that level and its Hessian computed (ORCA `TightOpt Freq`; the analytic
Hessian works in ORCA 6.0.1 for this functional, `NumFreq` is the declared fallback),
the basins are re-deduplicated with branch A's own rule, the same assembly as ticket 24
is run on the survivors, and a merge map says where every MACE basin landed:

    merge_map.dat   mace_basin  reference_basin (or -1)  status  rmsd_displacement_A
                    rmsd_to_representative_A  energy_eh  n_imaginary  hessian_route

`status` is `kept` (the basin is a reference basin), `merged` (it relaxed into another
reference basin: the same minimum at this level), or `saddle` (an imaginary mode at the
relaxed geometry: excluded from the reference ensemble, and a topology disagreement
worth a label later). The RMSD before is MACE geometry -> its own relaxed geometry
(the geometry shift of the level); the RMSD after is relaxed geometry -> the kept
representative it merged into (0 for a kept basin).

Engine files: `orca/<level>/basinNN/job.{inp,out,hess,xyz,...}`. Records:
`levels/<level>/thermo_msrrho.{out,toml}` (ticket-24 shape) and `merge_map.dat`.

LEVEL COMPARE
-------------
`levels/level_compare.{out,toml}` reads every `levels/<level>/thermo_msrrho.toml` a
molecule has and prints the three tiers: model error (the engine's level minus the
reference), level error (reference minus experiment), total (engine minus experiment).
An absent level is stated as PRESENT = false and no tier that needs it is computed.
"""
from pathlib import Path

import numpy as np

from .. import config
from ..conformer_search import conformers, degeneracy, symmetry
from ..potentials import engine
from ..qm_interfaces import orca
from ..store import basins as basins_mod
from ..store import branch_a_property, dat, layout, property as prop, report
from . import msrrho_ensemble as me
from . import thermo

REFERENCE_LEVEL = engine.REFERENCE_LEVEL
COMPARE_STEP = "level_compare"
COMPARE_PROGNAME = "openQHA level_compare"

COMPARE_SCHEMA = {
    "Calculation_Info": {
        "MOLECULE_DIR": ("String", None, "the molecule directory"),
        "QM9_INDEX": ("String", None, "the molecule"),
        "REFERENCE_LEVEL": ("String", None, "the level model error is measured against"),
        "ENGINE_LEVEL": ("String", None, "the potential's level"),
        "TEMPERATURE": ("Double", "K", "temperature"),
    },
    "Level": {
        "LEVEL": ("String", None, "level name"),
        "PRESENT": ("Boolean", None, "a thermo_msrrho record exists for this level"),
        "S_ABS": ("Double", "cal/mol/K", "absolute entropy at this level"),
        "G_TOTAL": ("Double", "kcal/mol", "G_total at this level"),
        "N_BASINS": ("Integer", None, "basins at this level"),
        "N_INCLUDED": ("Integer", None, "basins in the ensemble at this level"),
        "S_REF": ("Double", "cal/mol/K", "S_msRRHO of the reference basin"),
        "S_CONF": ("Double", "cal/mol/K", "S_CONF_PRIME + DS_BAR"),
    },
    "Tiers": {
        "MODEL_ERROR_S": ("Double", "cal/mol/K", "engine S_ABS - reference S_ABS"),
        "MODEL_ERROR_S_CONF": ("Double", "cal/mol/K", "engine conformational term - reference"),
        "LEVEL_ERROR_S": ("Double", "cal/mol/K", "reference S_ABS - experiment"),
        "TOTAL_ERROR_S": ("Double", "cal/mol/K", "engine S_ABS - experiment"),
        "S_EXPERIMENT": ("Double", "cal/mol/K", "declared experimental S"),
        "S_EXPERIMENT_SOURCE": ("String", None, "BibTeX key"),
        "MODEL_ERROR_G_REL": ("Double", "kcal/mol", "engine (G_total - G_ref) - reference (G_total - G_ref)"),
    },
}


# ====================================================================== reference level
def _mol_with_conformers(smiles, symbols, geometries):
    """An RDKit molecule in branch A's atom order (SMILES -> AddHs) carrying one
    conformer per geometry, so that `conformers.dedup` can apply branch A's rule."""
    from rdkit import Chem
    mol, _ = conformers.mol_from_smiles(smiles)
    mol = Chem.AddHs(mol)
    if [a.GetSymbol() for a in mol.GetAtoms()] != list(symbols):
        raise ValueError("branch A's atom order does not match the SMILES {}: {} vs {}".format(
            smiles, [a.GetSymbol() for a in mol.GetAtoms()], list(symbols)))
    mol.RemoveAllConformers()
    cids = []
    for pos in geometries:
        conf = Chem.Conformer(mol.GetNumAtoms())
        cid = mol.AddConformer(conf, assignId=True)
        conformers._set_conf(mol, cid, np.asarray(pos, dtype=float))
        cids.append(cid)
    return mol, cids


def run_calculation(molecule, level=REFERENCE_LEVEL, keywords=orca.REFERENCE_KEYWORDS,
                    nprocs=8, maxcore=3000, basins=None, qm9_index=None, cfg=None,
                    preset="crest", temperature_K=None, ptot=me.PTOT):
    """Re-optimise and Hessian every branch A basin (or the subset `basins`) at `level`,
    re-deduplicate, assemble, write the merge map and the level's thermo_msrrho record.
    ORCA is skipped for a basin whose engine folder already holds a finished job."""
    from ase.io import read
    molecule = Path(molecule)
    doc = prop.load(layout.records_dir(molecule) / "branchA.toml")
    info_a = doc.get("Calculation_Info") or {}
    T = float(temperature_K if temperature_K is not None else info_a.get("TEMPERATURE", thermo.T_REF))
    qid = qm9_index if qm9_index is not None else info_a.get("QM9_INDEX")
    rows = {int(r["INDEX"]): r for r in branch_a_property.basin_rows(doc)}
    files = {int(p.parent.name[len("basin"):]): p for p in basins_mod.basin_files(molecule)}
    wanted = sorted(files) if basins is None else [int(b) for b in basins]
    mace_level = engine.level_name(info_a.get("ENGINE")) if info_a.get("ENGINE") else None

    # g' of the MACE basins (run the degeneracy Calculation at the MACE level if absent)
    g_prime = {}
    if mace_level:
        deg_path = layout.level_dir(molecule, mace_level) / "degeneracy.toml"
        if not deg_path.is_file():
            degeneracy.run_calculation(molecule, mace_level)
        g_prime = {int(r["INDEX"]): (int(r["G_PRIME"]), str(r["G_PRIME_SOURCE"]))
                   for r in prop.load(deg_path).get("Basin", [])}

    # ---- ORCA per basin ------------------------------------------------------------
    relaxed = []
    for b in wanted:
        atoms = read(str(files[b]), format="extxyz")
        wd = layout.orca_level_dir(molecule, level, b)
        r = orca.optimise_and_hessian(atoms.get_chemical_symbols(), atoms.get_positions(), wd,
                                      keywords=keywords, nprocs=nprocs, maxcore=maxcore)
        rt = orca.verify_hess_frequencies(r["hess"])
        pos_mace = atoms.get_positions()
        pos_ref = np.asarray(r["positions_A"])
        relaxed.append(dict(mace_basin=b, symbols=atoms.get_chemical_symbols(),
                            masses=[float(m) for m in atoms.get_masses()],
                            positions=pos_ref, energy_eh=r["energy_eh"],
                            energy_eV=r["energy_eh"] * orca.EV_PER_HARTREE,
                            hessian=orca.hessian_to_ev_per_angstrom2(r["hess"]["hessian_eh_bohr2"]),
                            n_imaginary=r["n_imaginary"], hessian_route=r["hessian_route"],
                            roundtrip_cm=float(rt["max_deviation_cm_inv"]),
                            rmsd_displacement=degeneracy.kabsch_rmsd(pos_mace, pos_ref),
                            sigma_mace=int(rows[b]["SIGMA"]), g0=int(rows[b]["G0"]),
                            orca_version=r["orca_version"], seconds=r["seconds"]))

    # ---- re-deduplicate with branch A's rule ----------------------------------------
    minima = [x for x in relaxed if x["n_imaginary"] == 0]
    saddles = [x for x in relaxed if x["n_imaginary"] > 0]
    kept_map = {}
    if minima:
        mol, cids = _mol_with_conformers(info_a.get("SMILES"), minima[0]["symbols"],
                                         [x["positions"] for x in minima])
        kept, mapping, _warn = conformers.dedup(
            mol, cids, [x["energy_eV"] for x in minima],
            threshold_A=float(info_a.get("DEDUP_RMSD", conformers.DEDUP_RMSD_A)),
            ethr_kcal=info_a.get("DEDUP_ETHR"), bthr_rel=info_a.get("DEDUP_BTHR"))
        for cid, x in zip(cids, minima):
            target = mapping.get(cid, cid)
            kept_map[x["mace_basin"]] = kept.index(target)
    representative = {}
    for x in minima:
        rb = kept_map[x["mace_basin"]]
        representative.setdefault(rb, x)

    # ---- merge map ------------------------------------------------------------------
    merge_rows = []
    for x in relaxed:
        if x["n_imaginary"] > 0:
            status, rb, rmsd_after = "saddle", -1, None
        else:
            rb = kept_map[x["mace_basin"]]
            rep_x = representative[rb]
            status = "kept" if rep_x is x else "merged"
            rmsd_after = 0.0 if status == "kept" else degeneracy.kabsch_rmsd(x["positions"], rep_x["positions"])
        merge_rows.append(dict(mace_basin=x["mace_basin"], reference_basin=rb, status=status,
                               rmsd_displacement_A=x["rmsd_displacement"],
                               rmsd_to_representative_A=rmsd_after, energy_eh=x["energy_eh"],
                               n_imaginary=x["n_imaginary"], hessian_route=x["hessian_route"],
                               roundtrip_cm=x["roundtrip_cm"], seconds=x["seconds"]))
    lvl = layout.level_dir(molecule, level)
    lvl.mkdir(parents=True, exist_ok=True)
    dat.write_table(lvl / "merge_map.dat", merge_rows)

    # ---- assembly on the reference basins -------------------------------------------
    ref_basins = []
    for rb in sorted(representative):
        x = representative[rb]
        ops = symmetry.symmetry_operations([_z(s) for s in x["symbols"]], x["positions"])
        rec = me.basin_thermochemistry(rb, x["hessian"], x["masses"], x["positions"], x["energy_eV"],
                                       ops["sigma"], x["g0"], T, preset)
        gp = g_prime.get(x["mace_basin"], (1, "undeclared"))
        rec["g_prime"], rec["g_prime_source"] = gp
        rec["mace_basin"] = x["mace_basin"]
        rec["sigma_mace"] = x["sigma_mace"]
        ref_basins.append(rec)
    for x in saddles:
        rec = dict(index=len(ref_basins) + saddles.index(x) + 100, excluded=True,
                   excluded_reason="saddle at {}: {} imaginary mode(s)".format(level, x["n_imaginary"]),
                   E_el_kcal=x["energy_eV"] * me.EV_TO_KCAL, sigma=x["sigma_mace"], g0=x["g0"],
                   g_prime=1, g_prime_source="saddle", lowest_frequency_cm=float("nan"),
                   mace_basin=x["mace_basin"])
        ref_basins.append(rec)
    ens = me.assemble(ref_basins, T, ptot)
    ref = next(b for b in ref_basins if b["index"] == ens["reference_basin"])
    spread = thermo.preset_spread(ref["frequencies_cm"], ref["masses"], ref["positions"], temperature_K=T)
    exp = config.experimental_entropy(qid, cfg) if qid else None
    info = {"MOLECULE_DIR": str(molecule), "QM9_INDEX": qid, "TAG": info_a.get("TAG"),
            "LEVEL": str(level), "ENGINE": "ORCA {} ({})".format(relaxed[0]["orca_version"], keywords),
            "PRESET": preset, "TAU": float(thermo.MSRRHO_PRESETS[preset]["tau_cm"]),
            "ROTOR_CAP_RULE": str(thermo.MSRRHO_PRESETS[preset]["rotor_cap"]),
            "ITHR_POLICY": "refuse", "FSCAL": 1.0, "TEMPERATURE": T, "PRESSURE": float(thermo.P_STD),
            "REFERENCE_BASIN": ens["reference_basin"], "PTOT": float(ptot), "EXTRAPOLATION": "none"}
    me.write_records(lvl, info, ref_basins, ens, spread, exp)
    out = dict(ens)
    out.update(basins=ref_basins, merge_map=merge_rows, info=info, experimental=exp,
               record=lvl / "thermo_msrrho.toml", hessian_routes=sorted({x["hessian_route"] for x in relaxed}))
    return out


def _z(symbol):
    from ase.data import atomic_numbers
    return atomic_numbers[symbol]


# ====================================================================== level compare
def level_compare(molecule, qm9_index=None, cfg=None, reference_level=REFERENCE_LEVEL):
    """Read every `levels/<level>/thermo_msrrho.toml`, write `levels/level_compare.{out,toml}`."""
    molecule = Path(molecule)
    levels_dir = molecule / layout.LEVELS
    doc_a = prop.load(layout.records_dir(molecule) / "branchA.toml")
    info_a = doc_a.get("Calculation_Info") or {}
    qid = qm9_index if qm9_index is not None else info_a.get("QM9_INDEX")
    engine_level = engine.level_name(info_a.get("ENGINE")) if info_a.get("ENGINE") else None
    found = {}
    if levels_dir.is_dir():
        for d in sorted(levels_dir.iterdir()):
            p = d / "thermo_msrrho.toml"
            if d.is_dir() and p.is_file() and prop.status_of(p) == prop.NORMAL_TERMINATION:
                found[d.name] = prop.load(p)
    names = sorted(set(found) | {reference_level} | ({engine_level} if engine_level else set()))
    rows = []
    for name in names:
        d = found.get(name)
        row = {"LEVEL": name, "PRESENT": d is not None}
        if d is not None:
            res, ens_b = d["Result"], d["Ensemble"]
            ref_row = next((r for r in d["Basin"] if int(r["INDEX"]) == int(d["Calculation_Info"]["REFERENCE_BASIN"])), None)
            row.update({"S_ABS": res["S_ABS"], "G_TOTAL": res["G_TOTAL"], "N_BASINS": res["N_BASINS"],
                        "N_INCLUDED": res["N_INCLUDED"],
                        "S_REF": ref_row["S_MSRRHO"] if ref_row else None,
                        "S_CONF": ens_b["S_CONF_PRIME"] + ens_b["DS_BAR"]})
        rows.append(row)
    by = {r["LEVEL"]: r for r in rows}
    tiers = {}
    exp = config.experimental_entropy(qid, cfg) if qid else None
    ref = by.get(reference_level) if by.get(reference_level, {}).get("PRESENT") else None
    eng = by.get(engine_level) if engine_level and by.get(engine_level, {}).get("PRESENT") else None
    if ref and eng:
        tiers["MODEL_ERROR_S"] = eng["S_ABS"] - ref["S_ABS"]
        tiers["MODEL_ERROR_S_CONF"] = eng["S_CONF"] - ref["S_CONF"]
        g_rel_e = eng["G_TOTAL"] - _g_ref(found[engine_level])
        g_rel_r = ref["G_TOTAL"] - _g_ref(found[reference_level])
        tiers["MODEL_ERROR_G_REL"] = g_rel_e - g_rel_r
    if exp is not None:
        tiers["S_EXPERIMENT"], tiers["S_EXPERIMENT_SOURCE"] = exp
        if ref:
            tiers["LEVEL_ERROR_S"] = ref["S_ABS"] - exp[0]
        if eng:
            tiers["TOTAL_ERROR_S"] = eng["S_ABS"] - exp[0]
    info = {"MOLECULE_DIR": str(molecule), "QM9_INDEX": qid, "REFERENCE_LEVEL": reference_level,
            "ENGINE_LEVEL": engine_level, "TEMPERATURE": float(info_a.get("TEMPERATURE", thermo.T_REF))}
    blocks = {"Calculation_Info": info, "Level": rows, "Tiers": tiers}
    missing = prop.write(levels_dir / "level_compare.toml", blocks, COMPARE_SCHEMA,
                         prop.NORMAL_TERMINATION, COMPARE_PROGNAME)
    if missing:
        raise RuntimeError("level_compare.toml keys outside the schema: {}".format(missing))
    rep = report.Report("openQHA level_compare", "every level of {} beside the reference and "
                        "the experiment".format(qid))
    rep.section("levels")
    rep.table(["level", "present", "S_abs", "S_conf", "G_total", "basins", "included"],
              [[r["LEVEL"], r["PRESENT"], "%.3f" % r["S_ABS"] if r["PRESENT"] else "-",
                "%.3f" % r["S_CONF"] if r["PRESENT"] else "-", "%.4f" % r["G_TOTAL"] if r["PRESENT"] else "-",
                r.get("N_BASINS", "-"), r.get("N_INCLUDED", "-")] for r in rows])
    rep.section("tiers (cal/mol/K unless noted)")
    if not tiers:
        rep.text("  no tier can be computed: the reference level or the engine level is absent")
    for k, v in tiers.items():
        rep.kv(k, v if not isinstance(v, float) else "%.4f" % v)
    rep.note("model error = engine level minus reference level (the level MACE-OFF23 was trained "
             "to); level error = reference minus experiment; total = engine minus experiment. An "
             "absent level is stated, never zero.")
    rep.write(levels_dir / "level_compare.out", step=COMPARE_STEP)
    return dict(levels=rows, tiers=tiers, record=levels_dir / "level_compare.toml")


def _g_ref(doc):
    ref_idx = int(doc["Calculation_Info"]["REFERENCE_BASIN"])
    return next(float(r["G_I"]) for r in doc["Basin"] if int(r["INDEX"]) == ref_idx)
