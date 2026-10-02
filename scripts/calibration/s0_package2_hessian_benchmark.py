"""Package 2 -- the Hessian side branch, the frequency benchmark, and the rigid-rotor harmonic-oscillator reference free energy.

CALIBRATION. The docstring below settles the classification itself: the free
energies produced here are a reference sample, not stage 0's product.

Package 2. It does four things:

1. **7 species x lowest basin x one finite-difference Hessian**, with the Eckart projection before diagonalisation;
2. **the frequency benchmark**: mode by mode against the native QM9 B3LYP/6-31G(2df,p) 3N-6 frequencies;
3. **a free-energy error bar converted from the frequency error**, by three routes (measured direct displacement / uncorrelated / fully correlated);
4. **the rigid-rotor harmonic-oscillator reference free energy**: a complete four-term `(G - E_el)`, and the differences over the 11 edges.

> **The fourth overturns an earlier premise** ("no L1-min branch, produces no competing harmonic free energy").
> The user asked explicitly on 2026-08-27 to "actually produce free energies for reference".
> So the identity of these numbers is **a reference quantity / control**, not a stage 0 product --
> the stage 0 product is still only the density-of-states route.

**The reference level is not settled**: `r2SCAN-3c` or `CCSD(T)/cc-pVTZ` numerical frequencies, undecided.
Until then this script makes one **free preliminary benchmark** using the **B3LYP/6-31G(2df,p) frequencies QM9 ships**.
B3LYP/6-31G(2df,p) is not itself a high-level reference -- it is simply a second set of
frequencies already on disk, free, and of exactly the same provenance as stage 1. **What it gives
is a measured scale for "how much does a frequency difference move the free energy", not the absolute frequency error of MACE-OFF23-SC.** The absolute error still waits on a high-level reference.

Usage:
    python scripts/calibration/s0_package2_hessian_benchmark.py [--n-embed 50] [--skip-delta-scan]
Products:
    analysis/package2/<species>/{basin_census.json, lowest.xyz, hessian.json}
    analysis/frequency_benchmark.json
    analysis/hessian_rrho_free_energy.json
"""
import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

def _repo_root():
    """Directory holding the openQHA package, found by walking up from this file.

    Depth-independent on purpose: this file keeps working wherever under the
    repository it is moved to. The earlier move into `scripts/_superseded/` broke
    every `parents[1]` in the moved files silently, which is what this replaces.
    """
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


sys.path.insert(0, str(_repo_root()))

from openqha import conformers, config, engine, hessian, thermo

T_REF = None   # supplied by the configuration; see below
OUT = _repo_root() / "analysis"
PKG2 = OUT / "package2"

# --------------------------------------------------------------------------------------
# 7 species. The indices come from the 11 edges of the stage 2 production configuration (C3H6O1N0 {18,35,44,46,48} + C2H5O1N1 {19,36}).
#
# **External symmetry number and electronic ground-state degeneracy: declared explicitly, never derived automatically**.
# Each carries its reason; the contested one (oxetane) is decided at run time from the actual optimised structure.
CFG = config.load()
_P2 = config.package(2, CFG)

# The 7 species, their external symmetry numbers and electronic degeneracies all come from this
# repository own configuration, no longer derived from the stage 2 production configuration. A missing one raises; there is no default.
SPECIES = {}
for _e in config.edges(CFG):
    for _q in config.edge_species(_e):
        if _q not in SPECIES:
            _s = config.species(_q, CFG)
            SPECIES[_q] = dict(name=_s["name"], smiles=_s["smiles"],
                               sigma=int(_s["symmetry_number"]), g0=int(_s["electronic_degeneracy"]),
                               sigma_reason=_s["symmetry_reason"],
                               conditional=_s.get("conditional"))

DELTA_SCAN_SPECIES = _P2["delta_scan_species"]
T_REF = config.temperature(CFG)


def load_qm9_row(qid):
    return config.qm9_row(qid, CFG)


def load_edges():
    """The edge set comes from **this repository own configuration**. It used to be parsed verbatim
    from the stage 2 production configuration; when stage 0 became independent on 2026-08-28 it was
    transcribed once into configs/openqha.yaml, and this repository is the source of truth from then on. The origin of the transcription is in the configuration."""
    return config.edges(CFG)


def edge_species(edge):
    return config.edge_species(edge)


def planarity(positions, ring_idx):
    """Ring planarity: the maximum distance to the best-fit plane (A). Used to decide the symmetry number of oxetane."""
    p = np.asarray(positions)[list(ring_idx)]
    p = p - p.mean(0)
    _, _, vt = np.linalg.svd(p)
    return float(np.abs(p @ vt[2]).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-embed", type=int, default=conformers.N_EMBED_DEFAULT)
    ap.add_argument("--skip-delta-scan", action="store_true")
    ap.add_argument("--only", default=None, help="run a single QM9 index, for a smoke test")
    args = ap.parse_args()

    calc, engine_name, prov = engine.calculator()
    edges = load_edges()
    print("=" * 100)
    print("package 2 -- Hessian side branch and frequency benchmark   engine {}   T = {} K".format(engine_name, T_REF))
    print("=" * 100)
    print("configuration {}".format(CFG["_path"]))
    print("{} edge(s)".format(len(edges)))
    print("species: {}".format(len(SPECIES)))
    print("weights {}".format(prov["weights_path"]))
    print()

    qids = [args.only] if args.only else list(SPECIES)
    results = {}
    t_all = time.time()

    for qid in qids:
        spec = SPECIES[qid]
        row = load_qm9_row(qid)
        xyz_ref = config.qm9_xyz(qid, CFG)
        d = PKG2 / spec["name"]
        d.mkdir(parents=True, exist_ok=True)

        print("-" * 100)
        print("{}  {}  {}".format(qid, spec["name"], spec["smiles"]))
        t0 = time.time()

        # ---- 1. basin census (the package 1 module; only the lowest basin is taken here) ----
        rec, basins, mol = conformers.census(
            spec["smiles"], calc, name=spec["name"], n_embed=args.n_embed,
            fmax=conformers.FMAX_CENSUS_EV_A, reference_xyz=xyz_ref)
        print("   embedded {} -> deduplicated after the force field {} -> optimised on the potential {} -> {} basin(s) finally (threshold {} A), "
              "in {:.0f} s".format(
                  rec["n_embed_returned"], rec.get("n_after_forcefield_prune", "-"),
                  rec["n_conformers_optimised"], rec["n_basins"],
                  rec["dedup_threshold_A"], time.time() - t0))
        if rec["merge_energy_warnings"]:
            w = rec["merge_energy_warnings"]
            print("   ** merge consistency warning: {} pair(s) judged to be the same basin differ in energy by more than {} kcal/mol "
                  "(maximum {:.3f})".format(len(w), conformers.MERGE_ENERGY_WARN_KCAL,
                                         max(x["energy_difference_kcal"] for x in w)))
        print("   relative basin energies (kcal/mol): {}".format(
            " ".join("{:.3f}".format(x) for x in rec["basin_relative_kcal"][:8])))
        print("   {} within 1 kT, {} within 5 kT, Boltzmann weight of the lowest basin {:.4f}".format(
            rec["populations"]["n_within_1kT"], rec["populations"]["n_within_5kT"],
            rec["populations"]["weight_of_lowest"]))
        ref = rec.get("reference_geometry", {})
        if ref:
            print("   the native QM9 B3LYP geometry relaxes into basin {} (best rms deviation {:.4f} A)".format(
                ref["nearest_basin_rank_zero_based"],
                min(ref["rms_to_each_basin_A"].values())))
        if rec["n_graph_changed"]:
            print("   ** warning: the connectivity matrix changed during optimisation for {} conformer(s)".format(rec["n_graph_changed"]))
        conformers.dump_json(rec, d / "basin_census.json")

        # ---- 2. the conditional symmetry number (oxetane) -----------------------------------
        sigma = spec["sigma"]
        sigma_note = spec["sigma_reason"]
        cond = spec.get("conditional")
        if cond:
            tol = float(cond["planarity_tol_A"])
            ring = [i for i, z in enumerate(basins[0].numbers) if z in (6, 8)]
            flat = planarity(basins[0].positions, ring)
            sigma_note += "  |  measured ring planarity (maximum distance to the best-fit plane) = {:.4f} A".format(flat)
            if flat > tol:
                sigma = int(cond["symmetry_number_if_puckered"])
                sigma_note += "  -> puckered (> {} A), taking sigma = {}".format(tol, sigma)
            else:
                sigma_note += "  -> nearly planar (<= {} A), taking sigma = {}".format(tol, sigma)
            print("   {} conditional symmetry number: planarity {:.4f} A (threshold {}) -> sigma = {}".format(
                spec["name"], flat, tol, sigma))

        # ---- 3. one Hessian and one free energy per thermally accessible basin --------------
        # Acceptance criterion 3 and the first error source in ChemRxiv 2026 are both
        # the multi-conformer sum -- taking only the lowest basin misses a conformer that is not lowest in electronic energy but is lowest in free energy, and this system has one measured example.
        kt = thermo.KB_KCAL * T_REF
        rel = np.asarray(rec["basin_relative_kcal"])
        treat = [i for i in range(len(basins)) if rel[i] <= 5.0 * kt]
        per_basin, saddles = [], []
        for i in treat:
            a = basins[i]
            e_i, fm_i, ok_i, ns_i = conformers.optimise(
                a, calc, fmax=conformers.FMAX_HESSIAN_EV_A, steps=2000)
            if not ok_i:
                raise RuntimeError(
                    "{} basin {} did not converge to {} eV/A (stopped at {:.2e}) -- the residual force would contaminate the curvature, "
                    "so a Hessian on it is refused".format(spec["name"], i,
                                                  conformers.FMAX_HESSIAN_EV_A, fm_i))
            t1 = time.time()
            hh, asym_i = hessian.finite_difference_hessian(a, calc, delta=hessian.DELTA_A)
            hr = hessian.project_and_diagonalise(hh, a.get_masses(), a.get_positions())
            hr.update(hessian_asymmetry_eV_A2=asym_i, delta_A=hessian.DELTA_A,
                      seconds=time.time() - t1, energy_eV=e_i, max_force_eV_A=fm_i)
            if hr["n_imaginary"]:
                # not a minimum -> removed from the free-energy sum and recorded verbatim. **Never dropped
                # silently**: a structure the deduplication called a "basin" turning out to be a saddle point is itself a finding.
                nu_i = np.asarray(hr["frequencies_cm_inv"])
                saddles.append(dict(
                    basin=i, energy_eV=e_i, max_force_eV_A=fm_i,
                    n_imaginary=hr["n_imaginary"],
                    imaginary_frequencies_cm_inv=[float(x) for x in nu_i[nu_i < 0]],
                    relative_E_kcal=float((e_i - min(rec["basin_energies_eV"]))
                                          * conformers.EV_TO_KCAL),
                    is_reference_basin=bool(
                        ref and ref.get("nearest_basin_rank_zero_based") == i),
                    note="has an imaginary frequency and is not a minimum -- removed from the free-energy sum"))
                conformers.write_xyz(a, d / "saddle{}.xyz".format(i),
                                     "{} SADDLE {} on {}  E = {:.8f} eV  "
                                     "imaginary = {}".format(
                                         spec["name"], i, engine_name, e_i,
                                         np.round(nu_i[nu_i < 0], 2).tolist()))
                print("   ** basin {} has {} imaginary frequency/frequencies ({} cm^-1) -- a saddle point, not a minimum, "
                      "removed from the free-energy sum{}".format(
                          i, hr["n_imaginary"],
                          " ".join("{:.1f}".format(x) for x in nu_i[nu_i < 0]),
                          "; **and it is exactly the structure the reference geometry sits in**"
                          if ref and ref.get("nearest_basin_rank_zero_based") == i else ""))
                continue
            gi = thermo.g_minus_eel(a.get_masses(), a.get_positions(),
                                    hr["frequencies_cm_inv"], symmetry_number=sigma,
                                    degeneracy=spec["g0"], temperature_K=T_REF, qrrho=True)
            per_basin.append(dict(basin=i, energy_eV=e_i, max_force_eV_A=fm_i,
                                  hessian=hr, thermo=gi))
            conformers.write_xyz(a, d / "basin{}.xyz".format(i),
                                 "{} basin {} on {}  E = {:.8f} eV".format(
                                     spec["name"], i, engine_name, e_i))

        if not per_basin:
            raise RuntimeError("every thermally accessible structure of {} is a saddle point -- no usable minimum".format(
                spec["name"]))
        e0 = min(b["energy_eV"] for b in per_basin)
        for b in per_basin:
            b["relative_E_kcal"] = (b["energy_eV"] - e0) * conformers.EV_TO_KCAL
            b["relative_G_kcal"] = (b["relative_E_kcal"]
                                    + b["thermo"]["G_minus_Eel_kcal"]
                                    - per_basin[0]["thermo"]["G_minus_Eel_kcal"])
        print("   {} thermally accessible basin(s) (within 5 kT = {:.3f} kcal/mol), a Hessian for each:".format(
            len(per_basin), 5.0 * kt))
        for b in per_basin:
            print("      basin {}: E_rel {:+7.4f}   (G-E_el) {:+9.4f}   G_rel {:+7.4f} kcal/mol, "
                  "lowest frequency {:7.2f} cm^-1".format(
                      b["basin"], b["relative_E_kcal"], b["thermo"]["G_minus_Eel_kcal"],
                      b["relative_G_kcal"], min(b["hessian"]["frequencies_cm_inv"])))

        # multi-conformer sum:  G_tot = -kT ln sum_i exp(-G_i / kT)
        g_rel = np.array([b["relative_G_kcal"] for b in per_basin])
        g_multi = -kt * np.log(np.exp(-g_rel / kt).sum())
        weights = np.exp(-g_rel / kt); weights = weights / weights.sum()
        i_gmin = int(np.argmin(g_rel))
        multi = dict(
            n_basins_summed=len(per_basin),
            relative_G_kcal=[float(x) for x in g_rel],
            weights=[float(x) for x in weights],
            multiconformer_correction_kcal=float(g_multi),
            lowest_by_electronic_energy=int(np.argmin(
                [b["relative_E_kcal"] for b in per_basin])),
            lowest_by_free_energy=i_gmin,
            ranking_disagrees=bool(i_gmin != 0))
        # A caution about near-degenerate basins: "two basins" less than 1 kT apart are often two shallow
        # wells on the same nearly free rotor. Summing them as two independent oscillators double counts
        # -- the correct treatment is a one-dimensional hindered rotor (implemented in scripts/calibration/s0_lowfreq_and_separable_terms.py).
        near = [int(i) for i in range(len(g_rel))
                if i != 0 and abs(g_rel[i] - g_rel[0]) < kt]
        if near:
            multi["near_degenerate_basins"] = near
            multi["caveat"] = (
                "basin {} is less than 1 kT ({:.3f} kcal/mol) from the lowest basin in free energy. Such "
                "near-degenerate minima are usually shallow wells on one nearly free rotor, and summing them as "
                "independent oscillators **double counts**; this multi-conformer correction is therefore a "
                "**diagnostic and an upper bound**, not a result to use directly. The correct treatment is a one-dimensional hindered rotor.".format(near, kt))
            print("   ** near-degeneracy caution: basin {} is less than 1 kT from the lowest basin -- the "
                  "multi-conformer correction here is an upper bound, not a result".format(near))
        print("   multi-conformer sum: relative free energies {}; weights {}; correction {:+.4f} kcal/mol{}".format(
            " ".join("{:+.3f}".format(x) for x in g_rel),
            " ".join("{:.3f}".format(x) for x in weights), g_multi,
            "   ** the lowest in electronic energy and the lowest in free energy are not the same basin" if multi["ranking_disagrees"] else ""))

        # ---- 4. the frequency benchmark uses **the basin the reference geometry sits in** ---
        # The QM9 frequencies were computed on the QM9 geometry; comparing them against the frequencies
        # of a different conformer mixes a conformer difference into the curvature difference. Comparing within one conformer removes that contamination.
        ref_rank = ref.get("nearest_basin_rank_zero_based", 0) if ref else 0
        pick = next((k for k, b in enumerate(per_basin) if b["basin"] == ref_rank), None)
        if pick is None:
            pick = 0
            print("   ** the structure the reference geometry sits in (basin {}) cannot be used for the benchmark (a saddle point, or outside the thermally accessible window), "
                  "falling back to basin {} -- so a conformer difference re-enters the benchmark".format(
                      ref_rank, per_basin[0]["basin"]))
        atoms = basins[per_basin[pick]["basin"]]
        hrec = per_basin[pick]["hessian"]
        g = per_basin[pick]["thermo"]
        e_low = per_basin[pick]["energy_eV"]
        fmax_low = per_basin[pick]["max_force_eV_A"]
        nu = np.asarray(hrec["frequencies_cm_inv"])
        conformers.write_xyz(basins[0], d / "lowest.xyz",
                             "{} lowest-energy basin on {}  E = {:.8f} eV".format(
                                 spec["name"], engine_name, per_basin[0]["energy_eV"]))
        print("   rigid modes: rank {} -> {} removed; largest |zero eigenvalue| {:.3e}, smallest vibrational eigenvalue {:.3e}, "
              "separation ratio {:.3e}".format(
                  hrec["rigid_subspace_rank"], hrec["n_rigid_modes_removed"],
                  hrec["max_abs_rigid_eigenvalue"], hrec["min_vibrational_eigenvalue"],
                  hrec["separation_gap_ratio"]))
        print("   the benchmark uses basin {} (the one the reference geometry sits in), {} frequencies, lowest {:.2f} cm^-1".format(
            per_basin[pick]["basin"], len(nu), nu.min()))
        print("   A_vib {:+9.4f}   A_rot {:+9.4f}   A_trans {:+9.4f}   A_elec {:+9.4f}"
              "   -> (G - E_el) {:+9.4f} kcal/mol".format(
                  g["vibrational"]["A_vib_kcal"], g["rotational"]["A_rot_kcal"],
                  g["translational"]["value_kcal"], g["electronic"]["A_elec_kcal"],
                  g["G_minus_Eel_kcal"]))

        # ---- 5. the benchmark against the native QM9 B3LYP frequencies ----------------------
        qm9_nu = np.array(sorted(float(v) for v in row["frequencies"].split()))
        bench = thermo.direct_shift(nu, qm9_nu, T_REF)
        bench.update(
            conformer_used=int(per_basin[pick]["basin"]),
            conformer_is_reference_basin=bool(per_basin[pick]["basin"] == ref_rank),
            pairing="paired one by one in ascending frequency. The two sets now come from **the same conformer** "
                    "(the basin the native QM9 geometry relaxes into), so the conformer difference is excluded; but the mode "
                    "assignment still has no overlap test, so a per-mode deviation is overestimated wherever the mode order swaps",
            reference_level="B3LYP/6-31G(2df,p) (native QM9)",
            reference_is_high_level=False)
        print("   vs QM9 B3LYP: mean absolute deviation {:.2f}, rms deviation {:.2f}, maximum {:.2f} cm^-1; "
              "signed mean {:+.2f}".format(
                  bench["mean_absolute_deviation_cm_inv"],
                  bench["root_mean_square_deviation_cm_inv"],
                  bench["max_absolute_deviation_cm_inv"],
                  bench["signed_mean_deviation_cm_inv"]))
        print("   -> difference in A_vib (measured, assuming no correlation) = {:+.4f} kcal/mol".format(
            bench["delta_A_vib_kcal"]))

        # ---- 6. same-geometry diagnostic: another Hessian, on the QM9 geometry --------------
        from ase.io import read as ase_read
        ref_atoms = ase_read(str(xyz_ref))
        ref_atoms.calc = calc
        f_ref = float(np.abs(ref_atoms.get_forces()).max())
        h2, asym2 = hessian.finite_difference_hessian(ref_atoms, calc, delta=hessian.DELTA_A)
        hrec2 = hessian.project_and_diagonalise(h2, ref_atoms.get_masses(),
                                                ref_atoms.get_positions())
        nu2 = np.asarray(hrec2["frequencies_cm_inv"])
        same_geom = None
        if nu2.shape == qm9_nu.shape and hrec2["n_imaginary"] == 0:
            same_geom = thermo.direct_shift(nu2, qm9_nu, T_REF)
        same_geom_rec = dict(
            residual_force_on_qm9_geometry_eV_A=f_ref,
            n_imaginary=hrec2["n_imaginary"],
            frequencies_cm_inv=[float(x) for x in nu2],
            comparison=same_geom,
            caveat="the QM9 geometry is not a stationary point on the MACE surface (see the residual force above), and "
                   "the Eckart projection no longer separates rotation strictly off a stationary point -- this diagnostic measures a **difference of curvature** and must not be used as an error bar")
        print("   same-geometry diagnostic: residual force on the QM9 geometry {:.4f} eV/A, {} imaginary frequency/frequencies{}".format(
            f_ref, hrec2["n_imaginary"],
            ", mean absolute deviation {:.2f} cm^-1".format(
                same_geom["mean_absolute_deviation_cm_inv"]) if same_geom else ""))

        # ---- 7. the two statistical extremes of the error bar -------------------------------
        bars = {}
        for tag, sig in (("measured_rmse_vs_qm9_b3lyp",
                          bench["root_mean_square_deviation_cm_inv"]),
                         ("literature_egret1_mae_24.4", 24.4)):
            bars[tag] = thermo.error_bar_from_frequency_error(nu, sig, T_REF)
            bars[tag].pop("per_mode_sensitivity_kcal_per_cm")

        results[qid] = dict(
            qm9_index=qid, **{k: spec[k] for k in ("name", "smiles", "g0")},
            symmetry_number=sigma, symmetry_number_reason=sigma_note,
            n_atoms=len(atoms), energy_eV=e_low, max_force_eV_A=fmax_low,
            basin_census=dict(n_basins=rec["n_basins"],
                              relative_kcal=rec["basin_relative_kcal"],
                              populations=rec["populations"],
                              merge_energy_warnings=rec["merge_energy_warnings"],
                              reference_geometry=ref),
            per_basin=per_basin, saddles_rejected=saddles, multiconformer=multi,
            energy_lowest_basin=dict(
                energy_eV=per_basin[0]["energy_eV"],
                G_minus_Eel_kcal=per_basin[0]["thermo"]["G_minus_Eel_kcal"],
                G_minus_Eel_qrrho_kcal=per_basin[0]["thermo"]["G_minus_Eel_qrrho_kcal"],
                G_minus_Eel_multiconformer_kcal=(
                    per_basin[0]["thermo"]["G_minus_Eel_kcal"]
                    + multi["multiconformer_correction_kcal"])),
            hessian=hrec, thermo=g, benchmark_vs_qm9=bench,
            same_geometry_diagnostic=same_geom_rec, error_bars=bars,
            seconds=time.time() - t0)
        conformers.dump_json(results[qid], d / "hessian.json")
        print("   subtotal {:.0f} s".format(time.time() - t0))

    # ---- 8. displacement convergence ----------------------------------------------------
    delta_scan = None
    if not args.skip_delta_scan and DELTA_SCAN_SPECIES in results:
        print("-" * 100)
        print("finite-difference displacement convergence (measured on {})".format(
            SPECIES[DELTA_SCAN_SPECIES]["name"]))
        from ase.io import read as ase_read
        a = ase_read(str(PKG2 / SPECIES[DELTA_SCAN_SPECIES]["name"] / "lowest.xyz"))
        delta_scan = hessian.delta_convergence(a, calc)
        for r in delta_scan:
            print("   delta = {:.3f} A: lowest frequency {:8.2f} cm^-1, maximum deviation from 0.01 A {:.3f}, "
                  "mean {:.3f} cm^-1, symmetry residual {:.2e}".format(
                      r["delta_A"], min(r["frequencies_cm_inv"]),
                      r.get("max_deviation_from_0.01A_cm_inv", 0.0),
                      r.get("mean_abs_deviation_from_0.01A_cm_inv", 0.0),
                      r["asymmetry_eV_A2"]))

    # ---- 9. assembling the 11 edges -----------------------------------------------------
    print("=" * 100)
    print("rigid-rotor harmonic-oscillator reference: the thermodynamic term over 11 edges  Delta (G - E_el)")
    print("=" * 100)
    print("{:24s} {:>11s} {:>11s} {:>11s} {:>11s} {:>11s} {:>11s}".format(
        "edge", "dA_vib", "dA_rot", "dA_trans", "d(G-Eel)", "multiconf", "qRRHO"))
    print("-" * 100)
    edge_rows = {}
    for e in edges:
        a, b = edge_species(e)
        if a not in results or b not in results:
            continue
        ta = results[a]["per_basin"][0]["thermo"]
        tb = results[b]["per_basin"][0]["thermo"]
        la, lb = results[a]["energy_lowest_basin"], results[b]["energy_lowest_basin"]
        d_vib = tb["vibrational"]["A_vib_kcal"] - ta["vibrational"]["A_vib_kcal"]
        d_rot = tb["rotational"]["A_rot_kcal"] - ta["rotational"]["A_rot_kcal"]
        d_tr = tb["translational"]["value_kcal"] - ta["translational"]["value_kcal"]
        d_tot = lb["G_minus_Eel_kcal"] - la["G_minus_Eel_kcal"]
        d_multi = (lb["G_minus_Eel_multiconformer_kcal"]
                   - la["G_minus_Eel_multiconformer_kcal"])
        d_q = lb["G_minus_Eel_qrrho_kcal"] - la["G_minus_Eel_qrrho_kcal"]
        d_e = (lb["energy_eV"] - la["energy_eV"]) * conformers.EV_TO_KCAL
        edge_rows[e] = dict(reactant=a, product=b,
                            delta_A_vib_kcal=d_vib, delta_A_rot_kcal=d_rot,
                            delta_A_trans_kcal=d_tr,
                            delta_G_minus_Eel_kcal=d_tot,
                            delta_G_minus_Eel_multiconformer_kcal=d_multi,
                            delta_G_minus_Eel_qrrho_kcal=d_q,
                            delta_E_el_engine_kcal=d_e,
                            n_basins_summed=[results[a]["multiconformer"]["n_basins_summed"],
                                             results[b]["multiconformer"]["n_basins_summed"]],
                            note="**a reference quantity, not a stage 0 product** -- the stage 0 product is the density-of-states route")
        print("{:24s} {:11.4f} {:11.4f} {:11.4f} {:11.4f} {:11.4f} {:11.4f}".format(
            e, d_vib, d_rot, d_tr, d_tot, d_multi, d_q))
    print()
    print("the translational term is {:.2e} kcal/mol on every edge -- same formula, cancelling digit for digit, which is Delta n = 0 "
          "made visible".format(max(abs(r["delta_A_trans_kcal"]) for r in edge_rows.values())
                              if edge_rows else 0.0))

    # ---- 10. the error bar, at edge level -----------------------------------------------
    # The A_vib deviation of one species is not the quantity we want -- what we want is how much of it survives in the **difference**.
    print()
    print("=" * 100)
    print("error bar   how far Delta(G-E_el) moves when the frequency level changes from "
          "MACE-OFF23-SC to B3LYP/6-31G(2df,p)")
    print("=" * 100)
    print("{:24s} {:>14s} {:>14s} {:>14s}".format(
        "edge", "reactant dA_vib", "product dA_vib", "net shift on the edge"))
    print("-" * 100)
    shifts = {}
    for e, r in edge_rows.items():
        sa = results[r["reactant"]]["benchmark_vs_qm9"]["delta_A_vib_kcal"]
        sb = results[r["product"]]["benchmark_vs_qm9"]["delta_A_vib_kcal"]
        clean = (results[r["reactant"]]["benchmark_vs_qm9"]["conformer_is_reference_basin"]
                 and results[r["product"]]["benchmark_vs_qm9"]["conformer_is_reference_basin"])
        shifts[e] = dict(reactant_shift_kcal=sa, product_shift_kcal=sb,
                         edge_shift_kcal=sb - sa, same_conformer_both_sides=clean)
        print("{:24s} {:14.4f} {:14.4f} {:14.4f}{}".format(
            e, sa, sb, sb - sa, "" if clean else "   (the conformers do not match on one side; unusable)"))
    usable = [v["edge_shift_kcal"] for v in shifts.values()
              if v["same_conformer_both_sides"]]
    per_species = [v["benchmark_vs_qm9"] for v in results.values()]
    pooled = dict(
        n_species=len(per_species),
        mean_absolute_deviation_cm_inv=float(np.mean(
            [x["mean_absolute_deviation_cm_inv"] for x in per_species])),
        root_mean_square_deviation_cm_inv=float(np.sqrt(np.mean(
            [x["root_mean_square_deviation_cm_inv"] ** 2 for x in per_species]))),
        signed_mean_deviation_cm_inv=float(np.mean(
            [x["signed_mean_deviation_cm_inv"] for x in per_species])),
        max_absolute_deviation_cm_inv=float(np.max(
            [x["max_absolute_deviation_cm_inv"] for x in per_species])),
        n_edges_usable=len(usable),
        edge_shift_mean_abs_kcal=float(np.mean(np.abs(usable))) if usable else None,
        edge_shift_max_abs_kcal=float(np.max(np.abs(usable))) if usable else None,
        per_edge=shifts)
    print()
    print("all seven species together: mean absolute deviation {:.2f} cm^-1, rms deviation {:.2f} cm^-1, "
          "signed mean {:+.2f} cm^-1 (MACE systematically high), maximum {:.2f} cm^-1".format(
              pooled["mean_absolute_deviation_cm_inv"],
              pooled["root_mean_square_deviation_cm_inv"],
              pooled["signed_mean_deviation_cm_inv"],
              pooled["max_absolute_deviation_cm_inv"]))
    if usable:
        print("over the {} usable edge(s), the edge-level net shift is mean |{:.4f}|, maximum |{:.4f}| kcal/mol".format(
            len(usable), pooled["edge_shift_mean_abs_kcal"],
            pooled["edge_shift_max_abs_kcal"]))
    print()
    print("**This is not the frequency error bar of MACE-OFF23-SC.** B3LYP/6-31G(2df,p) is not a high-level")
    print("reference; it is itself tens of cm^-1 from coupled cluster. What the numbers above measure is")
    print("**how far the answer moves when the frequency level is changed** -- a scale, not an error.")
    print("The real error bar waits on the reference level decision (r2SCAN-3c or CCSD(T)/cc-pVTZ numerical frequencies).")

    payload = dict(
        error_bar_scale=pooled,
        generated_by="scripts/calibration/s0_package2_hessian_benchmark.py",
        package="package 2 -- the Hessian side branch and the frequency benchmark",
        engine=prov, temperature_K=T_REF,
        config_path=CFG["_path"], edges=edges,
        edges_provenance=CFG.get("edges_provenance"),
        n_species=len(results), species=results,
        delta_convergence=delta_scan,
        edge_thermodynamic_terms=edge_rows,
        reference_level_decision=dict(
            status="not yet settled",
            candidates=["r2SCAN-3c", "CCSD(T)/cc-pVTZ numerical frequencies"],
            interim="native QM9 B3LYP/6-31G(2df,p) -- free, but **not a high-level reference**",
            consequence="until a high-level reference is in place, the error bars in this file are a **scale**, not an **error**"),
        identity_note=("the rigid-rotor harmonic-oscillator free energy is a **reference quantity / control**; it overturns an earlier premise, "
                       "produced on request on 2026-08-27. "
                       "The stage 0 product is still only the density-of-states route."),
        total_seconds=time.time() - t_all)

    conformers.dump_json(payload, OUT / "frequency_benchmark.json")
    conformers.dump_json(
        dict(temperature_K=T_REF, engine=prov,
             species={k: dict(name=v["name"],
                              symmetry_number=v["symmetry_number"],
                              symmetry_number_reason=v["symmetry_number_reason"],
                              electronic_degeneracy=v["g0"],
                              n_basins=v["basin_census"]["n_basins"],
                              energy_lowest_basin=v["energy_lowest_basin"],
                              multiconformer=v["multiconformer"],
                              per_basin=[dict(basin=b["basin"],
                                              energy_eV=b["energy_eV"],
                                              relative_E_kcal=b["relative_E_kcal"],
                                              relative_G_kcal=b["relative_G_kcal"],
                                              frequencies_cm_inv=b["hessian"]["frequencies_cm_inv"],
                                              G_minus_Eel_kcal=b["thermo"]["G_minus_Eel_kcal"])
                                         for b in v["per_basin"]]) for k, v in results.items()},
             edges=edge_rows,
             identity_note=payload["identity_note"]),
        OUT / "hessian_rrho_free_energy.json")
    print()
    print("written: analysis/frequency_benchmark.json")
    print("      analysis/hessian_rrho_free_energy.json")
    print("      analysis/package2/<species>/{basin_census.json, lowest.xyz, hessian.json}")
    print("total time {:.0f} s".format(time.time() - t_all))


if __name__ == "__main__":
    main()
