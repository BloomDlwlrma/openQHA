"""02d -- can a quasi-harmonic `nu` stand in for a Hessian `omega`, term by term?

EXAMPLE. It computes; it decides nothing on its own. The verdict needs 02c's number
for the size of a quantum-chemistry error, and this script prints its result next to
that number rather than inventing a threshold.

THE QUESTION
------------
Branch B reads a frequency off a variance; branch C reads one off a curvature. In a
harmonic well they are the same number, and if that holds well enough on a real
molecule then

    G_total = E_el + G_trans + G_rot + sum_i [ ZPE_i + H_i - T*S_i ]

can be built from ONE spectrum instead of two, and the hybrid splice is unnecessary.
"Well enough" has to be measured separately for each of the three terms, because they
weight the spectrum completely differently: ZPE and enthalpy are dominated by the stiff
end, entropy by the soft end.

THREE STAGES, AND WHY THE ORDER MATTERS
---------------------------------------
1. HARMONIC LIMIT. A synthetic trajectory sampled analytically from the molecule's own
   MACE Hessian. The surface IS harmonic, so `nu == omega` exactly and any deviation is
   sampling noise in the covariance -- nothing else can contribute. Sweeping the frame
   count therefore measures **how many frames each thermodynamic term needs**, which is
   a different number for each of them.

2. REAL TRAJECTORY. The same analysis on molecular dynamics on the real surface. The
   deviation here is the harmonic-limit deviation PLUS the physics -- anharmonicity,
   internal rotation, basin escape. Stage 1 is what makes those separable.

3. HYBRID. `mode_match.hybrid_spectrum` builds `S_i^final` and the assembled
   `G - E_el` is reported for the pure-Hessian, pure-QHA and hybrid spectra side by
   side, so the price of each choice is a number rather than an argument.

    python examples/02d_qha_frequency_identity/s0_frequency_identity.py \
        --species dsgdb9nsd_000018 --tag 02d_prod --stage harmonic
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))

from openqha import config                                             # noqa: E402
from openqha.potentials import engine                                  # noqa: E402
from openqha.quasi_harmonic import mode_match, qha                     # noqa: E402
from openqha.quasi_harmonic import basin_residence as br               # noqa: E402
from openqha.store import basin_store                                  # noqa: E402
from openqha.thermochem import hessian as hess_mod                     # noqa: E402
from openqha.thermochem import gtotal, thermo                          # noqa: E402

#: 1 eV in kcal/mol -- the same constant stage 2 uses, not a second copy.
EV_TO_KCAL = 23.060547830618307

#: Frame counts for the harmonic-limit sweep. The top of the range is deliberately far
#: past anything a real trajectory will reach: the point is to show where each term
#: converges, and two of them do not converge inside the range a trajectory can afford.
HARMONIC_FRAME_COUNTS = (200, 500, 1000, 2500, 5000, 12500, 50000)


def basin_geometries(species, tag):
    """Every basin of one species, as (symbols, positions), from branch A's product.

    The basin xyz and a stored trajectory do NOT necessarily share an atom order --
    measured 2026-09-08 on acetone, where they differ. Everything downstream of here
    uses the order branch A wrote, and a trajectory whose `symbols` disagree is
    rejected rather than reordered: a silent reordering would scramble the mass
    weighting and every mode with it.
    """
    rec = basin_store.read(species, tag=tag)
    if rec is None:
        raise SystemExit(
            basin_store.missing_message(species, tag)
            + "\n  This example starts from the basins branch A found; to make them:\n"
            "  bash examples/run_chain.sh examples/02d_qha_frequency_identity/branchA.conf deimos")
    # Located by the store, not by the absolute path inside the record: the record was
    # written on the CPU cluster and read here on the GPU one (an113, 2026-09-13).
    lines = basin_store.xyz_for(species, rec, tag=tag).read_text(
        encoding="utf-8").splitlines()
    out, i = [], 0
    while i < len(lines) and lines[i].strip():
        n = int(lines[i].split()[0])
        block = lines[i + 2:i + 2 + n]
        out.append(([ln.split()[0] for ln in block],
                    np.array([[float(c) for c in ln.split()[1:4]] for ln in block])))
        i += n + 2
    return out, rec


def hessian_modes(symbols, positions, calc):
    """omega and its eigenvectors at a geometry branch A already relaxed.

    No re-optimisation: branch A's basins are MACE minima with zero imaginary
    frequencies, and relaxing them again with the same potential would only add a
    difference between this example and the record it reads from.
    """
    from ase import Atoms
    atoms = Atoms(symbols=symbols, positions=positions)
    atoms.calc = calc
    h, asym = hess_mod.analytic_hessian(atoms, calc)
    omega, vec = mode_match.projected_modes(h, atoms.get_masses(), positions, "hessian")
    # The same spectrum through the established route. If these two ever disagree, one
    # of the two projections is wrong and this is where it shows.
    ref = hess_mod.project_and_diagonalise(h, atoms.get_masses(), positions)
    dev = np.abs(np.sort(omega) - np.sort(ref["frequencies_cm_inv"])).max()
    if dev > 1e-6:
        raise ValueError(
            "mode_match.projected_modes and hessian.project_and_diagonalise disagree "
            "by {:.3e} cm^-1 -- the two projections are not the same computation"
            .format(dev))
    return h, omega, vec, atoms.get_masses(), float(asym), float(dev)


def qha_modes(frames, masses, reference):
    """nu and its eigenvectors, superimposed on `reference` rather than on the mean.

    Fitting to the Hessian's own geometry is what puts the two mode sets in one
    cartesian frame; fitting to the trajectory mean leaves a residual rotation between
    them and the overlap matrix then measures that rotation instead of the pairing.
    The covariance is still taken about the trajectory's own mean, which is the
    correct centre.
    """
    fitted, mean, fit_rec = qha.superimpose(frames, masses, reference=reference)
    cov = qha.mass_weighted_covariance(fitted, masses, mean)
    nu, vec = mode_match.projected_modes(cov, masses, mean, "covariance")
    return nu, vec, mean, fit_rec


def compare(nu, v_nu, omega, v_om, temperature_K, label):
    """One row of the comparison: pairing quality plus all four thermodynamic terms."""
    m = mode_match.match(nu, v_nu, omega, v_om)
    t_q = mode_match.thermodynamic_terms(nu, temperature_K)
    t_h = mode_match.thermodynamic_terms(omega, temperature_K)
    return dict(
        label=label, match=m, from_qha=t_q, from_hessian=t_h,
        delta=dict((k, float(t_q[k] - t_h[k]))
                   for k in ("ZPE_kcal", "E_vib_kcal", "TS_kcal", "A_vib_kcal")),
    )


def print_rows(rows, first_col, width=9):
    head = ("{:>" + str(width) + "} {:>10} {:>10} {:>10} {:>10} {:>8} {:>8} {:>5}")
    print(head.format(first_col, "max|dnu|", "dZPE", "dE_vib", "dT*S", "minO", "minBlk",
                      "dup"))
    for key, r in rows:
        m, d = r["match"], r["delta"]
        print(head.format(
            key, "%.2f" % max(abs(x) for x in m["delta_cm"]),
            "%+.4f" % d["ZPE_kcal"], "%+.4f" % d["E_vib_kcal"], "%+.4f" % d["TS_kcal"],
            "%.3f" % m["min_max_overlap"], "%.3f" % m["min_block_overlap"],
            m["n_duplicate_pairings"]))


def stage_harmonic(symbols, positions, calc, temperature_K, frame_counts, seed):
    """Stage 1. No potential is called after the Hessian, so the answer is exact."""
    h, omega, v_om, masses, asym, dev = hessian_modes(symbols, positions, calc)
    rows = []
    for n in frame_counts:
        frames, exact = qha.synthetic_harmonic_trajectory(
            h, masses, positions, temperature_K, n_frames=int(n), seed=int(seed))
        nu, v_nu, mean, _ = qha_modes(frames, masses, positions)
        rows.append((str(n), compare(nu, v_nu, omega, v_om, temperature_K,
                                     "harmonic_{}".format(n))))
    return dict(hessian_asymmetry_eV_A2=asym,
                projection_agreement_cm=dev,
                omega_cm=[float(x) for x in omega],
                masses_amu=[float(x) for x in masses],
                frames_per_dof=[float(n) / len(omega) for n in frame_counts],
                rows={k: v for k, v in rows}), rows, omega, v_om, masses, h


def stage_real(traj_dir, symbols, positions, omega, v_om, masses, temperature_K,
               fractions=(0.2, 0.4, 0.6, 0.8, 1.0)):
    """Stage 2. Prefixes of ONE trajectory: only the length changes."""
    meta = json.loads((traj_dir / "meta.json").read_text(encoding="utf-8"))
    if list(meta["symbols"]) != list(symbols):
        raise ValueError(
            "the trajectory's atom order {} is not branch A's {}. Refusing to reorder: "
            "a silent reordering scrambles the mass weighting and every mode with it."
            .format(meta["symbols"], list(symbols)))
    qha.assert_trajectory_identity(meta)
    frames = np.load(traj_dir / "frames.npy")
    dt_ps = float(meta.get("frame_spacing_fs", float("nan"))) / 1000.0
    rows = []
    for f in fractions:
        n = max(int(len(frames) * f), 3 * len(symbols) + 1)
        nu, v_nu, mean, _ = qha_modes(frames[:n], masses, positions)
        r = compare(nu, v_nu, omega, v_om, temperature_K, "real_{}".format(n))
        r["n_frames"] = int(n)
        r["ps"] = float(n * dt_ps)
        r["residence"] = br.basin_residence(frames[:n], list(symbols))
        rows.append(("{:.1f}ps".format(n * dt_ps), r))
    return dict(trajectory=str(traj_dir), meta=meta,
                rows={k: v for k, v in rows}), rows


def stage_gtotal(nu, omega, masses, min_positions, mean_positions, sigma, degeneracy,
                 e_el_kcal, temperature_K):
    """Stage 3. **The substitution the question is actually about.**

    `G_total` assembled twice from the same molecule, the same sigma and the same
    electronic energy -- once with the Hessian's `omega_i`, once with the
    quasi-harmonic `nu_k`. Every term is a SUM over its spectrum, so this needs no mode
    pairing, no overlap and no cutoff. Whether a pairing could be built is a separate
    question and stage 4 answers it; this stage does not depend on the answer.

    Each route is evaluated at ITS OWN geometry -- the Hessian at the minimum it was
    computed at, the quasi-harmonic route at the trajectory's mean structure -- because
    that is what each one actually has. The consequence is that `G_rot` can differ, and
    `gtotal.compare` labels that difference as geometry rather than letting it be read
    as an effect of the spectrum.
    """
    ref = gtotal.assemble(omega, masses, min_positions, sigma, degeneracy,
                          electronic_energy_kcal=e_el_kcal,
                          temperature_K=temperature_K, label="Hessian omega")
    test = gtotal.assemble(nu, masses, mean_positions, sigma, degeneracy,
                           electronic_energy_kcal=e_el_kcal,
                           temperature_K=temperature_K, label="QHA nu")
    return dict(hessian=ref, qha=test,
                comparison=gtotal.compare(ref, test, "hessian", "qha"))


def stage_hybrid(nu, v_nu, omega, v_om, masses, positions, symbols, sigma, degeneracy,
                 temperature_K, nu_cuts, min_block_overlap):
    """Stage 4. `G - E_el` from three spectra, so the price of the choice is a number."""
    m = mode_match.match(nu, v_nu, omega, v_om)
    out = []
    for cut in nu_cuts:
        spec, rec = mode_match.hybrid_spectrum(nu, omega, m, cut,
                                               min_block_overlap=min_block_overlap)
        g = thermo.g_minus_eel(masses, positions, spec, sigma, degeneracy,
                              temperature_K=temperature_K)
        out.append(dict(nu_cut_cm=float(cut), spectrum_record=rec,
                        terms=mode_match.thermodynamic_terms(spec, temperature_K),
                        G_minus_Eel_kcal=float(g["G_minus_Eel_kcal"])))
    pure = {}
    for name, spec in (("hessian", omega), ("qha", nu)):
        good = np.asarray(spec, float)
        good = good[np.isfinite(good) & (good > 0)]
        if len(good) != len(omega):
            pure[name] = dict(refused="spectrum has {} usable modes, the molecule has "
                                      "{}".format(len(good), len(omega)))
            continue
        g = thermo.g_minus_eel(masses, positions, good, sigma, degeneracy,
                              temperature_K=temperature_K)
        pure[name] = dict(terms=mode_match.thermodynamic_terms(good, temperature_K),
                          G_minus_Eel_kcal=float(g["G_minus_Eel_kcal"]))
    return dict(pure=pure, hybrid=out, match=m)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species", default="dsgdb9nsd_000018",
                    help="QM9 index. Acetone 000018, propanal 000035.")
    # The default is what examples/02d_qha_frequency_identity/branchA.conf writes. Until
    # 2026-09-11 it was "prod", which no conf in this repository sets.
    ap.add_argument("--tag", default="02d_prod",
                    help="branch A tag to read the basins from; = branchA.conf's TAG")
    ap.add_argument("--stage", default="harmonic",
                    choices=("harmonic", "real", "all"))
    ap.add_argument("--traj-tag", default=None,
                    help="branch B run tag under $S0_RUNS_ROOT/qha (stage real)")
    ap.add_argument("--basin", type=int, default=None,
                    help="one basin only; default every basin branch A found")
    ap.add_argument("--nu-cut", default="25,50,100,150",
                    help="cm^-1, comma separated")
    ap.add_argument("--min-block-overlap", type=float,
                    default=mode_match.DEFAULT_MIN_BLOCK_OVERLAP)
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = config.load()
    temperature = config.temperature(cfg)
    cuts = [float(x) for x in args.nu_cut.split(",")]
    geoms, rec_a = basin_geometries(args.species, args.tag)
    calc, engine_name, prov = engine.calculator(device="cpu")

    print("=" * 92)
    print("02d  nu(QHA) vs omega(Hessian) -- {}  tag {}  T = {} K".format(
        args.species, args.tag, temperature))
    print("engine {}   {} basin(s) from branch A".format(engine_name, len(geoms)))
    print("=" * 92)

    report = dict(example="02d", species=args.species, tag=args.tag,
                  temperature_K=temperature, engine=engine_name, provenance=prov,
                  nu_cuts_cm=cuts, min_block_overlap=args.min_block_overlap,
                  basins={})
    t0 = time.time()
    for b, (symbols, positions) in enumerate(geoms):
        if args.basin is not None and b != args.basin:
            continue
        basin_rec = rec_a["basins"][b]
        sigma = int(basin_rec["thermo"]["rotational"]["symmetry_number"])
        degen = int(basin_rec.get("electronic_degeneracy", 1))
        entry = dict(basin_index=b, sigma=sigma, electronic_degeneracy=degen)
        print()
        print("-" * 92)
        print("basin {}  sigma {}  rel {:.4f} kcal/mol".format(
            b, sigma, float(basin_rec["relative_kcal"])))
        print("-" * 92)

        harm, hrows, omega, v_om, masses, h = stage_harmonic(
            symbols, positions, calc, temperature, HARMONIC_FRAME_COUNTS, args.seed)
        entry["harmonic_limit"] = harm
        print("omega: {} modes, {:.2f} .. {:.2f} cm^-1;  hessian asymmetry {:.2e}"
              .format(len(omega), omega.min(), omega.max(),
                      harm["hessian_asymmetry_eV_A2"]))
        print()
        print("STAGE 1 -- harmonic limit (the surface IS harmonic; deviation = sampling"
              " noise only)")
        print_rows(hrows, "frames")

        if args.stage in ("real", "all"):
            if not args.traj_tag:
                raise SystemExit("--stage real needs --traj-tag")
            d = (Path(config.runs_dir("qha", cfg)) / args.traj_tag / args.species
                 / "basin{:02d}".format(b) / "seed00")
            if not (d / "frames.npy").exists():
                print("  no trajectory at {} -- stage 2 skipped for this basin"
                      .format(d))
                entry["real"] = dict(skipped=str(d))
            else:
                real, rrows = stage_real(d, symbols, positions, omega, v_om, masses,
                                         temperature)
                entry["real"] = real
                print()
                print("STAGE 2 -- real trajectory (deviation = stage 1 + the physics)")
                print_rows(rrows, "length")
                xd = [r["residence"]["distinct_basin_crossings"] for _, r in rrows]
                xs = [r["residence"]["symmetry_equivalent_crossings"] for _, r in rrows]
                print("  basin crossings at full length: {} distinct, {} "
                      "symmetry-equivalent".format(xd[-1], xs[-1]))
                nu_full, v_nu_full, mean_full, _ = qha_modes(
                    np.load(d / "frames.npy"), masses, positions)

                # ---- STAGE 3: the substitution itself. No pairing anywhere in it.
                e_el = float(basin_rec["energy_eV"]) * EV_TO_KCAL
                gt = stage_gtotal(nu_full, omega, masses, positions, mean_full,
                                  sigma, degen, e_el, temperature)
                entry["gtotal"] = gt
                print()
                print("STAGE 3 -- G_total with nu_k in place of omega_i, term by term "
                      "(kcal/mol)")
                print(gtotal.format_table([gt["hessian"], gt["qha"]], gt["comparison"]))
                c = gt["comparison"]
                print()
                print("  G_trans is identical by construction: no frequency enters "
                      "Sackur-Tetrode.")
                print("  G_rot moved {:+.4f} kcal/mol, and that is the GEOMETRY "
                      "(minimum vs trajectory mean), not the spectrum."
                      .format(c["rotational_geometry_shift_kcal"]))
                print("  everything the spectrum can touch: {:+.4f} kcal/mol"
                      .format(c["vibrational_only_delta_kcal"]))
                print()
                print("  which band carries which term -- Hessian omega")
                print(gtotal.format_bands(gt["hessian"]))
                print()
                print("  which band carries which term -- QHA nu")
                print(gtotal.format_bands(gt["qha"]))

                # ---- STAGE 4: could a per-mode splice be built at all?
                entry["hybrid"] = stage_hybrid(
                    nu_full, v_nu_full, omega, v_om, masses, positions, symbols,
                    sigma, degen, temperature, cuts, args.min_block_overlap)
                print()
                print("STAGE 4 -- could a per-mode hybrid be built? (G - E_el, "
                      "kcal/mol)")
                for hy in entry["hybrid"]["hybrid"]:
                    print("  {:>22} {:>14.4f}   ({} modes from QHA, {} rejected on "
                          "overlap)".format(
                              "hybrid nu_cut=%g" % hy["nu_cut_cm"],
                              hy["G_minus_Eel_kcal"],
                              hy["spectrum_record"]["n_from_qha"],
                              hy["spectrum_record"]["n_rejected_by_overlap"]))
        report["basins"][str(b)] = entry

    report["wall_seconds"] = time.time() - t0
    out = Path(args.out) if args.out else (
        ROOT / "analysis" / "qha" / args.tag /
        "{}_02d_frequency_identity.json".format(args.species))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print()
    print("written {}  ({:.1f} s)".format(out, report["wall_seconds"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
