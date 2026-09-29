"""The GFN2 seam: CREST `--entropy` on CREST's native engine beside our assembly.

WHY THIS CALCULATION EXISTS
---------------------------
`msrrho_ensemble` implements the assembly of Pracht & Grimme (Chem. Sci. 2021, 12, 6551)
inside openQHA. The only way to know the implementation matches the original is to run
the original on the engine it was validated on (GFN2-xTB) and feed its OWN inputs to our
code. So: CREST `--entropy` at GFN2 (at least twice; the sampling is stochastic), then

    algebraic terms   S'_conf (eq. 10), H_conf (eq. 14), Cp_conf (eq. 13) from CREST's
                      conformer energies and CREST's degeneracies must equal CREST's
                      printout to 1e-4 cal/mol/K: same numbers in, same numbers out.
    Hessian terms     S_ref (msRRHO of the reference structure) and dS_bar need per-
                      conformer Hessians, which CREST computes internally (numerical,
                      its own GFN2) and does not leave on disk. We compute them with
                      `xtb --hess` at CREST's geometries; the difference is the Hessian
                      implementation, not the assembly, and is reported as such.

WHAT CREST ACTUALLY DOES (read from crest-master/src/entropy on 2026-09-16)
--------------------------------------------------------------------------
* S'_conf uses `enantiofac` as g' (1, or 2 for every conformer in a point-group class in
  which a mirror pair was found: `intraconfRMSD`, propagated by the `symmetries` label).
* H_conf and Cp_conf use `introtscal` = g_rot * cores / g_sym, the number written to
  `cre_degen2`; a constant factor cancels there, so only the core count matters.
* Populations for dS_bar are Boltzmann weights of the conformer FREE energies with the
  `cre_degen2` degeneracies; S_ref is the msRRHO entropy of the INPUT structure (the
  `bhess` reference; with the calculator interface a plain Hessian at that geometry).
* The point-group label is the fragile part: on propanal the third conformer read `c1`
  in one run and `cs` in the next, and S'_conf moved by R ln 2 (3.1495 vs 2.7199). Our
  own assembly classifies chirality from the geometry (`degeneracy.symmetry_class`).

Records: `msrrho/thermo/gfn2.thermo_msrrho.{out,toml}` in the standard Records shape plus a `[Crest]`
block (CREST's numbers per run and their spread) and a `[Seam]` block (ours minus theirs,
term by term). GFN2 is never a reference for MACE.
"""
import math
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

from ..conformer_search import crest as crest_mod
from ..conformer_search import degeneracy, symmetry
from ..qm_interfaces import xtb
from ..store import layout, property as prop, report
from . import msrrho_ensemble as me
from . import thermo

LEVEL = "gfn2"
STEP = "thermo_msrrho"
PROGNAME = "openQHA thermo_msrrho (crest entropy seam)"
HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KCAL = 627.509474
#: CREST's translational entropy is at 1 atm with historical constants; ours at 1 bar.
#: Measured on acetone 2026-09-16: 38.081 vs 38.125 cal/mol/K. Applied when comparing.
CREST_TRANS_OFFSET_CAL = 38.1250 - 38.081
#: Agreement required of the algebraic terms (same inputs, same formula).
TOL_ALGEBRAIC_CAL = 1e-4
#: Agreement expected of the Hessian-dependent reference entropy (xtb analytic Hessian
#: against CREST's numerical one at the same geometry; measured 0.011 on propanal).
TOL_S_REF_CAL = 0.05

SCHEMA = dict(me.SCHEMA)
SCHEMA["Crest"] = {
    "N_RUNS": ("Integer", None, "CREST --entropy runs read"),
    "RUN": ("Integer", None, "run index"),
    "N_CONFORMERS": ("Integer", None, "conformers in that run"),
    "S_CONF": ("Double", "cal/mol/K", "CREST Sconf at 298.15 K"),
    "DS_RRHO": ("Double", "cal/mol/K", "CREST dSrrho at 298.15 K"),
    "S_CONF_TOTAL": ("Double", "cal/mol/K", "CREST S(total) = Sconf + dSrrho"),
    "H_CONF_E": ("Double", "kcal/mol", "CREST H(T)-H(0) (conf) on electronic energies"),
    "CP_CONF_E": ("Double", "cal/mol/K", "CREST Cp(conf) on electronic energies"),
    "H_CONF_G": ("Double", "kcal/mol", "CREST final H(T)-H(0) on free energies"),
    "CP_CONF_G": ("Double", "cal/mol/K", "CREST final Cp(total) on free energies"),
    "S_REF": ("Double", "cal/mol/K", "CREST msRRHO entropy of the reference structure"),
    "S_AVG": ("Double", "cal/mol/K", "CREST population-averaged msRRHO entropy"),
    "STHR": ("Double", "cm^-1", "CREST rotor cutoff"),
    "ITHR": ("Double", "cm^-1", "CREST imaginary cutoff"),
    "FSCAL": ("Double", None, "CREST frequency scaling"),
    "WALL": ("Double", "s", "CREST wall time"),
    "S_CONF_SPREAD": ("Double", "cal/mol/K", "max - min of Sconf over the runs"),
    "S_TOTAL_SPREAD": ("Double", "cal/mol/K", "max - min of S(total) over the runs"),
}
SCHEMA["Seam"] = {
    "RUN": ("Integer", None, "run the seam was evaluated on"),
    "S_CONF_OURS": ("Double", "cal/mol/K", "eq. 10 with CREST's energies and enantiofac"),
    "S_CONF_DELTA": ("Double", "cal/mol/K", "ours - CREST"),
    "H_CONF_OURS": ("Double", "kcal/mol", "eq. 14 with CREST's electronic energies and cre_degen2"),
    "H_CONF_DELTA": ("Double", "kcal/mol", "ours - CREST (electronic-energy block)"),
    "CP_CONF_OURS": ("Double", "cal/mol/K", "eq. 13 with CREST's electronic energies and cre_degen2"),
    "CP_CONF_DELTA": ("Double", "cal/mol/K", "ours - CREST (electronic-energy block)"),
    "H_CONF_G_OURS": ("Double", "kcal/mol", "eq. 14 on our free energies G_i with cre_degen2"),
    "H_CONF_G_DELTA": ("Double", "kcal/mol", "ours - CREST final (Hessian implementation)"),
    "CP_CONF_G_OURS": ("Double", "cal/mol/K", "eq. 13 on our free energies G_i with cre_degen2"),
    "CP_CONF_G_DELTA": ("Double", "cal/mol/K", "ours - CREST final (Hessian implementation)"),
    "S_REF_OURS": ("Double", "cal/mol/K", "msRRHO of the reference structure from xtb --hess, 1 atm"),
    "S_REF_DELTA": ("Double", "cal/mol/K", "ours - CREST (Hessian implementation)"),
    "DS_BAR_OURS": ("Double", "cal/mol/K", "population average minus reference, our Hessians"),
    "DS_BAR_DELTA": ("Double", "cal/mol/K", "ours - CREST (Hessian implementation)"),
    "N_CONFORMERS_EXCLUDED": ("Integer", None, "conformers excluded under the seam's imaginary-mode policy"),
    "N_KEPT_NEGATIVE": ("Integer", None, "modes below ithr kept negative (crest_native), over the included conformers"),
    "N_INVERTED": ("Integer", None, "modes in (ithr, 0) inverted, over the included conformers"),
    "ALGEBRAIC_TERMS_MATCH": ("Boolean", None, "|S_CONF, H_CONF, CP_CONF deltas| < 1e-4"),
    "G_PRIME_CREST": ("ArrayOfIntegers", None, "CREST enantiofac per conformer (from symmetries + mirror test)"),
    "G_PRIME_OURS": ("ArrayOfIntegers", None, "our g' per conformer (geometric chirality class)"),
    "CRE_DEGEN2": ("ArrayOfIntegers", None, "CREST cre_degen2 per conformer"),
    "CRE_DEGEN2_OURS": ("ArrayOfIntegers", None, "g_rot * cores per conformer, our port"),
}


# ====================================================================== CREST run
def run_crest_entropy(molecule, reference_xyz, run, threads=4, gfn=2, binary=None):
    """One `crest <ref> --entropy --gfn2 --keepdir` in `msrrho/crest_entropy/runNN/`. Skipped
    when that folder already holds a normally terminated run."""
    d = layout.crest_entropy_dir(molecule, run)
    out = d / "crest_entropy.out"
    if out.is_file() and "CREST terminated normally" in out.read_text(errors="replace"):
        return d
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(reference_xyz), str(d / "input.xyz"))
    cmd = [binary or crest_mod.crest_binary(), "input.xyz", "--entropy",
           "--gfn{}".format(int(gfn)), "--T", str(int(threads)), "--keepdir"]
    with open(out, "w") as fh:
        rc = subprocess.call(cmd, cwd=str(d), stdout=fh, stderr=subprocess.DEVNULL)
    if rc != 0 or "CREST terminated normally" not in out.read_text(errors="replace"):
        raise RuntimeError("crest --entropy failed in {} (rc {})".format(d, rc))
    return d


_NUM = r"([-+]?\d+\.\d+(?:E[-+]\d+)?)"


def parse_crest_entropy_out(path):
    """CREST's printed numbers at 298.15 K, and the setup lines."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    text = "\n".join(l for l in text.splitlines() if "OpenBLAS" not in l)
    rec = {}

    def grab(pattern, key, conv=float):
        m = re.search(pattern, text)
        if m:
            rec[key] = conv(m.group(1))

    grab(r"Sconf\s+=\s+" + _NUM, "S_conf")
    grab(r"\+ δSrrho\s+=\s+" + _NUM, "dS_rrho")
    grab(r"= S\(total\)\s+=\s+" + _NUM, "S_conf_total")
    grab(r"H\(T\)-H\(0\)\s+=\s+" + _NUM, "H_conf_G")        # final block: on free energies
    grab(r"Cp\(total\)\s+=\s+" + _NUM, "Cp_conf_G")
    m = re.search(r"Containing \d+ conformers :\s*\n\s*H\(298.15K\)-H\(0\) \(conf\)\s+" + _NUM
                  + r"\s*\n\s*Cp\(conf\)\s+" + _NUM + r"\s*\n\s*S\(conf\)\s+" + _NUM, text)
    if m:                                                   # ensemble block: on electronic energies
        rec["H_conf_E"] = float(m.group(1))
        rec["Cp_conf_E"] = float(m.group(2))
        rec["S_conf_E"] = float(m.group(3))
    grab(r"= G\(total\)\s+=\s+" + _NUM, "G_conf_total")
    grab(r"rotor cutoff\s+:\s+" + _NUM, "sthr")
    grab(r"imag\. cutoff\s+:\s+" + _NUM, "ithr")
    grab(r"scaling factor\s+:\s+" + _NUM, "fscal")
    grab(r"Nconf on file\s+:\s+(\d+)", "n_conformers", int)
    grab(r"\* wall-time:.*?(\d+\.\d+) sec", "wall_seconds")
    m = re.search(r"msRRHO\(bhess\) reference entropies:\s*\n((?:\s*\d+\.\d+\s+\d+\.\d+\s*\n)+)", text)
    if m:
        for line in m.group(1).splitlines():
            parts = line.split()
            if len(parts) == 2 and abs(float(parts[0]) - 298.15) < 1e-6:
                rec["S_ref"] = float(parts[1])
    m = re.search(r"\n\s*298\.15\s+" + _NUM + r"\s+" + _NUM + r"\s+" + _NUM + r"\s+(\d+)\s*\n", text)
    if m:
        rec["S_avg"] = float(m.group(1))
    rec["terminated_normally"] = "CREST terminated normally" in text
    return rec


def read_symmetries(run_dir):
    p = Path(run_dir) / "symmetries"
    if not p.is_file():
        return []
    return [l.split()[1] for l in p.read_text().splitlines() if l.strip()]


def read_cre_degen2(run_dir):
    p = Path(run_dir) / "cre_degen2"
    if not p.is_file():
        raise FileNotFoundError("cre_degen2 not found in {}".format(run_dir))
    lines = [l.split() for l in p.read_text().splitlines() if l.strip()]
    return [int(l[1]) for l in lines[1:1 + int(lines[0][0])]]


def read_conformers(run_dir):
    """(energy Eh, symbols, positions A) per conformer of `crest_conformers.xyz`."""
    out = []
    for comment, atoms in crest_mod.read_ensemble(Path(run_dir) / "crest_conformers.xyz"):
        out.append((float(comment.split()[0]), [a[0] for a in atoms],
                    np.array([[a[1], a[2], a[3]] for a in atoms])))
    return out


# ====================================================================== CREST's own algebra
def gibbs_shannon(e_rel_kcal, g, temperature_K):
    """CREST `entropy_S` (NIST eq. 30-32): S, Cp, H(T)-H(0) from relative energies and
    degeneracies. cal/mol/K, cal/mol/K, kcal/mol."""
    kt = thermo.KB_KCAL * temperature_K
    R = me.R_CAL
    e = np.asarray(e_rel_kcal, dtype=float)
    g = np.asarray(g, dtype=float)
    w = g * np.exp(-e / kt)
    z = float(w.sum())
    m1 = float((w * e).sum()) / z / kt
    m2 = float((w * e * e).sum()) / z / kt ** 2
    return R * (math.log(z) + m1), R * (m2 - m1 ** 2), R * temperature_K * m1 / 1000.0


def crest_enantiofac(run_dir, confs):
    """CREST's g' per conformer: 2 for every conformer whose `symmetries` label class
    holds a conformer with a mirror match, from our port of `intraconfRMSD` on the run's
    own rotamer file; 1 otherwise."""
    labels = read_symmetries(run_dir)
    if len(labels) != len(confs):
        labels = [c["symmetry_class"] for c in confs]
    flagged = {lab for lab, c in zip(labels, confs) if c["mirror_flag"]}
    return [2 if lab in flagged else 1 for lab in labels]


# ====================================================================== the Calculation
#: The seam reproduces CREST, so it takes CREST's own imaginary-mode regime: modes in
#: (ithr, 0) inverted, modes below ithr kept with zero entropy (measured on
#: propanal's third conformer, -68.4 cm^-1 in CREST's numerical Hessian, kept by CREST).
#: Since 2026-09-25 this seam is the ONLY caller of `crest_native`; the
#: record keeps its own counters (N_KEPT_NEGATIVE / N_INVERTED per basin and in [Seam]),
#: and its old [Imaginary_Spread] block is gone with the other records'.
SEAM_POLICY = "crest_native"


def evaluate_run(molecule, run, temperature_K=298.15, preset="crest", nprocs=4, gfn=2,
                 imaginary_policy=SEAM_POLICY):
    """Everything for one CREST entropy run: CREST's numbers, xtb Hessians at CREST's
    conformers and at the reference structure, our assembly, and the seam deltas."""
    d = layout.crest_entropy_dir(molecule, run)
    crest_rec = parse_crest_entropy_out(d / "crest_entropy.out")
    confs_deg = degeneracy.conformer_degeneracies(d)
    e_g = crest_enantiofac(d, confs_deg)
    degen2 = read_cre_degen2(d)
    conformers = read_conformers(d)
    if not (len(conformers) == len(confs_deg) == len(degen2)):
        raise ValueError("run {}: {} conformers, {} rotamer groups, {} cre_degen2 rows".format(
            run, len(conformers), len(confs_deg), len(degen2)))
    e0 = min(e for e, _s, _p in conformers)
    e_rel = [(e - e0) * HARTREE_TO_KCAL for e, _s, _p in conformers]
    s_conf_ours, _cp_e, _h_e = gibbs_shannon(e_rel, e_g, temperature_K)
    _s_d2, cp_ours, h_ours = gibbs_shannon(e_rel, degen2, temperature_K)

    # per-conformer msRRHO from xtb --hess at CREST's geometries (engine folder)
    from ase.data import atomic_masses, atomic_numbers
    basins = []
    for k, (e_eh, sym, pos) in enumerate(conformers):
        wd = layout.xtb_entropy_dir(molecule, run, k)
        h = _xtb_hessian(wd, sym, pos, gfn, nprocs)
        numbers = [atomic_numbers[s] for s in sym]
        masses = [float(atomic_masses[z]) for z in numbers]
        ops = symmetry.symmetry_operations(numbers, pos)
        rec = me.basin_thermochemistry(k, h, masses, pos, e_eh * HARTREE_TO_EV,
                                       ops["sigma"], 1, temperature_K, preset, imaginary_policy)
        rec["g_prime"] = int(confs_deg[k]["g_prime"])
        rec["g_prime_source"] = confs_deg[k]["g_prime_source"]
        rec["g_prime_crest"] = int(e_g[k])
        rec["cre_degen2"] = int(degen2[k])
        basins.append(rec)
    ens = me.assemble(basins, temperature_K)

    # CREST's reference: the input structure, plain Hessian at GFN2
    from ase.io import read
    ref = read(str(d / "crest_input_copy.xyz"))
    wd = layout.xtb_entropy_dir(molecule, run, 99)
    h = _xtb_hessian(wd, ref.get_chemical_symbols(), ref.get_positions(), gfn, nprocs)
    numbers = [atomic_numbers[s] for s in ref.get_chemical_symbols()]
    ops = symmetry.symmetry_operations(numbers, ref.get_positions())
    ref_rec = me.basin_thermochemistry(99, h, [float(atomic_masses[z]) for z in numbers],
                                       ref.get_positions(), 0.0, ops["sigma"], 1,
                                       temperature_K, preset, imaginary_policy)
    # (the seam writes PRESET_SPREAD = 0.0: CREST prints one preset; no spread is claimed)
    s_ref_ours = (ref_rec["S_msrrho_cal_per_K"] - CREST_TRANS_OFFSET_CAL
                  if not ref_rec["excluded"] else None)
    # dS_bar the CREST way: populations from G_i with cre_degen2 weights, excluded
    # conformers take the average of the others (CREST: "for failed hess calcs")
    inc = [b for b in basins if not b["excluded"]]
    if inc:
        kt = thermo.KB_KCAL * temperature_K
        G = np.array([b["G_i_kcal"] for b in inc])
        w = np.array([b["cre_degen2"] for b in inc]) * np.exp(-(G - G.min()) / kt)
        p = w / w.sum()
        s_avg_ours = float((p * np.array([b["S_msrrho_cal_per_K"] for b in inc])).sum()) - CREST_TRANS_OFFSET_CAL
    else:
        s_avg_ours = None
    ds_bar_ours = (s_avg_ours - s_ref_ours) if (s_avg_ours is not None and s_ref_ours is not None) else None

    if inc:
        g_rel = [b["G_i_kcal"] - min(x["G_i_kcal"] for x in inc) for b in inc]
        _s_g, cp_g_ours, h_g_ours = gibbs_shannon(g_rel, [b["cre_degen2"] for b in inc], temperature_K)
    else:
        cp_g_ours = h_g_ours = float("nan")
    seam = {"RUN": int(run),
            "S_CONF_OURS": s_conf_ours, "S_CONF_DELTA": s_conf_ours - crest_rec.get("S_conf", float("nan")),
            "H_CONF_OURS": h_ours, "H_CONF_DELTA": h_ours - crest_rec.get("H_conf_E", float("nan")),
            "CP_CONF_OURS": cp_ours, "CP_CONF_DELTA": cp_ours - crest_rec.get("Cp_conf_E", float("nan")),
            "H_CONF_G_OURS": h_g_ours, "H_CONF_G_DELTA": h_g_ours - crest_rec.get("H_conf_G", float("nan")),
            "CP_CONF_G_OURS": cp_g_ours, "CP_CONF_G_DELTA": cp_g_ours - crest_rec.get("Cp_conf_G", float("nan")),
            "N_CONFORMERS_EXCLUDED": len(basins) - len(inc),
            "N_KEPT_NEGATIVE": int(sum(b.get("n_kept_negative", 0) for b in inc)),
            "N_INVERTED": int(sum(b.get("n_inverted", 0) for b in inc)),
            "G_PRIME_CREST": list(e_g), "G_PRIME_OURS": [int(b["g_prime"]) for b in basins],
            "CRE_DEGEN2": list(degen2),
            "CRE_DEGEN2_OURS": [int(c["cre_degen2_equivalent"]) for c in confs_deg]}
    if s_ref_ours is not None and "S_ref" in crest_rec:
        seam.update({"S_REF_OURS": s_ref_ours, "S_REF_DELTA": s_ref_ours - crest_rec["S_ref"]})
    if ds_bar_ours is not None and "dS_rrho" in crest_rec:
        seam.update({"DS_BAR_OURS": ds_bar_ours, "DS_BAR_DELTA": ds_bar_ours - crest_rec["dS_rrho"]})
    seam["ALGEBRAIC_TERMS_MATCH"] = bool(
        abs(seam["S_CONF_DELTA"]) < TOL_ALGEBRAIC_CAL and abs(seam["H_CONF_DELTA"]) < TOL_ALGEBRAIC_CAL
        and abs(seam["CP_CONF_DELTA"]) < TOL_ALGEBRAIC_CAL)
    return dict(run=int(run), crest=crest_rec, basins=basins, ensemble=ens, seam=seam,
                conformer_degeneracies=confs_deg, e_rel_kcal=e_rel,
                imaginary_policy=imaginary_policy)


def _xtb_hessian(workdir, symbols, positions, gfn, nprocs):
    """xtb --hess in `workdir`, reusing the engine files when they are already there."""
    workdir = Path(workdir)
    hess = workdir / "hessian"
    if hess.is_file() and (workdir / "vibspectrum").is_file():
        from ..qm_interfaces import orca
        return orca.hessian_to_ev_per_angstrom2(xtb.parse_turbomole_hessian(hess, len(symbols)))
    rec = xtb.hessian_at(symbols, positions, gfn=gfn, workdir=workdir, keep=True, nprocs=nprocs)
    return rec["hessian_eV_A2"]


def run_calculation(molecule, reference_xyz=None, runs=(1, 2), run_crest=True, threads=4,
                    nprocs=4, temperature_K=298.15, preset="crest", gfn=2, level=LEVEL,
                    imaginary_policy=SEAM_POLICY):
    """CREST `--entropy` (each run in `runs` unless its folder is complete), xtb Hessians,
    our assembly, the seam; records `msrrho/thermo/gfn2.*`. Returns the full record."""
    molecule = Path(molecule)
    if run_crest:
        if reference_xyz is None:
            raise ValueError("run_crest needs the reference structure to start CREST from")
        for r in runs:
            run_crest_entropy(molecule, reference_xyz, r, threads=threads, gfn=gfn)
    evals = [evaluate_run(molecule, r, temperature_K, preset, nprocs, gfn, imaginary_policy)
             for r in runs]
    if len(evals) < 2:
        raise ValueError("at least two CREST entropy runs are required (the sampling is "
                         "stochastic); {} given".format(len(evals)))
    first = evals[0]
    layout.thermo_dir(molecule).mkdir(parents=True, exist_ok=True)

    s_confs = [e["crest"].get("S_conf", float("nan")) for e in evals]
    s_tots = [e["crest"].get("S_conf_total", float("nan")) for e in evals]
    crest_rows = []
    for e in evals:
        c = e["crest"]
        crest_rows.append({"RUN": e["run"], "N_CONFORMERS": c.get("n_conformers"),
                           "S_CONF": c.get("S_conf"), "DS_RRHO": c.get("dS_rrho"),
                           "S_CONF_TOTAL": c.get("S_conf_total"), "H_CONF_E": c.get("H_conf_E"),
                           "CP_CONF_E": c.get("Cp_conf_E"), "H_CONF_G": c.get("H_conf_G"),
                           "CP_CONF_G": c.get("Cp_conf_G"), "S_REF": c.get("S_ref"),
                           "S_AVG": c.get("S_avg"), "STHR": c.get("sthr"), "ITHR": c.get("ithr"),
                           "FSCAL": c.get("fscal"), "WALL": c.get("wall_seconds")})
    ens, basins = first["ensemble"], first["basins"]
    info = {"MOLECULE_DIR": str(molecule), "QM9_INDEX": None, "TAG": None, "LEVEL": level,
            "ENGINE": "crest+xtb GFN{}-xTB".format(gfn), "PRESET": preset,
            "TAU": float(thermo.MSRRHO_PRESETS[preset]["tau_cm"]),
            "ROTOR_CAP_RULE": str(thermo.MSRRHO_PRESETS[preset]["rotor_cap"]),
            "ITHR_POLICY": imaginary_policy, "FSCAL": 1.0, "TEMPERATURE": float(temperature_K),
            "PRESSURE": float(thermo.P_STD), "REFERENCE_BASIN": ens["reference_basin"],
            "PTOT": float(ens["ptot"]), "EXTRAPOLATION": "none"}
    rows = []
    for b in basins:
        row = {"INDEX": b["index"], "SIGMA": b["sigma"], "G0": b["g0"], "G_PRIME": b["g_prime"],
               "G_PRIME_SOURCE": b["g_prime_source"], "E_EL": b["E_el_kcal"],
               "LOWEST_FREQ": b["lowest_frequency_cm"], "EXCLUDED": bool(b["excluded"]),
               "N_IMAGINARY": b.get("n_imaginary"), "N_INVERTED": b.get("n_inverted", 0),
               "N_KEPT_NEGATIVE": b.get("n_kept_negative", 0),
               "N_BELOW_FLOOR": b.get("n_below_floor", 0),
               "INVERTED_CM": b.get("inverted_frequencies_cm") or None,
               "DROPPED_CM": b.get("dropped_frequencies_cm") or None,
               "POPULATION": b.get("population", 0.0)}
        if not b["excluded"]:
            row.update({"ZPE": b["ZPE_kcal"], "H_THERMAL": b["H_thermal_kcal"],
                        "G_ROT": b["G_rot_kcal"], "G_TRANS": b["G_trans_kcal"],
                        "S_VIB": b["S_vib_cal_per_K"], "S_VIB_HO": b["S_vib_HO_cal_per_K"],
                        "S_ROT": b["S_rot_cal_per_K"], "S_TRANS": b["S_trans_cal_per_K"],
                        "S_MSRRHO": b["S_msrrho_cal_per_K"], "H_I": b["H_i_kcal"], "G_I": b["G_i_kcal"]})
        rows.append(row)
    blocks = {"Calculation_Info": info, "Basin": rows,
              "Ensemble": {"S_CONF_PRIME": ens["S_conf_prime_cal_per_K"],
                           "S_CONF_PRIME_E": ens["S_conf_prime_E_cal_per_K"],
                           "DS_BAR": ens["dS_bar_cal_per_K"], "H_CONF": ens["H_conf_kcal"],
                           "CP_CONF": ens["Cp_conf_cal_per_K"]},
              "Result": {"S_ABS": ens["S_abs_cal_per_K"], "H_ABS": ens["H_abs_kcal"],
                         "G_TOTAL": ens["G_total_kcal"], "N_BASINS": ens["n_basins"],
                         "N_INCLUDED": ens["n_included"], "N_EXCLUDED": ens["n_excluded"],
                         "N_BASINS_90": ens["basins_90"], "PRESET_SPREAD": 0.0},
              "Crest": [dict(r, N_RUNS=len(evals),
                             S_CONF_SPREAD=float(np.nanmax(s_confs) - np.nanmin(s_confs)),
                             S_TOTAL_SPREAD=float(np.nanmax(s_tots) - np.nanmin(s_tots)))
                        for r in crest_rows],
              "Seam": [e["seam"] for e in evals]}
    missing = prop.write(layout.level_file(molecule, level, "thermo_msrrho.toml"), blocks, SCHEMA, prop.NORMAL_TERMINATION, PROGNAME)
    if missing:
        raise RuntimeError("thermo_msrrho.toml (gfn2) keys outside the schema: {}".format(missing))
    _write_report(layout.level_file(molecule, level, "thermo_msrrho.out"), info, evals)
    return dict(level=level, runs=evals, record=layout.level_file(molecule, level, "thermo_msrrho.toml"),
                S_conf_spread=float(np.nanmax(s_confs) - np.nanmin(s_confs)))


def _write_report(path, info, evals):
    rep = report.Report("openQHA thermo_msrrho at gfn2", "CREST --entropy on its native engine "
                        "beside our assembly (the GFN2 seam)")
    rep.section("conventions")
    for k in ("ENGINE", "PRESET", "TAU", "ITHR_POLICY", "FSCAL", "TEMPERATURE"):
        rep.kv(k, info[k])
    rep.section("CREST's own numbers per run (298.15 K)")
    rep.table(["run", "conf", "Sconf", "dSrrho", "S(total)", "H(conf,E)", "Cp(conf,E)", "S_ref", "|S| avg", "wall s"],
              [[e["run"], e["crest"].get("n_conformers"), "%.6f" % e["crest"].get("S_conf", float("nan")),
                "%.6f" % e["crest"].get("dS_rrho", float("nan")), "%.6f" % e["crest"].get("S_conf_total", float("nan")),
                "%.6f" % e["crest"].get("H_conf_E", float("nan")), "%.6f" % e["crest"].get("Cp_conf_E", float("nan")),
                "%.4f" % e["crest"].get("S_ref", float("nan")), "%.4f" % e["crest"].get("S_avg", float("nan")),
                "%.1f" % e["crest"].get("wall_seconds", float("nan"))] for e in evals])
    rep.section("the seam: CREST's inputs through our assembly, term by term")
    for e in evals:
        s = e["seam"]
        rep.table(["term", "ours", "CREST", "delta", "kind"],
                  [["S'_conf (eq. 10, enantiofac)", "%.6f" % s["S_CONF_OURS"], "%.6f" % e["crest"].get("S_conf", float("nan")), "%+.2e" % s["S_CONF_DELTA"], "algebraic"],
                   ["H_conf (eq. 14 on E, cre_degen2)", "%.6f" % s["H_CONF_OURS"], "%.6f" % e["crest"].get("H_conf_E", float("nan")), "%+.2e" % s["H_CONF_DELTA"], "algebraic"],
                   ["Cp_conf (eq. 13 on E, cre_degen2)", "%.6f" % s["CP_CONF_OURS"], "%.6f" % e["crest"].get("Cp_conf_E", float("nan")), "%+.2e" % s["CP_CONF_DELTA"], "algebraic"],
                   ["H_conf final (on G_i)", "%.6f" % s["H_CONF_G_OURS"], "%.6f" % e["crest"].get("H_conf_G", float("nan")), "%+.4f" % s["H_CONF_G_DELTA"], "Hessian"],
                   ["Cp final (on G_i)", "%.6f" % s["CP_CONF_G_OURS"], "%.6f" % e["crest"].get("Cp_conf_G", float("nan")), "%+.4f" % s["CP_CONF_G_DELTA"], "Hessian"],
                   ["S_ref (reference structure)", "%.4f" % s.get("S_REF_OURS", float("nan")), "%.4f" % e["crest"].get("S_ref", float("nan")), "%+.4f" % s.get("S_REF_DELTA", float("nan")), "Hessian"],
                   ["dS_bar", "%.4f" % s.get("DS_BAR_OURS", float("nan")), "%.4f" % e["crest"].get("dS_rrho", float("nan")), "%+.4f" % s.get("DS_BAR_DELTA", float("nan")), "Hessian"]],
                  title="run {}: algebraic terms {}; g' CREST {} ours {}; cre_degen2 {} ours {}; excluded {}".format(
                      e["run"], "MATCH" if s["ALGEBRAIC_TERMS_MATCH"] else "DIFFER",
                      s["G_PRIME_CREST"], s["G_PRIME_OURS"], s["CRE_DEGEN2"], s["CRE_DEGEN2_OURS"],
                      s["N_CONFORMERS_EXCLUDED"]))
    rep.note("Algebraic terms are the same formula on the same numbers and must agree to 1e-4. "
             "S_ref and dS_bar need Hessians CREST does not leave on disk; ours come from xtb "
             "--hess at the same geometries, so their delta is the Hessian implementation "
             "(numerical in CREST, analytic in xtb), not the assembly. CREST's translational "
             "term is at 1 atm; ours is shifted by -0.044 cal/mol/K before comparing.")
    rep.section("our assembly on CREST's conformers (run {})".format(evals[0]["run"]))
    ens = evals[0]["ensemble"]
    for k in ("S_ref_cal_per_K", "S_conf_prime_cal_per_K", "S_conf_prime_E_cal_per_K", "dS_bar_cal_per_K",
              "S_abs_cal_per_K", "G_total_kcal"):
        rep.kv(k, "%.4f" % ens[k])
    rep.write(path, step=STEP)
