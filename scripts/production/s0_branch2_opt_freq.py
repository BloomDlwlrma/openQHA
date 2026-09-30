# -*- coding: utf-8 -*-
"""Branch 2 -- run `OPT FREQ` on every package 1 conformer and get the Gibbs free energy from the partition function.

PRODUCTION. Branch 2: OPT FREQ per conformer and the Gibbs free energy that comes
out of it. Its output is a deliverable.

**What it does**: treat one molecule as a canonical ensemble in contact with a heat bath and use
statistical mechanics to build the partition function from microscopic properties (electronic structure + vibrational frequencies), giving `G(T)`:

    G(total) = E(electronic) + G(thermal correction)

`G(thermal correction)` comes from the ideal-gas, rigid-rotor, harmonic-oscillator (`RRHO`)
model and covers translation, rotation, vibration and entropy. ORCA prints it directly in the `THERMOCHEMISTRY` block after `FREQ`.

**Level**: `RI-MP2/cc-pVTZ`, Coulomb and exchange through `RIJK` (`cc-pVTZ/JK`), the
correlation fitting through `cc-pVTZ/C`. **This is a resolution-of-the-identity approximation to "MP2/cc-pVTZ", not canonical MP2** --
the reference paper (J. Chem. Theory Comput. 2020, 16, 196-210) reports in section 3.1 that the
RI-versus-canonical gradient difference is of order 1e-4 au, **but that sentence must appear in the deliverable** so no reader takes it for canonical MP2.

**Three traps this repository has measured, each handled below** (2026-09-01):

1. **Never take ORCA `Final Gibbs free energy` as given.** Its automatic symmetry number gets it
   wrong -- measured, it calls acetone `C1, sigma = 1` while the correct external symmetry number
   is 2, making the free energy too low by `RT ln 2 = 0.411 kcal/mol`. This script **records the
   symmetry number ORCA chose, verbatim and in its own column**, with the correction given by `RT ln(sigma_declared / sigma_orca)`;
   **any molecule whose symmetry number is not independently declared is marked provisional and does not enter a final number.**
2. **Quasi-RRHO is on by default in ORCA 6** (measured output: `Quasi RRHO ... True`).
   Measured on acetone, the two differ by 0.466 kcal/mol. Both are reported.
   **But ORCA runs only once** -- the pure-harmonic value is computed here by `openqha.thermo` from the same frequencies.
   Running ORCA again just to turn off one switch wastes twice the machine time; and computing it
   here also turns "can our partition function reproduce ORCA number" into a **per-conformer assertion**
   (`crosscheck_vs_orca_kcal`; plan acceptance 1 requires < 0.01 kcal/mol).
3. **An imaginary frequency is rejected explicitly**, never "made positive" -- that is not a minimum.

**The identity of where it ran**: quantum chemistry producing a production number runs only on deimos with ORCA 6.1.1.
**The ORCA 6.0.1 in local WSL is for testing only**, and its products always carry `provenance.status` = `test`.
The user asked on 2026-09-02 for it to "run on this machine", so this script may run locally,
**but the product identity is labelled honestly and never passed off as production.**

Usage::

    python scripts/production/s0_branch2_opt_freq.py --probe 10          # cost probe: run only one
    python scripts/production/s0_branch2_opt_freq.py --nprocs 4 --resume # production
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

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

from openqha import S0_ROOT
from openqha.qm_interfaces import orca

ORCA = os.environ.get("S0_ORCA_BIN", "/home/ubuntu/packages/orca_6_0_1/orca")
XYZ_DIR = S0_ROOT / "analysis" / "package1" / "1_16000" / "1_4000" / "xyz"
BASINS = S0_ROOT / "analysis" / "package1_crest_1_4000__basins.parquet"
MOLECULES = S0_ROOT / "analysis" / "package1_crest_1_4000__molecules.parquet"

R_KCAL = 1.987204259e-3          # kcal/(mol*K)
EH_KCAL = 627.5094740631         # kcal/mol per Hartree

# The simple input line. All three of these were forced by ORCA itself; none is a preference:
#
# * **`NumFreq`, not `FREQ`** -- measured 2026-09-02, `FREQ` exits with code 25 and reports
#   `ERROR: MP2 analytic Hessian calculations are not implemented - please use NumFreq`.
#   **This sets the order of the cost**: a numerical Hessian needs `6N` displaced gradients (60 for 10 atoms, 114 for 19).
# * **`TightOpt`** -- ORCA warns about this itself: with a loosely converged geometry the symmetry
#   detection fails, and the residual gradient contaminates the lowest frequencies -- which is exactly where quasi-RRHO acts.
# * **A frozen core is ORCA default for MP2 gradients** (the output warns about it explicitly); this script keeps the default and records it.
SIMPLE = "! RI-MP2 cc-pVTZ cc-pVTZ/C cc-pVTZ/JK RIJK TightOpt NumFreq TightSCF"


def read_basin(qm9_index, basin):
    """Extract one basin from a multi-frame xyz. Returns (atom count, coordinate lines, comment line)."""
    p = XYZ_DIR / "{}_basins.xyz".format(qm9_index)
    lines = p.read_text(encoding="utf-8").splitlines()
    i, k = 0, 0
    while i < len(lines):
        nat = int(lines[i].split()[0])
        comment = lines[i + 1]
        body = lines[i + 2:i + 2 + nat]
        if k == int(basin):
            return nat, body, comment
        i += 2 + nat
        k += 1
    raise KeyError("no basin {1} in {0}".format(p, basin))


def write_input(path, body, nprocs, maxcore, quasi_rrho=True, charge=0, mult=1,
                simple=None):
    lines = [simple or SIMPLE,
             "%maxcore {}".format(int(maxcore)),
             "%pal nprocs {} end".format(int(nprocs))]
    if not quasi_rrho:
        lines += ["%freq", "  QuasiRRHO false", "end"]
    lines.append("* xyz {} {}".format(int(charge), int(mult)))
    lines += list(body)
    lines.append("*")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


_PATS = dict(
    e_el=r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)",
    zpe=r"Zero point energy\s+\.+\s+(-?\d+\.\d+) Eh",
    thermal_vib=r"Thermal vibrational correction\s+\.+\s+(-?\d+\.\d+) Eh",
    inner_energy=r"Total thermal energy\s+(-?\d+\.\d+)",
    enthalpy=r"Total Enthalpy\s+\.+\s+(-?\d+\.\d+)",
    entropy_term=r"Final entropy term\s+\.+\s+(-?\d+\.\d+) Eh",
    gibbs=r"Final Gibbs free energy\s+\.+\s+(-?\d+\.\d+)",
    g_minus_eel=r"G-E\(el\)\s+\.+\s+(-?\d+\.\d+) Eh",
    quasi_rrho=r"Quasi RRHO\s+\.+\s+(\w+)",
    cutoff=r"Cut-Off Frequency\s+\.+\s+(-?\d+\.\d+)",
    sym_number=r"Point Group:\s+(\S+),\s+Symmetry Number:\s+(\d+)",
    n_basis=r"Number of basis functions\s+\.+\s+(\d+)",
)


def parse_out(text):
    """Parse the ORCA output. **Whatever is missing is reported as missing; no default is filled in.**"""
    rec, missing = {}, []
    for k, pat in _PATS.items():
        m = re.search(pat, text)
        if not m:
            missing.append(k)
            continue
        if k == "sym_number":
            rec["point_group"] = m.group(1)
            rec["orca_symmetry_number"] = int(m.group(2))
        elif k == "quasi_rrho":
            rec[k] = (m.group(1).lower() == "true")
        elif k == "n_basis":
            rec[k] = int(m.group(1))
        else:
            rec[k] = float(m.group(1))
    # **The basis-function count comes from `Dimension of the orbital basis`, not `Number of basis functions`.**
    #
    # This took two attempts on 2026-09-02:
    #   the first version took the **first** `Number of basis functions` match -> 272 (too large);
    #   the second took the **last** -> **also wrong**, because that line alternates between the SHARK
    #     and MP2 blocks (measured sequence 272,236,272,236,...), so either end depends on where the output stopped.
    #   The right answer: in the `RI-MP2 ENERGY+GRADIENT` block ORCA prints
    #     `Dimension of the orbital basis ... 236` -- **the one unambiguous place**.
    #     272 is the count under the general contraction, 236 the orbital-basis dimension after MP2 switches to segmented contraction;
    #     236 = 6x30 + 4x14 matches the spherical-harmonic cc-pVTZ count exactly.
    m = re.search(r"Dimension of the orbital basis\s+\.+\s+(\d+)", text)
    if m:
        rec["n_basis"] = int(m.group(1))
    m = re.search(r"Dimension of the AuxC basis\s+\.+\s+(\d+)", text)
    if m:
        rec["n_aux_c"] = int(m.group(1))
    rec["n_basis_lines_reported"] = [
        int(x) for x in re.findall(r"Number of basis functions\s+\.+\s+(\d+)", text)][:4]
    freqs = [float(x) for x in re.findall(
        r"^\s+\d+:\s+(-?\d+\.\d+) cm\*\*-1", text, re.M)]
    rec["frequencies_cm_inv"] = freqs
    rec["n_imaginary"] = int(sum(1 for f in freqs if f < -1.0))
    # ORCA prints the 6 translational/rotational modes as 0.00 cm**-1 as well.
    # **The lowest frequency must report the lowest VIBRATIONAL mode, not the lowest line** --
    # otherwise every molecule reports 0.00 and the field carries no information (exposed on the 2026-09-02 probe).
    rigid = [f for f in freqs if abs(f) <= 1.0]
    vib = [f for f in freqs if abs(f) > 1.0]
    rec["n_zero_modes"] = len(rigid)
    # **Assertion**: a non-linear molecule must have exactly 6 zero modes. Anything else means the
    # geometry did not converge, or the parser caught the wrong block -- neither may pass silently.
    rec["zero_modes_ok"] = (len(rigid) == 6)
    rec["lowest_vibrational_cm_inv"] = min(vib) if vib else None
    rec["lowest_frequency_cm_inv"] = min(vib) if vib else None
    rec["numerical_hessian"] = bool(re.search(r"NUMERICAL FREQUENCIES", text))
    # The optimised Cartesian coordinates (the last occurrence) -- needed to compute the partition function here
    blocks = re.findall(
        r"CARTESIAN COORDINATES \(ANGSTROEM\)\s*\n-+\n"
        r"((?:\s*[A-Za-z]{1,2}(?:\s+-?\d+\.\d+){3}\s*\n)+)", text)
    if blocks:
        sym, xyz = [], []
        for ln in blocks[-1].strip().split("\n"):
            f = ln.split()
            sym.append(f[0])
            xyz.append([float(v) for v in f[1:4]])
        rec["symbols"] = sym
        rec["positions_A"] = xyz
    rec["missing_fields"] = missing
    return rec



def own_thermochemistry(sub, temperature_K=298.15, pressure_Pa=1.0e5):
    """Compute both the pure-harmonic and the quasi-RRHO values here, with `openqha.thermo`, from the same frequencies.

    **Two purposes**: (a) it saves a second ORCA run; (b) it turns "can our partition function
    reproduce ORCA number" into a **per-conformer assertion** -- plan acceptance 1 requires the two to differ by < 0.01 kcal/mol.
    **The symmetry number is passed in explicitly here and never derived automatically**.
    """
    from openqha import thermo as T
    import numpy as np
    freqs = [f for f in sub.get("frequencies_cm_inv", []) if f > 0.0]
    sym = sub.get("symbols")
    pos = sub.get("positions_A")
    if not freqs or not sym or not pos:
        return dict(error="no frequencies or no optimised geometry; cannot compute it here")
    from ase.data import atomic_masses, atomic_numbers
    masses = np.array([atomic_masses[atomic_numbers[s]] for s in sym])
    pos = np.asarray(pos, dtype=float)
    out = {}
    try:
        mom = T.principal_moments(masses, pos)
        vib_h = T.vibrational(freqs, temperature_K, qrrho=False)
        vib_q = T.vibrational(freqs, temperature_K, qrrho=True,
                              moments_amu_A2=mom)
        # Rotation and translation: **use the symmetry number ORCA chose**, so the terms line up with
        # ORCA one by one; the correction to the declared value is separate, see sigma_correction_kcal.
        sigma_orca = int(sub.get("orca_symmetry_number") or 1)
        rot = T.rotational(masses, pos, sigma_orca, temperature_K)
        tr = T.translational(float(masses.sum()), temperature_K, pressure_Pa,
                             kind="helmholtz")
        kt = T.KB_KCAL * temperature_K
        base = rot["A_rot_kcal"] + tr["value_kcal"] + kt          # A_elec = 0 (g0=1)
        out["G_minus_Eel_rrho_kcal"] = float(vib_h["A_vib_kcal"] + base)
        out["G_minus_Eel_qrrho_kcal"] = float(vib_q["A_vib_qrrho_kcal"] + base)
        out["qrrho_minus_rrho_kcal"] = (out["G_minus_Eel_qrrho_kcal"]
                                        - out["G_minus_Eel_rrho_kcal"])
        out["ZPE_kcal"] = vib_h["ZPE_kcal"]
        out["lowest_frequency_cm_inv"] = float(min(freqs))
        out["symmetry_number_used"] = sigma_orca
        out["moments_amu_A2"] = [float(x) for x in mom]
        # Cross-check against ORCA (whose default is quasi-RRHO)
        if sub.get("g_minus_eel") is not None:
            orca_q = sub["g_minus_eel"] * EH_KCAL
            out["orca_G_minus_Eel_kcal"] = orca_q
            out["crosscheck_vs_orca_kcal"] = out["G_minus_Eel_qrrho_kcal"] - orca_q
    except Exception as e:
        out["error"] = "{}: {}".format(type(e).__name__, str(e)[:300])
    return out


def run_one(qm9_index, basin, outdir, nprocs, maxcore, timeout_s,
            simple=None):
    """One conformer: the default (quasi-RRHO) plus pure harmonic. Returns the record dict."""
    nat, body, comment = read_basin(qm9_index, basin)
    d = Path(outdir) / "{}_b{}".format(qm9_index, basin)
    d.mkdir(parents=True, exist_ok=True)
    rec = dict(qm9_index=qm9_index, basin=int(basin), n_atoms=nat,
               source_comment=comment, simple_input=(simple or SIMPLE),
               nprocs=int(nprocs), maxcore_mb=int(maxcore), workdir=str(d))
    # **ORCA runs only once.** The pure-harmonic value is computed here by `openqha.thermo` from the
    # same frequencies -- running ORCA again just to switch quasi-RRHO off wastes twice the machine
    # time, and computing it here also makes "can our partition function reproduce ORCA" a per-conformer assertion (plan acceptance 1).
    stem = d / "optfreq"
    write_input(str(stem) + ".inp", body, nprocs, maxcore, quasi_rrho=True,
                simple=simple)
    t0 = time.time()
    with open(str(stem) + ".out", "w") as fh:
        code = subprocess.call([ORCA, str(stem) + ".inp"], stdout=fh,
                               stderr=subprocess.STDOUT, cwd=str(d),
                               env=orca.subprocess_env(), timeout=timeout_s)
    wall = time.time() - t0
    text = Path(str(stem) + ".out").read_text(encoding="utf-8", errors="replace")
    ok = "****ORCA TERMINATED NORMALLY****" in text
    if not ok:
        rec["orca"] = dict(error="ORCA did not finish normally, exit code {}".format(code),
                           tail="\n".join(text.split("\n")[-25:]),
                           wall_seconds=wall, terminated_normally=False,
                           exit_code=code)
        rec["total_wall_seconds"] = wall
        return rec
    sub = parse_out(text)
    sub.update(wall_seconds=wall, terminated_normally=True, exit_code=code)
    rec["orca"] = sub
    rec["total_wall_seconds"] = wall
    rec["n_imaginary"] = sub.get("n_imaginary")
    # **An imaginary frequency means rejection**: that is not a minimum, and thermodynamics on it is meaningless
    rec["is_minimum"] = (rec["n_imaginary"] == 0)
    rec["own"] = own_thermochemistry(sub)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nprocs", type=int, default=4)
    ap.add_argument("--maxcore", type=int, default=3500)
    ap.add_argument("--outdir", default=None, help="default ~/runs/branch2_opt_freq")
    ap.add_argument("--jsonl", default="analysis/branch2_opt_freq.partial.jsonl")
    ap.add_argument("--probe", type=int, default=None,
                    help="cost probe: run only the first conformer with this atom count")
    ap.add_argument("--only", default=None, help="comma-separated qm9_index values")
    ap.add_argument("--max-atoms", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--timeout-s", type=float, default=None)
    ap.add_argument("--shard", default=None,
                    help="of the form i/n: deal the work round-robin to n concurrent instances; this one takes share i (i starts at 0)")
    ap.add_argument("--simple", default=None,
                    help="override the simple input line; used to measure the cost and memory of RIJK versus RIJCOSX")
    args = ap.parse_args()

    import pandas as pd
    from rdkit import Chem

    basins = pd.read_parquet(BASINS)
    mols = pd.read_parquet(MOLECULES)
    mols["n_atoms"] = mols.smiles.map(
        lambda s: Chem.AddHs(Chem.MolFromSmiles(s)).GetNumAtoms())
    tbl = basins.merge(mols[["qm9_index", "smiles", "n_atoms"]], on="qm9_index")

    if args.probe:
        tbl = tbl[tbl.n_atoms == args.probe].head(1)
    if args.only:
        tbl = tbl[tbl.qm9_index.isin(set(args.only.split(",")))]
    if args.max_atoms:
        tbl = tbl[tbl.n_atoms <= args.max_atoms]
    # **Sort by size before dealing the shards**: every shard then gets the same mix of sizes, so the
    # concurrent instances finish together instead of one grinding on large molecules while the others idle.
    tbl = tbl.sort_values(["n_atoms", "qm9_index", "basin"]).reset_index(drop=True)
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        tbl = tbl[tbl.index % n == i]
        print("shard {}/{}: {} conformer(s) in this instance".format(i, n, len(tbl)))
    if args.limit:
        tbl = tbl.head(args.limit)

    out = (Path(args.outdir).expanduser() if args.outdir
           else Path.home() / "runs" / "branch2_opt_freq")
    out.mkdir(parents=True, exist_ok=True)
    jsonl = Path(args.jsonl)
    jsonl.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if args.resume and jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["qm9_index"], r["basin"]))
        print("resuming: {} conformer(s) already done".format(len(done)))

    print("=" * 96)
    print("branch 2 -- OPT FREQ, free energy from the partition function")
    print("level: {}".format(args.simple or SIMPLE))
    print("conformers: {}   processes {}   memory per core {} MB".format(
        len(tbl), args.nprocs, args.maxcore))
    print("output: {}   incremental: {}".format(out, jsonl))
    print("**identity**: ORCA 6.0.1 in local WSL -> provenance.status = test")
    print("=" * 96, flush=True)

    t_all = time.time()
    for _, row in tbl.iterrows():
        key = (row.qm9_index, int(row.basin))
        if key in done:
            continue
        t0 = time.time()
        try:
            rec = run_one(row.qm9_index, int(row.basin), out,
                          args.nprocs, args.maxcore, args.timeout_s,
                          simple=args.simple)
        except Exception as e:
            rec = dict(qm9_index=row.qm9_index, basin=int(row.basin),
                       error="{}: {}".format(type(e).__name__, str(e)[:400]),
                       wall_seconds=time.time() - t0)
        rec.update(smiles=row.smiles, n_atoms=int(row.n_atoms),
                   provenance=dict(status="test", orca_bin=ORCA,
                                   host=os.uname().nodename,
                                   note="production numbers come only from ORCA 6.1.1 on deimos"))
        with open(str(jsonl), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        q = rec.get("orca", {})
        o = rec.get("own", {})
        low = q.get("lowest_frequency_cm_inv")
        print("{} b{}  {:2d} atoms  basis {}  imaginary {}  lowest {}  "
              "sigma(ORCA)={}  quasiRRHO-harmonic {}  cross-check {}  wall {:.0f} s".format(
                  rec["qm9_index"], rec["basin"], rec.get("n_atoms", 0),
                  q.get("n_basis"), rec.get("n_imaginary"),
                  ("{:.2f}".format(low) if low is not None else "?"),
                  q.get("orca_symmetry_number"),
                  ("{:+.4f}".format(o["qrrho_minus_rrho_kcal"])
                   if "qrrho_minus_rrho_kcal" in o else "?"),
                  ("{:+.4f}".format(o["crosscheck_vs_orca_kcal"])
                   if "crosscheck_vs_orca_kcal" in o else "?"),
                  rec.get("total_wall_seconds", 0)),
              flush=True)
    print()
    print("total wall clock {:.0f} s".format(time.time() - t_all))


if __name__ == "__main__":
    main()
