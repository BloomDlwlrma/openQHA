"""The reference level retries a soft saddle before any verdict.

INTEGRATION fixture run. No real ORCA: a fake binary answers `S0_ORCA_BIN`, so
`orca.optimise_and_hessian` -> `_run_job` -> the published file group -> `parse_hess` ->
`verify_hess_frequencies` -> the retry -> the Records are the real ones end to end. The
propanal fixture's basin-00 reference file group is removed from a copy and the fake
plays two attempts:

  attempt 1  a WINDOW saddle: the fixture's own basin-00 Hessian with its lowest
             eigenvalue shifted so the lowest mode is -6.84 cm^-1 (measured on Tianhe),
             `$vibrational_frequencies` rewritten to match;
  attempt 2  either the fixture's real minimum job (A: the retry resolves it), the
             saddle again (B: it stays a soft saddle), or a saddle below the floor
             (C: a plain saddle whose reason says it was retried).

A. kept: one retry from the first job's relaxed geometry, the file group REPLACED by the
   retry's job (byte-identical to the fixture), merge map `soft_saddle = false`, record
   SOFT_SADDLE = false.
B. excluded soft saddle: merge map `status = saddle`, `soft_saddle = true`,
   `lowest_frequency_cm` = the saddle's lowest mode; the record's excluded row carries
   SOFT_SADDLE = true and the same frequency; the Report says it was retried.
C. excluded plain saddle: the retry settled below the floor -- `soft_saddle = false`, the
   lowest frequency recorded, and the Report's reason says the retry ran and the mode sits
   below ithr (never "inside the inversion window").
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))
from openqha.qm_interfaces import orca                        # noqa: E402
from openqha.quasi_harmonic import mode_match                 # noqa: E402
from openqha.store import dat, layout, property as prop       # noqa: E402
from openqha.thermochem import hessian as hessian_mod         # noqa: E402
from openqha.thermochem import reference_level as rl          # noqa: E402

SRC = ROOT / "tests" / "data" / "propanal_molecule"
LEVEL = "wb97m-d3bj_def2-tzvppd"
BASIN = 0
TARGET_CM = -6.84
FAIL = []

#: A minimal ORCA stdout: the parser needs the version, the last FINAL SINGLE POINT
#: ENERGY, the RMS gradient and the route, and the file group needs the terminal line.
FAKE_OUT = """Program Version 6.0.1 - RELEASE
FINAL SINGLE POINT ENERGY      -192.345678901
RMS gradient 0.00000789

SCF Response
****ORCA TERMINATED NORMALLY****
"""


def check(label, ok, detail=""):
    print("  {:74s} {}".format(label, "ok" if ok else "FAIL " + str(detail)[:240]))
    if not ok:
        FAIL.append(label)


def _write_hess(path, symbols, masses, positions_bohr, hessian_eh_bohr2, freqs_cm):
    """A minimal `.hess` `parse_hess` reads: `$hessian`, `$atoms`, and
    `$vibrational_frequencies` as 6 rigid zeros then `freqs_cm` (3N-6 values)."""
    n3 = 3 * len(symbols)
    lines = ["$orca_hessian_file", "", "$act_atom", "  0", "", "$act_coord", "  0", "",
             "$act_energy", "        0.000000", "", "$multiplicity", "  1", "",
             "$hessian", str(n3)]
    lines.append(" ".join("{:5d}".format(c) for c in range(n3)))
    for r in range(n3):
        lines.append("{:5d} ".format(r) + " ".join("{:19.10E}".format(v) for v in hessian_eh_bohr2[r]))
    lines += ["", "$atoms", str(len(symbols))]
    for s, m, p in zip(symbols, masses, positions_bohr):
        lines.append(" {:2s} {:10.5f} {:18.12f} {:18.12f} {:18.12f}"
                     .format(s, float(m), float(p[0]), float(p[1]), float(p[2])))
    lines += ["", "$vibrational_frequencies", str(n3)]
    for i, nu in enumerate([0.0] * 6 + [float(x) for x in freqs_cm]):
        lines.append("{:5d}{:22.16f}".format(i, nu))
    lines += ["", "$end", ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _window_saddle(src_hess, out_path, target_cm=TARGET_CM):
    """The fixture Hessian with its lowest vibrational eigenvalue shifted to
    `target_cm`, the printed frequencies rewritten to match. Returns the lowest mode."""
    parsed = orca.parse_hess(src_hess)
    H = np.asarray(orca.hessian_to_ev_per_angstrom2(parsed["hessian_eh_bohr2"]), dtype=float)
    masses = np.asarray(parsed["masses_amu"], dtype=float)
    pos_bohr = np.asarray(parsed["positions_bohr"], dtype=float)
    freq, vec = mode_match.projected_modes(H, masses, pos_bohr / orca.BOHR_PER_ANGSTROM, "hessian")
    c = hessian_mod.CM_INV_PER_SQRT_EV_A2_AMU
    lam = np.sign(freq[0]) * (abs(float(freq[0])) / c) ** 2
    lam_target = -abs(target_cm / c) ** 2
    m3 = np.repeat(masses, 3)
    hm = H / np.sqrt(np.outer(m3, m3))
    H2 = (hm + (lam_target - lam) * np.outer(vec[:, 0], vec[:, 0])) * np.sqrt(np.outer(m3, m3))
    H2 = 0.5 * (H2 + H2.T)
    freqs2, _ = mode_match.projected_modes(H2, masses, pos_bohr / orca.BOHR_PER_ANGSTROM, "hessian")
    _write_hess(out_path, parsed["symbols"], masses, pos_bohr,
                H2 / (orca.EV_PER_HARTREE * orca.BOHR_PER_ANGSTROM ** 2), freqs2)
    return float(freqs2[0])


def _fake_orca(path):
    """The fake binary: attempt 1 = the saddle, then attempt 2 = `<state>/attempt2.*`;
    the call count and the two inputs are kept for the assertions."""
    code = '''#!{python}
import os, shutil
from pathlib import Path
state = Path(os.environ["FAKE_ORCA_STATE"])
first = not (state / "first").exists()
with (state / "calls").open("a") as fh:
    fh.write("x\\n")
tag = "attempt1" if first else "attempt2"
if first:
    (state / "first").touch()
shutil.copy(state / (tag + ".hess"), "job.hess")
shutil.copy(state / (tag + ".out"), "job.out")
shutil.copy("job.inp", state / (tag + ".inp"))
'''.format(python=sys.executable)
    Path(path).write_text(code, encoding="utf-8")
    os.chmod(path, 0o755)


def _input_geometry(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    i = next(k for k, line in enumerate(lines) if line.startswith("* xyz"))
    return np.array([[float(x) for x in line.split()[1:4]] for line in lines[i + 1:]
                     if len(line.split()) == 4])


def _scenario(tmp, name, attempt2_hess, attempt2_out):
    """One copy of the fixture, basin 00's file group removed, the fake ORCA installed."""
    mol = Path(tmp) / name
    shutil.copytree(SRC, mol)
    for ext in (".hess", ".out", ".inp", ".xyz"):
        p = layout.orca_level_file(mol, LEVEL, BASIN, ext)
        if p.is_file():
            p.unlink()
    state = Path(tmp) / (name + "_state")
    state.mkdir()
    lowest = _window_saddle(layout.orca_level_file(SRC, LEVEL, BASIN, ".hess"),
                            state / "attempt1.hess")
    (state / "attempt1.out").write_text(FAKE_OUT, encoding="utf-8")
    shutil.copy(attempt2_hess, state / "attempt2.hess")
    shutil.copy(attempt2_out, state / "attempt2.out")
    fake = Path(tmp) / (name + "_orca.py")
    _fake_orca(fake)
    os.environ["S0_ORCA_BIN"] = str(fake)
    os.environ["FAKE_ORCA_STATE"] = str(state)
    out = rl.run_calculation(mol, level=LEVEL, basins=[0, 1, 2])
    return mol, state, lowest, out


def main():
    src_hess = layout.orca_level_file(SRC, LEVEL, BASIN, ".hess")
    src_out = layout.orca_level_file(SRC, LEVEL, BASIN, ".out")
    with tempfile.TemporaryDirectory(prefix="reference_retry_") as tmp:
        # the generator itself: the written .hess must pass the round-trip and its lowest
        # mode must be the window value
        probe = Path(tmp) / "probe.hess"
        lowest = _window_saddle(src_hess, probe)
        saddle_out = Path(tmp) / "saddle.out"
        saddle_out.write_text(FAKE_OUT, encoding="utf-8")
        rt = orca.verify_hess_frequencies(orca.parse_hess(probe))
        check("the generated window saddle round-trips and its lowest mode is %.2f cm^-1" % lowest,
              abs(lowest - TARGET_CM) < 0.05 and abs(rt["lowest_cm_inv"] - TARGET_CM) < 0.05
              and rt["n_imaginary"] == 1, (lowest, rt["lowest_cm_inv"]))

        # ---- A. the retry reaches a minimum ------------------------------------------
        mol, state, lowest, out = _scenario(tmp, "resolve", src_hess, src_out)
        rows = {int(r["mace_basin"]): r for r in
                dat.read_table(layout.level_file(mol, LEVEL, "merge_map.dat"))}
        calls = (state / "calls").read_text(encoding="utf-8").split()
        check("A: the window saddle was retried once (two ORCA calls; basins 1-2 reused)",
              len(calls) == 2, len(calls))
        parsed1 = orca.parse_hess(state / "attempt1.hess")
        first_relaxed = np.asarray(parsed1["positions_bohr"], dtype=float) / orca.BOHR_PER_ANGSTROM
        dev = np.abs(_input_geometry(state / "attempt2.inp") - first_relaxed).max()
        check("A: the retry's input is the first job's ORCA-relaxed geometry (max dev %.1e A)" % dev,
              dev < 1e-6, dev)
        check("A: the basin is kept/merged with soft_saddle false, lowest_frequency_cm = %.2f"
              % rows[0]["lowest_frequency_cm"],
              rows[0]["status"] in ("kept", "merged") and rows[0]["soft_saddle"] is False
              and rows[0]["lowest_frequency_cm"] > 100)
        check("A: the retry REPLACED the file group (the .hess is the fixture's, byte for byte)",
              layout.orca_level_file(mol, LEVEL, BASIN, ".hess").read_bytes() == src_hess.read_bytes())
        doc = prop.load(layout.level_file(mol, LEVEL, "thermo_msrrho.toml"))
        check("A: no basin is excluded and every [[Basin]] row says SOFT_SADDLE = false",
              doc["Result"]["N_EXCLUDED"] == 0 and all(r["SOFT_SADDLE"] is False for r in doc["Basin"]))
        check("A: the merge map lists every MACE basin once with the new columns",
              sorted(rows) == [0, 1, 2] and all("soft_saddle" in r for r in rows.values()))

        # ---- B. the retry stays a saddle ---------------------------------------------
        mol, state, lowest, out = _scenario(tmp, "stay", probe, saddle_out)
        rows = {int(r["mace_basin"]): r for r in
                dat.read_table(layout.level_file(mol, LEVEL, "merge_map.dat"))}
        calls = (state / "calls").read_text(encoding="utf-8").split()
        check("B: two ORCA calls; the merge map is status saddle, soft_saddle true, lowest %.2f"
              % rows[0]["lowest_frequency_cm"],
              len(calls) == 2 and rows[0]["status"] == "saddle" and rows[0]["soft_saddle"] is True
              and abs(rows[0]["lowest_frequency_cm"] - lowest) < 1e-6)
        doc = prop.load(layout.level_file(mol, LEVEL, "thermo_msrrho.toml"))
        excluded = [b for b in doc["Basin"] if b["EXCLUDED"]]
        check("B: the record's excluded row carries SOFT_SADDLE = true and the lowest frequency",
              len(excluded) == 1 and excluded[0]["SOFT_SADDLE"] is True
              and abs(excluded[0]["LOWEST_FREQ"] - lowest) < 1e-6
              and excluded[0]["N_IMAGINARY"] == 1, excluded)
        check("B: the ensemble counts it excluded (N_BASINS 3, N_INCLUDED 2, N_EXCLUDED 1)",
              doc["Result"]["N_BASINS"] == 3 and doc["Result"]["N_INCLUDED"] == 2
              and doc["Result"]["N_EXCLUDED"] == 1)
        rtext = layout.level_file(mol, LEVEL, "thermo_msrrho.out").read_text(encoding="utf-8")
        check("B: the Report says the soft saddle was retried and names the window",
              "soft saddle" in rtext and "retried once" in rtext and "inversion window" in rtext)
        check("B: the other merge-map rows are not soft saddles and carry their lowest mode",
              all(rows[b]["soft_saddle"] is False and rows[b]["lowest_frequency_cm"] > 0 for b in (1, 2)),
              {b: (rows[b]["soft_saddle"], rows[b]["lowest_frequency_cm"]) for b in (1, 2)})

        # ---- C. the retry settles below the floor: a plain saddle that was retried -------
        deep = Path(tmp) / "deep.hess"
        deep_lowest = _window_saddle(src_hess, deep, target_cm=-61.68)
        mol, state, lowest, out = _scenario(tmp, "deep", deep, saddle_out)
        rows = {int(r["mace_basin"]): r for r in
                dat.read_table(layout.level_file(mol, LEVEL, "merge_map.dat"))}
        doc = prop.load(layout.level_file(mol, LEVEL, "thermo_msrrho.toml"))
        excluded = [b for b in doc["Basin"] if b["EXCLUDED"]]
        rtext = layout.level_file(mol, LEVEL, "thermo_msrrho.out").read_text(encoding="utf-8")
        flat = " ".join(rtext.split())          # the Report wraps its lines
        check("C: below the floor after the retry: status saddle, soft_saddle false, lowest %.2f"
              % rows[0]["lowest_frequency_cm"],
              rows[0]["status"] == "saddle" and rows[0]["soft_saddle"] is False
              and abs(rows[0]["lowest_frequency_cm"] - deep_lowest) < 1e-6)
        check("C: the record says the saddle below the floor was retried, not that it is soft",
              len(excluded) == 1 and excluded[0]["SOFT_SADDLE"] is False
              and abs(excluded[0]["LOWEST_FREQ"] - deep_lowest) < 1e-6
              and "below the floor (ithr = -50); retried once" in flat)
    if FAIL:
        print("FAIL: " + ", ".join(FAIL))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
