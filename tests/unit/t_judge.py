"""Ticket 14 of the Hessian-learning set: the judge's arithmetic, its set-aside rule and
its verdict lines -- no engine (a fake calculator answers with stored Hessians).

Asserted: a per-frame row reproduces `hessian_compare`'s numbers and `||A||_F^2/n_vib`
equals ticket 11's `projected_loss_full` to 1e-14; `ScaledCalculator(0.9)` gives
frequencies 0.9x to 1e-10 and a Hessian scaled by 0.81; the distribution of a molecule
(in_distribution beats out_of_molecule beats interpolation); class and distribution
aggregation add up; the `[Anharmonic]` rule sets aside a 20 cm^-1 mode, keeps a 40 cm^-1
one, and sets aside a 40 cm^-1 mode whose FD self-check is 7 cm^-1; a verdict line with
nothing to measure is `-` and never a silent PASS; the must-pass and must-fail lines
behave; `forgetting` compares two calculators on a frame file; the Record round-trips.
"""
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
from openqha.data import dataset                                 # noqa: E402
from openqha.qm_interfaces import orca                           # noqa: E402
from openqha.store import layout, property as prop               # noqa: E402
from openqha.thermochem import hessian as hessian_mod            # noqa: E402
from openqha.thermochem import hessian_compare as hc             # noqa: E402
from openqha.training import judge, phl                          # noqa: E402

FIX = ROOT / "tests" / "data" / "propanal_molecule"
LEVEL = "wb97m-d3bj_def2-tzvppd"
FAIL = []


def check(label, ok, detail=""):
    print("  {:78s} {}".format(label, "ok" if ok else "FAIL " + str(detail)[:200]))
    if not ok:
        FAIL.append(label)


class FakeCalc:
    """Answers with a stored Hessian (and, for the forgetting line, a scaled energy)."""

    def __init__(self, hessian, energy=0.0, forces=None):
        self.h = np.asarray(hessian, dtype=float)
        self.e = float(energy)
        self.f = forces
        self.r_max = 5.0

    def get_hessian(self, atoms=None):
        n = self.h.shape[0] // 3
        return self.h.reshape(3 * n, n, 3)

    def get_potential_energy(self, atoms=None, **kw):
        return self.e

    def get_forces(self, atoms=None):
        n = len(atoms) if atoms is not None else self.h.shape[0] // 3
        return np.zeros((n, 3)) if self.f is None else np.asarray(self.f)


def main():
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.data import atomic_masses, atomic_numbers

    parsed = orca.parse_hess(layout.orca_level_file(FIX, LEVEL, 0, ".hess"))
    symbols = list(parsed["symbols"])
    x_r = np.asarray(parsed["positions_bohr"]) / orca.BOHR_PER_ANGSTROM
    masses = np.array([atomic_masses[atomic_numbers[s]] for s in symbols])
    H_r = orca.hessian_to_ev_per_angstrom2(parsed["hessian_eh_bohr2"])
    H_t = np.load(FIX / "mace" / "basin00" / "hessian_at_{}.npy".format(LEVEL))
    atoms = Atoms(symbols=symbols, positions=x_r)

    # --- ScaledCalculator ---------------------------------------------------------------------
    base = FakeCalc(H_t)
    scaled = judge.ScaledCalculator(base, 0.9)
    h_s = judge.hessian_at(scaled, atoms)
    check("ScaledCalculator: the Hessian is scaled by s^2", np.abs(h_s - 0.81 * H_t).max() < 1e-12)
    f_base = hc.projected(H_t, masses, x_r)["freq"]
    f_scaled = hc.projected(h_s, masses, x_r)["freq"]
    check("... so every frequency is 0.9x (1e-10)", np.abs(f_scaled - 0.9 * f_base).max() < 1e-10,
          np.abs(f_scaled - 0.9 * f_base).max())
    check("... and the forces are scaled by s",
          np.abs(scaled.get_forces(atoms) - 0.9 * base.get_forces(atoms)).max() < 1e-12)

    # --- one frame row -----------------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        at = atoms.copy()
        at.calc = SinglePointCalculator(at, energy=-1.0, forces=np.zeros((len(at), 3)))
        at.info = dict(qm9_index="dsgdb9nsd_000018", generator="basin", basin=0, k=0, hessian=H_r)
        dataset._write_split(td / "test.{}.extxyz".format(LEVEL), [(at, "test")], reference=True)
        rows, anh = judge.frame_rows(td, "ds", LEVEL, FakeCalc(H_t), base_calc=FakeCalc(H_t),
                                     splits=("test",), index=[dict(qm9_index="dsgdb9nsd_000018",
                                                                   classes="aldehyde;ketone")])
        cmp_e = hc.compare_hessians(H_t, H_r, masses, x_r)
        r = rows[0]
        # the tolerances are what an extxyz round trip allows: positions are written to
        # 8 decimals (5e-9 A here), which moves the Eckart projector and with it every
        # projected quantity -- 2e-10 cm^-1 on the low-mode MAE, 3e-14 on the loss
        check("a frame row reproduces hessian_compare's numbers",
              len(rows) == 1 and abs(r["freq_mae_low_cm"] - cmp_e["FREQ_MAE_LOW_CM"]) < 1e-8
              and abs(r["freq_mae_cm"] - cmp_e["FREQ_MAE_CM"]) < 1e-8
              and abs(r["hessian_mae"] - cmp_e["HESSIAN_MAE"]) < 1e-12
              and abs(r["mixing"] - cmp_e["MIXING"]) < 1e-10, r)
        check("`loss_exact` = ticket 11's projected_loss_full (1e-12, the same round trip)",
              abs(r["loss_exact"] - phl.projected_loss_full(H_t, H_r, masses, x_r)) < 1e-12)
        check("the base columns are filled and the classes come from index.dat",
              r["base_freq_mae_cm"] == r["freq_mae_cm"] and r["classes"] == "aldehyde;ketone")
        check("a shipped molecule is in_distribution, not out_of_molecule",
              r["distribution"] == "in_distribution")

    check("a pinned, non-shipped molecule is out_of_molecule",
          judge.distribution_of("dsgdb9nsd_000044") == "out_of_molecule")
    check("any other molecule is interpolation", judge.distribution_of("dsgdb9nsd_099999") == "interpolation")

    # --- the anharmonic rule --------------------------------------------------------------------------
    anh = judge.anharmonic_modes([20.0, 40.0, 200.0], "m", 0)
    check("a 20 cm^-1 mode is set aside, a 40 cm^-1 one is not",
          len(anh) == 1 and anh[0]["MODE"] == 0 and "omega_r" in anh[0]["REASON"], anh)
    anh = judge.anharmonic_modes([40.0, 200.0], "m", 0, profiles={0: 7.0, 1: 1.0})
    check("a 40 cm^-1 mode whose FD self-check is 7 cm^-1 is set aside too",
          len(anh) == 1 and anh[0]["MODE"] == 0 and "FD self-check" in anh[0]["REASON"], anh)

    # --- aggregation ------------------------------------------------------------------------------------
    rows = [dict(qm9_index="a", distribution="interpolation", classes="epoxide", freq_mae_low_cm=2.0,
                 freq_mae_cm=4.0, hessian_mae=0.1, eigval_mae_eckart=0.2, loss_exact=1.0,
                 base_freq_mae_low_cm=4.0, base_freq_mae_cm=8.0, base_hessian_mae=0.2,
                 base_eigval_mae_eckart=0.4, base_loss_exact=2.0),
            dict(qm9_index="b", distribution="interpolation", classes="epoxide;amide", freq_mae_low_cm=4.0,
                 freq_mae_cm=6.0, hessian_mae=0.3, eigval_mae_eckart=0.4, loss_exact=3.0,
                 base_freq_mae_low_cm=8.0, base_freq_mae_cm=12.0, base_hessian_mae=0.6,
                 base_eigval_mae_eckart=0.8, base_loss_exact=6.0),
            dict(qm9_index="c", distribution="in_distribution", classes="-", freq_mae_low_cm=1.0,
                 freq_mae_cm=1.0, hessian_mae=0.05, eigval_mae_eckart=0.05, loss_exact=0.5,
                 base_freq_mae_low_cm=1.0, base_freq_mae_cm=1.0, base_hessian_mae=0.05,
                 base_eigval_mae_eckart=0.05, base_loss_exact=0.5)]
    dist, cls = judge.aggregate(rows)
    check("per distribution: the means and the molecule counts",
          [d["DISTRIBUTION"] for d in dist] == ["interpolation", "in_distribution"]
          and dist[0]["N_FRAMES"] == 2 and dist[0]["N_MOLECULES"] == 2
          and dist[0]["FREQ_MAE_LOW_CM"] == 3.0 and dist[0]["BASE_FREQ_MAE_LOW_CM"] == 6.0, dist)
    check("per class: a molecule counts for every class it is in",
          {c["CLASS"]: c["N_FRAMES"] for c in cls} == {"epoxide": 2, "amide": 1}
          and [c for c in cls if c["CLASS"] == "amide"][0]["FREQ_MAE_LOW_CM"] == 4.0, cls)

    # --- the verdict --------------------------------------------------------------------------------------
    lines = judge.verdict(dist, [], None, noise_floor_cm=10.0)
    by = {l["LINE"]: l for l in lines}
    check("the low-mode line passes at 3.0 cm^-1 and names the Label's grid noise",
          by["held_out_low_mode_mae_cm"]["RESULT"] == "PASS" and "grid noise" in by["held_out_low_mode_mae_cm"]["NOTE"])
    check("no msRRHO Record and no SPICE draw -> '-', never a silent PASS",
          by["model_error_s_ref_cal_per_mol_K"]["RESULT"] == "-" and by["forgetting"]["RESULT"] == "-")
    check("the in-distribution line reads the worst HIP metric against the base",
          abs(by["in_distribution_degradation"]["VALUE"]) < 1e-12
          and by["in_distribution_degradation"]["RESULT"] == "PASS", by.get("in_distribution_degradation"))
    bad = [dict(d) for d in dist]
    bad[0]["FREQ_MAE_LOW_CM"] = 30.0
    check("the low-mode line fails at 30 cm^-1",
          {l["LINE"]: l["RESULT"] for l in judge.verdict(bad, [], None)}["held_out_low_mode_mae_cm"] == "FAIL")
    worse = [dict(d) for d in dist]
    worse[1]["FREQ_MAE_CM"] = 2.0                                 # twice the base's on the shipped molecules
    check("the no-degradation line fails when an in-distribution metric doubles",
          {l["LINE"]: l["RESULT"] for l in judge.verdict(worse, [], None)}["in_distribution_degradation"] == "FAIL")
    thermo = [dict(QM9_INDEX="a", SOURCE="x", S_MSRRHO=70.0, MODEL_ERROR_S_REF=-0.35, N_ANHARMONIC=0)]
    check("the entropy line fails at 0.35 cal/mol/K",
          {l["LINE"]: l["RESULT"] for l in judge.verdict(dist, thermo, None)}["model_error_s_ref_cal_per_mol_K"] == "FAIL")
    forget = dict(N_FRAMES=10, E_RATIO=1.05, F_RATIO=1.30)
    check("the forgetting line fails at a 30 % force ratio",
          {l["LINE"]: l["RESULT"] for l in judge.verdict(dist, [], forget)}["forgetting"] == "FAIL")

    # --- forgetting on two calculators ---------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        from ase.io import write
        frames = []
        for i in range(3):
            a = Atoms("H2O", positions=np.array([[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]]) + 0.01 * i)
            a.info["REF_energy"] = 0.0
            a.arrays["REF_forces"] = np.zeros((3, 3))
            frames.append(a)
        write(str(Path(td) / "spice.xyz"), frames, format="extxyz")
        f = judge.forgetting(FakeCalc(np.eye(9), energy=0.03), FakeCalc(np.eye(9), energy=0.003),
                             Path(td) / "spice.xyz")
        check("forgetting: per-atom E RMSE and the ratio to the base",
              f["N_FRAMES"] == 3 and abs(f["ENGINE_E_RMSE_MEV_PER_ATOM"] - 10.0) < 1e-9
              and abs(f["E_RATIO"] - 10.0) < 1e-9, f)
        check("a missing SPICE file gives None, not an exception",
              judge.forgetting(FakeCalc(np.eye(9)), FakeCalc(np.eye(9)), Path(td) / "nope.xyz") is None)

    # --- the Record ---------------------------------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        info = {k: (0.0 if t[0] == "Double" else 0 if t[0] == "Integer" else [] if t[0].startswith("ArrayOf") else "x")
                for k, t in judge.SCHEMA["Calculation_Info"].items()}
        info.update(RUN="r", VERDICT="PASS")
        out = dict(info=info, frames=rows, distributions=dist, classes=cls,
                   thermochemistry=thermo, anharmonic=anh, forgetting=None, verdict=lines,
                   run_dir=Path(td))
        judge.write_record(out)
        rec = prop.load(Path(td) / "judge.toml")
        check("judge.toml round-trips with the five blocks and NORMAL TERMINATION",
              rec["Calculation_Status"]["STATUS"] == prop.NORMAL_TERMINATION
              and len(rec["Distribution"]) == 2 and len(rec["Class"]) == 2 and len(rec["Verdict"]) == len(lines),
              sorted(rec))
        check("judge.out and judge.dat are written",
              (Path(td) / "judge.out").is_file() and (Path(td) / "judge.dat").is_file())

    print("\n{} checks, {} failed".format(21, len(FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
