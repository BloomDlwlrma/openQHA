"""The package-1 batch screen applies the frequency floor (ticket 41).

UNIT. No CREST, no engine: `hessian_screen` is run on synthetic spectra through a
stubbed Hessian/projection pair; under a second.

The rule is ticket 37's, lifted into the batch path -- not a second rule: a candidate
whose lowest projected mode lies in the inversion window [ithr, 0) is a Basin, kept and
carrying its lowest frequency and its window count; only a lowest mode strictly below
ithr is a saddle, rejected and listed with its below-floor count and lowest frequency
(the same counters, from the same pure `census_verdict`).

The floor reaches the screen from the caller: the batch driver passes the configured
`package2.ithr_cm`; the parameter's default is the `crest` preset's own value. The old
`reject_imaginary` switch (any non-zero imaginary count threw the candidate out) is
deleted with the wiring on both screens (ruling Q1 of 2026-09-25).

Held here:

    A. the wired decision: a window candidate is kept with its counters, a below-floor
       candidate is rejected and listed with the same counters, a clean candidate is
       kept, and a mode exactly on the line is in the window;
    B. the floor comes from the caller, not from a literal in the screen;
    C. the verdict is `census_verdict`'s -- one rule, one source;
    D. `reject_imaginary` is gone from both signatures; the floor parameter is named;
    E. a record written before the floor rule stays readable: it renders through
       `record.write_log` (the superseded migration path re-renders old records) and
       converts to parquet rows, with the new fields reading as absent, not as zero.
"""
import inspect
import sys
from pathlib import Path


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))
FAIL = []


def check(name, ok, detail=None):
    print("  {} {}".format(name, "ok" if ok else "FAIL" + (" " + repr(detail) if detail is not None else "")))
    if not ok:
        FAIL.append(name)


import numpy as np                                      # noqa: E402
from ase import Atoms                                   # noqa: E402
from openqha.conformer_search import crest_census       # noqa: E402
from openqha.thermochem import hessian as _hessian_mod  # noqa: E402
from openqha.thermochem import thermo                   # noqa: E402


def run_screen(spectra, ithr=None):
    """`hessian_screen` on one synthetic spectrum per basin (no engine).

    The Hessian and the projection are stubbed; everything the screen does with them --
    the verdict, the counters, the record, the rejection list -- is real.
    """
    basins = [Atoms("H2O", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
              for _ in spectra]
    it = iter(spectra)

    def fake_hessian(atoms, calc, mode="analytic"):
        return np.eye(3 * len(atoms)) * 50.0, 0.0

    def fake_projection(h, masses, positions):
        nu = np.asarray(next(it), dtype=float)
        return dict(frequencies_cm_inv=[float(x) for x in nu],
                    n_imaginary=int((nu < 0.0).sum()),
                    n_rigid_modes_removed=6,
                    separation_gap_ratio=1.0e3)

    real_hess, real_proj = _hessian_mod.hessian, _hessian_mod.project_and_diagonalise
    _hessian_mod.hessian = fake_hessian
    _hessian_mod.project_and_diagonalise = fake_projection
    try:
        if ithr is None:
            return crest_census.hessian_screen(basins, None)
        return crest_census.hessian_screen(basins, None, ithr_cm=ithr)
    finally:
        _hessian_mod.hessian = real_hess
        _hessian_mod.project_and_diagonalise = real_proj


def main():
    # A -- the wired decision at the crest floor (-50 cm^-1): window kept, below-floor
    # rejected and listed, clean kept, a mode on the line in the window
    recs, keep, saddles = run_screen([[-6.84, 130.0, 420.0],
                                      [-195.79, 130.0],
                                      [40.0, 120.0],
                                      [-50.0, 120.0]])
    check("A: a window candidate is a basin with its lowest frequency and window "
          "count; a below-floor candidate is a saddle; the clean and on-the-line "
          "candidates are basins",
          keep == [0, 2, 3]
          and recs[0]["verdict"] == "basin" and recs[0]["n_inversion_window"] == 1
          and recs[0]["n_below_ithr"] == 0
          and recs[0]["lowest_frequency_cm_inv"] == -6.84
          and recs[1]["verdict"] == "saddle" and recs[1]["n_below_ithr"] == 1
          and recs[1]["n_inversion_window"] == 0
          and recs[2]["verdict"] == "basin" and recs[2]["n_inversion_window"] == 0
          and recs[3]["verdict"] == "basin" and recs[3]["n_inversion_window"] == 1
          and recs[3]["n_below_ithr"] == 0,
          [(r.get("verdict"), r.get("n_below_ithr"), r.get("n_inversion_window"))
           for r in recs])
    check("A2: the rejected candidate is listed with its below-floor count and lowest "
          "frequency, and nothing else is",
          len(saddles) == 1 and saddles[0]["index"] == 1
          and saddles[0]["n_below_ithr"] == 1
          and saddles[0]["lowest_frequency_cm_inv"] == -195.79
          and saddles[0]["n_imaginary"] == 1,
          saddles)
    check("A3: every record -- kept or rejected -- carries the same floor fields",
          all(("verdict" in r and "n_below_ithr" in r and "n_inversion_window" in r
                and "lowest_frequency_cm_inv" in r) for r in recs) and len(recs) == 4,
          recs[1])

    # B -- the caller's floor is what decides, not a literal: the same -6.84 spectrum
    # is a window at -50 and a saddle at -5
    recs_b, keep_b, saddles_b = run_screen([[-6.84, 130.0]], ithr=-5.0)
    check("B: a stricter floor rejects the same spectrum the crest floor admits",
          keep_b == [] and len(saddles_b) == 1
          and saddles_b[0]["n_below_ithr"] == 1
          and saddles_b[0]["lowest_frequency_cm_inv"] == -6.84,
          (keep_b, saddles_b))
    recs_d, keep_d, _ = run_screen([[-6.84, 130.0]])
    check("B2: with no floor passed the screen uses ITHR_CM_DEFAULT, the crest preset's "
          "own value -- no third copy of -50",
          keep_d == [0] and recs_d[0]["n_inversion_window"] == 1
          and crest_census.ITHR_CM_DEFAULT == thermo.MSRRHO_PRESETS["crest"]["ithr_cm"] == -50.0,
          (keep_d, crest_census.ITHR_CM_DEFAULT))

    # C -- one rule, one source: the wired verdict is census_verdict's on the same
    # spectrum (ticket 37's helper, built on floor_verdict / n_below_ithr / n_in_window)
    nu = [-30.0, 100.0, 640.0]
    v = crest_census.census_verdict(nu, -50.0)
    recs_c, keep_c, _ = run_screen([nu])
    r0 = recs_c[0]
    check("C: the screen's verdict and counters ARE census_verdict's -- no second rule",
          keep_c == [0] and r0["verdict"] == v["verdict"]
          and r0["n_below_ithr"] == v["n_below_ithr"]
          and r0["n_inversion_window"] == v["n_inversion_window"]
          and r0["lowest_frequency_cm_inv"] == v["lowest_frequency_cm_inv"],
          (r0, v))

    # D -- the switch is deleted with the wiring (both screens, ruling Q1's landing of
    # ticket 37's review row 7); the floor parameter is named
    sig = inspect.signature(crest_census.hessian_screen)
    sig_c = inspect.signature(crest_census.census_from_frames)
    check("D: reject_imaginary is gone from both screens' signatures; ithr_cm is the "
          "floor entry",
          "reject_imaginary" not in sig.parameters and "ithr_cm" in sig.parameters
          and "reject_imaginary" not in sig_c.parameters and "ithr_cm" in sig_c.parameters,
          (list(sig.parameters), list(sig_c.parameters)))

    # E -- old-record readability: a record written before the floor rule renders and
    # reads. Its saddle keeps the imaginary wording -- its lowest mode need not lie
    # below ithr, because the 2026-09-25 rule is what made that distinction -- and the
    # new fields read as absent ('-'), never as a zero verdict.
    import tempfile as _tempfile
    from openqha.store import record as _record
    old_rec = dict(
        qm9_index="dsgdb9nsd_000000", smiles="CCC=O",
        tighten_fmax_eV_A=1e-4, dedup_threshold_A=0.125, temperature_K=298.15,
        n_saddles_rejected=1,
        saddles=[dict(index=3, n_imaginary=1, lowest_frequency_cm_inv=-6.84)],
        basin_energies_eV=[-1.0], basin_relative_kcal=[0.0],
        basin_hessian=[dict(n_imaginary=0, n_rigid_modes_removed=6,
                            lowest_frequency_cm_inv=42.0,
                            frequencies_cm_inv=[42.0, 100.0])],
        populations=dict(boltzmann_weights=[1.0], weight_of_lowest=1.0))
    with _tempfile.TemporaryDirectory() as td:
        path = Path(td) / "old.log"
        _record.write_log(old_rec, path)
        text = path.read_text(encoding="utf-8")
    rows = _record.record_to_rows(old_rec)
    check("E: a pre-floor record renders and reads -- the saddle keeps the imaginary "
          "wording, the new fields read as absent",
          "saddle rejected: index 3 has 1 imaginary mode(s)" in text
          and "saddle rejected: index 3 has 1 mode(s) below ithr" not in text
          and rows[0]["n_below_ithr"] is None and rows[0]["n_inversion_window"] is None
          and rows[0]["ithr_cm"] is None and rows[0]["n_imaginary"] == 0,
          (rows[0].get("n_below_ithr"), [ln for ln in text.splitlines()
                                         if "saddle rejected" in ln]))

    print("PASS" if not FAIL else "FAIL: " + "; ".join(FAIL))
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
