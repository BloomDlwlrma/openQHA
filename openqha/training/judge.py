"""The ruler: what a fine-tuned potential is judged by (Algorithm 3 of design-phl-loss.md).

PRODUCTION. Ticket 14 of the Hessian-learning set.

The judge reads only SHIPPED paths: the full Cartesian Hessian from
`MACECalculator.get_hessian` (mace's `compute_hessians_vmap`, no graph) against the
Label, through `hessian_compare` -- the same four metric families every earlier
comparison in this repository used. **Nothing here calls the estimator or the training
loss.** The optimiser reads eq. 6; the judge reads eq. 1, exactly, from the full matrix.
That separation is the point: a loss that flatters itself cannot flatter the ruler.

What one judge run reports:

  per frame        `hessian_compare`'s metrics of the engine and of the base model at the
                   same geometry, plus `||A||_F^2 / n_vib` (eq. 1, from `phl`)
  per class        the structure classes of `index.dat`, so "how are we doing on
                   epoxides" has an answer
  per distribution `interpolation` (the by-frame test of the drawn molecules -- the split
                   measures interpolation WITHIN them, round-5 Q4's stated consequence),
                   `out_of_molecule` (the pinned seven: whole molecules never trained on)
                   and `in_distribution` (the shipped molecules, which MACE-OFF23 did see)
  anharmonic       reference modes the entropy tier must not be judged on (round-2 Q6 (a),
                   assumed): omega_r < ANHARMONIC_CM, or a `mode_curvature` self-check
                   above FD_SELF_CHECK_CM where a profile exists
  thermochemistry  read from the msRRHO Records on disk, never recomputed here (running
                   CREST and an optimisation inside the judge would make the judge a
                   producer of the numbers it judges)
  forgetting       E and F RMSE on a fixed SPICE draw, engine against base (round-2 Q7)
  verdict          one PASS / FAIL line per threshold (round-2 Q8, as the spec assumes)

Falsifiability. `--engine base` must PASS every no-degradation line and show exactly 0
against itself; `ScaledCalculator(calc, 0.9)` (forces x0.9, Hessian x0.81, so every
frequency x0.9) must FAIL the low-mode line. A judge that cannot fail is not a judge.
"""
import math
from pathlib import Path

import numpy as np

from ..data import dataset as dataset_mod
from ..store import dat, layout, property as prop, report
from ..thermochem import hessian as hessian_mod
from ..thermochem import hessian_compare as hc
from . import phl

PROGNAME = "openQHA hl_judge"
STEP = "judge"

#: a reference mode below this is set aside from the entropy tier (round-2 Q6 (a))
ANHARMONIC_CM = 30.0
#: ... and so is one whose along-mode finite-difference self-check exceeds this
FD_SELF_CHECK_CM = 5.0
#: "low mode" of the judge's headline number, as everywhere else in this repository
LOW_CM = hc.LOW_CM

#: the thresholds of round-2 Q8 as the spec assumes them; `judge.run(thresholds=)` overrides
THRESHOLDS = dict(
    low_mode_mae_cm=8.5,            # the in-distribution value (propanal, S0-C-41)
    model_error_s_ref=0.2,          # cal/mol/K, excluding the anharmonic modes
    in_distribution_degradation=0.15,   # no HIP metric worse than the base by more than this
    forgetting=0.15,                # SPICE E/F RMSE within this of the base's
)

#: the molecules MACE-OFF23 was trained on among the pinned seven (S0-C-40)
IN_DISTRIBUTION = ("dsgdb9nsd_000018", "dsgdb9nsd_000019", "dsgdb9nsd_000035", "dsgdb9nsd_000036")

#: the HIP metrics the no-degradation line watches
HIP_METRICS = ("HESSIAN_MAE", "FREQ_MAE_CM", "FREQ_MAE_LOW_CM", "EIGVAL_MAE_ECKART")

FRAME_ROW = {
    "qm9_index": ("String", None, "the molecule"),
    "generator": ("String", None, "basin / displaced / merged / saddle"),
    "basin": ("Integer", None, "basin index"),
    "k": ("Integer", None, "frame index within the generator"),
    "split": ("String", None, "the Dataset split the frame came from"),
    "distribution": ("String", None, "interpolation / out_of_molecule / in_distribution"),
    "classes": ("String", None, "structure classes, ';'-joined"),
    "n_low": ("Integer", None, "reference modes below the low cutoff"),
    "freq_mae_low_cm": ("Double", "cm^-1", "engine: MAE over the low reference modes"),
    "freq_mae_cm": ("Double", "cm^-1", "engine: MAE over all modes"),
    "hessian_mae": ("Double", "eV/A^2", "engine: element-wise MAE of the Cartesian Hessian"),
    "eigval_mae_eckart": ("Double", "eV/A^2/amu", "engine: MAE of the projected eigenvalues"),
    "mixing": ("Double", None, "engine: off-diagonal weight of D in the reference-mode basis"),
    "loss_exact": ("Double", "eV^2/A^4/amu^2", "eq. 1 exactly: ||A||_F^2 / n_vib"),
    "base_freq_mae_low_cm": ("Double", "cm^-1", "the base model on the same frame"),
    "base_freq_mae_cm": ("Double", "cm^-1", "the base model on the same frame"),
    "base_hessian_mae": ("Double", "eV/A^2", "the base model on the same frame"),
    "base_eigval_mae_eckart": ("Double", "eV/A^2/amu", "the base model on the same frame"),
    "base_loss_exact": ("Double", "eV^2/A^4/amu^2", "the base model on the same frame"),
    "n_anharmonic": ("Integer", None, "reference modes set aside from the entropy tier"),
    "noise_floor_cm": ("Double", "cm^-1", "REF_NOISE_FLOOR_CM: the rigid block of the unprojected REFERENCE Hessian -- a low-mode difference below it is unresolved, not model error (S0-C-44)"),
}

SCHEMA = {
    "Calculation_Info": {
        "PROGNAME": ("String", None, "the step that wrote this file"),
        "VERSION": ("String", None, "openQHA version"),
        "STATUS": ("String", None, "the completion marker"),
        "RUN": ("String", None, "the judged run (a train run name, or the engine name)"),
        "TAG": ("String", None, "the campaign tag"),
        "NAME": ("String", None, "the Dataset name"),
        "LEVEL": ("String", None, "the reference level of the Labels"),
        "DATASET_DIR": ("String", None, "the Dataset judged"),
        "ENGINE": ("String", None, "the potential under test"),
        "ENGINE_PARAMS_SHA256": ("String", None, "its parameter fingerprint"),
        "ENGINE_SCALE": ("Double", None, "1.0, or the calibration scale of a deliberately wrong potential"),
        "BASE_ENGINE": ("String", None, "the potential it is compared against"),
        "BASE_PARAMS_SHA256": ("String", None, "the base model's parameter fingerprint"),
        "MACE_VERSION": ("String", None, "mace.__version__"),
        "MACE_FORK_COMMIT": ("String", None, "commit of the mace checkout, or unknown"),
        "SPLITS": ("ArrayOfStrings", None, "the Dataset splits judged"),
        "N_FRAMES": ("Integer", None, "frames with a reference Hessian in those splits"),
        "N_MOLECULES": ("Integer", None, "molecules those frames came from"),
        "LOW_CUTOFF": ("Double", "cm^-1", "a reference mode below this is a low mode"),
        "ANHARMONIC_CM": ("Double", "cm^-1", "below this a mode leaves the entropy tier (round-2 Q6)"),
        "FD_SELF_CHECK_CM": ("Double", "cm^-1", "an along-mode self-check above this leaves it too"),
        "SECONDS": ("Double", "s", "wall time"),
        "VERDICT": ("String", None, "PASS when every threshold line passed, FAIL otherwise"),
    },
    "Distribution": {
        "DISTRIBUTION": ("String", None, "interpolation / out_of_molecule / in_distribution"),
        "N_FRAMES": ("Integer", None, "frames"),
        "N_MOLECULES": ("Integer", None, "molecules"),
        "FREQ_MAE_LOW_CM": ("Double", "cm^-1", "mean over frames of the low-mode MAE"),
        "FREQ_MAE_CM": ("Double", "cm^-1", "mean over frames of the full-spectrum MAE"),
        "HESSIAN_MAE": ("Double", "eV/A^2", "mean over frames"),
        "EIGVAL_MAE_ECKART": ("Double", "eV/A^2/amu", "mean over frames"),
        "LOSS_EXACT": ("Double", "eV^2/A^4/amu^2", "mean over frames of eq. 1"),
        "BASE_FREQ_MAE_LOW_CM": ("Double", "cm^-1", "the base model, same frames"),
        "BASE_FREQ_MAE_CM": ("Double", "cm^-1", "the base model, same frames"),
        "BASE_HESSIAN_MAE": ("Double", "eV/A^2", "the base model, same frames"),
        "BASE_EIGVAL_MAE_ECKART": ("Double", "eV/A^2/amu", "the base model, same frames"),
        "BASE_LOSS_EXACT": ("Double", "eV^2/A^4/amu^2", "the base model, same frames"),
    },
    "Class": {
        "CLASS": ("String", None, "the structure class"),
        "N_FRAMES": ("Integer", None, "frames of molecules in this class"),
        "N_MOLECULES": ("Integer", None, "molecules"),
        "FREQ_MAE_LOW_CM": ("Double", "cm^-1", "mean over frames"),
        "FREQ_MAE_CM": ("Double", "cm^-1", "mean over frames"),
        "LOSS_EXACT": ("Double", "eV^2/A^4/amu^2", "mean over frames"),
        "BASE_FREQ_MAE_LOW_CM": ("Double", "cm^-1", "the base model, same frames"),
        "BASE_FREQ_MAE_CM": ("Double", "cm^-1", "the base model, same frames"),
        "BASE_LOSS_EXACT": ("Double", "eV^2/A^4/amu^2", "the base model, same frames"),
    },
    "Thermochemistry": {
        "QM9_INDEX": ("String", None, "the molecule"),
        "SOURCE": ("String", None, "the msRRHO Record the numbers were read from, or - "),
        "S_MSRRHO": ("Double", "cal/mol/K", "the engine's msRRHO entropy at its own minima"),
        "MODEL_ERROR_S_REF": ("Double", "cal/mol/K", "engine S_REF - reference S_REF"),
        "N_ANHARMONIC": ("Integer", None, "modes set aside from the entropy tier"),
    },
    "Anharmonic": {
        "QM9_INDEX": ("String", None, "the molecule"),
        "BASIN": ("Integer", None, "basin index"),
        "MODE": ("Integer", None, "index of the reference mode, ascending"),
        "OMEGA_REF_CM": ("Double", "cm^-1", "the reference frequency of that mode"),
        "REASON": ("String", None, "below ANHARMONIC_CM, or an FD self-check above FD_SELF_CHECK_CM"),
    },
    "Forgetting": {
        "FILE": ("String", None, "the fixed SPICE draw"),
        "N_FRAMES": ("Integer", None, "frames evaluated"),
        "ENGINE_E_RMSE_MEV_PER_ATOM": ("Double", "meV/atom", "the potential under test"),
        "ENGINE_F_RMSE_MEV_A": ("Double", "meV/A", "the potential under test"),
        "BASE_E_RMSE_MEV_PER_ATOM": ("Double", "meV/atom", "the base model"),
        "BASE_F_RMSE_MEV_A": ("Double", "meV/A", "the base model"),
        "E_RATIO": ("Double", None, "engine / base"),
        "F_RATIO": ("Double", None, "engine / base"),
    },
    "Verdict": {
        "LINE": ("String", None, "the threshold"),
        "VALUE": ("Double", None, "what was measured"),
        "THRESHOLD": ("Double", None, "what it had to beat"),
        "RESULT": ("String", None, "PASS / FAIL / -"),
        "NOTE": ("String", None, "how to read it"),
    },
}


class ScaledCalculator:
    """A deliberately wrong potential: forces x `scale`, Hessian x `scale^2`, so every
    frequency is `scale` x the base model's. The judge's must-FAIL calibration -- a
    threshold that this passes is not a threshold."""

    def __init__(self, calc, scale):
        self.calc = calc
        self.scale = float(scale)
        for name in ("r_max", "device", "models"):
            if hasattr(calc, name):
                setattr(self, name, getattr(calc, name))

    def get_hessian(self, atoms=None):
        return np.asarray(self.calc.get_hessian(atoms)) * self.scale ** 2

    def get_forces(self, atoms=None):
        return np.asarray(self.calc.get_forces(atoms)) * self.scale

    def get_potential_energy(self, atoms=None, **kw):
        return float(self.calc.get_potential_energy(atoms, **kw))

    def calculate(self, *a, **k):
        return self.calc.calculate(*a, **k)

    def __getattr__(self, name):
        return getattr(self.calc, name)


def hessian_at(calc, atoms):
    """The engine's full Cartesian Hessian at this geometry, (3N, 3N) eV/A^2 -- the
    shipped path, symmetrised as `hessian_compare` expects."""
    n3 = 3 * len(atoms)
    h = np.asarray(calc.get_hessian(atoms)).reshape(n3, n3)
    return 0.5 * (h + h.T)


def anharmonic_modes(omega_ref_cm, qid, basin, profiles=None):
    """Reference modes the entropy tier must not be judged on (round-2 Q6 (a)): below
    `ANHARMONIC_CM`, or with an along-mode finite-difference self-check above
    `FD_SELF_CHECK_CM` in `profiles` (a `mode_curvature` Record's rows, when one exists)."""
    out = []
    for i, w in enumerate(omega_ref_cm):
        if float(w) < ANHARMONIC_CM:
            out.append(dict(QM9_INDEX=qid, BASIN=int(basin), MODE=int(i), OMEGA_REF_CM=float(w),
                            REASON="omega_r < {:.0f} cm^-1".format(ANHARMONIC_CM)))
            continue
        fd = (profiles or {}).get(i)
        if fd is not None and abs(float(fd)) > FD_SELF_CHECK_CM:
            out.append(dict(QM9_INDEX=qid, BASIN=int(basin), MODE=int(i), OMEGA_REF_CM=float(w),
                            REASON="FD self-check {:.1f} > {:.0f} cm^-1".format(float(fd), FD_SELF_CHECK_CM)))
    return out


def distribution_of(qid, pinned=dataset_mod.PINNED, in_distribution=IN_DISTRIBUTION):
    """Which row of the judge's table this molecule belongs to. `in_distribution` wins
    over `out_of_molecule`: a pinned molecule MACE-OFF23 was trained on is not held out,
    whatever the split says (S0-C-40)."""
    if qid in in_distribution:
        return "in_distribution"
    if qid in pinned:
        return "out_of_molecule"
    return "interpolation"


def frame_rows(dataset_dir, name, level, calc, base_calc=None, splits=("test",), index=None,
               progress=None):
    """One row per labelled frame of `splits`: `hessian_compare` for the engine and (when
    given) for the base model, plus eq. 1 exactly. Frames without a reference Hessian are
    skipped -- there is nothing to judge on them."""
    from ase.io import read
    dataset_dir = Path(dataset_dir)
    classes = {}
    if index is None:
        p = dataset_dir / "index.dat"
        index = dat.read_table(p) if p.is_file() else []
    for r in index:
        classes.setdefault(str(r.get("qm9_index")), str(r.get("classes") or "-"))

    rows, anharmonic = [], []
    for split in splits:
        path = dataset_dir / "{}.{}.extxyz".format(split, level)
        if not path.is_file():
            continue
        for atoms in read(str(path), index=":", format="extxyz"):
            if "hessian" not in atoms.info and "REF_hessian" not in atoms.info:
                continue
            flat = atoms.info.get("hessian", atoms.info.get("REF_hessian"))
            n3 = 3 * len(atoms)
            h_r = np.asarray(flat, dtype=float).reshape(n3, n3)
            masses, pos = atoms.get_masses(), atoms.positions
            qid = str(atoms.info.get("qm9_index", "-"))
            if progress:
                progress("{} {} b{} k{}".format(qid, atoms.info.get("generator"), atoms.info.get("basin"),
                                                atoms.info.get("k")))
            h_e = hessian_at(calc, atoms)
            cmp_e = hc.compare_hessians(h_e, h_r, masses, pos)
            row = dict(qm9_index=qid, generator=str(atoms.info.get("generator", "-")),
                       basin=int(atoms.info.get("basin", 0)), k=int(atoms.info.get("k", 0)),
                       split=split, distribution=distribution_of(qid), classes=classes.get(qid, "-"),
                       n_low=int(cmp_e["N_LOW"]),
                       freq_mae_low_cm=float(cmp_e["FREQ_MAE_LOW_CM"]),
                       freq_mae_cm=float(cmp_e["FREQ_MAE_CM"]),
                       hessian_mae=float(cmp_e["HESSIAN_MAE"]),
                       eigval_mae_eckart=float(cmp_e["EIGVAL_MAE_ECKART"]),
                       mixing=float(cmp_e["MIXING"]),
                       loss_exact=phl.projected_loss_full(h_e, h_r, masses, pos),
                       noise_floor_cm=float(cmp_e.get("REF_NOISE_FLOOR_CM", 0.0)))
            anh = anharmonic_modes(cmp_e["OMEGA_REF_CM"], qid, row["basin"])
            row["n_anharmonic"] = len(anh)
            if row["generator"] == "basin":
                anharmonic.extend(anh)
            if base_calc is not None:
                h_b = hessian_at(base_calc, atoms)
                cmp_b = hc.compare_hessians(h_b, h_r, masses, pos)
                row.update(base_freq_mae_low_cm=float(cmp_b["FREQ_MAE_LOW_CM"]),
                           base_freq_mae_cm=float(cmp_b["FREQ_MAE_CM"]),
                           base_hessian_mae=float(cmp_b["HESSIAN_MAE"]),
                           base_eigval_mae_eckart=float(cmp_b["EIGVAL_MAE_ECKART"]),
                           base_loss_exact=phl.projected_loss_full(h_b, h_r, masses, pos))
            rows.append(row)
    return rows, anharmonic


def _mean(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def aggregate(rows):
    """The two tables the judge is read from: per distribution and per class."""
    dist_rows = []
    for d in ("interpolation", "out_of_molecule", "in_distribution"):
        sel = [r for r in rows if r["distribution"] == d]
        if not sel:
            continue
        dist_rows.append(dict(
            DISTRIBUTION=d, N_FRAMES=len(sel), N_MOLECULES=len({r["qm9_index"] for r in sel}),
            FREQ_MAE_LOW_CM=_mean(sel, "freq_mae_low_cm"), FREQ_MAE_CM=_mean(sel, "freq_mae_cm"),
            HESSIAN_MAE=_mean(sel, "hessian_mae"), EIGVAL_MAE_ECKART=_mean(sel, "eigval_mae_eckart"),
            LOSS_EXACT=_mean(sel, "loss_exact"),
            BASE_FREQ_MAE_LOW_CM=_mean(sel, "base_freq_mae_low_cm"),
            BASE_FREQ_MAE_CM=_mean(sel, "base_freq_mae_cm"),
            BASE_HESSIAN_MAE=_mean(sel, "base_hessian_mae"),
            BASE_EIGVAL_MAE_ECKART=_mean(sel, "base_eigval_mae_eckart"),
            BASE_LOSS_EXACT=_mean(sel, "base_loss_exact")))
    names = sorted({c for r in rows for c in str(r["classes"]).split(";") if c and c != "-"})
    cls_rows = []
    for c in names:
        sel = [r for r in rows if c in str(r["classes"]).split(";")]
        cls_rows.append(dict(CLASS=c, N_FRAMES=len(sel), N_MOLECULES=len({r["qm9_index"] for r in sel}),
                             FREQ_MAE_LOW_CM=_mean(sel, "freq_mae_low_cm"), FREQ_MAE_CM=_mean(sel, "freq_mae_cm"),
                             LOSS_EXACT=_mean(sel, "loss_exact"),
                             BASE_FREQ_MAE_LOW_CM=_mean(sel, "base_freq_mae_low_cm"),
                             BASE_FREQ_MAE_CM=_mean(sel, "base_freq_mae_cm"),
                             BASE_LOSS_EXACT=_mean(sel, "base_loss_exact")))
    return dist_rows, cls_rows


def thermochemistry(root, tag, molecules, engine_level, reference_level, anharmonic_rows=()):
    """The entropy tier, READ from the msRRHO Records on disk -- never recomputed here.
    A judge that produced the numbers it judges would be marking its own work; running
    the pipeline (`s0_thermo_msrrho.py` with `S0_ENGINE=<engine>`) is a separate step,
    and a molecule whose Record is absent is reported as absent."""
    n_anh = {}
    for a in anharmonic_rows:
        n_anh[a["QM9_INDEX"]] = n_anh.get(a["QM9_INDEX"], 0) + 1
    out = []
    for qid in molecules:
        mol = layout.molecule_dir(root, tag, qid)
        path = layout.level_file(mol, reference_level, "thermo_msrrho.toml")
        row = dict(QM9_INDEX=qid, SOURCE="-", S_MSRRHO=None, MODEL_ERROR_S_REF=None,
                   N_ANHARMONIC=int(n_anh.get(qid, 0)))
        if Path(path).is_file():
            rec = prop.load(path)
            info = rec.get("Calculation_Info", {})
            row["SOURCE"] = str(path)
            for key in ("S_MSRRHO", "S_TOTAL", "S_ABS"):
                if info.get(key) is not None:
                    row["S_MSRRHO"] = float(info[key])
                    break
            if info.get("MODEL_ERROR_S_REF") is not None:
                row["MODEL_ERROR_S_REF"] = float(info["MODEL_ERROR_S_REF"])
        out.append(row)
    return out


def forgetting(calc, base_calc, frames_file, energy_key="REF_energy", forces_key="REF_forces"):
    """E and F RMSE of the engine and of the base model on a FIXED set of SPICE frames
    (round-2 Q7's judge). Energies per atom, so molecules of different size compare."""
    from ase.io import read
    path = Path(frames_file)
    if not path.is_file():
        return None
    frames = read(str(path), index=":", format="extxyz")
    acc = {"engine": ([], []), "base": ([], [])}
    for atoms in frames:
        e_ref = atoms.info.get(energy_key)
        if e_ref is None:
            try:
                e_ref = float(atoms.get_potential_energy())
            except Exception:                                    # noqa: BLE001
                continue
        f_ref = atoms.arrays.get(forces_key)
        if f_ref is None:
            try:
                f_ref = atoms.get_forces()
            except Exception:                                    # noqa: BLE001
                f_ref = None
        for which, c in (("engine", calc), ("base", base_calc)):
            at = atoms.copy()
            at.calc = c
            de = (float(at.get_potential_energy()) - float(e_ref)) / len(at)
            acc[which][0].append(de)
            if f_ref is not None:
                acc[which][1].append(np.asarray(at.get_forces()) - np.asarray(f_ref))
    def rmse(v):
        return float(np.sqrt(np.mean(np.concatenate([np.asarray(x).reshape(-1) for x in v]) ** 2))) if v else None
    e_eng, e_base = rmse(acc["engine"][0]), rmse(acc["base"][0])
    f_eng, f_base = rmse(acc["engine"][1]), rmse(acc["base"][1])
    return dict(FILE=str(path), N_FRAMES=len(frames),
                ENGINE_E_RMSE_MEV_PER_ATOM=None if e_eng is None else e_eng * 1000.0,
                ENGINE_F_RMSE_MEV_A=None if f_eng is None else f_eng * 1000.0,
                BASE_E_RMSE_MEV_PER_ATOM=None if e_base is None else e_base * 1000.0,
                BASE_F_RMSE_MEV_A=None if f_base is None else f_base * 1000.0,
                E_RATIO=None if not e_base else e_eng / e_base,
                F_RATIO=None if not f_base else f_eng / f_base)


def verdict(dist_rows, thermo_rows, forget_row, thresholds=None, noise_floor_cm=None):
    """One PASS / FAIL line per threshold of round-2 Q8. A line with nothing to measure
    is `-`, never a silent PASS. The Label's grid noise (S0-C-44) is printed beside the
    low-mode line: no threshold means anything below it."""
    t = dict(THRESHOLDS, **(thresholds or {}))
    lines = []
    held = [d for d in dist_rows if d["DISTRIBUTION"] in ("interpolation", "out_of_molecule")]
    if held:
        v = float(np.mean([d["FREQ_MAE_LOW_CM"] for d in held if d["FREQ_MAE_LOW_CM"] is not None]))
        note = "held-out low-mode MAE (< {:.0f} cm^-1 modes)".format(LOW_CM)
        if noise_floor_cm:
            note += "; the Label's own grid noise is ~{:.0f} cm^-1 (S0-C-44)".format(noise_floor_cm)
        lines.append(dict(LINE="held_out_low_mode_mae_cm", VALUE=v, THRESHOLD=t["low_mode_mae_cm"],
                          RESULT="PASS" if v <= t["low_mode_mae_cm"] else "FAIL", NOTE=note))
    errs = [abs(r["MODEL_ERROR_S_REF"]) for r in thermo_rows if r.get("MODEL_ERROR_S_REF") is not None]
    lines.append(dict(LINE="model_error_s_ref_cal_per_mol_K", VALUE=max(errs) if errs else None,
                      THRESHOLD=t["model_error_s_ref"],
                      RESULT=("-" if not errs else ("PASS" if max(errs) <= t["model_error_s_ref"] else "FAIL")),
                      NOTE="largest |engine - reference| entropy error, anharmonic modes excluded; "
                           "'-' means no msRRHO Record on disk for this engine"))
    ind = [d for d in dist_rows if d["DISTRIBUTION"] == "in_distribution"]
    if ind and ind[0].get("BASE_FREQ_MAE_CM") is not None:
        d = ind[0]
        worst, worst_name = None, "-"
        for key in ("FREQ_MAE_LOW_CM", "FREQ_MAE_CM", "HESSIAN_MAE", "EIGVAL_MAE_ECKART"):
            base = d.get("BASE_" + key)
            if base:
                ratio = d[key] / base - 1.0
                if worst is None or ratio > worst:
                    worst, worst_name = ratio, key
        lines.append(dict(LINE="in_distribution_degradation", VALUE=worst,
                          THRESHOLD=t["in_distribution_degradation"],
                          RESULT="PASS" if worst is not None and worst <= t["in_distribution_degradation"] else "FAIL",
                          NOTE="worst HIP metric against the base model ({}); a negative number is an improvement".format(worst_name)))
    if forget_row and forget_row.get("F_RATIO") is not None:
        v = max(forget_row["F_RATIO"], forget_row.get("E_RATIO") or 0.0) - 1.0
        lines.append(dict(LINE="forgetting", VALUE=v, THRESHOLD=t["forgetting"],
                          RESULT="PASS" if v <= t["forgetting"] else "FAIL",
                          NOTE="SPICE E/F RMSE against the base model's on {} frames".format(forget_row["N_FRAMES"])))
    else:
        lines.append(dict(LINE="forgetting", VALUE=None, THRESHOLD=t["forgetting"], RESULT="-",
                          NOTE="no SPICE draw on disk (scripts/tooling/s0_spice_test_draw.py)"))
    return lines


def run(root, tag, name, level, calc, engine_name, base_calc=None, base_engine=None, run_name=None,
        splits=("test",), scale=1.0, spice_file=None, thresholds=None, progress=None,
        engine_params_sha256=None, base_params_sha256=None, reference_level=None, write=True):
    """The whole judge: frames, aggregation, the entropy tier, forgetting, the verdict,
    and the Record under `<dataset>/judge/<run>/`."""
    import time
    t0 = time.time()
    dataset_dir = Path(dataset_mod.datasets_dir(root, tag, name))
    run_name = run_name or engine_name
    rows, anharmonic = frame_rows(dataset_dir, name, level, calc, base_calc=base_calc,
                                  splits=splits, progress=progress)
    if not rows:
        # A judge with nothing to judge must not answer PASS. The most common cause is a
        # Dataset that has no labelled frames in these splits (a by-molecule smoke set
        # puts them all in `test`, a fresh campaign in `pool`), or a path that is not a
        # Dataset directory at all.
        raise ValueError(
            "no labelled frame in {} for splits {} at level {}: nothing to judge. "
            "Check `04_dataset.py` put frames there (`index.dat` lists their splits) and that the "
            "level is the one the Labels were made at.".format(dataset_dir, ", ".join(splits), level))
    dist_rows, cls_rows = aggregate(rows)
    molecules = sorted({r["qm9_index"] for r in rows})
    thermo_rows = thermochemistry(root, tag, molecules, engine_name, reference_level or level,
                                  anharmonic_rows=anharmonic)
    forget_row = forgetting(calc, base_calc, spice_file) if (spice_file and base_calc is not None) else None
    floors = [r["noise_floor_cm"] for r in rows if r.get("noise_floor_cm")]
    lines = verdict(dist_rows, thermo_rows, forget_row, thresholds,
                    noise_floor_cm=max(floors) if floors else None)

    import mace
    from ..potentials import engine as engine_mod
    fork = engine_mod.mace_fork_info()
    info = dict(RUN=str(run_name), TAG=str(tag), NAME=str(name), LEVEL=str(level),
                DATASET_DIR=str(dataset_dir), ENGINE=str(engine_name),
                ENGINE_PARAMS_SHA256=str(engine_params_sha256 or "-"),
                ENGINE_SCALE=float(scale), BASE_ENGINE=str(base_engine or "-"),
                BASE_PARAMS_SHA256=str(base_params_sha256 or "-"),
                MACE_VERSION=mace.__version__, MACE_FORK_COMMIT=fork["mace_fork_commit"],
                SPLITS=[str(s) for s in splits], N_FRAMES=len(rows), N_MOLECULES=len(molecules),
                LOW_CUTOFF=float(LOW_CM), ANHARMONIC_CM=float(ANHARMONIC_CM),
                FD_SELF_CHECK_CM=float(FD_SELF_CHECK_CM), SECONDS=float(time.time() - t0),
                VERDICT="PASS" if all(l["RESULT"] != "FAIL" for l in lines) else "FAIL")
    out = dict(info=info, frames=rows, distributions=dist_rows, classes=cls_rows,
               thermochemistry=thermo_rows, anharmonic=anharmonic, forgetting=forget_row,
               verdict=lines, run_dir=dataset_dir / STEP / run_name)
    if write:
        write_record(out)
    return out


def write_record(out):
    d = Path(out["run_dir"])
    d.mkdir(parents=True, exist_ok=True)
    blocks = {"Calculation_Info": out["info"], "Distribution": out["distributions"],
              "Class": out["classes"], "Thermochemistry": out["thermochemistry"],
              "Anharmonic": out["anharmonic"],
              "Forgetting": [out["forgetting"]] if out["forgetting"] else [],
              "Verdict": out["verdict"]}
    missing = prop.write(d / (STEP + ".toml"), blocks, SCHEMA, prop.NORMAL_TERMINATION, PROGNAME)
    if missing:
        raise RuntimeError("judge.toml keys outside the schema: {}".format(missing))
    dat.write_table(d / (STEP + ".dat"), out["frames"], list(FRAME_ROW), FRAME_ROW)
    _write_report(d / (STEP + ".out"), out)


def _num(x, fmt="{:.3f}"):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else fmt.format(x)


def _write_report(path, out):
    info = out["info"]
    rep = report.Report(PROGNAME, "Judge of {!r} on {} at {}".format(info["ENGINE"], info["NAME"], info["LEVEL"]))
    rep.section("what was judged")
    for k in ("RUN", "TAG", "NAME", "LEVEL", "DATASET_DIR", "SPLITS", "N_FRAMES", "N_MOLECULES",
              "ENGINE", "ENGINE_PARAMS_SHA256", "ENGINE_SCALE", "BASE_ENGINE", "BASE_PARAMS_SHA256",
              "MACE_VERSION", "MACE_FORK_COMMIT", "LOW_CUTOFF", "ANHARMONIC_CM", "SECONDS"):
        rep.kv(k, info.get(k))
    rep.section("per distribution (the engine, then the base model)")
    rep.table(["distribution", "frames", "mols", "low MAE", "MAE", "||A||^2/n", "base low", "base MAE", "base ||A||^2/n"],
              [[d["DISTRIBUTION"], d["N_FRAMES"], d["N_MOLECULES"], _num(d["FREQ_MAE_LOW_CM"], "{:.2f}"),
                _num(d["FREQ_MAE_CM"], "{:.2f}"), _num(d["LOSS_EXACT"], "{:.4e}"),
                _num(d["BASE_FREQ_MAE_LOW_CM"], "{:.2f}"), _num(d["BASE_FREQ_MAE_CM"], "{:.2f}"),
                _num(d["BASE_LOSS_EXACT"], "{:.4e}")] for d in out["distributions"]],
              units=["", "", "", "cm^-1", "cm^-1", "", "cm^-1", "cm^-1", ""])
    if out["classes"]:
        rep.section("per structure class")
        rep.table(["class", "frames", "mols", "low MAE", "MAE", "base low", "base MAE"],
                  [[c["CLASS"], c["N_FRAMES"], c["N_MOLECULES"], _num(c["FREQ_MAE_LOW_CM"], "{:.2f}"),
                    _num(c["FREQ_MAE_CM"], "{:.2f}"), _num(c["BASE_FREQ_MAE_LOW_CM"], "{:.2f}"),
                    _num(c["BASE_FREQ_MAE_CM"], "{:.2f}")] for c in out["classes"]],
                  units=["", "", "", "cm^-1", "cm^-1", "cm^-1", "cm^-1"])
    if out["thermochemistry"]:
        rep.section("thermochemistry at the engine's own minima (READ from the msRRHO Records, not recomputed)")
        rep.table(["molecule", "S_msRRHO", "model error", "anharmonic modes", "source"],
                  [[r["QM9_INDEX"], _num(r["S_MSRRHO"], "{:.3f}"), _num(r["MODEL_ERROR_S_REF"], "{:+.3f}"),
                    r["N_ANHARMONIC"], r["SOURCE"]] for r in out["thermochemistry"]],
                  units=["", "cal/mol/K", "cal/mol/K", "", ""])
    if out["anharmonic"]:
        rep.section("modes set aside from the entropy tier (round-2 Q6)")
        rep.table(["molecule", "basin", "mode", "omega_ref", "reason"],
                  [[a["QM9_INDEX"], a["BASIN"], a["MODE"], _num(a["OMEGA_REF_CM"], "{:.2f}"), a["REASON"]]
                   for a in out["anharmonic"]], units=["", "", "", "cm^-1", ""])
    if out["forgetting"]:
        rep.section("forgetting (a fixed SPICE draw)")
        for k, v in out["forgetting"].items():
            rep.kv(k, v if not isinstance(v, float) else round(v, 4))
    rep.section("verdict")
    rep.table(["line", "value", "threshold", "result", "note"],
              [[l["LINE"], _num(l["VALUE"], "{:.4f}"), _num(l["THRESHOLD"], "{:.4f}"), l["RESULT"], l["NOTE"]]
               for l in out["verdict"]])
    rep.kv("VERDICT", info["VERDICT"])
    rep.write(path)
