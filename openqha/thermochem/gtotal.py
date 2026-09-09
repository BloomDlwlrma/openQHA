"""The full molecular Gibbs energy, term by term, from ONE spectrum.

    G_total = E_el + ZPE + [H(T) - H(0)] + G_trans + G_rot - T*S_vib

and the point of this module is that the whole thing can be evaluated twice -- once with
the Hessian's `omega_i`, once with the quasi-harmonic `nu_k` from a trajectory covariance
-- so that "may nu replace omega?" becomes a table of differences rather than an opinion.

NO MODE PAIRING IS INVOLVED HERE, AND THAT IS THE POINT
-------------------------------------------------------
Every term below is a SUM over the spectrum. Swapping one spectrum for the other is a
set-to-set substitution: it needs no permutation `pi`, no overlap, no cutoff. A hybrid
that takes mode `i` from one spectrum and mode `j` from the other needs all three, and
`mode_match` is where that lives. The two questions are different and only the first one
is the one usually being asked.

WHICH TERMS CAN POSSIBLY MOVE
-----------------------------
    G_trans   Sackur-Tetrode: depends on the molecular MASS and the pressure. No
              frequency enters it. **It is identical between the two routes, exactly.**
    G_rot     classical rigid rotor: depends on the moments of inertia and sigma. No
              frequency enters it either. It can differ between the routes ONLY through
              the geometry -- the Hessian's minimum against the trajectory's mean
              structure -- and that difference is a measurement, reported as
              `rotational_geometry_shift_kcal`.
    ZPE       0.5 h c sum(nu). Dominated by the STIFF end.
    H - H(0)  sum h c nu / (exp(h c nu / kT) - 1). Dominated by the SOFT end: a mode at
              3000 cm^-1 is not thermally excited at 300 K and contributes ~0.
    -T*S_vib  dominated by the SOFT end, like the enthalpy.

So the two routes can differ in exactly one place, the vibrational block, and the band
decomposition below says which end of the spectrum each vibrational term is asking about.
That is the whole argument for where a trajectory can and cannot help, made numerical.

UNITS: frequencies cm^-1, masses amu, positions angstrom, energies kcal/mol, T kelvin.
"""
import numpy as np

from . import thermo

#: Bands used throughout branch C for signed frequency deviations, reused here so a
#: frequency error in one band and a free-energy term in the same band can be put side
#: by side without redefining the boundaries.
BANDS = ((0.0, 500.0, "below_500"),
         (500.0, 1500.0, "500_to_1500"),
         (1500.0, float("inf"), "above_1500"))


def _clean(frequencies_cm):
    f = np.asarray(frequencies_cm, dtype=float)
    ok = np.isfinite(f) & (f > 0.0)
    return f[ok], int((~ok).sum())


def band_decomposition(frequencies_cm, temperature_K=thermo.T_REF):
    """How much of ZPE, of the thermal enthalpy and of T*S each band carries.

    This is the numerical form of the argument that ZPE and the thermal enthalpy are
    asking different questions of the same spectrum:

      * `ZPE = 0.5 h c sum(nu)` weights every mode by its frequency, so the stiff end
        dominates -- one C-H stretch at 3000 cm^-1 is worth 4.29 kcal/mol of ZPE while a
        torsion at 100 cm^-1 is worth 0.14.
      * `H(T) - H(0) = sum h c nu / (exp(h c nu / kT) - 1)` weights by thermal
        occupation, and a 3000 cm^-1 mode is not excited at 300 K, so the soft end
        dominates instead.

    A method that resolves the soft end well and the stiff end badly is therefore
    usable for one of these and not the other, and this function is how that is shown
    rather than asserted.
    """
    f, _ = _clean(frequencies_cm)
    kt = thermo.KB_KCAL * temperature_K
    zpe_i = 0.5 * thermo.HC_KCAL * f
    x = thermo.HC_KCAL * f / kt
    h_i = thermo.HC_KCAL * f / np.expm1(x)
    ts_i = np.array([thermo.s_mode_kcal_per_K(float(v), temperature_K)
                     for v in f]) * temperature_K

    out = dict(n_modes=int(f.size),
               ZPE_kcal=float(zpe_i.sum()),
               H_thermal_kcal=float(h_i.sum()),
               TS_kcal=float(ts_i.sum()),
               bands={})
    for lo, hi, name in BANDS:
        sel = (f >= lo) & (f < hi)
        out["bands"][name] = dict(
            n=int(sel.sum()),
            ZPE_kcal=float(zpe_i[sel].sum()),
            ZPE_fraction=float(zpe_i[sel].sum() / zpe_i.sum()) if zpe_i.sum() else 0.0,
            H_thermal_kcal=float(h_i[sel].sum()),
            H_thermal_fraction=(float(h_i[sel].sum() / h_i.sum())
                                if h_i.sum() else 0.0),
            TS_kcal=float(ts_i[sel].sum()),
            TS_fraction=float(ts_i[sel].sum() / ts_i.sum()) if ts_i.sum() else 0.0,
        )
    return out


def assemble(frequencies_cm, masses, positions, symmetry_number, degeneracy,
             electronic_energy_kcal=0.0, temperature_K=thermo.T_REF,
             pressure_Pa=thermo.P_STD, label=None):
    """Every term of `G_total` from one spectrum and one geometry.

    `symmetry_number` and `degeneracy` are passed straight through to `thermo`, which
    refuses to guess either. `electronic_energy_kcal` may be left at zero, in which case
    the record is a `G - E_el` and says so.

    An imaginary or zero frequency is refused rather than dropped: a spectrum with one
    is not a minimum, and averaging over the rest would produce a plausible number for a
    structure that has no thermodynamics.
    """
    f = np.asarray(frequencies_cm, dtype=float)
    bad = f[~(np.isfinite(f) & (f > 0.0))]
    if bad.size:
        raise ValueError(
            "{} non-positive or non-finite frequency/frequencies (lowest {:.2f}) -- "
            "this spectrum does not describe a minimum and G_total is not defined on "
            "it".format(bad.size, float(np.nanmin(bad))))

    g = thermo.g_minus_eel(masses, positions, f, symmetry_number, degeneracy,
                           temperature_K=temperature_K, pressure_Pa=pressure_Pa)
    vib, rot, tr, el = (g["vibrational"], g["rotational"], g["translational"],
                        g["electronic"])
    kt = thermo.KB_KCAL * temperature_K
    zpe = float(vib["ZPE_kcal"])
    h_vib_thermal = float(vib["E_vib_kcal"]) - zpe

    # Gibbs pieces, each one A + its own share of pV. The pV term belongs to the
    # molecule as a whole and is kept on the translational line, which is where the
    # ideal-gas G = A + kT convention puts it.
    g_trans = float(tr["value_kcal"]) + kt
    g_rot = float(rot["A_rot_kcal"])
    ts_vib = float(vib["S_vib_kcal_per_K"]) * temperature_K

    total = float(electronic_energy_kcal) + float(g["G_minus_Eel_kcal"])
    return dict(
        label=label,
        temperature_K=float(temperature_K), pressure_Pa=float(pressure_Pa),
        n_modes=int(f.size),
        symmetry_number=int(symmetry_number), electronic_degeneracy=int(degeneracy),
        E_el_kcal=float(electronic_energy_kcal),
        ZPE_kcal=zpe,
        H_thermal_vib_kcal=h_vib_thermal,
        E_rot_kcal=float(rot["E_rot_kcal"]),
        E_trans_kcal=1.5 * kt,
        pV_kcal=kt,
        G_trans_kcal=g_trans,
        G_rot_kcal=g_rot,
        A_elec_kcal=float(el["A_elec_kcal"]),
        TS_vib_kcal=ts_vib,
        minus_TS_vib_kcal=-ts_vib,
        A_vib_kcal=float(vib["A_vib_kcal"]),
        G_minus_Eel_kcal=float(g["G_minus_Eel_kcal"]),
        G_total_kcal=total,
        moments_amu_A2=list(rot["moments_amu_A2"]),
        lowest_cm_inv=float(f.min()), highest_cm_inv=float(f.max()),
        bands=band_decomposition(f, temperature_K),
    )


def compare(reference, test, ref_name="hessian", test_name="qha"):
    """`test - reference`, term by term, plus what can and cannot have moved.

    `G_trans` is asserted identical: nothing in the Sackur-Tetrode expression depends on
    a frequency, so a difference there is an input error, not a result. `G_rot` may
    differ, but only through the GEOMETRY, and the difference is labelled as such so it
    is never read as an effect of the spectrum.
    """
    terms = ("E_el_kcal", "ZPE_kcal", "H_thermal_vib_kcal", "E_rot_kcal",
             "E_trans_kcal", "pV_kcal", "G_trans_kcal", "G_rot_kcal",
             "minus_TS_vib_kcal", "G_minus_Eel_kcal", "G_total_kcal")
    delta = {k: float(test[k] - reference[k]) for k in terms}

    if abs(delta["G_trans_kcal"]) > 1e-9:
        raise ValueError(
            "G_trans differs by {:.3e} kcal/mol between the two routes. It depends only "
            "on the molecular mass, the temperature and the pressure -- no frequency "
            "enters it -- so a difference here means the two routes were given "
            "different inputs.".format(delta["G_trans_kcal"]))

    return dict(
        reference=ref_name, test=test_name, delta=delta,
        rotational_geometry_shift_kcal=delta["G_rot_kcal"],
        rotational_note=("G_rot depends on the moments of inertia and sigma, not on any "
                         "frequency. Any difference here is the GEOMETRY the two routes "
                         "were evaluated at -- the Hessian's minimum against the "
                         "trajectory's mean structure -- and never an effect of "
                         "substituting one spectrum for the other."),
        vibrational_only_delta_kcal=float(
            delta["ZPE_kcal"] + delta["H_thermal_vib_kcal"]
            + delta["minus_TS_vib_kcal"]),
        band_delta={
            name: dict(
                ZPE_kcal=float(test["bands"]["bands"][name]["ZPE_kcal"]
                               - reference["bands"]["bands"][name]["ZPE_kcal"]),
                H_thermal_kcal=float(test["bands"]["bands"][name]["H_thermal_kcal"]
                                     - reference["bands"]["bands"][name]["H_thermal_kcal"]),
                TS_kcal=float(test["bands"]["bands"][name]["TS_kcal"]
                              - reference["bands"]["bands"][name]["TS_kcal"]),
            ) for _lo, _hi, name in BANDS},
    )


def format_table(records, deltas=None):
    """A printable term-by-term table: one column per route, one row per term."""
    rows = [("E_el", "E_el_kcal"),
            ("ZPE", "ZPE_kcal"),
            ("H(T)-H(0) vib", "H_thermal_vib_kcal"),
            ("E_rot", "E_rot_kcal"),
            ("E_trans", "E_trans_kcal"),
            ("pV", "pV_kcal"),
            ("G_trans", "G_trans_kcal"),
            ("G_rot (sigma)", "G_rot_kcal"),
            ("-T*S_vib", "minus_TS_vib_kcal"),
            ("(G - E_el)", "G_minus_Eel_kcal"),
            ("G_total", "G_total_kcal")]
    names = [r.get("label") or "route{}".format(i) for i, r in enumerate(records)]
    width = max(12, max(len(n) for n in names) + 2)
    head = "{:>16}" + ("{:>" + str(width) + "}") * len(records)
    if deltas is not None:
        head += "{:>14}"
        out = [head.format("term", *names, "difference")]
    else:
        out = [head.format("term", *names)]
    out.append("-" * len(out[0]))
    for text, key in rows:
        cells = ["{:.4f}".format(r[key]) for r in records]
        line = ("{:>16}" + ("{:>" + str(width) + "}") * len(records)).format(text, *cells)
        if deltas is not None:
            line += "{:>14}".format("{:+.4f}".format(deltas["delta"][key]))
        out.append(line)
    return "\n".join(out)


def format_bands(record):
    """Which band carries which term, as fractions -- the 2.1 / 2.2 argument, measured."""
    b = record["bands"]
    head = "{:>14} {:>4} {:>12} {:>8} {:>14} {:>8} {:>12} {:>8}"
    out = [head.format("band / cm^-1", "n", "ZPE", "share", "H(T)-H(0)", "share",
                       "T*S", "share"),
           "-" * 90]
    for _lo, _hi, name in BANDS:
        d = b["bands"][name]
        out.append(head.format(
            name.replace("_", " "), d["n"],
            "{:.4f}".format(d["ZPE_kcal"]), "{:.1%}".format(d["ZPE_fraction"]),
            "{:.4f}".format(d["H_thermal_kcal"]), "{:.1%}".format(d["H_thermal_fraction"]),
            "{:.4f}".format(d["TS_kcal"]), "{:.1%}".format(d["TS_fraction"])))
    out.append(head.format("total", b["n_modes"],
                           "{:.4f}".format(b["ZPE_kcal"]), "100.0%",
                           "{:.4f}".format(b["H_thermal_kcal"]), "100.0%",
                           "{:.4f}".format(b["TS_kcal"]), "100.0%"))
    return "\n".join(out)
