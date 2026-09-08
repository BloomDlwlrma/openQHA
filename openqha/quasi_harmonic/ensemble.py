"""From per-basin free energies to the molecule's answer.

WHAT BRANCH B ACTUALLY DELIVERS
-------------------------------
Not one basin's entropy. The molecule's conformational free energy is a sum over every
basin branch A found:

    F_conf = -kT ln sum_i exp(-dG_i / kT)          dG_i relative to the lowest basin

and `dG_i` is assembled from two things measured in different places:

    dG_i = dE_el,i            branch A: the basin's electronic energy, relative
         - T * dS_i           branch B: its quasi-harmonic entropy, relative

WHY THIS MATTERS MORE THAN ANY SINGLE T*S
-----------------------------------------
**A setting that shifts every basin by the same amount changes F_conf by nothing.** The
exponentials are normalised, so a common offset cancels exactly. A 0.3 kcal/mol wobble in
one basin's T*S is alarming; the same wobble applied to all of them is invisible in the
answer.

That is the whole reason a parameter scan has to be run on a molecule with several
basins. Acetone has one (measured 2026-09-07), so on acetone `F_conf` is identically zero
and tells you nothing about the settings -- see examples/02 and examples/03.

WHAT IS AND IS NOT HERE
-----------------------
The symmetry number and the electronic degeneracy are NOT applied here. They enter
through `thermo.g_minus_eel`, which branch A and branch B both call per basin, so by the
time a basin's `g_minus_eel_kcal` reaches this module they are already in it. Applying
them again here would double-count, and that is exactly the kind of mistake that a
plausible-looking number hides.
"""
import numpy as np

from ..thermochem import thermo

#: kcal/(mol K). The same constant the rest of the package uses, imported rather than
#: written again -- two copies of a physical constant is one copy too many.
KB_KCAL = thermo.KB_KCAL


def relative_free_energies(basins, temperature_K=thermo.T_REF):
    """dG_i per basin, relative to the lowest, in kcal/mol.

    `basins` is a sequence of mappings with

        rel_electronic_kcal   the basin's electronic energy relative to the lowest
                              (branch A)
        TS_kcal               T * S for that basin (branch B). Optional: a basin with
                              no trajectory yet contributes its electronic term only,
                              and `has_entropy` records which happened.

    Returns (dG array, record). A basin whose entropy is missing is NOT silently given
    zero: it is flagged, because "we did not run it" and "its entropy is zero" are
    different statements and only one of them is ever true.
    """
    e = np.array([float(b.get("rel_electronic_kcal", 0.0)) for b in basins], dtype=float)
    ts = np.array([float(b["TS_kcal"]) if b.get("TS_kcal") is not None else np.nan
                   for b in basins], dtype=float)
    have = ~np.isnan(ts)
    g = e - np.where(have, ts, 0.0)
    g = g - g.min()
    return g, dict(
        n_basins=int(len(basins)),
        n_with_entropy=int(have.sum()),
        n_electronic_only=int((~have).sum()),
        temperature_K=float(temperature_K),
        warning=(None if have.all() else
                 "{} basin(s) have no quasi-harmonic entropy and contributed their "
                 "electronic term only. F_conf below is NOT the converged answer -- a "
                 "missing entropy is not a zero entropy.".format(int((~have).sum()))),
    )


def effective_basin_count(populations):
    """How many basins actually carry the answer: exp(-sum p ln p).

    1 when one basin holds everything, N when all N are equally occupied. It is the
    perplexity of the population distribution, and it is here because "23 basins" and
    "23 basins of which one holds 99%" are different molecules and the same headline.
    """
    p = np.asarray(populations, dtype=float)
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return float(np.exp(-(p * np.log(p)).sum()))


def conformational_free_energy(basins, temperature_K=thermo.T_REF):
    """F_conf and the Boltzmann populations, from per-basin (dE_el, T*S).

    F_conf is <= 0 by construction: it is the free energy of the multi-basin ensemble
    relative to occupying only the lowest basin, so adding basins can only lower it.
    A positive value coming out of here means the inputs are not relative energies.
    """
    g, rec = relative_free_energies(basins, temperature_K)
    kt = KB_KCAL * float(temperature_K)
    w = np.exp(-g / kt)
    z = float(w.sum())
    f = float(-kt * np.log(z))

    if f > 1e-9:
        raise ValueError(
            "F_conf came out positive ({:.4f} kcal/mol), which is impossible for an "
            "ensemble measured relative to its own lowest member. The inputs are "
            "probably absolute energies rather than relative ones.".format(f))

    populations = (w / z).tolist()
    rec.update(
        F_conf_kcal=f,
        partition_function_relative=z,
        delta_G_kcal=[float(x) for x in g],
        populations=[float(x) for x in populations],
        # The number of basins that carry the answer. 20 basins of which 19 sit 5 kcal
        # above the first is a one-basin molecule wearing a costume, and a correction
        # quoted without this is easy to over-read.
        effective_basins=effective_basin_count(populations),
        note=("F_conf is relative to occupying the lowest basin only, so it is <= 0. "
              "sigma and the electronic degeneracy are NOT applied here -- they are "
              "already inside each basin's g_minus_eel."),
    )
    return f, rec


def describe(basins, temperature_K=thermo.T_REF):
    """A printable table: per basin, its dG, its population, and whether it has entropy."""
    f, rec = conformational_free_energy(basins, temperature_K)
    lines = ["{:>6} {:>14} {:>12} {:>12} {:>12}  {}".format(
        "basin", "dE_el/kcal", "T*S/kcal", "dG/kcal", "population", "entropy?")]
    for i, b in enumerate(basins):
        ts = b.get("TS_kcal")
        lines.append("{:>6} {:>14.4f} {:>12} {:>12.4f} {:>12.4f}  {}".format(
            i, float(b.get("rel_electronic_kcal", 0.0)),
            "-" if ts is None else "{:.4f}".format(float(ts)),
            rec["delta_G_kcal"][i], rec["populations"][i],
            "yes" if ts is not None else "MISSING"))
    lines.append("")
    lines.append("F_conf = {:.4f} kcal/mol over {} basin(s), {} effective".format(
        f, rec["n_basins"], round(rec["effective_basins"], 2)))
    if rec.get("warning"):
        lines.append("WARNING: " + rec["warning"])
    return "\n".join(lines), rec
