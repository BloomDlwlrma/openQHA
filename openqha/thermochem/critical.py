"""Critical constants, near-critical state points, and the verdict on **whether this
route still holds for a given molecule**.

Package 3 takes the "near-critical gas phase plus thermodynamic extrapolation" route
(option A, chosen 2026-08-31), so every species needs its own critical
temperature before a state point can be fixed. This module keeps the three layers apart:

| Layer | Function | Identity |
|---|---|---|
| **database lookup** | `from_database(cas)` | **citable**; each entry carries its method name (HEOS / IUPAC / CRC ...) |
| **group-contribution estimate** | `from_joback(smiles)` | **an estimate**; for screening and ranking, and **never for fixing a production state point** |
| **state point and verdict** | `near_critical_state`, `assess` | computed from one of the two above, and **delivers a verdict on whether the route still holds** |

**How far Joback can be trusted** (back-tested against this repository's 7 species,
2026-08-31):

| Species | Joback `T_c` | Database `T_c` | Deviation |
|---|---:|---:|---:|
| acetone | 500.6 | 508.1 | -1.5% |
| oxetane | 500.7 | 507.9 | -1.4% |
| propanal | 489.5 | 505.0 | -3.1% |
| **acetamide** | **570.3** | **761.0** | **-25.1%** |

**Joback's groups contain no hydrogen bonding**, so amides, carboxylic acids and polyols
come out **systematically low**. `assess()` flags a molecule containing those groups
separately.
"""
import math

# CODATA 2018, the same set as openqha.thermo and openqha.vdos
KB_SI = 1.380649e-23
ATM_PA = 101325.0

# ---- verdict thresholds: all **conventions that may be changed**; the consequences are
# ---- written into what assess() returns ----------------------------------------------
T_POTENTIAL_LIMIT_K = 600.0      # above this, MACE-OFF23-SC's training distribution has
                                 # almost no structures at the corresponding conditions
SPAN_LIMIT_K = 300.0             # ceiling on the extrapolation span; beyond it a
                                 # temperature extrapolation is not credible
COLLISION_LIMIT_PS = 20.0        # ceiling on the collision interval; beyond it
                                 # thermalisation by collision is too slow
HARD_SPHERE_DIAMETER_A = 5.0     # molecular diameter for the hard-sphere estimate, rough


def lee_kesler_pr(tr, omega):
    """The Lee-Kesler corresponding-states vapour-pressure correlation, returning
    `P_sat / P_c`."""
    f0 = 5.92714 - 6.09648 / tr - 1.28862 * math.log(tr) + 0.169347 * tr ** 6
    f1 = 15.2518 - 15.6875 / tr - 13.4721 * math.log(tr) + 0.43577 * tr ** 6
    return math.exp(f0 + omega * f1)


def acentric_from_boiling(tb_K, tc_K, pc_Pa):
    """Back out the acentric factor from the normal boiling point (Joback does not give
    it).

    At the normal boiling point `P_sat = 1 atm`, which substituted into Lee-Kesler gives:
        ln(1 atm / P_c) = f0(T_br) + omega * f1(T_br)
        =>  omega = ( -ln(P_c[atm]) - f0(T_br) ) / f1(T_br)

    **The sign is `-ln(P_c)`.** Written as `+ln` it gives acetone `omega = -2.16` (the
    true value is 0.307) and makes the number density 4.5 times too high -- found and
    corrected on 2026-08-31 in a smoke test over 300 molecules.
    """
    tbr = tb_K / tc_K
    pc_atm = pc_Pa / ATM_PA
    f0 = 5.92714 - 6.09648 / tbr - 1.28862 * math.log(tbr) + 0.169347 * tbr ** 6
    f1 = 15.2518 - 15.6875 / tbr - 13.4721 * math.log(tbr) + 0.43577 * tbr ** 6
    return (-math.log(pc_atm) - f0) / f1


def from_database(cas):
    """Look up critical constants in `chemicals`. Returns a dict **carrying the method
    name**, so the provenance can be checked.

    **Note**: `chemicals` may itself return a group-contribution value (cyclopropanol, for
    instance, returns `JOBACK`). The caller must inspect `tc_sources[0]` to tell an
    experimental value from an estimate -- `assess()` does that for you.
    """
    from chemicals import Tc, Pc, omega as omega_f
    from chemicals.critical import Tc_methods, Pc_methods
    from chemicals.acentric import omega_methods
    tc, pc, w = Tc(cas), Pc(cas), omega_f(cas)
    if not (tc and pc):
        raise ValueError("the database has no critical constants for {}".format(cas))
    return dict(source="database", cas=cas, Tc_K=float(tc), Pc_Pa=float(pc),
                Pc_bar=float(pc) / 1e5,
                omega=(float(w) if w is not None
                       else acentric_from_boiling(_tb(cas), tc, pc)),
                tc_sources=Tc_methods(cas), pc_sources=Pc_methods(cas),
                omega_sources=omega_methods(cas))


def _tb(cas):
    from chemicals import Tb
    t = Tb(cas)
    if not t:
        raise ValueError("the database has no boiling point for {}, so the acentric "
                         "factor cannot be backed out".format(cas))
    return float(t)


def from_joback(smiles):
    """A Joback group-contribution estimate. **This is an estimate, not a measurement** --
    see the back-test table in the module docstring."""
    from thermo import Joback
    j = Joback(str(smiles))
    est = j.estimate()
    tb, tc, pc = est["Tb"], est["Tc"], est["Pc"]
    if not (tc and tc > 0 and pc and pc > 0):
        raise ValueError("Joback returned a non-positive Tc/Pc")
    return dict(source="joback", smiles=smiles, Tb_K=float(tb), Tc_K=float(tc),
                Pc_Pa=float(pc), Pc_bar=float(pc) / 1e5,
                omega=acentric_from_boiling(tb, tc, pc), joback_status=j.status)


def collision_interval_ps(density_nm3, temperature_K, molar_mass=58.08,
                          diameter_A=HARD_SPHERE_DIAMETER_A):
    """Hard-sphere estimate of the intermolecular collision interval, in ps.
    `z = sqrt(2) n sigma v_bar`.

    **This is a rough estimate** -- the diameter is a constant 5 A and is not adjusted to
    the actual size of the molecule. Its purpose is to judge "is thermalisation fast
    enough", not to give a physical quantity.
    """
    m_kg = molar_mass * 1.66053906660e-27
    v_bar = math.sqrt(8.0 * KB_SI * temperature_K / (math.pi * m_kg))    # m/s
    sigma_m2 = math.pi * (diameter_A * 1e-10) ** 2
    n_m3 = density_nm3 * 1e27
    z = math.sqrt(2.0) * n_m3 * sigma_m2 * v_bar                          # 1/s
    return 1e12 / z if z > 0 else float("inf")


def near_critical_state(tc_K, pc_bar, omega, reduced_temperature=0.886,
                        n_molecules=8, molar_mass=58.08):
    """Given critical constants, compute the state point of the near-critical saturated
    vapour.

    **The number density is computed from the ideal gas, `n = P/kT`, which is a lower
    bound** -- near saturation `Z < 1` and the true density is higher (acetone near
    `T_r = 0.886` has `Z` about 0.7-0.8). In production this should be recalibrated
    against the simulated virial pressure.
    """
    t = reduced_temperature * tc_K
    p_bar = lee_kesler_pr(reduced_temperature, omega) * pc_bar
    n_nm3 = p_bar * 1e5 / (KB_SI * t) / 1e27
    return dict(reduced_temperature=reduced_temperature, temperature_K=t,
                Psat_bar=p_bar, density_nm3=n_nm3,
                density_is_lower_bound=True,
                box_length_A=(n_molecules / n_nm3) ** (1.0 / 3.0) * 10.0,
                n_molecules=n_molecules,
                collision_interval_ps=collision_interval_ps(n_nm3, t, molar_mass),
                extrapolation_span_K=t - 298.15)


# ---- the verdict: does this route still hold for this molecule -----------------------
_HBOND_HINTS = (("C(=O)N", "amide"), ("C(=O)O", "carboxylic acid"), ("NC=O", "amide"))


def assess(constants, state, smiles=None,
           t_limit_K=T_POTENTIAL_LIMIT_K, span_limit_K=SPAN_LIMIT_K,
           collision_limit_ps=COLLISION_LIMIT_PS):
    """**Judge whether the "near-critical plus temperature extrapolation" route still
    holds for this molecule**, giving a reason for each finding.

    Returns a dict carrying `verdict` (`ok` / `caution` / `reject`) and a list of `flags`.
    **Every criterion can fail, every threshold is a convention that may be changed, and a
    failure reports the number rather than only a label.**
    """
    flags = []
    t = state["temperature_K"]

    if t > t_limit_K:
        flags.append(dict(
            flag="temperature_above_potential_range", value=t, limit=t_limit_K,
            severity="reject",
            reason=("the state temperature {:.1f} K exceeds {:.0f} K -- "
                    "MACE-OFF23-SC's training distribution holds almost no structures at "
                    "this temperature, the molecule may decompose, and its thermal "
                    "stability is in doubt".format(t, t_limit_K))))
    span = state["extrapolation_span_K"]
    if span > span_limit_K:
        flags.append(dict(
            flag="extrapolation_span_too_large", value=span, limit=span_limit_K,
            severity="reject",
            reason=("the extrapolation span {:.1f} K exceeds {:.0f} K -- "
                    "`S(T2)=S(T1)+integral Cp/T dT` is not credible over that "
                    "distance".format(span, span_limit_K))))
    ci = state["collision_interval_ps"]
    if ci > collision_limit_ps:
        flags.append(dict(
            flag="collisions_too_rare", value=ci, limit=collision_limit_ps,
            severity="caution",
            reason=("the collision interval {:.1f} ps exceeds {:.0f} ps -- "
                    "thermalisation by collision is too slow and the trajectory would "
                    "have to be very long".format(ci, collision_limit_ps))))

    src = constants.get("source")
    if src == "joback":
        flags.append(dict(
            flag="critical_constants_estimated", value="joback", severity="caution",
            reason=("the critical constants are a Joback group-contribution estimate, not "
                    "an experimental value; back-tested here: hydrocarbons, ethers and "
                    "ketones low by 1-3%, **amides low by 25%**")))
    elif src == "database":
        first = (constants.get("tc_sources") or [""])[0]
        if "JOBACK" in first.upper() or "WILSON" in first.upper():
            flags.append(dict(
                flag="database_returned_estimate", value=first, severity="caution",
                reason=("the database returned {} -- which is itself a group-contribution "
                        "estimate, not an experimental value".format(first))))
    if smiles:
        for pat, label in _HBOND_HINTS:
            if pat in str(smiles):
                flags.append(dict(
                    flag="strong_hydrogen_bonding", value=label, severity="caution",
                    reason=("contains a {} group; Joback's groups include no hydrogen "
                            "bonding, so the critical temperature of such molecules is "
                            "systematically underestimated".format(label))))
                break

    sev = [f["severity"] for f in flags]
    verdict = "reject" if "reject" in sev else ("caution" if flags else "ok")
    return dict(verdict=verdict, flags=flags,
                temperature_K=t, extrapolation_span_K=span,
                collision_interval_ps=ci,
                thresholds=dict(t_limit_K=t_limit_K, span_limit_K=span_limit_K,
                                collision_limit_ps=collision_limit_ps),
                note=("the thresholds are **conventions that may be changed**, not "
                      "physical constants; **they are never adjusted to let a particular "
                      "molecule through** (stage 1 governance)."))


def assess_smiles(smiles, reduced_temperature=0.886, n_molecules=8, cas=None):
    """One step: give a SMILES (or a CAS) and get the constants, the state point and the
    verdict."""
    if cas:
        try:
            c = from_database(cas)
        except Exception:
            c = from_joback(smiles)
    else:
        c = from_joback(smiles)
    st = near_critical_state(c["Tc_K"], c["Pc_bar"], c["omega"],
                             reduced_temperature, n_molecules)
    return dict(constants=c, state=st,
                assessment=assess(c, st, smiles=smiles))
