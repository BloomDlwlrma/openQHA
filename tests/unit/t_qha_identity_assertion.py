"""Branch B acceptance criterion 11: the identity assertion is shown what it must reject.

UNIT. Milliseconds, no engine.

A criterion must first be shown an example it should fail
---------------------------------------------------------
`qha.assert_trajectory_identity` exists because three specific mistakes -- a bias
potential, constraints, and a repartitioned hydrogen mass -- do not raise, do not warn and
do not look wrong. They return a healthy, correctly dimensioned entropy that is
systematically too high. A guard against that is worth exactly as much as the evidence
that it fires, so this test hands it one violation at a time and requires a refusal each
time, plus an acceptance of the clean case.

The metadata under test is the same dict `scripts/production/s0_B_qha_trajectory.py`
writes next to every trajectory, so a change to that writer that quietly drops a field
fails here rather than three steps downstream.
"""
import sys
from pathlib import Path


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


sys.path.insert(0, str(_repo_root()))
from openqha import qha                                     # noqa: E402


def clean():
    """What the branch B driver writes: no bias, no constraints, real masses, 1 fs."""
    return dict(bias_potential=None, constraints=None,
                hydrogen_mass_amu=1.008, timestep_fs=1.0,
                thermostat_fixcm=False,
                source="openQHA.branchB.langevin")


def clean_nose_hoover():
    """What the driver writes when the thermostat is Nose-Hoover.

    `fixcm` is absent on purpose: it is an ASE Langevin option and recording it here would
    be a field that looks checked and is not.
    """
    return dict(bias_potential=None, constraints=None,
                hydrogen_mass_amu=1.008, timestep_fs=1.0,
                thermostat="ase.md.nose_hoover_chain.NoseHooverChainNVT",
                thermostat_tdamp_fs=20.0, thermostat_chain_length=3,
                source="openQHA.branchB.nose_hoover")


def rejects(meta):
    try:
        qha.assert_trajectory_identity(meta)
        return False, "accepted"
    except ValueError as exc:
        return True, str(exc).splitlines()[-1].strip()[:110]


def main():
    print("=" * 92)
    print("Branch B criterion 11 -- the trajectory identity assertion")
    print("=" * 92)

    cases = []

    ok, why = rejects(clean())
    cases.append(("a clean branch B trajectory is ACCEPTED", not ok, why))

    for label, patch in [
        ("a metadynamics bias potential",
         dict(bias_potential="rmsd metadynamics, 0.5 kcal/mol gaussians")),
        ("a well-tempered bias from the branch C sampling loop",
         dict(bias_potential="Well_Potential")),
        ("SHAKE on all bonds",
         dict(constraints="h-bonds")),
        ("any constraint at all",
         dict(constraints=["FixBondLengths"])),
        ("hydrogen mass repartitioned to deuterium",
         dict(hydrogen_mass_amu=2.0)),
        ("hydrogen mass repartitioned to 4 amu",
         dict(hydrogen_mass_amu=4.0)),
        ("the CREST 5 fs timestep",
         dict(timestep_fs=5.0)),
        ("a 2 fs timestep, which is merely different rather than absurd",
         dict(timestep_fs=2.0)),
        ("frames taken from CREST",
         dict(source="crest imtd-gc metadynamics")),
        ("frames taken from the branch C active-learning sampler",
         dict(source="alf_sampling")),
        ("no metadata about the hydrogen mass at all",
         dict(hydrogen_mass_amu=None)),
        ("no metadata about the timestep at all",
         dict(timestep_fs=None)),
        ("ASE's Langevin default fixcm=True, measured at 429 K when 298.15 K was asked "
         "for on a 10-atom molecule, worth +0.70 kcal/mol on T*S",
         dict(thermostat_fixcm=True)),
        ("no record of how the thermostat treated the centre of mass",
         dict(thermostat_fixcm=None)),
    ]:
        meta = clean()
        meta.update(patch)
        ok, why = rejects(meta)
        cases.append(("REJECTED: " + label, ok, why))

    # ---- Nose-Hoover ---------------------------------------------------------------
    # The branch moved to Nose-Hoover on instruction. `fixcm` means nothing there, so the
    # fifth check is replaced rather than dropped, and the replacement needs its own
    # failing examples or it is not a check.
    ok, why = rejects(clean_nose_hoover())
    cases.append(("a Nose-Hoover trajectory at the measured coupling time is ACCEPTED",
                  not ok, why))

    for label, patch in [
        ("Nose-Hoover at the CONVENTIONAL 100 fs coupling -- measured to return the "
         "softest mode at 179 cm^-1 against a true 79.7, costing 1.15 kcal/mol",
         dict(thermostat_tdamp_fs=100.0)),
        ("Nose-Hoover at 419 fs, the coupling time matched to the lowest mode's period, "
         "which is the intuitive choice and is measured to be among the worst",
         dict(thermostat_tdamp_fs=419.0)),
        ("Nose-Hoover with no record of the coupling time",
         dict(thermostat_tdamp_fs=None)),
        ("plain Nose-Hoover, chain length 1 -- what GROMACS SILENTLY falls back to under "
         "the default leap-frog integrator",
         dict(thermostat_chain_length=1)),
        ("Nose-Hoover with no record of the chain length",
         dict(thermostat_chain_length=None)),
    ]:
        meta = clean_nose_hoover()
        meta.update(patch)
        ok, why = rejects(meta)
        cases.append(("REJECTED: " + label, ok, why))

    # The full CREST protocol -- all three violations at once -- must be refused, and the
    # message must name all three, because a guard that stops at the first problem leaves
    # the reader fixing them one run at a time.
    meta = clean()
    meta.update(bias_potential="rmsd metadynamics", constraints="all bonds (SHAKE)",
                hydrogen_mass_amu=2.0, timestep_fs=5.0, thermostat_fixcm=True,
                source="crest")
    keys = ("bias_potential", "constraints", "hydrogen_mass_amu", "timestep_fs",
            "thermostat_fixcm", "source")
    try:
        qha.assert_trajectory_identity(meta)
        all_named, detail = False, "accepted"
    except ValueError as exc:
        text = str(exc)
        named = sum(k in text for k in keys)
        all_named = named == len(keys)
        detail = "{} of {} problems named in one message".format(named, len(keys))
    cases.append(("the whole CREST protocol is refused and every violation is reported "
                  "in one message", all_named, detail))

    bad = 0
    for text, ok, detail in cases:
        print("[{}] {}".format("PASS" if ok else "FAIL", text))
        print("       {}".format(detail))
        bad += 0 if ok else 1
    print()
    print("{} of {} checks failed".format(bad, len(cases)))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
