"""Optional capabilities. Absence changes no number.

The arrangement is ACEsuit/mace's (`mace/modules/extensions.py`, `tests/extensions/`,
`.github/workflows/ci-extensions.yaml`), and so is the contract, which is the part that
matters:

  * locally, work whose capability is unavailable is SKIPPED;
  * a run that DECLARES the capability -- `S0_REQUIRE_CAPS=gromacs` -- FAILS instead.

A job can never again go green while silently skipping the thing it exists to test. That
is not a hypothetical here: acceptance criterion 2 reported "no comparison produced" twice
for reasons it does not measure, once because GROMACS was installed in a sibling conda
environment.

Adding an extension means three things, exactly as upstream: a module here, a probe in
`openqha/capabilities.py`, and a directory under `tests/extensions/<name>/`.

    gromacs   an independent covariance spectrum from a program that shares no Python
              with openQHA. Optional since MDAnalysis was measured at 8.0e-09 kcal/mol
              against our own superposition, where GROMACS measured 1.27e-04.
"""
