# `tests/extensions/`

One directory per optional capability, the arrangement ACEsuit/mace uses
(`tests/extensions/`, `.github/workflows/ci-extensions.yaml`). The contract is
`openqha/capabilities.py`:

* locally, a test whose capability is unavailable **skips**;
* a run that **declares** the capability fails instead:

```bash
S0_REQUIRE_CAPS=gromacs python tests/run_tests.py
```

That second line is the point. Acceptance criterion 2 twice reported "no comparison
produced" — once because GROMACS was installed in a sibling conda environment, once
because `--no-gmx` was passed — and both times a criterion that exists to catch an error
in the covariance spectrum went quiet instead of red. A job that guarantees a capability
must never again go green while skipping the thing it exists to test.

Adding an extension means three things, exactly as upstream: a module under
`openqha/extensions/`, a probe in `openqha/capabilities.py`, and a directory here.

| extension | what it checks | what its absence costs |
|---|---|---|
| `gromacs` | `gmx covar -mwa` against `openqha/qha.py` — an independent covariance spectrum from a program sharing no Python with us | nothing numerical. MDAnalysis is the routine check and agreed to **8.020e-09 kcal/mol**, where GROMACS agreed to **1.27e-04** |
