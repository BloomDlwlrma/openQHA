"""Moved to `openqha.extensions.gromacs`. This module re-exports it.

GROMACS became an EXTENSION on 2026-09-05: branch B's number no longer depends on
it, because MDAnalysis provides a mass-weighted independent superposition that
agrees to 8.0e-09 kcal/mol and needs no external binary. See
`openqha/extensions/gromacs.py` for the implementation and
`openqha/capabilities.py` for what 'extension' means here.

This shim is kept rather than deleted for two reasons. Anything outside this
repository that imports `openqha.gmx_io` keeps working. And the docstring below is
where four measurements live -- the 452x `gmx anaeig -entropy` defect, the
`-[no]mwa (no)` default that silently returns a wrong number, the `.gro`
quantisation against a 0.03 angstrom amplitude, and the .g96 1e-8 angstrom
resolution -- and a move that dropped them would be the deletion
`tests/unit/t_translation_preserved_numbers.py` was written to catch (retired 2026-09-09 with the migration it guarded).

Original module documentation, preserved verbatim:

GROMACS as the independent implementation of the covariance spectrum (branch B).

Why GROMACS is here, and what it is not
---------------------------------------
Stage 0's potential is MACE-OFF23_medium and its trajectories are driven by ASE.
**GROMACS never evaluates a single force in this repo.** It appears in exactly one
place: given the same trajectory, `gmx covar -mwa` produces an INDEPENDENT mass-weighted
covariance spectrum, and the difference against `openqha/qha.py` is branch B acceptance
criterion 2.

The `.tpr` is therefore only a carrier of atom names and MASSES -- `-mwa` needs masses
and `-fit` needs a reference structure. Bonded and non-bonded parameters in the topology
written here are never evaluated. This has to be said in the product as well, or a reader
will assume the thermodynamics sits on some classical force field, which is exactly the
reading (plan_B supplement 1.3, reading B) that would dismantle the level consistency of
D0-4.

Which of the two is the reference -- and the measurement that changed the answer
-------------------------------------------------------------------------------
Revision C of plan_B (section 2.6.4) made GROMACS the reference and `openqha/qha.py` the
cross-check, on the sound argument that a widely used implementation should not be
checked against one written this week.

**That arrangement does not survive contact with the tool.** `gmx anaeig -entropy`,
measured on the installed binary on 2026-09-03, refuses mass-weighted eigenvalues, uses
a formula that expects them anyway, and drops the six softest modes instead of the six
rigid ones -- a factor of 452 on a synthetic case whose answer is known in closed form.
The three findings, the upstream source lines behind them and the reproduction are in
`anaeig`'s docstring, and the tool is still run, with its output recorded as evidence and
flagged `is_reference=False`.

So the independence is split rather than lost:

  * the SPECTRUM (superposition, mass weighting, diagonalisation -- the part that could
    plausibly be wrong) is checked against `gmx covar -mwa`, which is independent and
    works;
  * the ENTROPY SUM on top of it is closed form, and `qha.harmonic_limit_check` verifies
    it against the analytic answer to machine precision (criterion 3).

No single command covers the whole chain any more. Every step still has an independent
check, and the report says which is which.

The switch that silently changes the answer
-------------------------------------------
`gmx covar -mwa` defaults to **no**. Confirmed on the installed binary (GROMACS
2026.3-conda_forge): `-[no]mwa (no)`. Every step of the quasi-harmonic derivation stands
on the MASS-WEIGHTED covariance, and without `-mwa` the tool still returns a number, with
no error and no warning, and that number is wrong. `covar()` below therefore has no way
to turn mass weighting off; `covar_plain()` exists only to feed the defective `anaeig`
the unweighted input it demands, and no repo number is computed from its output.

Trajectory format
-----------------
Frames are written as GROMOS-96 `.g96`, which GROMACS reads as a trajectory and which
stores coordinates as %15.9f in nm, i.e. 1e-8 angstrom. `.gro` was rejected: it stores 3
decimals in nm = 0.01 angstrom, and the mass-weighted amplitude of a C-H stretch at 300 K
is about 0.03 angstrom, so `.gro` quantisation would land squarely on the quantity being
measured.
"""
from ..extensions.gromacs import *            # noqa: F401,F403
from ..extensions.gromacs import (            # noqa: F401
    BOX_PADDING_NM, NEVSKIP_ROT_TRANS, box_and_shift, covar, covar_plain,
    cross_check, gmx_binary, grompp, parse_entropy, parse_xvg, run_gmx,
    superimpose_once, version, write_g96_trajectory, write_gro,
    write_grompp_mdp, write_top,
)
