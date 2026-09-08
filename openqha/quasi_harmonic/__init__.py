"""Branch B: quasi-harmonic analysis. Trajectories, covariance, spectra.

Modules: basin_residence, ensemble, gmx_io, mdtraj_io, mode_match, openmm_mace,
         perturb, qha, torsion_cv, vdos

Deliberately thin, as in ALF and MACE: importing this subpackage must not
import its modules. openQHA loads no torch, ase, rdkit or openmm at import
time, and `openqha.capabilities` is what reports which of them are present.
"""
