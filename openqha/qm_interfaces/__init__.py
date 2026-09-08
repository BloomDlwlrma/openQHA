"""Reference-level quantum chemistry. ALF calls this qm_interfaces.

Modules: orca, xtb

Deliberately thin, as in ALF and MACE: importing this subpackage must not
import its modules. openQHA loads no torch, ase, rdkit or openmm at import
time, and `openqha.capabilities` is what reports which of them are present.
"""
