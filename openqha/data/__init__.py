"""Locating input structures: QM9, curatedQM9, and the uncharacterized list.

Modules: curated_qm9, qm9_uncharacterized

Deliberately thin, as in ALF and MACE: importing this subpackage must not
import its modules. openQHA loads no torch, ase, rdkit or openmm at import
time, and `openqha.capabilities` is what reports which of them are present.
"""
