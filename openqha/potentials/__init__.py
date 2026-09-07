"""The potential and everything that carries it: the registry, the neighbour-list patch, the resident server. ALF calls this ml_interfaces.

Modules: engine, mace_patch, mace_server

Deliberately thin, as in ALF and MACE: importing this subpackage must not
import its modules. openQHA loads no torch, ase, rdkit or openmm at import
time, and `openqha.capabilities` is what reports which of them are present.
"""
