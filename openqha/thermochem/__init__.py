"""Thermochemistry shared by every branch: Hessians, partition functions, the state points a result is only valid at.

Modules: critical, hessian, thermo

Deliberately thin, as in ALF and MACE: importing this subpackage must not
import its modules. openQHA loads no torch, ase, rdkit or openmm at import
time, and `openqha.capabilities` is what reports which of them are present.
"""
