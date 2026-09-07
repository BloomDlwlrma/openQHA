"""Where results live and how they are found again: the sharded basin store, the product record, the report, and what is left to compute.

Modules: artifacts, basin_store, record, report, worklist

Deliberately thin, as in ALF and MACE: importing this subpackage must not
import its modules. openQHA loads no torch, ase, rdkit or openmm at import
time, and `openqha.capabilities` is what reports which of them are present.
"""
