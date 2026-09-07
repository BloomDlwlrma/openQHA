"""Branch A: conformational search. CREST iMTD-GC, deduplication, basins.

Modules: conformers, crest, crest_census, filters, refine_analysis, symmetry

Deliberately thin, as in ALF and MACE: importing this subpackage must not
import its modules. openQHA loads no torch, ase, rdkit or openmm at import
time, and `openqha.capabilities` is what reports which of them are present.
"""
