# 05: Re-run collect and the ensemble on the propanal sample, both routes

**What to build:** the shipped sample records are what the code now writes. Collect and the ensemble are re-run on the propanal engine files already on disk (no MD is redone) for the OpenMM and the ASE route; the eight old `.dat` under `examples/02b_qha_openmm_propanal/sample_records/` are deleted and the two `collect.dat` plus the refreshed `collect.out`, `collect.toml`, `ensemble.out`, `ensemble.toml` take their place.

**Blocked by:** 04 (End to end on the ASE chain, and the docs name one Table).

**Status:** done 2026-09-16

- [x] `sample_records/openmm/md_openmm/` and `sample_records/ase/md_ase/` each hold `collect.dat` and no other `.dat`
- [x] the numbers in the new `collect.dat` equal those of the old four tables (same engine files, same code path for the analysis)
- [x] the example README's tree matches the folder
