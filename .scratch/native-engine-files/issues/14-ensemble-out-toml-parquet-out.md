# 14: Ensemble as ensemble.out + ensemble.toml; parquet leaves the chain

**What to build:** the ensemble report writes `ensemble.toml` (the answer for programs) and `ensemble.out` (per atoms-set sections, the per-basin table, the complete record; last line `openQHA ensemble terminated normally`); 02d writes `02d_frequency_identity.toml` + `.out` the same way. pandas and pyarrow are no longer required by any chain step: the preflight in collect and the `openqha_require_modules pandas pyarrow` in chain_body are gone, the parquet writers in `report.py` stay for use outside the chain.

**Blocked by:** 13.

**Status:** done 2026-09-15

- [x] `ensemble.json` and `02d_frequency_identity.json` are no longer written; `.toml` + `.out` are
- [x] no chain step imports pandas or pyarrow; t_parquet_preflight retired; check_dependency no longer lists them as required
- [x] full suite green (30/30 with `--all`)
