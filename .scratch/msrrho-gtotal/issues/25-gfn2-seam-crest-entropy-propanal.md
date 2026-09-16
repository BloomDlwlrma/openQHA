# 25: GFN2 seam: crest --entropy on propanal, term by term

**What to build:** a CREST `--entropy` Calculation at GFN2 (environment `s0crest`, run at least twice because the sampling is stochastic) in the crest engine folder with the `entropy` setting stem, and a reader that takes CREST's conformers, `cre_degen2` and per-conformer frequencies and feeds them to our assembly. The Calculation leaves `levels/gfn2/thermo_msrrho.out` and `.toml` in the same shape as ticket 24's, with `[Calculation_Info]` naming CREST's version, `sthr`, `ithr`, `fscal` and the number of runs, and a `[Crest]` block holding CREST's own S'_conf, dS_bar, Cp_conf, H_conf, S_abs and their run-to-run spread. GFN2 then appears as a level in `level_compare`; it is never a reference for MACE.

**Blocked by:** 24 (the assembly and the record shape).

**Status:** done 2026-09-16

- [x] on propanal, our S'_conf, Cp_conf and H_conf from CREST's own inputs equal CREST's printout to 1e-4 cal/mol/K (measured 1e-6, both runs). dS_bar and S_ref are Hessian-dependent and CREST leaves no per-conformer Hessian on disk: S_ref agrees to +0.011 (xtb analytic vs CREST numerical Hessian at the same geometry), dS_bar to +0.10/+0.12; recorded as a separate tier, not forced to 1e-4
- [x] our g_rot * cores equals `cre_degen2` (6, 3, 3) for every propanal conformer; CREST's g' itself is enantiofac (the mirror flag propagated by point-group label), reproduced per run from its `symmetries` file, and it differs between the two runs (c1/cs flip on the third conformer)
- [x] two runs are recorded and their spread in S'_conf is printed; a single run is refused
- [x] `levels/gfn2/` holds exactly the two record stems; CREST's engine files stay in the crest engine folder
- [x] the integration test runs with a stand-in CREST that writes a fixed ensemble, `cre_degen2` and frequencies, so the reader and the assembly are exercised without xtb
