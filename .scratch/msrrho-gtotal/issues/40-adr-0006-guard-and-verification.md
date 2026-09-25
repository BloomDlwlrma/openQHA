# 40: ADR 0006, the campaign guard, and the verified suite

**What to build:** the ruling is written down where the next reader will find it, and the change set is verified end to end. ADR 0006 records: one production imaginary-mode policy (`invert_below` with the frequency floor, `ithr = -50 cm⁻¹` from CREST's `-ithr`); `refuse` removed; the sub-1 cm⁻¹ drop with mandatory recording and the three-or-more tripwire; the census floor and the inversion window; the convergence certification with the single second pass; the reference-level retry — with the provenance table (-50 / 1 / 25 / ORCA's default TolMaxG). The Hessian-learning workflow notes gain the D1 guard sentence: labels are computed at the frame's own geometry and are never optimised; training rows are stationary basin frames, the off-stationary generators are held out (Rodriguez 2025; PHL's NMS rationale). Ticket 28 and the feature spec's stale sentences are updated to point at this ticket set (ticket 34 itself is left untouched). The word rule is applied to the touched docs: "mode" only for a vibration; "policy" and "rule" for settings. Then the suite runs: full unit tests, the one MACE integration test, and a fixture-molecule end-to-end (census record → basin record → thermochemistry record → reference fixture), with the results recorded here.

**Blocked by:** 35, 36, 37, 38, 39

**Status:** done 2026-09-25 (the ruling is written down where the next reader will find it; the guard sentence landed; ticket 28 and the spec's stale sentences point at this ticket set; the change set is verified — unit 63/63, the MACE integration test pass, the new forcing second-pass test pass, the full suite `--all`, and the propanal fixture walked census record -> basin record -> thermochemistry record -> reference fixture through the production steps). One renumbering: the ADR is **0007**, not 0006 — 0006 is the PHL-verbatim ADR of 2026-09-23.

- [x] ADR accepted, with the provenance table and the two dated principles (interpret frequencies only at a same-method stationary point; always project and never compute thermochemistry with projection off) — written as `docs/adr/0007-one-imaginary-mode-policy-floor-and-the-two-principles.md`; the renumbering from this ticket's text is noted in the ADR and in the notes below.
- [x] Campaign guard sentence landed (`workflows/hessian_learning/README.md`, Step 03); ticket 28 and the spec's stale lines point at the new tickets.
- [x] Full unit suite green; the MACE integration test green; the fixture end-to-end recorded with its outputs in the ticket.
- [x] The word rule applied in the touched docs ("mode" only for a vibration; "policy" and "rule" for settings).

## Notes (implementer, 2026-09-25)

**The ADR is 0007.** Ticket 34 (and this ticket) planned the ruling as ADR 0006; that
number was taken by `0006-phl-verbatim-cartesian-target.md` (2026-09-23). The ruling is
`docs/adr/0007-one-imaginary-mode-policy-floor-and-the-two-principles.md`: the two dated
principles (D1 -- interpret frequencies only at a stationary point of the same method, and
its scope guard towards the Hessian-learning labels; D2 -- always project, never compute
thermochemistry with projection off), one production policy `invert_below` on
`ithr = -50 cm^-1`, `refuse` removed, the sub-1 cm^-1 drop with recording and the
three-or-more tripwire, the census floor and inversion window, the tighten certification
with the single second pass, the reference-level retry, and the provenance table
(-50 / 1 / 25 / ORCA's `TolMaxG = 1.543e-2 eV/A`). Ticket 34's text is left as written,
per this ticket's instruction.

**The guard sentence** landed in `workflows/hessian_learning/README.md`, Step 03, as a
marked block ("The D1 guard (2026-09-25): never optimise a frame before labelling it"):
labels are computed at the frame's own geometry and are never optimised; training rows are
stationary basin frames, the off-stationary generators (`displaced`, `merged`, `saddle`)
held out (ADR 0005; Rodriguez 2025; PHL's NMS rationale). It states D1 as a rule about
interpreting a frequency — exactly what must NOT be applied to a curvature label at a
fixed geometry.

**Ticket 28 / the spec.** Ticket 28's Status now points at the ticket set 34-40 and ADR
0007 (the four-molecule spread table was closed as superseded; it was never built). `spec.md` gained a
dated ruling note at the top and dated notes on every stale sentence: story 4, story 24,
the spectrum bullet (sub-1 as drop-with-recording + tripwire), the imaginary-mode policy
bullet, the "policies on real data" bullet, the imaginary-policy testing seam, and the
Round-4 ticket list. Nothing outside the imaginary-mode regime was touched (e.g. the
`levels/` spelling is ADR 0004's material, not this ruling's).

**Ticket 38's unverified item is closed** by the new forcing test, per the user's ruling
of 2026-09-25 on the handoff (section 3): a real-calculator census whose first tighten
stops above ORCA's line, so the second pass's engine-file append is exercised end to end
at last.

## The verification runs

(the torch/e3nn environment warnings are trimmed from the transcripts)

### 1. Full unit suite

```
$ python tests/run_tests.py
...
all 63 test(s) passed
```

### 2. The one MACE integration test (sections D/E are tickets 37's and 38's)

```
$ python tests/integration/t_mace_engine_folder.py
A. one folder per tightened conformer, three files each
  two conformer folders (conf00, conf01)                         ok
  conf00 holds exactly opt.traj opt.log conf.extxyz              ok
  conf01 holds exactly opt.traj opt.log conf.extxyz              ok
  conf.extxyz carries energy and forces                          ok
  opt.traj holds the optimisation steps (22 frames)              ok
B. one folder per surviving basin, two files each
  one basin (the two conformers deduplicate)                     ok
  record agrees: n_basins == 1                                   ok
  basin00 holds exactly basin.extxyz hessian.npy                 ok
  basin.extxyz names its conformer index                         ok
  basin.extxyz carries CREST's comment for that conformer        ok
  hessian.npy is 3N x 3N float64                                 ok
  hessian.npy is symmetric (max asym 0.0e+00)                    ok
  raw (heavy-atom diagonal blocks exceed hydrogen ones)          ok
C. nothing else under mace/
  no other entries                                               ok
D. the census screen used the frequency floor (ticket 37)
  the record names the floor it applied                          ok
  the basin carries its verdict, lowest mode and window count    ok
E. the census certified the tighten (ticket 38)
  the summary counts every class and rejects none                ok
  the basin's class and tighten_converged answer to the two lines ok

PASS
```

### 3. The forcing second-pass test (new; closes ticket 38's item)

```
$ python tests/integration/t_census_second_pass.py
A. the second pass fired; the record names what failed twice
  the rejected frame folded 2 steps (one per pass); the certified frame 0-1 ok
  one certified basin, one not_certified, nothing in the other classes ok
  the rejected frame is conformer 1 with its FINAL residual, above ORCA's line ok
  the certified basin is the only one that survives, with its class recorded ok
B. both runs wrote the same opt.log
  two 'Step ... fmax' headers -- one optimisation per pass         ok
  LBFGS step numbers 0, n, 0, n -- the second pass ran its own steps ok
C. the trajectory kept the first pass (append, not truncation)
  frames == n1 + n2 + 1: the first pass's initial state and both steps survive ok
  the file's FIRST frame is the input geometry (pass 1's start)    ok
  the LAST frame is conf.extxyz, to the writer's precision         ok
  conf.extxyz carries the residual the record rejected the frame with ok
D. no new engine-file name was born
  conf01 holds exactly opt.traj opt.log conf.extxyz                ok
  conf00 (no second pass) holds exactly the same three files       ok
  no opt2.* anywhere under mace/                                   ok

PASS
```

### 4. The fixture molecule end to end (census record -> basin record -> thermochemistry record -> reference fixture)

The census/basin record is the fixture's own `_records/branchA.toml` (`[Census]` block and
`[[Basin]]` rows); the thermochemistry and reference records are written fresh into a temp
copy, and the reference step reuses the fixture's finished ORCA jobs (0 s each, analytic
route — no ORCA binary needed):

```
$ cp -r tests/data/propanal_molecule /tmp/t40_e2e/propanal_e2e/dsgdb9nsd_000035
$ S0_RUNS_ROOT=/tmp/t40_e2e python scripts/production/s0_thermo_msrrho.py --species dsgdb9nsd_000035 --tag propanal_e2e --step mace
level mace-off23_medium: S_abs = 72.355 cal/mol/K  G_total = -121244.0834 kcal/mol  basins 3 (excluded 0)
experiment 72.75 [li2016lbh]: S_abs - experiment = -0.395
  basin  0: lowest    128.45 cm^-1  inverted 0  dropped |omega|<1 cm^-1 0
  basin  1: lowest     73.96 cm^-1  inverted 0  dropped |omega|<1 cm^-1 0
  basin  2: lowest     74.00 cm^-1  inverted 0  dropped |omega|<1 cm^-1 0
record  /tmp/t40_e2e/propanal_e2e/dsgdb9nsd_000035/msrrho/thermo/mace-off23_medium.thermo_msrrho.toml
$ S0_RUNS_ROOT=/tmp/t40_e2e python scripts/production/s0_thermo_msrrho.py --species dsgdb9nsd_000035 --tag propanal_e2e --step reference
level wb97m-d3bj_def2-tzvppd: S_abs = 72.193 cal/mol/K  G_total = -121244.2954 kcal/mol  reference basins 3 (excluded 0)  Hessian route analytic
  MACE basin 0 -> kept 0  shift 0.002 A  imaginary 0  0 s
  MACE basin 1 -> kept 2  shift 0.032 A  imaginary 0  0 s
  MACE basin 2 -> kept 1  shift 0.032 A  imaginary 0  0 s
record  /tmp/t40_e2e/propanal_e2e/dsgdb9nsd_000035/msrrho/thermo/wb97m-d3bj_def2-tzvppd.thermo_msrrho.toml
$ S0_RUNS_ROOT=/tmp/t40_e2e python scripts/production/s0_thermo_msrrho.py --species dsgdb9nsd_000035 --tag propanal_e2e --step compare
  mace-off23_medium            present True  S_abs 72.355
  wb97m-d3bj_def2-tzvppd       present True  S_abs 72.193
  MODEL_ERROR_S          +0.1624
  MODEL_ERROR_S_CONF_PRIME +0.1584
  MODEL_ERROR_S_REF      -0.0203
  S_EXPERIMENT           +72.7500
  S_EXPERIMENT_SOURCE    li2016lbh
  LEVEL_ERROR_S          -0.5572
  TOTAL_ERROR_S          -0.3949
record  /tmp/t40_e2e/propanal_e2e/dsgdb9nsd_000035/msrrho/thermo/level_compare.toml
```

The numbers are the ones the closing report recorded for propanal (S_abs 72.355 / 72.193;
model error +0.162, of which S'_conf +0.158; level error -0.557), now produced by the
current code from the fixture's records and engine files alone.

### 5. `python tests/run_tests.py --all`

```
$ python tests/run_tests.py --all
  ...
  pass integration t_census_second_pass.py                        8.45s
  pass integration t_mace_engine_folder.py                       13.48s
  ...
all 78 test(s) passed
```

(78 = 63 unit + 13 integration + 2 regression; the new `t_census_second_pass.py` runs in
8.5 s inside the group as well. The review's row-11 change — `h0` indexed by the record's
own conformer id — came after this transcript; the test was re-run standalone after it:
`py_compile` clean and PASS, the check set identical.)

## Review record (2026-09-25): the two-axis review of this change set — closed 2026-09-25: accepted, default keep

The two-axis review (Standards / Spec) of the working-tree change set raised the findings
below. The acceptance criteria are paid; **the user accepted every row below on 2026-09-25
(default keep)** — the fixed rows stand as fixed, the judgement calls stand as committed,
and a revert of any row is mechanical at the location its "where" column names (tickets
37/38's records are the pattern).

| # | finding | where | basis | disposition |
|---|---------|-------|-------|-------------|
| 1 | the Status claimed `--all` while its transcript was still a placeholder | this ticket | AGENTS.md: "a ticket is done only when its acceptance criteria are paid and its Status line carries the evidence" | **fixed before the commit** (section 5 now carries the 78/78 transcript) |
| 2 | "14 checks" for the forcing test — it runs 13 `check()` calls | ticket 38's Status | miscount (the transcript prints 13 `ok` lines) | **fixed** (13, in ticket 38 and here) |
| 3 | the spec header's "Decisions" line did not name ADR 0007 while the new ruling note does | `spec.md` lines 5-6 | the header is where a reader finds the decisions | **fixed** (the line now lists `docs/adr/0007`) |
| 4 | the ruling blockquote said the stale sentences "carry a dated note", but three were replaced outright | `spec.md` ruling note | accuracy of the note's own claim | **fixed wording** ("replaced or annotated") |
| 5 | the ADR's certification classes did not name `converged_second_pass` while its Consequences say "the four class counts" | ADR 0007, certification paragraph | ticket 40 asked the ADR to record the certification with the single second pass | **fixed** (the success class is named) |
| 6 | "the three-policy spread was never built" overstates: the per-record `[Imaginary_Spread]` block was built and removed; only the four-molecule spread table was never built | `spec.md` story 24 and the real-data bullet | ticket 34 ("every record carried a three-policy spread") | **fixed** (both sentences now say the table) |
| 7 | the new test duplicates `_repo_root()` / `FAIL` / `check()` (`tests/_testlib.repo_root` exists) | `t_census_second_pass.py` | DRY vs the standalone-program test convention | **keep** — identical to `t_mace_engine_folder.py` and `t_census_convergence.py`; tests/README makes each test a standalone program |
| 8 | the test's `except Exception -> SKIP` wraps `engine.calculator`, so a real engine regression skips green | `t_census_second_pass.py` | tests/README: "nothing distinguished 'checked and agreed' from 'did not check'" | **keep** — the same pattern as the existing integration test; MACE absence is the intended skip, and the group now fails on a broken engine elsewhere |
| 9 | the D1 guard block restates the sentence above it; the ADR repeats the D1-scope exemption in the bullet and Consequences | README Step 03; ADR 0007 | duplication | **keep** — the guard is the sentence a maintainer must read where labels are made; the ADR repeats it because the Consequences are the entry point |
| 10 | the spec's top ruling banner goes slightly beyond "the stale sentences" | `spec.md` ruling note | ticket 40's wording | **keep** — declared in the notes above; one deletion reverts it if the user prefers |
| 11 | `h0 = rec["hessian"]["0"]` hard-coded the conformer id the same section asserts | `t_census_second_pass.py` | prior art indexes by the record's own id (`t_mace_engine_folder.py` E) | **changed** to `rec["hessian"][str(rec["basin_conformer_ids"][0])]` |
