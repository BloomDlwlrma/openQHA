# `tests/`

```
python tests/run_tests.py                 # unit only — seconds, no engine
python tests/run_tests.py --all           # every group
python tests/run_tests.py --group regression
```

Each test is a standalone program returning an exit code, and the runner launches each in
its own process. That is deliberate: `regression/t_defect46_shebang.py` exists precisely
because a script behaved differently when **executed** than when imported (`D0-79`), and
importing tests into one process would erase the distinction part of this suite is about.

| Directory | What lives here |
|---|---|
| `unit/` | pure functions and file-level checks; seconds, no engine, no network |
| `integration/` | needs an engine, CREST or ORCA; slow |
| `regression/` | reproduces one numbered defect; the file name carries the number |
| `data/` | inputs and expected outputs for the regression tests, paired in the style of MSTor's `testrun/` and `testo/` |

`_testlib.py` holds the two helpers the repository-hygiene checks share. It is not a test.

## Where this suite came from

`tests/` held 228 lines (1.6% coverage) while `scripts/` held 10 026. That was never
because checks had not been written — **a great many had been written, and were sitting in
`scripts/` as one-off probes that ran once and were never run again.** plan_D section 2.3
is the list of those worth recovering as permanent regression tests; each has a specific
input and a definite expected output already recorded in a checkpoint.

## The checks that guard the restructure itself

These are what make branch D's acceptance criteria executable rather than aspirational:

| Test | The criterion it enforces |
|---|---|
| `unit/t_translation_preserved_numbers.py` | translating and moving files loses **no number** — compared against the pre-restructure backup |
| `unit/t_code_is_english.py` | live code carries no CJK characters; retired code is counted, not silently exempted |
| `unit/t_script_taxonomy.py` | every script declares its category, and the declaration matches its directory |
| `unit/t_repo_bootstrap.py` | nothing reaches the repository root by counting directory levels, or by the working directory |
| `unit/t_superseded_isolated.py` | nothing live imports or reads anything retired |
| `unit/t_artifacts_identity.py` | `write_artifact` genuinely refuses an artifact with no identity |

Three of these had to be corrected after their first run reported false positives on their
own aperture — a test that scans for a forbidden pattern will find that pattern in its own
source, and in any docstring explaining why the pattern was removed. Each carries a
comment recording what it got wrong, because the next such check will meet the same trap.
