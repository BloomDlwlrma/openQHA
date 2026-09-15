---
status: accepted
date: 2026-09-15
---

# Records are text reports, TOML and whitespace tables, never JSON or parquet

The files this repository writes about a run (the branch A record, the trajectory
record, collect's tables, the ensemble answer) were JSON and parquet. The user ruled that
they take the form the engines themselves use -- ORCA's `.out` and `.property.txt`,
CREST's `crest.out`, `input.toml`, `crest.energies`: one `.out` report per step whose last
line `openQHA <step> terminated normally` is the completion marker, TOML for what a
program reads back (the standard library reads it; `openqha.store.toml_out` writes it),
and whitespace `.dat` tables with one header line for collect's numbers.

## Considered options

- Keep JSON and add a `.out` beside it: rejected, the objection was to the JSON.
- ORCA-style `$block` key/value text instead of TOML: rejected, it needs a parser of our
  own where TOML needs none.

## Consequences

pandas and pyarrow are no longer required by any chain step (the parquet-engine preflight
of 2026-09-13 is gone with the tables). TOML has no null, so a None field is absent and
every reader uses `.get`. A report cut short by a kill has no terminal line and the step
is redone, which is the safe direction.

## Amendment (2026-09-15): property style, and nothing for a Batch

The first form of the TOML record carried everything the JSON had carried (about 330
keys for a three-basin molecule, a tenth of it read by any later step, a sixth of it
written twice). The user ruled that the `.toml` takes the shape of ORCA's
`.property.txt`: a `[Calculation_Status]` block with `STATUS = "NORMAL TERMINATION"`, a
`[Calculation_Info]` block with the inputs, and only the result blocks a later step reads;
prose, provenance and diagnostics are printed in the `.out` alone. Two words were fixed in
CONTEXT.md to make the unit clear: a Calculation (one step on one molecule or basin) owns
a Record; a Batch (one driver over many Calculations) owns none, its Slurm log lists every
Calculation's return code, STATUS and record path. The three JSON batch summaries are
gone, and so is the `<setting>` level under `_records` (the setting is in the file stem,
as for engine files). Spec: `.scratch/records-redesign/spec.md`.
