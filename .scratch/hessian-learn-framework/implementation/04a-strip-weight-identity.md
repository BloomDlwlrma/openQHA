# 04a: The weight-identity strip — engine, records, tooling

Type: task
Status: resolved
Serves: 04
Blocked by: 04
Part of: [hessian-learn-framework](../map.md)

> Execution slice for [SHA256 retirement](../decisions/04-sha256-retirement.md) — the
> openQHA-side half of the 2026-09-27 grilling (minimal strip; no `mace_off` delegation;
> fingerprint and pin fully retired; `check_weights_are_physical` kept). The training-side
> half (run/judge callers, registration output, the training slurm) is carried by
> [The move](../decisions/07-the-move.md); the reference sweep by
> [References sweep](../decisions/12-references-sweep.md). Wayfinder conventions apply:
> the `Status` protocol, no triage labels (`docs/agents/issue-tracker.md`).

## Problem Statement

Loading a potential currently carries a bespoke identity layer that does nothing for the
load: every calculator construction computes a state-dict SHA-256 fingerprint (a second
full read of the model file), every Record that names a potential writes one or more
`*_PARAMS_SHA256` fields plus a pin-status line, and the registry pin — when it exists —
only prints a warning when the numbers differ. mace's own load path hashes nothing; the
effort's destination is "weights load the native mace way". Meanwhile the operator still
needs Records to say *which* weights produced a number, and old Records must stay exactly
as they are.

## Solution

Remove the identity layer from the load path and from the Records it touched, and put the
identity back where the grilling put it: the registered engine name plus the resolved
weight file. No fingerprint, no pin, no pin status, no per-frame weight hash. `ENGINE`
and a new resolved-path field identify the weights in a frames Record; each frame carries
the engine name instead of a hash; the dataset Record lists the engines seen; the weights
tool shrinks to "which mace is imported + what weight files are present". Old Records are
untouched; new writers tolerate their absence. `check_weights_are_physical` and the two
path overrides stay exactly as they are.

## User Stories

1. As a campaign operator, I want loading a weight file to be exactly the native mace
   call with nothing of ours inspecting or comparing its bytes, so that no file mace
   would load can be refused by this repository.
2. As a campaign operator, I want the fingerprint's second full model read gone from
   calculator construction, so that every batch process starts without paying it.
3. As a campaign operator, I want a corrupted weight file (the 2026-09-10 class: right
   size, right tensor count, all-finite, one 2.084e+306 tensor) still refused, so the
   NaN class cannot silently recur.
4. As a campaign operator, I want a missing weight file to keep the guidance error
   (expected path, what the directory holds, both overrides), so a mis-copied file on a
   cluster is diagnosed in one look.
5. As a campaign operator, I want `S0_MACE_ROOT` and `S0_MACE_MODEL` to behave exactly
   as before, so no cluster script changes.
6. As a campaign operator, I want the registry to keep `source` and `note`, so every
   Record still says what to cite and why the engine exists.
7. As a reader of a frames Record, I want the resolved weight file path in the Record,
   so a name that lies (an `S0_MACE_MODEL` override) is visible in the Record itself.
8. As a reader of frames, I want each frame to carry the registered engine name, so a
   merged dataset can still show whether one or several weights produced it.
9. As a reader of a dataset Record, I want the set of engines seen, so the merge states
   its sources without a hash.
10. As a reader of an MD Record, I want its provenance section to keep the path, size
    and versions, so an old trajectory stays traceable without the hash lines.
11. As a reader of old Records, I want them byte-identical and still readable, so earlier
    campaign numbers stay citable and nothing needs migrating.
12. As the person registering a fine-tuned potential, I want a registry filename to be
    allowed a relative sub-path, so campaign revisions can live grouped
    (`mace_off23_<campaign>/`) while the base models stay flat.
13. As the person diagnosing a failed load, I want the missing-file message to show
    sub-directory names too, so "wrong directory", "wrong sub-directory" and "wrong
    filename" are distinguishable.
14. As a maintainer, I want the reduced weights tool to print only which mace is imported
    and what files are present, so the tooling reports facts and nothing else.
15. As a maintainer, I want tests to lock the new contract (retired keys absent; engine
    name and weights-file path present), so the machinery cannot creep back silently.
16. As the future publisher of fine-tuned models, I want the load path free of hashing,
    so publication-time checksums (a later ticket) are a publisher-side concern only.

## Implementation Decisions

- **The engine registry module** (`openqha.potentials.engine`): delete
  `fingerprint_state_dict`, `parameter_fingerprint`, their cache, the optional
  `params_sha256` registry field, the pin comparison and its stderr WARN, and the four
  provenance keys `params_sha256` / `n_tensors` / `params_bytes` / `params_pin_status`
  (the file-size `bytes` stays). `provenance()` keeps engine / source / note /
  weights_path / bytes / interface / mace & fork identity / dtype / patch state. Its
  docstring states the identity: engine name + resolved path (+ fork commit, dtype).
- **`check_weights_are_physical` and `S0_SKIP_WEIGHT_CHECK` are unchanged**, still called
  at calculator construction.
- **`model_path()`** keeps its resolution (root/filename, both overrides). Because a
  registry `filename` may now be a relative sub-path, the missing-file diagnostics list
  the root's top-level `*.model` files **and** its sub-directory names, keeping the
  "expected path first, then what is actually there" split. No auto-latest, no pattern
  resolution: an entry resolves to exactly one concrete file (a glob handed to
  `MACECalculator` would silently make a committee).
- **Storage convention documented in the module comments** (not a search rule): base
  models stay flat in the root; self-trained revisions live under
  `mace_off23_<campaign>/` as `<run>+<YYYYMMDD-HHMMSS>.model`. The registration-side
  grammar (registry key, `--register-copy` creating the sub-directory) lands in ticket 07.
- **Frames Record** (`openqha.data.frames`): drop `ENGINE_PARAMS_SHA256` and
  `ENGINE_PIN_STATUS` from the schema and the writes; add `WEIGHTS_FILE` = the resolved
  weight file path (from the provenance the build already holds), `-` when the calculator
  was injected. *(The grilling approved this field as `ENGINE_FILE`; renamed here because
  CONTEXT's **Engine file** vocabulary and the dataset index's existing `engine_file`
  (the molecule's engine-level xyz) make that name ambiguous — say the word to keep the
  old name.)*
- **Frames per-frame info key**: `engine` = registered engine name replaces
  `engine_params_sha256`. Old frame files keep their old attribute (they are data); all
  readers use `.get` with `-`.
- **Dataset builder** (`openqha.data.dataset`): the index column `engine` replaces
  `engine_params_sha256`; the dataset Record carries `ENGINES` (sorted unique engine
  names) in place of `ENGINE_PARAMS_SHA256`. The existing `engine_file` index field (the
  molecule's engine-level file) is untouched.
- **Branch-A property mapper** (`openqha.store.branch_a_property`): drop both fields
  from schema and mapping.
- **MD Record provenance print** (`openqha.quasi_harmonic.md_record`): drop
  `params_sha256` / `n_tensors` / `params_pin_status`; keep `weights_path` / `bytes` /
  `interface` / versions.
- **Tooling scripts** (`s0_hl_smoke_fit`, `s0_probe_calibration`): drop the field writes
  and prints; their schema and key lists follow.
- **`s0_check_weights.py` reduced form**: keep `print_mace()` exactly (mace version,
  module path, fork commit and dirty flag); the default output lists every registered
  engine whose file resolves (name, path, bytes). Delete `--pin`, `--json`, `--compare`,
  `file_sha256`, and every fingerprint call; the docstring states what the tool is now.
- **Schema policy**: retired keys are removed from the schemas (not kept with `None`);
  no migration, no backfill; readers stay tolerant. Old Records are byte-identical.
- **Deliberately absent**: any replacement hash on the load path, in Records, or in
  tests; no transplant of upstream's golden-hash practice; publication checksums are a
  publisher-side concern (ticket 13, blocked by 09).

## Testing Decisions

- A good test here asserts **external behaviour**: a Record's content, a resolver's
  output, a CLI's stdout — never the shape of the removal. "The retired key is absent" is
  a contract test worth keeping once (the machinery must not creep back); everything else
  is presence and value.
- Seams — reuse existing ones; no new seam layer is introduced:
  1. **Engine public API** (successor to the deleted fingerprint unit test, same seam):
     resolve a registered name against a temporary root; a relative sub-path filename
     resolves; `provenance()` carries exactly the kept keys; the missing-file error names
     the expected path and lists top-level files plus sub-directories.
  2. **Frames Record seam** (existing frames unit test): the new Record carries `ENGINE`
     and the resolved-path field and none of the retired keys; each frame carries
     `engine=`; injected-calculator builds write `-`.
  3. **Dataset seam** (existing dataset unit test): index rows carry `engine`; the Record
     carries `ENGINES`; old fixtures (with the old attribute) still build.
  4. **Integration seam** (the existing frames→engine integration test): with the real
     engine, the frame attribute equals the registered engine name.
  5. **Tooling seam** (`t_probe_calibration` and the reduced CLI): the stub provenance is
     updated; the reduced tool runs, prints the mace identity lines, exits 0.
- **Deleted**: the engine-fingerprint unit test as a whole, including its registration in
  the test-group list.
- **Prior art**: the existing frames / dataset / branch-A unit tests for Record-content
  assertions; the frames→engine integration test; the probe-calibration tooling test.
- **Fixtures** under `tests/data/` stay byte-identical (old Records are immutable); only
  assertions change.

## Out of Scope

- Training-side fields and callers — `FOUNDATION_PARAMS_SHA256` / `MODEL_PARAMS_SHA256` /
  `MODEL_N_TENSORS`, the `ENGINE_PARAMS_SHA256` alias, `FOUNDATION_FILE`, the registration
  output, `06_judge`'s arguments, `hl_train.slurm`: ticket 07.
- Documentation, slurm and reference sweep, including the three stale slurm readers that
  raise `KeyError` today: ticket 12.
- The publication repository, releases and checksums: the publication ticket (13,
  blocked by 09).
- `CONFIG_SHA256` (the training recipe's identity, kept) and the other non-weight hashes
  (`edge_list_sha256`, the qm9 list hash, curated content hashes, the mace-patch source
  hash).
- Any change to `check_weights_are_physical`'s behaviour or its skip variable.
- Migrating or re-reading old Records; the mace-side fork work.

## Acceptance

- [x] No fingerprint/pin symbol and no retired Record key remains in the sliced modules;
      a grep for them over those modules is clean.
- [x] A frames Record built with a real engine carries `ENGINE` and the resolved
      weights-file path; each frame carries `engine=<name>`.
- [x] A dataset built across old fixtures and new frames works; the index rows carry
      `engine`; the Record carries `ENGINES`.
- [x] `provenance()` returns without the four retired keys; the missing-file error still
      names the expected path and the directory contents (files and sub-directories).
- [x] `s0_check_weights.py` prints the mace identity and the presence listing; none of
      the deleted flags exists.
- [ ] openQHA's unit and integration groups pass; the deleted test is deregistered.
      -> unit: 63/63 (the new `t_engine_identity.py` included, `t_engine_fingerprint.py`
      gone from the group). integration: 12/13 -- see the Answer; the one failure is
      ticket 07's half of decision 04, not this slice.
- [x] Reported per the operating rule: files changed, checks run, anything not verified
      (see the Answer).

## Further Notes

- The grilling is recorded in ticket 04; this slice exists because its openQHA-side edits
  do not travel with [The move](../decisions/07-the-move.md) (the moved files are
  run/judge and their callers).
- The map's "Not yet specified" fog line about `s0_check_weights.py`'s reduced form is
  answered here; the 04 resolution removes that line.
- The two tooling scripts this slice edits are also touched by 01c/01d for their import
  addresses; whichever side lands second rebases trivially.
- Flagged for the registration/publication side, not decided here: `level_name()` derives
  a Level string from a self-trained registered name (e.g. `draw300-r4+…`); whether a
  fine-tuned revision is a new Level is decided when the first model is registered.
- Environment reminder for the implementer: run the openQHA test groups from the repo
  root in the WSL `openqha` env (see `openQHA/AGENTS.md`).

## Answer (2026-09-27, implemented in this commit)

The strip landed as specified. What changed:

- **Engine** (`openqha/potentials/engine.py`): `fingerprint_state_dict`,
  `parameter_fingerprint`, `_FINGERPRINTS`, the registry pin and the pin WARN are gone;
  `provenance()` keeps engine / source / note / weights_path / bytes / interface / mace and
  fork identity / dtype / patch state (exactly those keys; the test pins the set);
  `model_path()` resolves a relative sub-path and its missing-file error now adds the
  root's sub-directory names after the top-level `*.model` listing; the
  `mace_off23_<campaign>/<run>+<stamp>.model` storage convention is in the module
  comments; `check_weights_are_physical` and both overrides untouched.
- **Records**: frames Record carries `ENGINE` + `WEIGHTS_FILE` (`-` for an injected
  calculator), each frame `engine=<name>`; the dataset index column is `engine` and the
  Record carries `ENGINES` (sorted unique); branch-A's Property drops both fields; the MD
  Record's provenance print keeps `weights_path` / `bytes` / `interface` / versions. Old
  Records and the fixtures under `tests/data/` are untouched (the frames unit test
  rewrites a copy, not the fixture).
- **Tooling**: `s0_check_weights.py` is `print_mace()` + a name/path/bytes listing (the
  deleted flags are refused by argparse); `s0_probe_calibration.py` and
  `s0_hl_smoke_fit.py` drop the field writes and prints with their schema entries.
- **Tests**: `t_engine_fingerprint.py` deleted; `t_engine_identity.py` added at the same
  seam with the opposite contract (name→path resolution, the relative sub-path,
  `S0_MACE_MODEL`, the missing-file message, `provenance()`'s exact key set, the reduced
  CLI and its deleted flags); `t_frames.py`, `t_dataset.py` (one engine frame kept with
  the pre-2026-09-27 attribute, read as `-`), `t_branch_a_property.py`,
  `t_probe_calibration.py`, `t_frames_engine.py` and `t_judge_engine.py` adjusted.

Checks run (WSL, `openqha` env, repo root):

- `python tests/run_tests.py` -> **all 63 tests passed** (includes
  `t_engine_identity.py`; the deleted test is not discovered).
- `python tests/run_tests.py --group integration` -> **12 of 13 passed**; the failure is
  `t_train_engine`, which dies at `openqha/training/run.py:600` calling the deleted
  `engine.parameter_fingerprint`. That call site is the training-side half decision 04
  assigns to [The move](../decisions/07-the-move.md) -- this ticket declares it out of
  scope -- so the integration group cannot be fully green until 07 lands.
- `t_frames_engine.py` runs the frame-set step against `tests/data/propanal_molecule`
  **in place**: a run rewrites that fixture's `frames/` and leaves it dirty. It was
  restored to HEAD before this commit; the fixture is byte-identical.

Not done here, assigned elsewhere: the training-side fields and callers and
`hl_train.slurm` (07); the doc/slurm/reference sweep, including the three stale
`prov["sha256"]` readers, `README.md`, `environment-cuda.yml:160` and
`check_dependency.py` (12).

Flagged to 12 (not in its list yet): `hpc/slurm/hl_branchA.slurm:63`,
`hl_pipeline_debug.slurm:67` and `q5-rerun-parked.sh:420` grep the reduced tool's stdout
for `registry pin` / `params sha`; those strings are gone, so the `|| echo "... weights
check failed"` fallback now fires on every branch-A job. The reduced tool prints
`path` / `bytes` lines instead.

## Review record (2026-09-28, annotations)

Two-axis review of the slice's diff (against its parent `3eb4b74`), per the
`code-review` skill, with the dispositions below; re-baselined after 01c landed:

- unit group: **all 63 tests passed** (`python tests/run_tests.py`);
- integration group: **12 of 13 passed** -- the one failure is `t_train_engine`, unchanged.

**Standards (repo standards + the smell baseline).**

- Two rewritten docstring lines kept CONTEXT.md's `_Avoid_` words ("a run's potential",
  "makes a product reproducible") -> fixed before the commit ("What identifies a
  potential in every Record...", "makes a Record reproducible").
- The decision-04 rationale is retold in several module docstrings -> kept: the ticket
  asks each module to state the identity it now carries.
- `s0_check_weights.py` recomposes `model_root()/<filename>` instead of the resolver ->
  kept: the listing exists to test resolution WITHOUT the resolver's raise on a missing
  file.
- `t_frames_engine.py` hard-indexes the new keys -> kept: the test regenerates the Frame
  set first, so the new contract is always what it reads.
- Sanctioned deferrals (training-side callers, the slurm/doc sweep) -> made visible in
  this Answer; the flagged slurm `registry pin` greps are now in ticket 12's list
  (`e6f6a7a`).

**Spec (this ticket).**

- The `tests/data/propanal_molecule` fixtures were rewritten in place by the integration
  run -> restored to HEAD before the commit; a later `--all` rewrites them again, so
  restore (`git checkout -- tests/data/propanal_molecule`) and never commit that dirt.
- `t_judge_engine.py`: dropping `engine_params_sha256=` / `base_params_sha256=` from the
  `judge.run(...)` call is strictly the training-side field work -- but the call could
  not stand as written (the keys it read are gone), so the minimal fix keeps the caller
  working; the final shape stays with 01d.
- The `openqha.training` -> `openqha_hessian` import switches in the three shared files
  are 01c's and were excluded from 04a's commit (filtered staging); 01d closes the rest.
- The registration recipe in `engine.py` ran the tool before the entry existed -> fixed
  to copy, add the entry, then `s0_check_weights.py <name>` verifies.
- "The retired key is absent" is asserted per Record (frames / dataset / branch-A) rather
  than once -> kept: each is the contract of its own Record.

**Shared-worktree postscript.** 04a's commit was staged mine-only (the 01c session's
import hunks excluded, verified no revert). The ticket's `Status` line did not survive
that commit (a concurrent write race); `e6f6a7a` set it to `resolved`, committed the
`map.md` 04a line and moved the flagged slurm greps into ticket 12. The training-side
callers this Answer places with [The move](../decisions/07-the-move.md) are executed by
[01d](01d-the-switchover.md).
