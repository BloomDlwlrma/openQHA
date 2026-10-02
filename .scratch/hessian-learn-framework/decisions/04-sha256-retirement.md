# SHA256 retirement: replacements, record fields, and the load-time checks

Type: grilling
Status: resolved
Blocked by: 03
Part of: [hessian-learn-framework](../map.md)

## Question

Work item 1 of the user's 2026-09-26 message: remove SHA256, call models the native mace/mace-off way. Decide exactly what retires and what replaces it; the sweep of code, tests and docs follows from the answer.

Decide:

1. **The machinery.** Which of these retire: `fingerprint_state_dict()`/`parameter_fingerprint()`/`_FINGERPRINTS` (`openqha/potentials/engine.py:368–407`); the optional `params_sha256` field on `ENGINES` entries (only `MACE-OFF23_medium` is pinned, engine.py:153); `scripts/tooling/s0_check_weights.py`'s `--pin` (its real output is a paste-able registry entry); `--json`/`--compare` (machine-to-machine model comparison); the report-only `params_pin_status` line in `provenance()` (engine.py:250–322, WARNs but never gates). What, if anything, survives in reduced form — and what does registering a fine-tuned model become (`openqha/training/run.py registry_entry()` 845–865; `05_train.py --register/--register-copy`)?
2. **The replacement.** The native loading path per [Native model loading](03-native-model-loading.md): which callers change (`engine.calculator()`; `training/run.py --foundation_model` 377–382; `06_judge.py`; the OpenMM route's custom loader `openqha/quasi_harmonic/openmm_mace.py` 56–63), and how a fine-tuned model file is named/handed around once it is no longer "a registry entry with a pin".
3. **Load-time checks.** `check_weights_are_physical()` (engine.py:448–506 — the 2026-09-10 Tianhe NaN event is its reason to exist): keep? `S0_MACE_MODEL`/`S0_MACE_ROOT` path overrides: keep? What is the behaviour now for a missing or wrong model file?
4. **Records.** The fields that carry hashes stop being written — `ENGINE_PARAMS_SHA256`/`BASE_PARAMS_SHA256` (`training/judge.py:149–154`), `FOUNDATION_PARAMS_SHA256`/`MODEL_PARAMS_SHA256` (`training/run.py:111–113`), and the frames/dataset/branch-A copies (`data/frames.py:178`, `data/dataset.py:208`, `store/branch_a_property.py:35`). Say which retire for *new* Records and which change meaning (e.g. `MODEL_N_TENSORS`); confirm old Records stay readable untouched (Records are immutable). `CONFIG_SHA256` hashes the training config, not weights — state whether it is in scope.
5. **Sweep list** (execution lands in [The move](07-the-move.md) / [References sweep](12-references-sweep.md)): tests `t_engine_fingerprint.py`, hash assertions in `t_train_run.py`/`t_train_engine.py`, fixtures embedding `e986a6cf…`; the stale Slurm readers of `prov["sha256"]`/`["sha256_pinned"]` (`hpc/slurm/branchA_debug.slurm:58–60`, `branchA_deimos.slurm:73–74`, `branchB_traj_tianhe_a.slurm:108–109` — a KeyError today); docs comments that already drift (`environment-cuda.yml:160` "hash-checked on every load"; `check_dependency.py`'s checksum comment).

## Facts to stand on

- The fingerprint answers "same numbers?" via name+dtype+shape+raw bytes over the state_dict; it is a *reported* identity, never a gate (a difference only WARNs).
- There is no automatic verification at load; `check_weights_are_physical` is the only load-time refusal besides a missing file, and it is not an identity check.

## Answer (2026-09-27, grilling; all questions ruled by the user. OpenQHA-side slice: [04a](../implementation/04a-strip-weight-identity.md).)

**Route (the frame for items 1–2).** Minimal strip — no `mace_off` delegation.
`engine.calculator()` is already the native call (`MACECalculator(model_paths=…)`; mace's
load path hashes nothing), so "native" is realised by removing our own verification
layer, not by adopting mace's download/resolve helpers. The registry, the single flat root
and both path overrides stay. The helpers stay out of the production path because of
offline Tianhe, the ASL download behaviour, the committee engines, and MACE-OFF23-SC
being absent from the mace-off repository. Publication of our own fine-tunes imitates
mace-off's *distribution* conventions, not its fetching layer (below).

**1. The machinery — all of it retires.** `fingerprint_state_dict()`,
`parameter_fingerprint()`, `_FINGERPRINTS`; the optional `params_sha256` field on
`ENGINES` (the one pin, `MACE-OFF23_medium`, with its comment); `provenance()`'s
`params_sha256` / `n_tensors` / `params_bytes` / `params_pin_status` and the mismatch
WARN (the file-size `bytes` stays); `s0_check_weights.py` shrinks to `print_mace()` plus a
presence/size listing of registered files — `--pin`, `--json`, `--compare`, `file_sha256`
and every fingerprint call go. No transplant of upstream's golden hashing into the test
tier: upstream's hashes protect its own download/redistribution channel; an artifact
regression would be a new ticket done the upstream way, and once we are publishers,
checksums belong to the release ([13](13-publication.md)), never to the load path.

**2. The replacement — the load path does not change; the identity wording does.**
Nothing in `calculator()`/`model_path()` changes except the missing-file diagnostics'
directory listing (item 3). `provenance()`'s identity sentence becomes: engine name +
resolved path (+ fork commit, dtype). Registering a fine-tuned potential becomes: copy the
file into the decided sub-directory, add one fixed `ENGINES` entry (`filename` / `source` /
`note`; `source` = Dataset index + `CONFIG_SHA256` prefix), select with `S0_ENGINE`.

**3. Load-time checks.** `check_weights_are_physical()` and `S0_SKIP_WEIGHT_CHECK` are
kept — it is not an identity check and it is the one refusal that catches the 2026-09-10
Tianhe corruption class. `S0_MACE_ROOT` / `S0_MACE_MODEL` are kept unchanged. A missing
file keeps the guidance error; its "what the directory holds" line gains top-level
sub-directory names. A wrong-but-valid file: no refusal and, by design, no report — the
identity is now name + path.

**4. Records (new writes only; old Records untouched, immutable; schema keys removed,
not kept with `None`).**
- `provenance()` keeps engine / source / note / weights_path / bytes / interface / mace &
fork identity / dtype / patch state.
- frames Record: `ENGINE_PARAMS_SHA256` and `ENGINE_PIN_STATUS` retire; adds
`WEIGHTS_FILE` (the resolved weight file; `-` with an injected calculator). *(The grilling
approved the name `ENGINE_FILE`; renamed because CONTEXT's **Engine file** and the dataset
index's existing `engine_file` (the molecule's engine-level xyz) make it ambiguous.)*
- frames' per-frame key: `engine=<registered name>` replaces `engine_params_sha256`.
- dataset: index column `engine` replaces `engine_params_sha256`; the Record carries
`ENGINES` (sorted unique) in place of the `ENGINE_PARAMS_SHA256` array.
- branch-A property and the MD Record provenance block: the same fields retire.
- training (lands in [The move](07-the-move.md)): `FOUNDATION_PARAMS_SHA256`,
`MODEL_PARAMS_SHA256`, `MODEL_N_TENSORS` and the `ENGINE_PARAMS_SHA256` alias retire; adds
`FOUNDATION_FILE` (the resolved base path — without it an `S0_MACE_MODEL` override makes
`FOUNDATION_MODEL` lie).
- `CONFIG_SHA256` is **not in scope** (kept): it is the training recipe's identity and,
with publication, part of the public provenance. The non-weight hashes (edge list, qm9
list, curated content, mace-patch source) are untouched.

**Storage, naming, registration (Q6/Q7).** A registry `filename` may be a relative
sub-path; base models stay flat in the root; self-trained revisions live under
`mace_off23_<campaign>/` as `<run>+<YYYYMMDD-HHMMSS>.model` (UTC; basename globally
unique). One fixed registry entry per revision (e.g. `draw300-R4+20260927-101530`), never
a moving pointer: the load path never resolves "latest" itself (a glob handed to
`MACECalculator` silently makes a committee). `--register` prints the entry without a pin
line; `--register-copy` creates the sub-directory.

**Publication (Q8).** Our own fine-tuned models go out the mace-off way in *conventions*:
their own model repository (separate from `openQHA-Hessian`, whose history is being
replaced), tagged GitHub Releases for bytes that are immutable by convention, a README
with the load lines (`mace_off(model="<https URL>")` works unchanged), licence (ASL
derivative — to settle), citation, and release notes carrying index + config sha + fork
commit. An optional release-time `sha256sum` listing is publisher-side only. First
artifact: the round-1 model of ticket 09. Conventions live here; execution is ticket
[13](13-publication.md) (blocked by 09).

**5. Sweep list — assigned.**
- **A → [The move](07-the-move.md)**: `training/run.py`, `training/judge.py`, the
`05_train.py` / `06_judge.py` callers, `hpc/slurm/hl_train.slurm`'s provenance print, and
the assertions in `t_train_run.py` / `t_train_engine.py` / `t_judge_engine.py`.
- **B → [04a: The weight-identity strip](../implementation/04a-strip-weight-identity.md)**
(new slice, `Serves: 04`): the engine module, the reduced tool, the frames / dataset /
branch-A / MD-Record fields, the two tooling scripts, and their tests
(`t_engine_fingerprint.py` deleted; `t_frames.py`, `t_dataset.py`,
`t_branch_a_property.py`, `t_frames_engine.py`, `t_probe_calibration.py` adjusted).
- **C → [References sweep](12-references-sweep.md)**: the three stale Slurm readers
(`prov["sha256"]` — a KeyError today), the README registration section, the workflows
README lines 296–297/368–371, `check_dependency.py`'s checksum comment,
`environment-cuda.yml:160`, tutorials, `.scratch/hessian-learning-set` pointers — swept
under a full grep for `fingerprint|params_sha256|pin`. Fixtures under `tests/data/` stay
untouched (old Records are immutable); only assertions change.

**Flagged, not decided here.** `level_name()` derives a revision-flavoured Level string
from a self-trained registered name (`draw300-r4+…`); whether a fine-tuned revision is a
new Level is decided with [13](13-publication.md) / the first registration.

**Postscript (2026-10-02, user ruling — [identity without checksums](16-identity-without-checksums.md)).**
The kept list above is overwritten: `CONFIG_SHA256` retires with the record-slimming field
table (the recipe's identity becomes the config file the Record names, `CONFIG_FILE`; the
two schema-test pins on `CONFIG_SHA256` retire with it — slice
[16a](../implementation/16a-the-record-without-the-checksum.md)), and the non-weight hashes
retire too (the QM9 list, the curated pack, the artifact index, the edge list, the
mace-patch source — slice [16b](../implementation/16b-the-reference-data-without-the-checksums.md)).
The one sha256 use that stays is PRNG seed material (`frames.frame_seed`,
`dataset.valid_probes` / `frame_draw`). The Publication paragraph above reconciles the
same way: the release note's content is index + the config file the Record names (the
`CONFIG_SHA256` prefix is out; [13](13-publication.md) carries its dated line). Durable
record: `docs/adr/0014`.
