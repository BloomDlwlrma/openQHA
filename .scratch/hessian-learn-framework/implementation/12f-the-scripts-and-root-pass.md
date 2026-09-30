# 12f: The scripts and root pass

Serves: [12](../decisions/12-references-sweep.md) · spec: [spec-references-sweep.md](../spec-references-sweep.md).

**What to build:** The one comment/string rule set applied to the production scripts and the repository root files (including the dependency-check script's claims, the install script, and the environment-file comment). Process tokens stripped; event sentences become current-state sentences; measured-fact dates kept; user-visible strings changed together with the tests that pin them; no behaviour change.

**Blocked by:** 12a, 12b, 12c — the content modifications land first; this pass runs on a stable tree.

**Status:** resolved

- [x] Every touched file compiles (`py_compile`) or parses (`bash -n` for scripts).
- [x] The residual-citation scan for the subtree is empty.
- [x] The unit suite is green.

## Answer (2026-09-30, implemented in this commit)

**Subtree.** The production scripts and the repository-level files: `scripts/**`
(14 calibration, 14 production, 15 tooling), `configs/**` (14), the `examples/**`
sources (28), `patches/**` (2), and the root files `.gitignore`, `requirements.txt`,
`install_dependency.sh` — 90 files. `check_dependency.py` and the environment files
were re-verified rather than re-edited (12c had already fixed their claims). The
historical outputs (`examples/**/sample_records/`, `logs/`, `data/`) are excluded,
like notebook outputs. Count table first: **87 files / 521 hitting lines** (families:
ticket, ruling, grilling, ADR, CONTEXT, scratch paths, `.mem` paths, S0/D0 codes,
`checkpoint N`, plan refs, Q-codes, round refs, study-set names, `decision N`); the
sweep added `defect N`, `skills section`, `AUDIT section`, `plan section`, and the
`s0-1` lecture-notes name to the family list as it met them (the way 12c's
annotations added `plan_[A-D]`).

**What landed.** The one comment/string rule set, per file, citations out and
event sentences reworded to current-state (measurement dates kept). The notables:

- `requirements.txt` — the retired "recomputes their SHA-256 on every load" claim
  replaced by the native load model (resolved by name; a missing file raises; nothing
  inspects what is inside it). 12c fixed this claim in the check tool, the env files
  and the Slurm README but not here; the "nothing changes a basin, a sigma or a
  ruling" line lost its citation.
- `configs/**` — the citation stacks out (the refine reversal story, the SHAKE
  table's provenance, the reference-level block, the F7 note, the eight split
  notices), and the user-visible values reworded (`shake_fallback.note`,
  `reference_level_rejected.reason`, `superseding` text, the h100x note). The two
  `.mem` pointer fields under `conformers_meta` (`plan:`, `checkpoints:`) are
  dropped: dead pointers to non-public docs, nothing consumes them (the 12c
  pointer-row treatment; the YAML diff in evidence). The mdp headers' prohibition
  blocks and the parameter-class pointers cleaned.
- `examples/**` — the SEEDS comments unified ("one per basin; 2+ = blank control"),
  the pipefail notes to current-state ("deliberately not used: a failing step must
  not end the job"), per-file citations out (`D0-75` on the ORCA jobs, `ADR 0001/0002`
  pointer phrases, `checkpoint 3`).
- `patches/**` — `D0-49` out of the two patch comments; the `.patch` payload and the
  applier's injected text stay textually consistent.
- `install_dependency.sh` — the user-ruling headers to current-state statements; the
  gate echo and the cluster-rules echo cleaned.
- `scripts/**` — the dense ones: `s0_A_pipeline.py` (31 sites: plan_A refs, tickets
  24/26/37, ADR pointers, defect numbers, and the record strings they fed), the two
  trajectory drivers, the `s0_E_*` parsl drivers, `s0_mem_decide`,
  `openqha_index_artifacts`, `openqha_migrate_analysis` (the generated mapping
  table's citations), the calibration docstrings, and the `s0_*` tooling one-liners.

**Kept deliberately (identifier/data classes — recorded, not swept):**

- `openqha_index_artifacts.py`'s IDENTITY `decision` values (~60 lines) — recorded
  provenance of old artifacts, written into the tool's INDEX as data; the field is
  part of the artifacts model (`openqha/store/artifacts.py`, which 12d kept as the
  live API contract). Stripping them would change the tool's output and erase
  recorded provenance (the old-assets boundary: no repair).
- `s0_mem_decide.py`'s decision-namespace scheme (the `S0-*`/`D0-*` dict keys, the
  CLI choices, the scheme documentation) — the tool's data model; its prose
  citations (`(S0-G-2)` in messages and templates) WERE swept.
- Function/flag names (`plan_workers`, `plan_shards`, `plan_frames`, `plan_msrrho`,
  `--plan-only`), the `s0-1_conformer-to-free-energy.ipynb` filename (an external
  artifact's name in two tools' default paths), the `"s0-1_edge"` mapping key, and
  `.gitignore`'s `.mem/` ignore entry (behaviour: it keeps the directory out of the
  repository).

**Evidence.**

- Compile/parse: `py_compile` over the 48 changed `.py` files; `bash -n` over the
  9 changed shell files; `yaml.safe_load` over the 11 configs — all green.
- No code change, mechanically: an AST comparison with all string constants blanked
  reads identical against HEAD for every one of the 48 `.py` files; the 6 YAML files
  with value edits (`baseline`, `branchB_protocol`, `cluster_tianhe`, `conformers`,
  `electronic`, `hessian`) were diffed key-by-key — every change is a string reword
  or the two dropped pointer keys above.
- Residual scan over the subtree (families as above; sources only): **98 hitting
  lines, all in the kept identifier/data classes** (64 + 21 + 5 + 3 + 2 + 1 + 1 + 1);
  every comment/docstring/message hit is gone.
- Unit suite: **60/60** on the final tree
  (`C:\Users\10704\AppData\Local\Temp\12f_unit_final.log`). Not run here: `--all`
  (the retirement baseline belongs to 12h); nothing submitted to Tianhe; no
  notebook re-execution.

Reported per the operating rule — files changed: the 90 above (plus this ticket and
the map line). Checks run: as listed. Not verified: the Slurm-side runtime paths were
not re-executed; a behaviour change is excluded by the AST-equivalence check.
