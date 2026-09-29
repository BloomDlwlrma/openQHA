# 12c: The public documents (one pass)

Serves: [12](../decisions/12-references-sweep.md) · spec: [spec-references-sweep.md](../spec-references-sweep.md).

**What to build:** A single pass over every public document: the content corrections merged with the citation strip. Content: the workflow README's split story (whole-molecule split = the production default; per-frame = the smoke/fit mode), the fraction defaults, the package addresses, the fork naming, the engine wording; the campaign page's dataset guidance and log example aligned with the dataset actually built; the README's weight-register section and weights-table line; the dependency-check script's claims; the environment comment; the Slurm house README; the tutorials' source cells (current package/checkout; no re-execution). Strip: the one rule set — process tokens out, event sentences become current-state sentences, measured-fact dates kept. A per-file count table precedes any edit.

**Blocked by:** None (can start immediately).

**Status:** resolved

- [x] The count table is produced before any edit.
- [x] The split story, fractions, addresses and fork naming match the code and the build actually run.
- [x] No process tokens remain in the touched documents.
- [x] Tutorial source cells reference the current package and checkout; stored outputs untouched.
- [x] The campaign-reading unit tests stay green.

## Answer (2026-09-30, implemented in this commit)

**The count table (before any edit).** Process tokens per public file — the public face is
the tracked tree minus the scratch trackers, the ADRs, `CONTEXT.md` and `AGENTS.md`;
notebook counts are cell sources only (outputs excluded):

```
file                                                     ticket ruling grill ADR CONTEXT scratch S0-code roundN Qcode D0-code set-ref
README.md                                                     5      2     0   2     0       1       0      0     0       0       1
check_dependency.py                                           0      0     0   0     0       0       0      0     0       0       0
configs/README.md                                             0      0     0   0     0       0       1      0     0       1       0
configs/mdp/README.md                                         0      0     0   0     0       0       0      0     0       1       0
docs/branchA_production.md                                    0      5     0   0     0       0       1      0     0       5       0
docs/branchA_workflow.md                                      0      9     0   1     0       0       1      0     0      10       0
docs/branchB_production.md                                    0      0     0   0     0       0       0      0     0       1       0
docs/branchB_seeds_and_length.md                              0      2     0   0     0       0       0      0     0       0       0
docs/branchB_workflow.md                                      0      1     0   0     0       0       1      0     0       1       0
docs/hessian_learning_campaign.md                            10      9     0   0     0       0       3      5     2       0       0
docs/output_inventory.md                                      2      0     0   4     2       0       0      0     0       0       0
docs/records_inventory.md                                     3      0     0   0     0       0       0      0     0       0       0
docs/tianhe_install.md                                        0      0     0   0     0       0       5      0     0       0       0
docs/tianhe_runbook.md                                        0      2     0   2     0       0       0      1     0       4       0
docs/tutorials/T01_*.ipynb                                    0      1     0   0     0       0       0      0     0       0       0
docs/tutorials/T01b_*.ipynb                                   0      0     0   0     0       0       1      0     0       0       0
docs/tutorials/T04_*.ipynb                                    2      0     0   1     0       0      28      0     0       0       0
docs/tutorials/T05_*.ipynb                                   18      4     1   2     0       1      43      1     0       0       1
examples/README.md                                            0      0     0   0     0       0       0      0     0       1       0
examples/02c_hessian_benchmark_levels/README.md               0      0     0   0     0       0       0      0     0       1       0
hpc/README.md                                                 0      5     0   1     0       0       1      0     0       7       0
hpc/slurm/README.md                                           0      0     0   1     0       0       0      0     0       0       0
scripts/README.md                                             0      1     0   0     0       0       1      0     0       0       0
tests/README.md                                               0      0     0   0     0       0       1      0     0       1       0
workflows/hessian_learning/README.md                         18      4     0   2     3       0       6      8     7       0       0
environment*.yml (4 files, D0/ruling tokens)                  0      5     0   0     0       0       2      0     0       6       0
```
(`set-ref` = study-set names, e.g. `hessian-learning-set`, `msrrho-gtotal`; `D0-code` =
the second study-code family, added to the tally after the first pass found it. Pure-date
files — `docs/cite/README.md`, `docs/branchB_production.md`'s remaining hits, the
remaining `examples/*/README.md` — carry measurement dates only and were verified, not
edited; the annotations commit below adds `T02` and the last `plan_*`/hash-claim cleanups
the two-axis review found, after which the same scan reads 0 including `plan_[A-D]`.)

**What landed.** The named content fixes, merged with the citation strip in one pass per
file:

- `README.md` — "The potential's weights" rewritten to the native model (name → filename →
  the one directory; no hash, no fingerprint, no pin; the registry's three fields); the
  registering recipe now `05_train.py --register-copy` (stamped revision under
  `mace_off23_<campaign>/`, the printed `ENGINES` entry); `s0_check_weights.py` described
  as the identity + presence listing it is; "The mace fork" kept current (fork
  `BloomDlwlrma/mace@openqha-hessian`, `../openQHA-Hessian/install.sh`), the `.scratch/`
  effort pointer and every ticket/ADR token out; the data-table weights line corrected.
- `workflows/hessian_learning/README.md` — step 05/06 driver addresses → `openqha_hessian.run` /
  `openqha_hessian.judge`; the fork named `BloomDlwlrma/mace`; the loss address and argv →
  `openqha_hessian.phl_loss:build`; **the split story corrected** — `molecule` (whole-molecule)
  is the production default and `frame` the smoke/fit mode, and the fractions are **5 % in
  both modes** (verified against `openqha/data/dataset.py` `DEFAULT_SPLIT_MODE`,
  `VALID_FRACTION`/`TEST_FRACTION`/`FRAME_*` = 0.05 and `04_dataset.py`); the index column
  list fixed (`engine`, not "engine fingerprint"); the base/fine-tuned "fingerprints"
  replaced by weight files; the R4 block and judge examples use the printed engine name;
  the registering tail matches the new recipe. Every ticket/ruling/round/study token out;
  the retry-contract sentences the unit test pins kept verbatim.
- `docs/hessian_learning_campaign.md` — §7's split guidance flipped to the production
  default (the actual build ran `--split-by molecule`), §4's log example now reads
  "(split by molecule)", the engine names follow the printed name; tokens stripped with
  every pinned string kept.
- `docs/tianhe_install.md` — the fork note re-checked (fork/branch/base tag, the editable
  pair via `install.sh`, non-editable eval line), `S0-G-67`/`S0-G-63…67`/`S0-C-64`/`S0-C-67`
  out, the `.mem/decisions` pointer row dropped.
- `docs/tianhe_runbook.md`, `docs/branchA_production.md`, `docs/branchA_workflow.md`,
  `docs/branchB_production.md`, `docs/branchB_seeds_and_length.md`, `docs/branchB_workflow.md`,
  `docs/output_inventory.md`, `docs/records_inventory.md` — the one rule set (tokens out,
  event sentences → current-state sentences, measurement dates kept; the kept-history
  blocks reworded, not deleted).
- `hpc/README.md`, `hpc/slurm/README.md` — the "recomputes the MACE-OFF SHA-256" item
  replaced by the reduced tool's presence listing; `D0-*`, `plan_B`, ADR and ruling tokens
  out.
- `check_dependency.py` — "imported, executed or hashed" → "imported or executed"; the
  "path AND a checksum" comment → "a path"; `environment*.yml` + `install_dependency.sh`
  — "hash-checked on every load" → "resolved by name at load", the remaining env-file
  ruling/D0 tokens out; `scripts/production/s0_package1_crest_census.py` — the "hashes the
  weights only" comment corrected.
- `examples/README.md` — the broken practice-notebook link (`T02…Practice…` → `T01…Practice…`)
  fixed; table tokens out. `scripts/README.md`, `tests/README.md`, `configs/README.md`,
  `configs/mdp/README.md`, `examples/02c_hessian_benchmark_levels/README.md` — tokens out.
- Tutorials — **T04/T05 source cells** move to `openqha_hessian` imports, the mace fork
  checkout, the printed engine name, and lose every S0/ticket/grilling/`.scratch`/ADR token;
  **T01** loses the ruling token and its retired `engine.sha256` print (now the resolved
  weights filename); **T01b**'s token + `.mem` pointer out; **T02** — public but outside
  the first set — swept in the annotations commit (engine import, provenance prints, the
  prose's retired hash check). Stored outputs untouched: cell counts and output arrays
  byte-identical; the editor reset the edited cells' execution counters to null (8 cells)
  and T01b gains a trailing newline — no output object changed.

**Evidence.** Residual scan over every public `.md`/`.yml`/notebook-source after the pass:
**0** hits for `ticket|ruling|grilling|ADR N|CONTEXT.md|.scratch|S0-X-N|D0-*|checkpoint N|plan_[A-D]`
(the annotations commit below re-runs the scan with `plan_[A-D]` added). Compile/parse:
`py_compile` on both touched scripts, `bash -n` on `install_dependency.sh`, `yaml.safe_load`
on the four env files, `json.load` on the five notebooks — all green. Tests: `t_hl_campaign`
PASS (all pinned page/README strings intact; re-run after the annotations' split-wording
fix), `t_engine_identity` PASS, `t_engine_fork` 18/18. Stored outputs: output arrays
byte-identical in every notebook; the editor did reset the edited cells' `execution_count`
to null (8 cells) and T01b gains a trailing newline — no output object changed. Not run
here: the full suite — this pass touches documents, comments and one tutorial print (no
behaviour), and the sweep's suite evidence is 12a/12b's 73/73 + 9/9 on the same code tree.

**Reported, not acted on** (for the later passes): the remaining `user ruling` / `D0-*` /
`.mem` tokens in `install_dependency.sh`, `configs/*.yaml` and the production scripts
belong to [12f](../implementation/12f-the-scripts-and-root-pass.md)/[12g](../implementation/12g-the-hpc-and-workflows-pass.md);
`docs/branchA_production.md:533`'s "records redesign 2026-09-15" keeps its date (a document
version mark, not a ruling).

## Review record (2026-09-30, annotations commit)

Two-axis review of `dbbead3` (range `4eda3fa..dbbead3`), per the `code-review` skill, run
as two read-only sub-agents. Hard findings, all fixed in this commit:

- **Standards.** Five stale "hash correctly"/"SHA-256 on load"/"recomputes … SHA-256"
  claims surviving in files the pass had already touched
  (`docs/branchA_production.md`, `docs/tianhe_runbook.md`, `hpc/README.md`,
  `environment.yml`, `environment-tianhe.yml`, and `install_dependency.sh`'s weights-block
  comment) — replaced by resolve-by-name wording. **T01**'s licence line still advertised
  "only a path and a SHA-256"; **T02**, public but outside the named set, was broken by
  the identity strip (its cell printed `prov["sha256"]` and `prov["licence"]`, both
  retired; the prose promised a raising hash check) — swept here too. `examples/02d…/README`
  linked `.mem/plan/plan_AB…`; `examples/02a…/README` said "weights, hash-checked";
  `tests/README.md` still cited "plan_D section … checkpoint"; `scripts/README.md`,
  `examples/README.md` and `examples/02c…/README.md` still cited `plan_C`/`plan_D` — all
  removed (the first scan's pattern missed `plan_[A-C]`). `branchA_production.md`'s
  directory table + paragraph now allow the registry's `mace_off23_<campaign>/` sub-path
  (FLAT applies to the base weights), matching `engine.py` and this ticket's own recipe.
- **Spec.** The split story re-verified against `openqha/data/dataset.py`: `molecule` is
  the default, fractions are 0.05 in both modes — but the molecule mode's `valid` draw is
  **per molecule**, not "by frame" (that is the frame mode's generator); the workflow
  README's drawn-by sentence was corrected. Count-table corrections: `environment*.yml`
  is 4 files, not 5; the `set-ref`/`D0-code` columns are defined in the footnote; the
  "diffs are source-line only" claim now states the execution-counter reset explicitly.

Judgement calls (kept as written, flagged for the human): the three layout-history
blockquotes keep their aligned wording; the eight reset execution counters are recorded,
not restored (the output objects are byte-identical; a scripted restore would risk
reformatting whole files for a cosmetic field); the T01/T01b touches and the metadata
churn that come with any notebook-tool edit are accepted. Not verified locally: that the
round-1 build invoked `--split-by molecule` in the flesh — taken from the map's record and
the campaign page, both of which now agree with `dataset.py`.
