# 16c: The rulebook and the dates

Type: task
Status: resolved
Blocked by: None.
Serves: [16](../decisions/16-identity-without-checksums.md).

**What to build:** the repo's own documents state the new rule and reconcile the old
ones by date -- a new ADR records "identity without checksums" (what came out, the seed
exception, the kept fork identity, the relationship to ADR 0012); dated notes reconcile
decision 04's kept list, the record-slimming ruling, and the campaign transfer rule; the
runbooks' future-facing sha instructions and the two-side commit-locking discipline are
removed where they instruct, with one dated line each where an executed receipt is
deleted; requirements/environments stop carrying the fork-by-URL pin.

- [x] New ADR (next free number) states the ruling, the checksum-vs-seed boundary, and the Q9=A kept set.
- [x] Dated notes in: decision 04, the record-slimming ruling's pages, the campaign transfer rule.
- [x] Runbooks/deploy guidance: sha receipt steps gone; "both sides at the same commit" is no longer a requirement; executed receipts each get one dated line.
- [x] requirements/environment files: fork-by-URL lines removed; pin guidance closed out (the install path is the carried checkout, see 16d).

## Notes

- Dated notes date the change; they do not rewrite the old sentences (house rule).

## Answer (2026-10-02, implemented in this commit)

**ADR 0014 landed** — `docs/adr/0014-identity-without-checksums.md` (status accepted,
2026-10-02): what comes out (every sha256 the project adds of its own — the Record's
`CONFIG_SHA256`, the reference-data digests, the receipt steps that computed or compared
digests), the checksum-vs-seed boundary (the PRNG seed material stays — `frames.frame_seed`
and `dataset._rng` feeding `valid_probes` / `frame_draw`; upstream mace/CREST untouched),
the kept product identity (Q9=A: `check_fork`'s refusals + the Record's fork/package commit
fields), and the narrowed clauses of ADR 0012 (URL default, env-files-carry-the-fork, the
pure-eval ride), ADR 0013's bare-branch note and ticket 04's kept list.

**Dated notes (old sentences kept).**
- Decision [04](../decisions/04-sha256-retirement.md): a postscript — the kept list is
  overwritten (`CONFIG_SHA256` and its two schema-test pins retire — 16a; the non-weight
  hashes retire — 16b; the one remaining use is the seed material), and the Publication
  paragraph's release-note content reconciles to index + the config file.
- Record-slimming pages: [decisions/09](../decisions/09-round-1-run.md)'s record-field note
  gains the checksum-field paragraph; `workflows/hessian_learning/05_train.py`'s Record
  paragraph now says "the config file the Record names" (the config-SHA field retired,
  2026-10-02); the workflow README's step-05 row and register sentence — two pages the
  record-slimming sweep had missed (they still said `train.{out,toml,dat}` and the config
  SHA) — read the slim record + the config file, dated. (The `05_train.py` sentence is the
  one [16a](16a-the-record-without-the-checksum.md)'s Answer records as "updated by a
  concurrent session"; this slice owns it.)
- Campaign transfer rule (`docs/hessian_learning_campaign.md`): the "both sides size+sha
  equal" line is struck, dated — no substitute, the group read-back is the check;
  `--digest`/`--sha` leave the tool (16b).
- [13](../decisions/13-publication.md): the release-note content line's `CONFIG_SHA256` prefix
  is out (dated parenthetical).

**Runbooks/deploy guidance.** 09g: the sha receipt steps are gone (step 6's `sha256sum`
line deleted; receipt 7 and the receipt list read "tar paths + registry ls"); the (四)
revision note records it. "Both sides at the same commit" is no longer a requirement —
annotated at the 前置, in the checkout paragraph and in the note (each tree records its own
`git log -1` + clean `git status --porcelain`; 16e is the landing). Executed receipts: 08a's
step-6 fetch `sha256sum` and 09e's 3.2 four-hash step are deleted, each leaving one dated
line; the executed values stay where they were recorded (decision [08](../decisions/08-round-1-xyz.md)'s
Answer, [09d](09d-the-two-30k-draws.md)'s table, 09e's status) — history, not
rewritten. `docs/tianhe_runbook.md` §3's "the hash is recomputed on every load" is struck
and dated (loads do not hash). `hpc/slurm/install_env_tianhe.slurm` §9's comment no longer
says the environment files install the fork by URL (the carried checkouts + editable pair,
dated).

**Requirements/environments.** The fork-by-URL pin is removed from `requirements.txt`,
`requirements-minimal.txt`, `environment.yml`, `environment-cuda.yml`,
`environment-tianhe.yml` and `environment-tianhe-gpu.yml`; each pin comment is closed out to
"the sibling `mace/` checkout, installed by `openQHA-Hessian/install.sh`" (dated), and the
two Tianhe env headers keep the same story. `install_dependency.sh` — the executable face of
those files — carries the same closed-out guidance; its `import mace` probe now warns and
points at `install.sh` when mace is absent (it installs no mace at all), and the closing
message lists `install.sh` first.

**Checks run.** `bash -n` clean; shellcheck `-S warning` no delta vs HEAD
(`install_dependency.sh`; `install_env_tianhe.slurm` re-checked after the review fix); the
four environment files parse (`yaml.safe_load`); `py_compile` on `05_train.py`; residual
scans: no `git+https` in the six requirements/environment files, no sha receipt step left in
08a/09e/09g (only the dated notes mention sha256). openQHA suite `--all`: **73/73, rc 0**
(the landing tree; log `%TEMP%\oqc16c_suite.log`).

**Two-axis review** (per the `code-review` skill; two read-only sub-agents over the working
diff vs HEAD). Standards — one inconsistency (a link form in decisions/09) fixed in-pass;
the scope disclosure added; the "100755" ADR mode was a drvfs artifact
(`core.filemode=false`). Spec — three findings fixed in-pass (the two Tianhe env headers,
09g's "两树 id 相同" line, the 04/13 `CONFIG_SHA256` release-note mention); the "executed
receipts" reading applied: delete the steps, one dated line per deleted step, keep the
recorded values as history.

**Not verified.** The Tianhe side (16e; untouched by design). Sibling slices: 16a landed in
the package (`ada4e4d`) and 16d landed alongside this slice (package `9bfacef`; its tracker
commit was still pending at this commit); 16b was mid-flight in the shared tree — no
sibling files were touched by this slice. One found-but-untouched leftover, reported not
edited: `openqha/potentials/engine.py`'s registry narrative still says `source` = "index
path + training config SHA" (the pre-16a form; the file is the 16b session's live surface).

**Reported per the operating rule.** Files changed: 18 tracked files + the new ADR +
this record + the map line (the commit's diff is the list). Nothing else was staged.
