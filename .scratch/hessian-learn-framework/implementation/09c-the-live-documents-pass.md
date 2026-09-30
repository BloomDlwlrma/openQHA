# 09c: The live-documents pass -- the two-arm 30k Replay

Type: task
Status: resolved
Blocked by: 09a.
Serves: [09](../decisions/09-round-1-run.md) · spec: [spec-round-1-run.md](../spec-round-1-run.md).

**What to build:** One pass over the live operator-facing text: every place that
currently instructs or describes the round-1 production Replay as `4 x N` / 110,960 /
"R4" states the new standard instead — 30,000 frames, two arms (`replay30k_w1`,
`replay30k_w10`, weights 1 / 10) — credited to the mace-docs guidance and the 2026-09-30
user ruling. S0-era narrative, archived tickets and stored tutorial outputs stay
untouched.

- [x] The workflow README, the campaign page, the tutorial source cells (T04/T05) and
      the live comments (the workflow step, the package driver, the Slurm wrapper) carry
      the two-arm 30k instruction where they previously carried the `4xN` / 110,960
      instruction; lines that are S0 history say so.
- [x] A residual scan (`110960`, `4 x N_TRAIN_HESSIAN`, "R4" as a production string)
      over live text is clean outside archives and stored outputs.
- [x] The suites that read these documents stay green.

## Answer (2026-09-30, implemented in this commit; the package half is `openQHA-Hessian` @ `411abe1`)

**The pass.** One sweep over the live operator text — the round-1 production Replay stated
as the two-arm 30k draw, every remaining S0 ladder reference labelled as history, stored
notebook outputs untouched by design:

- `workflows/hessian_learning/README.md`: the block becomes "The production arms, end to
  end" — the **30,000-frame draw** of SPICE's train split at the mace-docs multihead
  guidance's size ("30000 is a good value"), drawn twice with one seed (weights 1 / 10; the
  weight lives in the file; user ruling 2026-09-30); the two draw commands (`--n 30000
  --seed 0 --weight 1` / `--weight 10` -> `spice_pt_replay30k_w1` / `_w10`); both arms
  launched together with round 1's re-selected knobs on the same `EXTRA` line; per-arm
  registration and the per-arm A_pipeline / msRRHO / judge chain; the step-05 example draws
  the production files, and its leftover ladder example is marked "an S0 scan row
  (history)"; the "The Replay." paragraph and the smoke sentence carry the S0 label / the
  both-arms note.
- `docs/hessian_learning_campaign.md` §7: the round-1 arms paragraph (two draws, one seed,
  `config_weight` 1 / 10), the two draw commands, the two-arm launch (one job per arm,
  submitted together) and the per-arm downstream chain.
- `docs/tutorials/T04` / `T05` source cells: production claims to the two-arm standard with
  the mace-docs credit; the S0 ladder (the R0–R4 rows and fields, `REPLAY_R4_FRAMES`, the
  coverage table, the §5 demo cell) labelled as history / "not the production Replay"; the
  §7 recipe and §8 (mechanics row, A5, A7) follow; stored outputs untouched (the notebooks
  are not re-executed here).
- the workflow step (`05_train.py` header example), the Slurm wrapper (`hl_train.slurm`: the
  production-arms example replaces the scan-row example; the `EXTRA` comment), the draw tool
  (`s0_spice_pt_draw.py`: docstring, examples, `--n` / `--seed` / `--weight` help) and the
  package driver (`run.py`: the `REPLAY_PER_HESSIAN_FRAME` docstring note, the Record note
  and the schema field description now say the S0 scan rows).
- in pass (same rule, surfaced by the scan): the examples index's T05 row ("R4 the one that
  runs" -> the two-arm standard) and the registry-name fixture in `t_engine_identity` moved
  off the S0 run name.

**The scan.** Sources-only over both repos (`.scratch` and `.mem/` (the S0-era narrative),
`_superseded`, `_backup`, data files and notebook stored outputs out of scope; `.ipynb` cells'
sources only) for `110960` /
`110,960` / `4 x N_TRAIN_HESSIAN` / `REPLAY_R4_FRAMES` / `R4` / `r4`: **53 hits, all
labelled S0 history or code identifiers** — S0-ladder phrases ("the S0 scan rows R0–R4 were
defined by", "the S0 ladder, history"), the build print and schema descriptions 09b's slice
labels "historical; not the production Replay" (09b's commit rides its own ticket; its edits
are in this worktree), `r4` variables (probe/ring/SMARTS) and 09b's
negative assertions. Zero hits read "the production Replay = R4"; `110960` has zero live
hits (the tracker's history and the provenance note keep it by design).

**Checks.** `py_compile` on the four edited `.py`; `bash -n` on `hl_train.slurm`; both
notebooks parse as JSON; the document-reading unit tests green (`t_hl_campaign`,
`t_dataset_cli`, `t_spice_pt_draw` 19/19, `t_frame_labels`, `t_script_taxonomy`,
`t_engine_identity`); full suites: openQHA `--all` **73/73** (SUITE_RC=0) and the package
`--all` **9/9** (SUITE_RC=0) — logs `/tmp/oqt09c_suite.log` + `/tmp/oqt09c_pkg_suite.log`
in the session's WSL.

**Judgment calls (recorded).** (1) The launch blocks carry round 1's re-selected flags on
`EXTRA` (spec story 14 / Q5: "the production launch lines carry the re-selected flags") —
without them the documented launch would train the schedule the spec rules out. (2)
Registration moved onto the run (`--register --register-copy` in the same `EXTRA`): a
post-training `05_train --register-copy` re-invocation would train again. (3) The draw tool
and the examples index were treated as live operator text (same rule; both carried the old
ladder language). (4) Stored notebook outputs keep the old text by design.

**Not verified.** Nothing runnable changed (text only; `py_compile` / `bash -n` / the suites
stand in). The Tianhe side stays to be executed — 09d's two draws, the timing job and the
two production jobs; 09b's slice carries the build print and schema in this worktree.

## Review record (2026-09-30, annotations -- the two-axis review of `5685aaf` / `411abe1`)

Two read-only sub-agents over the commit pair, per the `code-review` skill. Spec axis: the
requirements are present (the three boxes; every live site of the old statement re-stated; the
launch / registration / EXTRA story matches story 14 / Q5; stored outputs byte-identical; the
scan recount agrees). Standards axis: the EOL conventions hold (`.scratch` CRLF, deployed LF,
the fixture's stored CRLF) and the added live lines carry no stripped citations. Findings
applied in this commit:

- **Register form (hard).** The six live sites credited the ruling as "user ruling 2026-09-30";
  the tree's kept form is `(user, <date>)` (the ADRs, `hpc/`, scripts). All six now read
  `; user, 2026-09-30` — README, campaign page, T04 intro, T05 intro / §5 / §7.
- **`CONTEXT.md` glossary (judgement).** The Replay entry still said "one seed one file for a
  whole campaign" — the exact claim the pass retires. It now reads "one seed for the whole
  campaign (the production draw is written twice, at the two weights)", and "the same draw is a
  different ratio".
- **Record scope (spec).** The scan's exclusion list now names `.mem/` (kept S0-era narrative)
  and the 09b boundary is explicit (its build-print / schema half commits on its own ticket).

Recorded, not changed: the `EXTRA` string is written out in the three launch surfaces (README,
`hl_train.slurm`, T05 §7) on purpose — each is a standalone copy-paste surface; the era names
"round 1" / "S0" enter live files as the effort's vocabulary (the campaign page already keeps
its round names); the package driver's text stays generic machinery (it names the S0 ratio, not
the production size — the two-arm standard lives with the campaign texts); the
`t_engine_identity` fixture rename is recorded in the Answer.
