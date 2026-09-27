# References sweep: docs, slurm drift, tutorials

Type: task
Status: open
Blocked by: 04, 05, 06, 11
Part of: [hessian-learn-framework](../map.md)

## Question / work

The mechanical sweep of everything that still names the old arrangement; the content decisions come from 04/05/06/07, this ticket lands them.

- `docs/tianhe_install.md` (fork note 284–291; install lines 277/331); openQHA `README.md` "The mace fork" (202–219); `check_dependency.py`; `environment-cuda.yml:160` stale comment; `docs/hessian_learning_campaign.md`; `workflows/hessian_learning/README.md` (its `RETRY_FAILED` line 77 is already stale — decide whether that belongs here or in the campaign page's own upkeep).
- Tutorials `T04`/`T05`: module imports and the fork-commit assertions (mapping from [Identity after the split](06-identity-after-the-split.md)).
- The three stale Slurm scripts (`hpc/slurm/branchA_debug.slurm:58–60`, `branchA_deimos.slurm:73–74`, `branchB_traj_tianhe_a.slurm:108–109`) reading `prov["sha256"]` keys that no longer exist — a KeyError today.
- `.scratch/hessian-learning-set` references that would mislead a reader (ticket 10/36 texts): a pointer note, not a rewrite — old tickets are history.
- Reported by [Round-1 xyz](08-round-1-xyz.md) when its CLI fix landed (2026-09-26):
  - `workflows/hessian_learning/README.md` step-04 section: "frame, the default (production)" / "molecule (the smoke set)" are backwards since S0-C-65 (and the "(0.1)" fraction claims).
  - `hpc/slurm/hl_labels.slurm` tail (task 0): the `04_dataset.py` call's exit code is never captured (`exit "$rc"` is the assemble's), so a failed Dataset build exits 0 silently; the call also leans on the CLI's `--split-by` default (correct now, but implicit).
  - `openqha/data/dataset.py:640`: the refusal message says "Pass resplit=True (04_dataset.py --resplit)" — the workflow deliberately exposes no such flag (ruling 2026-09-26); keep the substring "resplit" (`tests/unit/t_dataset.py:395` pins it) while re-wording.

## Answer

<!-- resolver: append what was swept + evidence; set Status: resolved; add a line to the map's Decisions so far -->
