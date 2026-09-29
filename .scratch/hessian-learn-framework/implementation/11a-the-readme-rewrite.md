# 11a: The README rewrite — clean register, no History

Type: task
Status: open
Serves: 11
Blocked by: None
Part of: [hessian-learn-framework](../map.md)

> Execution slice of [The repo swap](../decisions/11-repo-swap.md) per
> [spec-repo-swap.md](../spec-repo-swap.md): decision 6 (the rewrite) and decision 7
> (the wording rule); stories 2-8, 13. The draft is public text: it is shown to the
> human before it is committed. Wayfinder conventions apply: the `Status` protocol, no
> triage labels (`docs/agents/issue-tracker.md`, ADR 0009).

## What to build

The package repository's `README.md`, rewritten clean in the register of hip and the
PHL source repository (references extracted at
`source-code/final-workflow-design/hessian-train/PHL-main/`), committed on the local
`main` on top of `ff92141` so that the swap's first push (11c) already carries the
finished text:

- title; one short paragraph: the training side of openQHA — not a MACE fork;
- the index links — openQHA `https://github.com/BloomDlwlrma/openQHA`, mace
  `https://github.com/BloomDlwlrma/mace` (branch `openqha-hessian`);
- Install — both `install.sh` modes, the run-it-last and eval-wheel notes, tightened;
- Verify — the runner, `--all`, the provenance header;
- Citation — the PHL paper (bibtex key `rodriguez2026projectedhessianlearningfast`,
  arXiv 2603.04523) and the openQHA software entry (`@misc{openqha}`, per openQHA's
  `docs/cite/`).

No History section, no old hashes or mapping talk, no badge. The README does not
discuss the predecessor at all; externally it is only "an early MACE fork experiment".

## Acceptance

- [ ] The draft is shown to the human before the commit (public text)
- [ ] The README carries every element above and none of the forbidden ones; both index links and both citation entries are exact
- [ ] One commit on the package repo's `main`; the diff touches `README.md` only — `install.sh`, `pyproject.toml`, `openqha_hessian/`, `tests/`, `.gitignore` unchanged (decision 13)
- [ ] Reported per the operating rule: files changed, checks run, anything not verified

## Answer

<!-- resolver: append the commit hash + the shown-to-human note; set Status: resolved; add a line to the map's Implementation -->
