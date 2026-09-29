# AGENTS.md — openQHA

## Scope rule: complete the ticket's scope; report anything beyond it (user ruling, 2026-09-25)

Finish the work that the current spec or ticket asks for. Do not stop that work because you
found something outside its scope.

If a change beyond the spec's scope looks necessary, do NOT make it directly. Complete the
requested in-scope work and report the extra change at the same time:

- what the extra change is;
- which ticket/spec line it comes from, if any;
- the options, and your recommendation.

The user decides on the extra change.

继续完成当前 spec 和 tickets 范围内的内容；对于超过 spec 范围的改动，在完成当前要求的内容同时停下来汇报，不要直接进行修改。

## Where work is tracked

Local markdown tickets (the tracker is described at the workspace root,
`docs/agents/issue-tracker.md`). Single-workflow features: `.scratch/<feature>/issues/<NN>-<slug>.md`.
Wayfinder efforts: `.scratch/<effort>/map.md` with the map's tickets in `decisions/`, derived
execution slices in `implementation/`, findings in `research/`, and `archive/` for superseded
tickets (ADR `docs/adr/0009-wayfinder-ticket-layout.md`). One ticket per file. A ticket is
done only when its acceptance criteria are paid and its Status line carries the evidence
(tests run, what was not verified).

## Environment and tests

- The repository runs under WSL (Ubuntu 24.04); anaconda is at `/home/ubuntu/anaconda3`.
- The full environment is conda env `openqha` (procrustes, the `0.3.16+openqha` mace fork,
  nbconvert).
- The mace fork's refs (`openqha-hessian` @ `1110ffb`, `base-v0.3.16` @ `5c2d761`) are frozen — no rebase, no force-push, no rewrite (ADR 0013).
- From the repository root, inside WSL with the `openqha` env active:
  - `python tests/run_tests.py` — the unit group (fast; the default);
  - `python tests/run_tests.py --all` — adds the integration group;
  - `tests/integration/t_mace_engine_folder.py` — the one MACE integration test.
- Report exactly: files changed, checks run, what was not verified.

## Submitting jobs

Submit with the default environment and the job's own variables as command prefixes --
`TAG=draw300 sbatch hpc/slurm/hl_branchA.slurm`. Never a restricted `--export` list:
it carries only what it names, and the identity variables it drops (USER, LOGNAME) are
what a compute node whose passwd map cannot resolve the uid then needs from the
environment -- job 7675703 (2026-09-25) died inside torch's import (getpass.getuser →
pwd.getpwuid) before any chemistry. `--export=ALL` is not written out either; it is the
default. A job builds its operative environment from the checkout (`hpc/env/common.sh`
→ `hpc/env/tianhe.sh`); the submitting shell supplies the variables named on the
command line.
(user ruling, 2026-09-26; replaces the explicit-list rule of 2026-09-25)

## Vocabulary

`CONTEXT.md` in this repository is the ubiquitous language (Basin, Frame, Record, Property
file, frequency floor, inversion window, ...). Code, comments and records use its terms and
avoid its `_Avoid_` list.

## Records are immutable

Historic Records stay readable as data. A change to a rule or to a record field is dated in
the ticket that made it, and old records are never rewritten.
