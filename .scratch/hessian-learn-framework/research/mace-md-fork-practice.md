# The mace-md fork practice (jharrymoore/mace@softcore) — base, changes, upkeep, upstream status; implications for `BloomDlwlrma/mace@openqha-hessian`

Type: research — feeds ticket 10 ("Upstream: submit the mace-side changes")
Date: 2026-09-29
Method: primary sources only — the local mace-md copies and mace-related files in this workspace; GitHub REST/raw endpoints for `jharrymoore/mace`, `ACEsuit/mace`, `jharrymoore/openmm-ml@ml_alchemy`, `jharrymoore/MACE-OFF23-SC`, and `BloomDlwlrma/mace`; upstream `CONTRIBUTING.md` on both `main` and `develop`. All web reads on 2026-09-29. No repository edits.

**Outcome (2026-09-29): not submitted — see [ticket 10](../decisions/10-upstream-pr.md) / [ADR 0013](../../../docs/adr/0013-permanent-private-mace-fork.md).**

## 1. mace-md and how it references its mace fork

**Copies present locally** (all four are `mace-md-master/`; byte-identical per the 27-file SHA-256 check recorded in [L3]-prior-note, not re-hashed here):
1. `source-code/mace-md-master/` (canonical) [L1];
2. `source-code/ref-papers/mace-training/mace-md-master/`;
3. `lambda-qm9-reaction-deltan_0-mace_v1/stage2-lambda-delta0-mace-learning/refs/source-code/mace-md-master/`;
4. `lambda-qm9-reaction-deltan_0-mace_v1/related-papers/workflow-design/mlps/mace-md-master/`.

Also present: `source-code/MACE-OFF23-SC-main/` = `README.md`, `LICENCE.md`, `MACE-OFF23-SC_swa.model` [L2] — the **model** repo, not a mace fork ("This repository contains the MACE-OFF23 potential as described in our recent publication (arXiv:2405.18171)" [M1]). **No copy of the author's mace fork exists anywhere in the workspace**: the only mace source tree is our own `mace/` checkout (`**/mace/mace/__init__.py` matches it alone).

**The fork pins (verbatim).** `mace-install.sh` (whole file) [L1]:

```bash
#!/bin/bash -l
mamba env create -f mace-openmm.yml

conda activate mace-openmm

pip install git+https://github.com/jharrymoore/mace.git@softcore
pip install git+https://github.com/jharrymoore/openmm-ml.git@ml_alchemy
pip install .
```

`mace-openmm-full.yml:19-24` [L1] — the only yml with a pip section; `mace-openmm.yml` (the file CI installs) has none:

```yaml
  - pip
  - pip:
    - mpi4py
    - git+https://github.com/jharrymoore/mace.git@softcore
    - git+https://github.com/jharrymoore/openmm-ml.git@ml_alchemy
    - git+https://github.com/jharrymoore/mace-md.git
```

`README.md:8`: "The base conda environment can be build from `mace-openmm.yml`, additional requirements can be found in `mace-install.sh`" (sic). The README's only other mace statement: "The pretrained `MACE-OFF23-SC` potential is provided under the ASL and is available [here](https://github.com/jharrymoore/MACE-OFF23-SC)". **No version of mace is stated anywhere**; the pin is a bare branch name, no commit, no tag. `setup.py` declares no `install_requires` and nothing about mace at all.

**CI never exercises the fork**: `.github/workflows/python-app.yml` installs only `mace-openmm.yml` (which has no mace entry) and runs `mace-md -h` [L1].

**Code that reaches into mace / documents the fork.** `mace_md/hybrid_md.py:7` `from mace.calculators import MACECalculator`; `mace_md/cli/entrypoint.py:2` `from mace import tools` [L1]. Neither the README nor any comment says why the fork is needed; the alchemical path flows through the author's **openmm-ml** fork: `hybrid_md.py:67` `from openmmml.models.macepotential import MACEPotentialImplFactory`; `mace_md/nnp_repex/utils.py:151` `class NNPAlchemicalState(AlchemicalState)` "for perturbing the `lambda_interpolate` value" [L1]. The call sites of the mace fork's actual API are in that openmm-ml fork (fetched raw; not present locally): `openmmml/models/macepotential.py` @ `ml_alchemy` has `addForces(..., decouple_indices: Optional[torch.Tensor] = None, ...)`, reads the OpenMM global parameter `lambda_interpolate`, and calls `self.model(self.inputDict, compute_force=False, decouple_indices=self.decouple_indices, lmbda=lambda_interpolate,)[self.returnEnergyType]`; it also adds `force.addGlobalParameter("lambda_interpolate", 1.0)` with the dhdl wiring left commented: `# enable calculation of dhdl` / `# force.addEnergyParameterDerivative("lambda_interpolate")` [J7]. That is exactly the keyword signature the `softcore` mace branch adds and that upstream `ScaleShiftMACE.forward` does not accept.

## 2. The fork itself — base, changes, upkeep

**Repo/branches.** `jharrymoore/mace`, a GitHub fork of `ACEsuit/mace` (created 2022-09-25; `default_branch: main`; `pushed_at: 2025-01-21T12:11:10Z` as of 2026-09-29) [J1]. Branches (16) [J2]: `QEq`, `ase_mlmm`, `coulomb`, `cuda-graphs`, `distrib_local`, `foundations_maceoff`, `intel`, `main`, `multi-GPU`, `multi-GPU-intel`, `openmm-harry`, `openmm-prod`, `refactor_data_jhm_merge`, **`softcore`**, `torchscript_merge_openmm`, `xpu-ddp-train`. So `softcore` is one of several long-lived feature branches (OpenMM/device lines included); only `softcore` is referenced by mace-md's install pins. The fork's own `main` is stale: tip `ff054d57…` is upstream's "Merge pull request #190 … fix table of contents readme", committed 2023-10-11 [J6].

**HEAD and activity.** `softcore` HEAD = `33ab71a01287f1f7c1638f9a5bf2a7edd77faed7` (2024-05-01T11:40:00Z), message "Merge branch 'softcore' of github.com:jharrymoore/mace into softcore" [J3]. That merge is a self-sync of his own two clones — the branch has had **no upstream sync ever after its base**. The repository's last push (2025-01-21) was for a different branch (`refactor_data_jhm_merge`, the head of PR #793) [J1][O7]; `softcore` has not moved since 2024-05-01.

**The base.** Compare `88d49f9…softcore` (in-fork): `merge_base_commit` = `88d49f9ed6925dec07d1777043a36e1fe4872ff3`; status `ahead`, `ahead_by: 5`, `behind_by: 0` [J4]. `88d49f9` = "Merge pull request #306 from ACEsuit/develop — Warning on float64 training with PyTorch 2.1", 2024-01-24T11:38:54Z, and it exists in `ACEsuit/mace` (commits API returns it; 6-file README-only delta) [O3]. Compare `v0.3.4…88d49f9` upstream: status `ahead`, `ahead_by: 3` (and `88d49f9…v0.3.4`: `behind_by: 3`) — i.e. **the base is exactly tag v0.3.4 (`6df88277a2971a819b1d6177e9acbd7dc76b7c54`, 2024-01-16, "Merge pull request #296 … Change stress input + update version") plus 3 commits** (`c110e88` warning; `480f5118` PR #305 merge; `88d49f9` PR #306 merge) [O4]. It corresponds to **no release tag**: it is a develop→main merge cut two weeks after v0.3.4, on main. The next release is 4 months later — v0.3.5 = `dee204f1f9d587f28fd792fdad1f45039ef71e94`, 2024-05-17, "Merge pull request #428 from ACEsuit/develop — fix bug test set stress key", cut directly after `1efd4887` (= PR #426, 2024-05-15) on the same main line [O5]. **Commits above the base: 5** (4 authored + 1 merge) [J4].

**The change shape** (net base→HEAD, compare API [J4]) — 4 files, +135/−7:

| file | +/− | what |
|---|---|---|
| `mace/modules/models.py` | +31/−2 | `ScaleShiftMACE.forward` gains `decouple_indices: Optional[torch.Tensor]`, `lmbda: Optional[torch.Tensor]`; `lmbda.requires_grad_(True)`; an "alchemical mask" (edge crosses above/below `max(decouple_indices)`) scales `edge_attrs`/`edge_feats` by `lmbda`; `get_outputs(...)` is called as a 4-tuple with `lmbda` and `training=True`; the output dict writes `"dhdl": dhdl`. Comments: "we want a mask into the edge indices for those pairs with an atom on each side of the decoupling plane" and "TODO: this method assumes we have contiguous atom indices in the alchemical region, and this comes first". A second, inert `# "dHdL": dhdl,` line is added to the other model's output dict. |
| `mace/modules/utils.py` | +35/−4 | new `compute_forces_dhdl(energy, positions, lmbda, training)` (returns `-forces, dhdl`); `get_outputs()` gains `lmbda`, `compute_dhdl` and returns `(forces, virials, stress, dhdl)`; when dhdl is computed via `compute_forces_dhdl`, virials/stress are `None`. |
| `mace/calculators/mace.py` | +10/−1 | `MACECalculator.__init__(..., decouple_indices=None, lmbda=None)`, stored and passed to each `model(...)` call in `calculate()`. |
| `tests/test_models.py` | +59/−0 | `test_alchemical_mace`: builds a `ScaleShiftMACE`, runs eager vs `jit.compile`, asserts equal energies and `dhdl` for `decouple_indices=torch.tensor([0])`, `lmbda=torch.tensor(0.5)`. |

Commit messages: `db46e2a` "add softcore scaling" (2024-04-06), `508cc05` "add alchemical softcore mace" (2024-04-06), `473fefc` "remove dhdl calculation" (2024-04-30), `f7306b8` "fixes for dhdl calculation" (2024-05-01; re-introduces dhdl via the new `compute_forces_dhdl` path), `33ab71a` self-merge [J3][J5]. It is a **narrow in-place patch to existing core files plus one test — no new module, no flag, no dependency**.

**Upkeep.** Base lag when the work started: ~2.5 months (2024-01-24 → 2024-04-06). Branch frozen since 2024-05-01 — no rebase, no upstream merge, no tag; while upstream released v0.3.5 (2024-05-17) through v0.3.16 (2026-05-10) [O2]. Consumers pin the branch name (`@softcore`) and get whatever it last was; nothing in mace-md records a commit (the prior note's install analysis applies: non-editable URL installs leave no `.git`).

## 3. Upstream relationship

- Search `repo:ACEsuit/mace author:jharrymoore`: 8 results, all PRs; **none is the softcore/alchemical change** [O6]. Merged: #62 "openmm-interop compatibility changes" (2022-12-29; body: "write back the interaction energy from the scale shift mace model"), #340 "Add Intel XPU device" (2024-03-11), #420 "log errors and handle checkpoint io on rank 0 only" (2024-05-14), #1225 "update mace-openmm docs" (opened 2025-10-15, merged 2025-10-20 by ilyes319). Closed unmerged: #212 "Multi gpu", #319/#320 "Add finetuning for maceoff models", #793 "changes to mace preprocess argparse to support configfile". On all, `author_association: COLLABORATOR`.
- Push provenance: PR #793's head is `jharrymoore:refactor_data_jhm_merge` (from the fork; base `ACEsuit:refactor_data`) [O7]; PR #1225's head is `ACEsuit:jhm/update-openmm-docs` (pushed into the upstream repo itself, merged 2025-10-20; 1 commit, +57/−105, 1 file) [O7]. So the fork is where his experiments live; accepted contributions go upstream as ordinary small PRs, sometimes bypassing the fork entirely.
- Upstream code search (`ACEsuit/mace`): `softcore` → no matches; `decouple_indices` → none; `dhdl` → none; `lmbda` → none (control query `mace_mp` returns matches) [O9]. As indexed on the default branch, **none of the softcore patch has landed upstream in any form**, and there is no PR proposing it.
- The public companion is the model repo `MACE-OFF23-SC` (README cites arXiv:2405.18171; ASL license; weights only) [M1]. The mace fork itself carries no README/doc/issue explaining the feature — its only "documentation" is the code and the model repo link.

## 4. Comparison with our fork

Ours (server-side, 2026-09-29) [B1]: `BloomDlwlrma/mace@openqha-hessian` = `4d2da09` (upstream v0.3.16, itself "Merge pull request #1468 from ACEsuit/develop", 2026-05-10) → **D** `5c2d7612eed88dc1463b5588a79c2d5f5718d322` ("base: drop the three bundled foundation-model binaries (fork-only) … never submit upstream") → **6 commits** → tip `1110ffbafd651a74d1d4678deb4748056d1dff0a` (7 commits above the tag; compare `ahead_by: 7`). Base→branch = 11 files, +869/−17; measured against upstream (including D's three `.model` removals) = 14 files, +869/−17. The six [B1][L3-02b]:

| # | new sha | subject (short) |
|---|---|---|
| 1 | `a31d0a6` | version `0.3.16+openqha` + `.gitignore` cleanup — fork-only, ticket 10 drops it |
| 2 | `1b53032` | "commit A": per-structure Hessian label — `--hessian_key` (REF_hessian) → `Configuration.properties["hessian"]` → `AtomicData.hessian/has_hessian/hessian_weight/sqrt_masses` |
| 3 | `61582b0` | "commit B": external loss hook — `--loss external --loss_module module:factory`; `evaluate` asks for the full Hessian when the loss declares `wants_hessian_at_eval` |
| 4 | `cce52c0` | "commit C": multihead finetuning keeps `--loss external`; `evaluate` puts the loss in eval mode, keeps the force graph on request, merges `eval_summary()` into logged metrics |
| 5 | `66a68f0` | "commit D": `--valid_probes_key` + dataset's fixed probes ride into the batch; removes `--hessian_mode_weighting` and the `modes` probe choice (now hard parse errors) |
| 6 | `1110ffb` | probe default becomes PHL's standard normal (`gaussian`); `_validated_probes` checks shape/finiteness only |

Files touched vs upstream's current names for the shared surface (§5): `mace/data/atomic_data.py`, `mace/data/utils.py`, `mace/tools/arg_parser.py`, `mace/tools/train.py` are on upstream's literal "core and shared" list; also `mace/cli/run_train.py`, `mace/tools/scripts_utils.py`, `mace/tools/default_keys.py`, `mace/__version__.py`, and three new test files (`tests/test_hessian_label.py`, `tests/test_external_loss.py`, `tests/test_valid_probes.py` — flat files, not "their own directory"). The commit messages assert repeatedly "default path unchanged" [B1].

Side by side:

| | mace-md (jharrymoore) | ours (BloomDlwlrma) |
|---|---|---|
| base | post-v0.3.4 `main` state, **not a tag** | upstream tag **v0.3.16** + fork-only D |
| branch | `softcore`, pinned by URL in install files | `openqha-hessian`, pinned by URL in `install.sh` |
| net size | 4 files, +135/−7, 1 test | 11 files (14 w/ D), +869/−17, 3 test files |
| core/shared touched | `models.py`, `modules/utils.py`, `calculators/mace.py` — 3 of upstream's named shared files | `data/atomic_data.py`, `tools/train.py`, `tools/arg_parser.py` + others |
| offering upstream | never proposed | ticket 10's job (open) |
| upkeep | none since 2024-05-01; base never rebased | rebuilt onto v0.3.16 on 2026-09-27; designed to re-base per adopted release |
| fork-only commits | none declared | D + version bump (to drop/mark in the PR) |
| consumer provenance | pure branch pin; no version marker | `0.3.16+openqha` + `mace_fork_info()`-checked editable clone |

## 5. What this implies for our modifications and the ticket-10 route (release-focus constraint)

**Facts to stand on.**
1. mace-md's way is: branch off the **released line**, keep the change narrow, pin the branch name in install files, never rebase, never upstream — and it has held for ~2.5 years only because the branch was simply frozen (and the pin has no fallback if it moves).
2. Upstream's routing rules, as fetched today and **the same text on `main` and `develop`** [O8]: "**main** carries releases. A release is cut by merging `develop` into `main` and releasing from there. **develop** is where v0.3 is developed"; "Bug fixes are always allowed … a fix may touch a shared file. That is the difference between fixing behaviour and adding it. **Fix the bug where it lives**"; "New features: **v1 first** … You may also need the feature in a released version before v1 ships. When v1 genuinely is not an option, v0.3 accepts the feature if it meets both conditions" — "**1. It is self-contained in its own directory.** … **2. It is tested on numbers, not on absence of errors.**" (tests "assert values … run on a committed, fixed input … live in their own directory under `tests/`"); "**Core and shared changes go straight to v1**: Changes to core behaviour, or to files that everything else depends on, are not accepted against v0.3", with the shared surface "in practice" listed as `mace/modules/models.py`, `blocks.py`, `symmetric_contraction.py`, `mace/modules/utils.py`, `wrapper_ops.py`, `mace/data/atomic_data.py`, `mace/tools/arg_parser.py`, `mace/tools/train.py`, `mace/calculators/mace.py`. The routing table row: "Change core behaviour, or touch shared files | **v1 only** | Not accepted against v0.3". And: "A fix reaches users at the next release cut … If you need a v0.3 fix sooner than that … say so on the issue rather than assuming either way."
3. The softcore fork's whole patch edits exactly three of those named shared files — and was never offered; the author's accepted upstream work is instead small, non-alchemical PRs (#62, #340, #420, #1225). No evidence anywhere of an alchemy-feature PR being attempted.
4. Our six commits edit 4 of the named shared files (data layer, train loop, arg parser) and add flag-level behaviour changes (new flags; two old flags now fail at parse time) [B1]. Ticket 10 itself frames part of the work as a bug ("multihead mode drops `--loss external`") and lists the PR set as "the four feature commits A–D plus their tests", dropping the version bump [L3-10].

**Inferences / recommendations (marked as inference).**
- *Inference:* under these rules a PR against `main` is the wrong target (main only receives develop cuts); the right target is `develop`. The release-focus constraint is satisfied the same way mace-md did it and we already do it — the *fork* branches off the released tag; the *PR* goes to develop and reaches users at the next cut. (Our branch being tag-anchored is the stronger version of mace-md's base: theirs is not reproducible by tag at all.)
- *Inference:* the commit set must be split by category before writing the PR text: (a) the multihead `--loss external` change reads as a **bug fix** ("Fix the bug where it lives" — fixes are always accepted, even in shared files; its observable effect, however, presupposes commit B's hook); (b) the Hessian label / external-loss hook / probes are **features** — the rule demands v1-first, and when v0.3 is claimed necessary, self-contained directories + pinned-number tests + a justification. Our tests assert values on in-test seeded random data (not committed fixed inputs) and, for the external-loss path, on behaviour/log output; they sit as flat `tests/test_*.py` files, not "in their own directory"; the commits keep the default paths unchanged (good for the "blast radius" test), but the code edits sit in shared files. Expect pushback of the "v1 only" kind, or plan to negotiate the v0.3 exemption via the "needed in a released version before v1 ships" clause.
- *Inference:* either way, the fork remains the delivery vehicle for this project until a release carries the changes — the mace-md case is precisely the "pin until released" hedge, and our install (editable clone for `mace_fork_info()`) already depends on it. So ticket 10's alternative acceptance — "a written decision (with the user) not to submit" — has a fully precedented fallback (mace-md has shipped this way since 2024), at the known costs: base rot (their patch, four months old at freeze time, is now untargetable against a released tree), no review, and an unpinned moving-branch ref for consumers.
- *Inference:* if we do submit, do it soon after a re-base onto a fresh release tag; the further the fork ages, the less the PR resembles mace-md's cheap option and the more it becomes a re-derivation (the softcore branch is the exhibit).

## 6. Unverified / cautions

- **v0.3.5 containment of the base**: not directly verified — the two compare queries spanning base→v0.3.5 exceed what this session's page fetcher extracts, and the HTML compare page rendered unreliable pagination. Verified facts: base = v0.3.4 + 3 (both directions), and v0.3.5 (2024-05-17) is a later develop→main cut on the same line. "The base's content ships in v0.3.5" is therefore an inference, not an observation.
- **Code search caveat**: the zero-hit results for `softcore`/`decouple_indices`/`dhdl`/`lmbda` are from a tokenless GitHub code search over the indexed ref (default branch `develop`); they are evidence of absence there, not a proof across every branch or history.
- **openmm-ml call chain**: evidenced from one raw fetch of `ml_alchemy`; I did not test at runtime that the mace fork is load-bearing, though the quoted call signature (`decouple_indices=…, lmbda=…`) would fail against upstream `ScaleShiftMACE.forward` (which lacks both kwargs) — that failure is an inference from the quoted code.
- **The four mace-md copies** are stated byte-identical on the prior note's 27-file SHA-256 check; not re-hashed in this pass.
- **CONTRIBUTING main vs develop**: both fetched; the sections compared read identically; no byte-wise diff performed, so "no differences" is a spot-comparison result.
- **Per-branch recency** of the fork's other branches (`openmm-prod`, `openmm-harry`, …) was not measured; only repo-wide `pushed_at` (2025-01-21, attributable to the `refactor_data_jhm_merge` push for PR #793 by time correlation).
- **mace-md CI** runs `mace-md -h` against an environment file containing no mace — evidently stale/broken; not investigated further (it does confirm the fork is not CI-tested).
- The `473fefc` → `f7306b8` dhdl flip-flop is described by its net end state; the intent behind "remove" then "fixes" is not documented beyond the commit messages.

## Sources

Web (all fetched 2026-09-29):
- [J1] `https://api.github.com/repos/jharrymoore/mace` — fork metadata (`pushed_at` 2025-01-21T12:11:10Z, `default_branch: main`, created 2022-09-25).
- [J2] `https://api.github.com/repos/jharrymoore/mace/branches?per_page=100` — 16 branches incl. `softcore` @ `33ab71a`.
- [J3] `https://api.github.com/repos/jharrymoore/mace/commits?sha=softcore&per_page=30` — HEAD + date + subjects; base's parent chain.
- [J4] `https://api.github.com/repos/jharrymoore/mace/compare/88d49f9ed6925dec07d1777043a36e1fe4872ff3...softcore` — `ahead_by: 5`; files `models.py` +31/−2, `utils.py` +35/−4, `calculators/mace.py` +10/−1, `tests/test_models.py` +59/−0; key diffs.
- [J5] per-commit: `.../commits/{db46e2a19563c02aa73b0d7208c015ee45578cc2, 508cc05ac490a52bb040a906786658d4a7d4c723, 473fefc6b1bfc935cb61e237becf18d0b64e86b2, f7306b88fe203499559f5babaa8f1fb94baaaa6c}` — messages, stats, patches.
- [J6] `https://api.github.com/repos/jharrymoore/mace/commits/ff054d572c259ecd046bd1e40becd66711254233` — fork `main` tip = upstream commit of 2023-10-11.
- [J7] `https://raw.githubusercontent.com/jharrymoore/openmm-ml/ml_alchemy/openmmml/models/macepotential.py` — `decouple_indices`/`lmbda` call site; `lambda_interpolate` global parameter; commented dhdl wiring.
- [M1] `https://raw.githubusercontent.com/jharrymoore/MACE-OFF23-SC/main/README.md` — model repo, ASL, arXiv:2405.18171.
- [O1] `https://api.github.com/repos/ACEsuit/mace/tags?per_page=30` — v0.3.16 latest (`4d2da09`); v0.3.5 = `dee204f1`; v0.3.4 = `6df88277`.
- [O2] `https://api.github.com/repos/ACEsuit/mace/releases?per_page=5` — v0.3.16 published 2026-05-10T18:06:35Z; release dates/notes.
- [O3] `https://api.github.com/repos/ACEsuit/mace/commits/88d49f9ed6925dec07d1777043a36e1fe4872ff3` — base exists upstream; 2024-01-24; PR #306 develop→main.
- [O4] `https://api.github.com/repos/ACEsuit/mace/compare/v0.3.4...88d49f9ed6925dec07d1777043a36e1fe4872ff3` (status `ahead`, 3) and `.../compare/88d49f9...v0.3.4` (status `behind`, 3).
- [O5] `https://api.github.com/repos/ACEsuit/mace/commits/dee204f1f9d587f28fd792fdad1f45039ef71e94` (v0.3.5, 2024-05-17) and `.../commits/1efd488730a9b1861fce0e5c0ac2d587caecc366` (PR #426, 2024-05-15).
- [O6] `https://api.github.com/search/issues?q=repo%3AACEsuit%2Fmace%20author%3Ajharrymoore` — 8 PRs, statuses, bodies.
- [O7] `https://api.github.com/repos/ACEsuit/mace/pulls/793` (head `jharrymoore:refactor_data_jhm_merge`, unmerged) and `/pulls/1225` (head `ACEsuit:jhm/update-openmm-docs`, merged 2025-10-20).
- [O8] `https://raw.githubusercontent.com/ACEsuit/mace/develop/CONTRIBUTING.md` and `https://raw.githubusercontent.com/ACEsuit/mace/main/CONTRIBUTING.md` — v1-rewrite routing text (quotes in §5).
- [O9] GitHub code search via tool, scope `ACEsuit/mace`: `softcore` / `decouple_indices` / `dhdl` / `lmbda` = no matches; `mace_mp` = matches (control).
- [B1] `https://api.github.com/repos/BloomDlwlrma/mace/commits?sha=openqha-hessian&per_page=10` and `.../compare/4d2da09413ac1407f37cdbb6b81fa28e4c15655e...openqha-hessian` — the 7 commits, subjects, 14 files, +869/−17.

Local:
- [L1] `source-code/mace-md-master/`: `mace-install.sh`, `mace-openmm.yml`, `mace-openmm-full.yml`, `README.md`, `setup.py`, `.github/workflows/python-app.yml`, `mace_md/hybrid_md.py`, `mace_md/cli/entrypoint.py`, `mace_md/nnp_repex/utils.py`.
- [L2] the three further mace-md copy paths (see §1) and `source-code/MACE-OFF23-SC-main/`.
- [L3] prior work in this repo: `openQHA/.scratch/hessian-learn-framework/research/mace-md-install-pattern.md` (install mechanics, 27-file identity check); `decisions/10-upstream-pr.md` (ticket text, commit-set pruning); `decisions/02-the-real-fork.md`; `implementation/02a-the-mace-fork.md`, `02b-the-rebuild.md` (six-commit table, +869/−17), `02d-clone-verify-rehome.md` (branch/tag SHAs, install evidence).
