# Mace's identity practice: what the shipped package records, what upstream `develop` does, and what pip records (PEP 610)

Serves [06-identity-after-the-split](../decisions/06-identity-after-the-split.md) — the "is this worry a false problem?" question. Primary sources only: the local `mace/` checkout (fork branch `openqha-hessian`), upstream `ACEsuit/mace` at `develop` (fetched 2026-09-28 from raw.githubusercontent.com), PEP 610, the Python `importlib.metadata` docs, and the live `.dist-info` artifacts of the WSL `openqha` env. Checked 2026-09-28. No repository code was modified.

## 1. The shipped stack (what actually runs here): a version string, and nothing readable

- Version is a static file: `mace/mace/__version__.py:1` → `__version__ = "0.3.16+openqha"` (upstream v0.3.16: `"0.3.16"`; the `+openqha` suffix is the fork's PEP 440 local version). Distribution version comes from `mace/setup.cfg:3` → `version = attr: mace.__version__`. No setuptools-scm (build deps: setuptools/wheel only).
- The only runtime git touch in the importable package: `mace/mace/tools/scripts_utils.py:215-224`, `print_git_commit()` — GitPython, `search_parent_directories=True`, every exception swallowed → returns the string `"None"`, logged at `debug` level only. Called from training (`mace/cli/run_train.py:132`); its value is written into the TorchScript compile block's `commit.txt` extra-file (`run_train.py:1086-1131`) — and that block is wrapped in `try/except Exception: pass`. Nothing reads `commit.txt` (`load_from_json` reads the key `config.json`; the write key is `config.yaml` — a mismatched pair with no in-tree caller).
- No dirty handling, no `.git` detection, no install-shape refusal anywhere: `run_train.py:126-129` only `logging.info`s the version. Grep over `mace/mace/**` for `subprocess|rev-parse|sha256|hashlib|dirty`: zero hits (digests exist only as unused fields in test reference JSONs).
- Weights: no hash at load or download; upstream's own sha256 pinning lives in its golden-test harness, never on a user load path.
- Saved model: a plain pickled `nn.Module` (`torch.save`), no metadata; the optional TorchScript side artifact carries `commit.txt` + `config.yaml`, unread.
- Net: the shipped package records nothing readable about its own code — a version string, plus a debug commit that lands unread inside an optional artifact.

## 2. Upstream `develop` (the current, work-in-progress answer, `tests/golden/`)

- `les_pin.py` — "Which `les` produced a number, and whether it is the one installed now." It keeps three commits apart: **pinned** (40-hex in `requirements/les.txt`), **installed** — "what pip actually put in site-packages, read from the `direct_url.json` a VCS install records (PEP 610)", and **reference** (the committed golden JSON). `installed_les_commit()` is literally `metadata.distribution("les").read_text("direct_url.json")` → `payload.get("vcs_info", {}).get("commit_id")`. `None` means "the question cannot be answered … a wheel from PyPI, a local editable checkout or a vendored copy all record no VCS provenance". The failure message instructs: "Install it the way CI does — `pip install -r requirements/les.txt` — rather than from a wheel or a local checkout."
- Why it exists (docstring): two `xfail`ed LES tests "do not reproduce against the pinned one"; "the information needed to tell 'the model changed' from 'the solver changed' was never written down, so neither test can say which happened, and both had to be abandoned rather than fixed."
- `train_anchor.py` — checkpoint provenance as a **sidecar**: the training argv is built explicitly and "recorded verbatim in the sidecar next to the checkpoint, so the recipe is a fact about the committed file rather than a paragraph that can drift away from it" (`.build.json` carries command/argv/seed/dtype/train_file/args).
- `feature_inventory.md` — v1 direction: checkpoints become "neutral-format (safetensors + manifest)"; `MACE_USE_CUEQ_CG` is DROPped because "an environment variable that silently changes model numerics is unreproducible and never lands in the run metadata".

Reading — mace's position where a number depends on code outside the artifact: record the code's commit, recover the **installed** commit from PEP 610 metadata, fail loudly when it cannot be established, and prescribe the install shape that makes it establishable. None of it reads `.git`, and none of it is runtime machinery — it lives in the test/golden tier.

## 3. What pip records — PEP 610 `direct_url.json` (verified against the spec)

| install shape | `direct_url.json` | commit recoverable from metadata? |
|---|---|---|
| `pip install git+https://…@BRANCH` (non-editable) | `vcs_info: {vcs, requested_revision?, commit_id}` — `commit_id` MUST be present | ✅ yes (install-time snapshot) |
| `pip install -e git+https://…@BRANCH` | `url` = local clone `file:` URL, `dir_info.editable: true`, **no `vcs_info`** (spec example, verbatim) | ❌ no — only in the clone's `.git` |
| `pip install -e /local/path` | `dir_info.editable: true`; VCS info MUST NOT be inferred (spec note) | ❌ no — only in the tree's `.git` |
| PyPI wheel (name + version) | no `direct_url.json` at all (spec: MUST NOT be created) | ❌ no — name/version only |

- Written by pip ≥ 20.1 (pip changelog 20.1b1: "pip now implements PEP 610").
- Install-time snapshot: no dirty field; not rewritten later (inferred from the spec's install-time framing, not a literal quote).
- Reading at runtime: `importlib.metadata.distribution(name).read_text("direct_url.json")` → `str | None` (Python ≥ 3.8; absent for name+version installs — handle `None`). Python 3.13 adds `dist.origin`.

## 4. Live artifacts — the two editable installs in the WSL `openqha` env (dumped 2026-09-28)

- `mace_torch-0.3.16+openqha.dist-info/direct_url.json`:
  `{"dir_info": {"editable": true}, "url": "file:///mnt/c/Users/10704/Documents/01_Free-Energy-alchemical/mace"}`
- `openqha_hessian-0.1.0.dist-info/direct_url.json`:
  `{"dir_info": {"editable": true}, "url": "file:///mnt/c/Users/10704/Documents/01_Free-Energy-alchemical/openQHA-Hessian"}`
- METADATA: `Name: mace-torch`, `Version: 0.3.16+openqha`; `Name: openqha-hessian`, `Version: 0.1.0`.
- Exactly the PEP 610 local-editable shape: no commit in metadata; the commit lives in the adjacent `.git` of each checkout.

## 5. What this means for ticket 06 (facts above; judgement flagged)

- "The package's commit is nowhere to be found" is false for the shapes we actually use: a local editable checkout gives commit + dirty from the adjacent `.git` (what `mace_fork_info` already does); a non-editable VCS install gives the commit from `direct_url.json` (mace `develop`'s own mechanism). Editable-VCS and local-editable metadata carry no commit, but a `.git` is present in both. The only genuine blind spot is a wheel/sdist — and mace's answer there is "do not use this shape for identity-critical code".
- Dirty state is only observable from `.git`. Mace never checks dirty — it sidesteps the problem by prescribing immutable (non-editable VCS) installs. Our editable dev checkouts make `.git` reading the only option, so a dirty gate is a product-contract choice of ours, with no mace precedent either way.
- Recording the producing code's commit next to a number has upstream precedent: mace `develop`'s LES golden references carry the solver commit and fail on mismatch. Ticket 06's package-identity fields extend that pattern rather than exceeding it.
- Mace's gating lives in the test tier (fail a comparison, with install instructions), not in a runtime refusal; whether our training driver refuses at runtime is a product choice (Q3), not a mace precedent to copy either way.
- Relevant to [05](../decisions/05-install-and-transport.md): mace's prescription for identity-critical installs is a pinned `pip install git+…@<40-hex>` — the shape that lets PEP 610 do the work. A wheel is a deliberate downgrade of that property; if 05 ships one, the provenance answer must come from somewhere else (publisher-side checksum, ticket 13).

Sources: local `mace/` files as cited above; `raw.githubusercontent.com/ACEsuit/mace/develop/tests/golden/{les_pin.py, train_anchor.py, feature_inventory.md}` (fetched 2026-09-28); PEP 610 `peps.python.org/pep-0610`; pip changelog `pip.pypa.io/en/stable/news` (20.1b1); `docs.python.org/3/library/importlib.metadata.html`; WSL dist-info artifacts (dumped 2026-09-28).
