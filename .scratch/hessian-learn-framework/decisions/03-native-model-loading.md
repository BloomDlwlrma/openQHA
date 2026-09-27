# Native model loading: what mace/mace-off offer without SHA256

Type: research
Status: resolved
Part of: [hessian-learn-framework](../map.md)

## Question

What is the native mace / mace-off way to load and call models, so that [SHA256 retirement](04-sha256-retirement.md) can decide with facts? Resolved by a research subagent against primary sources; findings at `research/native-model-loading.md` (this folder).

Investigate:

- (a) `mace-torch` v0.3.16 (the pinned base) vs current upstream: `mace.calculators.MACECalculator(model_paths=...)`; the `mace_off()` / `mace_mp()` helpers in `mace.calculators.foundations_models` — how a model is resolved (path, download URL, cache), what happens when the bundled `mace/calculators/foundations_models/*.model` files are absent (this fork's base case), any built-in checksums or version checks.
- (b) How a fine-tuned `.model` file is normally loaded (the `.model` container, `torch.load` vs the calculator), `--E0s` handling, dtype/device defaults.
- (c) MACE-OFF23 model distribution conventions (the `ACEsuit/mace-off` repo, ASL, HuggingFace?) and the loading lines MACE-OFF's own docs recommend.
- (d) A short "implications" section: what "remove SHA256, call models natively" could concretely mean for openQHA's engine registry — grounded in the above, with judgement clearly flagged (the decision itself belongs to ticket 04).

## Answer (2026-09-26, research subagent; full findings in [`research/native-model-loading.md`](../research/native-model-loading.md))

- **mace's load path has no identity check at all.** `MACECalculator(model_paths=…)` globs the string into `torch.load`; a missing file raises `ValueError`; multiple paths make a committee. Upstream v0.3.16 has zero sha256 under `mace/` — its only hashing precedent is the *test* tier (`tests/golden/`), "identity is measured, not assumed", at test time.
- The fork's calculators are byte-identical to upstream v0.3.16 (the 12-file delta is training-side only), so nothing fork-specific blocks native loading.
- `mace_off()` resolves keys → `raw.githubusercontent.com/ACEsuit/mace-off/main/mace_off23/*.model` → `~/.cache/mace/`; local paths and https URLs are accepted directly; the bundled `*.model`s being absent is the *normal* state (true of upstream wheels too). `MACE-OFF23_medium.model` is 18,350,596 B (matches openQHA's pin comment); MACE-OFF23-SC is not in the mace-off repo and stays an operator-supplied local file.
- **Implication for [SHA256 retirement](04-sha256-retirement.md):** `engine.calculator()` is *already* the native call, and the pin was always report-only. Two routes: (i) minimal — strip the fingerprint/pin plumbing, load exactly as now; (ii) delegate resolution to `mace_off(...)` — downloads/caching via `~/.cache/mace`, but offline clusters (Tianhe) get a new ASL-notice/download behaviour. Judgement (survival of `check_weights_are_physical`, the path overrides, the replacement provenance fields) is ticket 04's.
