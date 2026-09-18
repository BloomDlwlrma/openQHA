# 01: Engine identity by parameter fingerprint, and registering a fine-tuned potential

**What to build:** `engine.parameter_fingerprint(name=None)` -- SHA-256 over the model's `state_dict` sorted by key (name, dtype, shape, raw bytes), container-insensitive (a `torch.save`/`load` round trip gives the same digest), number-sensitive (one changed tensor changes it); `provenance()` gains `params_sha256` and `n_tensors`; `ENGINES` entries may carry `params_sha256` and, when present, `provenance()` classifies a mismatch (reports it in the record, does not refuse -- S0-G-73's ruling); `scripts/tooling/s0_check_weights.py` (`--pin` prints the registry lines; two machines compare three lines). README: how a fine-tuned model enters: copy `<name>.model` into `data/potentials/`, add an `ENGINES` entry with `filename`, `source` = the Dataset index path + training config SHA, `params_sha256` from `--pin`; select with `S0_ENGINE`.

**Blocked by:** nothing.

**Delivers:** the identity every Frame-set Record and `index.dat` will carry.

- [ ] fingerprint identical across a save/load round trip of MACE-OFF23_medium, different after one tensor is perturbed (unit test with a tiny synthetic state_dict; integration on the real weights)
- [ ] `provenance()` carries `params_sha256`, `n_tensors`; a registry pin mismatch is reported, not refused
- [ ] `s0_check_weights.py --pin` prints the four registry lines; README section "registering a fine-tuned potential"
