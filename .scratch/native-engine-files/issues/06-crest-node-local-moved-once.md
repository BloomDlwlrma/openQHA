# 06: CREST node-local, moved once

**What to build:** CREST runs in `/tmp/<user>/<jobid>/<qid>/` and, when it returns, its whole working directory is copied once into `crest/` of the molecule directory (the SHAKE fallback's into `crest_shake1/`), then the node-local copy is removed. Scratch reuse, settings matching and the record parser read `crest/`.

**Blocked by:** 05 (MACE engine folder).

**Status:** done 2026-09-14

- [x] `crest/` holds CREST's files verbatim plus `input.toml`, `<qid>.xyz` and `crest.out`
- [x] `crest_shake1/` exists beside it only when the fallback ran; the first attempt is never overwritten
- [x] The node-local directory is gone after the move; a failed CREST still leaves its `crest.out` in `crest/`
- [x] Reusing an existing `crest/` under matching settings skips CREST as before
- [x] Test point 5 with a stand-in crest binary writing the known file set
