# 01: Layout module

**What to build:** one pure module in the store package that answers every path question of the new tree: given (root, tag, qid) it returns the molecule directory, each engine folder (crest, crest_shake1, mace conformer and basin folders, openmm setting and basin folders, xtb, orca) and the records folder, applying the shard rule range 16 000 / chunk 1 000. Every later writer and reader asks this module and nothing else composes these paths.

**Blocked by:** None (can start immediately).

**Status:** done 2026-09-14

- [x] From (root, tag, qid) the module returns `<root>/<tag>/<range>/<chunk>/<qid>/` with the shard names `1_16000/1_1000`, `1_16000/1001_2000`, ..., `16001_32000/16001_17000`, ...
- [x] The seven shipped species land in `1_16000/1_1000`
- [x] Engine folder functions return the documented sub-paths; the openmm folder takes a setting name and a basin index; mace has a conformer and a basin form
- [x] The records folder is `_records/` inside the molecule directory
- [x] The module reads no configuration and touches no filesystem
- [x] Test point 1 exists and runs under the repository's test runner
