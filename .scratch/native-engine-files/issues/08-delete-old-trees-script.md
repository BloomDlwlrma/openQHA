# 08: Delete-old-trees script

**What to build:** one script that, on the machine it runs on, finds the old layout (the previous runs roots, `analysis/`, `data/basins/`, `logs/node_local/`) and with `--plan` (the default) lists each tree with its size and file count; with `--delete` it removes exactly those trees. It never touches the new root.

**Blocked by:** None (can start immediately).

**Status:** done 2026-09-14

- [x] `--plan` prints every old tree, its size and its file count, and removes nothing
- [x] `--delete` removes the listed trees and nothing else, and reports what it removed
- [x] A path under the new root is never listed, even when an old tree name appears inside it
- [x] Test point 7 on a temporary copy of the old layout
