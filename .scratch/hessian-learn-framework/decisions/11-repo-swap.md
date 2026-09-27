# The repo swap: replace openQHA-Hessian's content and clean the local tree

Type: task
Status: open
Blocked by: 05, 07
Part of: [hessian-learn-framework](../map.md)

## Question / work

Q5 execution: `BloomDlwlrma/openQHA-Hessian` keeps its name and URL and gets new content — the package, the installer, the README that says what this extension is and where mace lives — with no MACE-native files in it.

Work:

1. Land the new tree (from [The move](07-the-move.md) / [Install and transport](05-install-and-transport.md)); rewrite the README (extension ≠ mace; install; link to the fork; the old-history pointers); keep the GitHub description honest (today: "openQHA Extension: Active Hessian Learning Framework Instance for MLIPs - MACE").
2. Replace the GitHub default branch content (force-push or a fresh branch — state which); save the old history as a **local bundle** only; the old remote branch goes away.
3. Clean the working tree: untracked `logs/`, `results/`, `mace_torch.egg-info/`, `.pytest_cache/`, `mp_finetuning*.xyz` — removed or explicitly kept; the point is that the local tree stops looking like a second mace checkout.
4. Decide: a minimal CI workflow for the repo (install the fork + run the package's tests), since mace-md has one — if yes, it lands here.

## Answer

<!-- resolver: append what was done (branch, bundle path, CI) + evidence; set Status: resolved; add a line to the map's Decisions so far -->
