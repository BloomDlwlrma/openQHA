---
status: accepted
date: 2026-09-26
---

# Wayfinder efforts get a two-layer ticket tree: decisions/ and implementation/

## Context

The first `/wayfinder` effort (`openQHA/.scratch/hessian-learn-framework/`) charted its 12
tickets as `.scratch/<effort>/issues/NN-<slug>.md` — the same flat form this repo's
single-workflow features use (`/grill-with-docs` → `/to-spec` → `/to-tickets` → `/implement`:
`spec.md` + `issues/NN-<slug>.md` + triage labels). At the same level and in the same
shape, a wayfinder map's tickets are indistinguishable from a single workflow's tickets,
although they are different objects: a wayfinder ticket is a node on a route (a decision,
or a route task), and some of them decompose into several executable slices once resolved.

## Decision

A wayfinder effort uses its own tree:

- `map.md` — the single map (Destination / Notes / Decisions so far / Implementation /
  Not yet specified / Out of scope).
- `decisions/` — the map's tickets, one file per ticket, `NN-<slug>.md` (the route the map
  charted).
- `implementation/` — execution slices derived from a ticket with `/to-tickets` (named
  `<parent><letter>-<slug>.md`, `07a-…`, with a `Serves: NN` line), plus work that never
  belonged on the route.
- `research/` — findings files, one per research ticket (`<slug>.md`).
- `archive/` — superseded/invalidated tickets only; resolved tickets stay in place (the
  map's links and "refer by name" need stable paths); the whole effort moves here once the
  map closes.

Numbering is one space for the whole effort (unique, never renumbered or recycled); the
frontier scan covers both ticket folders; the claim/resolve protocol is
`Status: open/claimed/resolved` with the answer appended under `## Answer`. The full
convention lives in the workspace tracker doc, `docs/agents/issue-tracker.md` →
Wayfinding operations.

## Considered options

* **Keep the flat `issues/` form for wayfinder efforts too.** Rejected: the level
  confusion this change fixes — map tickets look like a workflow's implementation
  tickets; "which of these is a decision and which is work" is not readable from the path.
* **Classify tickets by deliverable instead** (answers → `decisions/`, changes →
  `implementation/`). Rejected: every ticket would need a decision-vs-work argument up
  front (most carry both), and it fights wayfinder's own wording (its tickets are "decision
  tickets"); the slice layer is instead created exactly when a ticket resolves into
  several executable pieces.
* **Move closed tickets to `archive/`.** Rejected: the map's "Decisions so far" links and
  the "refer by name" rule need stable paths; `archive/` is for superseded/invalidated
  tickets and for the closed effort.
* **Put the convention into the vendored skills** (they are pinned by `skills-lock.json`
  and sourced from `mattpocock/skills`). Rejected: repo-specific conventions live in the
  repo's tracker doc, which the skills already defer to.

## Consequences

- `/to-tickets` inside a wayfinder effort writes to `implementation/` and uses the
  effort's `Status` protocol (documented in the tracker doc, not in the vendored skill).
- The map gains an `Implementation` section as slices land; `Decisions so far` stays for
  the map's own tickets.
- `hessian-learn-framework` was migrated the day this was adopted: its 12 tickets all
  live in `decisions/`; the first slices will appear in `implementation/` as tickets
  resolve.
