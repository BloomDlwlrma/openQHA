# Publication: our own fine-tuned models, out the mace-off way

Type: task
Status: open
Blocked by: 09
Part of: [hessian-learn-framework](../map.md)

## Question / work

Publish our own fine-tuned (ORCA-Hessian-accuracy) MACE models so others can load them
the mace way — the premise of [SHA256 retirement](04-sha256-retirement.md)'s route ruling
("we cannot delegate; we learn mace-off; the first artifact is the round-1 minimal
model"). The conventions are fixed (see Facts); this ticket settles the rest and executes.

Settle at claim time:

- **Licence and citation**: the weights derive from MACE-OFF23 (ASL, academic use only);
  the model repository's README/`LICENSE` wording, the campaign's citation, and what the
  labels' provenance requires.
- **Home and release form**: the dedicated model repository (separate from
  `openQHA-Hessian`), its org/name; one tagged GitHub Release per revision; the README
  with the load lines (`mace_off(model="<https URL>")` and a local file; the
  `~/.cache/mace` note).
- **Release-note content**: Dataset index, `CONFIG_SHA256` prefix, `MACE_FORK_COMMIT`,
  file names/basenames, and the optional `sha256sum` listing (publisher-side integrity;
  never a load-time gate). *(2026-10-02 checksum ruling: the `CONFIG_SHA256` prefix is
  out — the release note names the config file the Record carries.)*
- **First release**: the round-1 model from ticket 09 (assembled via ticket 08), under the
  decided names (`mace_off23_<campaign>/<run>+<YYYYMMDD-HHMMSS>.model`).
- **The Level question**: `level_name()` derives a revision-flavoured Level from a
  self-trained registered name (`draw300-r4+…`); decide whether a fine-tuned revision is a
  new Level — it would enter `<level>.<step>` file naming and the thermo folders — or maps
  to the base level; decide before registering the first model.

## Facts to stand on

- `mace_off()` accepts any `https:` URL or a local path and caches by URL basename, so our
  releases load with mace's own documented lines unchanged
  ([research](../research/native-model-loading.md)).
- Upstream precedent: mace-off = plain files on `main` (mutable); mace-mp = release-tag
  URLs; code repository and model repository are separate. We take the organisation from
  the former and the tagged, immutable-by-convention bytes from the latter.
- The 04 rulings fix the local side: one fixed registry entry per revision; basename
  globally unique; no load-time pinning of any kind; a release-time `sha256sum` listing is
  allowed as a publisher artifact.

## Answer

<!-- resolver: append what was decided/done + evidence; set Status: resolved; add a line
to the map -->
