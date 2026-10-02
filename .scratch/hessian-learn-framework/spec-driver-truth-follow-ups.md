# Spec: Driver truth follow-ups -- the multihead control mirror, the extras overlay, and the `--key=value` form

Label: `ready-for-agent`. Tracker: `.scratch/hessian-learn-framework/`. Spec for
[14-driver-truth-follow-ups](decisions/14-driver-truth-follow-ups.md); rulings of
2026-10-01 (user, R1–R5 as recommended: R1 the fork's rule mirrored in the control
resolution, provenance by description; R2 extras folded as the authoritative native
override, the Record follows; R3 add `--ema-decay` and `--force-mh-ft-lr`, defer
`--weight-pt-head`, keep the ratio-0.0 standard and the `num_samples_pt` refusal;
R4 `fine_tuning_select` reference-only; R5 sequencing left to the landing session --
the round-1 arms are unaffected).

## Problem Statement

A fine-tune's artifacts can misstate the schedule the run actually trains. Three doors
are open in the driver today:

- The fork's multihead mode silently overwrites the parsed arguments -- lr 0.0001, EMA
  on, ema_decay 0.99999 -- while the driver's Record, dry-run header and `config.yaml`
  keep the requested values or the class defaults (lr 0.01, decay 0.99, a boolean EMA
  with no decay field at all). Every multihead run executed to date trained at
  1e-4 / 0.99999 while its Record said otherwise.
- The extras channel (`--mace-arg`, appended last) wins any argparse duplicate -- it is
  the native editing surface -- but the Record does not follow it, so an extras-supplied
  value (or the `force_mh_ft_lr` escape) can desynchronize the artifacts from the run
  again.
- A single-token `--mace-arg=--clip_grad=1.0` reaches mace fine, but `argv_pairs` reads
  it as a bare flag, so `config.yaml` -- whose promise is "every mace argument, as given
  (the file mace itself can re-run)" -- carries `clip_grad=1.0: True` instead of
  `clip_grad: 1.0`.

Round 1 works around all three at the run level (explicit `--lr 0.0001`; the pinned
two-token extras form). That protection lives in operator discipline, and it is lost the
moment a launch line forgets a flag. The driver should stop being able to lie --
including when defaults are used -- without taking away the caller's ability to edit the
knobs (defaults allowed, explicit values allowed).

## Solution

The driver resolves each run's effective control once: the nominal defaults, with the
run's extras folded in for the controlled keys, and the fork's multihead rule applied
last (the fork forces after parsing) -- unless `force_mh_ft_lr` is set. That single
resolution feeds the Record, the dry-run header and the emitted flags, so the artifacts
cannot disagree with the values the fork trains; when the mirror replaces a requested or
overlaid value it says so.

`EMA_DECAY` becomes a first-class control (a Record field, a driver flag, and the
fork's forced value via the mirror). `--force-mh-ft-lr` becomes a first-class flag --
the one native way to use one's own lr/EMA in multihead mode. `argv_pairs` learns the
`--key=value` form, so `config.yaml` reads back what mace's parser read. `--multiheads`
without a Replay file is refused (the fork would silently disable the mode). The pinned
round-1 launch lines keep working unchanged, and their trained values do not change.

## User Stories

1. As the operator, I want a multihead run's Record to state LR 0.0001, EMA True and
   EMA_DECAY 0.99999 even when the launch line omits the explicit flags, so that the
   artifact cannot describe a schedule the run did not train.
2. As the operator, I want a warning when the fork's rule replaces a value I requested
   (or defaulted), so that I learn my knob was ignored before the run starts.
3. As the operator, I want an explicit `--ema-decay` flag (default mace's 0.99), so
   that the EMA decay is editable like the other controls.
4. As the operator, I want the decay recorded as `EMA_DECAY`, so that the schedule in
   the Record is complete.
5. As the operator, I want a first-class `--force-mh-ft-lr`, so that the native escape
   for the multihead lr/EMA rule is reachable without raw extras.
6. As the operator, when I force, I want the Record to state the values I passed, so
   that the escape hatch is truthful too.
7. As the operator, I want extras-supplied values for controlled keys to be reflected
   in the Record as the final value, so that my editing surface cannot desynchronize
   the artifacts.
8. As the operator, I want `force_mh_ft_lr` given through extras to be honored (the
   mirror steps aside), so that the native channel works from either door.
9. As the operator, I want `--mace-arg=--clip_grad=1.0` (one token) to read back as
   `clip_grad: 1.0` in `config.yaml`, so that the config file re-runs and reads
   truthfully.
10. As the operator, I want the two-token form to keep working, so that round 1's
    pinned launch lines stay valid.
11. As the operator, I want `--multiheads` without a Replay file refused with a message
    naming the fix, so that a run cannot silently train single-head while claiming
    multihead.
12. As the operator, I want `--num_samples_pt` to remain refused, so that no flag
    pretends to control a size the Replay file owns.
13. As the reviewer, I want the LR / EMA / EMA_DECAY descriptions to state the fork's
    multihead rule and its `force_mh_ft_lr` exception, so that forced values are
    distinguishable from requested ones.
14. As the reviewer, I want a comparability note for Records written before this change
    (their multihead control values are requests; the runs trained 1e-4 / EMA on /
    0.99999 regardless), so that old Records are read correctly.
15. As the reviewer, I want the pinned surfaces' explanatory comments (workflow README
    production block, Slurm header, T05 production cell, story 14's amendment) to carry
    a dated note that the single-token hazard is fixed, so that the pin is not read as
    still-broken behavior.
16. As the maintainer, I want the fork's rule duplicated in our control resolution
    beside a pinned reference and with no fork change, so that the frozen fork and its
    deployment row stay valid.
17. As the maintainer, I want no effective-value change for the pinned round-1 launch
    lines, so that the timing job and the two arms are unaffected by this change.
18. As the maintainer, I want the resolution arithmetic and the `=`-form extras covered
    by the existing unit and integration seams, so that no new seam is introduced.
19. As the next session, I want the change committed on both repos with suites green
    and a dry-run on the dbg subset, so that deployment at the next sync is mechanical.
20. As the effort, I want the ticket's Answer and one map line at resolution, so that
    the index stays single-sourced.

## Implementation Decisions

- **The control resolution (the mirror).** The driver's single control resolution --
  already the source for the Record, the dry-run header and the emitted flags -- gains
  the fork's multihead rule: when the run is multihead and `force_mh_ft_lr` is not set,
  LR = 0.0001, EMA = True, EMA_DECAY = 0.99999. The constants are duplicated in our code
  beside a pinned reference to the fork's forcing block (the fork is not changed). The
  mirror warns when it replaces a requested or overlaid value, and is silent when the
  values coincide. `SWA_LR` is not recomputed by the mirror -- the fork does not
  recompute it either; it stays the value derived from the request (or the given value).
- **The extras fold.** Extras remain the native, authoritative override (argparse
  last-wins). The run parses its extras with the `argv_pairs` mapping and, for the
  controlled keys -- the set the driver emits and records: `lr`, `scheduler_patience`,
  `patience`, `eval_interval`, `ema`, `ema_decay`, `swa`, `start_swa`, `swa_lr`, the
  Stage-Two weights -- folds the last-wins values **under** the fork rule. The folded
  result is what the Record and the dry-run state; the emitted flags keep the driver's
  authored values and the extras tokens stay appended verbatim (the argv stays "as
  given").
- **`force_mh_ft_lr`.** A first-class `--force-mh-ft-lr` on the driver CLI (default
  off; the help carries mace's "not recommended"), honored also when it arrives through
  extras (mace's truth values). When set, the mirror steps aside and the folded values
  are the Record's.
- **`EMA_DECAY` becomes a control.** A new Double field in the Record schema and the
  control resolution, emitted as `--ema_decay` whenever EMA is on; default 0.99 (mace's
  own default); the mirror's 0.99999 in multihead. The driver CLI gains `--ema-decay`.
- **`argv_pairs` reads `--name=value`.** A single token of that shape splits at the
  first `=` (an empty value is allowed; any further `=` stay in the value); bare flags,
  two-token pairs and non-`--` tokens are unchanged. This mapping also feeds the extras
  fold.
- **Refusals.** `--multiheads` without a Replay file is refused, with an error naming
  `--pt-train-file` (the fork would silently disable the mode and train single-head).
  With that guard, the mirror keys on the run's `multiheads` flag alone.
  `--num_samples_pt` stays refused as today.
- **Provenance by description (the 15b pattern).** The LR / EMA / EMA_DECAY
  descriptions state the fork's rule, the `force_mh_ft_lr` exception, and a pre-change
  comparability note; no new boolean field is added.
- **Two promises, kept distinct.** `config.yaml` keeps "every mace argument, as given
  (the file mace itself can re-run)"; the Record and the dry-run state the effective
  resolution. Where a fork-forced value and an extras override coexist the two may
  differ -- documented, not fought.
- **Dated surface notes.** The pinned surfaces' explanatory comments (workflow README
  production block, Slurm header, T05 production cell, story 14's amendment) gain a
  dated note that the single-token hazard is fixed; the two-token pin itself stays for
  round 1.
- **Module surfaces (house references, no paths).** The package's run module (the
  control resolution, the argv builder, the Record schema, the extras fold, the
  refusal); the package's run tests (unit + the fixture integration); the openQHA
  driver CLI (the two new flags and help); the four pinned documents.

## Testing Decisions

- **What a good test is here.** Assert external behavior at the existing seams: the
  resolved control (Record fields, dry-run header), the emitted argv, the `config.yaml`
  mapping, and the refusal messages. Pin the fork's mirrored constants as literals
  beside the pinned reference; never assert training internals.
- **Seams (unchanged from the round-1 spec).** Unit: the package's run test (control
  arithmetic, argv building, `argv_pairs`). Integration: the run-to-Record fixture that
  trains a real mini multihead fine-tune. Acceptance: the driver dry-run on the dbg
  subset. No new seams are introduced.
- **Unit coverage.** The mirror's three modes (multihead -> the forced triple; forced
  -> untouched; single-head -> untouched); `SWA_LR` under the mirror; the `--ema_decay`
  emission; the `=`-form `argv_pairs` cases (first `=`, empty value, a value containing
  `=`, bare flag, two-token); the extras fold (a single-head override recorded; a
  multihead override then forced; `force_mh_ft_lr` through extras stepping the mirror
  aside); the multiheads-without-file refusal; the existing `num_samples_pt` refusal;
  the description pins.
- **Integration coverage.** The fixture multihead run's Record carries LR 0.0001,
  EMA True, EMA_DECAY 0.99999, cross-checked against the fork's log line; a dry-run
  case built from a single-token `=` extra writes `config.yaml` and reads back
  `clip_grad: 1.0`.
- **Verification checklist (for the landing session).**
  1. Package suite `--all` green (new unit + integration checks included).
  2. openQHA suite `--all` green (the driver flag/help change).
  3. A `05_train --dry-run` on the `draw300_r1dbg` subset (minutes, with the amended
     balance): with the explicit flags forgotten, the header shows the mirrored
     LR 0.0001; `config.yaml` carries `ema_decay: 0.99999` and -- with a single-token
     extra -- `clip_grad: 1.0`; the force flag flips the header back to the requested
     value; multiheads without a Replay file refuses.
  4. Commits on both repos with recorded hashes.
  5. Deployment rides the next Tianhe sync; the round-1 arms are unaffected either way
     (their launch lines already carry the same effective values).
  6. Done (2026-10-02): the acceptance re-run on both device paths -- CPU and CUDA
     green with the post-ruling tips; evidence in
     [14c](implementation/14c-the-close-out.md)'s postscript.
- **Prior art.** The 15b control/Record unit additions; the fixture integration's
  Record assertions; the 09f dry-run procedure.

## Out of Scope

- Fork changes: the rule is mirrored in our control resolution; the fork stays frozen.
- `--weight-pt-head` as a first-class flag (deferred; extras can pass it today).
- The ratio-0.0 standard becoming a flag (kept as the emitted standard).
- Refusing conflicting extras (the fold policy is adopted instead).
- `fine_tuning_select` adoption (reference-only; a future ticket if the selection
  approach changes).
- Recording mace keys outside the controlled set.
- Effective-value changes: the pinned round-1 launch lines' trained values are
  unchanged; the workaround flags become redundant but stay valid.
- The round-1 submissions and the Tianhe operations themselves.

## Further Notes

- **References.** The fork's forcing block (`mace/mace/cli/run_train.py:191-213`: the
  silent downgrade, then lr/EMA/decay assignment) and its parser defaults
  (`--ema_decay` 0.99, `--force_mh_ft_lr` false); the research note
  `research/mace-finetuning-parameters.md`; the MACE docs' multihead page; the ticket's
  [09f](../implementation/09f-the-local-gate.md) finding.
- **Sequencing (as of 2026-10-01 evening).** The timing job already runs on the
  post-[15f](../implementation/15f-the-yhbatch-name-defect.md) checkout; the cap and the
  two arms follow. This change lands on both repos and deploys at the next convenient
  sync -- it does not need a refresh window and does not hold anything. The pushes and
  the Tianhe operations stay the user's.
- **The 15b pattern.** Descriptions carry provenance and comparability notes (as
  `BALANCE_PROBE` / `BALANCE_N_PROBES` did for the balance); this spec follows it for
  the mirrored control values.
- **The pinned two-token form** on the launch surfaces stays; a dated fix note is added
  beside it so the rationale is not read as a live defect.
