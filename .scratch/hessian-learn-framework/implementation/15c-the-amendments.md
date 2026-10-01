# 15c: The amendments -- spec-phl-verbatim and 09g

Type: task
Status: resolved
Blocked by: [15b](15b-the-driver-and-the-record.md).
Serves: [15](../decisions/15-the-balance-on-the-probe-estimator.md) · spec: [spec-the-balance-on-the-probe-estimator.md](../spec-the-balance-on-the-probe-estimator.md).

**What to build:** the documents say the new rule. `spec-phl-verbatim.md` (the
`hessian-learning-set`) gains a dated amendment: story 5's "on the full Cartesian
matrix only" becomes "with the run's probe setting (gaussian k=4 by default); the exact
full-matrix reading remains the anchors' and the judge's job"; Derivation 3.6's
parenthetical "(the estimator's noise must not enter a weight)" is replaced by the
amended rationale -- a fixed-seed k=4 estimate carries a ~0.1-0.5% offset on the full
train file, far below the ~5-35% per-step noise the loss itself trains through, and the
estimator is recorded (`BALANCE_PROBE` / `BALANCE_N_PROBES`), not silent; the
implementation line "`epoch_zero_balance` computes the Cartesian value only" is
qualified the same way. `09g` (the round-1 runbook) is re-baselined: the balance no
longer reads as a full-matrix pass over the whole train file; the wall check's fixed
head becomes split + probe-balance + anchors; the timing submission stays held until
the change is deployed.

- [x] Amendment text landed with the 2026-10-01 date; conflicting original wording marked as superseded (history stays visible).
- [x] 09g lines updated (balance description, fixed head, held-submission sequencing).
- [x] Residual scan: no unqualified "full Cartesian matrix only" / "full-matrix pass" reading left in the touched files.

## Notes

- Land after 15b so the wording describes landed behavior; the numbers it cites are the
  spec's (and 15d's, once measured).

## Answer (2026-10-01)

**What landed.** The two documents say the new rule; every amendment is dated 2026-10-01 and
keeps the replaced wording visible (strikethrough beside the replacement):

- `spec-phl-verbatim.md` (the `hessian-learning-set`) -- four spots: story 5 ("on the full
  Cartesian matrix only" becomes "with the run's probe setting (gaussian k=4 by default; the
  exact full-matrix reading remains the anchors' and the judge's job)", and its justification
  becomes "the balance is measured under the trained term's own estimator"); Derivation 3.6
  ("(the estimator's noise must not enter a weight)" becomes the amended rationale -- a
  fixed-seed k=4 estimate carries a ~0.1-0.5 % offset on the full train file, far below the
  ~5-35 % per-step noise the loss itself trains through, and the Record keeps `BALANCE_PROBE` /
  `BALANCE_N_PROBES`); the step-3 implementation line ("computes the Cartesian value only")
  qualified the same way (probe setting beside the exact path); and the Further-notes sentence
  "with the full matrix" struck + dated (the review's catch).
- `09g` (the round-1 runbook) -- the header gains the dated revision note (balance按运行的 probe
  设置估计、分钟级;固定头 = split + probe-balance + 两锚点; **the timing submission stays held
  until the change is deployed**, user ruling 2026-10-01); the step-2 note's
  "全矩阵一次过整个 train 文件" struck and corrected; receipt 2's read-back and its list now
  carry `BALANCE_PROBE` / `BALANCE_N_PROBES` (`({BALANCE_PROBE} k={BALANCE_N_PROBES})`); the
  cross-arm step says the two arms share `HESSIAN_WEIGHT` by (train file, probe/k/seed).

**Residual scan.** Over the two touched files: "full Cartesian matrix only" (story 5),
"computes the Cartesian value only" (step-3 line), "with the full matrix" (Further notes) and
"全矩阵一次过整个 train 文件" (09g step 2) appear only struck-through/negated, each beside its dated
amendment; no "full-matrix pass" phrasing anywhere. The remaining "full matrix" mentions are the
judge/anchor/evaluation contexts and stay as they are.

**Ordering.** Landed after [15b](15b-the-driver-and-the-record.md)'s commit `05ac7b1`
(annotations `547c394`), so the text describes committed behavior: the balance's `L_H` is read
under the run's probe setting (the run's probe/k/seed; `BALANCE_PROBE` / `BALANCE_N_PROBES` in
the Record), and the exact reading stays with the anchors and the judge.

**Review.** Two-axis review of the working-tree diff vs `c66833c` (two read-only sub-agents, per
the `code-review` skill):
- Spec: no material gaps; the beyond-ticket additions (story 5's "so that" clause rewrite; the 09g
  receipt-2 read-back and the step-5 "(same probe/k/seed)" note) judged consistent with the
  framework spec's stories 4-6 and recorded here; no wrong field names, dates or numbers.
- Standards: one real leftover found and fixed in-pass -- `Further notes` still read "with the
  full matrix" unmarked; it is now struck + dated like the rest. Also fixed in-pass: the fixture
  measurement is qualified "pre-amendment, the exact reading"; the amendment marker is uniform
  ("amended 2026-10-01 -- hessian-learn-framework ticket 15: the balance on the probe estimator",
  outside the bold, in all four spots); 09g's 修订/修正 wording is unified and its receipt 2 and
  回执清单 carry `BALANCE_PROBE`/`BALANCE_N_PROBES`.

**Not verified.** The Tianhe deployment and the timing resumption are [15e](15e-the-close-out.md)'s
scope; this slice only re-baselined the text.
