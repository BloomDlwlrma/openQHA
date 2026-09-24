# Hessian learning -- grilling, round 6: two potentials, and where the surface one's modules go (2026-09-21)

The user's ask: **two MACE Hessian potentials**. One is the thermochemistry potential the
campaign is already building (entropy-weighted PHL, Hessian Labels at the stationary
frames, judged by the msRRHO entropy). The second is a **full Hessian potential-energy
surface** -- every mode right, curvature right away from the minima, the third derivatives
that VPT2 / reaction paths / transition states read -- and this round settles what that
second potential *is*, which modules it adds to `openQHA` and whether it needs anything
from the `openQHA-Hessian` fork.

## Facts checked so you don't have to

**What is already built and needs no second copy.** Tickets 10-14 are done: the HVP
(`openqha/training/hvp.py`), Algorithm 1 and the loss (`phl.py`, `phl_loss.py`), the
Label in the batch and the external-loss hook (fork commits `1b9c382` A and `e68390f` B on
`openqha-hessian`, base `8fac5d1` = `base-v0.3.16`), the driver (`05_train.py` +
`training/run.py`), the judge (`judge.py` + `06_judge.py`, 32 checks). The loss already
takes `probe ∈ {rademacher, gaussian, modes, cartesian}` and `mode_weighting ∈ {entropy,
none}`; T04 §3 derived `relative` (`w_i = 1/|λ_i|`) and `cartesian` (`B = I`) as the same
operator with a different `B`, so the surface loss is a weighting, not a new loss.

**The two weightings are less different than the framing suggests.** T04 §3 measured, on
propanal with the base model, the share of each loss that comes from each frequency band:

| weighting | < 300 cm⁻¹ | 300-1000 | 1000-2000 | 2000-4000 |
|---|---|---|---|---|
| absolute (T03's) | 0.2 % | 2.0 % | 26.0 % | 71.7 % |
| Cartesian (Rodriguez / HIP) | 0.6 % | 3.8 % | 54.1 % | 41.5 % |
| **entropy** (the thermo potential) | 76.7 % | 19.3 % | 4.0 % | 0.0 % |
| **relative** (the surface candidate) | 80.5 % | 10.0 % | 8.3 % | 1.2 % |

Both of the last two are dominated by the modes below 300 cm⁻¹ -- not because the
weighting says so but because that is where the base model's *relative* frequency error
is (1.69 % below 300, 0.26 % above 2000). The two potentials therefore differ by a factor
2.4 in the stretch band and hardly at all below 300. **The real difference between a
thermochemistry potential and a surface potential is the DATA and the JUDGE, not the
weight vector.**

**What the campaign's Labels do not contain.** Round-5 Q7 ruled (b): Hessians at
`basin / merged / saddle`, gradients only at `displaced`. T04 §1 and §5 measured what that
leaves unconstrained, on the 2-methyloxirane frame fixture (basin + 4 displaced frames
with E/F/H at BOTH levels):

- over a classical 298 K frame (RMS 0.06-0.13 Å) the reference Hessian itself changes by
  **15-30 %**; the basin Hessian alone mispredicts the force difference by 20-90 %, the
  trapezoid of two Hessians gets it to 1-19 % (eq. 1.1);
- the base model's projected error at a displaced frame is **ρ_k = 1.45-2.01** times its
  error at the basin (eq. 5.3, round-5 Q7 (c)'s probe, first measurement);
- it reproduces the reference's change of curvature along `d` to 4-8 % (eq. 5.2);
- two Hessians along one mode give the whole cubic slice `φ_··k` (eq. 5.4) -- MACE's
  `φ_kkk` on propanal's two torsions is ~0 (correct symmetry), −0.157 on the third mode.

**Cost of buying what is missing** (round 5's own numbers, 19-atom molecules, 4 ranks):
analytic Hessian 40-80 min per frame, gradient 3-5 min; campaign as ruled ≈ 8.4 days of
12 nodes; + one displaced Hessian per basin ≈ **11 days**; Hessians everywhere ≈ 22 days.

**What the fork can already do.** Commit A stores a per-structure `hessian`,
`has_hessian`, `hessian_weight`, `sqrt_masses`; commit B is `--loss external
--loss_module module:factory` plus `compute_hessian=True` at validation when the loss asks.
Neither knows what msRRHO, a structure class or a weighting is. A surface loss is another
`module:factory`.

**Frames.** `frames.GENERATORS = (basin, displaced, merged, saddle)`; the displaced draw is
classical at `TEMPERATURE_K = 298.15`, `MAX_RMS_A = 0.15`, 4 per basin, energy window 275
kcal/mol. `02_frames.py` already takes `--temperature`, `--max-rms`, `--n-displaced`,
`--distribution`, but every draw is written under the one name `displaced`.
`frame_labels.HESSIAN_GENERATORS` is a module constant, not a flag.

---

## Round 6 questions

❓ **Q1** - **What makes them two potentials.** Given the band table above, name the axis.
(a) **Data**: the surface potential is trained on the same Dataset *plus* Hessian Labels at
displaced frames (and, later, along-mode line points), with the weighting a secondary
choice; (b) **Loss**: same Dataset, `relative` instead of `entropy`, two runs from one
label campaign, no extra ORCA; (c) **Both**: extra Labels *and* the relative weighting;
(d) **Base model too**: also fine-tune MACE-OFF23_large or MACE-OFF24 for the surface, so
the two potentials differ in capacity as well.

➡️ **(c), stated in that order of importance.** (b) alone buys a 2.4× re-weighting of a
band that carries 1-8 % of either loss -- the smoke fit can measure it for free, but it is
not a second potential. What is not in the training set at all is the curvature away from
the minima, and ρ_k = 1.45-2.01 says the base model is measurably worse there than at the
basins. (d) is a third axis and belongs to a later round: it doubles every training and
judging cost and answers a different question (capacity vs objective).

---

❓ **Q2** - **One workflow or two.** `workflows/hessian_learning/` is steps 00-06 with one
Dataset per tag. (a) **One workflow, a named target**: `configs/training_targets.yaml`
holds `thermo` and `surface` (weighting, probes, `w_H` band, judge thresholds, which
generators carry Hessians), `05_train.py --target surface` and `06_judge.py --target
surface` read it, one code path, the Records carry `TARGET`; (b) **a second workflow
folder** `workflows/hessian_surface/` with its own 00-06; (c) no config file: flags only
(`--mode-weighting relative --hessian-generators basin displaced ...`), the difference
lives in the runbook and in each run's Record.

➡️ **(a).** (b) duplicates 00-04 verbatim -- the frames and the Labels are the same
Calculations -- and a duplicated driver is where the two potentials silently drift apart.
(c) is what (a) does anyway, minus the one place a reader can see both objectives side by
side and minus a name for the thing. `Training target` then becomes a CONTEXT noun (Q8).

---

❓ **Q3** - **The weighting module.** T04 §3 derived four `B` for one operator.
(a) extend `phl.py`: `WEIGHTINGS = (absolute, relative, entropy, cartesian)`, one
`weights_for(name, lam_r, ...)` with `λ_floor = λ(30 cm⁻¹)` for the relative case, and
`phl_loss.MODE_WEIGHTINGS` follows; (b) a separate `openqha/training/surface_loss.py`
subclassing the loss; (c) only add `relative`, leave `cartesian` out (it is the literature's
weighting, not ours, and T04 showed it is 96 % decided by the stretches).

➡️ **(a) with all four.** They are four lines of the same function and T04 proved the
estimator is unbiased and the `modes` set exact for every one of them; having `cartesian`
in the table is what lets the judge say "and this is what a Rodriguez/HIP-weighted loss
would have optimised" without a second implementation. (b) would fork the estimator, which
is the one piece that must stay identical between the two potentials.

---

❓ **Q4** - **Which frames get a Hessian Label for the surface set.** Today
`HESSIAN_GENERATORS = (basin, merged, saddle)`. (a) **+ one displaced frame per basin**
(round-5 Q7 (c), ~+2.6 days of 12 nodes, ~18,000 extra Hessians): gives ρ_k as a *training*
signal, not only a judge number; (b) **+ all four displaced frames per basin** (~22 days
total): the full off-minimum surface; (c) **stationary only**, and the surface potential is
(b) of Q1 after all; (d) **(a) + along-mode line points** on a subset (ticket 32's
`mode_curvature` machinery, 7 points per mode, the reference's cubic constants by eq. 5.4).

➡️ **(a) now, (d) on the seven pinned molecules only.** The marginal 2.6 days buys the one
thing the campaign cannot otherwise learn, and T04's ρ_k says the gap is real rather than
hypothetical. (b) triples the label bill for the fourth digit of the same information.
(d) on seven molecules is hours, and it is the only anharmonic *reference* we can cite.
Implementation: `HESSIAN_GENERATORS` becomes a per-run argument
(`03_labels.py --hessian-generators`, recorded in `labels.<level>.toml`), with the
`k`-selection rule for (a) written down (`k = 0`, the lowest-RMS frame of the basin).

---

❓ **Q5** - **Does the surface set need different FRAMES, not just different labels?** The
displaced draw is classical 298 K, RMS ≤ 0.15 Å -- the neighbourhood msRRHO visits. A
surface potential used for MD, a TS search or VPT2 sees more than that.
(a) same frames, only more Labels (Q4); (b) add a generator `displaced_hot` (classical
500-600 K or RMS ≤ 0.30 Å, 2 per basin) so the surface potential sees curvature it will be
asked for, written under its own generator name with its own Record fields; (c) add frames
from branch B's MD trajectories (`quasi_harmonic/`, already on disk for some molecules);
(d) (b) + (c).

➡️ **(b), 2 frames per basin, and only for the surface target.** A new generator name
costs nothing in the code (`frames.GENERATORS` is a tuple and every Record is keyed by
generator) and keeps the two potentials' provenance separable in `index.dat`. (c) mixes a
different sampling process (thermostatted dynamics, correlated frames) into a set whose
whole story is "seeded from MACE basins" -- it is a round-7 question if (b) is not enough.
Cost: 2 frames per basin at gradient price + Q4's Hessian on one of them.

---

❓ **Q6** - **The judge for the surface potential.** `judge.py` reports low-mode MAE,
full-spectrum MAE, `‖A‖²_F/n_vib`, the entropy tier and forgetting. T04 §8 Algorithm 5
lists what a surface judge adds. (a) extend `judge.py` with a `[Surface]` block:
per-band **relative** frequency MAE (T04 eq. 3.3), ρ_k at the labelled displaced frames,
the two invariance residuals (sum rule, rotation-gradient identity -- the base model gives
1e-7, the *reference* only 2-5 %, so this line also audits the Labels), and the cubic
slice against the line points where they exist; (b) a separate
`openqha/training/surface_judge.py`; (c) keep one judge and let the thresholds differ by
target only.

➡️ **(a).** One judge, one Record, a block that is present when the data for it is. The
thresholds are per target (Q2's config), and the surface target's are a round-7 question:
they must be measured on the smoke fit first, and the `[Surface]` block's floors are the
Label's own grid noise (24.4 cm⁻¹ on the methyloxirane basin) and its 2-5 % rotation
residual, not zero.

---

❓ **Q7** - **Does the fork need anything else?** Commits A and B are generic. The surface
target as sketched needs: a second `--loss_module` factory (openQHA side), a per-frame
`hessian_weight` (already in A), the full matrix at validation (already in B).
(a) **no further fork commits**, and the rule is written down: the fork only ever gets
changes that would make sense to ACEsuit, anything that knows about msRRHO, structure
classes or a weighting stays in `openqha/training/`; (b) a third commit C for a per-frame
*probe count* or per-frame weighting API; (c) a commit C that upstreams the whole loss.

➡️ **(a), with the rule as an ADR line.** The measured argument: the fork is 2 commits /
~400 lines against `base-v0.3.16`, and every rebase onto a new mace is that diff. (b) is
not needed -- the loss owns the probes and can vary `k` per frame inside itself. (c) is a
different project (and upstream would want the derivation, which is what T03/T04 are).

---

❓ **Q8** - **Nouns, registry and naming.** Two fine-tuned potentials will sit in `ENGINES`
beside MACE-OFF23_medium, and `ADR 0001`/`engine.py` says an entry is `filename`, `source`,
`note` (+ optional `params_sha256`) -- "three fields, no more". (a) CONTEXT gains
**Training target** (the objective a fine-tune optimises: its weighting, its probe set, its
Hessian generators and its judge thresholds -- named in `configs/training_targets.yaml`),
and the engine entry states the target inside `source` (`<index path> + <config SHA> +
target=surface`); (b) add a fourth field `target` to the registry; (c) no new noun -- call
them "the thermo model" and "the surface model" in prose.

➡️ **(a).** The registry stays three fields (the ADR's point is that the filename is the
identity and the registry is not a database), and `Training target` is the word the spec,
the config, the Records and the judge tables all need anyway. Proposed engine names:
`mace-off23-qha_thermo` and `mace-off23-qha_surface`, both with `source` naming the
Dataset index, the config SHA and the target.

---

❓ **Q9** - **Order and what the smoke fit must answer.** Ticket 15's smoke fit
(`s0_hl_smoke_fit.py`, running) measures cost per probe setting, the epoch-0 `w_H` balance,
a `w_H` scan and the replay ratio -- all with `mode_weighting = entropy`. The campaign's
`hl_labels` array has not been submitted at the reference level.
(a) **Decide Q4/Q5 before the array goes in**: the extra displaced Hessians and the hot
frames are ~3 days of the same 12 nodes if they ride along, two submissions and a second
pass through 6,458 molecules if they do not; and extend the smoke fit now with
`entropy | relative | absolute` at fixed `w_H` (three more 100-epoch runs on 65 frames --
minutes) so both targets are configured from one measurement; (b) finish the thermo
potential end to end first, then decide the surface set from its judge table; (c) split the
campaign: thermo labels for all 6,458 molecules, surface labels for a stratified 1,000.

➡️ **(a).** The ORCA bill is the irreversible part of this plan and the marginal cost of
riding along is 30 %; the training runs are hours and can be repeated. (b) pays twice for
one queue. (c) is a good fallback if the queue says no, and it is one flag
(`03_labels --hessian-generators ... --limit 1000 --stratify`) -- but it makes the two
potentials' training sets differ in molecules as well as in labels, which the judge then
has to control for.

---

## What waits for a later round

The surface target's PASS/FAIL thresholds (needs the smoke fit's numbers); the hot
generator's exact temperature and RMS ceiling (needs Q5); whether one shipped model with a
mixed weighting beats two (needs both judge tables); a larger base model (Q1 (d)); MD
frames (Q5 (c)); upstreaming anything to ACEsuit.
