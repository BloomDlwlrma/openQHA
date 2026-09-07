# Branch B in production

The quasi-harmonic half: what the settings are, why they are not the published protein
ones, what was measured before choosing them, and how to run it on Tianhe.

Companion to [`branchA_production.md`](branchA_production.md). The protocol itself lives
in [`../configs/branchB_protocol.yaml`](../configs/branchB_protocol.yaml) — one file, read
by both trajectory routes and by the Parsl driver.

---

## 1. A correction, stated first

On 2026-09-07 the published protocol of Rinaldo & Field (*Biophys. J.* 2003) was adopted
wholesale: **520 ps equilibration, 1.5 ns production, one frame per 0.5 ps, friction
50 ps⁻¹.** That was wrong, and it was wrong in a way worth writing down.

It is the right *paper* — this branch already cites it for its thermostat and for both
convergence diagnostics. What does not transfer is the **system**:

| | Rinaldo & Field | openQHA |
|---|---|---|
| system | solvated protein, 2586 heavy atoms | 10–19 atoms, vacuum |
| degrees of freedom | 3N = 7758 | 3N−6 = 24–51 |
| their limiting problem | too few frames for the modes | **cannot happen here** |
| our limiting problem | — | **the trajectory leaves the basin** |

Their constraint was that a covariance from T frames has at most T non-zero eigenvalues,
so 3000 frames could never resolve 7758 modes. Ours is the opposite end of the same
inequality, and a different failure entirely.

---

## 2. The failure that actually threatens us: leaving the basin

Quasi-harmonic analysis assumes the sampled configurations are **one multivariate
Gaussian**. Branch A enumerates the basins; branch B measures the entropy *inside* one.

**A trajectory that crosses a torsional barrier samples two basins.** The covariance then
contains the displacement *between* the minima as well as the fluctuation within them,
T·S is inflated, and nothing else in the chain notices.

Acetone's methyl barrier is ≈0.8 kcal/mol against kT = 0.59 at 298 K. It rotates freely.
Over 1.5 ns it turns many times.

**Two kinds of crossing, and only one is a different conformer:**

| | what happened | conformer count | Cartesian covariance |
|---|---|---|---|
| symmetry-equivalent | a methyl turned 120° into a copy of itself | unaffected — this is what σ is for | **still inflated**: the three H atoms swept a circle |
| distinct | the molecule is in a different well | wrong | **not an intra-basin entropy at all** |

The symmetry number corrects the *rotational* partition function and the counting of
distinct conformers. **It does nothing to the quasi-harmonic covariance.**
`openqha/quasi_harmonic/basin_residence.py` counts both.

### This inverts acceptance criterion 1

Criterion 1 fails when T·S is still rising over the last doubling. That has **two causes
demanding opposite actions**:

```
rising, no crossings   ->  the estimator has not converged   ->  run LONGER
rising, crossings      ->  the trajectory is leaving a basin ->  run SHORTER
```

They look identical in the saturation curve alone, and the earlier version of this
document said "run longer" unconditionally — which, in the second case, makes the number
worse while eventually making the criterion *look* satisfied (an inflated covariance
converges too). `basin_residence.interpret_saturation()` refuses to read one without the
other.

---

## 3. What was measured before choosing

`qha.synthetic_harmonic_trajectory` draws from the exact classical canonical distribution
of a molecule's own Hessian, so the true T·S is known in closed form and the **estimator**
error can be isolated from any question about dynamics.

Acetone, its own MACE-OFF23_medium analytic Hessian (`max|H−Hᵀ| = 1.4e−14 eV/Å²`),
8 independent draws per row:

```
ALL ATOMS (10 of 10), 24 modes, closed-form T*S = 2.9317 kcal/mol
  len/ps   step/ps   frames  frames/DOF   mean T*S       bias    spread
     500       2.0      250        10.4     2.9371    +0.0053    0.0412
     500       1.0      500        20.8     2.9281    -0.0037    0.0297
     500       0.5     1000        41.7     2.9268    -0.0050    0.0145
    1500       0.5     3000       125.0     2.9269    -0.0048    0.0044
```

Three conclusions, one of which corrects a prediction I made here earlier:

1. **The estimator is essentially unbiased across the whole range** — |bias| < 0.01
   kcal/mol even at 250 frames. I had predicted a finite-sample eigenvalue bias at
   frames/DOF ≈ 10; measured, at this system size, **it is not there**.
2. What frames buy is **precision**: the seed-to-seed spread falls from 0.041 to 0.004
   kcal/mol. Both are far inside the 1 kcal/mol target.
3. **All 24 modes are recovered at 2 ps spacing**, and T·S matches the closed form.

Point 3 is the direct disproof of a claim that is often made for downsampling —

> *"a longer sampling interval filters out high-frequency bond and angle thermal noise
> and keeps only the conformational fluctuation"*

**It does not.** The quasi-harmonic covariance is an **equal-time** average,
`⟨(xᵢ−⟨xᵢ⟩)(xⱼ−⟨xⱼ⟩)⟩`. Every retained frame still carries the full instantaneous
bond-length displacement, and a stiff mode contributes `kT/mω²` however sparsely you
sample. If downsampling filtered them, T·S would fall *below* the closed form. It does
not move.

The conclusion (use 1–2 ps) is still right — it just buys **decorrelation and less to
write**, not filtering. What actually separates conformational from vibrational is the
eigenvalue spectrum, and that separation is already done in mode space
(`entropy["S_per_mode_kcal_per_K"]`, `qha.mode_batch_convergence`).

---

## 4. The four settings

| setting | paper | openQHA | why |
|---|---|---|---|
| timestep | 1 fs | **1 fs** | unchanged; also `D0-C-30` |
| equilibration | 520 ps | **50 ps** | a 10-atom molecule in vacuum is not a solvated protein relaxing over 500 ps |
| production | 1.5 ns | **500 ps** (range 500–1000) | set by basin residence, not by the paper |
| sampling interval | 0.5 ps | **1.0 ps** (range 0.5–2.0) | decorrelation and storage; the estimator is fine at all three |
| friction / coupling | 50 ps⁻¹ Langevin | **NHC, 20 fs coupling** | see below — the numbers are not comparable |
| analysis atoms | non-hydrogen | **all** (`heavy` available) | different quantities; see below |

### Friction: a name collision worth untangling

The paper's *"a friction coefficient of 50 ps⁻¹"* is a **Langevin** friction. openQHA's
production route uses a **Nosé–Hoover chain**, where openmmtools' `collision_frequency =
50/ps` means a **coupling time of 20 fs**. Same number, different algorithm, different
meaning — they must not be traded for each other.

**For Langevin, equilibrium configurational averages do not depend on γ at all.** Langevin
samples the canonical distribution for any γ > 0. So weaker damping cannot "restore the
true fluctuation amplitude" — the amplitude is already correct. What γ changes is the rate
of configurational *diffusion*, hence how much of the basin a fixed-length trajectory
covers, and the finite-timestep integration bias. (Note that slower diffusion means
**fewer barrier crossings**, which per §2 is a point in favour of stronger damping here,
not against it.)

**For the Nosé–Hoover coupling time this repository has a measurement that points against
intuition.** On a harmonic surface where the answer is known in closed form, a 100 fs
coupling returned the softest mode at 179 cm⁻¹ against a true 79.7 and cost **1.15
kcal/mol** on T·S; 419 fs — the "intuitive" choice, matched to that mode's period — cost
1.23. **20 fs was the best of the sweep**, and `qha.assert_trajectory_identity` refuses
anything looser.

So the default is **unchanged until the scan says otherwise**, and the scan covers both
axes against the closed form: `scripts/calibration/s0_B_thermostat_choice.py` now has
Langevin at 0.1 / 1 / 5 / 10 / 50 ps⁻¹ and Nosé–Hoover `tdamp` from 5 to 1000 fs.

### Atom selection: not a cleaner version of the same number

Measured on acetone's own Hessian, closed-form T·S:

```
all atoms    (10 atoms, 24 modes)   2.9317 kcal/mol
heavy only   ( 4 atoms,  6 modes)   0.1383 kcal/mol
```

**These are different quantities and must never be compared.** The hydrogens carry 18 of
the 24 modes.

The case *for* heavy-atom analysis here is not the usual one (dropping X–H stretches,
where classical equipartition is worst at hν/kT ≈ 14). It is that **the atoms it drops are
exactly the ones a freely rotating methyl sweeps around a circle** — the dominant source
of non-Gaussian variance in these molecules. For a *difference* between conformers of one
molecule the X–H contributions largely cancel, which makes the heavy-atom estimator
better-conditioned for the conformational correction even though it is useless as an
absolute entropy.

It is `analysis_atoms` in the protocol, and the scan reports both.

---

## 5. Running the scan

```bash
# seconds, no MD, truth known in closed form
python examples/02_qha_openmm_acetone/s0_qha_parameter_scan.py --stage estimator

# hours to days -- on a cluster
python examples/02_qha_openmm_acetone/s0_qha_parameter_scan.py --stage dynamics \
    --thermostats nhc_20 nhc_100 langevin_1 langevin_5 langevin_50
```

The dynamics stage runs **one trajectory per thermostat** at the longest length and
finest interval, then scans length × interval × atom-set **post hoc** by truncating and
striding that one trajectory. Only the thermostat needs separate runs — which is what
takes the scan from a four-way grid to five trajectories.

Every row carries `distinct_basin_crossings` and `symmetry_equivalent_crossings` beside
its T·S, and a verdict that says which action a failed criterion 1 calls for.

**Nothing here is settled for a new molecule until the dynamics stage has been run on
it.** Acetone has two methyl rotors; a molecule with a hindered single bond will behave
differently.

---

## 6. Running production on Tianhe

Trajectories are a GPU job (TianheXY-A preferred); collection is CPU, one node.

```bash
# 30-minute smoke test first -- always
python -u scripts/production/s0_E_branchB_parsl.py --species dsgdb9nsd_000018 \
    --resource tianhe_a --route openmm --seeds 1 --prod-ps 20 --debug

# production
python -u scripts/production/s0_E_branchB_parsl.py --edges \
    --resource tianhe_a --route openmm

# then collect, back on the CPU cluster
python -u scripts/production/s0_E_branchB_collect_parsl.py --edges \
    --resource tianhe_cpu --tag prod
```

or without Parsl:

```bash
bash hpc/slurm/submit_branchB_tianhe_a.sh a_debug temp     # smoke
bash hpc/slurm/submit_branchB_tianhe_a.sh prod ai          # production
TAG=prod yhbatch hpc/slurm/branchB_collect.slurm
```

| | where | layout |
|---|---|---|
| trajectories | TianheXY-A, `ai`, 7 days | 56 workers/node, **7 sharing each card**, 5 nodes = 280 concurrent |
| trajectories | TianheXY-AI, `h100x` | 14 workers, all on the one allocated card |
| collection | TianheXY-C, `deimos` | 64 × 1 core, **one node** |

One molecule end to end on a single h100x allocation is
[`examples/02_qha_openmm_acetone`](../examples/02_qha_openmm_acetone/README.md).

**Cost is unmeasured on a card.** 500 ps at the only figure this repository has —
96.1 s/ps, one CPU thread, uncontended — is ~13 h per trajectory, and the only GPU number
is 3.5× *slower* than CPU on a T400. Read `seconds_per_ps_this_run` from the smoke run.

---

## 7. Where the answer lands

```
$S0_RUNS_ROOT/qha/<tag>/<species>/basinNN/seedNN/frames.npy   the trajectory
                                                 /meta.json    protocol + provenance
analysis/qha/<tag>/                                            T*S, spectrum, criteria
analysis/qha/<tag>/collect/<species>.json                      the completion marker
```

```python
from openqha.quasi_harmonic import qha, basin_residence as br
import numpy as np, json

frames = np.load("frames.npy")
meta = json.load(open("meta.json"))
rec = qha.analyse(frames, meta["masses_amu"], meta=meta)
res = br.basin_residence(frames, meta["symbols"])
print(rec["entropy"]["TS_QH_kcal"], res["stayed_in_one_basin"])
print(br.interpret_saturation(qha.saturation_curve(frames, meta["masses_amu"]), res))
```

**Read `stayed_in_one_basin` before reading T·S.** If it is false, the number is not an
intra-basin entropy, however well-converged it looks.

---

## 8. Open

1. **The dynamics stage has not been run.** Every setting in §4 is chosen on the
   estimator measurement plus argument; the basin-residence numbers do not exist yet.
2. **The friction scan has not been run** on the extended variant list.
3. **No GPU cost measurement** for branch B anywhere.
4. 500 vs 1000 ps is not settled — it is a range until the basin check picks a point,
   and the point may differ per molecule.
5. Whether `analysis_atoms: heavy` should become the default for the *correction* (as
   opposed to the absolute entropy) is open, and needs the conformer-difference numbers,
   not the single-conformer ones above.
