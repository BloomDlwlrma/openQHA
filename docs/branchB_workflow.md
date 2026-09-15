# Branch B — quasi-harmonic free energy from molecular dynamics

Branch B turns a molecular dynamics trajectory into a vibrational entropy, and from there
into `G - E_el`. It is the alternative to branch C's Hessian route, and the two share
their translational, rotational and electronic terms exactly — so the difference between
them is the vibrational treatment and nothing else.

This document is the operational one. The reasoning lives in
`.mem/plan/plan_B_quasi-harmonic-analysis.md`; the tutorial that runs it end to end is
`docs/tutorials/T01b_openQHA_Practice_BranchB_QHA.ipynb`.

---

## 1. What is computed

From a trajectory of superimposed configurations,

```
C   = <(x - <x>)(x - <x>)^T>          mass-weighted, 3N x 3N
nu_k = (1 / 2 pi c) sqrt(k_B T / lambda_k)
S    = sum over modes of the quantum harmonic oscillator entropy at nu_k
```

Two things about that are worth holding onto because they decide almost everything else:

* **It is a variance over CONFIGURATIONS.** There is no time correlation anywhere in it.
  Claims about velocity autocorrelation or the vibrational density of states do not
  transfer to this number, and `openqha/vdos.py` — which did read velocities — is retired.
* **The temperature in it is the CONFIGURED one, not the measured one.** A trajectory that
  is 10 % hot inflates every `lambda`, deflates every `nu`, and raises the entropy in a
  fixed direction, with nothing in the run looking wrong.

The second point is why this branch has an identity assertion rather than a checklist.

---

## 2. The three prohibitions, and the fourth and fifth

`openqha.qha.assert_trajectory_identity` refuses a trajectory rather than warning about
it. Each entry is something that produces a perfectly healthy-looking number.

| refused | why |
|---|---|
| a bias potential | metadynamics does not sample the Boltzmann distribution of the true surface; `lambda` too large, `nu` too low, entropy too HIGH |
| any constraint | SHAKE **removes** a degree of freedom rather than softening it, so the covariance is identically zero along it and the count of non-zero eigenvalues cannot reach `3N-6` |
| repartitioned hydrogen mass | `nu ~ 1/sqrt(mu)`; 2 amu turns a 3000 cm⁻¹ C–H stretch into roughly 2200 |
| `fixcm=True` (ASE Langevin only) | measured 429.3 ± 10.1 K when 298.15 K was asked for on a 10-atom molecule, costing +0.6991 kcal/mol on `T*S` |
| a Nosé–Hoover coupling time above 20 fs, or a chain shorter than 3 | see §4 |

That is also why a CREST trajectory can never reach a thermodynamic number here: it
violates the first three at once.

---

## 3. Two production routes, on purpose

Branch B produces trajectories two ways. Until 2026-09-14 they wrote the **same**
`frames.npy` + `meta.json` contract and were read by the **same** analysis, so a
disagreement between them was a measurement rather than a mystery. Since then the OpenMM
route writes OpenMM's own files into `<molecule>/md_openmm/basinNN/` and the analysis reads
those (`docs/output_inventory.md` section 6); the ASE route still writes the old contract
and is NOT read by collect until it is brought over -- an open item, not a decision.

| | OpenMM route (**primary**) | ASE route |
|---|---|---|
| driver | `scripts/production/s0_B_qha_trajectory_openmm.py` | `scripts/production/s0_B_qha_trajectory.py` |
| environment | `openqha` | `openqha` — **one environment since 2026-09-05** |
| integrator | `openmmtools.NoseHooverChainVelocityVerletIntegrator` | `ase.md.nose_hoover_chain.NoseHooverChainNVT` |
| chain propagation | 5-term Yoshida–Suzuki, 5 MTS (openmmtools' defaults) | 3-term YS, `tloop` MTS — **not adjustable** |
| MACE reaches it via | `openqha/quasi_harmonic/openmm_mace.py`, a traced `TorchForce` | `mace.calculators.MACECalculator` |
| neighbour list | **complete graph**, built here; no neighbour search at all | MACE's own, with `openqha/potentials/mace_patch.py` applied |
| resumes bit-identically | **no** — the chain's extended variables are not checkpointed | **yes** — `state.npz` carries positions and momenta |

Both are enabled in `hpc/configs/master.json` on purpose: they write the same product and
are read by the same analysis, so running both makes their agreement a measurement.
Prefer the **ASE** route for the saturation benchmark, because that one needs prefixes of
a trajectory that resumes bit-identically under a queue limit.

### One environment

Branch B used to need two, because `openmm-torch` pins the pytorch it was compiled
against. On 2026-09-05 that pin was accepted — pytorch 2.13.0 → 2.12.1, numpy 2.4.6 →
1.26.4 — and **the cost was measured, not assumed**: on identical geometry, weights and
frames, energy, forces, every Hessian frequency, `T*S` on 2000 fixed frames and every
quasi-harmonic frequency all came back **bit-identical**
(`scripts/calibration/s0_B_stack_fingerprint.py`).

Take that fingerprint before and after any future change to the environment files. A stack
change whose cost is unmeasured is a stack change nobody can defend.

```bash
# OpenMM route -- primary
python scripts/production/s0_B_qha_trajectory_openmm.py --species dsgdb9nsd_000018 \
    --tag omm01 --seeds 3 --prod-ps 25

# ASE route -- the second implementation, and the one that resumes bit-identically
python scripts/production/s0_B_qha_trajectory.py --species dsgdb9nsd_000018 \
    --tag prod01 --seeds 3 --prod-ps 25

# one analysis reads either
python scripts/production/s0_B_qha_analyse.py --species dsgdb9nsd_000018 --tag prod01

# what this installation can and cannot do, and what each absence costs
python -c "from openqha import capabilities; print(capabilities.summary())"
```

### The centre of mass runs free

`PIN_CENTRE_OF_MASS` is **False**. Three measurements settled it:

* the installed MACE is translation invariant with no patch at all — ΔE = 0.000000
  kcal/mol out to 100 000 Å;
* the OpenMM route builds a complete graph, so it has no box to anchor;
* `repeatC_free` (pinning off) survived **8/8** and agreed with the pinned run to ~1e-6 Å,
  and `repeatD` (start shifted by 1e-14 Å, an exact symmetry) also survived **8/8** with a
  maximum divergence of 2.5e-11 Å over 3 ps.

Under Nosé–Hoover in ASE it is not merely unnecessary but **impossible**: the integrator
caches positions and momenta and writes them back every step, so an attached re-centring
observer is silently overwritten. The driver refuses `--pin-com` there rather than
recording a pinning that did not happen.

### Why the OpenMM route needs no neighbour search

Branch B's molecules have 10 to 19 atoms. At that size the exact neighbour list **is** the
complete graph, and pairs beyond `r_max` contribute exactly zero because MACE's radial
cutoff function is exactly zero there. Measured on acetone, scaling the molecule up so
that most pairs fall outside the 5 Å cutoff:

| scale | max pair | filtered edges | complete edges | ΔE | max &#124;ΔF&#124; |
|---|---|---|---|---|---|
| 1.0 | 4.295 Å | 90 | 90 | 0.000e+00 eV | 6.939e-17 |
| 1.6 | 6.873 Å | 72 | 90 | 0.000e+00 | 0.000e+00 |
| 2.0 | 8.591 Å | 46 | 90 | 0.000e+00 | 0.000e+00 |

Bit-identical with half the edges outside the cutoff. That is what makes the graph static,
which is what makes `torch.jit.trace` sound — and tracing is required because
`torch.jit.script` **fails** on this model (e3nn 0.4.4 is not scriptable). It also means
this route cannot have the origin-anchoring defect: there is no box to anchor.

`s0_B_qha_trajectory_openmm.py` refuses to run if the OpenMM force does not reproduce the
ASE calculator. Measured: ΔE = 0.000e+00 eV, max &#124;ΔF&#124; = 4.8e-15 eV/Å.

---

## 4. The thermostat, and the number that decides it

Branch B runs a **Nosé–Hoover chain in NVT**. The coupling time is the whole decision, and
it is not a preference — it is measured against a closed form.

On a harmonic surface built from the production potential's own analytic Hessian,
`<q_k^2> = k_B T / omega_k^2` holds exactly, so a correctly sampled trajectory must return
the Hessian's own frequencies. `scripts/calibration/s0_B_thermostat_choice.py` measures
each thermostat's **bias against that**, not against the others.

200 ps × 4 seeds, acetone, lowest true mode **79.69 cm⁻¹**:

| `tdamp` | `T*S` error / kcal·mol⁻¹ | lowest recovered mode | 2⟨KE⟩/kT (3N = 30) | 2⟨U⟩/kT (3N−3 = 27) |
|---|---|---|---|---|
| 20 fs | −0.093 | 78.6 | 30.00 | 19.1 |
| 100 fs | −1.150 | **179.0** | 30.15 | 18.4 |
| 419 fs | −1.228 | **229.0** | 29.94 | 18.1 |
| 1000 fs | −1.221 | 227.9 | 30.17 | 18.3 |

Read the last two columns together. **The kinetic temperature is correct in every row.**
It is the configurational distribution that collapses — and that is the only thing
quasi-harmonic analysis reads. A temperature check cannot see this failure.

**20 fs is not our number.** It is `openmmtools`' documented default
(`collision_frequency = 50/ps`), and the same 50 ps⁻¹ appears in the QHA reference paper
this branch cites (`rinaldo2003`: *"a friction coefficient of 50 ps⁻¹ was employed for each
atom"*, and *"a piston collision frequency of 50 ps⁻¹"*).

**The cost is real and is recorded.** At 500 ps × 6 seeds the adopted setting measures
**−0.128 ± 0.064** kcal/mol, against Langevin's **+0.004 ± 0.015**. That is inside the
1.0 kcal/mol stage-0 target and outside the 0.10 kcal/mol tooling budget. It is a
known-cost decision (`S0-B-25`), not a free one.

### Three traps

* **GROMACS silently degrades it.** With the default leap-frog `integrator = md`,
  `grompp` prints *"leapfrog does not yet support Nose-Hoover chains, nhchainlength reset
  to 1"* — and `mdout.mdp` still records the length you asked for. Chain length 1 is plain
  Nosé–Hoover, whose configurational variance ratio measured **2.591** against a canonical
  1. `configs/mdp/s0_qha_production.mdp` therefore specifies `integrator = md-vv`.
* **ASE and OpenMM are not the same propagator.** ASE is hard-wired to a 3-term
  Yoshida–Suzuki decomposition, OpenMM defaults to 7. At 100 fs coupling the measured bias
  differed by a factor of four: −1.150 (ASE) against −0.312 (OpenMM).
* **`Stationary` + `ZeroRotation` do nothing under ASE's chain.** `NoseHooverChainNVT`
  caches positions and momenta at construction, propagates its own copies, and writes them
  back over the atoms every step — it never reads them back. An attached observer that
  modifies the atoms is silently overwritten. Measured: a run with re-centring attached was
  **bit-identical** to one without, with the centre of mass still 1422 Å away. The driver
  refuses `--pin-com` under Nosé–Hoover rather than recording a pinning that did not happen.

---

## 5. Acceptance criteria

`scripts/production/s0_B_qha_analyse.py` prints all of them and returns non-zero if any
fails. On the first real 25 ps × 3-seed run (acetone, MACE-OFF23_medium), 10 of 11 passed:

| # | criterion | measured |
|---|---|---|
| 1 | saturation: last doubling below 0.3 kcal/mol | **FAIL, +0.4684** — 25 ps is not converged |
| 2 | an independent **mass-weighted** implementation reproduces `T*S` to 0.05 kcal/mol | PASS, **1.27e-04** (`gmx covar -mwa`) and **8.02e-09** (MDAnalysis fitted our way) |
| 4 | exactly the rigid modes removed, cleanly separated | PASS, separation ratio 1.6e+13 |
| 5 | seed-to-seed noise floor below 1.0 kcal/mol | PASS, **0.0435** |
| 6 | `S_QH <= S_Schlitter` (analytic; a failure is a bug) | PASS |
| 7 | translation/rotation/electronic identical to the Hessian route | PASS |
| 8 | symmetry number declared, never derived | PASS, sigma = 2 |
| 9 | non-zero eigenvalues == `3N-6` | PASS, 24 of 24 |
| 10 | per-batch convergence: the mode COUNT must not still be growing | PASS |
| 11 | the identity assertion | PASS |

Criterion 1 failing is the system working. `andricioaei2001` needed ~4 ns where 200 ps was
not converged; 25 ps was never going to be enough, and the criterion is what says so
instead of a footnote.

**Criterion 2 needs `gmx` on `PATH` or in a sibling conda environment.** It lives in
`qm9fe` here, and `openqha/gmx_io.gmx_binary()` searches for it — because this criterion
once reported "no comparison produced" for no reason but which shell was active.

### GROMACS is an extension

`openqha/extensions/gromacs.py`, on the ACEsuit/mace extension pattern
(`mace/modules/extensions.py`, `tests/extensions/`). No branch B number depends on it.
The contract is in `openqha/capabilities.py`:

* locally, a missing extension **skips**;
* a run that declares it — `S0_REQUIRE_CAPS=gromacs` — **fails** instead.

That second line exists because criterion 2 twice reported "no comparison produced" for
reasons it does not measure, and a criterion that goes quiet instead of red teaches people
to ignore it.

```bash
conda install -c conda-forge gromacs      # activation hook fails under `set -u`
S0_REQUIRE_CAPS=gromacs python tests/run_tests.py
```

### The cross-checks are not equivalent, and the difference decomposes

Same 25 ps trajectory, 3125 frames, each library run as it comes and with our protocol
imposed:

| implementation | mass-weighted | reference | `T*S` | difference |
|---|---|---|---|---|
| `openqha.qha` (ours) | yes | iterated mean | 5.799829 | — |
| **MDAnalysis 2.10.0, iterated** | yes | iterated mean | 5.799829 | **+8.020e-09** |
| MDAnalysis 2.10.0, as it comes | yes | one frame | 5.828531 | +2.870e-02 |
| `gmx covar -mwa` | yes | our mean structure | — | +1.27e-04 |
| `mdtraj 1.11.1` | **no** | one frame | 8.207780 | **+2.407951** |

Three separate things, separated:

* **Not mass-weighting costs 2.41 kcal/mol** — 2.4× the whole stage-0 target, from the
  superposition alone. `mdtraj.Trajectory.superpose` cannot be told to weight by mass, so
  it is fitting a different thing and cannot serve as agreement.
* **Not iterating the reference costs 0.029 kcal/mol** — the bias that fitting to one
  arbitrary frame leaves, now a number rather than an argument.
* **The implementation itself costs 8.0e-09 kcal/mol** — our Kabsch/SVD against
  MDAnalysis' Theobald QCP quaternion, with weighting and reference protocol matched.
  Frequencies agree to 1.2e-04 cm⁻¹ over 24 modes, eigenvalues to 1.2e-07 relative. The
  residual is float32: MDAnalysis stores positions in single precision.

Because MDAnalysis needs no external binary, **criterion 2 no longer depends on finding
`gmx` on `PATH`** — it passes on either.

mdtraj also printed, from its C extension, `UNCONVERGED ROTATION MATRIX. RETURNING
IDENTITY=300` — for 300 of 3125 frames its float32 solver returned the identity rotation
instead of a fit, silently as far as the Python API is concerned. Read
`openqha/mdtraj_io.py` before trusting any RMSD-based analysis of these trajectories.

---

## 6. Fan-out

`scripts/production/s0_E_branchB_parsl.py` spreads (basin × seed) over an executor. Cost
anchor, MACE-OFF23_medium, 10 atoms, 1 thread, uncontended: **96.1 s/ps**. The OpenMM route
measured about 100 s/ps on the same molecule.

```bash
python scripts/production/s0_E_branchB_parsl.py --species dsgdb9nsd_000018 \
    --tag prod01 --seeds 3 --prod-ps 25 --config hpc/configs/qha_md.json
```

HPC configuration is in `hpc/`: `hpc/configs/qha_md.json` for the ASE route,
`hpc/configs/qha_md_openmm.json` for the OpenMM route, and one executor per task class in
`hpc/resource_configs/`.

---

## 7. What is still open

* **The trajectory blow-ups have no established cause.** Two explanations were written and
  both were withdrawn by measurement — a short-range hole in the potential, and a
  neighbour list that is not translation invariant. The second defect is real but lives in
  the MACE **develop** tree vendored under stage 2, not in the `mace_torch 0.3.16` that
  the ASE route imports. Do not cite either without new evidence.
## The production protocol (set 2026-09-07)

Everything measured above was taken at **25 ps**, and criterion 1 failed there by
+0.4684 kcal/mol against a 0.3 budget. Those records are kept as they are — they are the
evidence that the criterion works — and the production length is now the published one:

| | value | source |
|---|---|---|
| timestep | 1 fs | Rinaldo & Field, *Biophys. J.* 2003, p2 |
| equilibration | **520 ps** | ibid. — 20 ps was tried and rejected |
| production | **1500 ps** (1.5 ns) | ibid. |
| sampling interval | **0.5 ps** | ibid. |
| frames | **3000** | 1500 / 0.5 |

`configs/branchB_protocol.yaml` is the single source, and **both** trajectory drivers plus
the Parsl driver read it. Until 2026-09-07 they did not: the ASE route ran 50 + 200 ps and
the OpenMM route 2 + 25 ps, while the two are supposed to be an independent implementation
pair of *one* protocol — two implementations of two protocols measure nothing.

**Why 0.5 ps is better than the 8 fs it replaces**, not looser: the quasi-harmonic
covariance is an equal-time average, so the interval aliases nothing — it only decides how
many *independent* samples the estimate is built from. 25 000 frames spaced far inside the
correlation time carry the information of a few hundred.

**The constraint that defeated the source paper cannot reach us.** A covariance estimated
from T frames has at most T non-zero eigenvalues; Rinaldo & Field had 3N = 7758 degrees of
freedom and 3000 frames, so 4758 modes were missing by construction and no trajectory
length would have converged them (their p11). Our molecules are 10–19 atoms: 3N−6 = 24–51
against 3000 frames, a margin of 53–100×. `openqha.qha.min_frames_for()` checks it anyway.

**1.5 ns is where to start asking, not an answer.** Criterion 1 is still measured on the
trajectory that was actually produced.

* **25 ps is not converged.** The production length is set by the saturation curve, not by
  a constant in a file.
* Two rulings in `plan_B` §7 are still the user's: the molecule scope, and whether the
  quasi-harmonic or the rigid-rotor-harmonic number enters the final free energy.
