# Branch B in production — the whole picture, for developers

What the code is, which function does which piece, how to run it, and how to read what
comes out. Companion to [`branchA_production.md`](branchA_production.md); same shape.

**Nothing in this repository has ever been submitted to Tianhe.** Everything below is
either verified locally (stated where it is) or marked unverified.

---

## 1. What branch B computes, and where the line falls

Branch A enumerates the **basins**. Branch B measures the entropy **inside** one, and the
two meet in an ensemble:

```
        dG_i  =  dE_el,i          branch A   the basin's relative electronic energy
               - T * S_i          branch B   its quasi-harmonic entropy

    F_conf  =  -kT ln sum_i exp(-dG_i / kT)      the molecule's answer
```

`S_i` comes from the **covariance of the atomic positions**, not from a Hessian:

```
    sigma_ij = < (x_i - <x_i>)(x_j - <x_j>) >        i, j = 1 .. 3N
    C        = M^(1/2) sigma M^(1/2)                 mass-weighted
    lambda_k = eigenvalues of C                      amu * A^2
    nu_k     = sqrt(kT / lambda_k) / (2 pi c)        quasi-harmonic frequencies
```

The identity that makes this work is `<q_k^2> = kT / omega_k^2` for a classical harmonic
oscillator, so a correctly sampled trajectory returns the Hessian frequency it was built
from. **That is testable in closed form, and it is what
`qha.synthetic_harmonic_trajectory` exists for.**

```
BRANCH B -- the science. Changing anything here changes the answer.
  configs/branchB_protocol.yaml       HOW LONG, HOW OFTEN, WHICH THERMOSTAT
  openqha/quasi_harmonic/
      qha.py              the estimator: superposition, covariance, spectrum, entropy
      basin_residence.py  did the trajectory stay in its basin? (this gates the rest)
      ensemble.py         per-basin dG -> F_conf, populations, effective basin count
      openmm_mace.py      MACE as an OpenMM force, exact all-pairs graph
      mdtraj_io.py        an independent superposition, for acceptance criterion 2
      vdos.py, perturb.py, torsion_cv.py, gmx_io.py
  openqha/thermochem/thermo.py        partition functions; g_minus_eel
  scripts/production/
      s0_B_qha_trajectory.py          the ASE route          }  one implementation
      s0_B_qha_trajectory_openmm.py   the OpenMM route       }  PAIR of one protocol
      s0_B_qha_analyse.py             per-species analysis and acceptance criteria
      s0_B_report_ensemble.py         branch A + branch B  ->  F_conf

BRANCH E -- the execution layer. Changing anything here must change NO number.
  scripts/production/s0_E_branchB_parsl.py          trajectories, fanned out
  scripts/production/s0_E_branchB_collect_parsl.py  the analysis, fanned out
  hpc/resource_configs/{tianhe_a,tianhe_ai,tianhe_cpu}.py
  examples/run_chain.sh                             the whole chain, local | hpc
```

---

## 2. One basin, end to end — the call chain

`s0_B_qha_trajectory_openmm.py --species X --basins <xyz> --basin K --seed-index S`

```
branch A's basin K                     basin_store.paths_for(X, tag)[1]  ->  .basins.xyz
      |
      v  relax()                       minimise on the SAME force the dynamics will use
      |
      v  openmm_mace.build_system()    MACE as a TorchForce.
      |                                The graph is the COMPLETE graph -- exact for
      |                                10-19 atoms, no neighbour list, so it cannot have
      |                                the origin-anchoring defect mace_patch guards.
      |                                Verified: dE = 0.000e+00 eV vs the ASE calculator.
      v  build_integrator()            openmmtools NoseHooverChainVelocityVerlet at ITS
      |                                OWN defaults -- 50/ps (= 20 fs coupling), chain 5,
      |                                MTS 5, YS 5. Adopted, not chosen.
      v
equilibration                          protocol.equilibration_ps
      |                                drift measured, not assumed
      v
production                             protocol.production_ps, frame every
      |                                protocol.sampling_interval_ps
      |                                flushed every chunk_frames, .part-then-rename
      v
openmm/basinKK/ traj.dcd + state.csv   <molecule>/openmm/basinKK/  (since 2026-09-14;
+ start.pdb, system/integrator.xml,    the record meta.json under _records/md_openmm/<setting>/;
  state.xml, state.chk                 before: frames.npy + meta.json under $S0_RUNS_ROOT/qha/)
      |
      v  qha.assert_trajectory_identity(meta)      REFUSES a biased or constrained run,
      |                                            a coupling looser than 20 fs, a chain
      |                                            shorter than 3, or a fake H mass
      v  qha.superimpose()             mass-weighted, iterated to the mean structure
      v  qha.mass_weighted_covariance()
      v  qha.spectrum()                Eckart projection; rank check against 3N-6
      v  qha.entropy()                 quantum oscillator formula on classical fluctuations
      |
      +--> basin_residence.basin_residence()   did it stay in the basin?
      +--> qha.saturation_curve()              is T*S still rising?
      +--> basin_residence.interpret_saturation(saturation, residence)
```

### The functions a developer actually calls

| function | in | what it gives you |
|---|---|---|
| `qha.protocol(cfg)` | `quasi_harmonic/qha.py` | the length/interval/timestep, from the one config |
| `qha.analyse(frames, masses, meta)` | | superposition → covariance → spectrum → entropy |
| `qha.saturation_curve(frames, masses)` | | T·S against trajectory length (criterion 1) |
| `qha.min_frames_for(n_atoms)` | | the rank floor: 3N |
| `qha.synthetic_harmonic_trajectory(H, m, x)` | | exact canonical sampling; truth in closed form |
| `basin_residence.basin_residence(frames, syms)` | `basin_residence.py` | crossing counts, both kinds |
| `basin_residence.interpret_saturation(sat, res)` | | what a failed criterion 1 *means* |
| `basin_residence.heavy_atom_mask(syms)` | | the non-hydrogen selection |
| `ensemble.conformational_free_energy(basins, T)` | `ensemble.py` | F_conf, ΔG, populations |
| `ensemble.effective_basin_count(pops)` | | how many basins carry the answer |

---

## 3. The protocol lives in one file

[`configs/branchB_protocol.yaml`](../configs/branchB_protocol.yaml). **Both trajectory
routes and the Parsl driver read it** through `qha.protocol()`.

| | value | why |
|---|---|---|
| timestep | 1 fs | `D0-C-30` |
| equilibration | 50 ps | a 10-atom molecule in vacuum, not a solvated protein |
| production | 500 ps (range 500–1000) | **set by basin residence**, not by the source paper |
| sampling interval | 1.0 ps (range 0.5–2.0) | decorrelation and storage |
| thermostat | NHC, 20 fs coupling, chain 5 | measured best of a sweep |
| analysis atoms | `all` (`heavy` available) | different quantities — see §5 |

`sample_every_steps` is **derived** (`interval_ps × 1000 / timestep_fs`) and raises if it
is not a whole number. Storing both would let them disagree, and a sampling interval that
disagrees with its own step count is invisible in every product.

> **This is a correction.** The published protein protocol (520 ps + 1.5 ns at 0.5 ps) was
> adopted wholesale on 2026-09-07 and withdrawn the same day. It is the right paper — this
> branch already cites it for the thermostat — but a solvated protein barely explores its
> basin in 1.5 ns, while a 10-atom molecule crosses its torsional barriers constantly.
> Until 2026-09-07 the two routes were also running *different* protocols (50+200 ps
> against 2+25 ps), which made them two implementations of two things.

---

## 4. The check that gates everything: did it stay in the basin?

Quasi-harmonic analysis assumes **one multivariate Gaussian**. A trajectory that crosses a
torsional barrier puts the displacement *between* two minima into the covariance, T·S is
inflated, and nothing else in the chain notices.

`basin_residence` counts two kinds, and only one is a different conformer:

| | what happened | conformer count | Cartesian covariance |
|---|---|---|---|
| `symmetry_equivalent_crossings` | a methyl turned 120° into a copy of itself | unaffected — this is what σ is for | **still inflated** |
| `distinct_basin_crossings` | the molecule is in a different well | wrong | **not an intra-basin entropy** |

**This inverts acceptance criterion 1.** A rising saturation curve has two causes that
demand opposite actions, and they look identical on their own:

```
rising, no crossings   ->  the estimator has not converged   ->  run LONGER
rising, crossings      ->  the trajectory is leaving a basin ->  run SHORTER
```

`interpret_saturation()` refuses to read one without the other.

**Measured on acetone** (30 ps, 0.5 ps sampling): 28 symmetry-equivalent crossings, **zero**
distinct — and all-atom T·S climbing `5.3884 → 5.5844` while heavy-atom fell
`0.5269 → 0.4536`. The swept hydrogens inflate the all-atom covariance; the heavy-atom
estimator drops exactly those atoms.

---

## 5. Atom selection is two different quantities

Closed form on acetone's own MACE Hessian: **all atoms 2.9317**, **heavy only 0.1383**
kcal/mol. The hydrogens carry 18 of the 24 modes. **Never compare them.**

The case for `heavy` here is not the usual X–H one. It is that the atoms it drops are
exactly the ones a freely rotating methyl sweeps around a circle. For a *difference*
between basins the X–H terms largely cancel, so the heavy-atom estimator is
better-conditioned for the correction while being useless as an absolute entropy.

---

## 6. Running it

### The whole chain — one file, two modes

```bash
bash examples/run_chain.sh examples/02b_qha_openmm_propanal/chain.conf                    # local
bash examples/run_chain.sh examples/02b_qha_openmm_propanal/chain.conf ai
bash examples/run_chain.sh examples/02b_qha_openmm_propanal/chain.conf h100x
```

It runs the four production drivers in order and adds no science of its own. **There is no
smoke mode** — a short run is a different quantity that looks like an answer.

The script **submits itself**: invariant `#SBATCH` directives are in its header, and
partition / walltime / `--gpus` are `yhbatch` flags because a `#SBATCH` line cannot be
parameterised. The conf path is passed as the script's **argument** and `source`d by the
job — **not** `--export=ALL`, which would carry the submitting shell's environment into the
job and leave the settings only in a scheduler record.

| | local | `PARTITION=ai` | `PARTITION=h100x` |
|---|---|---|---|
| resource config | `local` | `tianhe_a` | `tianhe_ai` |
| route / platform | ASE / CPU | OpenMM / CUDA | OpenMM / CUDA |
| allocation | — | node, 8 cards, 56 cores, 7 d | **one card**, 14 CPUs, 3 d |
| layout | cores | 56 workers, 7 per card | 14 workers, all on the one card |

### At campaign scale

```bash
python -u scripts/production/s0_E_branchB_parsl.py --edges --resource tianhe_a --route openmm
python -u scripts/production/s0_E_branchB_collect_parsl.py --edges --resource tianhe_cpu
```

`--basins auto` takes the count from branch A's own record, so the number of basins and the
geometries they start from cannot disagree. **Until 2026-09-08 the driver passed no basin
file at all**, so `--basins 3` ran three copies of the QM9 reference geometry under three
indices — an F_conf that looked multi-basin and was not.

---

## 7. Where the answer lands, and the order to read it

Since 2026-09-14 (`docs/output_inventory.md` section 6):

```
<molecule>/md_openmm/basinNN/traj.dcd  start.pdb  state.csv ...        the trajectory (OpenMM's files)
<molecule>/_records/md_openmm/<setting>/basinNN/meta.json               protocol + provenance
<molecule>/_records/md_openmm/<setting>/collect__*.parquet, collect.log  per-molecule analysis
<molecule>/_records/md_openmm/<setting>/ensemble.json                   F_conf, ΔG, populations
<molecule> = <root>/<tag>/<range>/<chunk>/<species>
```

```python
from openqha.quasi_harmonic import qha, basin_residence as br, trajectory_reader
tr = trajectory_reader.read_trajectory(engine_dir, records_dir=records_dir, setting="default")
frames, meta = tr["positions_A"], tr["meta"]
rec = qha.analyse(frames, meta["masses_amu"], meta=meta)
res = br.basin_residence(frames, tr["symbols"])
print(br.interpret_saturation(qha.saturation_curve(frames, meta["masses_amu"]), res))
```

**Read in this order, and stop at the first failure:**

1. `distinct_basin_crossings` — non-zero and it is not an intra-basin entropy.
2. `rank_check.rank_is_full` and `n_frames` against `qha.min_frames_for(n_atoms)`.
3. `effective_basins` — 20 basins of which one holds 99% is a one-basin molecule.
4. `F_conf_kcal` — only if 1–3 are sound.

A basin with **no** entropy is `MISSING`, never zero, and `s0_B_report_ensemble.py` exits
non-zero. "We did not run it" and "its entropy is zero" are different statements.

---

## 8. Calibration tools

| | what it settles | cost |
|---|---|---|
| `examples/02_.../s0_qha_parameter_scan.py --stage estimator` | is the estimator accurate at N frames? Closed form, so bias is *known* | seconds |
| `examples/02_.../s0_debug_realmole.py --conf debug_realmole.conf` | do the settings move the deliverable? Real molecule, all basins, 150-cell grid | hours |
| `scripts/calibration/s0_B_thermostat_choice.py` | which thermostat samples the right configurational distribution | minutes |

The grid costs `n_basins × seeds × n_thermostats` trajectories, not 150: length is a
truncation, interval a stride, atom set a mask — all read off one trajectory.

---

## 9. Verified here, and not

| checked without a cluster | result |
|---|---|
| both routes read one protocol (parser intercepted, not `--help` grepped) | **pass** |
| estimator unbiased to <0.01 kcal/mol, 250→3000 frames, 8 seeds | **pass** |
| all 24 modes recovered at 2 ps spacing | **pass** |
| `ensemble`: a common T·S shift moves F_conf by exactly 0 | **pass** |
| `effective_basin_count` on the limiting cases (1, 2, 4) | **pass** |
| propanal: 3 basins, F_conf = −0.2354 kcal/mol from ΔE_el | **pass** |
| `run_chain.sh` local / ai / h100x / bad partition / no conf | **5/5** |
| branch A end to end on two molecules | **9/9 criteria each** |

**Unverified:** every cost on a card; whether 500 or 1000 ps is right for a given molecule
(it is a range until the basin check picks a point); the full parameter grid; and anything
at all on Tianhe.
