# `configs/mdp/` — MD protocols in `.mdp` form

> **Read this first, or you will misread what these files are.**
>
> **openQHA's potential is MACE-OFF, and no production run in this repository goes
> through GROMACS.** These `.mdp` files are **protocol specifications**: every dynamics
> parameter written down once, in a format that can be compared line by line — against
> the reference literature, against each other, and against the ASE implementations that
> actually run.
>
> GROMACS *can* read them (it is used as an independent cross-check of `S_QH` in branch
> B), but **production does not go that way**.

## Why the format was chosen

Because the branches disagree about dynamics, and the disagreement is the single most
expensive thing to get wrong in this project.

Branch A and branch B both run molecular dynamics on the same molecules, in the same
directories, minutes apart — with **settings that must never be exchanged**. Prose
saying so is easy to skim. Two files in one format, with a column for each, is not.

| | `s0_branchA_crest_metadynamics.mdp` | `s0_qha_production.mdp` |
|---|---|---|
| branch | A — conformer search | B — quasi-harmonic analysis |
| purpose | cross barriers | sample one basin at equilibrium |
| product | **geometries** | **a trajectory** |
| bias | metadynamics, 12 runs | **none** |
| `dt` | **5 fs** | **1 fs** |
| `constraints` | **all-bonds** (SHAKE) | **none** |
| hydrogen mass | **2 amu** (deuterium) | real, 1.008 amu |
| thermostat | Berendsen, `tau-t` 0.5 ps | v-rescale |
| `ref-t` | 300 K (realised 350–500 K) | 298.15 K, and checked |
| what runs it | CREST | ASE (`s0_B_qha_trajectory.py`) |
| may be read off it | which minima exist | frequencies, entropy, free energy |

**Read the last two rows together.** Branch A's trajectory cannot give you a frequency;
branch B's cannot tell you which minima exist. Neither is a deficiency — they are
different instruments, and the only error available is using one for the other.

## The three prohibitions

Branch B's file opens with three prohibitions rather than three settings, because each
of branch A's settings is individually fatal to a quasi-harmonic analysis:

1. **Bias potential** → metadynamics does not sample the Boltzmann distribution of the
   true surface. The covariance eigenvalues come out too large, so
   `nu = sqrt(kB*T/lambda)` comes out too low, so **entropy comes out too high**. The
   direction is fixed: it is a bias, not noise, and averaging does not remove it.
2. **Constraints** → SHAKE does not soften a degree of freedom, it *removes* it. The
   covariance is identically zero along every constrained bond, so the `3N-6` non-zero
   eigenvalues **cannot all be filled**. The count itself breaks.
3. **Deuterium mass** → C–H stretches drop from ~3000 to ~2200 cm⁻¹. Every vibrational
   term is then wrong, by an amount that still looks plausible.

The handover between the branches is **the basin geometries, never the trajectory**.

## Files

| file | branch | status |
|---|---|---|
| `s0_branchA_crest_metadynamics.mdp` | A | specification; CREST runs it |
| `s0_qha_production.mdp` | B | specification; ASE runs it, GROMACS cross-checks it |
| `_superseded/s0_stage1_equil_sd.mdp` | retired | package 3, stochastic-dynamics equilibration |
| `_superseded/s0_stage2_nvt_dos.mdp` | retired | package 3, density-of-states sampling |

The two retired files belonged to the multi-molecule near-critical gas box, a route
struck out. They are kept rather than deleted so that a retired protocol's
parameters stay quotable — the same rule that keeps overturned measurements in the
config files.

## Where the parameters actually live

The `.mdp` files are a *view*. The authority is:

| branch | authority |
|---|---|
| A | `configs/conformers.yaml`, section `crest:` |
| B | `configs/openqha.yaml`, section `package5:` |

If a value differs between an `.mdp` and its YAML, **the YAML is right and the `.mdp` is
a stale copy** — the code never reads these files.

## One thing the format cannot express

GROMACS has no key for *"add the current structure to an RMSD-space bias potential every
1 ps"*. Branch A's central mechanism has no `.mdp` equivalent at all, and the published
values are written into the file as comments instead.

That gap is worth noticing rather than papering over: **the one setting that most
distinguishes the two protocols is the one the shared format cannot hold.**
