# `examples/`

What openQHA publishes is **the method**, not this project's eleven edges (plan_D
section 6.2). Someone who installs openQHA wants "give me a molecule, compute its
conformational entropy", not "rerun some group's eleven edges". But the worked example
still has to be here and has to run, because a method library with no runnable example is
the repository-scale version of a criterion nobody has built a failing case for.

## What exists now

| Directory | What it does | State |
|---|---|---|
| `01_crest_composite_acetone/` | acetone through the CREST composite calculator: GFN sampling with MACE refinement over a socket | **runs** |
| `02_qha_openmm_acetone/` | acetone through branch B, both production routes: the closed-form check, the OpenMM force against the ASE one, two trajectories, an independent superposition | **runs** |

```
python examples/02_qha_openmm_acetone/s0_qha_openmm_demo.py
```

It runs the ASE route in whatever interpreter you start it with, and looks for a second
one that has OpenMM, openmm-torch and openmmtools — `S0_OPENMM_PYTHON`, else the `qm9fe`
environment. If it cannot find one, the OpenMM half is **skipped with a message** rather
than quietly omitted.

The trajectories are 3 ps, which is a smoke length and is labelled as one: acceptance
criteria 1 and 5 are refused below 20 ps, because a 0.4 ps run once passed the saturation
criterion and the reason it passed was that it had not begun to rise.

```
python -m openqha.mace_server --socket /tmp/s0_mace_engrad.sock &
S0_MACE_SOCKET=/tmp/s0_mace_engrad.sock \
    python examples/01_crest_composite_acetone/s0_crest_acetone_demo.py
```

It needs CREST on the path and a MACE-OFF model in place — see the engine registry in
`openqha/engine.py`, which searches for the model root and refuses to run on a checksum
it does not recognise.

## Tutorials live in `docs/tutorials/`

Two notebooks, moved there 2026-09-04 because they are documentation rather than
worked examples of this project's own campaign:

| | What it is |
|---|---|
| [`../docs/tutorials/T02_openQHA_Practice_CREST_conformers.ipynb`](../docs/tutorials/T02_openQHA_Practice_CREST_conformers.ipynb) | **Practice.** A conformer search end to end — the published iMTD-GC protocol and its boundary, the composite calculator, and the tighten → deduplicate → Hessian chain that turns an ensemble into a basin list. Worked on `OCCC(=O)CO` (24 conformers → 23 basins) and `OCCO` (10 basins, σ varying between them). |
| [`../docs/tutorials/T03_openQHA_Theory_AD_Hessian_and_PHL.ipynb`](../docs/tutorials/T03_openQHA_Theory_AD_Hessian_and_PHL.ipynb) | **Theory.** Where a Hessian comes from in a MACE model, why the analytic one beats finite differences, and how to supervise curvature with Hessian-vector products instead of Hessians — Projected Hessian Learning, with the two corrections it needs and a variance criterion that can fail. |

```bash
export S0_CREST_BIN=/path/to/crest            # T02 only
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
jupyter lab docs/tutorials/
```

Both find the repository root by walking up for `openqha/__init__.py`, so neither
depends on how deep it sits — which is why moving them cost nothing. Both execute
against the repository's own modules, so every number they print is the number the
production pipeline would get, and re-running them after a change is worth doing:
executing T03 contradicted four statements in its own prose, and executing T02
contradicted a fifth.

## What does not exist yet, stated plainly

plan_D section 6.3 names two examples. Neither is complete:

- **`01_single_molecule/`** — one molecule from SMILES all the way to `S_QH`. The acetone
  example above covers the conformer-search end of that chain and stops before the
  molecular dynamics and the quasi-harmonic analysis. Branch B's driver
  (`scripts/production/s0_B_qha_trajectory.py`) and analysis
  (`scripts/production/s0_B_qha_analyse.py`) exist; the example that ties them together
  does not.
- **`02_qm9_isomerisation/`** — the eleven-edge worked example of this project.

They are listed here rather than left out so that the gap is visible. `git init` is
gated on both of these running end to end (plan_D section 7.3), so this list is the
remaining distance to that gate.

## Data

What ships with the repository is the reference geometries of 7 species (11 KB) and a
7-row excerpt of the index, which is what lets package 2 reproduce with no external data
(`D0-41`). The full QM9 set is placed by `scripts/tooling/s0_prepare_data.py` and does not
go into version control.
