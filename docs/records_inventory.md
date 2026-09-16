# Everything this repository writes that is not an engine file, and what reads it

> **Superseded on 2026-09-15** by step 2: the records below became `branchA.out` +
> `basins.toml`, `md.out` + `md.toml`, `collect.out` + four `.dat` files (one Table `collect.dat` since 2026-09-16), `ensemble.out` +
> `ensemble.toml` (see `docs/output_inventory.md` section 7), and on the same day by the
> records redesign (section 8: property-style `.toml`, no setting level, no Batch record).
> This file is kept as the inventory those decisions were taken from.

Taken from the code on 2026-09-15, after tickets 01-10 of the layout change. This is the
input to step 2 (the form and home of the records). "Reader" is a program in this
repository that opens the file; "human" means nothing does.

## 1. In the molecule directory: `<root>/<tag>/<range>/<chunk>/<qid>/_records/`

| file | written by | read by | what it is for |
|---|---|---|---|
| `basins.json` | `s0_A_pipeline.py` | `basins.read_record()` -> ensemble report (relative electronic energies, sigma, degeneracy per basin), 02c, 02d, 02a debug script; `s0_E_branchA_parsl.py` (did the run finish) | the branch A record: gate, CREST settings and counts, census (tightening, dedup, Hessian), the labelled basins, the 11 acceptance criteria, machine and timing. The one source of "which basins, at what energy, with what symmetry number" |
| `basins.xyz` | `s0_A_pipeline.py` | nothing (since ticket 07; the geometries are read from `mace/basinNN/basin.extxyz`) | the basins as one multi-frame xyz with a comment line per basin; a convenience copy of the engine files |
| `<route>/<setting>/basinNN/meta.json` | the trajectory driver (openmm or ase) | `s0_B_qha_analyse.py` (the identity assertion: thermostat, timestep, bias, constraints, hydrogen mass; the masses the driver integrated with), the driver itself (resume: seed, frames on disk), the parsl driver (summary line), 02d (identity assertion) | the trajectory's record: seed, segments, settings, force check against ASE, engine provenance, relaxation, equilibration, production statistics, the engine file names, the job id |
| `<route>/<setting>/basinNN/frames.npy` | the trajectory driver | nothing (kept by ticket 03's text; the DCD / md.traj is the trajectory) | float64 copy of the sampled positions. A duplicate |
| `openmm/<setting>/basinNN/equilibrated.json` | the OpenMM driver | the OpenMM driver (resume after a kill during production, without re-equilibrating) | the equilibration statistics, flushed the moment equilibration ends |
| `ase/<setting>/basinNN/progress.json` | the ASE driver | nothing | frames done / asked, integrity check, wall |
| `<route>/<setting>/basinNN/driver.log` | `s0_E_branchB_parsl.py` | human | stdout + stderr of the trajectory driver subprocess |
| `<route>/<setting>/summary.json` | the trajectory driver | nothing | one line per basin: frames, complete, wall, temperature, COM drift |
| `<route>/<setting>/collect.log` | `s0_B_qha_analyse.py` | human | the analysis report, readable |
| `<route>/<setting>/collect__trajectories.parquet` | `s0_B_qha_analyse.py` | `s0_B_report_ensemble.py` (T*S per basin, the numbers that are summed) | per-trajectory entropies, frequencies, saturation, rates |
| `<route>/<setting>/collect__criteria.parquet` | `s0_B_qha_analyse.py` | `s0_B_report_ensemble.py` (the verdict count printed with F_conf) | the 10 criteria and their verdicts |
| `<route>/<setting>/collect__assembly.parquet`, `collect__blank_control.parquet` | `s0_B_qha_analyse.py` | nothing | G - E_el per basin; the blank-control (seed spread) table |
| `<route>/<setting>/collect.json` | `s0_E_branchB_collect_parsl.py` | `s0_E_branchB_collect_parsl.py` (resume: the completion marker) | the collect driver's summary of one molecule: n_trajectories, criteria passed, command |
| `<route>/<setting>/collect.driver.log` | `s0_E_branchB_collect_parsl.py` | human | stdout + stderr of the analysis subprocess |
| `<route>/<setting>/ensemble.json` | `s0_B_report_ensemble.py` | nothing (it IS the answer) | F_conf over the basins, populations, per-basin detail, the collect verdict count, the crossings |
| `<route>/<setting>/02d_frequency_identity.json` | 02d | nothing (the example's answer) | the four stages of the identity check |
| `<route>/<setting>/basinNN/gmx/` | `s0_B_qha_analyse.py --with-gmx` (off by default) | itself, during the run | GROMACS cross-check working files (`.gro .top .mdp .g96`, `covar` output) |

## 2. Per tag: `<root>/<tag>/_records/`

| file | written by | read by | what it is for |
|---|---|---|---|
| `branchB_parsl_summary.json` | `s0_E_branchB_parsl.py` | nothing | the plan (resource, route, tasks) and every task's return code and seconds |
| `collect_batch.json` | `s0_E_branchB_collect_parsl.py` | nothing | the plan and every molecule's collect result |
| `parsl/<job>.<pid>/` | parsl | human (debugging a block that did not start) | parsl's own run directory: `parsl.log`, submit scripts, executor and worker logs, certificates |

## 3. Still in the repository checkout

| file | written by | read by | what it is for |
|---|---|---|---|
| `analysis/branchE/<tag>/batch.json` | `s0_E_branchA_parsl.py` (the campaign branch A driver, not the chains) | nothing | the batch plan and per-molecule results of a parsl branch A run |
| `logs/<jobname>_<jobid>.{out,err}` | Slurm (`run_chain.sh` sets `--output/--error`) | human | the job's stdout and stderr: the environment banner, every step's output |
| `logs/02d2_<row>_<arrayjob>_<task>.log` | `array.slurm` | human | one chain log per array row |
| `logs/openqha_pair_<jobid>_<tag>.log` | `chain_body.sh` (02ab only) | human | one log per molecule of the pair |
| `logs/trace/<job or pid>/engrad_trace.log` | `s0_mace_engrad.py` when the conf sets `MACE_TRACE` | human (diagnosing a gradient that never came back) | one line per gradient call, both ends of the socket |
| `examples/02d-2_qha_settings_array/generated/<row>.conf`, `manifest` | `submit_array.sh` | `array.slurm`, `chain_body.sh` | the conf each array row sources; the row order |

## 4. Node-local, gone with the node: `$S0_SCRATCH` = `/tmp/<user>/<jobid>/`

| file | written by | read by | what it is for |
|---|---|---|---|
| `s0_mace_pool_<pid>_<i>.sock` | `crest.start_servers` | CREST's engrad client | the MACE server socket |
| `s0_mace_pool_<pid>_<i>.sock.log` | `crest.start_servers` | human | the MACE server's stdout |
| `openqha_crest/<qid>/` | CREST | `s0_A_pipeline.py` (moved into `crest/` when CREST returns) | CREST's working directory while it runs |

## 5. What this says

Three files carry state a program needs later: `basins.json` (the basin list's energies
and symmetry numbers), `meta.json` (the identity assertion and resume), `collect.json`
(collect's completion marker). Two parquet tables carry numbers a later step sums
(`collect__trajectories`, `collect__criteria`). Everything else is either the answer
itself (`ensemble.json`, `02d_*.json`), a duplicate (`basins.xyz`, `frames.npy`,
`summary.json`), or a log nobody opens unless something went wrong.
