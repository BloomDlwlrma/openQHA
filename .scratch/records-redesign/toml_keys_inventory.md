# Every key the three TOML records carry today, and what each is for

Taken from a real local run on 2026-09-15 (propanal, `dsgdb9nsd_000035`, OpenMM route,
setting `default`). This is the fact sheet for the records redesign (grilling round 1).
Files read: `_records/basins.toml` (561 lines), `_records/md_openmm/default/basin00/md.toml`
(167 lines), `_records/md_openmm/default/ensemble.toml` (56 lines), and CREST's own
`crest/input.toml` (28 lines) for comparison.

Each key is classed as one of:

- **input**: a setting we chose before the run (what CREST's `input.toml` and ORCA's `.inp` hold)
- **result**: a number produced by the run that a LATER STEP reads back
- **report**: a number or verdict produced by the run that only a human reads
- **provenance**: which software, which weights, which machine
- **pointer**: a path to a file that already exists in the molecule tree
- **duplicate**: the same value written a second time in the same file

## 0. What CREST itself writes as TOML (`crest/input.toml`, 28 lines)

| key | class | what it is |
|---|---|---|
| `input`, `runtype`, `threads` | input | the xyz, iMTD-GC, thread count |
| `[calculation] optlev` | input | optimisation level |
| `[[calculation.level]] method = gfn2` | input | the workhorse |
| `[[calculation.level]] method = generic, binary, gradtype, gradfile, refine` | input | the MACE engrad client |
| `[dynamics] tstep, shake, hmass` | input | the published dynamics protocol |

Every key is an input. CREST writes the results to `crest.out`, `crest.energies`,
`crest_conformers.xyz`. That is the model the user is pointing at.

## 1. `basins.toml` (branch A record; written by the branch A pipeline; read by the ensemble report, 02a/02c/02d, the branch A parsl driver)

### Top level (13 keys)

| key | class | what it is |
|---|---|---|
| `generated_by` | provenance | the script name |
| `branch` | provenance | "A" |
| `qm9_index`, `label`, `name`, `identified_by`, `smiles` | input | which molecule |
| `composite_notation` | provenance | "RI-MP2/cc-pVTZ // MACE-OFF23_medium", the paper's notation |
| `protocol_source` | provenance | the two citations for SHAKE/5 fs/2 amu and iMTD-GC |
| `wall_seconds_total` | report | wall of the whole branch A run |
| `all_criteria_passed` | result | the one boolean the parsl driver reads to say "done" |
| `molecule_dir`, `output_dir` | pointer | the molecule directory and `_records/` |
| `tag` | input | the tag the run was made under |

### `[engine]` (10 keys + `[engine.neighbour_list_patch]` 4 keys + `[engine.neighbour_list_patch.installed]` 6 keys)

| key | class | what it is |
|---|---|---|
| `engine`, `source`, `note` | provenance | MACE-OFF23_medium, the arXiv link, the "production default since" note |
| `weights_path`, `bytes` | provenance | the `.model` file and its size |
| `interface`, `mace_torch_version`, `mace_module_path`, `torch_version`, `dtype` | provenance | the software stack |
| `neighbour_list_patch.applied`, `.sites`, `.what`, `.why` | provenance | the monkey-patch we apply to MACE's neighbour search |
| `neighbour_list_patch.installed.module_path`, `.sha256`, `.sizing`, `.defect_present`, `.n_edges`, `.probe_shift_A` | provenance | the probe that decides whether the installed MACE has the origin-anchoring defect |

Written identically into `md.toml` (see section 2). Twenty keys, none read by a later step.

### `[crest_version]` (2 keys)

`version`, `commit`: provenance. Written a second time as `[crest.crest]`.

### `[settings]` (16 keys)

| key | class | what it is |
|---|---|---|
| `workhorse`, `refine`, `runtype`, `optlev`, `shake`, `tstep_fs`, `hydrogen_mass_amu`, `threads` | input | what we asked CREST to do (the same values are in `crest/input.toml`) |
| `fmax_eV_A` | input | the tightening convergence criterion |
| `dedup_rmsd_A`, `dedup_criterion`, `dedup_ethr_kcal`, `dedup_bthr_relative` | input | the three-fold CREGEN deduplication thresholds |
| `hessian_mode` | input | analytic or finite difference |
| `temperature_K`, `symmetry_tolerance_A` | input | the thermochemistry temperature and the sigma detection tolerance |

The only block in this file that is what CREST's `input.toml` is: the settings.
`branchA` criterion 1 and 11 compare these against `crest/input.toml`.

### `[gate]` (5 keys)

`smiles`, `passed`, `reason`, `gates_enabled`, `smiles_source`: report. The molecule
filter (F0/F1/F3/F4/F5/F7) verdict before CREST ran.

### `[crest]` (21 keys + `[crest.crest]` 2 + `[crest.settings]` 9 + `[crest.products]` 3)

| key | class | what it is |
|---|---|---|
| `workdir`, `ran_in` | pointer | `crest/` in the molecule tree, and the node-local scratch it ran in |
| `binary` | provenance | the crest executable |
| `seconds`, `wall_seconds`, `returncode`, `terminated_normally`, `ok` | report | how the run ended (`seconds` and `wall_seconds` are the same number twice) |
| `n_terminated_early`, `n_completed_successfully`, `total_engrad_calls` | report | counts parsed from `crest.out` |
| `n_cregen_nan`, `n_energies_unparsable`, `n_nonfinite_energies`, `n_zero_energies` | report | sanity counts from `crest.energies` |
| `n_conformers`, `energy_spread` | report | what CREST reported |
| `shake_used`, `used_shake_fallback` | result | which SHAKE actually ran (criterion 10 reads it) |
| `reused_scratch`, `wall_is_valid_cost` | report | criterion 2's inputs |
| `[crest.crest] version, commit` | duplicate | of `[crest_version]` |
| `[crest.settings]` 9 keys | duplicate | of `[settings]`, plus `backend`, `engine_client` (the engrad script path) |
| `[crest.products]` 3 keys | report | byte sizes of `crest_conformers.xyz`, `crest_best.xyz`, `crest_rotamers.xyz` |

### `[census]` (29 keys + 6 sub-tables)

| key | class | what it is |
|---|---|---|
| `name`, `smiles`, `source` | duplicate | of the top level |
| `n_frames_in`, `n_frames_from_crest`, `n_reference_geometries_pooled` | report | how many frames went into tightening |
| `n_not_converged`, `n_graph_changed`, `max_residual_force_eV_A`, `fmax_criterion_eV_A` | report | tightening outcome |
| `opt_steps_per_frame`, `opt_steps_total` | report | optimiser steps |
| `tightened_energies_eV` | report | energy of every frame after tightening |
| `dedup_threshold_A`, `dedup_criterion`, `dedup_ethr_kcal`, `dedup_bthr_relative` | duplicate | of `[settings]` |
| `n_merges_blocked_by_energy`, `n_merges_blocked_by_rotational`, `dedup_blocked_note`, `dedup_metric`, `merge_energy_warnings` | report | deduplication outcome (criterion 12 reads the counts) |
| `n_saddles_rejected`, `saddles` | report | frames rejected for imaginary modes |
| `n_basins`, `basin_conformer_ids`, `basin_energies_eV`, `basin_relative_kcal` | result | the basin list (also in `[[basins]]`, so partly duplicate) |
| `temperature_K` | duplicate | of `[settings]` |
| `free_energy_note` | report | a sentence saying sigma is never derived automatically |
| `[census.atom_order_check]` 5 keys | report | atom order identity across frames, and how it was checked |
| `[census.duplicate_map]` | report | which conformer merged into which |
| `[census.populations]` 5 keys | report | Boltzmann weights over the basins at T |
| `[census.hessian.<id>]` 7 keys per basin | report | `n_imaginary`, `hessian_mode`, `n_rigid_modes_removed`, `separation_gap_ratio`, `hessian_asymmetry_eV_A2`, `lowest_frequency_cm_inv`, `frequencies_cm_inv` (the 3N-6 frequencies; `mace/basinNN/hessian.npy` holds the matrix they come from) |
| `[census.crest_vs_repo]` 8 keys | report | CREST's conformer count against our basin count (criterion 7) |
| `[census.reference_geometry_basin]` 3 keys | report | where the QM9 reference geometry landed (criterion 6) |

### `[machine]` (5 keys)

`loadavg_1min/5min/15min`, `cpu_count`, `node`: provenance.

### `[mace]` (3 keys)

`folder`, `basins`, `hessians`: pointer. Lists of the `mace/basinNN/basin.extxyz` and
`hessian.npy` paths, which the tree already spells.

### `[[basins]]` (one table per basin; 7 keys + `symmetry` 23 keys + `symmetry.tolerance_sweep` 6 tables + `thermo` 5 keys + 4 sub-tables)

| key | class | what it is |
|---|---|---|
| `basin_index`, `energy_eV`, `relative_kcal` | result | the basin, its electronic energy (the ensemble report sums these) |
| `electronic_degeneracy`, `electronic_degeneracy_source` | result | g0 and where it came from (declared in config) |
| `lowest_frequency_cm_inv`, `n_imaginary` | report | the Hessian's verdict |
| `symmetry.sigma`, `sigma_source`, `declared_sigma`, `detected_sigma`, `agrees_with_declaration` | result | sigma (the ensemble report reads it) and the declared/detected agreement |
| `symmetry.tolerance_A`, `n_proper`, `n_improper`, `n_graph_automorphisms`, `graph_automorphism_note`, `max_accepted_rmsd_A`, `min_rejected_rmsd_A`, `margin_A`, `sigma_stable_below_tolerance`, `sigma_flip_tolerance_A`, `sigma_stable_over_whole_sweep`, `sigma_values_over_sweep`, `group_closure_defect` | report | the sigma detection diagnostics (criterion 9) |
| `symmetry.pymsym_point_group`, `pymsym_sigma`, `pymsym_note` | report | the pymsym label, never the source of sigma |
| `symmetry.tolerance_sweep."0.01".."0.4"` | report | sigma at six tolerances |
| `thermo.temperature_K`, `pressure_Pa` | duplicate | of `[settings]` / a constant |
| `thermo.A_minus_Eel_kcal`, `pV_kcal`, `G_minus_Eel_kcal` | result | the harmonic G - E_el per basin (the ensemble report reads `G_minus_Eel_kcal`) |
| `thermo.vibrational` 6 keys | report | A_vib, E_vib, S_vib, ZPE, n_modes, lowest frequency |
| `thermo.rotational` 9 keys | report | A_rot, q_rot, sigma, moments, rotational constants and temperatures, E_rot, S_rot, T/theta |
| `thermo.translational` 6 keys | report | the Sackur-Tetrode piece |
| `thermo.electronic` 2 keys | report | A_elec, degeneracy |

### `[[criteria]]` (9 tables, 4 keys each)

`number`, `criterion`, `passed`, `detail`: report. The branch A acceptance checks 1, 11,
2, 4, 6, 7, 9, 10, 12 with a one-line detail each. Only the aggregate
`all_criteria_passed` is read by a program.

## 2. `md.toml` (one trajectory's record; written by the trajectory driver; read by the driver itself on resume, the analysis, the batch driver's summary, 02d)

### Top level (27 keys)

| key | class | what it is |
|---|---|---|
| `hydrogen_mass_amu`, `timestep_fs`, `thermostat`, `thermostat_tdamp_fs`, `thermostat_chain_length`, `temperature_K`, `sample_every_steps`, `frame_spacing_fs`, `ensemble`, `platform`, `centre_of_mass_pinned_to_origin` | input | the dynamics settings; the analysis asserts identity against them |
| `route`, `setting` | input | openmm/ase and the setting name |
| `qm9_index`, `basin_index` | input | which molecule, which basin |
| `seed`, `seed_formula` | result | the seed (the driver rereads it on resume) |
| `symbols`, `masses_amu` | result | the masses the driver integrated with (the analysis uses them) |
| `source`, `protocol_source`, `protocol_status` | provenance | which module, the Rinaldo & Field citation, "adopted by user ruling" |
| `engine_folder`, `geometry_source` | pointer | `md_openmm/basinNN/` and `mace/` |
| `equilibration_done` | result | resume marker: production may continue without re-equilibrating |

### `[engine_files]` (7 keys)

`start.pdb`, `system.xml`, `integrator.xml`, `traj.dcd`, `state.csv`, `state.xml`,
`state.chk`: pointer. Written a second time as `[production.files]`.

### `[integrator]` (5 keys)

`implementation`, `collision_frequency_per_ps`, `chain_length`, `num_mts`,
`num_yoshidasuzuki`: input (the Nose-Hoover chain parameters).

### `[force]` (12 keys + empty `[force.platform_properties]`)

`model_path`, `cutoff_A`, `n_atoms`, `n_edges`, `neighbour_list`, `dtype`, `traced`,
`trace_device`, `device_constants`, `device_mismatches`, `openmm_platform`,
`n_particles`: provenance of the TorchScript MACE force inside OpenMM. Written a
second time as `[relaxation.force_field]`.

### `[force_check_against_ase]` (15 keys)

The first 8 duplicate `[force]`; `engine`, `energy_traced_eV`,
`max_delta_force_traced_eV_per_A`, `energy_openmm_eV`, `energy_ase_eV`,
`delta_energy_eV`, `max_delta_force_eV_per_A`, `agrees`: report. The check that the
OpenMM force equals the ASE force at the start geometry.

### `[engine]` (20 keys)

Identical to `basins.toml` `[engine]`: provenance, duplicate across files.

### `[relaxation]` (3 keys + `[relaxation.force_field]` 12 keys)

`energy_before_kJ`, `energy_after_kJ`, `energy_drop_kcal`: report. `force_field`:
duplicate of `[force]`.

### `[equilibration]` (6 keys)

`ps`, `wall_seconds`, `potential_first_half_kJ`, `potential_second_half_kJ`,
`temperature_second_half_K`, `drift_kJ`: report.

### `[production]` (19 keys + `[production.engine_files]` 2 + `[production.files]` 7)

| key | class | what it is |
|---|---|---|
| `ps`, `n_frames`, `n_target`, `complete` | result | how long, how many frames, is it done (the analysis and the batch driver read these) |
| `wall_seconds`, `seconds_per_ps`, `seconds_per_frame_this_run`, `seconds_per_ps_this_run` | report | cost (two of these are the same number) |
| `stopped_on_wall_budget`, `resumed`, `n_frames_already_on_disk`, `n_frames_generated_this_run` | report | resume bookkeeping |
| `temperature_mean_K`, `temperature_deviation_K`, `temperature_error_K` | report | thermostat check (the last two are the same number) |
| `centre_of_mass_drift_A` | report | COM drift over production |
| `engine_files.folder`, `engine_files.frames_in_dcd` | pointer / duplicate | the folder again; frames in the DCD (= `n_frames`) |
| `files.*` 7 keys | duplicate | of `[engine_files]` |

### `[[segments]]` (1 table per run that appended frames; 3 keys)

`seed`, `frames_from`, `frames_to`: result. The resume history the driver reads.

## 3. `ensemble.toml` (the answer; written by the ensemble report; read by nothing)

| key | class | what it is |
|---|---|---|
| `species`, `tag`, `basin_tag`, `temperature_K`, `n_basins_branch_a` | input | what was summed |
| `trajectory_root` | pointer | `md_openmm/` |
| `[collect_criteria] passed, total` | report | 7 of 10 |
| `[results.all] n_basins, n_with_entropy, n_electronic_only, temperature_K, F_conf_kcal, partition_function_relative, delta_G_kcal, populations, effective_basins, note, distinct_crossings, symmetry_crossings` | result | F_conf and the populations: the number the whole chain exists for |
| `[results.all.per_basin_detail.<i>] TS_kcal, spread_kcal, n_seeds, n_frames, source, distinct_crossings, symmetry_crossings` | report | per basin |

## 4. The `.dat` tables (collect; read by the ensemble report)

| file | columns | read by |
|---|---|---|
| `collect.trajectories.dat` | species basin seed n_frames TS_QH_kcal TS_Schlitter_kcal S_QH_kcal_per_K A_vib_kcal lowest_frequency_cm_inv highest_frequency_cm_inv n_nonzero expected_modes rigid_ratio saturation_last_doubling_kcal wall_seconds seconds_per_ps | ensemble (`TS_QH_kcal`) |
| `collect.criteria.dat` | species criterion measured passed | ensemble (the count) |
| `collect.assembly.dat` | species basin G_minus_Eel_kcal A_vib_kcal S_vib_kcal_per_K terms_match_hessian_route | nothing |
| `collect.blank.dat` | species basin n_seeds TS_QH_mean_kcal TS_QH_spread_kcal TS_QH_rms_about_mean_kcal standard_error_kcal note | nothing |

## 5. Counts

| file | keys | input | result (read later) | report | provenance | pointer | duplicate |
|---|---|---|---|---|---|---|---|
| `basins.toml` | about 330 (3 basins) | 22 | about 20 | about 200 | about 45 | 6 | about 25 |
| `md.toml` | about 145 | 20 | 12 | 40 | 45 | 9 | about 30 |
| `ensemble.toml` | 45 | 6 | 12 | 26 | 0 | 1 | 0 |

What a program reads back later is about one tenth of what is written. The rest is the
`.out` report's material (it is also printed there in prose) or provenance written by
every step alike.

## 6. What is written per BATCH today (not per calculation)

| file | written by | read by | content |
|---|---|---|---|
| `<root>/<tag>/_records/branchB_parsl_summary.json` | the branch B parsl driver (one Slurm job, many trajectories) | nothing | resource config, route, the task list, each task's return code and seconds |
| `<root>/<tag>/_records/collect_batch.json` | the collect parsl driver | nothing | the plan and every molecule's collect result |
| `<repo>/analysis/branchE/<tag>/batch.json` | the branch A parsl driver (campaign) | nothing | the plan and per-molecule results |
| `<root>/<tag>/_records/parsl/<job>.<pid>/` | parsl | human | parsl's own logs |
| `<repo>/logs/<jobname>_<jobid>.{out,err}` | Slurm | human | the job's stdout/stderr |
| `_records/md_<route>/<setting>/basinNN/driver.log` | the branch B parsl driver | human | stdout of ONE trajectory driver subprocess (per calculation, but written by the batch) |

No file today says "this Slurm job ran these N calculations and here is how each ended"
in the `.out` + `.toml` form; the three JSON files do that in three different shapes.
