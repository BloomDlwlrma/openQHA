# What the chain writes, file by file

Sections 1-5: the state before 2026-09-14 (kept as the baseline). Section 6: what replaced it.

Inventory taken from the code on 2026-09-14 (`s0_A_pipeline.py`, `crest.py`,
`basin_store.py`, `s0_E_branchB_parsl.py`, `s0_B_qha_trajectory_openmm.py`,
`s0_E_branchB_collect_parsl.py`, `s0_B_qha_analyse.py`, `s0_B_report_ensemble.py`,
`chain_body.sh`, `run_chain.sh`, `hpc/env/tianhe.sh`, `array.slurm`). It is the state
BEFORE the reorganisation the user asked for on 2026-09-14 ("only the clean result of
CREST, MACE and OpenMM; nothing you write"), kept so the change can be judged against it.

Tags in the trees below:

    [CREST]   written by the crest binary itself
    [MACE]    a MACE result -- there is none as a file: see section 5
    [OpenMM]  an OpenMM result -- there is none as a file: see section 5
    [ours]    written by this repository's code (record, log, summary, copy)
    [slurm]   written by the scheduler
    [parsl]   written by parsl

## 1. The runs root

`config.runs_root()`: `S0_RUNS_ROOT`, else `runtime.runs_root` in `configs/openqha.yaml`,
else `~/runs/openQHA`. On Tianhe `hpc/env/tianhe.sh` sets

    S0_SCRATCH   = $HOME/runs/<jobid>[_<S0_SCRATCH_TAG>]      (per job; per card_row in 02d-2)
    S0_RUNS_ROOT = $S0_SCRATCH/runs

so every path in this section sits under `~/runs/<jobid>/runs/` on the cluster, and the
job's scratch is NOT removed at exit (the base is not a tmp directory) -- it stays there
AND is copied to `logs/node_local/<jobid>/` (section 2).

    <runs_root>/
      branchA/<tag>/<qid>/                    CREST working directory (run_crest)
        <qid>.xyz                             [ours]  copy of the start geometry
        <label>_seed.xyz                      [ours]  SMILES molecules only: the embedded start
        input.toml                            [ours]  the CREST input this code generates
        crest.out                             [CREST stdout, redirected into this file by run()]
        crest_conformers.xyz                  [CREST] the ensemble -- what branch A reads
        crest_best.xyz                        [CREST]
        crest_rotamers.xyz                    [CREST]
        crestopt.xyz                          [CREST] (when present)
        crest.energies  cre_members  ensemble_energies.log         [CREST]
        crest.restart  crest_0.mdrestart  crest_dynamics.trj        [CREST] restart / MTD
        confcross.xyz  crest_input_copy.xyz  coord  wbo             [CREST]
        gfnff_topo  gfnff_adjacency                                  [CREST] topology
      branchA/<tag>/<qid>_shake1/             [same set] only when the SHAKE=2 run terminated early
      qha/<tag>/<qid>/
        summary.json                          [ours]  one line per (basin, seed): frames, complete, wall
        basinNN/seedMM/
          frames.npy                          [ours]  positions pulled from the OpenMM State, float64 A,
                                                      shape (n_frames, N, 3). This IS the trajectory.
          meta.json                           [ours]  seed, segments, settings, force check, engine, timing
          frames.part.npy                     transient; renamed into frames.npy
          gmx/                                [ours + gmx] only with collect --with-gmx (default off)
      level_benchmark/<tag>/<qid>/basinNN/    02c only
        gfn2/   rimp2/optfreq.inp .out .hess  [ours input; xtb / ORCA output]
      parsl/NNN/                              [parsl] run_dir for each parsl driver (branch B, collect):
        parsl.log  submit_scripts/  <executor>/ manager and worker logs
      trace/engrad_trace.log                  [ours]  only when the conf sets MACE_TRACE

    <S0_SCRATCH>/                             (= socket dir on Tianhe)
      s0_mace_pool_<pid>_<i>.sock             [ours]  the MACE server socket
      s0_mace_pool_<pid>_<i>.sock.log         [ours]  MACE server stdout

## 2. logs/ (in the repository)

    logs/
      <jobname>_<jobid>.out  .err             [slurm] job stdout / stderr (run_chain.sh)
      02d2_<row>_<arrayjob>_<task>.log        [ours]  02d-2: one chain log per row
      openqha_pair_<jobid>_<tag>.log          [ours]  02ab only
      node_local/<jobid>[_cardK_rowJ]/        [ours]  cp -a of the WHOLE scratch at job exit:
        MANIFEST.txt                                  ls -laR of the scratch before the copy
        runs/branchA/...  runs/qha/...  runs/parsl/... sockets' .log -- a second copy of section 1

## 3. analysis/ (in the repository)

    analysis/
      branchA/<tag>/<qid>/
        basins.json                           [ours]  the branch A record (gate, crest, census, basins, criteria)
        basins.xyz                            [ours]  relaxed basin geometries, one frame per basin
        driver.log                            [ours]  only via s0_E_branchA_parsl.py
      branchE/<tag>/batch.json                [ours]  only via s0_E_branchA_parsl.py
      qha/<tag>/
        branchB_parsl_summary.json            [ours]  plan + per-task return codes
        <qid>/basinNN_seedM/driver.log        [ours]  stdout+stderr of each trajectory driver
        <qid>.log                             [ours]  collect: the human-readable analysis report
        <qid>__trajectories.parquet           [ours]  collect: per-trajectory TS, frequencies, rates
        <qid>__blank_control.parquet          [ours]  collect
        <qid>__assembly.parquet               [ours]  collect: G - E_el per basin
        <qid>__criteria.parquet               [ours]  collect: the verdicts
        collect/<qid>.log                     [ours]  stdout+stderr of s0_B_qha_analyse.py
        collect/<qid>.json                    [ours]  completion marker (written last, by rename)
        collect/batch.json                    [ours]  plan + results of the collect driver
        <qid>_ensemble.json                   [ours]  report: F_conf over the ensemble
        <qid>_02d_frequency_identity.json     [ours]  02d only
      levels/<tag>/<qid>_02c_level_benchmark.json   [ours]  02c only

## 4. data/basins/ (the store)

    data/basins/<tag>/<range>/<chunk>/
      <qid>.basins.json                       [ours]  byte-identical copy of analysis/branchA/<tag>/<qid>/basins.json
      <qid>.basins.xyz                        [ours]  byte-identical copy of basins.xyz

Branch B reads the basins from HERE (`--basins auto`), not from analysis/.

## 5. Which of these are engine results

CREST is the only engine that writes files of its own. Its result is
`crest_conformers.xyz` (with `crest_best.xyz`, `crest_rotamers.xyz`, `crest.energies`);
everything else in that directory is CREST scratch, plus three files of ours
(`input.toml`, `<qid>.xyz`, the `crest.out` redirect).

MACE writes nothing. Inside CREST its gradients pass through `genericinp.engrad` in a
calcspace that CREST deletes after each call. In branch A steps 3-6 (relax, Hessian) the
MACE numbers exist only inside `basins.json` and as the geometries in `basins.xyz`.

OpenMM writes nothing either: the trajectory driver attaches no Reporter. Positions are
read from the Context state every `sample_every` steps and saved by this code as
`frames.npy`. So `frames.npy` is the OpenMM result in the only form it has; `meta.json`
and everything under analysis/ is ours.

For one molecule, one basin, one seed on the `qha` chain the engine results are FOUR
files (`crest_conformers.xyz`, `crest_best.xyz`, `crest_rotamers.xyz`, `frames.npy`) and
the relaxed geometry `basins.xyz`; this code writes about 25 more, and on Tianhe the copy
to `logs/node_local/` doubles the run tree.

## 6. Since 2026-09-14: the molecule tree (ADR 0001, ADR 0002)

Everything above this line is the state the user objected to. What replaced it:

    <root>/<tag>/<range>/<chunk>/<qid>/
      crest/                      CREST's working directory, verbatim (+ input.toml, <qid>.xyz, crest.out)
      crest_shake1/               the SHAKE fallback attempt, only when it ran
      mace/confNN/                opt.traj  opt.log  conf.extxyz      every tightened conformer
      mace/basinNN/               basin.extxyz  hessian.npy           every surviving basin
      md_openmm/basinNN/             start.pdb system.xml integrator.xml traj.dcd state.csv
                                  state.xml state.chk                  (the default setting)
                                  start_<setting>.pdb ... traj_<setting>.dcd ...
                                  (every other setting, same folder, its name in the file)
      xtb/basinNN/  orca/basinNN/ 02c only
      _records/                   everything this repository writes about the run:
        basins.json  basins.xyz                       branch A's record
        openmm/<setting>/basinNN/{meta.json, driver.log, frames.npy, equilibrated.json}
        openmm/<setting>/{collect.log, collect__*.parquet, collect.json,
                          collect.driver.log, ensemble.json, 02d_frequency_identity.json}
    <root>/<tag>/_records/        records about a whole tag: branchB_parsl_summary.json,
                                  collect_batch.json

    root   = <prefix>/HDD_POOL/<acct>/<user>/sherwin/runs, prefix /XYFS02 for partitions
             ai and cn, /XYAIFS00 for a100x h100x hx a800x v100x (hpc/env/root.sh derives
             it; S0_RUNS_ROOT set explicitly wins; off-cluster ~/runs/openQHA)
    shard  = range 16 000 / chunk 1 000  (openqha/store/layout.py, the only place it is spelled)

Engine files only in the engine folders. CREST runs node-local (`$S0_SCRATCH/openqha_crest/`)
and its finished directory is copied once into `crest/`; nothing else is copied anywhere.
The per-job scratch, the exit-trap copy to `logs/node_local/`, the basin store
`data/basins/` and `analysis/branchA`, `analysis/qha` are gone; a script lists and, on
request, deletes the old trees (`scripts/tooling/s0_delete_old_layout.py --plan`).

Readers: `openqha.store.basins` (the basins, from `mace/`), `openqha.quasi_harmonic.
trajectory_reader` (the trajectory, from `openmm/`); every driver takes `--basin-tag`
(the tag the molecule directory is under) and `--setting`.

Not covered by the change: the ASE-route trajectory driver (`s0_B_qha_trajectory.py`,
CPU cross-check) still writes `frames.npy` + `meta.json` under the old runs root and the
collect step no longer finds those; the Slurm `.out/.err` and parsl run directories stay
under the repository's `logs/` until step 2 decides the records.

## 7. Step 2 (2026-09-15): the records, CREST/ORCA style

The five records a program needs later, and everything beside them, took the form CREST
and ORCA use: a text report per step whose last line says the step terminated normally
(the completion marker), TOML for what a program reads back (CREST's own settings
format; Python 3.11 reads it with the standard library), and whitespace tables for
collect's numbers. Nothing is JSON or parquet any more, and nothing is written twice.

    <molecule>/_records/
      branchA.out                       branch A's report; last line = the marker
      basins.toml                       branch A's record (basins, criteria, settings, CREST, census)
      md_<route>/<setting>/basinNN/md.out      the trajectory's report
      md_<route>/<setting>/basinNN/md.toml     the trajectory's record (seed, identity inputs, resume)
      md_<route>/<setting>/collect.trajectories.dat  .criteria.dat  .assembly.dat  .blank.dat
      md_<route>/<setting>/collect.out          collect's report, written LAST; last line = the marker
      md_<route>/<setting>/ensemble.out + ensemble.toml              F_conf over the basins
      md_<route>/<setting>/02d_frequency_identity.out + .toml        02d only
    <root>/<tag>/_records/
      branchB_parsl_summary.json  collect_batch.json   (the parsl drivers' batch summaries; still JSON)
      parsl/<job>.<pid>/                                parsl's own logs

Gone: basins.json, basins.xyz, meta.json, frames.npy, summary.json, progress.json,
equilibrated.json, collect.log, collect.json, collect.driver.log, the four parquet
tables, ensemble.json, 02d_frequency_identity.json. The parquet-engine preflight and the
pandas/pyarrow requirement of the chain went with them.

Readers: `openqha.store.toml_out` (TOML in and out), `openqha.store.dat` (tables),
`openqha.store.report` (`.out` and `terminated_normally`), `openqha.quasi_harmonic.md_record`.
(Superseded the same afternoon by section 8.)

## 8. Records redesign (2026-09-15, afternoon): one Record per Calculation, ORCA property style

Two words (CONTEXT.md): a **Calculation** is one step on one molecule or basin and owns a
**Record**, a **Report** (`.out`) plus a **Property file** (`.toml`); a **Batch** is one
driver over many Calculations and owns nothing, its Slurm log is the report. The
`<setting>` level under `_records/` is gone: the setting is in the file stem, as for
engine files. The Property file takes ORCA's `.property.txt` shape carried into TOML:

    [Calculation_Status]
    PROGNAME = "openQHA branchA"     # String: the step that wrote this file
    VERSION  = "0.3.0"               # String: openQHA version
    STATUS   = "NORMAL TERMINATION"  # String: the completion marker
    [Calculation_Info]   the inputs (upper-case keys, a "# Type, unit: doc" comment each)
    ...                  only the result blocks a later step reads; [[Basin]] etc. with INDEX

STATUS is the completion marker for programs (RUNNING while a trajectory driver may still
be resumed; a file without it never got that far); the Report's last line stays the
marker for people. Provenance, diagnostics, criterion detail lines, the sigma tolerance
sweep, the thermochemistry breakdown and every file list are printed in the Report only.

    <molecule>/_records/
      branchA.out   branchA.toml                     [Calculation_Info] [CREST_Run] [Census] [[Basin]] [Criteria]
      driver.log                                     the driver's stdout when a Batch ran it
      md_openmm/basinNN/md.out  md.toml  driver.log  [Calculation_Info] [Masses] [Force_Check] [Equilibration]
                                                     [Production] [[Segment]]; md_s2.* for setting s2
      md_openmm/collect.out  collect.toml            [Calculation_Info] [Criteria]; collect_s2.* for setting s2
      md_openmm/collect.dat                          the Table: [trajectories] [blank] [assembly], every
                                                     column explained above its header (section 9)
      md_openmm/ensemble.out  ensemble.toml          [Calculation_Info] [[Result]] [[Basin]]
      md_openmm/02d_frequency_identity.out  .toml    02d only: [[Basin]] [[Hybrid]]
      md_ase/...                                     the same for the ASE route
    <root>/<tag>/_records/parsl/<job>.<pid>/         parsl's own logs; nothing else per tag

The Batch table every parsl driver prints (branch A, branch B, collect), one aligned line
per Calculation, the common columns first and the driver's own after:

    species  basin  seed  rc  seconds  STATUS  record  ...
    batch wall 5.9 s   3/3 ran to a record
    slurm job 7351234

`rc` is the return code, STATUS is read from the Property file after the run (FAILED
when absent with rc != 0, NO RECORD when absent with rc 0), `record` the absolute path.

Gone: `basins.toml` (now `branchA.toml`), the `_records/md_<route>/<setting>/` level,
`branchB_parsl_summary.json`, `collect_batch.json`, `analysis/branchE/<tag>/batch.json`,
`analysis/branchA/<tag>/<qid>/driver.log`. `scripts/tooling/s0_delete_old_layout.py --plan`
lists these leftovers under an existing root.

Writers and readers: `openqha.store.property` (the shape), `openqha.store.branch_a_property`,
`openqha.quasi_harmonic.md_record`, `openqha.quasi_harmonic.chain_records`,
`openqha.store.batch_table`. Sample Records: `examples/02b_qha_openmm_propanal/sample_records/`
(both routes) and `examples/02d_qha_frequency_identity/sample_records/`.

## 9. One Table per Calculation (2026-09-16): `collect.dat`

Collect's four `.dat` became one Table, `collect.dat` (`collect_s2.dat` for a setting), in
the form CREST's `crest.energies` takes with two additions: a `[section]` line before each
of its three tables, and one comment line per column above each header in the Property
file's form, `Type, unit: doc`, generated from a column schema so an undocumented column is
a test failure. Column names are unchanged.

    [trajectories]                one row per trajectory: n_frames, TS_QH_kcal, TS_Schlitter_kcal,
                                  S_QH_kcal_per_K, A_vib_kcal, lowest/highest_frequency_cm_inv,
                                  n_nonzero, expected_modes, rigid_ratio,
                                  saturation_last_doubling_kcal, wall_seconds, seconds_per_ps
    [blank]                       one row per basin: n_seeds, TS_QH_mean/spread/rms_about_mean_kcal,
                                  standard_error_kcal, note
    [assembly]                    one row per basin: G_minus_Eel_kcal, A_vib_kcal, S_vib_kcal_per_K,
                                  terms_match_hessian_route

The criteria are not a table: each verdict is a sentence with its measure in `collect.out`
(once), and the counts a later step reads are `[Criteria]` in `collect.toml`; the ensemble
reads them there. The expanded dump at the end of `collect.out` no longer repeats the
rows the Table and the `Criteria` section hold. Glossary: CONTEXT.md **Table**; decision:
ADR 0003, amendment of 2026-09-16. Writer and reader: `openqha.store.dat`
(`write_tables`, `read_tables`), `openqha.quasi_harmonic.chain_records` (`COLUMNS`,
`write_collect_table`, `read_collect_table`).
