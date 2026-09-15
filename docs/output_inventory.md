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
