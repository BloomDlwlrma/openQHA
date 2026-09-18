# 03: Reference E-F-H labels per frame (`openqha/data/frame_labels.py`, `03_labels.py`, tianhe Batch)

**What to build:** `frame_labels.run(molecule, level="wb97m-d3bj_def2-tzvppd", generators=None, nprocs=4, maxcore=None)`: per frame an ORCA single point + analytic Hessian (`orca.LEVELS[level]["single_point"] + " Freq"`, blocks per level) under `<molecule>/orca/<level>/frames/<generator>_b<basin>_k<k>/job.*` (full `.out` kept, finished skipped by the terminal line); writes `<molecule>/frames/<generator>.<level>.extxyz` with positions identical to the MACE file (refuses at > 1e-8 A), `energy`, `forces`, `hessian` in eV, eV/A, eV/A^2 from the `.hess`; Record `frames/labels.<level>.{out,toml}` (frames done, wall time per frame, ORCA version, `hessian_route`, noise floor per frame). Batch: `workflows/hessian_learning/03_labels.py --tag --limit N --stratify` writes the frame list and `hpc/slurm/frame_labels_tianhe.slurm` (one node per job, `xargs -P 16`, `%pal nprocs 4`, `%maxcore` = node memory / 64 x 0.75, `$TMPDIR` scratch, sbatch on TianheXY-CN); `--local` runs the same list here. Fact to verify first: the ORCA binary and version on TianheXY-CN.

**Blocked by:** 02.

**Delivers:** the smoke set (7 molecules, ~80 frames) labelled on 1 node; the 200-molecule draw on 12 nodes.

- [ ] a fake `.hess` at the fixture's basin frame produces the extxyz with the fixture's known frequencies after projection (round trip < 0.5 cm^-1); a 1e-6 A position mismatch is refused
- [ ] the Slurm script and frame list for the 7 molecules: 16 concurrent, 4 ranks, maxcore from the node, skip-finished; dry run locally on one frame keeps the full `.out`
- [ ] OPEN until the tianhe smoke Batch returns: wall time and memory per frame in the Record; every frame of the 7 molecules labelled
