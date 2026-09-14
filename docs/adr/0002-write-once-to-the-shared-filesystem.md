---
status: accepted
date: 2026-09-14
---

# The molecule tree is written once, to the shared filesystem

On Tianhe every job wrote its tree under a per-job scratch (`~/runs/<jobid>/`), then at
exit copied the whole scratch into `logs/node_local/<jobid>/` and, because that scratch
was not a tmp directory, left the original in place as well: two copies of every
trajectory. The scratch existed for the MACE server's Unix sockets, which cannot live on
Lustre. The user ruled: the molecule tree is written straight to the shared filesystem
and stays there; the exit copy is removed; sockets go to `/tmp/$USER/$SLURM_JOB_ID`, a
node-local directory that is gone with the node.

## Consequences

A job killed at its walltime leaves whatever the engines had flushed; nothing depends on
an exit trap running. The logs that used to be worth carrying back (parsl run
directories, socket logs) need a home of their own, decided with the other records.
