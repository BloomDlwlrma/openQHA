# Sentinel files and scaling cadence: two operator questions from the labels sweep

Date: 2026-09-27. Serves: the labels-driver operator-lever design (Q1 «STOP file», Q2
«launch discipline») raised while the Tianhe sweep ran; the driver-behaviour facts are from
the live run's own `parsl.log` (run_dir `labels_pid1740052`), quoted in the campaign
session. Method: primary sources only, each claim followed to the source that owns it;
every fetch is listed in the verification log at the end.

## 1. Is a «STOP file» (a sentinel the process polls) an established pattern?

**Finding: no standard names it — but «file existence is the control signal» is first-party
practice, not an ad-hoc invention.**

- **systemd** records a scheduled shutdown by writing a sentinel file; other components
  decide against it. Primary source, `systemd/systemd` `src/login/logind-shutdown.h`
  (verified by code search 2026-09-27):

      #define SHUTDOWN_SCHEDULE_FILE "/run/systemd/shutdown/scheduled"

- **Linux-PAM** ships an auth/account decision made purely by a file's existence.
  `pam_nologin(8)` (man7.org, fetched 2026-09-27): «pam_nologin is a PAM module that
  prevents users from logging into the system when `/var/run/nologin` or `/etc/nologin`
  **exists**. The contents of the file are displayed to the user.»
- Family members in the same shape, named for familiarity only (not individually fetched
  here): `/etc/nologin` itself (`nologin(8)`), lock files taken to serialize (e.g. Git's
  `*.lock`), container markers such as `/.dockerenv`.

**Fit for this driver.** The operator's constraint is that no signal channel exists: no ssh
into the node that hosts the tmux-held driver (random login assignment, no node-to-node
ssh), the scheduler cannot signal a login-node process, and `scancel` is not a stop (parsl
replaces dead blocks while tasks remain). The shared filesystem is the only medium both
sides can write — and the campaign already coordinates through it (frame locks `.running`,
failure archives `<stem>.failed.out`). A STOP file continues that idiom. Ranked
alternatives: a signal (unavailable), a control socket/RPC (a new surface parsl does not
provide), scheduler-native stop (the driver is outside Slurm), admin intervention (a human
dependency).

**Boundaries to write down with the lever.** The driver checks between task harvests
(~30 s cadence); on sight it runs parsl's cleanup — the blocks *in its ledger* are
cancelled, a block outside the ledger (the 2026-09-27 «ghost») needs a manual `scancel`;
a stale STOP file should refuse a fresh start until removed; the tmux session dies with
the process by default (tmux `remain-on-exit` off).

## 2. Is a 5-second scaling tick normal?

**Finding: 5 s is parsl's own default, and periodic second-to-teens-cadence re-evaluation
is the norm for elastic scaling loops.**

- **parsl**, `parsl/config.py` (fetched from `Parsl/parsl` master 2026-09-27):

      strategy_period: Union[float, int] = 5,
      # docstring: "How often the scaling strategy should be executed. Default is 5 seconds."
      max_idletime: float = 120.0,
      # docstring: "The maximum idle time allowed for an executor before strategy
      #             could shut down unused blocks. Default is 120.0 seconds."

  The Tianhe install is parsl 2026.09.07; the driver's own Config dump reports the same
  values (`strategy='simple'`, `strategy_period=5`, `max_idletime=120.0` in
  `labels_pid1740052/000/parsl.log`), i.e. nothing was tuned.
- **Kubernetes HPA** — the reference implementation of elastic scaling, for comparison
  (kubernetes.io HPA docs, fetched 2026-09-27): «Kubernetes implements horizontal pod
  autoscaling as a control loop that runs intermittently (it is not a continuous process).
  The interval is set by the `--horizontal-pod-autoscaler-sync-period` parameter to the
  kube-controller-manager (and the default interval is 15 seconds).»
- **What the parsl tick costs and buys** (measured on this run): each tick logs one
  evaluation and issues one provider status query (sacct for the known job ids) — a
  millisecond-scale cost at a 5 s period (~17 k/day). In exchange the same loop is the
  sweep's entire self-management: it replaces blocks that die (`case 2b` after a tracked
  block's 72 h walltime), scales in when `active_tasks` reaches 0 (`case 1b` after
  `max_idletime`), and otherwise only states «not scaling out» at the cap. Message strings
  verified against `parsl/jobs/strategy.py` (fetched) and the live log — e.g.
  `Strategy case 2a: active_blocks 12 >= max_blocks 12 so not scaling out`, every 5 s
  while the ledger sits at its cap.

## Verification log (2026-09-27)

- Fetched: `raw.githubusercontent.com/Parsl/parsl/master/parsl/config.py`;
  `raw.githubusercontent.com/Parsl/parsl/master/parsl/jobs/strategy.py`;
  `kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/`;
  `man7.org/linux/man-pages/man8/pam_nologin.8.html`;
  `man7.org/linux/man-pages/man8/nologin.8.html`.
- Code search: `systemd/systemd` for `shutdown/scheduled` → `src/login/logind-shutdown.h`.
- Live evidence: `$S0_RUNS_ROOT/parsl/labels_pid1740052/000/parsl.log` (config dump;
  strategy ticks every 5 s; the ghost-block submit error of 2026-09-27), quoted in the
  campaign session.
- Deliberately not claimed: `/run/reboot-required` (would need the update-notifier source;
  search returned nothing — omitted), and the familiarity-only family members in §1.
