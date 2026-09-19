"""Schedulers whose command names Parsl does not already know.

Branch E, plan_E section 3.1. This is the ONLY non-configuration code in hpc/ --
everything else in this directory is a resource description.

The problem, verified on the installed parsl (2026.08.10) with
`inspect.getsource(SlurmProvider)` rather than from memory:

    docstring   "This provider uses sbatch to submit, sacct for status and
                 scancel to cancel jobs."
    submit      self.execute_wait("sbatch {0}".format(script_path))
    status      tries `sacct`, falls back to `squeue`
    cancel      _cmd = "scancel"

So **four** command names are hard-coded, not three. Tianhe uses `yhbatch` /
`yhrun` while keeping `#SBATCH` directives -- it is a Slurm derivative with
renamed front-end commands -- so `TianheSlurmProvider` overrides exactly those
names and inherits everything else.

What this deliberately does NOT do
----------------------------------
It does not put a shim named `sbatch` on PATH, and it does not alias anything.
That would work, and it would make "how was this job submitted" a fact you cannot
read off the record. This repo has already lost a dataset to a condition that was
invisible in its own products (defect 57), so the substitution is made in a named
class that appears in the run record instead.
"""
import logging
import pathlib

logger = logging.getLogger(__name__)

try:
    from parsl.providers import SlurmProvider
except ImportError:  # pragma: no cover - the message is the useful output
    SlurmProvider = None


#: The four names Parsl hard-codes, plus the launcher.
SLURM_COMMANDS = dict(submit="sbatch", status="sacct", status_fallback="squeue",
                      cancel="scancel", launcher="srun")

#: Tianhe (TianheXY-AI). Directives stay `#SBATCH`; only the front-end commands are
#: renamed.
#:
#: **All five are confirmed as of 2026-09-05** (`type -a` on the login node). They were
#: two of five until then, and the reason that mattered is worth keeping: a wrong submit
#: command fails loudly, while a wrong status command does not -- it makes Parsl believe
#: every job is still pending and the queue silently stops making progress. That is why
#: they were never guessed, and why `execute_wait` now raises on a failed status query.
#:
#: They are the Slurm names, which is what a Slurm derivative usually
#: keeps, and `preflight()` below reports which of the candidates actually exist on
#: PATH. **Run preflight on the login node before step 1** and fill in whatever it
#: finds. Branch E acceptance criterion 3 covers the rendered script; this covers
#: the commands that act on it.
TIANHE_COMMANDS = dict(
    submit="yhbatch",            # measured 2026-09-05: /usr/bin/yhbatch (no `sbatch`)
    launcher="yhrun",            # measured 2026-09-05: /usr/bin/yhrun
    status="sacct",              # measured 2026-09-05: /usr/bin/sacct
    status_fallback="squeue",    # measured 2026-09-05: /usr/bin/squeue
    cancel="scancel",            # measured 2026-09-05: /usr/bin/scancel
)

#: Which of the above are measured rather than assumed. Carried into every record.
#:
#: ALL FIVE, as of 2026-09-05: `type -a` on the login node found every one of them, and
#: also found the yh* variants (yhacct, yhqueue, yhcancel). Both families are present.
#:
#: The plain Slurm names are kept for status and cancel on purpose. parsl's SlurmProvider
#: PARSES the output of these commands, and the yh* variants are not guaranteed to print
#: byte-identical output; swapping to a name whose format nobody has read would trade a
#: verified path for an unverified one. `submit` must stay `yhbatch` -- `sbatch` was not
#: in the list at all.
TIANHE_CONFIRMED = ("submit", "launcher", "status", "status_fallback", "cancel")

#: TianheXY-CN, the CPU cluster (`deimos`, `debug`): STOCK SLURM. `sbatch`, not `yhbatch`
#: -- measured by the user 2026-09-09 (S0-G-74): the yh* wrappers are the GPU clusters',
#: and the CPU login node submits with the plain names. Until 2026-09-18 tianhe_cpu.py
#: built `TianheSlurmProvider`, whose map would have sent every block through `yhbatch`;
#: it had never submitted (the CPU chain went through examples/run_chain.sh, which reads
#: the partition table), and the labels Batch of the Hessian-learning set is the first
#: driver to submit its own blocks there.
TIANHE_CN_COMMANDS = dict(SLURM_COMMANDS)

COMMANDS = {"slurm": SLURM_COMMANDS, "tianhe": TIANHE_COMMANDS, "tianhe_cn": TIANHE_CN_COMMANDS}

#: What `preflight` looks for, in preference order.
CANDIDATES = dict(
    submit=("yhbatch", "sbatch"),
    launcher=("yhrun", "srun"),
    status=("yhacct", "sacct"),
    status_fallback=("yhqueue", "squeue", "yhinfo"),
    cancel=("yhcancel", "scancel"),
)


def preflight(site="tianhe"):
    """Report which scheduler commands actually exist on this machine.

    Run this on the login node BEFORE the first submission. It writes nothing and
    submits nothing; it answers the one question the repo cannot answer from here.
    """
    import shutil
    declared = COMMANDS[site]
    confirmed = TIANHE_CONFIRMED if site == "tianhe" else tuple(declared)
    out = {}
    for role, names in CANDIDATES.items():
        found = [n for n in names if shutil.which(n)]
        out[role] = dict(
            configured=declared.get(role),
            configured_exists=bool(declared.get(role)
                                   and shutil.which(declared[role])),
            found_on_path=found,
            evidence=("measured on this site's job scripts" if role in confirmed
                      else "ASSUMED -- not yet seen on this machine"),
        )
    out["_verdict"] = (
        "usable" if all(v["configured_exists"] for k, v in out.items()
                        if not k.startswith("_"))
        else "at least one configured command is not on PATH -- edit "
             "hpc/providers.py TIANHE_COMMANDS before submitting anything")
    return out


def normalise_walltime(walltime):
    """Slurm's `D-HH:MM:SS` (also `D-HH:MM`, `D-HH`, `HH:MM`) as the `HH:MM:SS` parsl's
    `wtime_to_minutes` reads -- it does `hours, mins, seconds = s.split(':')` and
    `int(hours)`, so the day form the site's own scripts use failed the first labels
    Batch on tianhe with "invalid literal for int() with base 10: '3-00'" (2026-09-19).
    Parsl renders `--time` in minutes, so nothing else changes."""
    s = str(walltime).strip()
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    parts = s.split(":")
    if len(parts) == 1:
        h, m, sec = int(parts[0]), 0, 0
    elif len(parts) == 2:
        h, m, sec = int(parts[0]), int(parts[1]), 0
    else:
        h, m, sec = int(parts[0]), int(parts[1]), int(parts[2])
    return "{:02d}:{:02d}:{:02d}".format(days * 24 + h, m, sec)


def _swap(text, mapping):
    """Replace whole-word command names in a command string."""
    import re
    out = text
    for old, new in mapping.items():
        if old == new:
            continue
        out = re.sub(r"(?<![\w/-])" + re.escape(old) + r"(?![\w-])", new, out)
    return out


if SlurmProvider is not None:

    class TianheSlurmProvider(SlurmProvider):
        """SlurmProvider with Tianhe's front-end command names.

        Everything else -- the `#SBATCH` directives, the block model, `max_blocks`,
        the walltime handling -- is inherited unchanged, because everything else is
        the same.

        Two notes that belong with the class, not in a README:

        1. **Pass the GPU count as `gpus_per_node`, not as a `scheduler_options`
           string.** Tianhe refuses a submission with no explicit `-G` (D0-C-25),
           and a parameter gets rendered into the template by Parsl itself while a
           string is never checked. `SlurmProvider.__init__` already accepts
           `gpus_per_node` and `gres`; verified on the installed version.

        2. **Look at the rendered script before the first real submission.** Branch
           E acceptance criterion 3 is written against the rendered script, not
           against the config -- a config that reads correctly and renders wrongly
           is exactly the failure this class exists to avoid. `render_only()` below
           writes it out without submitting anything.
        """

        site = "tianhe"

        def __init__(self, *args, **kwargs):
            # BEFORE super().__init__(), not after. parsl's SlurmProvider.__init__
            # probes the scheduler before it returns (slurm.py:216), and that probe
            # goes through the overridden execute_wait below -- which reads
            # self._command_map. Setting the map afterwards made every construction
            # raise AttributeError, on every machine. The class had never been
            # constructed here (branch E only ever ran the `local` config), which is
            # what let it survive: written is not the same as executed.
            if "walltime" in kwargs:
                kwargs["walltime"] = normalise_walltime(kwargs["walltime"])
            src, dst = COMMANDS["slurm"], COMMANDS[self.site]
            self._command_map = {
                src["submit"]: dst["submit"],
                src["status"]: dst["status"],
                src["status_fallback"]: dst["status_fallback"],
                src["cancel"]: dst["cancel"],
            }
            logger.info("TianheSlurmProvider: command map %s", self._command_map)
            # The loud-status check below is armed only after construction. parsl's
            # SlurmProvider.__init__ runs `sacct -X` once to discover what the scheduler
            # supports, and "not found" is a legitimate answer to that on a machine with
            # no scheduler. Raising during the probe would make this class impossible to
            # construct off-cluster and would block the render check (branch E acceptance
            # criterion 3), which is supposed to run BEFORE anyone has a cluster.
            self._status_check_armed = False
            super().__init__(*args, **kwargs)
            self._status_check_armed = True

        def execute_wait(self, cmd, *args, **kwargs):
            """Every scheduler call in SlurmProvider goes through here.

            Overriding this one method covers submit, status and cancel at once,
            including the `sacct` -> `squeue` fallback, without having to copy any
            of Parsl's logic -- so a change upstream cannot leave one of the four
            names behind.
            """
            swapped = _swap(cmd, self._command_map)
            if swapped != cmd:
                logger.debug("TianheSlurmProvider: %r -> %r", cmd, swapped)
            retcode, stdout, stderr = super().execute_wait(swapped, *args, **kwargs)

            # A wrong SUBMIT fails loudly by itself: no job id comes back and Parsl
            # raises. A wrong STATUS does not -- the command is simply not found, the
            # output is empty, no job matches, and every job stays PENDING for ever while
            # the queue quietly stops. Three of the five names here are still UNVERIFIED
            # (see TIANHE_CONFIRMED), so the failure that cannot be seen is the one worth
            # converting into one that can.
            if (retcode != 0 and getattr(self, "_status_check_armed", False)
                    and self._is_status_call(swapped)):
                raise RuntimeError(
                    "Tianhe scheduler STATUS command failed (exit {}): {!r}\n"
                    "  stdout: {!r}\n  stderr: {!r}\n"
                    "This is raised rather than ignored on purpose. Parsl treats an "
                    "unreadable status as 'still pending', so an unrecognised command "
                    "name would stall every job with no error anywhere.\n"
                    "  Check the name:  python -c \"import sys; sys.path.insert(0,'hpc');"
                    " import providers, json; print(json.dumps("
                    "providers.preflight('tianhe'), indent=1))\"\n"
                    "  Then edit TIANHE_COMMANDS in hpc/providers.py.".format(
                        retcode, swapped, (stdout or b"")[:200], (stderr or b"")[:200]))
            return retcode, stdout, stderr

        def _is_status_call(self, cmd):
            """Is this command string one of the STATUS queries?

            Matched against the SITE names, because `execute_wait` has already swapped
            them by this point.
            """
            dst = COMMANDS[self.site]
            names = [dst.get("status"), dst.get("status_fallback")]
            first = (cmd or "").strip().split(None, 1)[0] if (cmd or "").strip() else ""
            return any(n and first == n for n in names)

        def render_only(self, path, job_name="openqha_render_check", tasks_per_node=1,
                        nodes_per_block=None):
            """Write the job script Parsl WOULD submit, and submit nothing.

            This is what branch E acceptance criterion 3 inspects, and it must work
            WITHOUT a scheduler -- the point is to read the script before there is a
            cluster to submit it to.

            `template_string` is a MODULE-level name in parsl
            (`parsl.providers.slurm.slurm.template_string`), not an attribute of the
            provider. Reading it as `self.template_string` made this method raise
            AttributeError, so the check that was meant to precede every first submission
            had never once run.

            The job_config is assembled the way `SlurmProvider.submit` assembles it, so
            what lands on disk is the script that would really be submitted rather than
            something that merely resembles it.
            """
            from parsl.providers.slurm.slurm import template_string

            nodes = self.nodes_per_block if nodes_per_block is None else nodes_per_block
            job_config = {
                "submit_script_dir": self.script_dir,
                "nodes": nodes,
                "tasks_per_node": tasks_per_node,
                "walltime": self.walltime,
                "scheduler_options": self.scheduler_options,
                "worker_init": self.worker_init,
                "partition": self.partition,
                "account": self.account,
                "qos": getattr(self, "qos", None),
                "constraint": getattr(self, "constraint", None),
                # parsl validates these two even though the template body only
                # substitutes ; without them _write_submit_script raises
                # SchedulerMissingArgs.
                "job_stdout_path": str(pathlib.Path(str(path)).with_suffix(".out")),
                "job_stderr_path": str(pathlib.Path(str(path)).with_suffix(".err")),
                "user_script": "echo 'render only -- nothing is executed'",
            }
            job_config["user_script"] = self.launcher(
                job_config["user_script"], tasks_per_node, nodes)
            self._write_submit_script(template_string, str(path), job_name, job_config)
            return pathlib.Path(str(path)).read_text(encoding="utf-8")

    class TianheCNSlurmProvider(TianheSlurmProvider):
        """The CPU cluster's provider: the same class, the stock Slurm command names
        (`TIANHE_CN_COMMANDS`, S0-G-74). The map is then the identity and `_swap` leaves
        every command alone; the loud status check and `render_only` are inherited."""

        site = "tianhe_cn"


else:  # pragma: no cover

    class TianheSlurmProvider(object):
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "parsl is not installed, so TianheSlurmProvider cannot be built.\n"
                "    pip install parsl\n"
                "Branch E step 0 is to get the local chain running first; the "
                "cluster providers are step 1.")

    class TianheCNSlurmProvider(TianheSlurmProvider):
        pass


def describe():
    """What this module claims about each site. Goes into the run record."""
    return dict(
        sites={k: dict(v) for k, v in COMMANDS.items()},
        tianhe_confirmed_commands=list(TIANHE_CONFIRMED),
        tianhe_assumed_commands=[k for k in TIANHE_COMMANDS
                                 if k not in TIANHE_CONFIRMED],
        parsl_available=SlurmProvider is not None,
        note=("Command names verified against inspect.getsource(SlurmProvider) on "
              "parsl 2026.08.10, not quoted from documentation. Four names are "
              "hard-coded upstream: sbatch, sacct, squeue, scancel. Of the Tianhe "
              "replacements only yhbatch and yhrun are measured; run preflight() "
              "on the login node to settle the rest."),
        array_jobs=("Tianhe supports --array (user, 2026-09-03), but Parsl does not "
                    "use it: it queues through max_blocks. Recorded because it "
                    "keeps _superseded/cluster_2/s0_submit_array.slurm open as a "
                    "fallback shape if Parsl ever proves unworkable there."),
    )
