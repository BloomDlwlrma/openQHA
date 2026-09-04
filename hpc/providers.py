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
#: **Only two of these five are confirmed.** `yhbatch` and `yhrun` appear verbatim
#: in the site's own job scripts (user, 2026-08-31; D0-C-24). The status and cancel
#: commands have NOT been seen on that machine, and this repo does not get to invent
#: them: a wrong status command does not fail loudly, it makes Parsl believe every
#: job is still pending, and the queue silently stops making progress.
#:
#: So they are left at the Slurm names, which is what a Slurm derivative usually
#: keeps, and `preflight()` below reports which of the candidates actually exist on
#: PATH. **Run preflight on the login node before step 1** and fill in whatever it
#: finds. Branch E acceptance criterion 3 covers the rendered script; this covers
#: the commands that act on it.
TIANHE_COMMANDS = dict(
    submit="yhbatch",            # confirmed: site job scripts
    launcher="yhrun",            # confirmed: site job scripts
    status="sacct",              # UNVERIFIED -- candidates: sacct, yhacct
    status_fallback="squeue",    # UNVERIFIED -- candidates: squeue, yhqueue, yhinfo
    cancel="scancel",            # UNVERIFIED -- candidates: scancel, yhcancel
)

#: Which of the above are measured rather than assumed. Carried into every record.
TIANHE_CONFIRMED = ("submit", "launcher")

COMMANDS = {"slurm": SLURM_COMMANDS, "tianhe": TIANHE_COMMANDS}

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
            super().__init__(*args, **kwargs)
            src, dst = COMMANDS["slurm"], COMMANDS[self.site]
            self._command_map = {
                src["submit"]: dst["submit"],
                src["status"]: dst["status"],
                src["status_fallback"]: dst["status_fallback"],
                src["cancel"]: dst["cancel"],
            }
            logger.info("TianheSlurmProvider: command map %s", self._command_map)

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
            return super().execute_wait(swapped, *args, **kwargs)

        def render_only(self, path, job_name="openqha_render_check"):
            """Write the job script Parsl WOULD submit, and submit nothing.

            This is what branch E acceptance criterion 3 inspects.
            """
            from parsl.utils import RepresentationMixin  # noqa: F401  (import check)
            script = self._write_submit_script(
                self.template_string, str(path), job_name,
                self._get_job_config() if hasattr(self, "_get_job_config") else {})
            return script


else:  # pragma: no cover

    class TianheSlurmProvider(object):
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "parsl is not installed, so TianheSlurmProvider cannot be built.\n"
                "    pip install parsl\n"
                "Branch E step 0 is to get the local chain running first; the "
                "cluster providers are step 1.")


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
