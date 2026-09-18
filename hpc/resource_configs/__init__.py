"""Machine-specific Parsl resource configurations.

One module per machine, each exposing `config(...)` and `describe()`. Nothing here
knows anything about chemistry: choosing a machine must never change a number.
That separation is the whole point of the directory, and it is what makes branch E
acceptance criterion 1 -- same input, same result, local or cluster -- a thing that
can be checked.

Adapted in shape from ALF (LANL, BSD-3-Clause), `alframework/parsl_resource_configs`.
"""
from . import local          # noqa: F401

#: Every machine this repository knows how to submit to, and what belongs on each.
#: Kept here rather than inside an error string so it cannot drift out of date -- which
#: it had: the message still named only `local` and `deimos` after three Tianhe configs
#: were added.
SITES = {
    "local": "this workstation. Step 0: prove the chain here before any cluster.",
    "deimos": "the group's own Slurm cluster.",
    "tianhe_cpu": "TianheXY-C. Branch A (crest), branch B collection (collect), "
                  "Hessian-learning labels (labels).",
    "tianhe_a": "TianheXY-A. Branch B trajectories (qha) and branch C training "
                "(train), one card per task, 8 per allocation. Preferred GPU site.",
    "tianhe_ai": "TianheXY-AI. Branch C training, one card per allocation.",
}

__all__ = ["local", "SITES", "load"]


def load(name):
    """Import a resource configuration by name."""
    import importlib
    try:
        return importlib.import_module("{}.{}".format(__name__, name))
    except ImportError as exc:
        raise ImportError(
            "no resource configuration named {!r}. Available:\n{}\n"
            "Add a new machine by copying local.py -- see hpc/README.md.\n"
            "({})".format(
                name,
                "\n".join("  {:12s} {}".format(k, v) for k, v in sorted(SITES.items())),
                exc))
