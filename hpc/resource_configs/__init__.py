"""Machine-specific Parsl resource configurations.

One module per machine, each exposing `config(...)` and `describe()`. Nothing here
knows anything about chemistry: choosing a machine must never change a number.
That separation is the whole point of the directory, and it is what makes branch E
acceptance criterion 1 -- same input, same result, local or cluster -- a thing that
can be checked.

Adapted in shape from ALF (LANL, BSD-3-Clause), `alframework/parsl_resource_configs`.
"""
from . import local          # noqa: F401

__all__ = ["local", "deimos", "load"]


def load(name):
    """Import a resource configuration by name."""
    import importlib
    try:
        return importlib.import_module("{}.{}".format(__name__, name))
    except ImportError as exc:
        raise ImportError(
            "no resource configuration named {!r}. Available: local, deimos.\n"
            "Add a new machine by copying local.py -- see hpc/README.md.\n"
            "({})".format(name, exc))
