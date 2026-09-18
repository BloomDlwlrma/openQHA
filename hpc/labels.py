"""Executor labels: the single place they are defined, and the check that they match.

WHY THIS FILE EXISTS
--------------------
A Parsl app names the executor it wants; a Parsl config provides executors with labels.
If the two disagree, **the config builds cleanly, the job script renders cleanly, and at
run time the app matches no executor.** That happened here on 2026-09-05: a new config
labelled its executor `openqha_crest_tianhe` while the driver bound
`executors=["openqha_crest"]`. Nothing complained until it would have mattered.

A string repeated in two files is a defect waiting for someone to edit one of them. So
the labels live here, and `check(config)` refuses a Config whose labels are not in the
registry.

THE NAMING PATTERN
------------------
Taken from ALF's, which is the convention this project's HPC work already follows:

    <project>_<role>_executor              the normal pool for that role
    <project>_<role>_standby_executor      optional: lower priority, shorter walltime

The role is what the task IS, never where it runs. `openqha_crest_executor` runs the
conformer search whether that is a laptop, the CPU cluster or a login node -- which is
the whole point of the execution layer: replacing it with a for-loop must not change a
single number.

This is not a technicality. On 2026-09-07 branch B trajectories moved from the CPU
cluster to the GPU one and `openqha_qha_executor` did not have to change, because it
never named a machine. The `cluster` column below is a note about where the role is
USUALLY scheduled, not part of the contract.

**A label never names a machine.** `openqha_crest_tianhe` was wrong twice over: it named
the site, and it did not match.
"""

#: role -> (label, what runs there, which cluster it belongs on)
ROLES = {
    "crest": (
        "openqha_crest_executor",
        "Branch A: CREST iMTD-GC conformer search with the GFN2-xTB workhorse, plus "
        "MACE refinement over a socket. CPU-bound; xtb has no GPU path.",
        "CPU cluster (TianheXY-C)",
    ),
    "qha": (
        "openqha_qha_executor",
        "Branch B: unbiased quasi-harmonic trajectories, through OpenMM with "
        "openqha/openmm_mace.py on the CUDA platform. ONE TRAJECTORY PER CARD -- "
        "parallelism belongs between trajectories, never inside one.",
        "GPU cluster (TianheXY-A), 8 cards per allocation. The CPU cluster still "
        "offers this role for the ASE fallback route, which is the implementation "
        "pair that makes the OpenMM numbers checkable.",
    ),
    "collect": (
        "openqha_collect_executor",
        "Branch B: turning trajectories into a number. Reads frames, does the "
        "quasi-harmonic analysis, writes one result per molecule. ONE CORE PER "
        "MOLECULE, 64 at a time, on ONE node -- it is short and I/O-bound on Lustre, "
        "so more nodes would buy metadata contention rather than throughput.",
        "CPU cluster (TianheXY-C)",
    ),
    "labels": (
        "openqha_labels_executor",
        "Hessian-learning set: reference E-F-H labels per frame from ORCA (single point + "
        "EnGrad + analytic Hessian at the frame's fixed geometry; workflows/hessian_learning/"
        "03_labels.py). ONE FRAME PER TASK, 4 ORCA ranks each, 16 frames per 64-core node "
        "(user ruling 2026-09-18): an analytic Hessian on a 10-atom molecule is minutes, "
        "and 16 independent frames fill a node better than one 64-rank job.",
        "CPU cluster (TianheXY-C)",
    ),
    "qm": (
        "openqha_qm_executor",
        "Branch C: reference labels from ORCA (RI-MP2). MPI-parallel, and the whole "
        "allocation belongs to one calculation.",
        "CPU cluster (TianheXY-C)",
    ),
    "train": (
        "openqha_train_executor",
        "Branch C: MACE fine-tuning with the PHL loss. The one part of this repository "
        "that is genuinely faster on a GPU.",
        "GPU cluster (TianheXY-AI)",
    ),
}

#: Standby variants. ALF's pattern: same role, lower priority or shorter walltime, used
#: when a cheaper queue is available. Optional everywhere -- a config that omits one is
#: valid, and nothing schedules to a label that does not exist.
STANDBY = {role: label.replace("_executor", "_standby_executor")
           for role, (label, _d, _c) in ROLES.items()}

#: Every label this project may legitimately use.
ALL = sorted([lab for lab, _d, _c in ROLES.values()] + list(STANDBY.values()))

#: Labels used before 2026-09-05, mapped to their replacements. Kept so an old config or
#: an old driver fails with a sentence instead of scheduling into a void.
LEGACY = {
    "openqha_crest": "openqha_crest_executor",
    "openqha_qha": "openqha_qha_executor",
    "openqha_qha_openmm": "openqha_qha_executor",
    "openqha_crest_tianhe": "openqha_crest_executor",
}


def label(role, standby=False):
    """The canonical label for a role. Raises on an unknown role rather than inventing one."""
    if role not in ROLES:
        raise KeyError("no such executor role: {!r}. Known roles: {}".format(
            role, ", ".join(sorted(ROLES))))
    return STANDBY[role] if standby else ROLES[role][0]


def resolve(name):
    """Map a possibly-legacy label to the canonical one, or return it unchanged.

    Returns (canonical, was_legacy). A caller that wants to be strict can refuse when
    `was_legacy` is true; a caller that wants to keep working can warn and carry on.
    """
    if name in ALL:
        return name, False
    if name in LEGACY:
        return LEGACY[name], True
    return name, False


def check(config, expect=None):
    """Validate the labels on a built Parsl Config.

    `expect` is the label the driver is about to bind, if it knows. The check that
    matters is not "is this label spelled correctly" but **"is the label the app asks for
    actually present in this config"** -- the failure it exists to prevent is a silent
    no-match at run time.

    Returns a dict describing what was found; raises ValueError on a real mismatch.
    """
    found = [ex.label for ex in config.executors]
    unknown = [l for l in found if l not in ALL and l not in LEGACY]
    legacy = [l for l in found if l in LEGACY]

    if expect is not None:
        canonical, _ = resolve(expect)
        if expect not in found and canonical not in found:
            raise ValueError(
                "no executor labelled {!r} in this config.\n"
                "  config provides : {}\n"
                "  known labels    : {}\n"
                "A Parsl app that names a missing executor does not fail at build time -- "
                "it fails to schedule, quietly, which is why this is checked here."
                .format(expect, found, ALL))

    return dict(found=found, unknown=unknown, legacy=legacy,
                legacy_map={l: LEGACY[l] for l in legacy})


def describe():
    """The label table, for a checkpoint or a `--help`."""
    rows = []
    for role, (lab, what, where) in sorted(ROLES.items()):
        rows.append(dict(role=role, label=lab, standby=STANDBY[role],
                         runs=what, cluster=where))
    return rows


if __name__ == "__main__":
    for r in describe():
        print("{label}\n    role     {role}\n    standby  {standby}\n"
              "    cluster  {cluster}\n    runs     {runs}\n".format(**r))
