"""What is left to compute: scan what is done, subtract it, and say how you decided.

A 133 885-molecule campaign is never one run. It is a run, a queue limit, a restart, a
node that went away, another restart. So the worklist has to be derived from what is on
disk, not from a range typed on a command line.

FOUR LESSONS TAKEN FROM `00_QM9_reaction_eng/hkuhpc/sbatch_tianhe/count_remain_fixed.py`
----------------------------------------------------------------------------------------
That script is this project's own earlier solution to the same problem on the same
filesystem, and its fix notes are the interesting part:

  * **One `scandir` per directory, never one `stat` per molecule.** On Lustre a
    per-molecule `os.path.isfile()` over 133k molecules is hundreds of thousands of
    metadata round-trips and was measured "tens of times" slower. `basin_store.completed()`
    already globs per directory, so this module builds on it rather than re-walking.
  * **QM9 indices are not contiguous.** A range minus the completed set contains many
    indices that have no input at all; without filtering them the remaining list is
    inflated and the job spends its allocation printing "not found". Here the filter is
    the F0-F7 gates plus the presence of a geometry, which is stricter and is the same
    rule production uses.
  * **Print the completion criterion.** If the scanner and the worker disagree about what
    "done" means, molecules are skipped for ever and never recovered. So the criterion is
    stated in the returned record and printed by the driver -- not left implicit.
  * **One definition of "completed", shared.** `is_complete()` below is that definition;
    the scanner and any resubmit tooling must both call it rather than each testing for a
    file they happen to know about.
"""
from __future__ import annotations

from . import basin_store, config as _config, filters

#: What has to exist for a molecule to count as done.
#:
#: Deliberately BOTH files. The JSON is the record and the xyz is the deliverable; a run
#: killed between the two writes leaves the first without the second, and treating that
#: as complete loses the geometries silently. `basin_store.write()` renames each into
#: place after writing a `.part`, so a half-written file never has the final name -- but
#: a job can still die between the two renames.
COMPLETION = "both {qid}.basins.json and {qid}.basins.xyz exist in the shard"


def is_complete(qm9_index, cfg=None, tag=None):
    """The single definition of "done". Scanner and resubmitter must both use it."""
    j, x = basin_store.paths_for(qm9_index, cfg, tag)[:2]
    return j.exists() and x.exists()


def completed(cfg=None, tag=None, chunk_dir=None):
    """Indices already computed, by directory listing rather than per-molecule stat."""
    return basin_store.completed(cfg=cfg, tag=tag, chunk_dir=chunk_dir)


def remaining(candidates, cfg=None, tag=None, chunk_dir=None, apply_gates=True,
              require_geometry=True, progress=None):
    """Which of `candidates` still need computing, and an account of how that was decided.

    Returns (worklist, record). The record is meant to be printed and stored: a resume
    that cannot say why it skipped 2522 molecules is a resume nobody can audit.
    """
    cfg = cfg or _config.load()
    done = completed(cfg=cfg, tag=tag, chunk_dir=chunk_dir)

    todo, skipped_done, skipped_gate, skipped_missing = [], [], [], []
    gate_reason = {}
    #: Lookups that raised something other than "not there". Counted apart from genuine
    #: absences on purpose: the first version of this function caught every exception and
    #: reported a TypeError as "no geometry", so 199 of 200 molecules were dropped for a
    #: reason that was not true.
    lookup_errors = {}
    for i, qid in enumerate(candidates):
        if progress and i % 5000 == 0:
            progress("screening {}/{}".format(i, len(candidates)))
        idx = basin_store._to_int(qid)
        if idx in done:
            skipped_done.append(idx)
            continue
        if require_geometry:
            # The STRING identifier: config.qm9_xyz builds `qm9_index + ".xyz"`.
            try:
                path = _config.qm9_xyz(_qid(idx), cfg)
            except FileNotFoundError:
                path = None                      # a genuine absence
            except Exception as exc:             # noqa: BLE001
                # NOT an absence. Record it as what it is and do not pretend the
                # molecule has no geometry.
                key = type(exc).__name__
                lookup_errors[key] = lookup_errors.get(key, 0) + 1
                skipped_missing.append(idx)
                continue
            if path is None or not _exists(path):
                skipped_missing.append(idx)
                continue
        if apply_gates:
            try:
                smiles = _config.qm9_smiles(_qid(idx), cfg)
                ok, gate, why = filters.screen(smiles, cfg, identifier=_qid(idx))
            except Exception as exc:                       # noqa: BLE001
                # A gate that cannot be evaluated is NOT a gate that rejected. It is
                # counted under its own exception name, so "dropped by F5" and "the
                # screen crashed" never look the same in the record.
                key = type(exc).__name__
                lookup_errors[key] = lookup_errors.get(key, 0) + 1
                ok, gate = False, "screen_raised:" + key
            if not ok:
                skipped_gate.append(idx)
                gate_reason[gate] = gate_reason.get(gate, 0) + 1
                continue
        todo.append(_qid(idx))

    record = dict(
        completion_criterion=COMPLETION,
        tag=tag, chunk_dir=chunk_dir,
        n_candidates=len(candidates),
        n_already_done=len(skipped_done),
        n_dropped_no_geometry=len(skipped_missing),
        n_dropped_by_gate=len(skipped_gate),
        dropped_by_gate_breakdown=gate_reason,
        n_remaining=len(todo),
        n_lookup_errors=sum(lookup_errors.values()),
        lookup_errors=lookup_errors,
        gates_applied=bool(apply_gates),
        geometry_required=bool(require_geometry),
        note=("Counts are derived by listing shard directories, not by stat-ing each "
              "molecule: on a parallel filesystem the latter is hundreds of thousands "
              "of metadata round-trips."),
        warning=("n_lookup_errors > 0 means some molecules were dropped because a "
                 "lookup RAISED, not because they are genuinely absent or rejected. "
                 "That is a bug to fix, not a property of the data -- investigate "
                 "before trusting n_remaining."
                 if lookup_errors else None),
    )
    return todo, record


def _qid(idx):
    return "dsgdb9nsd_{:06d}".format(int(idx))


def _exists(path):
    import pathlib
    try:
        return pathlib.Path(str(path)).exists()
    except OSError:
        return False


def index_range(start, end):
    """[start, end] inclusive, as QM9 identifiers. Not filtered -- `remaining` does that."""
    if start < 1 or end < start:
        raise ValueError("bad range: {}..{}".format(start, end))
    return [_qid(i) for i in range(int(start), int(end) + 1)]
