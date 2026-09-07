"""Species selection gates F0-F6 -- rewritten in meaning from stage 1, **the code is
this repository's own**.

User ruling 2026-08-28: **consider only neutral molecules with no unpaired electrons
and 0 net charge**. Stage 1's `docs/02_data_ingestion-and-filtering.md` section 4.1
broke this into gates that can be counted separately; each is rewritten here (stage 0
is an independent repository and imports nothing from stage 1).

| Gate | Criterion | What it removes |
|---|---|---|
| F0 | `MolFromSmiles` plus sanitisation succeed | molecules no route can recognise at all |
| F1 | exactly one connected fragment | multi-fragment species (complexes, salts) |
| F3 | formal charges sum to 0 | ions |
| F4 | 0 unpaired electrons | carbenes / nitrenes / formal diradicals |
| F5 | no atom carries a non-zero formal charge | zwitterions / ylides / 1,3-dipoles |
| F6 | elements are a subset of {H, C, N, O} | fluorine-bearing species (**off by default** in stage 0; the reason is in the configuration) |
| F7 | not on QM9's official "uncharacterized" list | the 3054 whose **deposited geometry and deposited SMILES are not the same molecule** |

**The order is normative and short-circuits**: the first gate to fail decides, so every
rejection is attributed to exactly one cause.

**The difference between F3 and F5**: F3 measures the **sum** of formal charges (is it
an ion?); F5 measures **existence** (does any atom carry a charge?). F5 is strictly
stronger, and what it catches is the zwitterion whose net charge is zero.

**Why F7 comes last**: F0-F6 judge the **SMILES**; F7 judges **whether the record is
self-consistent** -- whether the deposited geometry and the deposited SMILES are the
same molecule. Putting F7 last leaves every existing F0-F6 attribution **unchanged**,
so F7's count reads exactly as "molecules that passed all the other gates but whose
deposited geometry cannot be trusted". F7 needs the **QM9 index** (the list is
published by index), which makes it the only gate that cannot work from the SMILES
alone -- **enabling F7 without supplying an index raises, it does not skip silently**.

> **F7 was added on 2026-09-02 (defect 63).** Before it, `dsgdb9nsd_003838` (index
> SMILES `N=C1N=CON=N1`, while the deposited geometry parses to `[NH][C][N]C=O` and
> `N#N`, **two fragments**) ran all the way through CREST (**1399 seconds**) only to be
> caught at the graph-isomorphism step.

> **This gate was added on 2026-08-28.** Before it, package 1 ran all 4000 molecules of
> indices 1 to 4000, of which **175 carried unpaired electrons** (4.38%, e.g.
> `CO[C](C)[NH]`), **5 had SMILES that would not parse because of unconventional
> valence** (zwitterions), and **1 was an explicit zwitterion**. Those 5 were at the
> time **rescued** by "assigning formal charges from valence" (`D0-38`) -- **which was
> wrong**: what that produced was precisely the zwitterions F5 exists to remove. See
> `D0-41`.
"""
import pathlib as _pathlib

from .. import config as _config

GATE_ORDER = ("F0", "F1", "F3", "F4", "F5", "F6", "F7")

GATE_DESCRIPTION = {
    "F0": "SMILES parses and sanitises",
    "F1": "exactly one connected fragment",
    "F3": "formal charges sum to 0",
    "F4": "0 unpaired electrons",
    "F5": "no atom carries a non-zero formal charge",
    "F6": "elements are a subset of {H, C, N, O}",
    "F7": "not on QM9's official \"uncharacterized\" list (deposited geometry and "
          "deposited SMILES are the same molecule)",
}

_CONFIG_KEY = {
    "F0": "F0_parses", "F1": "F1_single_fragment", "F3": "F3_net_charge_zero",
    "F4": "F4_no_radicals", "F5": "F5_no_formal_charge", "F6": "F6_elements_HCNO",
    "F7": "F7_qm9_geometry_consistent",
}

ALLOWED_ELEMENTS = {"H", "C", "N", "O"}

#: Default scope of F7. `"all"` = remove all 3054 on the list (user request 2026-09-02).
F7_SCOPE_DEFAULT = "all"


def enabled_gates(cfg=None):
    cfg = cfg or _config.load()
    f = cfg.get("species_filter", {})
    if not f.get("enabled", True):
        return ()
    g = f.get("gates", {})
    return tuple(k for k in GATE_ORDER if g.get(_CONFIG_KEY[k], False))


def screen(smiles, cfg=None, gates=None, identifier=None):
    """Judge one SMILES. Returns (passed, failing gate or None, a one-line reason).

    **Short-circuits**: it returns at the first gate that fails, so attribution is
    unique.

    `identifier` is the QM9 index (`dsgdb9nsd_XXXXXX` or an integer) and **only F7 uses
    it**. When F7 is enabled and `identifier` is None this **raises ValueError** --
    an enabled gate that silently passes everything for want of an input is worse than
    no gate at all.
    """
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    gates = enabled_gates(cfg) if gates is None else tuple(gates)
    if "F7" in gates and f7_mode(cfg) == "off":
        gates = tuple(g for g in gates if g != "F7")
    if not gates:
        return True, None, "filtering is not enabled"

    mol = Chem.MolFromSmiles(smiles)
    if "F0" in gates and mol is None:
        return False, "F0", "SMILES failed to parse or sanitise: {}".format(smiles)
    if mol is None:
        # F0 is off but the molecule still will not parse -- the later gates have
        # nothing to work on, so report it honestly rather than counting it as a pass
        return False, "F0", ("SMILES failed to parse (F0 is not enabled, but there is "
                             "no molecule to judge): {}".format(smiles))

    if "F1" in gates:
        n = len(Chem.GetMolFrags(mol))
        if n != 1:
            return False, "F1", "{} connected fragments (complex or salt)".format(n)
    if "F3" in gates:
        q = Chem.GetFormalCharge(mol)
        if q != 0:
            return False, "F3", "net formal charge {:+d} (ion)".format(q)
    if "F4" in gates:
        r = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
        if r:
            sites = [a.GetSymbol() + str(a.GetIdx()) for a in mol.GetAtoms()
                     if a.GetNumRadicalElectrons()]
            return False, "F4", ("{} unpaired electron(s), on {} "
                                 "(carbene/nitrene/diradical)".format(r, ",".join(sites)))
    if "F5" in gates:
        charged = [(a.GetSymbol() + str(a.GetIdx()), a.GetFormalCharge())
                   for a in mol.GetAtoms() if a.GetFormalCharge() != 0]
        if charged:
            return False, "F5", "atoms carrying a formal charge {} (zwitterion/ylide)".format(charged)
    if "F6" in gates:
        bad = sorted({a.GetSymbol() for a in mol.GetAtoms()} - ALLOWED_ELEMENTS)
        if bad:
            return False, "F6", "contains {} -- outside {{H,C,N,O}}".format(",".join(bad))
    if "F7" in gates:
        if identifier is None:
            raise ValueError(
                "F7 needs the QM9 index (the list is published by index), but no "
                "identifier was passed. Either supply the index or turn "
                "F7_qm9_geometry_consistent off in the configuration -- "
                "**silently skipping an enabled gate is not allowed**.")
        from ..data import qm9_uncharacterized
        row = qm9_uncharacterized.entry(identifier)
        if row is not None and _f7_in_scope(identifier, cfg):
            mode = f7_mode(cfg)
            if mode == "curated":
                # The molecule is on the list, but curatedQM9 may have repaired it. The
                # gate then passes and the REPAIRED GEOMETRY IS USED -- config.qm9_xyz
                # resolves it, and the product records `geometry_source` so which
                # structure a number came from is never a guess.
                from ..data import curated_qm9
                path, kind = curated_qm9.find(identifier, cfg)
                if path is not None:
                    return True, None, (
                        "on QM9's uncharacterized list, but curatedQM9 has a "
                        "{} geometry for it ({}); f7_mode=curated keeps it and uses "
                        "that geometry".format(kind, path.name))
                return False, "F7", (
                    "on QM9's uncharacterized list AND absent from curatedQM9 -- one "
                    "of the molecules the repair workflow could not fix. GDB17 SMILES "
                    "{!r}, deposited geometry parses to {!r}".format(
                        row["smiles_gdb17"], row["smiles_b3lyp_xyz"]))
            return False, "F7", (
                "on QM9's official uncharacterized list: the GDB17 SMILES is {!r}, "
                "while the deposited B3LYP geometry parses to {!r} -- not the same "
                "molecule".format(row["smiles_gdb17"], row["smiles_b3lyp_xyz"]))
    return True, None, "passed {}".format("/".join(gates))


#: The three ways F7 may treat QM9's uncharacterized list. `curated` is recommended.
F7_MODES = ("drop_all", "curated", "off")

#: Default. `drop_all` reproduces every product made before 2026-09-04; switching the
#: default is a separate decision from making the switch exist.
F7_MODE_DEFAULT = "drop_all"


def f7_mode(cfg=None):
    """How F7 treats the uncharacterized list.

    drop_all  remove all 3054 on the list. Conservative and lossy: upstream repaired
              2988 of them and found only 66 genuinely unstable, so this drops ~2.2%
              of QM9 to avoid 0.05%.
    curated   pass a listed molecule IF curatedQM9 has a geometry for it, and use that
              geometry. Removes only what the repair workflow could not fix.
    off       do not apply F7. Debugging only -- it lets through molecules whose
              deposited geometry is a different molecule from their SMILES, which is
              what cost this repository 1399 s of CREST on dsgdb9nsd_003838.

    `curated` needs the archive on disk. If it is missing, this RAISES rather than
    quietly falling back to drop_all: silently changing which molecules are in a
    dataset, because a directory was absent, is exactly the kind of condition that
    cannot be read off the products afterwards.
    """
    cfg = cfg or _config.load()
    mode = str(cfg.get("species_filter", {}).get("f7_mode", F7_MODE_DEFAULT))
    if mode not in F7_MODES:
        raise ValueError(
            "f7_mode is {!r}; it must be one of {}".format(mode, ", ".join(F7_MODES)))
    if mode == "curated":
        from ..data import curated_qm9
        if not curated_qm9.available(cfg):
            raise FileNotFoundError(
                "f7_mode = 'curated' needs the curatedQM9 archive, which was not "
                "found. Unpack it under data/qm9/{} or set S0_CURATED_QM9.\n"
                "Refusing to fall back to drop_all: that would silently change which "
                "molecules are in the dataset, and nothing in the products would say "
                "so.".format(curated_qm9.DEFAULT_DIRNAME))
    return mode


def f7_scope(cfg=None):
    """The scope of F7. See `_f7_in_scope` and `f7_note` in the configuration."""
    cfg = cfg or _config.load()
    return str(cfg.get("species_filter", {}).get("f7_scope", F7_SCOPE_DEFAULT))


def _f7_in_scope(identifier, cfg=None):
    """Whether a molecule on the list **does** fall within F7's scope.

    Two scopes, both supported by measurement (the numbers are in `f7_note` in the
    configuration):

    * `"all"` (default) -- remove all 3054 on the list. The reason: upstream calls them
      "**uncharacterized**", that is, **the identity of the molecule is not
      established**; for a data set built on isomerisation reactions, taking an identity
      that OpenBabel guessed from a geometry as the molecule's identity is a risk we
      would be carrying ourselves.
    * `"identity_from_geometry_only"` -- remove only those whose identity in our index
      is **not** derived from the geometry. This repository's `qm9_smiles` carries a
      column `qm9_identity_source`: where its value is `relaxed`, the SMILES was itself
      **parsed from the B3LYP geometry**, so "geometry disagrees with SMILES" **does not
      hold for us** (measured: 1749 of the 3054 on the list are `relaxed`). What is
      genuinely inconsistent for us is the remaining `gdb17` (1270) and `none` (35) --
      and `dsgdb9nsd_003838` is one of the `gdb17` kind.

    **A third value is not allowed**: any other string raises rather than being treated
    silently as the default.
    """
    scope = f7_scope(cfg)
    if scope == "all":
        return True
    if scope != "identity_from_geometry_only":
        raise ValueError(
            "species_filter.f7_scope must be 'all' or 'identity_from_geometry_only', "
            "read {!r}".format(scope))
    return _identity_source(identifier, cfg) != "relaxed"


_IDENTITY_SOURCE = {}


def identity_source_available(cfg=None):
    """Can `qm9_identity_source` be read at all?

    False when the QM9 index is not present -- which is the DEFAULT state of a fresh
    clone, because the 119 MB index is not shipped. In that state `_identity_source`
    returns "" for every molecule and `f7_scope="identity_from_geometry_only"` therefore
    behaves exactly like `"all"`.

    That is the safe direction, and it is also invisible: without this function a caller
    cannot tell "this molecule's identity came from GDB17" from "there is no index".
    A test that cannot tell them apart passes for the wrong reason, which is what
    happened (t_filters_f7, 2026-09-05).
    """
    cfg = cfg or _config.load()
    try:
        src_path = str(_config.qm9_index_csv(cfg))
    except Exception:
        return False
    return bool(_IDENTITY_SOURCE.get(src_path)) or _pathlib.Path(src_path).is_file()


def _identity_source(identifier, cfg=None):
    """The `qm9_identity_source` column of the QM9 index. **Read once and cached.**

    This function is called once per molecule, 133884 times; calling `config.qm9_row()`
    each time would re-read the whole 133884-row index every time, which is O(n^2) and
    measured to take minutes or more. A lookup that finds nothing returns the empty
    string -- the caller then treats "not found" as inconsistent, which is the
    conservative direction.
    """
    import csv
    cfg = cfg or _config.load()
    try:
        src_path = str(_config.qm9_index_csv(cfg))
    except Exception:
        return ""
    if src_path not in _IDENTITY_SOURCE:
        table = {}
        try:
            with open(src_path, encoding="utf-8", newline="") as fh:
                for r in csv.DictReader(fh):
                    table[r["qm9_index"]] = r.get("qm9_identity_source", "")
        except Exception:
            table = {}
        _IDENTITY_SOURCE[src_path] = table
    key = identifier if isinstance(identifier, str) else "dsgdb9nsd_{:06d}".format(
        int(identifier))
    return _IDENTITY_SOURCE[src_path].get(key, "")


def screen_many(pairs, cfg=None):
    """`pairs` is an iterable of (identifier, SMILES).

    Returns (identifiers that passed, rejection records, counts).
    """
    import collections
    gates = enabled_gates(cfg)
    kept, rejected, counts = [], [], collections.Counter()
    for ident, smi in pairs:
        ok, gate, why = screen(smi, cfg, gates, identifier=ident)
        if ok:
            kept.append(ident)
            counts["passed"] += 1
        else:
            rejected.append(dict(identifier=ident, smiles=smi, gate=gate, reason=why))
            counts[gate] += 1
    return kept, rejected, dict(counts)
