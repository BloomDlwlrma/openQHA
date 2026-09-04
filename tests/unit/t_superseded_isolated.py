"""Unit test: nothing live depends on anything retired.

UNIT. Branch D, plan_D acceptance criterion 7 and section 8.5 criterion 4. Seconds.

Retired work is KEPT, never deleted -- memory-discard.md section 7: a refuted result
stays written down so it does not get re-cited as live. That policy only holds if the
retired material is genuinely off every path. Otherwise "kept as evidence" quietly
becomes "still in use", and a number that was ruled out goes on being produced.

Two ways that can happen, and this test looks for both:

A. a live module imports a retired one;
B. live CODE names a path under a `_superseded/` directory.

Documentation is not a dependency
---------------------------------
Saying in a comment where something WENT is provenance, and it is exactly what
`memory-discard.md` asks for. So comments and docstrings are blanked before the scan
(`tests/_testlib.code_only`); only executable references count. The first version of
this test lacked that and reported seven hits, every one of them a sentence explaining
a retirement -- which would have taught the repository to stop writing those sentences.

Three files are exempt by name, each for a stated reason, because their PURPOSE is to
handle retired material. An exemption without a reason is just a hole.

Run::  python tests/unit/t_superseded_isolated.py
"""
import re
import sys
from pathlib import Path

for _p in Path(__file__).resolve().parents:          # depth-independent, like everywhere else
    if (_p / "openqha" / "__init__.py").is_file():
        sys.path.insert(0, str(_p / "tests"))
        break
from _testlib import code_only, repo_root, uncommented  # noqa: E402

ROOT = repo_root(__file__)
LIVE = ("openQHA", "scripts", "tests", "examples", "hpc", "configs")
SKIP_PARTS = {"__pycache__", "source-code", "_backup"}
RETIRED = "_superseded"

#: Files whose job is to manage retired material. Each entry carries its reason.
EXEMPT = {
    "scripts/tooling/openqha_migrate_analysis.py":
        "it is the migration tool; writing into analysis/_superseded/ is what it does",
    "scripts/tooling/openqha_index_artifacts.py":
        "it indexes every artifact, retired ones included, and reports their status",
    "tests/unit/t_superseded_isolated.py":
        "this test",
    "tests/unit/t_script_taxonomy.py":
        "it checks that retired scripts stay separated and keep their README",
    "tests/unit/t_code_is_english.py":
        "it counts the retired files still in the original language",
}

#: Single occurrences acknowledged one at a time, each with its reason and each
#: verified not to be an access. Prose can also live inside a string VALUE rather than
#: a docstring -- a note in a dictionary, say -- and no textual rule can tell that from
#: a real dependency. Rather than widen the rule until it stops catching things, such
#: cases are listed here individually, so that adding one is a deliberate act with a
#: justification attached instead of a quiet hole.
ACKNOWLEDGED = {
    ("hpc/providers.py", "_superseded/cluster_2/s0_submit_array.slurm"):
        "a note recording that the direct array-submission shape is kept as a fallback "
        "if Parsl proves unworkable on Tianhe; providers.py never opens the file",
}

FAIL = []


def live_files(exts):
    for top in LIVE:
        d = ROOT / top
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*")):
            if not p.is_file() or p.suffix not in exts:
                continue
            if set(p.parts) & SKIP_PARTS or RETIRED in p.parts or p.name.endswith(".bak"):
                continue
            if p.relative_to(ROOT).as_posix() in EXEMPT:
                continue
            yield p


def retired_names():
    """Stems of retired scripts, read from disk so the check cannot go stale."""
    names = set()
    for base in (ROOT / "scripts" / RETIRED, ROOT / RETIRED):
        if base.is_dir():
            for p in base.rglob("*"):
                if p.is_file() and p.suffix in (".py", ".sh"):
                    names.add(p.stem)
    return names


def main():
    names = retired_names()
    print("retired scripts on disk: {}".format(len(names)))
    print("exempt by stated reason: {}".format(len(EXEMPT)))
    for f, why in sorted(EXEMPT.items()):
        print("    {:<52s} {}".format(f, why))

    print("\nA. no live module imports a retired one")
    n = 0
    import_re = re.compile(r"^\s*(?:from|import)\s+([A-Za-z_][\w.]*)", re.M)
    for p in live_files({".py"}):
        for m in import_re.finditer(code_only(p.read_text(encoding="utf-8"))):
            head = m.group(1).split(".")[0]
            if head in names or head == RETIRED:
                n += 1
                FAIL.append("{} imports {}".format(p.relative_to(ROOT), m.group(1)))
                print("  FAIL  {} imports {}".format(p.relative_to(ROOT), m.group(1)))
    print("  {} offending import(s)".format(n))

    print("\nB. no live CODE names a path under a _superseded/ directory")
    path_re = re.compile(r"[\w./-]*_superseded/[\w./-]+")
    n = n_doc = n_ack = 0
    seen_ack = set()
    for p in live_files({".py", ".yaml", ".yml", ".toml", ".template", ".sh", ".json",
                         ".mdp"}):
        raw = p.read_text(encoding="utf-8", errors="replace")
        stripped = code_only(raw) if p.suffix == ".py" else uncommented(raw)
        n_doc += len(path_re.findall(raw)) - len(path_re.findall(stripped))
        rel = p.relative_to(ROOT).as_posix()
        for m in path_re.finditer(stripped):
            if (rel, m.group(0)) in ACKNOWLEDGED:
                n_ack += 1
                seen_ack.add((rel, m.group(0)))
                print("  ack   {:<52s} {}".format(rel, m.group(0)))
                continue
            n += 1
            FAIL.append("{}: {}".format(rel, m.group(0)))
            print("  FAIL  {:<52s} {}".format(rel, m.group(0)))
    print("  {} unexplained reference(s); {} acknowledged; {} mention(s) in comments, "
          "which are provenance and allowed".format(n, n_ack, n_doc))

    stale = sorted(k for k in ACKNOWLEDGED if k not in seen_ack)
    if stale:
        for k in stale:
            FAIL.append("acknowledgement no longer matches anything: {}".format(k))
            print("  FAIL  stale acknowledgement {} -- remove it".format(k))

    print("\nC. the retired directory still carries its explanation")
    readme = ROOT / "scripts" / RETIRED / "README.md"
    if readme.parent.is_dir() and not readme.is_file():
        FAIL.append("{} missing".format(readme.relative_to(ROOT)))
        print("  FAIL  {} missing".format(readme.relative_to(ROOT)))
    elif readme.is_file():
        print("  ok    {}".format(readme.relative_to(ROOT)))

    print()
    if FAIL:
        print("{} problem(s):".format(len(FAIL)))
        for f in FAIL:
            print("  - " + f)
        return 1
    print("retired work is off every live path")
    return 0


if __name__ == "__main__":
    sys.exit(main())
