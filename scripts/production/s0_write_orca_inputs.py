"""Write the stage 0 DLPNO-CCSD(T)/cc-pVTZ inputs to the hkuhpc / deimos convention.

PRODUCTION. Writes the ORCA inputs the reference calculations are run from.

**Why this script exists**: all the user's other work is produced with the same settings,
and stage 0 must match them, or the systematic errors of the DLPNO truncation, the auxiliary
basis approximation and the program version cannot cancel against that batch.

The convention, verbatim from:
  * `00_QM9_reaction_eng/hkuhpc/sbatch_tianhe/gen_orca_input.sh`
        - keyword line `! DLPNO-CCSD(T) <basis> <aux>`; the aux for cc-pVTZ is `cc-pVTZ/JK RIJK cc-pVTZ/C`
        - `%mdci  TCutPairs 1e-6  printlevel 4 end`
        - `%loc   LocMet AHFB  OCC true end`
        - the geometry is given as `*xyzfile 0 1 <path>`
  * the cluster table in `00_QM9_reaction_eng/hkuhpc/sbatch_tianhe/core-bind/orca.md`:
        deimos = ORCA **5.0.4**, 64 cores, 512 GB   |   intel = ORCA 6.1.0, 32 cores, 128 GB
    stage 0 uses **deimos / 5.0.4**.
  * every PMI* and SLURM* variable must be unset before starting, and ORCA started by absolute path (a Tianhe requirement).

**This script only writes inputs; it never calls ORCA** -- there is no ORCA on this machine.

Usage:
    python scripts/production/s0_write_orca_inputs.py            # read the geometry back from the existing .inp
Products:
    docs/orca_inputs/deimos_504/<edge>_<species>.{xyz,inp}
    docs/orca_inputs/deimos_504/run_hint.sh
"""
import json
import sys
from pathlib import Path


def _repo_root():
    """Directory holding the openQHA package, found by walking up from this file.

    Depth-independent on purpose: this file keeps working wherever under the
    repository it is moved to. The earlier move into `scripts/_superseded/` broke
    every `parents[1]` in the moved files silently, which is what this replaces.
    """
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))
from openqha.potentials import engine  # noqa: E402

ORCA = ROOT / "docs" / "orca_inputs"
OUT = ORCA / "deimos_504"

EDGE = "C2H5O1N1_19_36"
SPECIES = ["acetamide", "N-methylformamide"]

# ---- Parameter classes -------------------------------------------------------------------
# Method and thresholds: literature values / matching conventions -- changing them makes this incomparable with the user's other work
KEYWORD_LINE = "! DLPNO-CCSD(T) cc-pVTZ cc-pVTZ/JK RIJK cc-pVTZ/C"
MDCI_BLOCK = "%mdci\n   TCutPairs 1e-6\n   printlevel 4\nend"
LOC_BLOCK = "%loc\nLocMet AHFB\nOCC true\nend"
ORCA_VERSION = "5.0.4"          # deimos
# Resource budget: node-dependent. The Tianhe worker rewrites these two lines at run time; never trust the values written here
MAXCORE_MB = 3900
NPROCS = 4

TEMPLATE = """{keyword}
%maxcore {maxcore}
%pal nprocs {nprocs} end
{mdci}
{loc}
*xyzfile 0 1 {xyzname}
"""


def read_geometry(name):
    """Read the MACE-OFF23-SC optimised geometry (elements + coordinates) back from an existing ORCA input."""
    p = ORCA / "{}_{}".format(EDGE, name) / "{}_{}.inp".format(EDGE, name)
    if not p.exists():
        raise FileNotFoundError("no existing input, so the geometry cannot be read back: {}".format(p))
    lines = p.read_text(encoding="utf-8").split("\n")
    i = next(i for i, l in enumerate(lines) if l.startswith("* xyz"))
    rows = []
    for l in lines[i + 1:]:
        if l.strip() == "*":
            break
        f = l.split()
        rows.append((f[0], float(f[1]), float(f[2]), float(f[3])))
    if not rows:
        raise ValueError("empty geometry: {}".format(p))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    record = dict(edge=EDGE, orca_version=ORCA_VERSION, cluster="deimos",
                  keyword_line=KEYWORD_LINE,
                  mdci="TCutPairs 1e-6, printlevel 4",
                  loc="LocMet AHFB, OCC true",
                  maxcore_mb=MAXCORE_MB, nprocs=NPROCS,
                  geometry_source="the lowest MACE-OFF23-SC optimised conformer -- digit-for-digit identical to the existing .inp",
                  composite_notation=engine.composite_notation(reference="dlpno-ccsdt_cc-pvtz", name="MACE-OFF23-SC"),
                  provenance="00_QM9_reaction_eng/hkuhpc/sbatch_tianhe/gen_orca_input.sh "
                             "and core-bind/orca.md",
                  files={})
    print("=" * 88)
    print("writing inputs to the deimos / ORCA {} convention   edge {}".format(ORCA_VERSION, EDGE))
    print("=" * 88)
    print(KEYWORD_LINE)
    print(MDCI_BLOCK)
    print(LOC_BLOCK)
    print()
    for name in SPECIES:
        rows = read_geometry(name)
        stem = "{}_{}".format(EDGE, name)
        xyz = OUT / (stem + ".xyz")
        inp = OUT / (stem + ".inp")
        xyz.write_text(
            "{}\n{}  MACE-OFF23-SC optimised lowest conformer\n".format(len(rows), stem)
            + "".join("{:2s} {:18.10f} {:18.10f} {:18.10f}\n".format(*r) for r in rows),
            encoding="utf-8")
        inp.write_text(TEMPLATE.format(keyword=KEYWORD_LINE, maxcore=MAXCORE_MB,
                                       nprocs=NPROCS, mdci=MDCI_BLOCK, loc=LOC_BLOCK,
                                       xyzname=xyz.name), encoding="utf-8")
        record["files"][name] = dict(inp=str(inp.relative_to(ROOT)),
                                     xyz=str(xyz.relative_to(ROOT)),
                                     n_atoms=len(rows))
        print("  {:20s} {} atom(s) -> {} + {}".format(name, len(rows), inp.name, xyz.name))

    hint = OUT / "run_hint.sh"
    hint.write_text("""#!/usr/bin/env bash
# stage 0 electronic term -- the deimos / ORCA 5.0.4 calling convention (taken point by point from the hkuhpc and stage 2 cluster scripts)
#
# 1) Tianhe/deimos require every PMI* and SLURM* variable to be unset before ORCA starts
# 2) start ORCA by absolute path; do not rely on PATH
# 3) rewrite %maxcore and %pal at run time -- never trust the node-dependent values written here
# 4) compute in a temporary directory, then move the finished product into place atomically
#
# The two species must use **the same ORCA version, the same keyword line and the same thresholds**,
# or the DLPNO truncation error does not cancel between the two ends of the reaction.
# `set -u` is deliberately not used: a failing step must not end the job.
ORCA_BIN="${ORCA_BIN:?set ORCA_BIN to the absolute orca binary path (deimos: ORCA 5.0.4)}"

for var in $(env | awk -F= '{print $1}' | grep -E '^(PMI|SLURM)'); do unset "$var"; done

for stem in __STEMS__; do
  tmp=$(mktemp -d)
  cp "${stem}.inp" "${stem}.xyz" "$tmp/"
  ( cd "$tmp" && "$ORCA_BIN" "${stem}.inp" > "${stem}.out" 2>&1 )
  mv "$tmp/${stem}.out" ./
  rm -rf "$tmp"
done
""".replace("__STEMS__", " ".join("{}_{}".format(EDGE, n) for n in SPECIES)),
                    encoding="utf-8")
    print()
    print("  calling hint ->", hint.name)

    (OUT / "provenance.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print("  provenance   -> provenance.json")
    print()
    print("**The two existing single points (ORCA 6.0.1, TightSCF TightPNO, no RIJK) do not")
    print("follow this convention, are therefore void, and must not be mixed with results")
    print("produced under it.**")


if __name__ == "__main__":
    main()
