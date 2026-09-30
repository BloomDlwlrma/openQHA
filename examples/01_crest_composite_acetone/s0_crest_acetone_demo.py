"""Acetone through the CREST composite calculator -- the smallest end-to-end example.

EXAMPLE. Not a production driver and not a calibration: it runs ONE molecule with the
shipped settings and prints what came back, so that a reader can see the machinery work
before reading `scripts/production/s0_A_pipeline.py`.

    python -m openqha.potentials.mace_server --socket /tmp/s0_mace_engrad.sock &
    S0_MACE_SOCKET=/tmp/s0_mace_engrad.sock \\
        python examples/01_crest_composite_acetone/s0_crest_acetone_demo.py

What it is NOT
--------------
This runs the CREST stage only -- sampling plus MACE refinement. It does not tighten to
fmax = 1e-4 eV/A, compute Hessians, reject saddle points, derive symmetry numbers or
apply the acceptance criteria. **A conformer count from here is not a basin count**, and
this repository has been caught by that distinction before: on this very molecule CREST
reported 2 conformers 0.8118 kcal/mol apart, and after tightening their energies were
identical to the last digit. For the whole chain use:

    python scripts/production/s0_A_pipeline.py --species dsgdb9nsd_000018

Everything comes from the config
--------------------------------
Nothing here hard-codes a setting. `configs/conformers.yaml` is the authority, so this
example follows the shipped protocol automatically -- including the change of `refine`
from "opt" to "sp" on 2026-09-04, which this file would otherwise still be advertising.
The dynamics package (SHAKE all bonds + 5 fs + hydrogen mass 2 amu; Grimme, JCTC 2019,
15, 2847) is passed as ONE package, because passing two of the three is how a run departs
from the published protocol without anything saying so.
"""
import json
import os
import sys
from pathlib import Path


def _repo_root():
    """Directory holding the openQHA package, found by walking up from this file.

    Depth-independent on purpose: this file keeps working wherever under the repository
    it is moved to. An earlier move into `scripts/_superseded/` broke every `parents[1]`
    in the moved files silently, which is what this replaces.
    """
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


sys.path.insert(0, str(_repo_root()))
from openqha import S0_ROOT, config, crest  # noqa: E402

CFG = config.load()
C = CFG["crest"]
OUT = S0_ROOT / "analysis" / "package1" / "crest_acetone"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    xyz = OUT / "acetone.xyz"
    if not xyz.exists():
        raise FileNotFoundError(
            "no starting geometry at {}. Copy one in, for example\n"
            "    cp data/reference-geometries/dsgdb9nsd_000018.xyz {}".format(xyz, xyz))

    print("=" * 92)
    print("Acetone -- CREST composite calculator "
          "({} workhorse samples, MACE refines)".format(C["workhorse"]))
    print("=" * 92)
    v = crest.crest_version()
    print("CREST        {} (commit {})".format(v["version"], v["commit"]))
    print("mlip support {}   <- needs CREST 3.1; this repo drives 3.0.2 through "
          "`generic`".format(crest.supports_mlip()))
    print("settings     runtype={} workhorse={} refine={} optlev={} threads={} "
          "backend={}".format(C["runtype"], C["workhorse"], C["refine"], C["optlev"],
                              C["threads"], C["backend"]))
    print("dynamics     shake={} tstep={} fs hmass={} amu   <- one package, not three "
          "knobs".format(C["shake"], C["tstep_fs"], C["hydrogen_mass_amu"]))
    print("socket       {}".format(os.environ.get("S0_MACE_SOCKET", C["socket"])))
    print()

    # The dynamics package goes in whole. `shake` without `tstep_fs` would silently take
    # CREST's own timestep, and the run would no longer be the published protocol while
    # still looking like it.
    rec = crest.run(OUT, xyz, runtype=C["runtype"], threads=int(C["threads"]),
                    optlev=C["optlev"], refine=C["refine"], backend=C["backend"],
                    workhorse=C["workhorse"], shake=int(C["shake"]),
                    tstep_fs=float(C["tstep_fs"]),
                    hmass_amu=float(C["hydrogen_mass_amu"]),
                    engine_client=S0_ROOT / C["engine_client"], timeout_s=7200)

    print("wall         {:.0f} s = {:.2f} h".format(rec["seconds"], rec["seconds"] / 3600))
    print("finished     {}".format(rec["terminated_normally"]))
    print("terminated EARLY   {}   <- criterion: must be 0. If it is not, the production "
          "driver retries this molecule once at shake=1 and marks it "
          "(crest.run_with_shake_fallback)".format(rec["n_terminated_early"]))
    print("completed successfully  {}".format(rec["n_completed_successfully"]))
    print("energy+gradient calls   {}".format(rec["total_engrad_calls"]))
    print("products     {}".format(rec["products"]))
    print("record ok    {}".format(rec["ok"]))

    ens = OUT / "crest_conformers.xyz"
    if ens.exists():
        frames = crest.read_ensemble(ens)
        rec["n_conformers"] = len(frames)
        print()
        print("conformers   {}   <- CREST's count, NOT a basin count (see the module "
              "docstring)".format(len(frames)))
        e = []
        for c, _ in frames:
            try:
                e.append(float(c.split()[0]))
            except (ValueError, IndexError):
                pass
        if e:
            e0 = min(e)
            print("relative energies as CREST reports them (kcal/mol): {}".format(
                " ".join("{:.4f}".format((x - e0) * 627.5094740631) for x in e)))
            print("  these are workhorse energies on workhorse geometries; the pipeline "
                  "re-optimises and re-ranks them.")

    (OUT / "run_record.json").write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    print("written: {}".format(OUT / "run_record.json"))
    if not rec["ok"]:
        print()
        print("** the record did not pass its own check; tail of the output:")
        print(rec.get("tail", ""))


if __name__ == "__main__":
    main()
