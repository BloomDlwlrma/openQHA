"""The hkuhpc job bundle for a numerical reference level (ticket 32).

One directory per (molecule, level) holding one ORCA input per MACE basin, a worker
script, a SLURM script and a README with the point-count estimate, written here and
submitted by the user. The conventions are hkuhpc's (`sbatch_tianhe/core-bind/orca.md`,
`smoke_debug_32nodes_8mols.sbatch`): several 8-rank ORCA jobs per node rather than one
wide one (DLPNO-CCSD(T) scales poorly past 8), OS scheduling with ORCA's CPU binding off,
PMI/SLURM variables unset before the native launch, the terminal line looked for in the
last 100 lines, results copied back only on normal termination. `%maxcore` follows the
75 % rule (DLPNO overruns it 2-3x).

Point-count estimate for a molecule of N atoms at a level without analytic gradients:
    optimisation   ~ n_steps x (2 x 3N) energies  (two-sided numerical gradient)
    NumFreq        (2 x 3N) gradients x (2 x 3N) energies = (6N)^2
so oxetane (N = 10) is ~600 + 3,600 single points per basin.
"""
import textwrap
from pathlib import Path

from . import orca

DEFAULTS = dict(partition="intel", time="7-00:00:00", cpus_per_node=32, ranks_per_job=8,
                mem_gb=128, orca_bin_dir="/lustre1/g/chem_yangjun/orca6.1.0/orca-6.1.0-f.0_linux_x86-64/bin")
#: measured 2026-09-17: one DLPNO-CCSD(T)/cc-pVTZ (TightPNO) single point on oxetane
#: (10 atoms), 8 MPI ranks, local ORCA 6.0.1 -- the unit of the wall-time estimate
SECONDS_PER_POINT_8RANKS = 93.0


def point_estimate(n_atoms, n_opt_steps=10, seconds_per_point=SECONDS_PER_POINT_8RANKS):
    grad = 2 * 3 * int(n_atoms)
    total = int(n_opt_steps) * grad + grad * grad
    return dict(opt=int(n_opt_steps) * grad, numfreq=grad * grad, total=total,
                days_8ranks=total * seconds_per_point / 86400.0)


def write_bundle(out_dir, jobs, level, nprocs=None, cfg=None, **overrides):
    """`jobs`: list of dicts (name, symbols, positions, charge=0, mult=1). Writes
    <name>.inp per job, worker.sh, run.sbatch, README.md. Returns the paths."""
    s = dict(DEFAULTS)
    s.update({k: v for k, v in overrides.items() if v is not None})
    ranks = int(nprocs or s["ranks_per_job"])
    concurrent = max(1, int(s["cpus_per_node"]) // ranks)
    maxcore = int(0.75 * int(s["mem_gb"]) * 1024 / int(s["cpus_per_node"]))
    spec = orca.level_spec(level)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names, estimates = [], []
    for j in jobs:
        (out_dir / (j["name"] + ".inp")).write_text(
            orca.input_text(j["symbols"], j["positions"], spec["keywords"], ranks, maxcore,
                            j.get("charge", 0), j.get("mult", 1), spec["blocks"]), encoding="utf-8")
        names.append(j["name"])
        estimates.append((j["name"], len(j["symbols"]), point_estimate(len(j["symbols"]))))
    (out_dir / "jobs.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    worker = textwrap.dedent("""\
        #!/bin/bash
        # one ORCA job: $1 = job name (the .inp beside this script). Idempotent: a finished
        # .out is skipped. Only .out, .hess, .xyz, .engrad, .property.txt are published.
        set -uo pipefail
        name="$1"; here="$(cd "$(dirname "$0")" && pwd)"
        done_dir="$here/results/$name"; mkdir -p "$done_dir"
        if [ -f "$done_dir/$name.out" ] && tail -n 100 "$done_dir/$name.out" | grep -q "ORCA TERMINATED NORMALLY"; then
            echo "[$name] SKIP (finished)"; exit 0; fi
        scr="${ORCA_SCR:-/tmp}/openqha_$name.$$"; mkdir -p "$scr"; cp "$here/$name.inp" "$scr/"; cd "$scr"
        cleanup() { st=$?; cd /; rm -rf -- "$scr"; exit "$st"; }; trap cleanup EXIT
        for var in $(env | awk -F= '{print $1}' | grep -E '^(PMI|SLURM)'); do unset "$var"; done
        export OMPI_MCA_rmaps_base_oversubscribe=1 ORCA_SKIP_CPU_BIND=1 OMPI_MCA_btl=^openib
        export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
        "${ORCA_BIN_DIR}/orca" "$name.inp" > "$name.out" 2>&1
        if tail -n 100 "$name.out" | grep -q "ORCA TERMINATED NORMALLY"; then
            for ext in out hess xyz engrad property.txt; do [ -f "$name.$ext" ] && cp "$name.$ext" "$done_dir/"; done
            echo "[$name] DONE"
        else
            cp "$name.out" "$done_dir/$name.failed.out"; echo "[$name] FAILED"; exit 1
        fi
        """)
    (out_dir / "worker.sh").write_text(worker, encoding="utf-8")
    sbatch = textwrap.dedent("""\
        #!/bin/bash
        #SBATCH --job-name=openqha_{level}
        #SBATCH --partition={partition}
        #SBATCH --time={time}
        #SBATCH --nodes=1
        #SBATCH --ntasks=1
        #SBATCH --cpus-per-task={cpus}
        #SBATCH --hint=nomultithread
        #SBATCH --mem={mem}G
        #SBATCH --output=logs/%x_%j.out
        # {concurrent} ORCA jobs of {ranks} MPI ranks each, dispatched by xargs; the OS
        # scheduler places the ranks (ORCA's own binding is off in worker.sh).
        set -uo pipefail
        export ORCA_BIN_DIR="{orca_bin_dir}"
        export PATH="$ORCA_BIN_DIR:$PATH"; export LD_LIBRARY_PATH="$ORCA_BIN_DIR:${{LD_LIBRARY_PATH:-}}"
        [ -x "$ORCA_BIN_DIR/orca" ] || {{ echo "ORCA not found at $ORCA_BIN_DIR" >&2; exit 1; }}
        export ORCA_SCR="${{ORCA_SCR:-$PWD/scratch}}"; mkdir -p "$ORCA_SCR" logs results
        xargs -a jobs.txt -P {concurrent} -I{{}} bash worker.sh {{}}
        """).format(level=level, partition=s["partition"], time=s["time"], cpus=s["cpus_per_node"],
                    mem=s["mem_gb"], concurrent=concurrent, ranks=ranks, orca_bin_dir=s["orca_bin_dir"])
    (out_dir / "run.sbatch").write_text(sbatch, encoding="utf-8")
    # Slurm opens --output BEFORE the script runs (hpc skill rule 6): the mkdir inside the
    # script is too late, so the bundle ships with the directory
    (out_dir / "logs").mkdir(exist_ok=True)
    lines = ["# {} on hkuhpc".format(level), "",
             "Keywords: `! {}`".format(spec["keywords"]), "",
             "Blocks: `{}`".format(spec["blocks"].replace("\n", " ")), "",
             "Route: {} (ORCA has no analytic gradient at this level: Opt NumGrad + NumFreq).".format(spec["route"]), "",
             "{} ranks per job, {} jobs concurrently per node, %maxcore {} MB (75 % of {} GB / {} cores).".format(
                 ranks, concurrent, maxcore, s["mem_gb"], s["cpus_per_node"]), "",
             "Point-count estimate per job (two-sided numerical gradients, ~10 optimisation steps):", "",
             "| job | atoms | opt | NumFreq | total single points | days at 8 ranks |", "|---|---|---|---|---|---|"]
    for name, n, e in estimates:
        lines.append("| {} | {} | {} | {} | {} | {:.1f} |".format(name, n, e["opt"], e["numfreq"], e["total"], e["days_8ranks"]))
    lines += ["", "Wall time from the measured {:.0f} s per DLPNO-CCSD(T)/cc-pVTZ (TightPNO) point on a 10-atom".format(SECONDS_PER_POINT_8RANKS),
              "molecule at 8 ranks (local ORCA 6.0.1, 2026-09-17); the jobs run concurrently, so the",
              "bundle takes about the longest job. Set --time accordingly."]
    lines += ["", "Submit with `sbatch run.sbatch` from this directory; `results/<job>/` receives the full",
              "`.out` (never trimmed), `.hess`, `.xyz` of every normally terminated job; a failed job leaves",
              "`<job>.failed.out`. Re-submitting skips finished jobs. Copy `results/<job>/` back to",
              "`<molecule>/orca/{}/basinNN/job.*` and run `s0_thermo_msrrho.py --step reference --level {}`.".format(level, level)]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dict(dir=out_dir, inputs=[out_dir / (n + ".inp") for n in names], worker=out_dir / "worker.sh",
                sbatch=out_dir / "run.sbatch", readme=out_dir / "README.md", estimates=estimates,
                maxcore=maxcore, concurrent=concurrent)
