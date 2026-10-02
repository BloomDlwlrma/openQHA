# Installing openQHA on Tianhe — a handbook

For **TianheXY-CN**, **TianheXY-A** and **TianheXY-AI**. Everything here was either measured
on the machine or measured against conda-forge on 2026-09-08; anything that was not is
marked **[unverified]** so you know which is which before you spend an allocation.

If you want the machine itself — partitions, quotas, scheduler commands — read
[`tianhe_runbook.md`](tianhe_runbook.md). This file is only about getting the software in
place.

---

## 0. The short version

```bash
source ~/init_conda.sh
source /APP/u22/ai_x86/toolshs/setproxy.sh 172.16.31.200 3138

# XYFS02 -- shared by TianheXY-CN and TianheXY-A, so ONE checkout serves deimos and ai.
# TianheXY-AI (h100x) is on XYAIFS00, a different filesystem: a separate copy, see
# docs/tianhe_runbook.md section 0b and hpc/tools/xfer_tianhe_ai.sh.
cd /XYFS02/HDD_POOL/<acct>/<user>/<you>/openQHA-main
bash install_dependency.sh --tianhe-cuda --both     # builds openqha AND openqha-gpu
```

Then, per job — never both at once:

```bash
OPENQHA_ROLE=cpu source hpc/env/tianhe.sh    # branch A, QHA collection
OPENQHA_ROLE=gpu source hpc/env/tianhe.sh    # branch B trajectories, branch C training
```

If that works, you are done. The rest of this file is **why**, and what to do when it
does not.

---

## 1. Five facts that decide whether an install works here

Every one of these cost a real failure to establish. None is optional.

| # | Fact | What it costs to ignore |
|---|---|---|
| 1 | **A login node has no DNS.** Nothing resolves until `setproxy.sh` runs | 40 retry lines and a `CondaHTTPError` blaming a mirror that was never down |
| 1b | **A COMPUTE node has no outbound network at all** — not even to the proxy (measured 2026-09-12 on an45) | `Failed to connect to 172.16.31.200 port 3138`, i.e. the proxy is *known* and unreachable. Compare row 1, where the proxy is unknown and the message is `Failed to resolve`. **Read which of the two it is**: one means run `setproxy.sh`, the other means you are on the wrong kind of node. Install from a login node, or `mamba install --offline` from the package cache |
| 2 | **Channels go through the TUNA mirror**, configured in your `~/.condarc`. Do not rewrite it | — (it is already correct; changing it is the mistake) |
| 3 | **Use `mamba`, not `conda`** | 1 h 37 min at 100% CPU and still going, versus 5 min 30 s |
| 4 | **A login node has no GPU driver**, so `__cuda` is absent and every `*cuda*` build is unsatisfiable | `pytorch … is not installable because it requires __cuda` |
| 5 | **Two environments, split along the BLAS** | An unsolvable environment, or a slow CREST, depending on which way you merge them |

### 1.1 The proxy (fact 1)

```bash
source /APP/u22/ai_x86/toolshs/setproxy.sh 172.16.31.200 3138
curl -sI --max-time 15 \
  https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/noarch/repodata.json | head -1
```

Expect `HTTP/1.1 200`. If it hangs, stop — everything after will hang too, and a hung job
is charged for its whole walltime.

The failure this prevents looked like this, and it is worth being able to read:

```
CondaHTTPError: HTTP 000 CONNECTION FAILED for url
<https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/linux-64/repodata.json>
Failed to resolve 'mirrors.tuna.tsinghua.edu.cn' ([Errno -3] Temporary failure in name resolution)
```

**Read the second line, not the first.** "Failed to resolve" is DNS. It is not a refused
connection, not a 404, not a stale mirror. TUNA was fine; nothing could resolve at all.

Host and port are recorded in [`../configs/cluster_tianhe.yaml`](../configs/cluster_tianhe.yaml)
and are **[unverified]** in the sense that they came from the site on 2026-08-31 and have
not been re-confirmed since. `TIANHE_PROXY_HOST` / `TIANHE_PROXY_PORT` override them.

### 1.2 The mirror (fact 2)

The same `~/.condarc` is in place on all three clusters:

```yaml
channels: [defaults]
default_channels: [<TUNA>/anaconda/pkgs/{main,r,msys2}]
custom_channels:  {conda-forge: <TUNA>/anaconda/cloud, pytorch: <TUNA>/anaconda/cloud}
```

The environment files ask for `conda-forge` + `nodefaults`, so `conda-forge` resolves
through `custom_channels` to `<TUNA>/anaconda/cloud/conda-forge` — the only channel a
solve here touches. **This is correct. Do not change it.**

Measured 2026-09-08: that path returns HTTP 200 and its `linux-64/repodata.json` (443 MB)
was last modified the same day. `pypi.tuna.tsinghua.edu.cn` serves the pip packages too.

If you build by hand, `--override-channels -c conda-forge` is the command-line equivalent
of `nodefaults`. Without it your `channels: [defaults]` is merged in as well.

### 1.3 The solver (fact 3)

Measured on `ln302%TianheXY-AI`, 2026-09-08:

| solver | result on `environment-tianhe-gpu.yml` |
|---|---|
| classic | **1 h 37 min at 100% CPU, 3.5 GB RSS, still running** |
| libmamba | **5 min 30 s**, including the repodata download |

That is not "slower". A classic solve is also *silent* — a spinner and nothing else for
over an hour — which is indistinguishable from a hang, and it burns a core on a shared
login node the whole time.

`miniforge3` always ships `mamba`, and your own `init_conda.sh` already exports
`MAMBA_EXE`. Use it. `install_dependency.sh` now prefers it automatically and **refuses
to start a classic solve** rather than going quiet for ninety minutes.

> A trap worth knowing: a banner reading `solver : libmamba` did **not** mean libmamba was
> used. The installer probed `conda create --help` for `--solver` while running
> `conda env create` — different parsers. The check passed, the flag did nothing, and the
> classic solver ran anyway. Fixed, but if you are driving conda by hand, prefer `mamba`
> over trusting a flag.

### 1.4 `__cuda` (fact 4)

```
pytorch =2.5.1 cuda120* is not installable because it requires
└─ __cuda =* *, which is missing on the system.
```

conda-forge's CUDA builds depend on the `__cuda` **virtual package**, which conda and
mamba synthesise from the NVIDIA driver they can see. A login node has no card and no
driver, so `__cuda` is absent and every `*cuda*` build becomes unsatisfiable.

```bash
export CONDA_OVERRIDE_CUDA=12.3
```

This affects **solving only** — nothing is faked at runtime. It is honest here because the
compute nodes this environment runs on do have a driver (550.54.15, a CUDA 12.4 driver;
minor-version compatibility covers a 12.3 runtime). `install_dependency.sh` sets it
automatically, reading the value from the `cuda-version=` pin so the two cannot drift.

> Do not "verify" that this is unnecessary on a workstation that has a driver — the test
> is self-confirming. Check `conda info | grep -A6 "virtual packages"` and confirm
> `__cuda` is genuinely absent first. A truncated read of exactly that output is what made
> this look unnecessary once.

### 1.5 Two environments (fact 5)

| | `openqha` | `openqha-gpu` |
|---|---|---|
| BLAS | OpenBLAS, **OpenMP build**, pinned; `nomkl` | MKL (the solver's choice) |
| torch | CPU, **2.12.1** | CUDA 12.3, 2.5.1 |
| numpy | **1.26.4** | 1.26.4 |
| CREST + xtb | yes | **no** |
| OpenMM stack | no | **yes** |
| runs | branch A, QHA collection | branch B trajectories, branch C training |

Three requirements, no common solution:

* conda-forge's `crest` links the **pthreads** OpenBLAS while CREST is OpenMP-parallel.
  Measured 2026-09-04 (`dsgdb9nsd_000018`, `-T 4`): pthreads 38.0 s and 4164 warning
  lines, OpenMP 18.5 s and none. Hence the pins.
* A CUDA build of pytorch **depends on MKL**, so `nomkl` and a CUDA torch cannot both hold.
* Measured 2026-09-07: `openmmtools` + pins solves; `openmmtools` + CUDA torch solves;
  `openmmtools` + CUDA torch + `cuda-version=12.3` + pins **fails**. Every pair, never the
  triple.

So the pins live where CREST and the quasi-harmonic diagonalisation are, and the GPU half
takes MKL — where neither of those runs. **The diagonalisation, the only step whose
numbers a BLAS could move, always executes in the collection pass, in `openqha`, against
OpenBLAS.**

**Only one cluster on your account?** Build both on it with `--both`; branch A then runs on
the CPU cores of a GPU allocation.

> **Never `conda activate openqha` and then `conda activate openqha-gpu`.** Conda *stacks*
> rather than replaces, so both prefixes stay on the loader's search path — one carrying
> OpenBLAS, one MKL — and which you get is decided by link order, not by the name you
> typed. It imports cleanly and returns numbers from a library you did not choose.
> `hpc/env/tianhe.sh` deactivates down to base first and reports the BLAS it ended up with
> into every job log.
>
> Products crossing between the two environments must go through parquet/HDF5/xyz —
> **never pickle**. The two now agree on numpy 1.26.4, but they differ in torch and in the
> BLAS underneath, and a pickle carries whatever the writing side happened to have.

---

## 2. Route A — the installer

```bash
source ~/init_conda.sh
cd <repo>

bash install_dependency.sh --tianhe        # TianheXY-CN  -> openqha
bash install_dependency.sh --tianhe-a      # TianheXY-A   -> openqha-gpu
bash install_dependency.sh --tianhe-cuda   # TianheXY-AI  -> openqha-gpu
bash install_dependency.sh --tianhe-cuda --both   # both, on one cluster
```

It handles facts 1, 3 and 4 for you: sources the proxy, probes the mirror your condarc
names and stops in 15 seconds if it cannot reach it, prefers mamba, refuses a classic
solve, and sets `CONDA_OVERRIDE_CUDA` from the yml.

Useful overrides:

| variable | effect |
|---|---|
| `TIANHE_PROXY_HOST` / `_PORT` | site proxy moved |
| `OPENQHA_PIP_INDEX=<url>` | different PyPI index; `=` (empty) means pypi.org |
| `OPENQHA_SOLVE_WITH=conda` | use conda + libmamba instead of mamba |
| `OPENQHA_SKIP_NET_CHECK=1` | skip the 15-second probe |
| `OPENQHA_FORCE_ANACONDA=1` | bypass TUNA for one run — **only** for `PackagesNotFoundError`, see §5 |
| `OPENQHA_ALLOW_CLASSIC=1` | permit a classic solve (you do not want this) |

### On a compute node instead

Solving is one core and several GB of RSS for minutes; login nodes are shared and often
cgroup-limited, so the same solve is slower there and can be killed.
[`../hpc/slurm/install_env_tianhe.slurm`](../hpc/slurm/install_env_tianhe.slurm) moves it:

```bash
ROLE=both yhbatch -p h100x --gpus=1 -t 02:00:00 hpc/slurm/install_env_tianhe.slurm
```

**[unverified, and it is the whole ballgame]** — whether TianheXY-AI compute nodes have
outbound network at all. Many sites give login nodes a proxy and compute nodes nothing.
The job probes and exits in ~15 seconds rather than burning the allocation, so submitting
it once with a short walltime is a cheap way to find out.

---

## 3. Route B — by hand, with exact versions

Use this when you want to watch each step, or when route A fails and you are isolating
which package is the problem. Versions are what libmamba actually resolved on 2026-09-08
against conda-forge, linux-64.

```bash
source ~/init_conda.sh
source /APP/u22/ai_x86/toolshs/setproxy.sh 172.16.31.200 3138

M="mamba install -y --override-channels -c conda-forge"
```

### 3.1 `openqha-gpu`

**The two CUDA lines belong to this environment and to nothing else.** Section 3.2 is the
CPU environment; it has no CUDA anything, and the deimos jobs that use it never load a
CUDA module (`examples/chain_body.sh` loads it only when the partition is a GPU one).

```bash
module load CUDA/12.3
export CONDA_OVERRIDE_CUDA=12.3

mamba create -y -n openqha-gpu --override-channels -c conda-forge python=3.11.16
conda activate openqha-gpu
```

```bash
$M cuda-version=12.3
$M numpy=1.26.4
$M scipy=1.13.1
$M pyyaml=6.0.3
$M pandas=2.3.3
$M pyarrow=18.1.0
$M h5py=3.16.0
$M ase=3.29.0
$M rdkit=2025.09.5
$M "pytorch=2.5.1=cuda120*"
$M openmm=8.2.0
$M "openmm-torch=1.5=cuda120*"
$M openmmtools=0.25.0
$M mdanalysis=2.10.0
$M pip=26.2.1

pip install pymsym==0.3.5 parsl==2026.9.7
# the mace FORK and the training package, editable, from the two checkouts beside this
# repository (mace/ and openQHA-Hessian/, carried by the transfer tool) -- run it last
bash ../openQHA-Hessian/install.sh
python -c "import mace; print(mace.__version__)"                 # 0.3.16+openqha
```

Pulled in as dependencies, for the record: `libtorch 2.5.1=cuda120_h6f417b9_303`,
`cudnn 9.10.2.21`, `libblas 3.9.0=37_h5875eb1_mkl`, `mkl 2024.2.2`, `llvm-openmp 23.1.0`.

> **The mace fork.** The fork is `BloomDlwlrma/mace` (branch
> `openqha-hessian`, base tag `base-v0.3.16` = upstream v0.3.16 minus three bundled model
> binaries), and the training side is the `openQHA-Hessian` package. A non-editable fork
> is enough for eval; TRAINING needs the **editable** pair: the fork and the
> package, from the two checkouts. Both travel with the repository --
> `hpc/tools/xfer_tianhe_ai.sh push-repo` carries `mace/` and `openQHA-Hessian/` beside
> `openQHA/`, `.git` included -- and `install_env_tianhe.slurm` section 9 runs
> `openQHA-Hessian/install.sh` (local-path mode) inside every environment it manages.
> Run it last -- the environments install no mace of their own (2026-10-02 ruling): the
> fork and the package arrive only through `install.sh`, from the sibling checkouts, and
> the editable pair is what training
> needs. `openqha.potentials.engine.provenance()` records the fork's commit as `mace_fork_commit`;
> a non-editable install (or a wheel) answers `unknown` and `05_train` refuses it.
> `scripts/tooling/s0_check_weights.py` prints the two lines.
>
> **Offline, stated once.** The conda/pip stack is installed where there is network -- a
> workstation, or the Tianhe login side behind the site proxy. The mace fork and the
> training package never need it: both travel by the transfer tool (the carried `mace/`
> and `openQHA-Hessian/` checkouts) and `install.sh` installs them from disk, offline.
>
> The branch carries the fork's commits on top of the base tag: the version bump, **A** the
> per-structure Hessian label (`--hessian_key`), **B** the external-loss hook (`--loss
> external --loss_module`), **C** multihead fine-tuning with that hook and the `evaluate`
> changes, **D** (2026-09-23) `--hessian_mode_weighting` and `--hessian_probe modes`
> removed -- there is one target -- and `--valid_probes_key` added: the
> per-structure fixed probe set the dataset draws and the loss reads at evaluation,
> and the probe default (tip `1110ffb`). There is no pinned sha to
> keep in step: `engine.provenance()` reads the checkout's own commit and `05_train`
> refuses a dirty or unknown one, so the only thing that must be true is that the checkout
> is committed.

### 3.2 `openqha`

```bash
mamba create -y -n openqha --override-channels -c conda-forge python=3.11.16
conda activate openqha
```

```bash
$M nomkl=1.0
$M "libopenblas=0.3.34=openmp*"
$M "libblas=3.11.0=*openblas"
$M "liblapack=3.11.0=*openblas"
$M numpy=1.26.4
$M scipy=1.17.1
$M pyyaml=6.0.3
$M pandas=3.0.5
$M pyarrow=25.0.0
$M matplotlib=3.11.1
$M h5py=3.16.0
$M ase=3.29.0
$M rdkit=2026.03.6
$M "pytorch=2.12.1=cpu_generic*"
$M crest=3.0.2
$M xtb=6.7.1
$M pip=26.2.1

pip install pymsym==0.3.5 parsl==2026.9.7
# the mace FORK and the training package, editable, from the two checkouts beside this
# repository (mace/ and openQHA-Hessian/, carried by the transfer tool) -- run it last
bash ../openQHA-Hessian/install.sh
python -c "import mace; print(mace.__version__)"                 # 0.3.16+openqha
```

> ### ⚠ OPEN: this environment produces a potential that returns `NaN` for every structure
>
> **Measured on TianheXY-CN, three runs, 2026-09-10.** `MACE-OFF23_medium` returns a
> non-finite energy on **every** call — 1628 of 1628, then 1719 of 1719, then again after
> the environment was rebuilt. The same code and the same acetone geometry return
> **−5259.489825324396 eV** on this project's workstation.
>
> **What has been ruled out, by measurement rather than by argument:**
>
> | | ruled out because |
> |---|---|
> | the weight file | same size (18 350 596 B), same 77 tensors, same 2 265 399 parameters, **every one finite** |
> | numpy | was 2.4.6, now **1.26.4** — same as the workstation. Still `NaN` |
> | pytorch | was 2.13.0, now **2.12.1 `cpu_generic`** — same as the workstation. Still `NaN` |
> | e3nn, mace-torch | 0.4.4 / 0.3.16 on both |
> | matscipy (the neighbour-list backend) | 1.2.0 on both |
> | BLAS | `nomkl` + `libopenblas 0.3.34 openmp*` + `libblas *_openblas` on both |
> | OpenMP runtime | `_openmp_mutex 8_kmp_llvm` + `llvm-openmp 23.1` on both |
>
> The pins above are still the right ones — they now match a machine that works, and the
> same downgrade was measured on 2026-09-05 to change **no number at all**
> (`s0_B_stack_fingerprint.py`: energy, forces, every Hessian frequency, T·S on 2000
> frames and every quasi-harmonic frequency, all `0.000e+00`). **But they are not the
> fix, and this box will not claim they are.**
>
> **Where to look next**, in the order the probe checks them:
>
> ```bash
> python scripts/tooling/s0_probe_potential.py
> ```
>
> Its section 5 starts *below* every library: it compares a `torch` matmul against numpy
> on the same matrices, and prints the CPU model. With every package version matched, what
> remains underneath is the kernel **OpenBLAS selects for this CPU at runtime**. If that
> check fails, no version pin can help and the things to try are
> `OPENBLAS_CORETYPE=Haswell`, `OPENBLAS_NUM_THREADS=1`, or a `pthreads` OpenBLAS build.
> If it passes, the section then isolates the neighbour list, e3nn's primitives with no
> MACE involved, and the model's own layers.

### 3.3 Three notes on these lists

**Only three specs keep a build selector, because the version alone does not say what you
want.** `pytorch=…=cuda120*` / `=cpu_generic*` separates the CUDA build from the CPU one;
`openmm-torch=1.5=cuda120*` likewise — a `cpu` build there leaves the GPU environment
silently useless for the one thing it exists for; `libopenblas=0.3.34=openmp*` separates
OpenMP from pthreads, which is the 4164-warning, 38.0 s vs 18.5 s difference.

**Because every version is pinned, installing one at a time is safe.** Normally each
`conda install` re-solves and can quietly downgrade something installed earlier; with exact
pins a later step either succeeds or reports a conflict.

**Do not run `pip install -r requirements.txt` in `openqha-gpu`.** That file lists
`torch>=2.0`, and pip may replace the conda CUDA build with a PyPI wheel. The three
packages above are the only ones with no conda-forge package. In `openqha` the full
requirements file is fine and is what adds the notebook stack, `mdtraj` and `MDAnalysis` —
which is why `conda list` there shows them as `pypi_0`.

> The `mace-torch` pins are gone: the environment and requirements files install no
> mace at all (2026-10-02 ruling) -- the fork arrives through `install.sh` from the
> sibling checkout -- so nothing here installs the PyPI wheel. For bit-identical
> reproduction, pin `torch==` and stay on one mace fork commit -- every product record
> carries `engine/torch_version` and `engine/mace_fork_commit`.

---

## 4. Verify

```bash
conda activate openqha-gpu
conda list | grep -E "openmm-torch|^pytorch|^libblas|cuda-version"
python -c "import torch, openmm, openmmtorch, openmmtools, MDAnalysis as m; \
           print(torch.__version__, torch.version.cuda, openmm.__version__, m.__version__)"
```

`openmm-torch` and `pytorch` must both read `cuda120*`. A `cpu*` build there is the failure
this environment exists to prevent, and it will not announce itself at run time.

```bash
conda activate openqha
python check_dependency.py            # ends with: branch A READY / NOT READY
crest --version && xtb --version
```

`branch B is NOT ready` **in `openqha` is expected** — openmm is deliberately not in the
CPU environment. Check branch B in `openqha-gpu`.

Which BLAS did you actually get:

```bash
ls "$CONDA_PREFIX"/conda-meta/libblas-*.json "$CONDA_PREFIX"/conda-meta/libopenblas-*.json 2>/dev/null
```

`libblas-…_openblas` + `libopenblas-…-openmp_…` in `openqha`; `libblas-…_mkl` in
`openqha-gpu`. Sourcing `hpc/env/tianhe.sh` reports the same thing into the job log.

> **Do not read this from `numpy.show_config()` or from the `.so` symlink.** Under
> conda-forge numpy links the `libblas` metapackage, so `show_config()` reports
> `"name": "blas"` for *either* provider; and `libblas.so.3` resolves to
> `libopenblasp-*.so` even in an OpenMP-pinned environment — the `p` is conda-forge's file
> naming, not "pthreads". Both were checked on 2026-09-08 and both give the wrong answer.

---

## 5. When it fails

| What you see | Cause | Fix |
|---|---|---|
| `Failed to resolve …` / `HTTP 000` / a hang | No proxy (fact 1) | `source …/setproxy.sh 172.16.31.200 3138` |
| 40 × `Retrying … notices.json` | Channel notices over a dead link | Cosmetic; gone once the proxy is set. `CONDA_NUMBER_CHANNEL_NOTICES=0` silences it |
| `Solving environment:` for ≥20 min | Classic solver (fact 3) | Use `mamba`. Check with `ps -o pid,etime,pcpu,rss,comm -u "$USER" \| grep -iE 'conda\|mamba'` — ~100% CPU means solving, ~0% means blocked |
| `__cuda =* *, which is missing` | No GPU driver (fact 4) | `export CONDA_OVERRIDE_CUDA=12.3` |
| `PackagesNotFoundError` / unsatisfiable spec | **Content**, not network — TUNA may lack that build | Drop the build selector, keep the version. Still stuck: `OPENQHA_FORCE_ANACONDA=1` for one run |
| `EnvironmentNameNotFound: openqha-gpu` | The solve never completed; no environment was made | Re-run; nothing to clean up |
| `branch B is NOT ready` in `openqha` | Expected — openmm is not in the CPU environment | Use `openqha-gpu` |
| No `crest` on `PATH` | You are in the GPU environment | `OPENQHA_ROLE=cpu source hpc/env/tianhe.sh` |
| A channel outside TUNA in the printed list | `$CONDA_PREFIX/.condarc` in the miniforge3 base — it sits *above* `~/.condarc` and nothing shadows it | `conda config --show-sources` |

**Distinguish a network failure from a content failure before reaching for
`OPENQHA_FORCE_ANACONDA`.** It bypasses the mirror; it does nothing for a missing proxy,
and using it there only changes which mirror gets blamed.

---

## 6. Still unverified

1. **No openQHA job has ever been submitted to Tianhe.** The install path is measured; the
   science on it is not.
2. **Whether TUNA carries every pin this project needs.** The versions in §3 were resolved
   against `conda.anaconda.org`. TUNA is a full conda-forge mirror updated the same day, so
   they should be present — "should" is not "measured". A gap surfaces as
   `PackagesNotFoundError` in about a minute, not as a long grind.
3. **Whether compute nodes have outbound network** (§2).
4. **Whether `$HOME` is the same filesystem on every cluster.** `conda env list` shows one
   environment under both `/HOME/…` and `/XYAIFS00/HOME/…`, while `init_conda.sh` sets
   `MAMBA_ROOT_PREFIX=/XYFS01/HOME/…` — three mount names, one logical path. If XYFS01 and
   XYAIFS00 are different filesystems, an environment built on TianheXY-CN is **not** the
   one TianheXY-AI sees. The compute-node job prints the resolved path and `df`; read it
   before assuming.
5. **Branch A on GPU-cluster CPU cores has never been timed.** `--both` makes it possible;
   one card's package on TianheXY-AI is 12–14 cores, which is not the shape of
   TianheXY-CN's 64-core exclusive node.
6. **The proxy host and port** are the site's 2026-08-31 values and have not been
   re-confirmed.

---

## 7. Where the reasoning lives

| Question | File |
|---|---|
| The machine: partitions, quotas, scheduler | [`tianhe_runbook.md`](tianhe_runbook.md) |
| What to run and where results land | [`branchA_production.md`](branchA_production.md) |
| Why the GPU environment is what it is | [`../environment-tianhe-gpu.yml`](../environment-tianhe-gpu.yml) header |
| Why the CPU environment pins its BLAS | [`../environment-tianhe.yml`](../environment-tianhe.yml) header |
