# ORCA under Slurm — the `SLURM_TASKS_PER_NODE` force-terminate: primary sources

Research note, 2026-09-25. Evidence base for the draw300 labels failure on TianheXY-CN
(every frame of the `TAG=draw300` array died in ORCA "Startup" with
`[file orca_tools/qcmsg.cpp, line 394]: .... aborting the run`) and for the fix /
parallel-framework design in [`grilling-round-13-orca-parallel-framework.md`](grilling-round-13-orca-parallel-framework.md).

Produced by a research subagent; the load-bearing citations (ORCA manual §2.5, OpenMPI
`v4.1.8` `ras_slurm_component.c`) were spot-checked against the same sources by the main
session. Primary sources: the ORCA 6.1 manual (FACCTs), the OpenMPI `v4.1.8` source tree
on `open-mpi/ompi`, and `slurm.schedmd.com`. The orcaforum could not be reached from the
research environment; nothing below depends on it. Claim labels used inline:
**[doc]** = official documentation, **[src]** = read in program source,
**[ours]** = measured in this repository / on the workstation.

---

## 0. Executive summary

1. **Root cause confirmed, exactly as suspected.** In OpenMPI 4.1.x, the presence of
   **`SLURM_JOBID` alone** makes three ORTE components eligible: `ras/slurm` (priority 50),
   `plm/slurm` (priority 75) and, for daemons, `ess/slurm` (priority 50). `ras/slurm` then
   *requires both* `SLURM_NODELIST` **and** `SLURM_TASKS_PER_NODE`; with `SLURM_NODELIST`
   present and `SLURM_TASKS_PER_NODE` missing it prints exactly the help text seen in the
   ORCA output and returns `ORTE_ERR_NOT_FOUND`, which `ras_base_allocate.c` treats as
   fatal (`ORTE_ERROR_LOG` + `ORTE_FORCED_TERMINATE`).
2. **The narrow unset loop is the bug.** The worker removed `SLURM_TASKS_PER_NODE` (the
   pattern `SLURM_TASK…` prefix-matches it) but kept `SLURM_JOBID`, `SLURM_NODELIST`,
   `SLURM_JOB_CPUS_PER_NODE`, `SLURM_NNODES`, `SLURM_NODEID`. `SLURM_JOBID` is OpenMPI's
   "I am in Slurm" switch; the rest is a half-armed Slurm environment.
3. **Recommended fix: option (a), remove *all* `SLURM_*`/`PMI_*` from the ORCA child
   environment only** (narrowest choke point:
   `openqha/qm_interfaces/orca.py::subprocess_env()`), plus a binding off-switch
   (`--bind-to none` or `OMPI_MCA_hwloc_base_binding_policy=none`) as an ORCA-documented
   precaution. This disarms `ras/slurm`, `plm/slurm` and `ess/slurm` in one stroke.
4. **Option (b) (`OMPI_MCA_ras=^slurm`) is *not* sufficient alone** — with `SLURM_JOBID`
   still exported, `plm/slurm` (priority 75) would take over and launch the ORTE daemons
   **via `srun`**. If MCA exclusions are used at all, they must exclude `ras`, `plm`
   **and** `ess`.
5. OpenMPI's docs do **not** contain a "unset `SLURM_*` before mpirun" FAQ item in the
   reachable primary sources; the official documentation covers *component exclusion*
   instead. The unset remedy is derived from the component-selection logic in the source
   (primary) and matches the deterministic local tests reproduced on the workstation.

---

## 1. ORCA official guidance (FACCTs 6.1 manual)

### 1.1 The `%pal` block and how ORCA launches MPI

*Manual §2.5 "Parallel and Multi-Process Runs":*
<https://www.faccts.de/docs/orca/6.1/manual/contents/essentialelements/parallel.html>

- Parallelism is requested in the input: `! PAL4` or `%pal nprocs 4 end` (any positive
  integer via the block).
- The driver then launches the modules itself **[doc]**: "The parallelized modules of
  ORCA are started by the (serial) ORCA-Driver. If the driver finds `PAL4` or `%pal`
  `nprocs` `4` `end` (e.g.) in the input, it will start up the parallel modules instead
  of the serial ones." — the documented counterpart of the observed
  `Calling Command: mpirun -np 4 .../orca_startup_mpi ...`.
- Boxed warning **[doc]**: "**Do not start the ORCA driver with mpirun!**"
- *Troubleshooting §1.7.1* repeats it: "If ORCA runs for you in serial, but not in
  parallel, first make sure that you are not running the main ORCA executable with
  mpirun."
- *Troubleshooting §1.7.3* shows exactly the failure *shape* seen here
  (`ORCA finished by error termination in <ORCA MODULE>` / `Calling Command: mpirun -np 8 …`)
  and advises reading the `WARNINGS` block: "At this point, looking at the warnings
  issued by ORCA at startup can be very helpful."

### 1.2 Passing extra MPI options through ORCA (the documented hook)

Same page, item 7 of "Single Node Runs":

- `/mypath_orca_executables/orca MyMol.inp "--bind-to none"` — "It is possible to pass
  additional MPI-parameters to mpirun by adding these arguments to the ORCA call - all
  arguments enclosed in a single pair of quotes".
- Also documented: `"--prefix /my-openmpi-folder"`, `"-x LD_LIBRARY_PATH -x PATH"`
  (for multi-node env propagation), and ORCA honours the variable `RSH_COMMAND` for
  multi-node NumCalc.

### 1.3 CPU binding: default behaviour and off-switches

*Manual §2.5.2.1 Tip* (verbatim quotes):

> "mpirun automatically binds processes as of the start of the v1.8 series. For NumCalc
> this can result in all displacements being run on the same set of cores, leading to
> severe performance degradation. There are different workarounds for this:
> • You can switch off this behaviour by passing `bind-to none` as additional
> MPI-parameter to ORCA.
> • Alternatively you can disable the binding via environment setting:
> `OMPI_MCA_hwloc_base_binding_policy none`
> • The most efficient solution is to use a resource manager (such as SLURM, PBS or
> others) and make sure Open MPI was built to support it. These resource managers will
> make sure that each job will run on a different set of cores."

So: **ORCA does not bind ranks itself; its bundled OpenMPI does, and ORCA documents the
two off-switches above.** The third bullet describes a *managed* allocation where the RM
hands each launcher a distinct CPU set — which is **not** our pattern (16 mpiruns share
one allocation), so we own placement and should switch OMPI binding off
(cf. `grilling-round-13`, Q3).

### 1.4 Throughput: how many ranks per job are efficient?

The manual gives efficiency ceilings, not a jobs-per-node recipe **[doc]**:

> "The efficiency of the parallel modules is such that for RI-DFT perhaps up to 16
> processors are a good idea while for hybrid DFT and Hartree-Fock a few more processors
> are appropriate. Above this, the overhead becomes significant and the parallelization
> loses efficiency. Coupled-cluster calculations usually scale well up to at least 8
> processors but probably it is also worthwhile to try 16."

and, for numerical work:

> "For Numerical Frequencies or Gradient runs it makes sense to choose nprocs = 4 or 8
> times 6*Number of Atoms."

Related official tools for *not* running one wide job: the `%pal` block's
`nprocs_world`/`nprocs_group` split (e.g. `nprocs 32 … nprocs_group 4` = "8 displacements
simultaneously") and the shorthand groupings `!PAL16(4x4)`, `!PAL32(8x4)`, etc.

**Negative finding:** the 6.1 manual chapters most likely to contain it (Parallel and
Multi-Process Runs; Architecture; Troubleshooting; General Recommendations) and the
manual index were checked: **no explicit ORCA statement was found recommending "run
several independent ORCA jobs on one node instead of one wide job."** What is documented
is (i) the per-method efficiency ceilings above and (ii) the multi-process mode. If the
project wants an ORCA-endorsed throughput quote, it should come from the forum
(unreachable) or a FACCTs workshop — flagged as an open item (§7).

### 1.5 Batch systems / Slurm in ORCA docs

- No dedicated Slurm page; the only queueing-system statements found **[doc]**: single-node
  item "If you start ORCA within a queueing system, you also don't need to provide a
  nodefile. The queueing system will care for it", and the PBS wrapper example.
- **`ORCA_SKIP_CPU_BIND` is not documented anywhere reachable**: the manual index
  (<https://www.faccts.de/docs/orca/6.1/manual/genindex.html>) contains no
  environment-variable entries at all; the binding chapter offers only `--bind-to none`
  and `OMPI_MCA_hwloc_base_binding_policy none`. Treat `ORCA_SKIP_CPU_BIND` as
  unverified / not-an-ORCA-variable unless a forum source says otherwise.
- The `shared_openmpi4xx` builds are documented by example in §1.2.2
  (`orca_6_1_0_linux_x86-64_shared_openmpi416.run`); PATH and LD_LIBRARY_PATH handling is
  §1.2.3.

---

## 2. OpenMPI v4.1.x — the failure mechanism (source, `v4.1.8`)

### 2.1 What makes the Slurm path engage: `SLURM_JOBID` alone

`orte/mca/ras/slurm/ras_slurm_component.c::orte_ras_slurm_component_query` **[src]**:

```c
if (NULL == getenv("SLURM_JOBID") && !mca_ras_slurm_component.dyn_alloc_enabled) {
    /* disqualify ourselves */
    *priority = 0; *module = NULL; return ORTE_ERROR;
}
...
*priority = 50;
*module = (mca_base_module_t *) &orte_ras_slurm_module;
return ORTE_SUCCESS;
```

`orte/mca/ras/base/ras_base_select.c`: `mca_base_select("ras", …)` simply selects the
**highest-priority** component that returns a module, so with `SLURM_JOBID` set,
`ras/slurm` wins.

### 2.2 What `ras/slurm` requires, and the exact fatal return

`orte/mca/ras/slurm/ras_slurm_module.c::orte_ras_slurm_allocate` **[src]**:

```c
slurm_node_str = getenv("SLURM_NODELIST");
if (NULL == slurm_node_str) { ... "SLURM_NODELIST" ...; return ORTE_ERR_NOT_FOUND; }
...
} else {
    /* get the number of process slots we were assigned on each node */
    tasks_per_node = getenv("SLURM_TASKS_PER_NODE");
    if (NULL == tasks_per_node) {
        orte_show_help("help-ras-slurm.txt", "slurm-env-var-not-found", 1,
                       "SLURM_TASKS_PER_NODE");
        free(regexp);
        return ORTE_ERR_NOT_FOUND;
    }
    ...
    tmp = getenv("SLURM_CPUS_PER_TASK");
    if (NULL != tmp) { cpus_per_task = atoi(tmp); ... } else { cpus_per_task = 1; }
}
```

`orte/mca/ras/slurm/help-ras-slurm.txt`, entry `[slurm-env-var-not-found]` **[src]**:

```
While trying to determine what resources are available, the
SLURM resource allocator expects to find the following environment variables:
    SLURM_NODELIST
    SLURM_TASKS_PER_NODE
However, it was unable to find the following environment variable:
    %s
```

This is character-for-character the text in the Tianhe output, with `%s` =
`SLURM_TASKS_PER_NODE`.

### 2.3 Why it is fatal

`orte/mca/ras/base/ras_base_allocate.c::orte_ras_base_allocate` handles only three
non-fatal `allocate()` returns (`ORTE_ERR_ALLOCATION_PENDING`,
`ORTE_ERR_SYSTEM_WILL_BOOTSTRAP`, `ORTE_ERR_TAKE_NEXT_OPTION`). Any other error falls
through to **[src]**:

```c
ORTE_ERROR_LOG(rc);
OBJ_DESTRUCT(&nodes);
ORTE_FORCED_TERMINATE(ORTE_ERROR_DEFAULT_EXIT_CODE);
```

which produces the `ORTE_ERROR_LOG: Not found in file .../ras_base_allocate.c at line
193` and `FORCE-TERMINATE ... ras_base_allocate.c(195)` lines (line numbers in the 4.1.8
build; the sequence is this function).

Note the two sub-cases: with `SLURM_JOBID` set but `SLURM_NODELIST` missing, the module
fails in the *same* fatal way, just naming `SLURM_NODELIST`. The true trigger is
**`SLURM_JOBID` present + any required variable absent**. The narrow unset loop deleted
the wrong subset: `SLURM_NODELIST` present, `SLURM_TASKS_PER_NODE` gone.

### 2.4 The local fallback (why removing all `SLURM_*` works)

With no component chosen (`SLURM_JOBID` absent), `orte_ras_base_allocate` falls through
rankfile → dash-host → hostfile(s) → and finally "addlocal" **[src]**:

```c
/* if nothing was found by any of the above methods, then we have no
 * earthly idea what to do - so just add the local host */
node->slots = 1;
opal_list_append(&nodes, &node->super);
orte_hnp_is_allocated = true;
```

The job proceeds on the local node — consistent with both deterministic tests:
`env -u SLURM_JOBID SLURM_NODELIST=fake1 mpirun -np 2 hostname` works;
`SLURM_JOBID=1 SLURM_NODELIST=fake1 mpirun -np 2 hostname` dies in the ras stage before
anything else matters **[ours]**.

### 2.5 Other ORTE components that read `SLURM_*` — important for fix ranking

**`plm/slurm`** — `orte/mca/plm/slurm/plm_slurm_component.c::orte_plm_slurm_component_query`
**[src]**:

```c
/* Are we running under a SLURM job? */
if (NULL != getenv("SLURM_JOBID")) {
    *priority = 75; ... return ORTE_SUCCESS;
}
```

With priority 75 it wins over `plm/rsh`. Its `launch_daemons()` builds an **`srun`**
command to start the ORTE daemons (`"srun"`, `"--ntasks-per-node=1"`,
`"--kill-on-bad-exit"`, `"--mpi=none"`, `"--ntasks=…"`), i.e. inside a Slurm allocation
OpenMPI 4.1 launches its orteds as Slurm steps. It also sets `SLURM_CPU_BIND=none` for
the daemon environment (with a source comment about the Slurm-19 `--cpu_bind`→`--cpu-bind`
rename).

**`ess/slurm`** — `ess_slurm_component.c::orte_ess_slurm_component_query` requires
`ORTE_PROC_IS_DAEMON && SLURM_JOBID && orte_process_info.my_hnp_uri`; it beats `ess/env`,
whose query says it is "the env module, only used by daemons that are launched by ssh so
allow any enviro-specifc modules to override us" (priority 1). `ess_slurm_module.c::slurm_set_name()`
then reads `SLURM_NODEID` and `SLURMD_NODENAME` **[src]**:

```c
slurm_nodeid = atoi(getenv("SLURM_NODEID"));
...
if (NULL == (tmp = getenv("SLURMD_NODENAME"))) { ORTE_ERROR_LOG(ORTE_ERR_NOT_FOUND); return ORTE_ERR_NOT_FOUND; }
orte_process_info.nodename = strdup(tmp);
```

Consequence: **`SLURM_JOBID` is a three-way switch** (ras + plm + ess). Fixes that keep
`SLURM_JOBID` must account for all three; fixes that remove it need none.

### 2.6 Stopping the Slurm path deliberately: MCA component exclusion

**Implementation (v4.1.8), `opal/mca/base/mca_base_component_find.c::mca_base_component_parse_requested`**
**[src]**:

```c
static char negate[] = "^";
...
*include_mode = requested[0] != negate[0];          /* first char ^ => exclude mode */
requested += strspn(requested, negate);             /* skip leading ^ */
if (NULL != strstr(requested, negate)) {            /* ^ only at the very front */
    opal_show_help("help-mca-base.txt", "framework-param:too-many-negates", true, requested_orig);
    return OPAL_ERROR;
}
*requested_component_names = opal_argv_split(requested, ',');
```

So `ras ^slurm` excludes `slurm` from the `ras` framework; `slurm,^lsf` is an error.

**Documentation (Open MPI in-repo manual, `docs/mca.rst`, "Selecting which Open MPI
components are used at run time")** **[doc]**:

> "Each MCA framework has a top-level MCA parameter that helps guide which components are
> selected to be used at run-time. … It takes a comma-delimited list of component names,
> and may be optionally prefixed with `^`."
> `# Tell Open MPI to exclude the tcp and uct BTL components and implicitly include all the rest`
> `shell$ mpirun --mca btl ^tcp,uct ...`
> "Note that `^` can *only* be the prefix of the *entire* comma-delimited list because
> the inclusive and exclusive behavior are mutually exclusive."

Environment form: same doc — "Any environment variable named `OMPI_MCA_<param_name>` will
be used" → `OMPI_MCA_ras=^slurm`, `OMPI_MCA_plm=^slurm`, `OMPI_MCA_ess=^slurm`.

### 2.7 Oversubscription and binding knobs (4.1 names)

- `docs/mca.rst` migration table: 4.x-era `rmaps_base_oversubscribe` = "Nodes are allowed
  to be oversubscribed, even on a managed system, and overloading of processing
  elements"; current name is `mapby=…:oversubscribe` (PRTE). For 4.1.8, the 4.x spelling
  `--mca rmaps_base_oversubscribe 1` / `OMPI_MCA_rmaps_base_oversubscribe=1` applies.
- Managed-allocation default: the ras/slurm dynamic path sets `ORTE_MAPPING_NO_OVERSUBSCRIBE`
  explicitly ("default to no-oversubscribe-allowed for managed systems"). This is the
  reason a *correctly-shaped but 1-slot* Slurm view (`--ntasks=1`) would **still** fail
  `mpirun -np 4` once the first error is fixed. Relevant to option (c).
- Binding (ORCA's own documented off-switches, §1.3): `--bind-to none` appended through
  ORCA's quoted extra-args, or `OMPI_MCA_hwloc_base_binding_policy=none`.

### 2.8 Does OpenMPI document "unset SLURM_*" anywhere?

**Not found.** The reachable official docs (in-repo `docs/…`, man pages) document
*component selection/exclusion* and MCA parameters; no FAQ entry prescribing "unset
`SLURM_*` before calling mpirun inside an allocation" was found. Treat that remedy as
**derived from the source logic** (primary) plus the empirical tests, not as a documented
Open MPI recommendation. Open question, §7.

---

## 3. Slurm official documentation

### 3.1 Environment variables: batch script vs job steps

`sbatch.html` ("OUTPUT ENVIRONMENT VARIABLES"): **"The Slurm controller will set the
following variables in the environment of the batch script."** Relevant entries
(verbatim) **[doc]**:

- `SLURM_TASKS_PER_NODE` — "Number of tasks to be initiated on each node. Values are
  comma separated and in the same order as SLURM_JOB_NODELIST. If two or more consecutive
  nodes are to have the same task count, that count is followed by `(x#)` … e.g.
  `SLURM_TASKS_PER_NODE=2(x3),1`…"
- `SLURM_JOBID` — "The ID of the job allocation. See SLURM_JOB_ID. Included for backwards
  compatibility."
- `SLURM_NODELIST` — "List of nodes allocated to the job. See SLURM_JOB_NODELIST.
  Included for backwards compatibility."
- `SLURM_JOB_CPUS_PER_NODE` — "Count of CPUs available to the job on the nodes in the
  allocation, using the format `CPU_count[(xnumber_of_nodes)][,…]` …"
- `SLURM_CPUS_PER_TASK` — "Number of cpus requested per task. Only set if either the
  --cpus-per-task option or the --tres-per-task=cpu=# option is specified."
- `SLURM_NTASKS` — "Set to value of the --ntasks option, if specified. … NOTE: This is
  also an input variable for srun, so if set it will effectively set the --ntasks option
  for srun when called from the batch script." (relevant if `srun` is ever called from
  this script)

`sbatch --export`: **"Note that `SLURM_*` variables are always propagated."**

`srun.html` ("OUTPUT ENVIRONMENT VARIABLES") sets an analogous set **inside steps**, plus
step-scoped ones: `SLURM_STEP_TASKS_PER_NODE` ("Number of processes per node within the
step"), `SLURM_STEP_NUM_TASKS`, `SLURM_STEP_ID`, `SLURM_PROCID`, `SLURM_LOCALID`,
`SLURM_TASKS_PER_NODE`, `SLURMD_NODENAME`, etc. This is why the ORCA children (run from
the batch script, not via srun) inherit the *batch-level* variants.

### 3.2 sbatch semantics we rely on (short quotes)

- `--exclusive`: "the job is allocated all CPUs and GRES on all nodes in the allocation,
  but is only allocated as much memory as it requested. … **To request all the memory on
  a node, use `--mem=0`.**" Also "NOTE: This option is mutually exclusive with
  `--oversubscribe`."
- `--mem`: "**A memory size specification of zero is treated as a special case and grants
  the job access to all of the memory on each node.**" and "Memory requests will not be
  strictly enforced unless Slurm is configured to use an enforcement mechanism" (cgroup
  caveat).
- `-c/--cpus-per-task`: "…the controller knows that each task requires 3 processors on
  the same node…" (our 64 fits here).
- `-n/--ntasks`: "**sbatch does not launch tasks**, it requests an allocation of
  resources and submits a batch script. This option advises the Slurm controller that job
  steps run within the allocation will launch a maximum of number tasks…" (our
  `--ntasks=1` sets `SLURM_TASKS_PER_NODE=1` — harmless for Slurm, fatal for ORTE, §2).
- `--hint=nomultithread`: "Don't use extra threads with in-core multi-threading;
  restricts tasks to one thread per core. Only supported with the task/affinity plugin."
  (applies to *srun steps*; our 16 mpiruns are outside that machinery).
- `--overcommit` vs `--oversubscribe`: see also Slurm FAQ "Why does the srun --overcommit
  option not permit multiple jobs to run on nodes?".

### 3.3 The MPI guide (`mpi_guide.html`)

- Three modes **[doc]**: (1) "Slurm directly launches the tasks and performs
  initialization of communications through the PMI-1, PMI-2 or PMIx APIs"; (2) "Slurm
  creates a resource allocation for the job and then **mpirun launches tasks using
  Slurm's infrastructure (srun)**"; (3) "Slurm creates a resource allocation for the job
  and then **mpirun launches tasks using some mechanism other than Slurm**, such as SSH
  or RSH. These tasks are initiated outside of Slurm's monitoring or control… **The use
  of pam_slurm_adopt is strongly recommended.**"
- Open MPI section: "If OpenMPI is configured with `--with-pmi=` pointing to either
  Slurm's PMI-1 … or PMI-2 … OMPI jobs can then be launched directly using the srun
  command. This is the preferred mode of operation since accounting features and affinity
  done by Slurm will become available." Starting with OMPI 3.1, PMIx is natively
  supported (`srun --mpi=pmix` / `MpiDefault=pmix`). **The ORCA shared build's OpenMPI is
  not configured against this site's Slurm PMI, and ORCA's driver invokes its own
  `mpirun` regardless — so we are in mode 2/3, not mode 1.**

### 3.4 Many independent MPI programs per node inside one job

- **FAQ, "How can I run multiple jobs from within a single script?"** **[doc]**: "A Slurm
  job is just a resource allocation. You can execute **many job steps** within that
  allocation, either **in parallel or sequentially**. Some jobs actually launch thousands
  of job steps this way. The job steps will be allocated nodes that are not already
  allocated to other job steps." (<https://slurm.schedmd.com/faq.html>)
- **`srun` step visibility**: by default (since 20.11) steps are *exclusive*; `--exact` =
  "Allow a step access to only the resources requested for the step. … NOTE: Parallel
  steps will either be blocked or rejected until requested step resources are available
  **unless `--overlap` is specified**."; `--overlap` = "allows steps to share all
  resources (CPUs, memory, and GRES) with all other steps." The srun man page Example 9
  demonstrates exactly this: "`--overlap` allows both steps to start at the same time.
  The `--exclusive` flag makes the second step wait until the first has finished."
  Example 7 shows the parallel-steps shell pattern (`srun -n4 prog1 & srun -n3 prog2 & … wait`).
- **CPU binding for such a pattern**: `srun --cpu-bind=[quiet|verbose,]type`; "**If
  `--cpu-bind` is not used, the default binding mode will depend upon Slurm's
  configuration and the step's resource allocation**"; "Explicitly specified masks or
  bindings are only honored when the job step has been allocated every available CPU on
  the node." `--hint=nomultithread` is allocation-scoped. **But note**: none of this
  binding machinery applies to processes launched with plain `mpirun` — Slurm only binds
  *steps*.
- **Caveat for a hypothetical srun-based rewrite**: srun man page PERFORMANCE: "**Do not
  run srun or other Slurm client commands that send remote procedure calls to slurmctld
  from loops in shell scripts or other programs.**" (our `xargs -P16` would become 16
  concurrent step creations per node).

---

## 4. Synthesis I — root cause, confirmed with citations

| Step | Fact | Source |
|---|---|---|
| 1 | `--nodes=1 --exclusive --ntasks=1 --cpus-per-task=64 --mem=0` ⇒ batch env gets `SLURM_JOBID`, `SLURM_NODELIST`, `SLURM_TASKS_PER_NODE=1`, `SLURM_CPUS_PER_TASK=64`, `SLURM_JOB_CPUS_PER_NODE=64`, `SLURM_NNODES=1`, … | sbatch man, OUTPUT ENVIRONMENT VARIABLES |
| 2 | The unset loop `grep -E '^(PMI|SLURM_(CPU|TASK|NTASKS|NPROCS|STEP))'` matches any name whose prefix is `SLURM_TASK`, hence **removes `SLURM_TASKS_PER_NODE`**; but `SLURM_JOBID`, `SLURM_NODELIST`, `SLURM_JOB_CPUS_PER_NODE`, `SLURM_NNODES`, `SLURM_NODEID` survive | `hpc/slurm/hl_labels.slurm:63` + regex semantics |
| 3 | In the ORCA child, `SLURM_JOBID` present ⇒ `ras/slurm` query returns priority 50 and wins | `ras_slurm_component.c` |
| 4 | `ras/slurm` needs `SLURM_NODELIST` **and** `SLURM_TASKS_PER_NODE`; missing `TASKS_PER_NODE` ⇒ the exact help text and `ORTE_ERR_NOT_FOUND` | `ras_slurm_module.c` + `help-ras-slurm.txt` |
| 5 | `ORTE_ERR_NOT_FOUND` is not a tolerated return ⇒ `ORTE_ERROR_LOG` + `ORTE_FORCED_TERMINATE` | `ras_base_allocate.c` |
| 6 | Result: every `mpirun` inside ORCA's startup dies in seconds → ORCA aborts in "Startup" | observed = predicted |

**Why "TASKS_PER_NODE missing but NODELIST present" is exactly fatal:** the component is
armed by `SLURM_JOBID` alone; `NODELIST` gets it past the first check and *then* the
absent `TASKS_PER_NODE` triggers the not-found return. (Had `NODELIST` also been removed,
the same fatal pass would occur earlier naming `NODELIST`.)

**Latent second failure:** even if `TASKS_PER_NODE=1` had been kept, the managed
allocation would offer the ORCA mpirun **1 slot** and managed systems default to
no-oversubscription (source: "default to no-oversubscribe-allowed for managed systems";
MCA `rmaps_base_oversubscribe`), so `-np 4` would then fail on slots. So there is no
variant of "keep a 1-task Slurm view" that works.

---

## 5. Synthesis II — fix options, ranked

### (a) Unset **all** `SLURM_*` / `PMI_*` for the ORCA child only — **recommended**

Where: the narrowest choke point is `openqha/qm_interfaces/orca.py::subprocess_env()`
(it already builds the child env and prepends `S0_ORCA_PATH`/`S0_ORCA_LIB`); alternatively
and/or the worker shell. Keep the job script's own Slurm variables for bookkeeping.

- Source support: `SLURM_JOBID` is the sole selector for `ras/slurm` (§2.1), `plm/slurm`
  (§2.5) and `ess/slurm` (§2.5). With it absent, no Slurm component is selected; `ras`
  falls back to the local node (§2.4); `plm` falls back to rsh (local fork); `ess/env`
  (priority 1) is used; the local tests reproduce this end-to-end **[ours]**.
- Residual risks: (i) the ORCA child loses Slurm's view, so nothing is bound by the RM —
  the worker already pins with `taskset`; add the binding off-switch anyway (below) so
  OMPI's own hwloc binding cannot fight or extend beyond the taskset range; (ii) if OMPI
  ever reports "not enough slots" (it should not: np=4 ≤ 4 pinned/allowed cores), use
  `OMPI_MCA_rmaps_base_oversubscribe=1` (§2.7); (iii) ORCA's `-x`/`--prefix` behaviours
  are unaffected.
- Minimal experiment (offline, already run on the workstation):
  1. `env SLURM_JOBID=1 SLURM_NODELIST=fake1 mpirun -np 2 hostname` → fails (baseline);
  2. `env -u SLURM_JOBID SLURM_NODELIST=fake1 mpirun -np 2 hostname` → runs;
  3. `env SLURM_JOBID=1 SLURM_NODELIST=fake1 SLURM_TASKS_PER_NODE=1 mpirun -np 2 hostname`
     → gets past ras (then may hit the slot check if np > allowed cores), demonstrating
     the *first* error is exactly the missing var.

### (b) MCA exclusions (`OMPI_MCA_ras=^slurm`, …) — workable but must cover **all three** components

- Syntax support: v4.1.8 parser (§2.6) + Open MPI's own docs on `^` (§2.6).
- **If only `ras` is excluded**, `plm/slurm` (priority 75, armed by the same `SLURM_JOBID`)
  takes over the *daemon launch* and will call `srun` (§2.5) — a completely different
  launch path, per-ORCA-job srun steps, Slurm's step semantics and the "srun from loops"
  warning. Not what we want when the point is 16 independent mpiruns.
- If keeping `SLURM_JOBID` but excluding properly, use all three:
  `OMPI_MCA_ras=^slurm`, `OMPI_MCA_plm=^slurm`, `OMPI_MCA_ess=^slurm` (env form; `--mca`
  equivalents can also be passed through ORCA's quoted args, remembering the quotes).
- Residual risks: three knobs to keep consistent; future OMPI components reading
  `SLURM_*` are not covered; behaves identically to (a) only insofar as *nothing* else
  consumes the kept variables. Simplest mental model: (b) ≈ (a) with extra steps and more
  failure modes.

### (c) Declare a Slurm shape that matches 16×4 (`--ntasks=16 --cpus-per-task=4`) and let OpenMPI's Slurm code work

- What changes: header `--ntasks=16 --cpus-per-task=4` (keeping `--exclusive --mem=0`),
  *and* the unset loop must stop deleting `SLURM_TASKS_PER_NODE` (keep `SLURM_CPUS_PER_TASK`
  too, or let the ras module default it to 1 — either passes validation, §2.2). `ras/slurm`
  then sees 16 slots on 1 node; each `mpirun -np 4` fits; and `plm/slurm` will launch the
  ORTE daemons via `srun --ntasks-per-node=1 … --mpi=none` **once per ORCA job** (§2.5).
- Source support: ras/slurm slot parsing (`2(x3),1` syntax) and plm/slurm launch
  construction (§2.5); Slurm's cpus-per-task semantics (sbatch man).
- Residual risks: (i) 16 concurrent `srun` daemon launches per node at job start (and
  again at teardown) — see the srun PERFORMANCE warning; (ii) binding: `plm/slurm` sets
  `SLURM_CPU_BIND=none` **only for the daemons**; the 4 MPI ranks are still bound by
  hwloc defaults, and with 16 mpiruns each starting "at the beginning" of the topology
  they can pile onto the same cores — the failure mode ORCA's manual Tip warns about; a
  binding off-switch would still be needed; (iii) a hard dependency on `srun`/slurmd
  round-trips inside every ORCA job — the very thing (a) avoids; (iv) Slurm's accounting
  sees a 16-task allocation but no steps for the actual compute.

### (d) Auxiliary knobs (supporting, not primary)

- `OMPI_MCA_hwloc_base_binding_policy=none` and/or ORCA-quoted `"--bind-to none"` —
  directly ORCA-documented (§1.3). Use with (a) or (c).
- `OMPI_MCA_rmaps_base_oversubscribe=1` (4.x name; docs §2.7) — only if a "not enough
  slots" error ever appears after the main fix.
- `OMPI_MCA_ras_slurm_use_entire_allocation=1` — switches ras/slurm to read
  `SLURM_JOB_CPUS_PER_NODE` (=64 here) instead of `SLURM_TASKS_PER_NODE` (§2.2). **Do not
  ship this**: it would make each `mpirun -np 4` believe it owns 64 slots, silently
  abandoning the 16×4 intent. Useful only as a diagnostic to confirm which variable the
  module is reading.
- No `ORCA_*` binding/skip variables are documented (§1.5); do not rely on
  `ORCA_SKIP_CPU_BIND`.

---

## 6. Synthesis III — the 16×4 layout on a 64-core node

- **ORCA's own numbers** justify 4 ranks/job and discourage wide jobs for this workload
  class: RI-DFT "up to 16 processors … above this, the overhead becomes significant";
  hybrid DFT/HF "a few more"; CC "at least 8 … also … 16". The labels run
  `! wB97M-D3BJ def2-TZVPPD TightOpt Freq TightSCF`-shaped work (hybrid meta-GGA +
  analytic Hessian) — squarely in the "overhead grows past the teens" regime, so 16×4 is
  consistent with ORCA guidance; but note the manual does **not** phrase it as a
  jobs-per-node recipe (§1.4).
- **Slurm's view of our pattern**: 16 independent `mpirun`s inside one allocation is
  MPI-guide **mode 2/3**: Slurm neither launches, binds, nor accounts those processes.
  Slurm's *official* tool for "many independent programs on one node" is **steps**
  (`srun` with `--exact`/`--overlap`), per the FAQ and srun man; our pattern deliberately
  sidesteps steps (and thus sidesteps per-step binding and accounting). Defensible for
  ORCA (whose driver must own the `mpirun`), but it means **all** placement guarantees
  must come from our side (`taskset`), and all Slurm-aware-launcher leakage must be
  prevented (this incident).
- **Memory**: `%maxcore` is per rank — ORCA manual: "`MaxCore` is given in MB and is the
  amount of maximum memory to allocate PER PROCESSING CORE! … it is generally advised to
  set `MaxCore` such that `MaxCore * nprocs <= 0.75(Available Memory)`." The arithmetic
  64 ranks × 6000 MB = 384 GB = 0.75 × 512 GB matches that rule exactly. Slurm side:
  `--exclusive` alone does not give the node's memory — "To request all the memory on a
  node, use `--mem=0`"; keep both. Enforcement is cgroup-dependent.
- **If workers are ever converted to `srun` steps** (not recommended for ORCA as-is): use
  `--exact --overlap` to run 16 steps concurrently, `--cpu-bind`/`--hint` for placement,
  and expect one `srun` RPC to slurmctld per step creation; also remember `srun -n4 orca`
  would launch *four whole ORCA drivers*, not four ranks, unless OpenMPI is PMI-integrated
  with Slurm — ORCA's documented model is that the driver itself calls `mpirun` ("Do not
  start the ORCA driver with mpirun!"), so step-wrapping is a larger redesign.

---

## 7. Open questions / sources not reached

1. **ORCA forum** (<https://orcaforum.kofo.mpg.de/>): portal and search endpoints returned
   "Not Found" from the research environment; no forum posts (developer or community)
   were examined. Anything forum-based (e.g. jobs-per-node advice, `ORCA_SKIP_CPU_BIND`)
   is **unverified** here.
2. **OpenMPI v4.1 HTML docs**: `docs.open-mpi.org/en/v4.1.x/faq.html`,
   `…/man1/mpirun.1.html` repeatedly failed content extraction; `open-mpi.org/doc/v4.1/man1/mpirun.1.php`
   returned only a page stub; `man7/mca.7.php` 404. The `^` semantics are therefore cited
   from (i) the v4.1.8 **source** and (ii) the **current** in-repo manual `docs/mca.rst`.
3. **No OpenMPI FAQ/primary doc found that recommends unsetting `SLURM_*`** before
   `mpirun` inside an allocation. The remedy is source-derived (§2) and empirically
   confirmed by the workstation tests.
4. **`ess/slurm` selection in the non-`srun` launch path**: its query also needs
   `orte_process_info.my_hnp_uri` (set for mpirun-spawned daemons) and it outranks
   `ess/env`; which ess component the actual failing/passing runs select was not
   measured. Confirm locally with `--mca ess_base_verbose 10` (and `ras_base_verbose`,
   `plm_base_verbose`) if that should be on the record.
5. **Exact slot arithmetic when ras is excluded**: the local fallback sets `slots = 1`
   for the added host (§2.4), yet the workstation test runs `-np 2` fine — OMPI's slot
   handling for unmanaged local runs allows this on a multi-core host. If a worker ever
   sees "not enough slots", apply §5(d). Not fully traced in source.
6. **Slurm docs** do not discuss `SLURM_*` leaking into launchers either; the closest
   official statements are the `--export` propagation note and the MPI guide's mode-3
   caveats (pam_slurm_adopt).
7. **Why the 2026-09-19 smoke run passed** while the array failed is still a
   reconstruction (most plausibly: the smoke run went through the parsl in-allocation
   worker init, which already unset everything `^(PMI|SLURM)_`, rather than the xargs
   route's narrow loop). Unverified; the smoke job's captured environment was not
   available.

---

## 8. Answer to the research questions, in one line each

1. **ORCA**: driver starts modules via its own `mpirun`; never start the driver with
   mpirun; efficiency ~≤16 procs for DFT-type work; binding is OpenMPI's, switched off by
   ORCA-documented `--bind-to none` / `OMPI_MCA_hwloc_base_binding_policy none`; no
   Slurm-specific page, no documented `ORCA_*` variable for this, no explicit "many jobs
   per node" recommendation found.
2. **OpenMPI 4.1.x**: `SLURM_JOBID` arms `ras/slurm` (prio 50) — and `plm/slurm` (75) and
   `ess/slurm` (50); `ras/slurm` requires `SLURM_NODELIST` + `SLURM_TASKS_PER_NODE`;
   missing the latter → `slurm-env-var-not-found` help text + `ORTE_ERR_NOT_FOUND` →
   `ORTE_ERROR_LOG`/`ORTE_FORCED_TERMINATE`; supported stop: `^slurm` MCA exclusion (all
   three frameworks if that route is taken); unset-all is the source-faithful remedy but
   is not a documented FAQ item.
3. **Slurm**: `SLURM_TASKS_PER_NODE` **is** exported to the batch script (sbatch man);
   env vars are "always propagated" by `sbatch --export`; `--exclusive` needs `--mem=0`
   for full memory; `srun` steps are the official multi-program vehicle (`--exact`,
   `--overlap`, `--cpu-bind`, `--hint`); plain `mpirun`s sit outside Slurm's
   binding/accounting (MPI guide modes 2–3).
4. **Fix ranking**: (a) unset all `SLURM_*`/`PMI_*` in the ORCA child env — recommended,
   smallest blast radius; (b) MCA exclusions — only as `ras+plm+ess ^slurm`, otherwise
   `plm/slurm` will hijack the launch via `srun`; (c) 16×4 Slurm shape — supported but
   re-introduces srun-based daemon launch and binding pitfalls; (d)
   `rmaps_base_oversubscribe`, `hwloc_base_binding_policy=none`/`--bind-to none` as
   supporting controls; `ras_slurm_use_entire_allocation` diagnostic only. Layout 16×4 is
   consistent with ORCA's efficiency guidance; Slurm-wise it is a mode-2/3 pattern that
   Slurm will not bind or account, so prevention of Slurm-env leakage into the launcher is
   mandatory and memory should keep the `--exclusive + --mem=0` + per-rank `%maxcore`
   0.75 rule.
