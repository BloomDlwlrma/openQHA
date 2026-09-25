# Soft modes, Eckart projection and the imaginary-mode floor — primary sources

Research note, 2026-09-25. Evidence base for the imaginary-mode ruling (production = ONE
policy: `invert_below`, `ithr = -50.0 cm^-1`) and for the two handling principles adopted
the same day (the user's rulings, paraphrased verbatim):

> **D1 — sign flips on an unconverged geometry.** "使用做同方法的几何优化再算频率；否则数字
> 无意义" — optimise with the same method first and interpret the frequencies after (Gaussian's
> rule); plus ORCA's quality practice (tight SCF, central differences, fine grids, tightened
> geometry criteria, small increments).
>
> **D2 — unprojected rigid-body residue.** Always project (`ProjectTR`), enforce translation
> invariance (acoustic sum rule, `TransInvar`); never compute thermochemistry with projection
> off — with projection off, TR frequencies crossing the 1 cm^-1 cutoff make the entropy jump
> ("exhibits a finite jump").

Claim types: **[doc]** = quoted from the program's own documentation; **[src]** = read in the
program's source; **[ours]** = measured/found in this repository.

---

## 1. "The projection itself is wrong" — what that means mechanically

Eckart projection removes the 6 (5 for linear molecules) rigid-body vectors from the
mass-weighted Hessian: `K = P H_m P`, `P = I - v v^T`. After projection the rigid-body
subspace lies in the kernel **by construction** (eigenvalues at machine zero). ORCA says
exactly this **[doc, ORCA 6.0 manual 6.5]**:

> "The six frequencies … correspond to the rotations and translations of the molecule. They
> have been projected out of the Hessian before the calculation of the frequencies and thus,
> the zero values do not tell you anything about the quality of the Hessian that has been
> diagonalized."

So a projected mode with |omega| < 1 cm^-1 cannot be "rigid-body residue that was not
projected out". It is either **(i)** coordinate/instrument inconsistency — the Hessian not
computed at the projected geometry, or not converged tightly enough (VASP **[doc]**: "A
looser EDIFF introduces noise into the force constants that can manifest as spurious
low-frequency modes"; ORCA **[doc]**: "The numerical error in the frequencies may reach
50 cm^-1"); **(ii)** an incomplete rigid basis (near-linear molecules: rank < 6; near-symmetric
ones: nearly dependent rotation vectors — our `rigid_body_vectors` takes the rank from an SVD
and identifies rigid modes by overlap, never "the six smallest" **[ours]**); or **(iii)** a
genuinely ultra-soft vibrational direction, which cannot be distinguished from noise by a
single number. This repository's own defect 64 sits on the same boundary: finite differences
at delta = 0.01 A resolve only +-9 to 20 cm^-1 while the condemned modes were 9-24 cm^-1
**[ours, `crest_census.HESSIAN_MODE_DEFAULT` comment]**.

Note: the deepest sense of "the projection is wrong" is D2's — one must never *turn it off*:
with projection off, the frequencies of the TR modes are non-zero numbers that cross the
thermochemistry cutoff, and the entropy jumps (quote in §3).

## 2. Sign flips on an unconverged geometry — sources and the adopted handling

- **Gaussian [doc, `gaussian.com/freq/`]**: "This transformation is only valid at a stationary
  point. Thus, it is **meaningless** to compute frequencies at any geometry other than a
  stationary point for the method used for frequency determination." … "The recommended
  practice is to compute frequencies following a previous geometry optimization using the
  same method."
- **VASP [doc, wiki "How to handle imaginary phonon modes"]**: "An unrelaxed or poorly
  converged cell can produce spurious imaginary modes from Pulay stress or residual forces.
  If imaginary modes appear unexpectedly, **first tighten the ionic relaxation … and recompute
  the phonons before drawing conclusions**." … "Always verify that the structure is fully
  relaxed before computing phonons. Residual forces from a geometry optimisation that
  converged on NSW rather than EDIFFG are a frequent source of spurious imaginary modes." …
  "For the phonon run, EDIFF = 1E-8 and PREC = Accurate are the recommended minimum."
- **ORCA [doc, manual 6.5]**: "Significant negative frequencies indicate saddle points of the
  energy hypersurface and **prove that the optimization has not resulted in an energy
  minimum**." To reduce noise: "Your SCF is tightly converged. A convergence accuracy of at
  least 1e-7 Eh in the total energy and 1e-6 in the density is desirable." … central
  differences … "Possibly, the convergence criteria of the geometry optimization need to be
  tightened in order to get fully converged results." … "decrease the numerical increment to
  0.001 Bohr or so".
- **VASP magnitude classes [doc, same page]** — when, after a converged setup, a negative mode
  remains, it is classified by size: imaginary part **< 0.5 meV (about 4 cm^-1)** = acoustic
  mode / numerical noise, "can be ignored"; **> 5-20 meV (about 40-160 cm^-1)** = "Genuine
  soft mode → structural instability". ORCA's "error may reach 50 cm^-1" belongs to the same
  interval.
- **The policy-style alternative (what CREST/Pracht-Grimme do) [src]**: never silently, never
  ignored, but *semantics plus accounting* (`crest/src/entropy/thermocalc.f90`): `vibthr = 1.0`
  — a mode is kept only `if (abs(freq(i)) .gt. vibthr)`; then "invert imaginary modes":
  `if (vibs(i) .lt. 0 .and. vibs(i) .gt. ithr) vibs(i) = -vibs(i)`; below `ithr` the mode stays
  negative and is counted (`nimag`).

**Adopted (D1)**: the root-cause rule — same-method optimisation first, frequencies after;
the census/tighten layer keeps fmax = 1e-4 eV/A (about 2e-6 Eh/bohr; ORCA's default
"normal" convergence is TolMaxG = 3e-4 Eh/bohr, i.e. about 150x looser — ORCA 6.0 manual
7.26) and the analytic Hessian (defect 64). The policy layer (`invert_below`, `ithr = -50`)
handles what remains *at* the minimum; the "< 1 cm^-1" assertion handles broken projection.
The three numbers (-50, 25, 1) are not inventions — see §4. ORCA's own note that "floppy
structures with many possible rotations around single bonds and soft dihedral angle modes
are tricky" (manual 7.26, footnote 1) describes exactly the population that raises the
extra-optimisation-pass question (Q8).

## 3. Unprojected rigid-body residue — sources and the adopted handling

- **ORCA [doc, manual 6.5 & 7.27]**: defaults `ProjectTR True` ("project out translation and
  rotation degrees of freedom in frequency calculation and thermochemistry analysis?") and
  `TransInvar True` ("enforce translation invariance while calculating the Hessian?" — the
  acoustic sum rule, "reduces the effect of noise from numerical integration coming from DFT
  or COSX"; not applied to Partial/Hybrid Hessians). Turning projection OFF is allowed only as
  a diagnostic: "the deviations represent a metric of the numerical error of the Hessian
  calculation" — and "it is **strongly discouraged** to turn off `PROJECTTR` when calculating
  thermochemical quantities (especially entropies and Gibbs free energies). This is because
  when the frequencies of translational and rotational modes exceed `CutOffFreq` (which is
  1 cm^-1 by default), their contributions to the partition function will be calculated using
  the formulas for vibrations. As a result, the calculated entropy is inaccurate … and in
  particular exhibits a **finite jump** when the (theoretically zero) frequencies of the
  translational and rotational modes cross `CutOffFreq`."
  The `%freq` block also carries `QuasiRRHO true` with `QRRHORefFreq 25` (the block's value;
  its comment notes the original paper used 100) and `CentralDiff true`, `Increment 0.005`.
- **VASP [doc]**: "The last three modes are the translational modes (**they are usually
  disregarded**)." And: three acoustic modes always appear near zero — "this is a numerical
  artefact of the finite-differences procedure, not a structural instability."
- **CREST [src]**: the thermo layer has no projection concept; near-zero modes never enter it,
  they are dropped by `vibthr = 1.0` before the thermodynamic sums (`thermocalc.f90`).

**Adopted (D2)**: project always (our spectra are Eckart-projected; `separation_gap_ratio`
reported, criterion 4 gate `> 1e8` **[ours]**), and keep the 1 cm^-1 floor as the enforcement
of "projection was on and clean": `thermo.VIBTHR_CM = 1.0` raises instead of dropping — a
projected spectrum cannot legitimately carry |nu| < 1 cm^-1, so the same number ORCA uses as
a drop is used here as an **assertion**. Whether production keeps the assertion or switches
to ORCA-style drop-with-recording is **OPEN** (Q6, 2026-09-25: decided to be discussed
further; both sides restated in the session).

## 4. Number provenance (verified in sources)

| number | first-party source |
|---|---|
| `ithr = -50 cm^-1` | CREST man page **[src]**: "-ithr _float_: Imaginary mode inversion cutoff. [_default_: **-50.0** cm^-1]" (`crest/docs/man/crest.adoc`); implemented in `thermocalc.f90`. ORCA's noise scale "may reach 50 cm^-1" **[doc]** and VASP's 40-160 cm^-1 class bracket the same interval. |
| `tau = 25 cm^-1` | CREST `-sthr` default 25 ("Vibrational/rotational entropy interpolation threshold (tau)", man page; `classes.f90: sthr = 25.0`, note "in xtb 50.0") **[src]**; ORCA `%freq` shows `QRRHORefFreq 25` **[doc]**. Our `MSRRHO_PRESETS["crest"] tau_cm = 25.0` copies this. |
| `1 cm^-1` | CREST `vibthr = 1.0` (drop) **[src]** = ORCA `CutOffFreq 1.0` "Threshold for frequencies to be considered in spectra, thermochemistry and printout" **[doc]**. Our `VIBTHR_CM = 1.0` (assertion) and `orca.optimise_and_hessian`'s `n_imaginary = (nu < -1.0).sum()` **[ours]** use the same floor. |
| real-world reference | Pracht & Grimme, *Chem. Sci.* **2021**, 12, 6551 (the msRRHO assembly; the CREST sources above are its implementation). |

## 5. openQHA mapping

| phenomenon | source practice | openQHA apparatus |
|---|---|---|
| sign flip, unconverged geometry | tighten first (Gaussian/VASP/ORCA); interpret only after | tighten fmax=1e-4 + analytic Hessian (defect 64) **[exists]**; non-converged frames: one extra pass, then flag (Q8, proposed) |
| converged minimum with modes in (-50, 0) | CREST: invert; below ithr: keep negative / count | production `invert_below(-50)` (Q1/Q2 settled); census floor to be added (Q3); reference level: re-optimise rather than invert (Q7 revised) |
| broken projection / |nu| < 1 | ORCA/CREST: drop (their inputs may be unprojected) | `VIBTHR_CM` assertion = enforcement that projection was on (Q6: keep) |
| TR invariance | ORCA `TransInvar` default | our analytic Hessians are exactly translation-invariant; rigid eigenvalues and the separation gap are reported; no extra ASR needed **[checked]** |
| auditability | ORCA defaults are documented | `ProjectTR`/`TransInvar`/`CutOffFreq` are NOT pinned anywhere in our ORCA inputs yet (Q10, proposed) |

## 6. Scope guard (do not over-apply D1)

D1 governs the **interpretation** of frequencies: the census minimum verdict, the
thermochemistry at basins, the reference-level basins. It must **not** be applied to the
Hessian-learning labels: the campaign labels frames "at the frame's fixed geometry — a single
point, **never an optimisation**" (`workflows/hessian_learning/README.md`), displaced frames
are deliberately off-stationary and their label is the raw Cartesian Hessian — that is what
the PHL fine-tune trains on (ADR 0005). Adding an optimisation before those frequency jobs
would destroy the campaign's design.

## 7. Literature practice on Hessian labels away from stationary points (Q9)

Read from the local collection `source-code/final-workflow-design/hessian-train/` (text
extracted 2026-09-25; PHL text complete, MACE-AD / PFT partially skipped by the extractor):

- **PHL (arXiv:2603.04523, 2026)** — dataset family: RTP (35,087 equilibrium reactants /
  products / transition states; "excluded from training", i.e. the benchmark), IRC (34,248
  geometries from 600 pathways), NMS (62,527 far-from-equilibrium geometries "designed to
  rigorously test the robustness of the extrapolation under large structural distortions").
  Reference Hessians at Gaussian16 omegaB97XD/6-31G(d), analytic. Off-equilibrium curvature
  labels are a deliberately built dataset class.
- **Rodriguez (JCTC 2025, "Does Hessian Data Improve the Performance of MLIPs")** — models
  "trained only to stable points on the potential energy surface (minima and transition
  states with atomic forces close to zero)", then evaluated on the IRC and NMS Hessian
  RMSEs; E-F-H training cuts the NMS Hessian RMSE 128 -> 38 kcal/mol/A^2 (tables 2-4).
  Stationary-only training, off-stationary held-out evaluation — the protocol ADR 0005
  follows.
- **ANI-1 (Smith, Isayev, Roitberg, Sci. Data 2017)** — Normal Mode Sampling introduced to
  generate off-equilibrium single points from an energy-minimised molecule (energy + forces
  only, no Hessian); the origin of this repository's displaced draw.
- **PFT (Koker et al.)** — phonon fine-tuning matches MLIP Hessians to DFT force constants
  from finite-displacement workflows about relaxed structures; "off-equilibrium remains an
  open research" question in that line.
- **Contrasting dataset conventions** (cited in PHL): Hessian-QM9 = equilibrium geometries
  only; HORM = 1.84 M Hessians for reactive geometries including transition states. Both are
  raw curvature at the given geometry; the convention is declared, not imposed.

**Guard (recommendation, Q9)**: frames are labelled at their own geometry — never optimised
first. D1 governs the interpretation of frequencies (census verdict, thermochemistry,
reference basins), not curvature labels; the training rows are stationary basin frames, the
off-stationary generators are held-out extrapolation tests.

## Sources

- ORCA 6.0 manual, 6.5 Vibrational Frequencies — https://www.faccts.de/docs/orca/6.0/manual/contents/typical/frequencies.html
- ORCA 6.0 manual, 7.27 Frequency calculations (the `%freq` block) — https://www.faccts.de/docs/orca/6.0/manual/contents/detailed/frequencies.html
- VASP wiki, IBRION (EDIFF <= 1E-6 for phonons) — https://www.vasp.at/wiki/index.php/IBRION
- VASP wiki, Phonons from finite differences (translation modes "usually disregarded") — https://www.vasp.at/wiki/index.php/Phonons_from_finite_differences
- VASP wiki, How to handle imaginary phonon modes (< 0.5 meV noise vs > 5-20 meV genuine; tighten first) — https://www.vasp.at/wiki/index.php/How_to_handle_imaginary_phonon_modes
- Gaussian, Freq keyword ("meaningless … other than a stationary point") — https://gaussian.com/freq/
- CREST sources (man page `-ithr`/`-sthr`; `src/entropy/thermocalc.f90`, `thermo.f90`) — https://github.com/crest-lab/crest
- Pracht & Grimme, Chem. Sci. 2021, 12, 6551 (msRRHO)
- Local collection (Q9): `source-code/final-workflow-design/hessian-train/` — PHL (arXiv:2603.04523), Rodriguez (JCTC 2025), ANI-1 (Sci. Data 2017), PFT, MACE-AD (ChemRxiv 2025); extracted text in `openQHA/_dbg/pdf_txt/` (throwaway)
