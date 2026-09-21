# ANI-1 normal-mode sampling as the only displaced-frame generator (design, 2026-09-21)

The user's ask: adopt ANI-1's NMS for the displaced frames -- temperature from ANI's table,
no RMS ceiling (only the 275 kcal/mol energy window, as ANI), the judge bins frames by RMS
displacement -- and remove every other basin-sampling scheme (the classical and quantum
harmonic draws) from the Hessian-learning workflow. This file holds the source, the
derivation in our coordinates, the code change, the cost on named molecules, and the
per-basin count. Nothing is implemented until the tickets are approved.

## 1. ANI-1's NMS, as published (Smith, Isayev, Roitberg, Sci. Data 4, 170193 (2017), "Normal mode sampling")

Pipeline: GDB-11 subset (1-8 heavy atoms, C/N/O) -> RDKit 3D + H -> MMFF94 pre-optimisation
-> wB97x/6-31G(d) optimisation in Gaussian 09 (the first stationary point reached) ->
normal modes and force constants at that geometry (UltraFine grid) -> N = S x K
conformations by NMS, single-point energies only -> conformations above 275 kcal/mol over
the molecule's lowest are stored separately ("HE") and not trained on.

| heavy atoms | molecules | max T of the draw | S (points per degree of freedom) | conformations |
|---|---|---|---|---|
| 1 | 3 | 2000 K | 500 | 10,800 |
| 2 | 13 | 1500 K | 450 | 51,360 |
| 3 | 20 | 1000 K | 425 | 151,200 |
| 4 | 61 | 600 K | 400 | 658,080 |
| 5 | 267 | 600 K | 200 | 1,823,040 |
| 6 | 1,406 | 600 K | 30 | 1,712,208 |
| 7 | 7,760 | 600 K | 20 | 7,329,384 |
| 8 | 47,932 | **450 K** | **5** | 12,951,737 |

K = 3 N_a - 6 (5 if linear). The draw (their eq. 1): for the N_f modes q_i with force
constants K_i, draw N_f uniform random numbers c_i such that sum c_i is in [0, 1], set the
harmonic energy of mode i equal to the c_i-scaled thermal energy, and solve

    R_i = +/- sqrt( 3 c_i N_a k_B T / K_i ),   sign Bernoulli(1/2),   x = x_0 + sum_i R_i q_i.

No displacement ceiling; the only filter is the 275 kcal/mol window applied afterwards on
the DFT energy. Rodriguez 2025 uses the same NMS on IRC intermediate structures (excluding
the reaction-coordinate mode) to build its extrapolation set.

## 2. The same draw in our coordinates

Our modes are the eigenvectors L_k of the Eckart-projected mass-weighted Hessian
K~ = P M^-1/2 H M^-1/2 P with eigenvalues lambda_k = omega_k^2 (eV A^-2 amu^-1), and a
mass-weighted normal coordinate q_k (amu^1/2 A) has harmonic energy (1/2) lambda_k q_k^2.
ANI's condition "harmonic energy of mode i = c_i x (3/2) N_a k_B T" therefore reads

    (1/2) lambda_k q_k^2 = (3/2) c_k N_a k_B T
    q_k = +/- sqrt( 3 c_k N_a k_B T / lambda_k ),                          (N1)
    dx  = M^-1/2 sum_k q_k L_k ,                                            (N2)

with c_k = s u_k / sum_j u_j, u_k ~ U(0,1), s ~ U(0,1) (so sum_k c_k = s is uniform on
[0, 1]). Consequences, exact for the harmonic energy E_h = sum_k (1/2) lambda_k q_k^2:

    E_h = (3/2) s N_a k_B T,   E[E_h] = (3/4) N_a k_B T,   E_h <= (3/2) N_a k_B T.        (N3)

So the draw's total harmonic energy is bounded (ANI's window is therefore rarely hit by the
harmonic part; it catches the anharmonic clashes when the displaced geometry is fed to the
real potential), and the mean is 3/4 N_a k_B T against the classical equipartition draw's
N_f k_B T / 2 -- for N_a = 10, N_f = 24: 7.5 k_B T against 12 k_B T at the same T. That is
why ANI's 450 K draw and our classical 298 K draw have almost the same mean amplitude
(section 4). Per mode the amplitude is not Boltzmann: c_k is a uniform partition, so the
per-mode energies are Dirichlet-like with a hard cap, and the tail of the RMS displacement
is bounded (propanal 450 K: max 0.27 A over 4000 draws, classical 298 K: 0.37 A).

## 3. The code change (ticket)

`openqha/thermochem/hessian.py::thermal_displacements` -- a third branch:

    elif distribution == "nms":                       # ANI-1 eq. 1 in mass-weighted coordinates (N1)
        u = rng.random(len(lam_v)); s = rng.random()
        c = s * u / u.sum()
        q = np.sqrt(3.0 * c * len(m) * KB_EV * temperature_K / lam_v) * rng.choice([-1.0, 1.0], size=len(lam_v))

inside the per-sample loop (the classical/quantum branches draw `q = rng.normal(0, sigma_q)`
there), with the Record gaining per frame `c_sum` (= s) and `harmonic_energy_kcal`
(= (3/2) s N_a k_B T, also computed from q as a check). `max_rms_displacement_A=None` is
already accepted by the loop (no rejection).

`openqha/data/frames.py`: `DISTRIBUTION = "nms"`, `TEMPERATURE_K = 450.0`,
`MAX_RMS_A = None`; the generator name stays `displaced` (one generator, one distribution,
as the user ruled: no classical/quantum alternative remains in this workflow), the Frame set
Record keeps `DISTRIBUTION`, `TEMPERATURE_K`, `MAX_RMS_A` so old and new Frame sets are
told apart by their Records. `02_frames.py`: `--distribution` removed; `--temperature`
default 450; `--max-rms` default none. The energy window `ENERGY_WINDOW_KCAL = 275`
(already ANI's number) stays the only filter, applied on the engine energy as today.

Removed from the workflow: the `classical` and `quantum` choices in `frames.py` and
`02_frames.py`. `hessian.thermal_displacements` keeps its two harmonic branches because two
other callers use them for other purposes (`scripts/calibration/s0_composite_energy_force.py`,
branch B's QHA sampling record); they are not reachable from the Hessian-learning workflow.
Existing labelled `displaced` frames (the 65-frame smoke set, the methyloxirane fixture) were
classical 298 K draws: they stay as data, their Records say so, and no new frame is drawn
that way.

Judge: the RMS-displacement bins (0 / < 0.08 / < 0.15 / >= 0.15 A) read every frame's
`rms_displacement_A`; with no ceiling the last bin is populated.

Tests (`tests/unit/t_frames_nms.py`): over 4000 draws the harmonic energy has mean
(3/4) N_a k_B T within 3 % and never exceeds (3/2) N_a k_B T; signs are balanced; the same
seed reproduces the frame; the Record carries `c_sum` and `harmonic_energy_kcal`; the two
harmonic branches are bit-identical to before (regression on the fixture).

## 4. Measured on the fixtures (4000 draws on the reference basin Hessian, `nms_probe.py`)

| molecule | draw | T | RMS mean / p95 / max (A) | E_h mean / p95 / max (kcal/mol) | RMS > 0.15 A |
|---|---|---|---|---|---|
| propanal (N_a 10) | classical (today) | 298 | 0.118 / 0.217 / 0.365 | 7.1 / 10.7 / 17.0 | 22 % |
| propanal | NMS | 298 | 0.092 / 0.155 / 0.217 | 4.4 / 8.4 / 8.9 | 7 % |
| propanal | **NMS** | **450** | 0.113 / 0.190 / 0.266 | 6.6 / 12.6 / 13.4 | 23 % |
| propanal | NMS | 600 | 0.130 / 0.220 / 0.307 | 8.8 / 16.8 / 17.9 | 37 % |
| 2-methyloxirane (N_a 10) | classical | 298 | 0.084 / 0.155 / 0.272 | 7.1 / 10.7 / 17.0 | 6 % |
| 2-methyloxirane | **NMS** | **450** | 0.080 / 0.133 / 0.184 | 6.6 / 12.6 / 13.4 | 1 % |

(E_h max = (3/2) N_a k_B T: 8.9 / 13.4 / 17.9 kcal/mol at 298 / 450 / 600 K, N_a = 10, as (N3) says.)

## 5. Cost, on named molecules

ORCA 6.0.1, wB97M-D3(BJ)/def2-TZVPPD, per frame: 10 atoms -- analytic Hessian 226 s on
8 cores = 0.5 core-h (S0-C-38; E/F alone ~0.05); 19 atoms (a 9-heavy-atom QM9 molecule) --
40-80 min on 4 ranks = 2.7-5.3 core-h, take 4 (round 5); E/F alone 3-5 min x 4 = 0.27.
One day of the 12 x 64-core allocation = 18,432 core-h.

| molecule | N_a, K | ANI's S | frames per molecule under S x K | with H at every frame | E/F only |
|---|---|---|---|---|---|
| propanal C3H6O (4 heavy) | 10, 24 | 400 | 9,600 | 4,800 core-h | 480 core-h |
| 2-methyloxirane C3H6O (4 heavy) | 10, 24 | 400 | 9,600 | 4,800 | 480 |
| a 9-heavy QM9 molecule | 19, 51 | 5 (ANI's 8-heavy row) | 255 | 1,020 core-h | 69 core-h |
| campaign, 6,458 such molecules | | 5 | 1.65 M | **6.6 M core-h = 358 days** | 0.45 M = 24 days |

Against the frame counts this project can label with H (3 basins per molecule on draw300):

| per basin | NMS frames per molecule | + 5 stationary frames | H on all, per molecule | campaign (6,458) |
|---|---|---|---|---|
| 2 | 6 | 11 | 44 core-h | 284 k core-h = **15 days** |
| 4 | 12 | 17 | 68 core-h | 439 k = **24 days** |
| ANI S x K | 255 | 260 | 1,040 core-h | 6.7 M = **365 days** |

## 6. The per-basin count, and why ANI's S x K is not the number

ANI's N = S x K is set for a different problem: a model trained FROM SCRATCH on ENERGIES
ONLY at a cheap level (wB97x/6-31G(d) single points, ~1 min each), which needs dense
coverage of a wide window to learn the surface shape from energies alone. Three things
differ here: (i) the base model already carries 951,813 SPICE E/F frames, so coverage of
the thermal window is not what the fine-tune buys; (ii) every frame here carries E, F and
H -- one Hessian frame is worth K x (K+1)/2 second-derivative numbers plus 3N first
derivatives, against ANI's one number per conformation, which is the data-efficiency
Rodriguez measured (E-F-H at 2 % of the data beats E-F at 80 %); (iii) the label costs
1,000x ANI's. So the count per basin is set by the label bill, and ANI's S is the number
of *energy* samples that replaces what a Hessian says in one.

Proposal: **4 NMS frames per basin at 450 K, H on all** (17 frames per molecule, the same
count as today, ~24 days of 12 nodes for the campaign; the stationary-only-H fallback of
the framework's step 1 is ~11 days), and the 65-frame smoke set regenerated with NMS
frames beside its classical ones so the two draws are judged on the same molecules before
the campaign's array goes in. If the bill must be 15 days, 2 per basin.
