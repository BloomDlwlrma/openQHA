# Projected Hessian learning for MACE-OFF23: derivation, algorithm, and the rewrite of mace-torch 0.3.16

Design note for the Hessian-learning step (round-2 grilling, Q4/Q5/Q7/Q13), 2026-09-18.
Ruling it rests on: HIP's full-matrix metrics are the ruler, PHL's Hessian-vector-product
loss is the loss, neither is the potential. Everything below is stated so that it can fail:
the estimator is unbiased (proof), the full E-F-H loss is the same estimator with a
deterministic probe set (proof), and each code change names the file and the line it
touches in `mace-torch 0.3.16` as installed in the `openqha` environment.

## 1. Notation

| symbol | meaning |
|---|---|
| `x ∈ R^{3N}` | Cartesian coordinates of one structure (a Label site: a basin, a displaced point, a line point) |
| `E_θ(x)`, `F_θ = −∇_x E_θ`, `H_θ = ∇_x² E_θ ∈ R^{3N×3N}` | MACE energy, forces, Hessian (parameters θ) |
| `E_r, F_r, H_r` | the reference-level Label at the same `x` (wB97M-D3(BJ)/def2-TZVPPD, DefGrid3 if Q12 rules so); `H_r` is the raw Cartesian Hessian, never pre-projected |
| `M = diag(m_1,m_1,m_1, …, m_N,m_N,m_N)` | atomic masses (amu) |
| `K = M^{-1/2} H M^{-1/2}` | mass-weighted Hessian (eV Å^-2 amu^-1) |
| `V ∈ R^{3N×n_rig}` | orthonormal translation/rotation vectors in mass-weighted coordinates at `x` (`n_rig` = 6, 5 for a linear molecule; `hessian.rigid_body_vectors`) |
| `P = I − V Vᵀ` | Eckart projector (symmetric, idempotent, `P V = 0`) |
| `K̃ = P K P` | projected mass-weighted Hessian; `K̃ L = L Λ`, `Λ = diag(λ_i)`, `ω_i = sign(λ_i) √|λ_i|` in cm^-1 |
| `n_vib = 3N − n_rig` | number of vibrational modes |
| `A_θ = P M^{-1/2} (H_θ − H_r) M^{-1/2} P = K̃_θ − K̃_r` | the error operator: symmetric, `A V = 0` |
| `D = L_rᵀ K̃_θ L_r` | the engine's curvature in the reference-mode basis (`hessian_compare` family 4; `D_ii = OMEGA_ALONG_REF_CM[i]²` in eV Å^-2 amu^-1) |

The Label is the Cartesian `H_r` at `x`. At a displaced (non-stationary) `x` the rigid block
`VᵀK_rV` is not zero (the rotational block feels the gradient, 29 cm^-1 on the analytic
propanal minimum, ~100 at a foreign geometry, S0-C-43/44); `P` removes it on both sides
identically, so the loss compares curvature only. It is the same `P` on both sides that
makes the comparison meaningful; a Label stored already projected would hide which `P`
was used and is forbidden.

## 2. The loss and its derivation

### 2.1 The target quantity

The quantity thermochemistry depends on is the vibrational spectrum of `K̃`, so the loss
is the squared Frobenius norm of the projected mass-weighted error, per mode:

    L_H(θ) = (1/n_vib) ‖A_θ‖_F²  =  (1/n_vib) ‖P M^{-1/2} (H_θ − H_r) M^{-1/2} P‖_F²        (1)

Properties. (i) `L_H = 0` iff `K̃_θ = K̃_r`, i.e. identical spectrum AND identical modes; a
rigid-block difference contributes nothing (`P` kills it). (ii) Weyl's inequality gives
`|λ_i^θ − λ_i^r| ≤ ‖A‖_2 ≤ ‖A‖_F`: driving (1) to zero drives every sorted eigenvalue,
hence every frequency, to the reference. (iii) In the reference-mode basis,
`L_rᵀ A L_r = D − Λ_r`, so

    ‖A‖_F² = ‖D − Λ_r‖_F² = Σ_i (D_ii − λ_i^r)² + Σ_{i≠j} D_ij²                          (2)

the diagonal term is the per-mode curvature error `hessian_compare` already reports
(`OMEGA_ALONG_REF_CM` vs `OMEGA_REF_CM`), the off-diagonal term is its `MIXING`. The loss
is exactly family 4, squared and summed. (`L_r` is orthonormal and spans range(P), so the
Frobenius norm is basis-invariant on that subspace.)

### 2.2 Mode weighting (Q5): the entropy's own sensitivity

Let `W = diag(w_i)`, `w_i ≥ 0`, one weight per reference mode, and define the weighted
projector `P_W = L_r W^{1/2} L_rᵀ` (symmetric; `P_I = P`). Then

    L_H^W(θ) = (1/n_vib) ‖P_W M^{-1/2} (H_θ − H_r) M^{-1/2} P_W‖_F²
             = (1/n_vib) Σ_{i,j} w_i w_j (D_ij − λ_i^r δ_ij)²                             (3)

Recommended weights: `w_i = |∂S_msRRHO/∂ω_i|(T=298 K) / max_i|·|`, evaluated on the
reference spectrum with the crest preset (the msRRHO interpolation weight is inside
`∂S/∂ω`, so modes under τ are not over-driven; a mode with `ω_r < PROFILE_BELOW_CM` keeps
its Label but gets the weight the formula gives it, which is large -- Q6 decides whether the
*judge* sets it aside, the loss does not). (3) is still a quadratic form of a symmetric
operator, so §2.3 applies unchanged with `P_W` in place of `P`.

### 2.3 The Hutchinson estimator (PHL) and its unbiasedness

For any symmetric `A ∈ R^{n×n}` and a random `v ∈ R^n` with `E[v] = 0`, `E[v vᵀ] = I`
(Rademacher `v_i ∈ {−1,+1}` or standard Gaussian):

    E_v ‖A v‖²  =  E_v [ vᵀ Aᵀ A v ]  =  tr( Aᵀ A · E[v vᵀ] )  =  tr(AᵀA)  =  ‖A‖_F²        (4)

Apply it to `A = A_θ`. Because `P` is idempotent and `M^{-1/2}` diagonal,

    A v = P M^{-1/2} (H_θ − H_r) ṽ ,     ṽ := M^{-1/2} P v                                  (5)

so one draw needs `H_θ ṽ` and `H_r ṽ` only -- the model side is a Hessian-vector product,
the reference side a matrix-vector product on the stored Label. With `k` independent
probes,

    L̂_H^{(k)}(θ) = (1/(n_vib k)) Σ_{j=1}^{k} ‖ P M^{-1/2} ( H_θ ṽ_j − H_r ṽ_j ) ‖²           (6)

    E[L̂_H^{(k)}] = L_H     for every k ≥ 1                                                  (7)

(7) is (4) applied term by term; it holds for `P_W` as well (`A_W = P_W M^{-1/2} ΔH M^{-1/2} P_W`,
probes `ṽ = M^{-1/2} P_W v`). Variance, Rademacher probes, `B = AᵀA`:

    Var[L̂_H^{(k)}] = (2 / (n_vib² k)) ( ‖B‖_F² − Σ_i B_ii² )                                 (8)

so `k = 2-4` probes per structure give a gradient whose noise averages out over a batch
and over epochs (PHL's finding; plan_C §8.6 band), and the estimator never biases the
optimum.

### 2.4 The Hessian-vector product by automatic differentiation

`F_θ = −∇_x E_θ` is itself the output of `torch.autograd.grad` with `create_graph=True`
during training (mace `compute_forces(..., training=True)`). For a constant probe `ṽ`,

    H_θ ṽ  =  ∇_x ( ∇_x E_θ · ṽ )  =  −∇_x ( F_θ · ṽ )                                       (9)

one more backward pass over the force graph, with `create_graph=True` so that the result
is differentiable with respect to θ. No full Hessian is formed; memory is one third-order
graph for one vector, not `3N` of them. Forward-over-reverse (`torch.func.jvp` of the
gradient) is numerically identical and cheaper in one tape; both are third-order at
training time (PFT, plan_C §8.7). Reverse-over-reverse is used here because it needs no
functional rewrite of the MACE forward.

### 2.5 The full E-F-H loss is the same estimator with a deterministic probe set

Take the probes to be the `n_vib` reference modes, `v_j = L_{r,j}` (orthonormal in
range(P), `P v_j = v_j`). Then

    Σ_{j=1}^{n_vib} ‖A L_{r,j}‖²  =  ‖A L_r‖_F²  =  ‖A‖_F²                                    (10)

because `A = P A P` and `L_r L_rᵀ = P`. So `L̂_H` with `probe = "modes"` and `k = n_vib` is
**exactly** (1), with zero variance, at the cost of `n_vib` HVPs -- the Rodriguez full
Hessian loss on the same code path (and `3N` Cartesian unit probes `e_j` give it too,
since `Σ_j ‖A e_j‖² = ‖A‖_F²`, at `n_rig` more HVPs). One implementation therefore serves
both Q4 options: `probe="modes"` measures the ceiling on the seven-molecule smoke set,
`probe="rademacher", k=2-4` trains the 200-molecule set. Per-mode probes also make (3)
explicit: with `probe="modes"` and weights, term `j` is `w_j² Σ_i w_i² (D_ij − λ_i δ_ij)²`.

### 2.6 The total loss and its gradient

    L(θ) = w_E L_E + w_F L_F + w_H L̂_H^{(k)}                                                (11)

with `L_E`, `L_F` mace's existing per-config-weighted squared errors. Gradient of the
Hessian term, per probe (`ρ_j := P M^{-1/2} (H_θ ṽ_j − H_r ṽ_j)`):

    ∂L̂_H/∂θ = (2/(n_vib k)) Σ_j ρ_jᵀ P M^{-1/2} ∂(H_θ ṽ_j)/∂θ                                 (12)

which autograd produces from (9) when the HVP carries a graph. Cost per batch: one forward,
one backward for forces (graph kept), `k` HVP backward passes -- **one per probe index for
the whole batch**, because structures in a batch are disconnected graphs and
`−∇_x Σ_n F_{θ,n}·ṽ_{n,j}` returns every structure's `H_n ṽ_{n,j}` at once -- and one
parameter backward through all of it. Roughly `(2 + 2k)` forward-equivalents against
`(2 + 6N)` for the full matrix.

Units and the weight band. `H` in eV Å^-2, `K` in eV Å^-2 amu^-1; `L_H` per mode. PHL and
Rodriguez report `w_H/w_F ≈ 0.25-0.30` on *Cartesian, unweighted* Hessians; with the
mass weighting our `L_H` is smaller by ~`1/m` for the heavy modes and larger for the
hydrogenic ones, so the band is a starting point, not a value: set `w_H` so that
`w_H L̂_H ≈ w_F L_F` on the base model at epoch 0 (the smoke set gives both numbers), scan
×0.3 / ×3 around it (plan_C §5.2), report all three.

### 2.7 Forgetting (Q7): co-training as multihead fine-tuning

mace-torch 0.3.16 already implements the replay that PFT calls co-training:
`--foundation_model mace-off23-medium --multiheads_finetuning True --pt_train_file
<spice_subset.xyz> --num_samples_pt <n>` trains a second head on a sample of the
pretraining data alongside the fine-tuning head (E and F only, on SPICE frames that are on
disk). The Hessian term applies to the fine-tuning head; the pretraining head keeps the
E/F loss. The judge for forgetting (acceptance 7a rewritten) is E/F error on a fixed
5,000-frame draw of the SPICE test split within 15 % of the base model's.

## 3. Algorithm listings

```
Algorithm 1  probes(x, M, H_r, mode="rademacher", k, W=None)      -- per structure, no autograd
  V  <- rigid_body_vectors(M, x)                                    # 3N x n_rig, orthonormal, mass-weighted
  P  <- I - V V^T
  if W is not None:  L_r, Λ_r <- eigh(P K_r P) restricted to range(P);  P <- L_r W^{1/2} L_r^T
  if mode == "rademacher": v_j ~ Uniform{-1,+1}^{3N},  j = 1..k
  if mode == "modes":      v_j <- L_{r,j},             j = 1..n_vib        (k := n_vib; exact, eq. 10)
  ṽ_j <- M^{-1/2} P v_j                                                   # the vector the model sees
  r_j <- P M^{-1/2} H_r ṽ_j                                                # reference side, a matmul
  return {ṽ_j}, {r_j}, P, M^{-1/2}
```

```
Algorithm 2  one training step (batch B of structures n = 1..|B|)
  for n in B:  ({ṽ_{n,j}}, {r_{n,j}}, P_n, M_n^{-1/2}) <- Algorithm 1 (x_n, M_n, H_{r,n})   # in the data loader
  E_θ, F_θ <- model(B, training=True)                                     # forces with create_graph=True
  for j = 1..k:                                                            # k backward passes for the whole batch
      g_j <- autograd.grad( -Σ_n F_{θ,n} · ṽ_{n,j},  x,  create_graph=True )        # = [H_{θ,n} ṽ_{n,j}]_n  (eq. 9)
      for n in B:  ρ_{n,j} <- P_n M_n^{-1/2} g_{n,j} - r_{n,j}
  L_H <- (1/|B|) Σ_n (1/(n_vib,n k)) Σ_j ‖ρ_{n,j}‖²                                  # eq. 6
  L   <- w_E L_E(B) + w_F L_F(B) + w_H L_H
  θ   <- optimizer.step(∇_θ L)                                             # third-order graph through the HVPs
  (multihead replay: the pretraining head's E/F batch is drawn and stepped by mace's own loop)
```

```
Algorithm 3  the ruler (evaluation, held-out set)                         -- inference only
  for each structure: H_θ <- MACECalculator.get_hessian(atoms)            # shipped path, create_graph=False
                      hessian_compare.compare_hessians(H_θ, H_r, masses, x)   # HIP families 1-4, floors, D, MIXING
  report in- and out-of-distribution rows separately (Q8); thermochemistry at own minima (Q13)
```

```
Algorithm 4  acceptance (each must be able to fail)
  A1 unbiasedness:  on one molecule, mean of L̂^{(k)} over 200 Rademacher draws -> (1) within 1 % as k grows (plan_C acc. 3)
  A2 exactness:     probe="modes", k = n_vib  ==  (1) to 1e-10; Cartesian e_j probes likewise             (eq. 10)
  A3 projection:    adding eps V V^T (mass-weighted) to H_r changes L̂ by 0                                (P V = 0)
  A4 gradient:      finite-difference of L̂ w.r.t. one parameter == autograd to 1e-6 (third-order graph is right)
  A5 must-pass:     θ = base, H_r := H_base  -> L_H = 0 and zero gradient
  A6 must-fail:     H_r := 0.81 H_base (0.9x frequencies) -> L_H = (0.19)^2 ‖K̃_base‖_F^2 / n_vib and ∇ ≠ 0
  A7 two estimators: reverse-over-reverse (9) and forward-over-reverse agree in mean (plan_C acc. 16)
```

## 4. The rewrite of mace-torch 0.3.16, file by file

Installed at `<env>/lib/python3.11/site-packages/mace/`. Nothing in the shipped Hessian
path is changed; the training path is added beside it. Every addition is a keyword with a
default that reproduces current behaviour, so `mace_run_train` without the new flags is
unchanged.

### 4.1 `modules/utils.py` -- the HVP, next to `compute_hessians_vmap` (line 113)

```python
def compute_hessian_vector_products(
    forces: torch.Tensor,        # [n_nodes, 3], built with create_graph=True
    positions: torch.Tensor,     # [n_nodes, 3], requires_grad
    probes: torch.Tensor,        # [k, n_nodes, 3], constant (already M^-1/2 P v), whole batch
    training: bool = True,
) -> torch.Tensor:               # [k, n_nodes, 3] = H v_j per structure (batch is block-diagonal)
    """H v = -d(F . v)/dx, one backward pass per probe over the whole batch (eq. 9).
    `create_graph=training` keeps the result differentiable w.r.t. the parameters -- the
    one thing `compute_hessians_vmap` (create_graph=False, inference) cannot give."""
    out = []
    for j in range(probes.shape[0]):
        g = torch.autograd.grad(
            outputs=[-(forces * probes[j]).sum()],
            inputs=[positions],
            retain_graph=True,
            create_graph=training,
            allow_unused=True,
        )[0]
        out.append(torch.zeros_like(positions) if g is None else g)
    return torch.stack(out, dim=0)
```

`get_outputs(...)` (line 167) gains `hessian_probes: Optional[torch.Tensor] = None` and,
after the force block,

```python
    hvp = None
    if hessian_probes is not None:
        assert forces is not None
        hvp = compute_hessian_vector_products(forces, positions, hessian_probes, training=training)
```

with `training=(training or compute_hessian or compute_edge_forces or hessian_probes is not None)`
in both `compute_forces_virials` / `compute_forces` calls (lines 193, 200), and `hvp`
appended to the returned tuple. `compute_hessians_vmap` / `compute_hessians_loop` stay as
they are: they are the ruler's path.

### 4.2 `modules/models.py` -- `MACE.forward` (line 276) and `ScaleShiftMACE.forward` (line 455)

Signature: add `hessian_probes: Optional[torch.Tensor] = None` after `compute_hessian`.
Call (lines 401 and 583): pass `hessian_probes=hessian_probes`, unpack the sixth output
`hvp`. Return dict: add `"hvp": hvp`. The same three-line change in `modules/extensions.py`
(lines 110/259/294 and 600/931/969) if those model classes are ever fine-tuned. The
`Optional[torch.Tensor]` argument keeps the forward TorchScript-compilable (the calculator
compiles the model; an `Optional` tensor default `None` is supported, as `compute_hessian`
already shows for a bool).

### 4.3 `calculators/mace.py` -- nothing removed, one method added after `get_hessian` (line 770)

```python
    def get_hessian_vector_products(self, atoms, probes):
        """H v_j for the given Cartesian probes [k, N, 3] (acceptance A1/A2/A7, and the
        along-mode curvature D_ii = L_r^T K L_r without forming H). Inference: no graph."""
        batch = self._atoms_to_batch(atoms).to_dict()
        p = torch.as_tensor(probes, dtype=torch.get_default_dtype(), device=self.device)
        out = self.models[0](self._clone_batch(batch), hessian_probes=p, compute_stress=False, training=False)
        return out["hvp"].detach().cpu().numpy()
```

`get_hessian` (the ruler) is untouched.

### 4.4 `data/atomic_data.py` and `data/utils.py` -- the Label in the batch

The reference Hessian is per structure and `3N×3N`, which does not collate as a node
tensor when `N` varies. Store it flattened, one 1-D tensor per graph, and concatenate along
dim 0 like `ptr` does for nodes:

```python
# AtomicData.__init__ (line 58): new optional fields
hessian: Optional[torch.Tensor],          # [9 N^2] flattened row-major (3N x 3N), eV/A^2
hessian_weight: Optional[torch.Tensor],   # [], per config
sqrt_masses: torch.Tensor,                # [n_nodes], amu^1/2 (from atomic numbers, ase.data)
```

`__cat_dim__` (torch_geometric `Data`, `tools/torch_geometric/data.py:168`): 1-D tensors
concatenate along 0 by default, so `batch.hessian` is the concatenation and the per-graph
slice is `[9 n_k^2]` with `n_k = ptr[k+1]-ptr[k]`; the loss computes the offsets
`cumsum(9 n_k^2)`. `from_config` (line 168): read `config.hessian` (a `[3N,3N]` numpy
array or None), `config.hessian_weight`. `data/utils.py`: `Configuration` gains
`hessian`, `hessian_weight`; `config_from_atoms` reads `atoms.info[hessian_key]` -- the
extxyz Label file carries it as an info array of `9N²` floats (`REF_hessian`; ASE writes
info arrays as space-separated values, 3,600 numbers for 10 atoms -- accepted, the files
are ours) and `atoms.info.get("hessian_weight", 1.0)`; `--hessian_key` joins
`--forces_key` / `--stress_key` in `arg_parser.py`. `KeySpecification.info_keys` gains
`hessian_key`.

The probes are drawn per batch, not stored: they are a function of `(x, M, H_r, mode, k, W)`
and change every epoch for Rademacher.

### 4.5 `modules/loss.py` -- the Hessian term and the loss class

```python
def projected_hvp_error(ref: Batch, pred: TensorDict, ddp=None) -> torch.Tensor:
    """(1/n_vib) Σ_j ‖P M^-1/2 (H_θ ṽ_j − H_r ṽ_j)‖² per graph (eq. 6), weighted by
    ref.weight * ref.hessian_weight, averaged over the batch. pred["hvp"]: [k, n_nodes, 3]
    from the probes ref.hessian_probes [k, n_nodes, 3]; ref.hessian_ref_products [k, n_nodes, 3]
    (= P M^-1/2 H_r ṽ_j, made with the probes in the loader); ref.projector_rows: the
    per-graph P M^-1/2 applied on the fly through ref.sqrt_masses and ref.rigid_vectors."""
    ...
class WeightedEnergyForcesHessianLoss(torch.nn.Module):
    def __init__(self, energy_weight=1.0, forces_weight=1.0, hessian_weight=1.0,
                 n_probes=4, probe="rademacher", mode_weighting=None): ...
    def make_probes(self, batch) -> torch.Tensor:        # Algorithm 1 for the batch; called BEFORE the model
    def forward(self, ref, pred):
        return (self.energy_weight * weighted_mean_squared_error_energy(ref, pred)
              + self.forces_weight * mean_squared_error_forces(ref, pred)
              + self.hessian_weight * projected_hvp_error(ref, pred))
```

The probes have to exist before the forward (they enter it), so the loss owns
`make_probes(batch)`; the loader precomputes `P`, `M^{-1/2}`, `L_r` per structure once (a
`3N×3N` eigendecomposition of `K̃_r`, ~µs at `3N ≤ 60`) and stores them on the `AtomicData`
like `hessian` (flattened, concatenated). Per-graph slicing in the loss uses `ref.ptr`.

### 4.6 `tools/train.py` (line 418, 492, 574) and `tools/scripts_utils.py` (line 659)

```python
        probes = loss_fn.make_probes(batch) if output_args.get("hessian") else None
        output = model(batch_dict, training=True, compute_force=output_args["forces"],
                       compute_virials=output_args["virials"], compute_stress=output_args["stress"],
                       hessian_probes=probes)
        loss = loss_fn(pred=output, ref=batch)
```

`get_loss_fn`: `elif args.loss == "energy_forces_hessian": WeightedEnergyForcesHessianLoss(
args.energy_weight, args.forces_weight, args.hessian_weight, args.n_hessian_probes,
args.hessian_probe, args.hessian_mode_weighting)`; `configure_model` sets
`output_args["hessian"] = args.loss == "energy_forces_hessian"`. `arg_parser.py`:
`--hessian_weight` (float), `--hessian_key` (str, `REF_hessian`), `--n_hessian_probes`
(int, 4), `--hessian_probe` (`rademacher | gaussian | modes | cartesian`),
`--hessian_mode_weighting` (`none | entropy`), `--hessian_weight_stage_two`.
Evaluation inside training (`tools/train.py` `evaluate`, line ~574) reports the same term
with `probe="modes"` so the logged validation Hessian loss is exact (eq. 10), not sampled.

### 4.7 What the fine-tuning command becomes

```
mace_run_train --name off23_hess --foundation_model mace-off23-medium \
    --multiheads_finetuning True --pt_train_file spice_subset.xyz --num_samples_pt 5000 \
    --train_file hessian_set_train.xyz --valid_file hessian_set_valid.xyz \
    --energy_key REF_energy --forces_key REF_forces --hessian_key REF_hessian \
    --loss energy_forces_hessian --energy_weight 1 --forces_weight 100 --hessian_weight <scanned> \
    --n_hessian_probes 4 --hessian_probe rademacher --hessian_mode_weighting entropy \
    --default_dtype float64 --max_num_epochs 100 --device cuda
```

## 5. Where this sits against the rest of the design

- `hessian_compare` (openQHA) is the ruler; nothing in it changes. Family 4's `D` is the
  loss in the mode basis (eq. 2), so the ruler and the loss are the same quantity read two
  ways -- the judge (Q8) reads the ruler, the optimiser reads the estimator.
- The Label is the Cartesian `H_r` at a fixed geometry; the gradient term of a foreign
  geometry (S0-C-43) is inside both `H_θ` and `H_r` identically and is not an error; the
  projection removes the rigid block where the gradient shows up as rotation.
- Reference Hessians carry the grid noise of S0-C-44 (~10 cm^-1 on the softest mode with
  DefGrid2); a Label set should be produced with one grid (Q12), and the judge's threshold
  cannot be tighter than that floor.
- The along-mode line points (`mode_curvature`, ticket 32) are Label sites with a known
  reference curvature along one direction; with `probe="modes"` restricted to that mode
  they are a one-probe Label (a `D_ii` target without a full `H_r`) -- Q3(c).
