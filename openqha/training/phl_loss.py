"""The projected Hessian loss as a mace loss module: eq. 11 with the HVP estimator
(eq. 6) in training and the exact full-matrix term (eq. 1) in evaluation.

PRODUCTION. Ticket 11 of the Hessian-learning set.

This is the `nn.Module` mace's `train()` calls as `loss_fn(pred=output, ref=batch)`. It
reaches mace through the fork's external-loss hook (ticket 13: `--loss external
--loss_module openqha.training.phl_loss:build`); the E and F terms are mace's own
(`mace.modules.loss`, public), the Hessian term is here. The batch fields it reads are
the ones the fork's commit A adds (ticket 12): `hessian` (the Labels, flattened 9 n_k^2
per graph and concatenated like `ptr`), `has_hessian` [n_graphs], `hessian_weight`
[n_graphs], `sqrt_masses` [n_nodes]; and mace's `ptr`, `weight`, `positions`.

Training: the HVP is taken INSIDE the loss from `pred["forces"]` and `ref["positions"]`
(`hvp.hvp_from_forces`, create_graph=True) -- one backward pass per probe index for the
whole batch, since a batch is block-diagonal. Evaluation: when the forward was asked for
`compute_hessian` (the fork's `evaluate` does so when `wants_hessian_at_eval` is set),
`pred["hessian"]` is the full matrix and the term is eq. 1 exactly -- the logged
validation Hessian loss is the ruler's quantity, never sampled.

Per-structure constants (P_W, M^-1/2, L_r, the entropy weights) depend only on the Label,
the masses and the positions, so they are computed once per frame and cached by the
Label's bytes: the eigendecomposition is microseconds at 3N <= 60, the entropy weights
are 2 n_vib msRRHO evaluations and are not.
"""
import hashlib

import numpy as np
import torch

from ..thermochem import hessian as hessian_mod
from . import hvp as hvp_mod
from . import phl

MODE_WEIGHTINGS = ("entropy", "none")


def _field(ref, name):
    """A batch field, by item or attribute (mace's Batch answers both; a test's stand-in
    may answer one)."""
    try:
        return ref[name]
    except (TypeError, KeyError, IndexError):
        return getattr(ref, name)


def _has(ref, name):
    try:
        return _field(ref, name) is not None
    except (AttributeError, KeyError):
        return False


class FrameConstants:
    """What Algorithm 1 needs of one frame, computed once: reference modes, the
    (weighted) projector, M^-1/2, n_vib -- numpy for the probes, torch for the loss."""

    __slots__ = ("hessian_r", "masses", "positions", "modes_r", "lam_r", "weights", "projector",
                 "inv_sqrt_m", "n_vib", "projector_t", "inv_sqrt_m_t")

    def __init__(self, hessian_r, masses, positions, mode_weighting, temperature_K, preset):
        self.hessian_r = np.asarray(hessian_r, dtype=float)
        self.masses = np.asarray(masses, dtype=float)
        self.positions = np.asarray(positions, dtype=float)
        self.modes_r, self.lam_r = phl.reference_modes(self.hessian_r, self.masses, self.positions)
        self.n_vib = int(self.modes_r.shape[1])
        if mode_weighting == "entropy":
            omega = hessian_mod.eigenvalues_to_cm_inv(self.lam_r)
            self.weights, _ds = phl.entropy_weights(omega, self.masses, self.positions,
                                                    temperature_K=temperature_K, preset=preset)
        else:
            self.weights = None
        self.projector = phl.weighted_projector(self.modes_r, self.weights)
        _m3, self.inv_sqrt_m = phl.mass_vectors(self.masses)
        self.projector_t = None
        self.inv_sqrt_m_t = None

    def torch_constants(self, like):
        """P_W and M^-1/2 as tensors of `like`'s dtype and device (made once)."""
        if self.projector_t is None or self.projector_t.dtype != like.dtype or self.projector_t.device != like.device:
            self.projector_t = torch.as_tensor(self.projector, dtype=like.dtype, device=like.device)
            self.inv_sqrt_m_t = torch.as_tensor(self.inv_sqrt_m, dtype=like.dtype, device=like.device)
        return self.projector_t, self.inv_sqrt_m_t


def graph_labels(ref):
    """Per graph: (index, n_atoms, H_r [3n, 3n] numpy or None, masses, positions) read
    from the batch fields of ticket 12. Graphs without a Label give None."""
    ptr = _field(ref, "ptr").detach().cpu().numpy()
    n_k = ptr[1:] - ptr[:-1]
    has = _field(ref, "has_hessian").detach().cpu().numpy().astype(bool) if _has(ref, "has_hessian") \
        else np.ones(len(n_k), dtype=bool)
    hess = _field(ref, "hessian").detach().cpu().numpy() if _has(ref, "hessian") else np.zeros(0)
    sqrt_m = _field(ref, "sqrt_masses").detach().cpu().numpy()
    pos = _field(ref, "positions").detach().cpu().numpy()
    out, off = [], 0
    for g, n in enumerate(n_k):
        size = 9 * int(n) ** 2 if has[g] else 0
        h = None
        if has[g]:
            if off + size > hess.size:
                raise ValueError("batch.hessian holds {} numbers; graph {} needs {} more at offset {}".format(
                    hess.size, g, size, off))
            h = hess[off:off + size].reshape(3 * n, 3 * n)
        off += size
        a, b = int(ptr[g]), int(ptr[g + 1])
        out.append((g, int(n), h, sqrt_m[a:b] ** 2, pos[a:b]))
    if off != hess.size:
        raise ValueError("batch.hessian holds {} numbers but the labelled graphs account for {}".format(hess.size, off))
    return out


class WeightedEnergyForcesHessianLoss(torch.nn.Module):
    """w_E L_E + w_F L_F + w_H L_H (eq. 11): mace's weighted E and F terms plus the
    projected Hessian term -- the estimator (eq. 6) when `pred` has no "hessian", the
    exact full-matrix value (eq. 1 / eq. 3) when it has."""

    #: the fork's `evaluate` reads this and asks the forward for the full Hessian
    wants_hessian_at_eval = True

    def __init__(self, energy_weight=1.0, forces_weight=1.0, hessian_weight=1.0, n_probes=4,
                 probe="rademacher", mode_weighting="entropy", seed=None, temperature_K=298.15,
                 preset="crest", cache_size=4096):
        super().__init__()
        if probe not in phl.PROBE_MODES:
            raise ValueError("probe must be one of {}; got {!r}".format(phl.PROBE_MODES, probe))
        if mode_weighting not in MODE_WEIGHTINGS:
            raise ValueError("mode_weighting must be one of {}; got {!r}".format(MODE_WEIGHTINGS, mode_weighting))
        dt = torch.get_default_dtype()
        self.register_buffer("energy_weight", torch.tensor(float(energy_weight), dtype=dt))
        self.register_buffer("forces_weight", torch.tensor(float(forces_weight), dtype=dt))
        self.register_buffer("hessian_weight", torch.tensor(float(hessian_weight), dtype=dt))
        self.n_probes = int(n_probes)
        self.probe = probe
        self.mode_weighting = mode_weighting
        self.seed = seed
        self.temperature_K = float(temperature_K)
        self.preset = preset
        self.rng = np.random.default_rng(seed)
        self._cache = {}
        self._cache_size = int(cache_size)
        # the last values of the three terms, for the training log
        self.last_terms = dict(energy=None, forces=None, hessian=None, hessian_exact=None, n_labelled=0)

    # ---- per-frame constants ---------------------------------------------------------
    def constants(self, hessian_r, masses, positions):
        key = hashlib.sha1(np.ascontiguousarray(hessian_r).tobytes()).digest() + \
            hashlib.sha1(np.ascontiguousarray(positions).tobytes()).digest()
        c = self._cache.get(key)
        if c is None:
            c = FrameConstants(hessian_r, masses, positions, self.mode_weighting, self.temperature_K, self.preset)
            if len(self._cache) >= self._cache_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = c
        return c

    # ---- Algorithm 1 for a batch -----------------------------------------------------
    def make_probes(self, ref, like):
        """Probes for the whole batch: a tensor [k_max, n_nodes, 3] (zero rows for
        graphs without a Label and for j >= k_g), and per labelled graph
        (g, slice, k_g, v~ [k_g, 3n], r_j [k_g, 3n] numpy, constants, denominator)."""
        labels = graph_labels(ref)
        ptr = _field(ref, "ptr").detach().cpu().numpy()
        n_nodes = int(ptr[-1])
        per_graph = []
        k_max = 0
        for g, n, h_r, masses, pos in labels:
            if h_r is None:
                continue
            c = self.constants(h_r, masses, pos)
            vt, r, info = phl.make_probes(masses, pos, c.hessian_r, mode=self.probe, k=self.n_probes,
                                          weights=c.weights, rng=self.rng)
            per_graph.append((g, slice(int(ptr[g]), int(ptr[g + 1])), info["k"], vt, r, c, info["denominator"]))
            k_max = max(k_max, info["k"])
        probes = torch.zeros((k_max, n_nodes, 3), dtype=like.dtype, device=like.device)
        for g, sl, k_g, vt, _r, _c, _den in per_graph:
            probes[:k_g, sl, :] = torch.as_tensor(vt.reshape(k_g, -1, 3), dtype=like.dtype, device=like.device)
        return probes, per_graph

    # ---- the two Hessian terms -----------------------------------------------------------
    def projected_hvp_error(self, ref, pred):
        """Eq. 6 per labelled graph, weighted by weight * hessian_weight, averaged over
        the labelled graphs; 0 (with a graph) when the batch has no Label."""
        forces = pred["forces"]
        positions = _field(ref, "positions")
        probes, per_graph = self.make_probes(ref, forces)
        if not per_graph:
            self.last_terms.update(hessian=0.0, n_labelled=0)
            return 0.0 * forces.sum()
        hv = hvp_mod.hvp_from_forces(forces, positions, probes, create_graph=torch.is_grad_enabled())
        w_cfg = _field(ref, "weight")
        w_h = _field(ref, "hessian_weight") if _has(ref, "hessian_weight") else torch.ones_like(w_cfg)
        terms = []
        for g, sl, k_g, _vt, r, c, den in per_graph:
            p_w, inv_m = c.torch_constants(forces)
            hv_g = hv[:k_g, sl, :].reshape(k_g, -1)                                  # [k_g, 3n]
            rho = (hv_g * inv_m[None, :]) @ p_w - torch.as_tensor(r, dtype=forces.dtype, device=forces.device)
            terms.append(w_cfg[g] * w_h[g] * (rho * rho).sum() / den)                # n_vib k (stochastic) or n_vib (eq. 10)
        raw = torch.stack(terms)
        self.last_terms.update(hessian=float(raw.detach().mean()), n_labelled=len(terms))
        return raw.mean()

    def projected_hessian_error_full(self, ref, pred):
        """Eq. 1 (eq. 3 with weights) per labelled graph from the model's full Hessian
        `pred["hessian"]` ([3 n_nodes, n_nodes, 3] as mace's `compute_hessians_vmap`
        returns it for a batch, or [3n, 3n] for one graph), same weighting and mean."""
        h_all = pred["hessian"]
        ptr = _field(ref, "ptr").detach().cpu().numpy()
        n_nodes = int(ptr[-1])
        h_all = h_all.reshape(3 * n_nodes, n_nodes, 3)
        w_cfg = _field(ref, "weight")
        w_h = _field(ref, "hessian_weight") if _has(ref, "hessian_weight") else torch.ones_like(w_cfg)
        terms = []
        for g, n, h_r, masses, pos in graph_labels(ref):
            if h_r is None:
                continue
            c = self.constants(h_r, masses, pos)
            a, b = int(ptr[g]), int(ptr[g + 1])
            h_t = h_all[3 * a:3 * b, a:b, :].reshape(3 * n, 3 * n)
            p_w, inv_m = c.torch_constants(h_t)
            d = (h_t - torch.as_tensor(h_r, dtype=h_t.dtype, device=h_t.device)) * (inv_m[:, None] * inv_m[None, :])
            op = p_w @ d @ p_w
            terms.append(w_cfg[g] * w_h[g] * (op * op).sum() / c.n_vib)
        if not terms:
            self.last_terms.update(hessian_exact=0.0, n_labelled=0)
            return 0.0 * h_all.sum()
        raw = torch.stack(terms)
        self.last_terms.update(hessian_exact=float(raw.detach().mean()), n_labelled=len(terms))
        return raw.mean()

    # ---- eq. 11 --------------------------------------------------------------------------
    def forward(self, ref, pred, ddp=None):
        from mace.modules.loss import mean_squared_error_forces, weighted_mean_squared_error_energy
        loss_e = weighted_mean_squared_error_energy(ref, pred, ddp)
        loss_f = mean_squared_error_forces(ref, pred, ddp)
        if "hessian" in pred and pred["hessian"] is not None:
            loss_h = self.projected_hessian_error_full(ref, pred)
        else:
            loss_h = self.projected_hvp_error(ref, pred)
        self.last_terms.update(energy=float(loss_e.detach()), forces=float(loss_f.detach()))
        return self.energy_weight * loss_e + self.forces_weight * loss_f + self.hessian_weight * loss_h

    def __repr__(self):
        return ("{}(energy_weight={:.3f}, forces_weight={:.3f}, hessian_weight={:.3f}, "
                "n_probes={}, probe={!r}, mode_weighting={!r}, seed={!r})").format(
                    self.__class__.__name__, float(self.energy_weight), float(self.forces_weight),
                    float(self.hessian_weight), self.n_probes, self.probe, self.mode_weighting, self.seed)


def build(args):
    """The factory the fork's `--loss external --loss_module openqha.training.phl_loss:build`
    calls with mace's parsed arguments (ticket 13's flags; every one has a default)."""
    return WeightedEnergyForcesHessianLoss(
        energy_weight=getattr(args, "energy_weight", 1.0),
        forces_weight=getattr(args, "forces_weight", 1.0),
        hessian_weight=getattr(args, "hessian_weight", 1.0),
        n_probes=getattr(args, "n_hessian_probes", 4),
        probe=getattr(args, "hessian_probe", "rademacher"),
        mode_weighting=getattr(args, "hessian_mode_weighting", "entropy"),
        seed=getattr(args, "seed", None),
    )
