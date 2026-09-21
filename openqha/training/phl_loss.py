"""The Hessian loss as a mace loss module: eq. 11 with the HVP estimator (eq. 6 / 6')
in training and the SAME estimator on four probes fixed per frame in evaluation.

PRODUCTION. Ticket 11 of the Hessian-learning set; the Cartesian target and the
fixed-probe validation of ticket 21 (S0-C-53, S0-C-55; 2026-09-21).

This is the `nn.Module` mace's `train()` calls as `loss_fn(pred=output, ref=batch)`. It
reaches mace through the fork's external-loss hook (ticket 13: `--loss external
--loss_module openqha.training.phl_loss:build`); the E and F terms are mace's own
(`mace.modules.loss`, public), the Hessian term is here. The batch fields it reads are
the ones the fork's commit A adds (ticket 12): `hessian` (the Labels, flattened 9 n_k^2
per graph and concatenated like `ptr`), `has_hessian` [n_graphs], `hessian_weight`
[n_graphs], `sqrt_masses` [n_nodes]; and mace's `ptr`, `weight`, `positions`.

THE TARGET (`mode_weighting`). `cartesian` -- the default since S0-C-53 -- is the raw
matrix, PHL's eq. 2.1': rho_j = H_theta v_j - H_r v_j over 9 N^2 k, no mass weighting,
no Eckart projection, no reference modes. `entropy` and `none` are the projected
quantities of T03 (P_W M^-1/2 on both sides, eqs. 1-6), kept as diagnostics.

TRAINING. The HVP is taken INSIDE the loss from `pred["forces"]` and `ref["positions"]`
(`hvp.hvp_from_forces`, create_graph=True) -- one backward pass per probe index for the
whole batch, since a batch is block-diagonal. The probes are fresh draws from the
module's generator (mace's seed) every step.

EVALUATION (S0-C-55). The fork's `evaluate` puts the loss in eval mode (commit C) and,
because `wants_force_graph_at_eval` is set, calls the model with the force graph kept;
the Hessian term is then the same estimator on `VALID_N_PROBES` Rademacher probes whose
generator is seeded from the frame's Label bytes -- identical every epoch for a frame,
different between frames -- so the validation curve is a fixed, cheap, unbiased reading
of the target (about 4 HVPs per labelled frame against 3N for the full matrix). That
term enters the total validation loss mace's scheduler, checkpoint and Stage Two read.
Over a validation pass the three terms are accumulated and handed back through
`eval_summary()` (commit C), which joins them to mace's `results/*.txt` as
`valid_energy_term`, `valid_forces_term`, `valid_hessian_term`. `wants_hessian_at_eval`
is False: the fork's full-matrix hook stays available but unused; the full matrix is
the judge's tool (`hessian_error_full` remains for it and for the tests).

Per-structure constants (the projector, M^-1/2, L_r, the entropy weights on the
projected path; the frame's seed and its fixed probes on every path) depend only on the
Label, the masses and the positions, so they are computed once per frame and cached by
the Label's bytes.
"""
import hashlib
import logging

import numpy as np
import torch

from ..thermochem import hessian as hessian_mod
from . import hvp as hvp_mod
from . import phl

#: `cartesian` (the default, eq. 1'), `entropy` (eq. 3 with |dS/d omega| weights), `none` (eq. 1)
MODE_WEIGHTINGS = ("cartesian", "entropy", "none")
DEFAULT_MODE_WEIGHTING = "cartesian"
#: the validation estimator (S0-C-55): k fixed Rademacher probes per frame
VALID_PROBE = "rademacher"
VALID_N_PROBES = 4
VALID_PROBES_LABEL = "{} k={} fixed".format(VALID_PROBE, VALID_N_PROBES)


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


def frame_seed(hessian_r):
    """The frame's own probe seed: the first 8 bytes of SHA-1 over the Label's bytes.
    The same Label gives the same seed anywhere; two frames with different Labels differ."""
    digest = hashlib.sha1(np.ascontiguousarray(np.asarray(hessian_r, dtype=float)).tobytes()).digest()
    return int.from_bytes(digest[:8], "little")


class FrameConstants:
    """What Algorithm 1 needs of one frame, computed once: on the projected path the
    reference modes, the (weighted) projector, M^-1/2, n_vib; on the Cartesian path the
    identity and ones (nothing diagonalised); on both the frame's seed and its fixed
    validation probes -- numpy for the probes, torch for the loss."""

    __slots__ = ("hessian_r", "masses", "positions", "metric", "modes_r", "lam_r", "weights", "projector",
                 "inv_sqrt_m", "n_vib", "seed", "projector_t", "inv_sqrt_m_t", "_fixed")

    def __init__(self, hessian_r, masses, positions, mode_weighting, temperature_K, preset):
        self.hessian_r = np.asarray(hessian_r, dtype=float)
        self.masses = np.asarray(masses, dtype=float)
        self.positions = np.asarray(positions, dtype=float)
        n3 = 3 * len(self.masses)
        self.metric = "cartesian" if mode_weighting == "cartesian" else "projected"
        if self.metric == "cartesian":
            self.modes_r, self.lam_r, self.weights = None, None, None
            self.projector = np.eye(n3)
            self.inv_sqrt_m = np.ones(n3)
            self.n_vib = int(n3)
        else:
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
        self.seed = frame_seed(self.hessian_r)
        self.projector_t = None
        self.inv_sqrt_m_t = None
        self._fixed = {}

    def torch_constants(self, like):
        """P_W and M^-1/2 (or I and ones) as tensors of `like`'s dtype and device (made once)."""
        if self.projector_t is None or self.projector_t.dtype != like.dtype or self.projector_t.device != like.device:
            self.projector_t = torch.as_tensor(self.projector, dtype=like.dtype, device=like.device)
            self.inv_sqrt_m_t = torch.as_tensor(self.inv_sqrt_m, dtype=like.dtype, device=like.device)
        return self.projector_t, self.inv_sqrt_m_t

    def probes(self, mode, k, rng=None):
        """Algorithm 1's probes for this frame. With `rng` (training) a fresh draw; without
        (evaluation) the FIXED set from the frame's own seed, cached -- the same k vectors
        on every call, every epoch (S0-C-55)."""
        if rng is not None:
            return phl.make_probes(self.masses, self.positions, self.hessian_r, mode=mode, k=k,
                                   weights=self.weights, rng=rng, metric=self.metric)
        key = (mode, int(k))
        if key not in self._fixed:
            self._fixed[key] = phl.make_probes(self.masses, self.positions, self.hessian_r, mode=mode, k=k,
                                               weights=self.weights, rng=np.random.default_rng(self.seed),
                                               metric=self.metric)
        return self._fixed[key]


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
    Hessian term -- the estimator (eq. 6 / 6') from the forces' graph when `pred` has no
    "hessian" (training: fresh probes; evaluation: the frame's fixed probes), the exact
    full-matrix value when it has (the judge's path)."""

    #: the fork's `evaluate` reads these: no full Hessian at evaluation (S0-C-55), but
    #: the force graph kept so the estimator can take its HVPs there
    wants_hessian_at_eval = False
    wants_force_graph_at_eval = True

    def __init__(self, energy_weight=1.0, forces_weight=1.0, hessian_weight=1.0, n_probes=4,
                 probe="rademacher", mode_weighting=DEFAULT_MODE_WEIGHTING, seed=None, temperature_K=298.15,
                 preset="crest", cache_size=4096, valid_probe=VALID_PROBE, valid_n_probes=VALID_N_PROBES):
        super().__init__()
        if probe not in phl.PROBE_MODES:
            raise ValueError("probe must be one of {}; got {!r}".format(phl.PROBE_MODES, probe))
        if mode_weighting not in MODE_WEIGHTINGS:
            raise ValueError("mode_weighting must be one of {}; got {!r}".format(MODE_WEIGHTINGS, mode_weighting))
        if mode_weighting == "cartesian" and probe == "modes":
            raise ValueError("probe 'modes' needs reference modes; the cartesian target has none (use 'cartesian')")
        dt = torch.get_default_dtype()
        self.register_buffer("energy_weight", torch.tensor(float(energy_weight), dtype=dt))
        self.register_buffer("forces_weight", torch.tensor(float(forces_weight), dtype=dt))
        self.register_buffer("hessian_weight", torch.tensor(float(hessian_weight), dtype=dt))
        self.n_probes = int(n_probes)
        self.probe = probe
        self.mode_weighting = mode_weighting
        self.metric = "cartesian" if mode_weighting == "cartesian" else "projected"
        self.seed = seed
        self.temperature_K = float(temperature_K)
        self.preset = preset
        self.valid_probe = valid_probe
        self.valid_n_probes = int(valid_n_probes)
        self.rng = np.random.default_rng(seed)
        self._cache = {}
        self._cache_size = int(cache_size)
        # the last values of the three terms, for the training log
        self.last_terms = dict(energy=None, forces=None, hessian=None, hessian_exact=None, n_labelled=0)
        self._eval_sums = None
        self._reset_eval_sums()

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
        (g, slice, k_g, v~ [k_g, 3n], r_j [k_g, 3n] numpy, constants, denominator).
        In training mode the probes are fresh draws (`self.probe`, `self.n_probes`); in
        eval mode the frame's fixed validation set (`valid_probe`, `valid_n_probes`)."""
        labels = graph_labels(ref)
        ptr = _field(ref, "ptr").detach().cpu().numpy()
        n_nodes = int(ptr[-1])
        per_graph = []
        k_max = 0
        for g, n, h_r, masses, pos in labels:
            if h_r is None:
                continue
            c = self.constants(h_r, masses, pos)
            if self.training:
                vt, r, info = c.probes(self.probe, self.n_probes, rng=self.rng)
            else:
                vt, r, info = c.probes(self.valid_probe, self.valid_n_probes)
            per_graph.append((g, slice(int(ptr[g]), int(ptr[g + 1])), info["k"], vt, r, c, info["denominator"]))
            k_max = max(k_max, info["k"])
        probes = torch.zeros((k_max, n_nodes, 3), dtype=like.dtype, device=like.device)
        for g, sl, k_g, vt, _r, _c, _den in per_graph:
            probes[:k_g, sl, :] = torch.as_tensor(vt.reshape(k_g, -1, 3), dtype=like.dtype, device=like.device)
        return probes, per_graph

    # ---- the two Hessian terms -----------------------------------------------------------
    def hvp_error(self, ref, pred):
        """Eq. 6 (eq. 6' on the Cartesian path) per labelled graph, weighted by
        weight * hessian_weight, averaged over the labelled graphs; 0 (with a graph)
        when the batch has no Label."""
        forces = pred["forces"]
        positions = _field(ref, "positions")
        probes, per_graph = self.make_probes(ref, forces)
        if not per_graph:
            self.last_terms.update(hessian=0.0, n_labelled=0)
            return 0.0 * forces.sum()
        if forces.grad_fn is None:
            raise RuntimeError(
                "the forces carry no graph, so no Hessian-vector product can be taken: the model must be "
                "called with training=True (the fork's evaluate does so when the loss sets "
                "wants_force_graph_at_eval, commit C)")
        # the third-order graph (to the parameters) only in training; a validation HVP is a value.
        # torchmetrics runs a metric's update under no_grad (mace's evaluate wraps the loss in
        # one), so grad is re-enabled here: the force graph exists, it only has to be walked
        with torch.enable_grad():
            hv = hvp_mod.hvp_from_forces(forces, positions, probes, create_graph=bool(self.training))
        if not self.training:
            hv = hv.detach()
        w_cfg = _field(ref, "weight")
        w_h = _field(ref, "hessian_weight") if _has(ref, "hessian_weight") else torch.ones_like(w_cfg)
        terms = []
        for g, sl, k_g, _vt, r, c, den in per_graph:
            p_w, inv_m = c.torch_constants(forces)
            hv_g = hv[:k_g, sl, :].reshape(k_g, -1)                                  # [k_g, 3n]
            if c.metric == "cartesian":
                rho = hv_g - torch.as_tensor(r, dtype=forces.dtype, device=forces.device)
            else:
                rho = (hv_g * inv_m[None, :]) @ p_w - torch.as_tensor(r, dtype=forces.dtype, device=forces.device)
            terms.append(w_cfg[g] * w_h[g] * (rho * rho).sum() / den)                # n_vib k / 9N^2 k (stochastic) or n_vib / 9N^2
        raw = torch.stack(terms)
        self.last_terms.update(hessian=float(raw.detach().mean()), n_labelled=len(terms))
        return raw.mean()

    projected_hvp_error = hvp_error          # the name of ticket 11; the projected path is one of its two

    def hessian_error_full(self, ref, pred):
        """Eq. 1 (eq. 3 with weights; eq. 1' on the Cartesian path) per labelled graph
        from the model's full Hessian `pred["hessian"]` ([3 n_nodes, n_nodes, 3] as
        mace's `compute_hessians_vmap` returns it for a batch, or [3n, 3n] for one
        graph), same weighting and mean."""
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
            d = h_t - torch.as_tensor(h_r, dtype=h_t.dtype, device=h_t.device)
            if c.metric == "cartesian":
                terms.append(w_cfg[g] * w_h[g] * (d * d).sum() / (c.n_vib * c.n_vib))
            else:
                p_w, inv_m = c.torch_constants(h_t)
                op = p_w @ (d * (inv_m[:, None] * inv_m[None, :])) @ p_w
                terms.append(w_cfg[g] * w_h[g] * (op * op).sum() / c.n_vib)
        if not terms:
            self.last_terms.update(hessian_exact=0.0, n_labelled=0)
            return 0.0 * h_all.sum()
        raw = torch.stack(terms)
        self.last_terms.update(hessian_exact=float(raw.detach().mean()), n_labelled=len(terms))
        return raw.mean()

    projected_hessian_error_full = hessian_error_full

    # ---- the validation pass ---------------------------------------------------------------
    def _reset_eval_sums(self):
        self._eval_sums = dict(energy=0.0, forces=0.0, hessian=0.0, n_graphs=0, n_labelled=0, n_batches=0,
                               last_batch=None)

    def _accumulate(self, ref, loss_e, loss_f, hessian_mean, n_labelled):
        s = self._eval_sums
        # torchmetrics' full-state update calls the loss twice on the same batch object (the
        # global and the per-batch state); the second call carries the same numbers and
        # would only double the counts, so a batch is accumulated once
        if s["last_batch"] is not None and s["last_batch"] is ref:
            return
        s["last_batch"] = ref
        n_graphs = int(_field(ref, "ptr").numel() - 1)
        s["energy"] += float(loss_e.detach()) * n_graphs
        s["forces"] += float(loss_f.detach()) * n_graphs
        s["hessian"] += float(hessian_mean) * n_labelled
        s["n_graphs"] += n_graphs
        s["n_labelled"] += n_labelled
        s["n_batches"] += 1

    def eval_summary(self):
        """The three terms averaged over the validation pass (E and F per graph, the
        Hessian per labelled graph), as the fork's `evaluate` merges them into the
        metrics it logs; resets the accumulators. Called once per validation pass."""
        s = self._eval_sums
        out = dict(valid_energy_term=(s["energy"] / s["n_graphs"]) if s["n_graphs"] else None,
                   valid_forces_term=(s["forces"] / s["n_graphs"]) if s["n_graphs"] else None,
                   valid_hessian_term=(s["hessian"] / s["n_labelled"]) if s["n_labelled"] else None,
                   valid_hessian_n_labelled=int(s["n_labelled"]),
                   valid_probes="{} k={} fixed".format(self.valid_probe, self.valid_n_probes),
                   valid_target=self.mode_weighting)
        if s["n_batches"]:
            logging.info("openQHA loss (valid): energy=%s forces=%s hessian=%s n_labelled=%d probes=%s target=%s",
                         "-" if out["valid_energy_term"] is None else "{:.6e}".format(out["valid_energy_term"]),
                         "-" if out["valid_forces_term"] is None else "{:.6e}".format(out["valid_forces_term"]),
                         "-" if out["valid_hessian_term"] is None else "{:.6e}".format(out["valid_hessian_term"]),
                         out["valid_hessian_n_labelled"], out["valid_probes"], self.mode_weighting)
        self._reset_eval_sums()
        return out

    # ---- eq. 11 --------------------------------------------------------------------------
    def forward(self, ref, pred, ddp=None):
        from mace.modules.loss import mean_squared_error_forces, weighted_mean_squared_error_energy
        loss_e = weighted_mean_squared_error_energy(ref, pred, ddp)
        loss_f = mean_squared_error_forces(ref, pred, ddp)
        if "hessian" in pred and pred["hessian"] is not None:
            loss_h = self.hessian_error_full(ref, pred)
            h_value = self.last_terms["hessian_exact"]
        else:
            loss_h = self.hvp_error(ref, pred)
            h_value = self.last_terms["hessian"]
        self.last_terms.update(energy=float(loss_e.detach()), forces=float(loss_f.detach()))
        if not self.training:
            self._accumulate(ref, loss_e, loss_f, h_value or 0.0, self.last_terms["n_labelled"])
        return self.energy_weight * loss_e + self.forces_weight * loss_f + self.hessian_weight * loss_h

    def __repr__(self):
        return ("{}(energy_weight={:.3f}, forces_weight={:.3f}, hessian_weight={:.3f}, "
                "n_probes={}, probe={!r}, mode_weighting={!r}, seed={!r}, valid_probes={!r})").format(
                    self.__class__.__name__, float(self.energy_weight), float(self.forces_weight),
                    float(self.hessian_weight), self.n_probes, self.probe, self.mode_weighting, self.seed,
                    "{} k={} fixed".format(self.valid_probe, self.valid_n_probes))


def build(args):
    """The factory the fork's `--loss external --loss_module openqha.training.phl_loss:build`
    calls with mace's parsed arguments (ticket 13's flags; every one has a default; the
    target defaults to `cartesian`, S0-C-53)."""
    return WeightedEnergyForcesHessianLoss(
        energy_weight=getattr(args, "energy_weight", 1.0),
        forces_weight=getattr(args, "forces_weight", 1.0),
        hessian_weight=getattr(args, "hessian_weight", 1.0),
        n_probes=getattr(args, "n_hessian_probes", 4),
        probe=getattr(args, "hessian_probe", "rademacher"),
        mode_weighting=getattr(args, "hessian_mode_weighting", DEFAULT_MODE_WEIGHTING),
        seed=getattr(args, "seed", None),
    )
