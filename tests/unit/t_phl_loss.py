"""Ticket 11 of the Hessian-learning set: the loss module (`openqha.training.phl_loss`)
on a toy potential and a stand-in batch carrying ticket 12's fields -- no engine.

Two toy potentials (T03's pair MLP + confinement) with different parameters play the
model (theta) and the reference; H_r is the reference toy's exact Hessian. Asserted:
`probe = modes` through the HVP path equals the full-matrix path and the numpy exact
loss to 1e-12 (eq. 10 on the loop); A4: autograd dL/dtheta equals a central finite
difference to 1e-6; a two-molecule batch equals the mean of the single-molecule
losses to 1e-13; a third, unlabelled frame in the batch changes nothing; a batch with
no Label gives 0 and still backpropagates; A5 (H_r := H_theta -> 0, zero gradient) and
A6 (H_r := 0.81 H_theta -> the closed form, non-zero gradient) through the module;
a batch whose `hessian` field has the wrong length is refused; `build(args)` returns
the class with the flags and `repr` names them; the Rademacher estimator with k = 4
over 400 seeds has its mean within 3 sigma of the exact loss and the variance eq. 8 gives.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def _repo_root():
    for _p in Path(__file__).resolve().parents:
        if (_p / "openqha" / "__init__.py").is_file():
            return _p
    raise RuntimeError("openQHA package not found above " + __file__)


ROOT = _repo_root()
sys.path.insert(0, str(ROOT))
from openqha.training import phl, phl_loss                      # noqa: E402

FAIL = []


def check(label, ok, detail=""):
    print("  {:78s} {}".format(label, "ok" if ok else "FAIL " + str(detail)[:200]))
    if not ok:
        FAIL.append(label)


class ToyPotential(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.pair = torch.nn.Sequential(torch.nn.Linear(1, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))
        self.conf = torch.nn.Parameter(torch.tensor(0.05 + 0.01 * seed))

    def forward(self, x, batch=None):
        n = x.shape[0]
        idx = torch.zeros(n, dtype=torch.long) if batch is None else batch
        same = (idx[:, None] == idx[None, :]).triu(1)
        diff = x[:, None, :] - x[None, :, :]
        d = torch.sqrt((diff ** 2).sum(-1)[same] + 1e-12).unsqueeze(-1)
        e_pair = self.pair(d).sum()
        com = x.mean(0, keepdim=True) if batch is None else torch.stack([x[idx == g].mean(0) for g in idx.unique()])[idx]
        return e_pair + self.conf * ((x - com) ** 2).sum()

    def energies(self, x, batch):
        return torch.stack([self(x[batch == g]) for g in batch.unique()])


class FakeBatch(dict):
    """mace's Batch answers both ref["x"] and ref.x; so does this."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def hessian_of(model, x):
    return torch.autograd.functional.hessian(lambda p: model(p), x.detach()).reshape(3 * x.shape[0], -1).detach().numpy()


def make_batch(frames, model):
    """frames: list of (positions [n,3] numpy, masses [n], H_r [3n,3n] or None). Returns
    (ref, pred) with pred built from `model` with the force graph kept."""
    pos = np.vstack([f[0] for f in frames])
    ptr = np.cumsum([0] + [len(f[0]) for f in frames])
    batch = np.concatenate([np.full(len(f[0]), g) for g, f in enumerate(frames)])
    x = torch.tensor(pos, requires_grad=True)
    bt = torch.tensor(batch)
    E = model.energies(x, bt)
    F = -torch.autograd.grad(E.sum(), x, create_graph=True)[0]
    ref = FakeBatch(
        ptr=torch.tensor(ptr), batch=bt, positions=x,
        weight=torch.ones(len(frames)), energy_weight=torch.ones(len(frames)),
        forces_weight=torch.ones(len(frames)), hessian_weight=torch.ones(len(frames)),
        energy=E.detach() + 0.1, forces=F.detach() * 0.9,
        hessian=torch.tensor(np.concatenate([f[2].reshape(-1) for f in frames if f[2] is not None] or [np.zeros(0)])),
        has_hessian=torch.tensor([f[2] is not None for f in frames]),
        sqrt_masses=torch.tensor(np.sqrt(np.concatenate([f[1] for f in frames]))),
    )
    pred = dict(energy=E, forces=F)
    return ref, pred


def full_hessian_pred(model, ref, pred):
    """pred with the full batch Hessian in mace's [3 n_nodes, n_nodes, 3] layout."""
    x = ref["positions"]
    H = torch.autograd.functional.hessian(lambda p: model.energies(p, ref["batch"]).sum(), x.detach())
    n = x.shape[0]
    return dict(pred, hessian=H.reshape(3 * n, n, 3))


def main():
    torch.set_default_dtype(torch.float64)
    rng = np.random.default_rng(3)
    toy = ToyPotential(0)                     # the model
    ref_toy = ToyPotential(5)                 # the reference level
    n_a, n_b = 5, 4
    xa = rng.standard_normal((n_a, 3)) * 1.2
    xb = rng.standard_normal((n_b, 3)) * 1.2
    ma = rng.uniform(1.0, 16.0, n_a)
    mb = rng.uniform(1.0, 16.0, n_b)
    Ha = hessian_of(ref_toy, torch.tensor(xa))
    Hb = hessian_of(ref_toy, torch.tensor(xb))
    Ha_t = hessian_of(toy, torch.tensor(xa))
    Hb_t = hessian_of(toy, torch.tensor(xb))
    exact_a = phl.projected_loss_full(Ha_t, Ha, ma, xa)
    exact_b = phl.projected_loss_full(Hb_t, Hb, mb, xb)

    # --- probe = modes on the loop = the full-matrix path = the numpy exact loss ----------
    loss = phl_loss.WeightedEnergyForcesHessianLoss(probe="modes", mode_weighting="none")
    ref, pred = make_batch([(xa, ma, Ha)], toy)
    est = loss.projected_hvp_error(ref, pred)
    full = loss.projected_hessian_error_full(ref, full_hessian_pred(toy, ref, pred))
    check("probe=modes through the HVP path = numpy exact loss (1e-12)", abs(float(est) - exact_a) < 1e-12, (float(est), exact_a))
    check("full-matrix path = numpy exact loss (1e-12)", abs(float(full) - exact_a) < 1e-12, (float(full), exact_a))
    check("the HVP-path term carries a graph; the full-matrix one (detached input) does not need one",
          est.grad_fn is not None)
    total = loss(ref, pred)
    check("forward = w_E L_E + w_F L_F + w_H L_H with mace's E/F terms; last_terms filled",
          abs(float(total) - (loss.last_terms["energy"] + loss.last_terms["forces"] + loss.last_terms["hessian"])) < 1e-12
          and loss.last_terms["n_labelled"] == 1, loss.last_terms)
    pred_h = full_hessian_pred(toy, ref, pred)
    loss(ref, pred_h)
    check("forward with pred['hessian'] takes the exact term (hessian_exact set)",
          loss.last_terms["hessian_exact"] is not None and abs(loss.last_terms["hessian_exact"] - exact_a) < 1e-12)

    # --- A4: gradient by autograd = finite difference ---------------------------------------
    ref, pred = make_batch([(xa, ma, Ha)], toy)
    L = loss.projected_hvp_error(ref, pred)
    params = list(toy.parameters())
    g = torch.autograd.grad(L, params, allow_unused=True)
    names = [n for n, _ in toy.named_parameters()]
    j0 = names.index("pair.0.weight")          # own parameters come first in parameters(): conf, then the MLP
    p0 = params[j0]                            # first Linear weight [16, 1]
    i = (3, 0); h = 1e-5
    with torch.no_grad():
        orig = float(p0[i])
    def loss_at(val):
        with torch.no_grad():
            p0[i] = val
        r_, pr_ = make_batch([(xa, ma, Ha)], toy)
        v = float(loss.projected_hvp_error(r_, pr_).detach())
        with torch.no_grad():
            p0[i] = orig
        return v
    fd = (loss_at(orig + h) - loss_at(orig - h)) / (2 * h)
    check("A4: autograd dL/dtheta = central finite difference (1e-6 relative)", abs(float(g[j0][i]) - fd) < 1e-6 * max(1.0, abs(fd)),
          (float(g[j0][i]), fd))

    # --- batch identity and masking ---------------------------------------------------------------
    ref2, pred2 = make_batch([(xa, ma, Ha), (xb, mb, Hb)], toy)
    two = float(loss.projected_hvp_error(ref2, pred2))
    check("two-molecule batch = mean of the single-molecule losses (1e-13)", abs(two - 0.5 * (exact_a + exact_b)) < 1e-13,
          (two, exact_a, exact_b))
    ref3, pred3 = make_batch([(xa, ma, Ha), (xb, mb, None), (xb * 1.1, mb, Hb)], toy)
    exact_c = phl.projected_loss_full(hessian_of(toy, torch.tensor(xb * 1.1)), Hb, mb, xb * 1.1)
    three = float(loss.projected_hvp_error(ref3, pred3))
    check("an unlabelled frame in the batch contributes nothing (mean over the two labelled, 1e-13)",
          abs(three - 0.5 * (exact_a + exact_c)) < 1e-13 and loss.last_terms["n_labelled"] == 2, (three, exact_a, exact_c))
    ref0, pred0 = make_batch([(xb, mb, None)], toy)
    z = loss.projected_hvp_error(ref0, pred0)
    z.backward()
    check("a batch with no Label: term 0 and backward() runs", float(z) == 0.0 and loss.last_terms["n_labelled"] == 0)
    bad = FakeBatch(ref); bad["hessian"] = ref["hessian"][:-1]
    try:
        loss.projected_hvp_error(bad, pred)
        check("a hessian field of the wrong length is refused", False)
    except ValueError as exc:
        check("a hessian field of the wrong length is refused", "batch.hessian holds" in str(exc))

    # --- A5 / A6 through the module -------------------------------------------------------------------
    ref5, pred5 = make_batch([(xa, ma, Ha_t)], toy)
    L5 = loss.projected_hvp_error(ref5, pred5)
    g5 = torch.autograd.grad(L5, params, allow_unused=True)
    check("A5 must-pass: H_r := H_theta -> 0 and zero gradient (1e-16)",
          float(L5) < 1e-16 and max(float(x.abs().max()) for x in g5 if x is not None) < 1e-10, float(L5))
    ref6, pred6 = make_batch([(xa, ma, 0.81 * Ha_t)], toy)
    L6 = loss.projected_hvp_error(ref6, pred6)
    g6 = torch.autograd.grad(L6, params, allow_unused=True)
    P, _inv, _V = phl.projector(ma, xa)
    K = P @ phl.mass_weighted(Ha_t, ma) @ P
    a6 = 0.19 ** 2 * np.sum(K * K) / phl.reference_modes(Ha_t, ma, xa)[0].shape[1]
    check("A6 must-fail: H_r := 0.81 H_theta -> 0.19^2 ||K~||_F^2/n_vib and a non-zero gradient",
          abs(float(L6) / a6 - 1) < 1e-10 and max(float(x.abs().max()) for x in g6 if x is not None) > 1e-6, (float(L6), a6))

    # --- Rademacher k = 4 on the loop -------------------------------------------------------------------
    var = phl.estimator_variance(Ha_t, Ha, ma, xa, k=4)["rademacher"]
    n_seeds = 400
    vals = []
    for s in range(n_seeds):
        l4 = phl_loss.WeightedEnergyForcesHessianLoss(probe="rademacher", n_probes=4, mode_weighting="none", seed=s)
        r_, pr_ = make_batch([(xa, ma, Ha)], toy)
        vals.append(float(l4.projected_hvp_error(r_, pr_).detach()))
    vals = np.array(vals)
    sig_mean = np.sqrt(var / n_seeds)
    check("Rademacher k=4 over {} seeds: mean within 3 sigma of the exact loss ({:.2f} sigma), sample variance within 25 % of eq. 8".format(
          n_seeds, abs(vals.mean() - exact_a) / sig_mean),
          abs(vals.mean() - exact_a) < 3 * sig_mean and abs(vals.var() / var - 1) < 0.25, (vals.mean(), exact_a, vals.var(), var))

    # --- entropy weighting and build() -----------------------------------------------------------------------
    lw = phl_loss.WeightedEnergyForcesHessianLoss(probe="modes", mode_weighting="entropy")
    ref, pred = make_batch([(xa, ma, Ha)], toy)
    est_w = float(lw.projected_hvp_error(ref, pred))
    c = lw.constants(Ha, ma, xa)
    check("entropy weighting: probe=modes reproduces eq. 3 with the cached weights (1e-12)",
          abs(est_w - phl.projected_loss_full(Ha_t, Ha, ma, xa, weights=c.weights)) < 1e-12 and c.weights.max() == 1.0)
    check("the per-frame constants are cached (one entry after two calls on the same frame)",
          len(lw._cache) == 1)
    args = argparse.Namespace(energy_weight=1.0, forces_weight=100.0, hessian_weight=7.5, n_hessian_probes=2,
                              hessian_probe="gaussian", hessian_mode_weighting="none", seed=11)
    b = phl_loss.build(args)
    check("build(args) returns the class with the flags",
          isinstance(b, phl_loss.WeightedEnergyForcesHessianLoss) and float(b.hessian_weight) == 7.5 and b.n_probes == 2
          and b.probe == "gaussian" and b.mode_weighting == "none" and b.seed == 11 and float(b.forces_weight) == 100.0)
    check("repr names every setting", "hessian_weight=7.500" in repr(b) and "probe='gaussian'" in repr(b) and "seed=11" in repr(b), repr(b))
    check("wants_hessian_at_eval is set (the fork's evaluate reads it)", b.wants_hessian_at_eval is True)
    try:
        phl_loss.WeightedEnergyForcesHessianLoss(probe="hutchinson")
        check("an unknown probe is refused at construction", False)
    except ValueError:
        check("an unknown probe is refused at construction", True)

    print("\n{} checks, {} failed".format(19, len(FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
