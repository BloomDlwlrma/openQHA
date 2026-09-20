"""Hessian learning: the loss side of fine-tuning MACE-OFF23 on reference E-F-H Labels.

PRODUCTION. What lives here decides the fine-tuned potential's numbers.

The pattern is mace-md's (`source-code/mace-md-master`): everything that can be written
against mace's PUBLIC API lives in this package -- the Hessian-vector product (`hvp`),
the probes, mode weights and projected loss (`phl`, `phl_loss`, tickets 11), the judge
(`judge`, ticket 14) -- and the little that must touch mace's internals (a Hessian field
on the batch, an external-loss hook) lives as two generic commits on the fork
`BloomDlwlrma/openQHA-Hessian`, branch `openqha-hessian`, whose commit every training
Record carries (`engine.mace_fork_info`). Nothing in `openqha.extensions` is involved:
that subpackage is for optional cross-checks no production number depends on, and the
fine-tuned potential is a production number.

Derivations and the numbers every test here is checked against:
`docs/tutorials/T03_openQHA_Theory_Projected_Hessian_Loss.ipynb`, the long form of
`.scratch/hessian-learning-set/design-phl-loss.md`.
"""
