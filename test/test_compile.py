"""Tests for ``torch.compile`` compatibility.

These tests document the current state of ``torch.compile`` support and serve as
a starting point for making curvlinops operators compile-friendly.
"""

import torch
from torch import manual_seed, rand
from torch.nn import Linear, MSELoss, Sequential

from curvlinops import HessianLinearOperator


def _setup_hessian():
    """Create a small HessianLinearOperator for testing.

    Returns:
        Tuple of (operator, random_vector).
    """
    manual_seed(0)
    model = Sequential(Linear(4, 3), Linear(3, 2))
    loss_fn = MSELoss()
    params = dict(model.named_parameters())
    X = rand(2, 4)
    y = rand(2, 2)
    data = [(X, y)]
    H = HessianLinearOperator(model, loss_fn, params, data, check_deterministic=False)
    v = rand(H.shape[1])
    return H, v


def test_hessian_matvec_has_graph_breaks():
    """Compiling ``HessianLinearOperator @ v`` produces graph breaks.

    ``torch.compile`` encounters graph breaks (``cached_property``'s ``RLock``,
    ``vmap`` + compile incompatibility) and silently falls back to eager
    execution. The result is correct but there is no speedup.

    This test documents the current incompatibility. When ``torch.compile``
    support is added, this test should be updated to assert zero graph breaks.
    """
    H, v = _setup_hessian()

    def matvec(op, vec):
        return op @ vec

    torch._dynamo.reset()
    try:
        explanation = torch._dynamo.explain(matvec)(H, v)

        # torch.compile cannot fully trace the operator (e.g. due to
        # cached_property's RLock, vmap interaction, or other internals)
        assert explanation.graph_break_count > 0

        # Despite graph breaks, the result is correct (falls back to eager)
        r_eager = H @ v
        r_compiled = torch.compile(matvec)(H, v)
        assert torch.allclose(r_eager, r_compiled, atol=1e-5)
    finally:
        torch._dynamo.reset()
