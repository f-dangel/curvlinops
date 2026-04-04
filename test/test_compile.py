"""Tests for ``torch.compile`` compatibility.

These tests verify that curvlinops operators can be compiled with
``torch.compile`` without graph breaks, enabling potential speedups from
the ``torch.compile`` compiler.
"""

from torch import compile as torch_compile
from torch import manual_seed, rand
from torch._dynamo import explain
from torch._dynamo import reset as dynamo_reset
from torch.nn import Linear, MSELoss, Sequential
from torch.testing import assert_close

from curvlinops import GGNLinearOperator, HessianLinearOperator
from curvlinops.examples import trace_gradient_and_loss


def _setup_problem():
    """Create a small test problem with two batches of different size.

    Returns:
        Tuple of (model, loss_fn, params, data).
    """
    manual_seed(0)
    model = Sequential(Linear(4, 3), Linear(3, 2))
    loss_fn = MSELoss()
    params = dict(model.named_parameters())
    data = [(rand(2, 4), rand(2, 2)), (rand(3, 4), rand(3, 2))]
    return model, loss_fn, params, data


def _assert_no_graph_breaks(fn, *args):
    """Assert that ``fn(*args)`` compiles with zero graph breaks.

    Also verifies that the compiled result matches eager execution.

    Args:
        fn: Function to compile.
        *args: Arguments to pass to ``fn``.
    """
    dynamo_reset()
    try:
        explanation = explain(fn)(*args)
        assert explanation.graph_break_count == 0, (
            f"Expected 0 graph breaks, got {explanation.graph_break_count}.\n"
            f"Break reasons: {explanation.break_reasons}"
        )

        r_eager = fn(*args)
        r_compiled = torch_compile(fn)(*args)
        assert_close(r_eager, r_compiled, atol=1e-5, rtol=1e-5)
    finally:
        dynamo_reset()


def test_gradient_and_loss_no_graph_breaks():
    """Per-batch gradient+loss traced with ``make_fx`` compiles with 0 graph breaks."""
    model, loss_fn, params, data = _setup_problem()
    X, y = data[0]
    traced = trace_gradient_and_loss(model, loss_fn, params, X, y)
    _assert_no_graph_breaks(traced, params, X, y)


def test_hessian_matvec_no_graph_breaks():
    """``HessianLinearOperator @ v`` compiles with zero graph breaks."""
    model, loss_fn, params, data = _setup_problem()
    H = HessianLinearOperator(model, loss_fn, params, data, check_deterministic=False)
    v = rand(H.shape[1])
    _assert_no_graph_breaks(lambda op, vec: op @ vec, H, v)


def test_ggn_matvec_no_graph_breaks():
    """``GGNLinearOperator @ v`` (exact) compiles with zero graph breaks."""
    model, loss_fn, params, data = _setup_problem()
    G = GGNLinearOperator(model, loss_fn, params, data, check_deterministic=False)
    v = rand(G.shape[1])
    _assert_no_graph_breaks(lambda op, vec: op @ vec, G, v)


def test_ggn_mc_matvec_compiles_correctly():
    """``GGNLinearOperator @ v`` (MC) compiles correctly.

    ``fork_rng`` in ``_matmat`` causes graph breaks (context manager not
    traceable by dynamo), but the per-batch computation is still compiled.
    """
    model, loss_fn, params, data = _setup_problem()
    G = GGNLinearOperator(
        model, loss_fn, params, data, check_deterministic=False, mc_samples=1
    )
    v = rand(G.shape[1])

    dynamo_reset()
    try:
        explanation = explain(lambda op, vec: op @ vec)(G, v)
        # Graph breaks are from fork_rng (context manager), not the computation
        assert all("fork_rng" in str(br.reason) for br in explanation.break_reasons)

        r_eager = G @ v
        r_compiled = torch_compile(lambda op, vec: op @ vec)(G, v)
        assert_close(r_eager, r_compiled, atol=1e-5, rtol=1e-5)
    finally:
        dynamo_reset()
