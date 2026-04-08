"""Tests for ``torch.compile`` compatibility.

These tests verify that curvlinops operators can be compiled with
``torch.compile`` without graph breaks, enabling potential speedups from
the ``torch.compile`` compiler.

Note: We use ``torch._dynamo.explain`` rather than ``torch.compile(fullgraph=True)``
because ``fullgraph=True`` cannot proxy user-defined types (like our operators) at the
graph boundary. ``explain`` correctly traces *into* the operator's ``__matmul__`` and
confirms that all internal tensor ops are captured in a single graph without breaks.
"""

from contextlib import contextmanager

from pytest import mark
from torch import compile as torch_compile
from torch import manual_seed, rand
from torch._dynamo import explain
from torch._dynamo import reset as dynamo_reset
from torch.nn import Conv2d, Flatten, Linear, MSELoss, Sequential
from torch.random import fork_rng
from torch.testing import assert_close

from curvlinops import (
    EFLinearOperator,
    EKFACLinearOperator,
    GGNLinearOperator,
    HessianLinearOperator,
    KFACLinearOperator,
)
from curvlinops.computers.kfac_make_fx import make_compute_kfac_batch


def _setup_mlp_problem():
    """Create a small MLP test problem with two batches of different size.

    Returns:
        Tuple of (model, loss_fn, params, data).
    """
    manual_seed(0)
    model = Sequential(Linear(4, 3), Linear(3, 2))
    loss_fn = MSELoss()
    params = dict(model.named_parameters())
    data = [(rand(2, 4), rand(2, 2)), (rand(3, 4), rand(3, 2))]
    return model, loss_fn, params, data


def _setup_cnn_problem():
    """Create a small CNN test problem (Conv2d triggers non-contiguous tensors).

    Returns:
        Tuple of (model, loss_fn, params, data).
    """
    manual_seed(0)
    model = Sequential(
        Conv2d(3, 4, 3, padding=1), Conv2d(4, 5, 3, padding=1), Flatten(), Linear(180, 3)
    )
    loss_fn = MSELoss()
    params = dict(model.named_parameters())
    X = rand(2, 3, 6, 6)
    y = rand(2, 3)
    return model, loss_fn, params, [(X, y)]


SETUPS = [_setup_mlp_problem, _setup_cnn_problem]
SETUP_IDS = ["mlp", "cnn"]


@contextmanager
def _dynamo_explain(fn, *args):
    """Run ``torch._dynamo.explain`` on ``fn`` and yield the explanation.

    Resets dynamo state before and after. Also verifies that the compiled
    result matches eager execution.

    Args:
        fn: Function to explain/compile.
        *args: Arguments to pass to ``fn``.

    Yields:
        The ``ExplainOutput`` from ``torch._dynamo.explain``.
    """
    dynamo_reset()
    try:
        yield explain(fn)(*args)

        r_eager = fn(*args)
        r_compiled = torch_compile(fn)(*args)
        assert_close(r_eager, r_compiled, atol=1e-5, rtol=1e-5)
    finally:
        dynamo_reset()


def _assert_no_graph_breaks(linop):
    """Assert that ``linop @ v`` compiles with zero graph breaks."""
    v = rand(linop.shape[1])
    with _dynamo_explain(lambda op, vec: op @ vec, linop, v) as result:
        assert result.graph_break_count == 0


@mark.parametrize("setup_fn", SETUPS, ids=SETUP_IDS)
def test_hessian_matvec_no_graph_breaks(setup_fn):
    """``HessianLinearOperator @ v`` compiles with zero graph breaks."""
    model, loss_fn, params, data = setup_fn()
    H = HessianLinearOperator(model, loss_fn, params, data, check_deterministic=False)
    _assert_no_graph_breaks(H)


@mark.parametrize("setup_fn", SETUPS, ids=SETUP_IDS)
def test_ggn_matvec_no_graph_breaks(setup_fn):
    """``GGNLinearOperator @ v`` (exact) compiles with zero graph breaks."""
    model, loss_fn, params, data = setup_fn()
    G = GGNLinearOperator(model, loss_fn, params, data, check_deterministic=False)
    _assert_no_graph_breaks(G)


@mark.parametrize("setup_fn", SETUPS, ids=SETUP_IDS)
def test_ggn_mc_matvec_compiles_correctly(setup_fn):
    """``GGNLinearOperator @ v`` (MC) compiles correctly.

    ``fork_rng`` in ``_matmat`` causes graph breaks (context manager not
    traceable by dynamo), but the per-batch computation is still compiled.
    """
    model, loss_fn, params, data = setup_fn()
    G = GGNLinearOperator(
        model, loss_fn, params, data, check_deterministic=False, mc_samples=1
    )
    v = rand(G.shape[1])
    with _dynamo_explain(lambda op, vec: op @ vec, G, v) as result:
        assert all("fork_rng" in str(br.reason) for br in result.break_reasons)


@mark.parametrize("setup_fn", SETUPS, ids=SETUP_IDS)
def test_ef_matvec_no_graph_breaks(setup_fn):
    """``EFLinearOperator @ v`` compiles with zero graph breaks."""
    model, loss_fn, params, data = setup_fn()
    E = EFLinearOperator(model, loss_fn, params, data, check_deterministic=False)
    _assert_no_graph_breaks(E)


KFAC_LIKE_CLS = [KFACLinearOperator, EKFACLinearOperator]
BACKENDS = ["hooks", "make_fx"]


@mark.parametrize("setup_fn", SETUPS, ids=SETUP_IDS)
@mark.parametrize("cls", KFAC_LIKE_CLS, ids=lambda c: c.__name__)
@mark.parametrize("backend", BACKENDS)
def test_kfac_like_matvec_no_graph_breaks(cls, backend, setup_fn):
    """(E)KFAC matvec compiles with zero graph breaks for both backends."""
    model, loss_fn, params, data = setup_fn()
    num_per_example_loss_terms = data[0][1][0].numel()
    K = cls(
        model,
        loss_fn,
        params,
        data,
        check_deterministic=False,
        separate_weight_and_bias=False,
        num_per_example_loss_terms=num_per_example_loss_terms,
        backend=backend,
    )
    _assert_no_graph_breaks(K)


@mark.parametrize("setup_fn", SETUPS, ids=SETUP_IDS)
def test_kfac_precompute_no_graph_breaks(setup_fn):
    """KFAC per-batch factor computation traced with ``make_fx`` compiles correctly.

    The CNN case is critical: Conv2d layers produce non-contiguous tensors
    (via ``movedim`` and ``autograd.grad(is_grads_batched=True)``), which would
    fail during ``torch.compile`` if ``einops.rearrange`` (traces as
    ``aten.view``) were used instead of ``flatten``/``unsqueeze``.
    """
    model, loss_fn, params, data = setup_fn()
    X, y = data[0]
    traced, *_ = make_compute_kfac_batch(
        model, loss_fn, params, X, y, separate_weight_and_bias=False
    )

    def traced_seeded(params, X, y):
        with fork_rng():
            manual_seed(0)
            return traced(params, X, y)

    with _dynamo_explain(traced_seeded, params, X, y) as result:
        assert result.graph_break_count == 0
