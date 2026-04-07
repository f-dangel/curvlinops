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

from pytest import mark, raises
from torch import compile as torch_compile
from torch import manual_seed, no_grad, rand
from torch._dynamo import explain
from torch._dynamo import reset as dynamo_reset
from torch.nn import Conv2d, Linear, MSELoss, Sequential
from torch.testing import assert_close

from curvlinops import (
    EFLinearOperator,
    EKFACLinearOperator,
    GGNLinearOperator,
    HessianLinearOperator,
    KFACLinearOperator,
)
from curvlinops.computers._base import _BaseKFACComputer
from curvlinops.computers.io_collector import with_kfac_io
from curvlinops.computers.kfac_make_fx import (
    _build_param_groups_from_io,
    make_compute_kfac_batch,
)
from curvlinops.examples import trace_gradient_and_loss
from curvlinops.kfac_utils import FisherType, KFACType
from curvlinops.utils import _make_fx, make_functional_call


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


def test_gradient_and_loss_no_graph_breaks():
    """Per-batch gradient+loss traced with ``make_fx`` compiles with 0 graph breaks."""
    model, loss_fn, params, data = _setup_problem()
    X, y = data[0]
    traced = trace_gradient_and_loss(model, loss_fn, params, X, y)
    with _dynamo_explain(traced, params, X, y) as result:
        assert result.graph_break_count == 0


def test_hessian_matvec_no_graph_breaks():
    """``HessianLinearOperator @ v`` compiles with zero graph breaks."""
    model, loss_fn, params, data = _setup_problem()
    H = HessianLinearOperator(model, loss_fn, params, data, check_deterministic=False)
    _assert_no_graph_breaks(H)


def test_ggn_matvec_no_graph_breaks():
    """``GGNLinearOperator @ v`` (exact) compiles with zero graph breaks."""
    model, loss_fn, params, data = _setup_problem()
    G = GGNLinearOperator(model, loss_fn, params, data, check_deterministic=False)
    _assert_no_graph_breaks(G)


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
    with _dynamo_explain(lambda op, vec: op @ vec, G, v) as result:
        assert all("fork_rng" in str(br.reason) for br in result.break_reasons)


def test_ef_matvec_no_graph_breaks():
    """``EFLinearOperator @ v`` compiles with zero graph breaks."""
    model, loss_fn, params, data = _setup_problem()
    E = EFLinearOperator(model, loss_fn, params, data, check_deterministic=False)
    _assert_no_graph_breaks(E)


KFAC_LIKE_CLS = [KFACLinearOperator, EKFACLinearOperator]
BACKENDS = ["hooks", "make_fx"]


@mark.parametrize("cls", KFAC_LIKE_CLS, ids=lambda c: c.__name__)
@mark.parametrize("backend", BACKENDS)
def test_kfac_like_matvec_no_graph_breaks(cls, backend):
    """(E)KFAC matvec compiles with zero graph breaks for both backends."""
    model, loss_fn, params, data = _setup_problem()
    K = cls(
        model,
        loss_fn,
        params,
        data,
        check_deterministic=False,
        separate_weight_and_bias=False,
        num_per_example_loss_terms=2,
        backend=backend,
    )
    _assert_no_graph_breaks(K)


def test_kfac_fx_fake_traced_fn_requires_per_batch_size_tracing():
    """A fake-traced batch function has hardcoded shapes and fails on other sizes."""
    model, loss_fn, params, data = _setup_problem()
    model_func = make_functional_call(model)
    X_2, y_2 = data[0]  # batch size 2
    X_3, y_3 = data[1]  # batch size 3

    io_fn, io_param_names, layer_hparams = with_kfac_io(
        model_func, X_2, params, FisherType.TYPE2
    )
    mapping, io_groups = _build_param_groups_from_io(io_param_names, False)
    grad_outputs_computer = _BaseKFACComputer._set_up_grad_outputs_computer(
        loss_fn, FisherType.TYPE2, 1
    )
    rearrange_fn = lambda output, y: (output, y)  # noqa: E731

    for p in params.values():
        p.requires_grad_(True)

    batch_fn = make_compute_kfac_batch(
        io_fn,
        io_param_names,
        layer_hparams,
        mapping,
        io_groups,
        KFACType.EXPAND,
        FisherType.TYPE2,
        loss_fn.reduction,
        2,
        grad_outputs_computer,
        rearrange_fn,
    )

    traced = _make_fx(batch_fn)(params, X_2, y_2)

    # Same batch size works
    with no_grad():
        traced(params, X_2, y_2)

    # Different batch size fails (shapes are hardcoded in the traced graph)
    with raises(RuntimeError):
        with no_grad():
            traced(params, X_3, y_3)


def _setup_cnn_problem():
    """Create a small CNN test problem (Conv2d triggers non-contiguous tensors).

    Returns:
        Tuple of (model, loss_fn, params, data).
    """
    manual_seed(0)
    model = Sequential(Conv2d(3, 4, 3, padding=1), Conv2d(4, 5, 3, padding=1))
    loss_fn = MSELoss()
    params = dict(model.named_parameters())
    X = rand(2, 3, 6, 6)
    y = rand(2, 5, 6, 6)
    return model, loss_fn, params, [(X, y)]


@mark.parametrize(
    "setup_fn",
    [_setup_problem, _setup_cnn_problem],
    ids=["mlp", "cnn"],
)
def test_kfac_precompute_no_graph_breaks(setup_fn):
    """KFAC per-batch factor computation traced with ``make_fx`` compiles correctly.

    The CNN case is critical: Conv2d layers produce non-contiguous tensors
    (via ``movedim`` and ``autograd.grad(is_grads_batched=True)``), which would
    fail during ``torch.compile`` if ``einops.rearrange`` (traces as
    ``aten.view``) were used instead of ``flatten``/``unsqueeze``.
    """
    model, loss_fn, params, data = setup_fn()
    X, y = data[0]
    model_func = make_functional_call(model)

    # Trace IO collection
    io_fn, io_param_names, layer_hparams = with_kfac_io(
        model_func, X, params, FisherType.TYPE2
    )
    mapping, io_groups = _build_param_groups_from_io(io_param_names, False)

    # Set up grad_outputs_computer and rearrange_fn
    grad_outputs_computer = _BaseKFACComputer._set_up_grad_outputs_computer(
        loss_fn, FisherType.TYPE2, 1
    )
    rearrange_fn = lambda output, y: (output, y)  # noqa: E731

    for p in params.values():
        p.requires_grad_(True)

    batch_fn = make_compute_kfac_batch(
        io_fn,
        io_param_names,
        layer_hparams,
        mapping,
        io_groups,
        KFACType.EXPAND,
        FisherType.TYPE2,
        loss_fn.reduction,
        1,
        grad_outputs_computer,
        rearrange_fn,
    )

    traced = _make_fx(batch_fn)(params, X, y)
    with _dynamo_explain(traced, params, X, y) as result:
        assert result.graph_break_count == 0

    # Also verify full compilation succeeds (explain doesn't catch inductor
    # failures from aten.view on non-contiguous tensors, e.g. from einops)
    dynamo_reset()
    compiled = torch_compile(traced)
    with no_grad():
        compiled(params, X, y)
