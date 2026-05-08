r"""Tests for :class:`LayerIO` and :class:`LayerIOSnapshot`.

These are targeted unit tests for the new orchestration layer. End-to-end
correctness is covered by the broader KFAC/EKFAC/KFOC test suites, which
exercise ``LayerIO`` transitively through ``MakeFxKFACComputer``.
"""

from pytest import raises
from torch import manual_seed, randint, randn
from torch.nn import CrossEntropyLoss, Linear, MSELoss, Sequential

from curvlinops.computers.io_collector import LayerIO
from curvlinops.kfac_utils import FisherType
from curvlinops.utils import make_functional_call


def _setup_mlp(in_features: int = 6, out_features: int = 4):
    """Build a tiny MLP, its functional model, params, and CE loss.

    Params have ``requires_grad=True`` so direct ``populate`` calls (without
    going through ``enable_param_grads``) can run ``autograd.grad`` successfully.
    Tests that probe ``enable_param_grads``'s save/restore semantics flip
    ``requires_grad`` explicitly.

    Args:
        in_features: Input dimension of the first ``Linear``.
        out_features: Output dimension of the second ``Linear``.

    Returns:
        Tuple ``(model_func, loss, params)`` ready for ``LayerIO`` construction.
    """
    manual_seed(0)
    model = Sequential(Linear(in_features, 8), Linear(8, out_features)).eval()
    loss = CrossEntropyLoss()
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    model_func = make_functional_call(model)
    return model_func, loss, params


def test_enable_param_grads_preserves_requires_grad():
    """A frozen param stays frozen after ``enable_param_grads`` exits."""
    model_func, loss, params = _setup_mlp()
    params["0.weight"].requires_grad_(False)
    params["1.bias"].requires_grad_(False)

    io = LayerIO(model_func, loss, params, randn(4, 6), fisher_type=FisherType.MC)
    with io.enable_param_grads(params):
        for p in params.values():
            assert p.requires_grad
    # After: original state restored
    assert not params["0.weight"].requires_grad
    assert not params["1.bias"].requires_grad
    assert params["0.bias"].requires_grad
    assert params["1.weight"].requires_grad


def test_empirical_rejects_intermediate_as_batch_false():
    """Constructor raises for the unsupported EMPIRICAL + intermediate_as_batch=False combo."""
    model_func, loss, params = _setup_mlp()
    with raises(
        ValueError,
        match="intermediate_as_batch=False is not supported with FisherType.EMPIRICAL",
    ):
        LayerIO(
            model_func,
            MSELoss(),
            params,
            randn(4, 6),
            fisher_type=FisherType.EMPIRICAL,
            intermediate_as_batch=False,
        )


def test_forward_only_per_sample_grads_raises():
    """``per_sample_grads`` raises a clear error under ``FORWARD_ONLY``.

    ``standardized_io`` returns ``g=None`` in this mode; calling the einsum on
    ``None`` would otherwise crash with a cryptic error. Verify the public
    API fails predictably.
    """
    model_func, loss, params = _setup_mlp()
    X = randn(4, 6)
    y = randint(0, 4, (4,))

    io = LayerIO(model_func, loss, params, X, fisher_type=FisherType.FORWARD_ONLY)
    snap = io.snapshot(*io.populate(params, X, y))
    for group in io.mapping:
        with raises(
            RuntimeError,
            match="per_sample_grads is undefined for FisherType.FORWARD_ONLY",
        ):
            snap.per_sample_grads(group)
