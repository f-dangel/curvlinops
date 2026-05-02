r"""Tests for :class:`LayerIO` and :class:`LayerIOSnapshot`.

These are targeted unit tests for the new orchestration layer. End-to-end
correctness is covered by the broader KFAC/EKFAC/KFOC test suites, which
exercise ``LayerIO`` transitively through ``MakeFxKFACComputer``.
"""

from einops import einsum
from pytest import raises
from torch import allclose, manual_seed, randint, randn
from torch.nn import CrossEntropyLoss, Linear, MSELoss, Sequential

from curvlinops.computers.io_collector import LayerIO
from curvlinops.kfac_utils import FisherType, KFACType
from curvlinops.utils import _make_fx, make_functional_call


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


def test_bootstrap_metadata_is_populated():
    """``LayerIO`` exposes mapping/io_groups/io_param_names after construction."""
    model_func, loss, params = _setup_mlp()
    X = randn(4, 6)

    io = LayerIO(model_func, loss, params, X, fisher_type=FisherType.TYPE2)

    assert io.fisher_type == FisherType.TYPE2
    assert io.kfac_approx == KFACType.EXPAND
    # 4 groups: W and b for each of 2 Linear layers (separate_weight_and_bias=True default)
    keys = {tuple(g.values()) for g in io.mapping}
    assert keys == {("0.weight",), ("0.bias",), ("1.weight",), ("1.bias",)}
    assert set(io.io_param_names) == {"Linear0", "Linear1"}
    # Each detected layer maps both W and b
    for layer in io.io_param_names.values():
        assert set(layer) == {"W", "b"}


def test_per_shape_cache_grows_on_new_shape():
    """``ensure_io_fn`` reuses cached io_fn for known shapes, builds for new ones."""
    model_func, loss, params = _setup_mlp()
    X1 = randn(4, 6)
    X2 = randn(7, 6)
    y1 = randint(0, 4, (4,))
    y2 = randint(0, 4, (7,))

    io = LayerIO(model_func, loss, params, X1, fisher_type=FisherType.MC)
    assert len(io._io_fns) == 1

    fn1_again = io.ensure_io_fn(X1, params)
    assert fn1_again is next(iter(io._io_fns.values()))
    assert len(io._io_fns) == 1  # cache hit

    io.ensure_io_fn(X2, params)
    assert len(io._io_fns) == 2  # new shape → new entry

    # Both shapes are usable
    li1, log1 = io.populate(params, X1, y1)
    li2, log2 = io.populate(params, X2, y2)
    assert li1["Linear0"].shape == (4, 6)
    assert li2["Linear0"].shape == (7, 6)
    assert log1["Linear0"].shape[1] == 4
    assert log2["Linear0"].shape[1] == 7


def test_populate_returns_expected_shapes():
    """``populate`` returns layer inputs and output grads with correct shapes."""
    model_func, loss, params = _setup_mlp(in_features=5, out_features=3)
    X = randn(8, 5)
    y = randint(0, 3, (8,))

    io = LayerIO(model_func, loss, params, X, fisher_type=FisherType.TYPE2)
    layer_inputs, layer_output_grads = io.populate(params, X, y)

    # Layer inputs: (batch, d_in)
    assert layer_inputs["Linear0"].shape == (8, 5)
    assert layer_inputs["Linear1"].shape == (8, 8)
    # Layer output grads: (V, batch, d_out). V = num backprop directions
    # (= num classes for type-2 + CE). Pin V here so the shape contract is enforced.
    assert layer_output_grads["Linear0"].shape == (3, 8, 8)
    assert layer_output_grads["Linear1"].shape == (3, 8, 3)


def test_snapshot_standardized_io_for_W_and_bias_groups():
    """``standardized_io`` returns ``a`` for W groups, ``None`` for bias-only."""
    model_func, loss, params = _setup_mlp(in_features=4, out_features=2)
    X = randn(5, 4)
    y = randint(0, 2, (5,))

    io = LayerIO(model_func, loss, params, X, fisher_type=FisherType.TYPE2)
    snap = io.snapshot(*io.populate(params, X, y))

    expected_d_in = {"0.weight": 4, "1.weight": 8}
    for group in io.mapping:
        a, g = snap.standardized_io(group)
        if "W" in group:
            assert a is not None
            # Standardized format: [batch, shared, d_in]
            assert a.shape[2] == expected_d_in[group["W"]]
        else:
            assert a is None
        # Not FORWARD_ONLY: g present with shape [V, B, S, d_out]
        assert g is not None
        assert g.dim() == 4


def test_per_sample_grads_W_matches_explicit_einsum():
    """``per_sample_grads`` for W groups equals the inline ``g (otimes) a`` einsum."""
    model_func, loss, params = _setup_mlp(in_features=4, out_features=2)
    X = randn(6, 4)
    y = randint(0, 2, (6,))

    io = LayerIO(
        model_func,
        loss,
        params,
        X,
        fisher_type=FisherType.TYPE2,
        intermediate_as_batch=False,
    )
    snap = io.snapshot(*io.populate(params, X, y))

    for group in io.mapping:
        if "W" not in group:
            continue
        a, g = snap.standardized_io(group)
        expected = einsum(
            g, a, "vec batch shared out, batch shared inp -> vec batch out inp"
        )
        actual = snap.per_sample_grads(group)
        assert allclose(actual, expected)


def test_per_sample_grads_bias_only_sums_over_shared():
    """``per_sample_grads`` for bias-only groups = sum over shared axis."""
    model_func, loss, params = _setup_mlp(in_features=4, out_features=2)
    X = randn(6, 4)
    y = randint(0, 2, (6,))

    io = LayerIO(
        model_func,
        loss,
        params,
        X,
        fisher_type=FisherType.TYPE2,
        intermediate_as_batch=False,
    )
    snap = io.snapshot(*io.populate(params, X, y))

    for group in io.mapping:
        if "W" in group:
            continue
        _, g = snap.standardized_io(group)
        expected = einsum(g, "vec batch shared row -> vec batch row")
        actual = snap.per_sample_grads(group)
        assert allclose(actual, expected)


def test_joint_weight_bias_group_includes_bias_pad():
    """``separate_weight_and_bias=False`` produces joint W+b groups with d_in+1."""
    model_func, loss, params = _setup_mlp(in_features=4, out_features=2)
    X = randn(5, 4)
    y = randint(0, 2, (5,))

    io = LayerIO(
        model_func,
        loss,
        params,
        X,
        fisher_type=FisherType.TYPE2,
        separate_weight_and_bias=False,
    )
    # Each layer becomes one joint group {W, b}
    for group in io.mapping:
        assert set(group) == {"W", "b"}

    snap = io.snapshot(*io.populate(params, X, y))
    for group in io.mapping:
        a, _ = snap.standardized_io(group)
        # Joint groups: a.shape[-1] = d_in + 1 (bias padding column of ones)
        # Linear layer input dim was 4 or 8; expect 5 or 9
        assert a.shape[-1] in {5, 9}


def test_enable_param_grads_preserves_requires_grad():
    """A frozen param stays frozen after ``enable_param_grads`` exits.

    Canonical unit-level guarantee for the autograd-ownership contract from
    `PR #301 <https://github.com/f-dangel/curvlinops/pull/301>`_. Once
    ``EKFAC`` and ``KFOC`` migrate to ``LayerIO`` (follow-up PRs), the
    operator-level integration tests
    (``test_{kfac,ekfac,kfoc}_make_fx_preserves_requires_grad``) become
    redundant with this one and can be dropped.
    """
    model_func, loss, params = _setup_mlp()
    # Freeze one param explicitly
    params["0.weight"].requires_grad_(False)
    params["1.bias"].requires_grad_(False)

    io = LayerIO(model_func, loss, params, randn(4, 6), fisher_type=FisherType.MC)
    with io.enable_param_grads(params):
        # Inside: all params should be enabled for the trace
        for p in params.values():
            assert p.requires_grad
    # After: original state restored
    assert not params["0.weight"].requires_grad
    assert not params["1.bias"].requires_grad
    assert params["0.bias"].requires_grad
    assert params["1.weight"].requires_grad


def test_make_fx_traces_populate():
    """``populate`` traces under ``enable_param_grads`` without raising."""
    model_func, loss, params = _setup_mlp(in_features=4, out_features=2)
    X = randn(6, 4)
    y = randint(0, 2, (6,))

    io = LayerIO(model_func, loss, params, X, fisher_type=FisherType.TYPE2)

    def populate(p, x, t):
        return io.populate(p, x, t)

    with io.enable_param_grads(params):
        traced = _make_fx(populate)(params, X, y)
    # Sanity: traced graph is callable and returns the right structure
    li, log = traced(params, X, y)
    assert set(li) == {"Linear0", "Linear1"}
    assert set(log) == {"Linear0", "Linear1"}


def test_empirical_rejects_intermediate_as_batch_false():
    """Constructor raises for the unsupported EMPIRICAL + intermediate_as_batch=False combo."""
    model_func, loss, params = _setup_mlp()
    with raises(ValueError, match="EMPIRICAL"):
        LayerIO(
            model_func,
            MSELoss(),
            params,
            randn(4, 6),
            fisher_type=FisherType.EMPIRICAL,
            intermediate_as_batch=False,
        )


def test_forward_only_populate_returns_empty_grads():
    """``FisherType.FORWARD_ONLY`` skips backward; ``layer_output_grads`` is empty."""
    model_func, loss, params = _setup_mlp()
    X = randn(4, 6)
    y = randint(0, 4, (4,))

    io = LayerIO(model_func, loss, params, X, fisher_type=FisherType.FORWARD_ONLY)
    layer_inputs, layer_output_grads = io.populate(params, X, y)
    assert set(layer_inputs) == {"Linear0", "Linear1"}
    assert layer_output_grads == {}


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
        with raises(RuntimeError, match="FORWARD_ONLY"):
            snap.per_sample_grads(group)


def test_metadata_assertion_on_shape_change():
    """Cached metadata must match across shapes; mock a mismatch and assert raise.

    We poke the bootstrap metadata after construction so that ``ensure_io_fn``
    sees a divergence on the next-shape trace, exercising the safety net.
    """
    model_func, loss, params = _setup_mlp()
    X1 = randn(4, 6)
    X2 = randn(7, 6)

    io = LayerIO(model_func, loss, params, X1, fisher_type=FisherType.MC)
    # Simulate divergence: corrupt the cached metadata
    io._io_param_names = {**io._io_param_names, "FakeLayer": {"W": "fake.weight"}}
    with raises(RuntimeError, match="parameter-name metadata"):
        io.ensure_io_fn(X2, params)
