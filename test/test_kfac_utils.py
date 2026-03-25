"""Test utility functions specific to KFAC (canonical space converters)."""

from pytest import mark
from torch import cat, manual_seed, rand_like
from torch.nn import Conv2d, Linear, Sequential

from curvlinops.kfac_utils import FromCanonicalLinearOperator, ToCanonicalLinearOperator
from curvlinops.utils import allclose_report


@mark.parametrize("separate_weight_and_bias", [True, False], ids=["separate", "joint"])
def test_CanonicalLinearOperator(separate_weight_and_bias: bool):
    """Test that canonicalization properly reorders, groups, and ungroups parameters."""
    manual_seed(0)

    # Define an unconventional order of the parameter space, mixing weights and biases.
    # NOTE This network is non-functional, we only care about its parameters
    net = Sequential(Conv2d(6, 5, 4), Linear(5, 3), Linear(3, 2, bias=False))

    # Natural order would be: w1, b1, w2, b2, w3
    # Create unconventional order: w1, b2, b1, w3, w2
    natural_params = dict(net.named_parameters())
    # Natural order would be: w1, b1, w2, b2, w3
    # Create unconventional order: w1, b2, b1, w3, w2
    keys = list(natural_params.keys())
    new_order = [keys[i] for i in [0, 3, 1, 4, 2]]
    params = {k: natural_params[k] for k in new_order}
    assert list(params.keys()) == [
        "0.weight",  # w1
        "1.bias",  # b2
        "0.bias",  # b1
        "2.weight",  # w3
        "1.weight",  # w2
    ]

    # Define param_groups: with separate treatment each role is its own group,
    # with joint treatment weight and bias form one group
    if separate_weight_and_bias:
        param_groups = [
            {"W": "0.weight"},
            {"b": "0.bias"},
            {"W": "1.weight"},
            {"b": "1.bias"},
            {"W": "2.weight"},
        ]
    else:
        param_groups = [
            {"W": "0.weight", "b": "0.bias"},
            {"W": "1.weight", "b": "1.bias"},
            {"W": "2.weight"},
        ]

    # Extract param shapes, device, and dtype
    param_shapes = {name: p.shape for name, p in params.items()}
    device = next(iter(params.values())).device
    dtype = next(iter(params.values())).dtype

    # Verify correct behavior of canonicalization for this case
    x = [rand_like(p) for p in params.values()]
    # Order in params follows the unconventional order: w1, b2, b1, w3, w2
    x_w1, x_b2, x_b1, x_w3, x_w2 = x

    x_canonical = (
        [x_i.flatten() for x_i in [x_w1, x_b1, x_w2, x_b2, x_w3]]
        if separate_weight_and_bias
        else [
            # Conv kernel is 4d, we flatten it to 2d first before appending the bias
            cat([x_w1.flatten(start_dim=1), x_b1.unsqueeze(-1)], dim=-1).flatten(),
            cat([x_w2, x_b2.unsqueeze(-1)], dim=-1).flatten(),
            x_w3.flatten(),
        ]
    )

    # Multiplication with canonicalization operator should produce x_canonical
    to_canonical = ToCanonicalLinearOperator(param_shapes, param_groups, device, dtype)
    to_canonical_x = to_canonical @ x
    assert len(to_canonical_x) == len(x_canonical)
    assert all(allclose_report(x1, x2) for x1, x2 in zip(to_canonical_x, x_canonical))

    # Multiplication of x_canonical with from_canonical operator should produce x
    from_canonical = FromCanonicalLinearOperator(
        param_shapes, param_groups, device, dtype
    )
    from_canonical_x = from_canonical @ x_canonical
    assert len(from_canonical_x) == len(x)
    assert all(allclose_report(x1, x2) for x1, x2 in zip(from_canonical_x, x))

    # Check that the transpose operator is the inverse
    for P, v in zip([to_canonical, from_canonical], [x, x_canonical]):
        PTP_v = P.adjoint() @ (P @ v)
        assert len(PTP_v) == len(v)
        assert all(allclose_report(v1, v2) for v1, v2 in zip(PTP_v, v))
