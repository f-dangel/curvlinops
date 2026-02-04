"""Test utility functions related to KFAC."""

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
    natural_params = list(net.parameters())
    new_order = [0, 3, 1, 4, 2]
    params = [natural_params[idx] for idx in new_order]

    # Define param_positions to map back to layers
    param_positions = [
        {"weight": 0, "bias": 2},  # layer1: weight at pos 0, bias at pos 2
        {"weight": 4, "bias": 1},  # layer2: weight at pos 4, bias at pos 1
        {"weight": 3},  # layer3: weight at pos 3
    ]

    # Verify correct behavior of canonicalization for this case
    x = [rand_like(p) for p in params]
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
    to_canonical = ToCanonicalLinearOperator(
        params, param_positions, separate_weight_and_bias=separate_weight_and_bias
    )
    to_canonical_x = to_canonical @ x
    assert len(to_canonical_x) == len(x_canonical)
    for x1, x2 in zip(to_canonical_x, x_canonical):
        assert allclose_report(x1, x2)

    # Multiplication of x_canonical with from_canonical operator should produce x
    from_canonical = FromCanonicalLinearOperator(
        params, param_positions, separate_weight_and_bias=separate_weight_and_bias
    )
    from_canonical_x = from_canonical @ x_canonical
    assert len(from_canonical_x) == len(x)
    for x1, x2 in zip(from_canonical_x, x):
        assert allclose_report(x1, x2)
