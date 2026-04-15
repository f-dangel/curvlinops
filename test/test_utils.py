"""Test general utility functions."""

from pytest import raises

from curvlinops.utils import _has_single_batch, split_list


def test_split_list():
    """Test list splitting utility function."""
    assert split_list(["a", "b", "c", "d"], [1, 3]) == [["a"], ["b", "c", "d"]]
    assert split_list(["a", "b", "c"], [3]) == [["a", "b", "c"]]

    with raises(ValueError):
        split_list(["a", "b", "c"], [1, 3])


def test_has_single_batch():
    """``_has_single_batch`` accepts exactly one batch, rejects otherwise, and doesn't exhaust past the second element."""
    assert _has_single_batch([("x", "y")]) is None

    with raises(ValueError, match="got 0"):
        _has_single_batch([])

    with raises(ValueError, match="got more than one"):
        _has_single_batch([("x", "y"), ("x", "y")])

    steps = [0]

    def infinite():
        while True:
            steps[0] += 1
            yield ("x", "y")

    with raises(ValueError, match="got more than one"):
        _has_single_batch(infinite())
    assert steps[0] == 2
