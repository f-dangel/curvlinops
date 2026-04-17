"""Test general utility functions."""

from pytest import raises

from curvlinops.utils import _assert_single_element, split_list


def test_split_list():
    """Test list splitting utility function."""
    assert split_list(["a", "b", "c", "d"], [1, 3]) == [["a"], ["b", "c", "d"]]
    assert split_list(["a", "b", "c"], [3]) == [["a", "b", "c"]]

    with raises(ValueError):
        split_list(["a", "b", "c"], [1, 3])


def test_assert_single_element():
    """``_assert_single_element`` accepts exactly one element, rejects otherwise, and doesn't exhaust past the second element."""
    assert _assert_single_element(["x"]) is None

    with raises(ValueError, match="got 0"):
        _assert_single_element([])

    with raises(ValueError, match="got more than one"):
        _assert_single_element(["x", "y"])

    steps = 0

    def infinite():
        nonlocal steps
        while True:
            steps += 1
            yield "x"

    with raises(ValueError, match="got more than one"):
        _assert_single_element(infinite())
    assert steps == 2
