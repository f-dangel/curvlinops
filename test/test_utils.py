"""Test general utility functions."""

from pytest import raises

from curvlinops.utils import split_list


def test_split_list():
    """Test list splitting utility function."""
    assert split_list(["a", "b", "c", "d"], [1, 3]) == [["a"], ["b", "c", "d"]]
    assert split_list(["a", "b", "c"], [3]) == [["a", "b", "c"]]

    with raises(ValueError):
        split_list(["a", "b", "c"], [1, 3])
