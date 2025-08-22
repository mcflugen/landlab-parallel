import numpy as np
import pytest
from numpy.testing import assert_array_equal

from landlab_parallel.ghosts import is_ghost


@pytest.mark.parametrize("mode", ("d4", "d8", "odd-r"))
@pytest.mark.parametrize(
    "partitions",
    (
        [[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1]],
        [[0, 0, 2, 2]],
        np.asarray([[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1]]),
    ),
)
def test_is_ghost_shape_and_type(mode, partitions):
    actual = is_ghost(partitions)
    assert actual.dtype == bool
    assert actual.shape == np.asarray(partitions).shape


@pytest.mark.parametrize("mode", ("D4", "d5", "", None, 4))
def test_is_ghost_with_bad_mode(mode):
    with pytest.raises(ValueError):
        is_ghost([[0, 0, 1], [0, 1, 1], [1, 1, 1]], mode=mode)


@pytest.mark.parametrize(
    "mode,expected",
    [
        (
            "d4",
            [
                [0, 0, 1, 1],
                [0, 1, 1, 0],
                [1, 1, 0, 0],
            ],
        ),
        (
            "d8",
            [
                [0, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 0],
            ],
        ),
        (
            "odd-r",
            [
                [0, 0, 1, 1],
                [1, 1, 1, 0],
                [1, 1, 1, 0],
            ],
        ),
    ],
)
def test_is_ghost(mode, expected):
    partitions = [
        [0, 0, 0, 1],
        [0, 0, 1, 1],
        [0, 1, 1, 1],
    ]
    actual = is_ghost(partitions, mode=mode)
    assert_array_equal(actual, np.asarray(expected, dtype=bool))
