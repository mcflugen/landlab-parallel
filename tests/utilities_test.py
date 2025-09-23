import numpy as np
import pytest
from numpy.testing import assert_array_equal

from landlab_parallel.utilities import build_csr_array
from landlab_parallel.utilities import roll_values


@pytest.mark.parametrize("dtype", [None, int, float, bool, np.uint8, np.float32])
@pytest.mark.parametrize("n_rows", [0, 1, 5, 10])
def test_dtype_with_only_empty_rows(dtype, n_rows):
    actual_offset, actual_values = build_csr_array([()] * n_rows, dtype=dtype)
    expected_offset = np.zeros(n_rows + 1, dtype=np.int64)
    expected_values = np.array([], dtype=dtype)

    assert actual_offset.dtype == expected_offset.dtype
    assert actual_values.dtype == expected_values.dtype
    assert_array_equal(actual_offset, expected_offset)
    assert_array_equal(actual_values, expected_values)


@pytest.mark.parametrize("pos", [0, 1, 2])
def test_empty_row(pos):
    array = [[0], [1]]
    array.insert(pos, [])

    actual_offset, actual_values = build_csr_array(array)
    expected_values = [0, 1]
    expected_offset = [0, 1, 2]
    expected_offset.insert(pos, pos)

    assert_array_equal(actual_offset, expected_offset)
    assert_array_equal(actual_values, expected_values)


@pytest.mark.parametrize(
    "array,dtype",
    (
        ([[0], [1], [2.0]], float),
        ([[0], [1], [2]], int),
        ([[0], [1], [2 + 0j]], complex),
    ),
)
def test_guess_type(array, dtype):
    _, actual = build_csr_array(array)
    expected = [0, 1, 2]

    assert actual.dtype == dtype
    assert_array_equal(actual, expected)


@pytest.mark.parametrize("pos", [0, 1, 2])
def test_roll_empty_row(pos):
    array = [0, 3, 6]
    array.insert(pos, pos * 3)

    assert_array_equal(
        roll_values(array, [1, 2, 3, 4, 5, 6], direction="left"), [2, 3, 1, 5, 6, 4]
    )
    assert_array_equal(
        roll_values(array, [1, 2, 3, 4, 5, 6], direction="right"), [3, 1, 2, 6, 4, 5]
    )


@pytest.mark.parametrize(
    "direction,expected", (("right", [2, 0, 1]), ("left", [1, 2, 0]))
)
def test_roll_single_row_wraps_left_right(direction, expected):
    assert_array_equal(roll_values([0, 3], [0, 1, 2], direction), expected)


def test_roll_multiple_rows_with_empty_middle():
    indptr = [0, 3, 3, 7]
    values = [10, 11, 12, 13, 14, 15, 16]
    left = roll_values(indptr, values, "left")
    right = roll_values(indptr, values, "right")

    assert_array_equal(left, [11, 12, 10, 14, 15, 16, 13])
    assert_array_equal(right, [12, 10, 11, 16, 13, 14, 15])


@pytest.mark.parametrize("dtype", (float, int, bool, complex))
@pytest.mark.parametrize("direction", ("left", "right"))
def test_roll_keeps_type(dtype, direction):
    expected = np.array([1, 2, 0] if direction == "left" else [2, 0, 1], dtype=dtype)
    actual = roll_values([0, 3], np.asarray([0, 1, 2], dtype=dtype), direction)
    assert actual.dtype == expected.dtype
    assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "direction", ("up", "", "leftfoo", "rightfoo", "left ", " right", None)
)
def test_roll_invalid_side_raises(direction):
    with pytest.raises(ValueError):
        roll_values([0, 2], [1, 2], direction=direction)
