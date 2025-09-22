import numpy as np
import pytest
from numpy.testing import assert_array_equal

from landlab_parallel.utilities import build_csr_array


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
