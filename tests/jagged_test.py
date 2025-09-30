import numpy as np
import pytest

from landlab_parallel.jagged import validate_jagged


@pytest.mark.parametrize(
    "offsets,values,n_rows",
    [
        ([0, 2, 5], [0.0, 1, 5, 6, 7], None),
        ([0, 0, 3], [10, 20, 30], None),
        ([0], [], None),
        ([0], [], 0),
        (np.array([0, 1, 3], dtype=np.int64), np.array([1, 2, 3.0]), 2),
    ],
)
def test_validate_jagged_return(offsets, values, n_rows):
    actual_offsets, actual_values = validate_jagged(offsets, values, n_rows=n_rows)

    np.testing.assert_array_equal(actual_offsets, np.asarray(offsets))
    np.testing.assert_array_equal(actual_values, np.asarray(values))


@pytest.mark.parametrize(
    "offsets,values,msg_substr",
    [
        (np.array([[0, 2, 5]]), [0, 1, 2, 3, 4], "offsets must be a 1D array"),
        (np.array([0.0, 2.0, 5.0]), [0, 1, 2, 3, 4], "array of int"),
        ([], [], "length >= 1"),
        ([1, 2, 5], [0, 1, 2, 3, 4], "first value of offsets"),
        ([0, 3, 2], [0, 1, 2], "non-decreasing"),
    ],
)
def test_validate_jagged_invalid_offsets_structure(offsets, values, msg_substr):
    with pytest.raises(ValueError, match=msg_substr):
        validate_jagged(offsets, values)


@pytest.mark.parametrize(
    "offsets, values, msg_substr",
    [
        ([0, 2, 5], np.array([[0, 1, 2, 3, 4]]), "values must be a 1D array"),
        ([0, 2, 4], [0.0, 1, 5, 6, 7], "mismatch"),
        ([0, 2, 5], [10, 11, 12], "mismatch"),
    ],
)
def test_validate_jagged_invalid_values(offsets, values, msg_substr):
    with pytest.raises(ValueError, match=msg_substr):
        validate_jagged(offsets, values)


@pytest.mark.parametrize(
    "offsets,values,n_rows",
    [
        ([0, 2, 5], [1, 2, 3, 4, 5], 2),
        ([0], [], 0),
    ],
)
def test_validate_jagged_n_rows_good(offsets, values, n_rows):
    validate_jagged(offsets, values, n_rows=n_rows)


@pytest.mark.parametrize(
    "offsets,values,n_rows",
    [
        ([0, 2, 5], [1, 2, 3, 4, 5], 3),
        ([0, 2, 5], [1, 2, 3, 4, 5], 1),
    ],
)
def test_validate_jagged_n_rows_bad(offsets, values, n_rows):
    with pytest.raises(ValueError, match="number of rows"):
        validate_jagged(offsets, values, n_rows=n_rows)


@pytest.mark.parametrize(
    "offsets,values",
    [
        ([0, 0, 0, 2, 2, 2, 5], [7, 8, 9, 10, 11]),
        ([0, 0, 0, 0, 4], [1, 2, 3, 4]),
    ],
)
def test_validate_jagged_empty_rows_patterns(offsets, values):
    actual_offsets, actual_values = validate_jagged(offsets, values)
    np.testing.assert_array_equal(actual_offsets, np.asarray(offsets))
    np.testing.assert_array_equal(actual_values, np.asarray(values))


def test_validate_jagged_large_input_smoke():
    rng = np.random.default_rng(2010)
    lengths = rng.integers(low=0, high=6, size=1000)
    offsets = np.empty(len(lengths) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    values = np.arange(offsets[-1], dtype=np.int64)

    actual_offsets, actual_values = validate_jagged(offsets, values)
    np.testing.assert_array_equal(actual_offsets, offsets)
    np.testing.assert_array_equal(actual_values, values)
