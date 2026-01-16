import numpy as np
import pytest
from numpy.testing import assert_array_equal

from landlab_parallel.utilities import build_csr_array
from landlab_parallel.utilities import map_reverse_pairs
from landlab_parallel.utilities import roll_values
from landlab_parallel.utilities import unique_pairs
from landlab_parallel.utilities import wedge_is_inside_target


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


def test_map_reverse():
    pairs = [
        [0, 1],
        [1, 2],
        [1, 0],
        [2, 1],
    ]
    assert_array_equal(map_reverse_pairs(pairs), [2, 3, 0, 1])


def test_map_reverse_with_multiple_reverse_pairs():
    pairs = [
        [0, 1],
        [1, 2],
        [1, 0],
        [2, 1],
        [1, 0],
    ]
    assert_array_equal(map_reverse_pairs(pairs), [2, 3, 0, 1, 0])


def test_map_reverse_to_sleft():
    pairs = [
        [0, 1],
        [1, 2],
        [1, 0],
        [2, 1],
        [1, 1],
    ]
    assert_array_equal(map_reverse_pairs(pairs), [2, 3, 0, 1, 4])


def test_map_reverse_with_missing_error():
    pairs = [
        [0, 1],
        [1, 2],
        [2, 1],
    ]
    with pytest.raises(ValueError):
        map_reverse_pairs(pairs, if_missing="raise")
    with pytest.raises(ValueError):
        map_reverse_pairs(pairs)


@pytest.mark.parametrize("fill_value", [-1, -999, 9999, "ignore"])
def test_map_reverse_with_missing_fill(fill_value):
    pairs = [
        [0, 1],
        [1, 2],
        [2, 1],
    ]
    assert_array_equal(
        map_reverse_pairs(pairs, if_missing=fill_value),
        [-1 if fill_value == "ignore" else fill_value, 2, 1],
    )


def test_map_reverse_empty():
    actual = map_reverse_pairs([])
    assert_array_equal(actual, [])
    assert actual.dtype == np.int64


@pytest.mark.parametrize("if_missing", (None, "", "raisefoo", " ignore"))
def test_map_reverse_with_bad_keyword(if_missing):
    pairs = [
        [0, 1],
        [1, 2],
        [2, 1],
    ]
    with pytest.raises(ValueError):
        map_reverse_pairs(pairs, if_missing=if_missing)


def test_unique_pairs_rows_are_normalized():
    actual = unique_pairs([[0, 1], [2, 4], [1, 0], [3, 2]])

    assert np.all(actual[:, 0] <= actual[:, 1])
    assert_array_equal(actual, [[0, 1], [2, 3], [2, 4]])


def test_unique_pairs_with_negative_values_and_mixed_order():
    actual = unique_pairs([[5, -2], [-2, 5], [3, 3], [1, -10], [-10, 1]])

    assert np.all(actual[:, 0] <= actual[:, 1])
    assert_array_equal(actual, [[-10, 1], [-2, 5], [3, 3]])


@pytest.mark.parametrize("dtype", (np.int32, bool, np.float32, float, int))
def test_unique_pairs_dtype_is_preserved(dtype):
    pairs = np.array([[0, 1], [2, 4], [1, 0], [3, 2]], dtype=dtype)
    actual = unique_pairs(pairs).dtype
    assert actual == dtype


@pytest.mark.parametrize("array", ([], [0, 1], [[0, 1, 2], [3, 4, 5]]))
def test_unique_pairs_invalid_shape(array):
    with pytest.raises(ValueError, match="pairs must be a 2D array"):
        unique_pairs(array)


def test_unique_pairs_equivalence_to_numpy_unique():
    rng = np.random.default_rng(1945)
    pairs = rng.integers(-1000, 1000, size=(10_000, 2), dtype=np.int64)

    expected = np.unique(np.sort(pairs, axis=1), axis=0)

    actual = unique_pairs(pairs)

    assert_array_equal(actual, expected)


def test_unique_pairs_non_contiguous_input():
    pairs = np.arange(40).reshape((10, -1))

    view = pairs[:, ::2]
    assert not view.flags["C_CONTIGUOUS"]

    actual = unique_pairs(view)

    expected = np.unique(np.sort(view, axis=1), axis=0)
    assert_array_equal(actual, expected)


def test_unique_pairs_multiple_applications_unchanged():
    rng = np.random.default_rng(1973)
    pairs = rng.integers(0, 100, size=(1000, 2))
    out_1 = unique_pairs(pairs)
    out_2 = unique_pairs(out_1)

    assert_array_equal(out_1, out_2)


@pytest.mark.parametrize("side", ("", None, "foo", " left", "right ", "LEFT"))
def test_wedge_bad_side(side):
    with pytest.raises(ValueError, match="unknown key"):
        wedge_is_inside_target([0, 1, 2], [1, 0], [0, 1], [True, True], side=side)


@pytest.mark.parametrize("side", ("left", "right"))
@pytest.mark.parametrize(
    "tail,head",
    (
        ([0, 1], [0]),
        ([0], [0, 1]),
    ),
)
def test_wedge_bad_head_tail_size(side, tail, head):
    with pytest.raises(ValueError, match="mismatch in length"):
        wedge_is_inside_target([0, 1, 2], tail, head, [True, True], side=side)


@pytest.mark.parametrize("side", ("left", "right"))
def test_wedge_empty_indptr(side):
    with pytest.raises(ValueError, match="offsets must have length"):
        wedge_is_inside_target([], [0, 1], [1, 0], [True, True], side=side)


@pytest.mark.parametrize("side", ("left", "right"))
def test_wedge_empty_head_tail(side):
    actual = wedge_is_inside_target([0], [], [], [], side=side)
    assert_array_equal(actual, [])
    assert actual.dtype == bool


@pytest.mark.parametrize("side", ("left", "right"))
def test_wedge_all_in(side):
    indptr = [0, 2, 4, 6, 8]
    tails = [0, 0, 1, 1, 2, 2, 3, 3]
    heads = [1, 2, 0, 3, 0, 3, 1, 2]

    actual = wedge_is_inside_target(indptr, tails, heads, [True] * 4, side=side)
    assert_array_equal(actual, [True] * 8)


def test_wedge_left_right():
    # 1  2  3
    #  \ | /
    #    0
    is_in_target = [True, False, True, True]

    tails = [0, 0, 0, 1, 2, 3]
    heads = [1, 2, 3, 0, 0, 0]
    indptr = [0, 3, 4, 5, 6]

    actual = wedge_is_inside_target(indptr, tails, heads, is_in_target, side="left")
    assert_array_equal(actual[1:3], [False, True])

    actual = wedge_is_inside_target(indptr, tails, heads, is_in_target, side="right")
    assert_array_equal(actual[0:2], [False, True])
