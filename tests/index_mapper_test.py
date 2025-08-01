import numpy as np
import pytest

from landlab_parallel.index_mapper import IndexMapper


@pytest.mark.parametrize(
    "shape",
    [
        (3,),
        (4, 5),
        (2, 3, 4),
    ],
)
def test_identity_mapping_round_trip(shape):
    mapper = IndexMapper(shape)
    indices = np.arange(np.prod(shape))
    global_indices = mapper.local_to_global(indices)
    assert np.array_equal(global_indices, indices)
    assert np.array_equal(mapper.global_to_local(global_indices), indices)


@pytest.mark.parametrize(
    "shape, submatrix",
    [
        ((4, 5), [(1, 3), (2, 5)]),
        ((3, 4, 2), [(1, 3), (0, 2), (1, 2)]),
    ],
)
def test_submatrix_round_trip(shape, submatrix):
    mapper = IndexMapper(shape, submatrix=submatrix)
    local_shape = tuple(high - low for low, high in submatrix)
    indices = np.arange(np.prod(local_shape))

    coords = np.unravel_index(indices, local_shape)
    global_coords = [coords[dim] + submatrix[dim][0] for dim in range(len(shape))]
    expected_global = np.ravel_multi_index(global_coords, shape)

    result = mapper.local_to_global(indices)
    assert np.array_equal(result, expected_global)
    assert np.array_equal(mapper.global_to_local(result), indices)


@pytest.mark.parametrize(
    "shape, submatrix",
    [
        ((3, 4), [(0, 3)]),
        ((3, 4), [(0, 3), (0, 4), (0, 1)]),
    ],
)
def test_dimension_mismatch(shape, submatrix):
    with pytest.raises(ValueError):
        IndexMapper(shape, submatrix=submatrix)


@pytest.mark.parametrize(
    "shape, submatrix",
    [
        ((3, 4), [(-1, 2), (0, 4)]),
        ((3, 4), [(1, 4), (0, 4)]),
        ((3, 4), [(0, 2), (0, 5)]),
    ],
)
def test_invalid_submatrix_bounds(shape, submatrix):
    with pytest.raises(ValueError):
        IndexMapper(shape, submatrix=submatrix)
