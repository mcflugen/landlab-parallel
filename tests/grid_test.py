from numpy.testing import assert_array_equal

from landlab_parallel.grid import create_landlab_grid


def test_grid_d4():
    partitions = [
        [0, 0, 0, 1],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
    ]
    grid = create_landlab_grid(partitions, id_=0, mode="d4")
    assert grid.shape == (4, 4)
    assert_array_equal(
        grid.status_at_node.reshape(grid.shape),
        [
            [1, 1, 1, 1],
            [1, 0, 1, 4],
            [1, 0, 1, 4],
            [1, 1, 1, 1],
        ],
    )
