import numpy as np
import pytest
from numpy.testing import assert_array_equal

from landlab_parallel.tiler import D4Tiler
from landlab_parallel.tiler import Tile


@pytest.mark.parametrize(
    "partitions, expected",
    [
        (
            [
                [0, 0, 1, 1, 1],
                [0, 0, 0, 1, 1],
                [0, 1, 1, 1, 1],
            ],
            12,
        ),
        (
            [
                [1, 1, 1, 1, 1],
                [1, 1, 0, 1, 1],
                [1, 1, 1, 1, 1],
            ],
            9,
        ),
        (
            [
                [1, 1, 1, 1, 1],
                [1, 1, 0, 1, 0],
                [1, 1, 1, 0, 0],
            ],
            12,
        ),
        (
            [
                [1, 1, 1, 1, 1],
                [1, 1, 0, 1, 1],
                [1, 0, 1, 0, 1],
            ],
            15,
        ),
    ],
)
def test_get_tile_size(partitions, expected):
    assert D4Tiler(partitions, halo=1).get_tile_size(0) == expected


@pytest.mark.parametrize("halo", (0, 1, 2, 3))
def test_tiler_halo(halo):
    n = 2 * halo + 1
    partitions = np.ones((n + 2, n + 2), dtype=int)
    partitions[n // 2, n // 2] = 0

    tiler = D4Tiler(partitions, halo=halo)
    assert tiler.get_tile_size(0) == n**2
    assert tiler.getvalue(0).shape == (n, n)


@pytest.mark.parametrize("halo", (0, 1, 2, 3))
def test_tiler_halo_with_pymetis(halo):
    tiler = D4Tiler.from_pymetis((32, 32), 2, halo=halo)
    assert len(tiler) == 2


def test_tile():
    tile = Tile(
        (0, 0),
        (64, 64),
        [
            [1, 1, 1, 2, 2],
            [1, 1, 2, 2, 2],
            [1, 2, 2, 2, 3],
        ],
        1,
    )
    expected = {2: [3, 7, 11]}
    actual = tile._ghost_nodes

    assert actual.keys() == expected.keys()
    for owner in tile._ghost_nodes:
        assert_array_equal(actual[owner], expected[owner])
