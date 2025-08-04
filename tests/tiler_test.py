import numpy as np
import pytest

from landlab_parallel.tiler import D4Tiler


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
    assert D4Tiler(partitions).get_tile_size(0) == expected


@pytest.mark.parametrize("halo", (0, 1, 2, 3))
def test_tiler_halo(halo):
    n = 2 * halo + 1
    partitions = np.ones((n + 2, n + 2), dtype=int)
    partitions[n // 2, n // 2] = 0

    tiler = D4Tiler(partitions, halo=halo)
    assert tiler.get_tile_size(0) == n**2
    assert tiler.getvalue(0).shape == (n, n)
