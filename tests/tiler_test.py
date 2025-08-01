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
