from __future__ import annotations

import numpy as np


def _get_d4_adjacency(shape: tuple[int, int]) -> list[list[int]]:
    """Return D4 adjacency list for a raster grid.

    Parameters
    ----------
    shape : tuple of int
        Number of rows and columns in the grid.

    Returns
    -------
    list[list[int]]
        Adjacency list using D4 connectivity.
    """
    nodes = np.pad(
        np.arange(shape[0] * shape[1]).reshape(shape),
        pad_width=1,
        mode="constant",
        constant_values=-1,
    )

    d4_neighbors = np.stack(
        [
            nodes[1:-1, 2:],
            nodes[2:, 1:-1],
            nodes[1:-1, :-2],
            nodes[:-2, 1:-1],
        ],
        axis=-1,
    )

    return [[int(x) for x in row[row != -1]] for row in d4_neighbors.reshape(-1, 4)]


def _get_d8_adjacency(shape: tuple[int, int]) -> list[list[int]]:
    """Return D8 adjacency list for a raster grid.

    Parameters
    ----------
    shape : tuple of int
        Number of rows and columns in the grid.

    Returns
    -------
    list[list[int]]
        Adjacency list using D8 connectivity.
    """
    nodes = np.pad(
        np.arange(shape[0] * shape[1]).reshape(shape),
        pad_width=1,
        mode="constant",
        constant_values=-1,
    )

    d8_neighbors = np.stack(
        [
            nodes[1:-1, 2:],
            nodes[2:, 2:],
            nodes[2:, 1:-1],
            nodes[2:, :-2],
            nodes[1:-1, :-2],
            nodes[:-2, :-2],
            nodes[:-2, 1:-1],
            nodes[:-2, 2:],
        ],
        axis=-1,
    )

    return [[int(x) for x in row[row != -1]] for row in d8_neighbors.reshape(-1, 8)]


def _get_odd_r_adjacency(shape: tuple[int, int]) -> list[list[int]]:
    """Return adjacency list for a hex grid with odd-r layout.

    Parameters
    ----------
    shape : tuple of int
        Number of rows and columns in the grid.

    Returns
    -------
    list[list[int]]
        Adjacency list using odd-r connectivity.
    """
    nrows, ncols = shape
    rows, cols = np.meshgrid(np.arange(nrows), np.arange(ncols), indexing="ij")
    node_ids = rows * ncols + cols

    even_offsets = np.array([[0, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0]])
    odd_offsets = np.array([[0, 1], [1, 1], [1, 0], [0, -1], [-1, 0], [-1, 1]])

    adjacency: list[list[int]] = [[] for _ in range(nrows * ncols)]

    for parity, offsets in enumerate([even_offsets, odd_offsets]):
        parity_mask = rows % 2 == parity

        row_indices = rows[parity_mask]
        col_indices = cols[parity_mask]
        base_ids = node_ids[parity_mask]

        for dr, dc in offsets:
            r = row_indices + dr
            c = col_indices + dc

            valid = (0 <= r) & (r < nrows) & (0 <= c) & (c < ncols)
            src = base_ids[valid]
            dst = r[valid] * ncols + c[valid]

            for s, d in zip(src, dst):
                adjacency[s].append(int(d))

    return adjacency
