from __future__ import annotations

import landlab
import numpy as np
from numpy.typing import ArrayLike

from landlab_parallel.ghosts import _d4_ghosts
from landlab_parallel.ghosts import _odd_r_ghosts


def create_landlab_grid(
    partitions: ArrayLike,
    spacing: float | tuple[float, float] = 1.0,
    ij_of_lower_left: tuple[int, int] = (0, 0),
    id_: int = 0,
    mode="d4",
) -> landlab.ModelGrid:
    """Create a Landlab grid from a partition matrix.

    Parameters
    ----------
    partitions : array_like
        Partition matrix describing ownership of each node.
    spacing : float or tuple of float, optional
        Grid spacing in the x and y directions.
    ij_of_lower_left : tuple of int, optional
        Index of the lower-left node of the tile within the full grid.
    id_ : int, optional
        Identifier of the local tile.
    mode : {"odd-r", "d4"}, optional
        Grid type describing connectivity.

    Returns
    -------
    landlab.ModelGrid
        The constructed grid with boundary conditions set.
    """
    is_their_node = np.asarray(partitions) != id_

    if mode == "odd-r":
        if not isinstance(spacing, float):
            raise TypeError(
                "Invalid spacing. The spacing for odd-r layout must be scalar"
                f" but got {spacing}."
            )
        shift: float = 0.5 if ij_of_lower_left[0] % 2 else 0.0
        xy_of_lower_left = (
            (ij_of_lower_left[1] + shift) * spacing,
            ij_of_lower_left[0] * spacing * np.sqrt(3.0) / 2.0,
        )
    elif mode == "d4":
        xy_of_lower_left = tuple(np.multiply(ij_of_lower_left, spacing))

    if mode == "d4":
        grid = landlab.RasterModelGrid(
            is_their_node.shape,
            xy_spacing=spacing,
            xy_of_lower_left=xy_of_lower_left,
        )
        get_ghosts = _d4_ghosts
    elif mode == "odd-r":
        grid = landlab.HexModelGrid(
            is_their_node.shape,
            spacing=spacing,
            xy_of_lower_left=xy_of_lower_left,
            node_layout="rect",
        )
        get_ghosts = _odd_r_ghosts

    is_ghost_node = get_ghosts(~is_their_node).reshape(-1)
    is_their_node.shape = (-1,)

    grid.status_at_node.fill(landlab.NodeStatus.CORE)
    grid.status_at_node[is_their_node] = np.where(
        is_ghost_node[is_their_node],
        landlab.NodeStatus.FIXED_VALUE,
        landlab.NodeStatus.CLOSED,
    )

    return grid
