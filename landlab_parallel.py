from __future__ import annotations

import os
import tempfile
import xml.etree.ElementTree as ET
from abc import ABC
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
from typing import IO
from typing import Self
from xml.dom import minidom

import landlab
import meshio
import numpy as np
import pymetis
from numpy.typing import ArrayLike
from numpy.typing import NDArray

__version__ = "0.1.0"


def get_my_ghost_nodes(
    partitions: ArrayLike, my_id: int = 0, mode: str = "d4"
) -> dict[int, NDArray[np.int_]]:
    """Return ghost node indices for a given partition.

    Parameters
    ----------
    partitions : array_like
        Partition matrix describing ownership of each node.
    my_id : int, optional
        Identifier of the local partition.
    mode : {"d4", "d8", "odd-r", "raster"}, optional
        Connectivity scheme used to determine neighbors.

    Returns
    -------
    dict[int, ndarray]
        Mapping of neighbor rank to the indices of ghost nodes.

    Examples
    --------
    >>> partitions = [
    ...     [0, 0, 1],
    ...     [0, 1, 1],
    ...     [2, 2, 1],
    ... ]
    >>> result = get_my_ghost_nodes(partitions, my_id=0)
    >>> {int(rank): nodes.tolist() for rank, nodes in result.items()}
    {1: [2, 4], 2: [6]}
    """
    if mode in ("d4", "raster"):
        get_ghosts = _d4_ghosts
    elif mode == "odd-r":
        get_ghosts = _odd_r_ghosts
    elif mode == "d8":
        get_ghosts = _d8_ghosts
    else:
        raise ValueError(f"{mode}: mode not understood")

    data_array = np.asarray(partitions)
    is_my_node = data_array == my_id
    is_ghost = get_ghosts(is_my_node)
    neighbors = np.unique(data_array[~is_my_node & is_ghost])

    return {
        rank: np.ravel_multi_index(
            np.nonzero(is_ghost & (data_array == rank)), data_array.shape
        )
        for rank in neighbors
    }


class Tile:
    """A single tile of a partitioned grid."""

    def __init__(
        self,
        offset: tuple[int, ...],
        shape: tuple[int, ...],
        partitions: ArrayLike,
        id_: int,
        mode: str = "raster",
    ):
        """Create a tile.

        Parameters
        ----------
        offset : tuple of int
            Index of the lower-left corner of the tile within the full array.
        shape : tuple of int
            Shape of the full domain.
        partitions : array_like
            Partition matrix describing ownership of each node.
        id_ : int
            Identifier of the local tile.
        mode : {"d4", "d8", "odd-r", "raster"}, optional
            Connectivity scheme used to determine neighbors.
        """
        self._shape = tuple(shape)
        self._offset = tuple(offset)
        self._partitions = np.asarray(partitions)
        self._id = id_

        self._index_mapper = IndexMapper(
            self._shape,
            submatrix=[
                (o, o + self._partitions.shape[dim]) for dim, o in enumerate(offset)
            ],
        )

        self._ghost_nodes = get_my_ghost_nodes(self._partitions, my_id=id_, mode=mode)

    def local_to_global(self, indices: ArrayLike) -> NDArray[np.int_]:
        """Convert local node indices to global indices.

        Parameters
        ----------
        indices : array_like of int
            Local indices to convert.

        Returns
        -------
        ndarray of int
            The corresponding global node indices.

        """
        return self._index_mapper.local_to_global(indices)

    def global_to_local(self, indices: ArrayLike) -> NDArray[np.int_]:
        """Convert global node indices to local indices.

        Parameters
        ----------
        indices : array_like of int
            Global indices to convert.

        Returns
        -------
        ndarray of int
            The corresponding local node indices.
        """
        return self._index_mapper.global_to_local(indices)


class Tiler(Mapping, ABC):
    """Base class for tiling utilities."""

    def __init__(self, partitions: ArrayLike):
        """Initialize the tiler.

        Parameters
        ----------
        partitions : array_like
            Partition matrix describing ownership of each node.
        """
        self._partitions = np.asarray(partitions)
        self._shape = self._partitions.shape

        self._tiles = {
            int(tile): tuple(
                slice(*bound)
                for bound in self.get_tile_bounds(self._partitions, tile, halo=1)
            )
            for tile in np.unique(self._partitions)
        }

    def __getitem__(self, key: int) -> tuple[slice, ...]:
        """Return slice bounds for ``key``.

        Parameters
        ----------
        key : int
            Identifier of the tile.

        Returns
        -------
        tuple of slice
            Bounds of the tile within the full array.
        """
        return self._tiles[key]

    def __iter__(self) -> Iterator[int]:
        """Iterate over tiles.

        Returns
        -------
        iterator of int
            Iterator over tile ids.
        """
        return iter(self._tiles)

    def __len__(self) -> int:
        """Number of tiles.

        Returns
        -------
        int
            The number of tiles in the tiler.
        """
        return len(self._tiles)

    def getvalue(self, tile: int) -> NDArray:
        """Return the partition slice for ``tile``.

        Parameters
        ----------
        tile : int
            Identifier of the tile to extract.

        Returns
        -------
        ndarray
            Slice of the partition array corresponding to ``tile``.
        """
        return self._partitions[*self[tile]]

    def get_tile_bounds(
        self, partitions: ArrayLike, tile: int, halo: int = 0
    ) -> list[tuple[int, int]]:
        """Return bounds of ``tile`` with optional halo.

        Parameters
        ----------
        partitions : array_like
            Partition matrix describing ownership of each node.
        tile : int
            Tile identifier.
        halo : int, optional
            Width of the halo to add around the tile.

        Returns
        -------
        list of tuple of int
            Start and stop indices for each dimension.
        """
        raise NotImplementedError()

    def scatter(self, data: ArrayLike) -> dict[int, NDArray]:
        """Split an array by tile.

        Parameters
        ----------
        data : array_like
            Array of values associated with the full domain.

        Returns
        -------
        dict[int, ndarray]
            Mapping of tile id to a copy of the tile's data.
        """
        data = np.asarray(data).reshape(self._shape)
        return {tile: data[*bounds].copy() for tile, bounds in self.items()}

    def gather(
        self, tile_data: dict[int, NDArray], out: NDArray | None = None
    ) -> NDArray:
        """Reassemble an array from tile data.

        Parameters
        ----------
        tile_data : dict[int, array_like]
            Mapping of tile id to data arrays.
        out : ndarray, optional
            Array to fill with gathered data.

        Returns
        -------
        ndarray
            Array assembled from the provided tile data.
        """
        if out is None:
            out = np.empty(self._shape)

        for tile, data in tile_data.items():
            array = out[*self[tile]]
            mask = self.getvalue(tile) == tile
            array[mask] = data.reshape(mask.shape)[mask]

        return out

    @classmethod
    def from_pymetis(cls, shape: tuple[int, int], n_tiles: int) -> Self:
        """Partition ``shape`` into ``n_tiles`` using PyMetis.

        Parameters
        ----------
        shape : tuple of int
            Shape of the grid to partition.
        n_tiles : int
            Desired number of tiles.

        Returns
        -------
        Tiler
            New tiler instance built from the generated partitions.
        """
        _, partitions = pymetis.part_graph(n_tiles, adjacency=cls.get_adjacency(shape))

        return cls(np.asarray(partitions).reshape(shape))

    @classmethod
    def get_adjacency(cls, shape: tuple[int, int]) -> list[list[int]]:
        raise NotImplementedError("get_adjacency")


class D4Tiler(Tiler):
    """Tiler for raster grids with D4 connectivity.

    Examples
    --------
    >>> from landlab_parallel import D4Tiler

    >>> partitions = [
    ...     [0, 0, 1, 1, 1],
    ...     [0, 0, 0, 1, 1],
    ...     [0, 2, 2, 1, 1],
    ...     [3, 3, 2, 2, 1],
    ...     [3, 3, 2, 2, 2],
    ... ]
    >>> tiler = D4Tiler(partitions)
    >>> tiler.n_tiles
    4
    >>> tiler.get_tile(0)
    array([[0, 0, 1, 1],
           [0, 0, 0, 1],
           [0, 2, 2, 1],
           [3, 3, 2, 2]])

    >>> data = [
    ...     [0.0, 1.0, 2.0, 3.0, 4.0],
    ...     [5.0, 6.0, 7.0, 8.0, 9.0],
    ...     [10.0, 11.0, 12.0, 13.0, 14.0],
    ...     [15.0, 16.0, 17.0, 18.0, 19.0],
    ...     [20.0, 21.0, 22.0, 23.0, 24.0],
    ... ]
    >>> tile_data = tiler.scatter(data)
    >>> tile_data[1]
    array([[ 1.,  2.,  3.,  4.],
           [ 6.,  7.,  8.,  9.],
           [11., 12., 13., 14.],
           [16., 17., 18., 19.],
           [21., 22., 23., 24.]])

    >>> for array in tile_data.values():
    ...     array /= 10.0
    ...
    >>> tile_data[1] *= 10.0
    >>> tiler.gather(tile_data)
    array([[ 0. ,  0.1,  2. ,  3. ,  4. ],
           [ 0.5,  0.6,  0.7,  8. ,  9. ],
           [ 1. ,  1.1,  1.2, 13. , 14. ],
           [ 1.5,  1.6,  1.7,  1.8, 19. ],
           [ 2. ,  2.1,  2.2,  2.3,  2.4]])
    """

    def get_tile_bounds(
        self, partitions: ArrayLike, tile: int, halo: int = 0
    ) -> list[tuple[int, int]]:
        """Bounds of ``tile`` using D4 connectivity.

        Parameters
        ----------
        partitions : array_like
            Partition matrix describing ownership of each node.
        tile : int
            Tile identifier.
        halo : int, optional
            Width of the halo to include around the tile.

        Returns
        -------
        list of tuple of int
            Start and stop indices for each dimension.
        """
        partitions = np.asarray(partitions)

        indices = np.nonzero(partitions == tile)

        return [
            (
                int(max(indices[dim].min() - halo, 0)),
                int(min(indices[dim].max() + halo + 1, partitions.shape[dim])),
            )
            for dim in range(partitions.ndim)
        ]

    @classmethod
    def get_adjacency(cls, shape: tuple[int, int]) -> list[list[int]]:
        """Return adjacency list for a D4 grid.

        Parameters
        ----------
        shape : tuple of int
            Shape of the grid.

        Returns
        -------
        list[list[int]]
            Adjacency list using D4 connectivity.
        """
        return _get_d4_adjacency(shape)


class OddRTiler(Tiler):
    """Tiler for hexagonal grids using odd-r layout."""

    def get_tile_bounds(
        self, partitions: ArrayLike, tile: int, halo: int = 0
    ) -> list[tuple[int, int]]:
        """Bounds of ``tile`` for an odd-r grid.

        Parameters
        ----------
        partitions : array_like
            Partition matrix describing ownership of each node.
        tile : int
            Tile identifier.
        halo : int, optional
            Width of the halo to include around the tile.

        Returns
        -------
        list of tuple of int
            Start and stop indices for each dimension.
        """
        partitions = np.asarray(partitions)

        if partitions.ndim != 2:
            raise ValueError(
                f"{partitions.shape!r}: invalid number of dimensions"
                f" ({partitions.ndim != 2})"
            )

        indices = np.nonzero(partitions == tile)

        start_row = int(max(indices[0].min() - halo, 0))
        stop_row = int(min(indices[0].max() + halo + 1, partitions.shape[0]))
        start_col = int(max(indices[1].min() - halo, 0))
        stop_col = int(min(indices[1].max() + halo + 1, partitions.shape[1]))

        if start_row % 2 != 0:
            start_row -= 1
        return [(start_row, stop_row), (start_col, stop_col)]

    @classmethod
    def get_adjacency(cls, shape: tuple[int, int]) -> list[list[int]]:
        """Return adjacency list for an odd-r grid.

        Parameters
        ----------
        shape : tuple of int
            Shape of the grid.

        Returns
        -------
        list[list[int]]
            Adjacency list using odd-r connectivity.
        """
        return _get_odd_r_adjacency(shape)


class IndexMapper:
    """Map between local and global node indices."""

    def __init__(
        self,
        shape: Sequence[int],
        submatrix: Sequence[tuple[int, int]] | None = None,
    ) -> None:
        """Create an index mapper.

        Parameters
        ----------
        shape : tuple of int
            Shape of the full domain.
        submatrix : tuple of tuple of int, optional
            Lower and upper bounds for each dimension.
        """
        self._shape = tuple(shape)
        if submatrix is None:
            self._limits = [(0, self._shape[dim]) for dim in range(len(self._shape))]
        else:
            self._limits = [(limit[0], limit[1]) for limit in submatrix]

        if len(self._limits) != len(self._shape):
            raise ValueError()
        if any(
            limit[0] < 0 or limit[1] > dim
            for limit, dim in zip(self._limits, self._shape)
        ):
            raise ValueError()

    def local_to_global(self, indices: ArrayLike) -> NDArray[np.int_]:
        """Map local indices to global indices.

        Parameters
        ----------
        indices : array_like of int
            Local indices to convert.

        Returns
        -------
        ndarray of int
            Global indices corresponding to ``indices``.
        """
        coords = np.unravel_index(
            np.asarray(indices, dtype=int),
            [limit[1] - limit[0] for limit in self._limits],
        )
        return np.ravel_multi_index(
            [coords[dim] + self._limits[dim][0] for dim in range(len(coords))],
            self._shape,
        )

    def global_to_local(self, indices: ArrayLike) -> NDArray[np.int_]:
        """Map global indices to local indices.

        Parameters
        ----------
        indices : array_like of int
            Global indices to convert.

        Returns
        -------
        ndarray of int
            Local indices within the submatrix.
        """
        coords = np.unravel_index(np.asarray(indices, dtype=int), self._shape)
        return np.ravel_multi_index(
            [coords[dim] - self._limits[dim][0] for dim in range(len(coords))],
            [limit[1] - limit[0] for limit in self._limits],
        )


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


def _submatrix_bounds(
    array: ArrayLike,
    value: int | None = None,
    halo: int = 0,
) -> list[tuple[int, int]]:
    """Find the bounds of a submatrix.

    Parameters
    ----------
    array : array_like
        Array to search for the submatrix.
    value : int or None, optional
        Value defining the submatrix. If ``None`` any non-zero entry is used.
    halo : int, optional
        Number of cells to extend around the submatrix.

    Returns
    -------
    list[tuple[int, int]]
        Start and stop indices for each dimension.

    Examples
    --------
    >>> from landlab_parallel import _submatrix_bounds
    >>> partitions = [
    ...     [0, 0, 1, 1, 1],
    ...     [0, 0, 0, 1, 1],
    ...     [0, 2, 2, 1, 1],
    ...     [3, 3, 2, 2, 1],
    ...     [3, 3, 2, 2, 2],
    ... ]
    >>> _submatrix_bounds(partitions, 2)
    [(2, 5), (1, 5)]
    >>> _submatrix_bounds(partitions, 3, halo=1)
    [(2, 5), (0, 3)]
    >>> bounds = _submatrix_bounds(partitions, 3, halo=1)

    >>> partitions = np.asarray(partitions)
    >>> partitions[slice(*bounds[0]), slice(*bounds[1])]
    array([[0, 2, 2],
           [3, 3, 2],
           [3, 3, 2]])
    """
    array = np.asarray(array)

    if value is None:
        indices = np.nonzero(array)
    else:
        indices = np.nonzero(array == value)

    return [
        (
            int(max(indices[dim].min() - halo, 0)),
            int(min(indices[dim].max() + halo + 1, array.shape[dim])),
        )
        for dim in range(array.ndim)
    ]


def create_landlab_grid(
    partitions: ArrayLike,
    spacing: float | tuple[float, float] = 1.0,
    ij_of_lower_left: tuple[int, int] = (0, 0),
    id_: int = 0,
    mode="raster",
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
    mode : {"raster", "odd-r", "d4"}, optional
        Grid type describing connectivity.

    Returns
    -------
    landlab.ModelGrid
        The constructed grid with boundary conditions set.
    """
    is_their_node = np.asarray(partitions) != id_

    if mode == "odd-r":
        if not isinstance(spacing, float):
            raise ValueError("spacing must be scalar for odd-r layout")
        shift: float = 0.5 if ij_of_lower_left[0] % 2 else 0.0
        xy_of_lower_left = (
            (ij_of_lower_left[1] + shift) * spacing,
            ij_of_lower_left[0] * spacing * np.sqrt(3.0) / 2.0,
        )
    elif mode == "raster":
        xy_of_lower_left = tuple(np.multiply(ij_of_lower_left, spacing))

    if mode in ("d4", "raster"):
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

    grid.status_at_node[is_their_node] = np.where(
        is_ghost_node[is_their_node],
        landlab.NodeStatus.FIXED_VALUE,
        landlab.NodeStatus.CLOSED,
    )

    return grid


def _d4_ghosts(partitions: ArrayLike) -> NDArray[np.bool_]:
    """Identify nodes that are ghost nodes.

    Parameters
    ----------
    partitions : array_like of int
        Partition matrix describing ownership of each node.

    Returns
    -------
    ndarray or bool
        Nodes that are ghosts.

    Examples
    --------
    >>> import numpy as np
    >>> from landlab_parallel import _d4_ghosts

    >>> partitions = [
    ...     [0, 0, 1, 1, 1],
    ...     [0, 0, 0, 1, 1],
    ...     [0, 2, 2, 1, 1],
    ...     [3, 3, 2, 2, 1],
    ...     [3, 3, 2, 2, 2],
    ... ]
    >>> _d4_ghosts(partitions).astype(int)
    array([[0, 1, 1, 0, 0],
           [0, 1, 1, 1, 0],
           [1, 1, 1, 1, 0],
           [1, 1, 1, 1, 1],
           [0, 1, 1, 0, 1]])

    Ghost nodes of partition 1.

    >>> is_partition_1 = np.asarray(partitions) == 1
    >>> (_d4_ghosts(is_partition_1) & ~is_partition_1).astype(int)
    array([[0, 1, 0, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 0, 1, 0],
           [0, 0, 0, 0, 1]])
    """
    partitions = np.pad(
        partitions,
        pad_width=((1, 1), (1, 1)),
        mode="edge",
    )

    right = partitions[2:, 1:-1]
    top = partitions[1:-1, 2:]
    left = partitions[:-2, 1:-1]
    bottom = partitions[1:-1, :-2]

    core = partitions[1:-1, 1:-1]

    return (core != right) | (core != top) | (core != left) | (core != bottom)


def _d8_ghosts(partitions: ArrayLike) -> NDArray[np.bool_]:
    """Identify nodes that are ghost nodes, considering diagonals.

    Parameters
    ----------
    partitions : array_like of int
        Partition matrix describing ownership of each node.

    Returns
    -------
    ndarray or bool
        Nodes that are ghosts.

    Examples
    --------
    >>> import numpy as np
    >>> from landlab_parallel import _d8_ghosts

    >>> partitions = [
    ...     [0, 0, 1, 1, 1],
    ...     [0, 0, 0, 1, 1],
    ...     [0, 2, 2, 1, 1],
    ...     [3, 3, 2, 2, 1],
    ...     [3, 3, 2, 2, 2],
    ... ]
    >>> _d8_ghosts(partitions).astype(int)
    array([[0, 1, 1, 1, 0],
           [1, 1, 1, 1, 0],
           [1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1],
           [0, 1, 1, 1, 1]])

    Ghost nodes of partition 1.

    >>> is_partition_1 = np.asarray(partitions) == 1
    >>> (_d8_ghosts(is_partition_1) & ~is_partition_1).astype(int)
    array([[0, 1, 0, 0, 0],
           [0, 1, 1, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 1, 1, 0],
           [0, 0, 0, 1, 1]])
    """
    partitions = np.pad(
        partitions,
        pad_width=((1, 1), (1, 1)),
        mode="edge",
    )

    right = partitions[1:-1, 2:]
    top_right = partitions[2:, 2:]
    top = partitions[2:, 1:-1]
    top_left = partitions[2:, :-2]
    left = partitions[1:-1, :-2]
    bottom_left = partitions[:-2, :-2]
    bottom = partitions[:-2, 1:-1]
    bottom_right = partitions[:-2, 2:]

    core = partitions[1:-1, 1:-1]

    neighbors = np.stack(
        [right, top_right, top, top_left, left, bottom_left, bottom, bottom_right]
    )

    return np.any(core != neighbors, axis=0)


def _odd_r_ghosts(partitions: ArrayLike) -> NDArray[np.bool_]:
    """Identify nodes that are ghost nodes on an odd-r layout.

    Parameters
    ----------
    partitions : array_like of int
        Partition matrix describing ownership of each node.

    Returns
    -------
    ndarray or bool
        Nodes that are ghosts.

    Examples
    --------
    >>> import numpy as np
    >>> from landlab_parallel import _odd_r_ghosts

    >>> partitions = [
    ...     [0, 0, 1, 1, 1],
    ...     [0, 0, 0, 1, 1],
    ...     [0, 2, 2, 1, 1],
    ...     [3, 3, 2, 2, 1],
    ...     [3, 3, 2, 2, 2],
    ... ]
    >>> _odd_r_ghosts(partitions).astype(int)
    array([[0, 1, 1, 1, 0],
           [1, 1, 1, 1, 0],
           [1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1],
           [0, 1, 1, 0, 1]])

    Ghost nodes of partition 1.

    >>> is_partition_1 = np.asarray(partitions) == 1
    >>> (_odd_r_ghosts(is_partition_1) & ~is_partition_1).astype(int)
    array([[0, 1, 0, 0, 0],
           [0, 1, 1, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 1, 1, 0],
           [0, 0, 0, 0, 1]])
    """
    partitions = np.pad(partitions, pad_width=((1, 1), (1, 1)), mode="edge")

    right = partitions[1:-1, 2:]
    top_right = partitions[2:, 2:]
    top = partitions[2:, 1:-1]
    top_left = partitions[2:, :-2]
    left = partitions[1:-1, :-2]
    bottom_left = partitions[:-2, :-2]
    bottom = partitions[:-2, 1:-1]
    bottom_right = partitions[:-2, 2:]

    core = partitions[1:-1, 1:-1]

    row_indices = np.indices(core.shape)[0]
    is_even_row = (row_indices % 2) == 0
    is_odd_row = ~is_even_row

    is_ghost = np.zeros_like(core, dtype=bool)

    for neighbor in (right, top, top_left, left, bottom_left, bottom):
        is_ghost[is_even_row] |= core[is_even_row] != neighbor[is_even_row]

    for neighbor in (right, top_right, top, left, bottom, bottom_right):
        is_ghost[is_odd_row] |= core[is_odd_row] != neighbor[is_odd_row]

    return is_ghost


def vtu_dump(
    grid: landlab.ModelGrid,
    stream: IO[str] | None = None,
    include: str = "*",
    exclude: Sequence[str] | None = None,
    z_coord: float = 0.0,
    at: str = "node",
) -> str | None:
    """Return a VTU representation of a grid.

    Parameters
    ----------
    grid : landlab.ModelGrid
        Grid containing fields to output.
    stream : file-like object or None, optional
        Stream to write VTU data to. If ``None`` the string is returned.
    include : sequence of str or "*", optional
        Fields to include.
    exclude : sequence of str or None, optional
        Fields to exclude.
    z_coord : float, optional
        Elevation to use for 2D grids.
    at : {"node", "cell"}, optional
        Location of values within the grid.

    Returns
    -------
    str or None
        VTU representation or ``None`` if written to ``stream``.
    """
    mask = grid.status_at_node == grid.BC_NODE_IS_CLOSED
    saved_fields = {
        name: grid.at_node[name]
        for name in grid.at_node
        if np.issubdtype(grid.at_node[name].dtype, np.floating)
    }
    for name, array in saved_fields.items():
        grid.at_node[name] = array.copy()
        grid.at_node[name][mask] = np.nan

    with tempfile.NamedTemporaryFile(suffix=".vtk", mode="w+", delete=False) as tmp:
        tmp.write(
            landlab.io.legacy_vtk.dump(
                grid, include=include, exclude=exclude, z_coord=z_coord, at=at
            )
        )
        tmp.flush()
        mesh = meshio.read(tmp.name)

    for name, array in saved_fields.items():
        grid.at_node[name] = array

    with tempfile.NamedTemporaryFile(suffix=".vtu", mode="r+", delete=False) as tmp:
        tmp.close()
        meshio.write(tmp.name, mesh)
        with open(tmp.name, encoding="utf-8") as f:
            contents = f.read()

    content = "\n".join(
        [
            line
            for line in minidom.parseString(contents)
            .toprettyxml(indent="  ")
            .splitlines()
            if line.strip()
        ]
    )

    if stream is None:
        return content
    else:
        stream.write(content)
        return None


def pvtu_dump(grid: landlab.ModelGrid, vtu_files: Sequence[str] = ()) -> str:
    """Return a PVTU file referencing multiple VTU files.

    Parameters
    ----------
    grid : landlab.ModelGrid
        Grid from which point fields are derived.
    vtu_files : sequence of str, optional
        Paths to VTU files that comprise the full grid.

    Returns
    -------
    str
        XML string describing the parallel unstructured grid.
    """
    vtkfile = ET.Element(
        "VTKFile", type="PUnstructuredGrid", version="1.0", byte_order="LittleEndian"
    )
    pgrid = ET.SubElement(vtkfile, "PUnstructuredGrid", GhostLevel="0")

    ppoints = ET.SubElement(pgrid, "PPoints")
    ET.SubElement(
        ppoints, "PDataArray", type="Float32", NumberOfComponents="3", Name="Points"
    )

    if grid.at_node:
        pdata = ET.SubElement(pgrid, "PPointData")
        for name in grid.at_node:
            ET.SubElement(
                pdata, "PDataArray", type="Float32", Name=name, NumberOfComponents="1"
            )

    for f in vtu_files:
        ET.SubElement(pgrid, "Piece", Source=os.path.basename(f))

    tree = ET.ElementTree(vtkfile)

    root = tree.getroot()
    if root is None:
        raise RuntimeError("tree has no root")
    parsed = minidom.parseString(ET.tostring(root))

    return parsed.toprettyxml(indent="  ")
