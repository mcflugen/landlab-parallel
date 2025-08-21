from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray


def is_ghost(partitions: ArrayLike, mode: str = "d4") -> NDArray[np.bool_]:
    """Identify nodes that are ghosts of another partition.

    A ghost node is a node that is connected to a node owned
    by another partition.

    Parameters
    ----------
    partitions : array_like
        Partition matrix describing ownership of each node.
    mode : {"d4", "d8", "odd-r"}, optional
        Connectivity scheme used to determine neighbors.

    Examples
    --------
    >>> import numpy as np
    >>> partitions = np.asarray(
    ...     [
    ...         [0, 0, 0, 1, 2, 2, 2],
    ...         [0, 0, 1, 1, 1, 2, 2],
    ...         [0, 1, 1, 1, 1, 1, 2],
    ...     ]
    ... )
    >>> is_ghost(partitions).astype(int)
    array([[0, 0, 1, 1, 1, 0, 0],
           [0, 1, 1, 0, 1, 1, 0],
           [1, 1, 0, 0, 0, 1, 1]])
    """
    mode_handlers = {
        "d4": _d4_ghosts,
        "d8": _d8_ghosts,
        "odd-r": _odd_r_ghosts,
    }

    try:
        get_ghosts = mode_handlers[mode]
    except KeyError:
        valid_choices = ", ".join(repr(key) for key in sorted(mode_handlers))
        raise ValueError(
            f"Mode not understood. Must be one of {valid_choices} but got {mode!r}."
        )

    return get_ghosts(partitions)


def is_ghost_of_partition(
    partitions: ArrayLike, partition: int, mode: str = "d4"
) -> NDArray[np.bool_]:
    """Identify nodes that are ghosts of a given partition.

    Parameters
    ----------
    partitions : array_like
        Partition matrix describing ownership of each node.
    partition : int
        Identifier of the local partition.
    mode : {"d4", "d8", "odd-r"}, optional
        Connectivity scheme used to determine neighbors.

    Examples
    --------
    >>> import numpy as np
    >>> partitions = np.asarray(
    ...     [
    ...         [0, 0, 0, 1, 2, 2, 2],
    ...         [0, 0, 1, 1, 1, 2, 2],
    ...         [0, 1, 1, 1, 1, 1, 2],
    ...     ]
    ... )
    >>> is_ghost_of_partition(partitions, 1).astype(int)
    array([[0, 0, 1, 0, 1, 0, 0],
           [0, 1, 0, 0, 0, 1, 0],
           [1, 0, 0, 0, 0, 0, 1]])
    """
    is_local_partition = np.asarray(partitions) == partition
    return is_ghost(is_local_partition, mode=mode) & (~is_local_partition)


def is_non_ghost_of_partition(
    partitions: ArrayLike, partition: int, mode: str = "d4"
) -> NDArray[np.bool_]:
    """Identify nodes that are only part of a given partition.

    Parameters
    ----------
    partitions : array_like
        Partition matrix describing ownership of each node.
    partition : int
        Identifier of the local partition.
    mode : {"d4", "d8", "odd-r"}, optional
        Connectivity scheme used to determine neighbors.

    Returns
    -------
    ndarray of bool

    Examples
    --------
    >>> import numpy as np
    >>> partitions = np.asarray(
    ...     [
    ...         [0, 0, 0, 1, 2, 2, 2],
    ...         [0, 0, 1, 1, 1, 2, 2],
    ...         [0, 1, 1, 1, 1, 1, 2],
    ...     ]
    ... )
    >>> is_non_ghost_of_partition(partitions, 2).astype(int)
    array([[0, 0, 0, 0, 0, 1, 1],
           [0, 0, 0, 0, 0, 0, 1],
           [0, 0, 0, 0, 0, 0, 0]])
    """
    is_local_partition = np.asarray(partitions) == partition
    return ~is_ghost(is_local_partition, mode=mode) & is_local_partition


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
    mode : {"d4", "d8", "odd-r"}, optional
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
    partitions_array = np.asarray(partitions)
    is_my_node = partitions_array == my_id
    _is_ghost = is_ghost(is_my_node, mode=mode)
    neighbors = np.unique(partitions_array[~is_my_node & _is_ghost])

    return {
        rank: np.ravel_multi_index(
            np.nonzero(_is_ghost & (partitions_array == rank)), partitions_array.shape
        )
        for rank in neighbors
    }


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
    >>> from landlab_parallel.ghosts import _d4_ghosts

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
    >>> from landlab_parallel.ghosts import _d8_ghosts

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
    >>> from landlab_parallel.ghosts import _odd_r_ghosts

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
