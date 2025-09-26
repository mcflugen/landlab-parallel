from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import DTypeLike
from numpy.typing import NDArray


def build_csr_array(
    rows: Sequence[Sequence[Any]],
    dtype: DTypeLike = None,
):
    """Represent a jagged array in compressed sparse row form.

    Store all row elements in a single 1-D array ``values`` and an
    offsets array ``offset_to_row`` such that the i-th row is
    ``values[offset_to_row[i]:offset_to_row[i+1]]``.

    Parameters
    ----------
    rows : array_like of array_like
        Ragged numeric rows.
    dtype : numpy.dtype, optional
        Target dtype for ``values`` (e.g., ``np.float64``). If omitted, a common
        numeric dtype is inferred across rows using NumPy’s casting rules.
        For completely empty input, defaults to ``float64``.

    Returns
    -------
    offset_to_row : ndarray of int64 of shape (n_rows + 1,)
        CSR-style start indices into ``values``.
    values : ndarray of shape (n_elements,)
        Concatenation of all row elements, cast to ``dtype`` if provided.

    Examples
    --------
    >>> offset, data = build_csr_array([[0, 1], [5, 6, 7], [], [9]])
    >>> offset
    array([0, 2, 5, 5, 6])
    >>> data
    array([0, 1, 5, 6, 7, 9])
    >>> build_csr_array([[]])
    (array([0, 0]), array([], dtype=float64))
    >>> build_csr_array([])
    (array([0]), array([], dtype=float64))
    """
    n_rows = len(rows)
    if n_rows == 0:
        return np.array([0], dtype=np.int64), np.array([], dtype=dtype)

    length_of_row = [len(row) for row in rows]
    offset_to_row = np.empty(n_rows + 1, dtype=np.int64)
    offset_to_row[0] = 0
    np.cumsum(length_of_row, out=offset_to_row[1:])

    if offset_to_row[-1] == 0:
        return offset_to_row, np.array([], dtype=dtype)

    if dtype is None:
        for row in range(n_rows):
            if len(rows[row]) > 0:
                first_non_empty_row = row
                break
        dtype = np.asarray(rows[first_non_empty_row]).dtype
        for row in range(first_non_empty_row + 1, n_rows):
            if len(rows[row]):
                dtype = np.result_type(dtype, np.asarray(rows[row]).dtype)

    values = np.empty(offset_to_row[-1], dtype=dtype)
    for row in range(n_rows):
        values[offset_to_row[row] : offset_to_row[row + 1]] = rows[row]

    return offset_to_row, values


def roll_values(
    offset_to_row: ArrayLike, values: ArrayLike, direction: Literal["left", "right"]
) -> NDArray:
    """Roll the values by one place for each row of a matrix in CSR form.

    Parameters
    ----------
    offset_to_row : array_like of int
        Offsets to rows.
    values : array_like
        Array of values to roll.
    direction : {"left", "right"}
        Direction to roll values.

    Examples
    --------
    >>> roll_values([0, 2, 5], [0, 1, 2, 3, 4], direction="left")
    array([1, 0, 3, 4, 2])
    >>> roll_values([0, 2, 5], [0, 1, 2, 3, 4], direction="right")
    array([1, 0, 4, 2, 3])
    >>> roll_values([0, 2, 5], [1.0, 2.0, 3.0, 4.0, 5.0], direction="right")
    array([2., 1., 5., 3., 4.])
    """
    offset_to_row = np.asarray(offset_to_row)
    values = np.asarray(values)

    n_items_per_row = np.diff(offset_to_row)

    is_non_empty_row = n_items_per_row > 0

    first_item_at_row = offset_to_row[:-1][is_non_empty_row]
    last_item_at_row = (offset_to_row[1:] - 1)[is_non_empty_row]

    out = np.empty_like(values)
    if direction == "right":
        out[1:] = values[:-1]
        out[first_item_at_row] = values[last_item_at_row]
    elif direction == "left":
        out[:-1] = values[1:]
        out[last_item_at_row] = values[first_item_at_row]
    else:
        raise ValueError(f"direction must be either 'left' or 'right' ({direction!r})")

    return out


def map_reverse_pairs(
    pairs: ArrayLike,
    if_missing: Literal["raise", "ignore"] | int = "raise",
) -> NDArray[np.int64]:
    """Map each pair (a, b) to its reverse (b, a).

    Parameters
    ----------
    pairs : array_like of int or shape (n_pairs, 2)
        Pairs of integers.
    if_missing : {"raise", "ignore"} or int, optional
        What to do if a pair's reverse is missing:
        - "raise": raise ValueError
        - "ignore": leave as -1
        - int: fill missing entries with this integer

    Returns
    -------
    ndarray[int64]
        For each pair i, the index j such that (pairs[i,:] -> pairs[j, ::-1]).
    """
    int_pair = np.dtype([("tail", np.int64), ("head", np.int64)])

    pairs = np.ascontiguousarray(pairs, dtype=np.int64)
    pairs = pairs.view(dtype=int_pair).reshape(-1)

    if if_missing in ("ignore", "raise"):
        fill_value = -1
    elif isinstance(if_missing, int):
        fill_value = if_missing
    else:
        raise ValueError(
            "bad value for if_missing keyword, must be 'ignore', 'raise' or an integer"
            f" ({if_missing!r})"
        )

    n_pairs = pairs.shape[0]

    reversed_pairs = np.empty_like(pairs)
    reversed_pairs["tail"] = pairs["head"]
    reversed_pairs["head"] = pairs["tail"]

    sorted_indices = np.argsort(pairs)
    sorted_pairs = pairs[sorted_indices]

    indices = np.searchsorted(sorted_pairs, reversed_pairs)

    found = np.zeros(n_pairs, dtype=bool)

    has_a_reverse = indices < n_pairs
    found[has_a_reverse] = (
        sorted_pairs[indices[has_a_reverse]] == reversed_pairs[has_a_reverse]
    )

    out = np.full(n_pairs, fill_value, dtype=np.int64)
    if np.any(found):
        out[found] = sorted_indices[indices[found]]

    if if_missing == "raise" and not np.all(found):
        raise ValueError("Some pairs have no reverse.")

    return out


def unique_pairs(pairs: ArrayLike) -> NDArray:
    """Find unique pairs, ignoring pair ordering.

    Parameters
    ----------
    pairs : (n, 2) array_like
        Each row is a pair. Elements are sorted within each row so (a, b) == (b, a).

    Returns
    -------
    (m, 2) ndarray
        Unique pairs.

    Examples
    --------
    >>> nodes_at_link = np.asarray(
    ...     [
    ...         [0, 1],
    ...         [2, 4],
    ...         [1, 0],
    ...         [3, 2],
    ...     ]
    ... )
    >>> unique_pairs(nodes_at_link)
    array([[0, 1], [2, 3], [2, 4]])
    """
    pairs = np.asarray(pairs)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must be a 2D array with shape (n_pairs, 2)")

    normalized_pairs = np.sort(pairs, axis=1)
    pair_dtype = np.dtype(
        [("a", normalized_pairs.dtype), ("b", normalized_pairs.dtype)]
    )
    records = normalized_pairs.view(pair_dtype).ravel()

    return np.unique(records).view(pairs.dtype).reshape((-1, 2))


def wedge_is_inside_target(
    indptr: ArrayLike,
    tail: ArrayLike,
    head: ArrayLike,
    is_in_target: ArrayLike,
    side: Literal["left", "right"],
):
    """Determine whether each edge's wedge lies entirely inside a target partition.

    A *wedge* is defined as the triangle formed by an edge and one of its
    neighboring edges in the adjacency structure. This function checks,
    for each edge, whether the edge's tail node, head node, and its
    left- or right-adjacent neighbor node are all marked as being inside
    the target partition.

    Parameters
    ----------
    indptr : (n_nodes + 1,) array_like of int
        CSR row pointer array. For each node *i*, the neighbors of *i*
        are stored in ``head[indptr[i]:indptr[i+1]]``.
    tail : (n_edges,) array_like of int
        Tail node index for each edge. Must align with ``head``.
    head : (n_edges,) array_like of int
        Head node index for each edge. Must align with ``tail``.
    is_in_target : (n_nodes,) array_like of bool
        Boolean mask indicating which nodes are in the target partition.
    side : {'left', 'right'}
        Which wedge to check. For an edge (tail → head), the wedge is
        formed with the next neighbor of ``head`` when traversed in the
        given direction around the adjacency list:
        - 'left': use the clockwise neighbor (implemented as a roll right)
        - 'right': use the counter-clockwise neighbor (implemented as a roll left)

    Returns
    -------
    inside_wedge : (n_edges,) ndarray of bool
        Boolean mask indicating, for each edge, whether the wedge defined by
        (tail, head, neighbor) lies entirely inside the target partition.

    Examples
    --------
    >>> from landlab_parallel.adjacency import _get_d4_adjacency

    >>> nodes = [
    ...     [0, 1, 2],
    ...     [3, 4, 5],
    ... ]
    >>> partitions = [
    ...     [1, 1, 0],
    ...     [1, 1, 0],
    ... ]
    >>> is_in_target = np.asarray(partitions) == 1

    >>> adjacency = _get_d4_adjacency((2, 3))
    >>> indptr, head = build_csr_array(adjacency)
    >>> tail = np.repeat(np.arange(6), np.diff(indptr))

    >>> wedge_is_inside_target(indptr, tail, head, is_in_target, side="left").astype(
    ...     int
    ... )
    array([1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0])
    >>> wedge_is_inside_target(indptr, tail, head, is_in_target, side="right").astype(
    ...     int
    ... )
    array([1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0])
    """
    indptr = np.asarray(indptr, dtype=np.intp).ravel()
    tail = np.asarray(tail, dtype=np.intp).ravel()
    head = np.asarray(head, dtype=np.intp).ravel()
    is_in_target = np.asarray(is_in_target, dtype=bool).ravel()

    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right' (got {side!r})")
    if tail.size != head.size:
        raise ValueError(
            f"tail and head must have the same size ({tail.size} vs {head.size})"
        )
    if indptr.size == 0:
        raise ValueError("indptr must be a non-empty 1D array")
    if tail.size == 0:
        return np.zeros(0, dtype=bool)

    map_side_to_direction: dict[str, Literal["left", "right"]] = {
        "left": "right",
        "right": "left",
    }
    neighbor = roll_values(indptr, head, direction=map_side_to_direction[side])

    return np.logical_and.reduce(
        (
            is_in_target[tail],
            is_in_target[head],
            is_in_target[neighbor],
        )
    )
