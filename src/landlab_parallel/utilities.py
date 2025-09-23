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
