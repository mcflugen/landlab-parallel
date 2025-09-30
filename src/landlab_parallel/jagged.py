from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray


def validate_jagged(
    offsets: ArrayLike,
    values: ArrayLike,
    n_rows: int | None = None,
) -> tuple[NDArray, NDArray]:
    """Validate an offsets/values pair representing a jagged array.

    Parameters
    ----------
    offsets : array_like
        CSR row pointer array.
    values : array_like
        Values for rows.
    n_rows : int, optional
        If the number of rows is known, validate with this value.

    Returns
    -------
    tuple of (ndarray, ndarray)
        The CSR pair as numpy arrays

    Examples
    --------
    >>> validate_jagged([0, 2, 5], [0.0, 1, 5, 6, 7])
    (array([0, 2, 5]), array([0., 1., 5., 6., 7.]))

    >>> validate_jagged([0, 2, 4], [0.0, 1, 5, 6, 7])
    Traceback (most recent call last):
    ...
    ValueError: mismatch in length of values array (5) and the last item of offsets (4).
    """
    offsets = np.asarray(offsets)
    values = np.asarray(values)

    if offsets.ndim != 1:
        raise ValueError(f"offsets must be a 1D array ({offsets.ndim})")
    if offsets.size == 0:
        raise ValueError("offsets must have length >= 1 (got length 0)")
    if not np.issubdtype(offsets.dtype, np.integer):
        raise ValueError(f"offsets must be an array of int ({offsets.dtype})")
    if offsets[0] != 0:
        raise ValueError(f"first value of offsets must be 0 ({offsets[0]})")
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("offsets must be non-decreasing")
    if n_rows is not None and offsets.size != n_rows + 1:
        raise ValueError(
            f"size of offsets ({offsets.size}) does not match the given"
            f" number of rows ({n_rows})"
        )

    if values.ndim != 1:
        raise ValueError(f"values must be a 1D array ({values.ndim})")
    if values.size != offsets[-1]:
        raise ValueError(
            f"mismatch in length of values array ({values.size}) and the last item"
            f" of offsets ({offsets[-1]})."
        )

    return offsets, values
