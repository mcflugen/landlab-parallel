from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray


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
