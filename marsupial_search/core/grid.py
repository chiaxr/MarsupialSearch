"""Regular 2D grid geometry used for the terrain and belief maps.

Arrays are indexed ``[row, col]`` with row along +y and col along +x, so
``imshow(array, origin="lower", extent=grid.extent)`` displays them correctly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GridSpec:
    origin: tuple[float, float]  # world (x, y) of the lower-left corner of cell (0, 0)
    resolution: float  # cell size in metres
    shape: tuple[int, int]  # (rows, cols)

    @property
    def rows(self) -> int:
        return self.shape[0]

    @property
    def cols(self) -> int:
        return self.shape[1]

    @property
    def extent(self) -> tuple[float, float, float, float]:
        x0, y0 = self.origin
        return (x0, x0 + self.cols * self.resolution, y0, y0 + self.rows * self.resolution)

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        r = int(np.floor((y - self.origin[1]) / self.resolution))
        c = int(np.floor((x - self.origin[0]) / self.resolution))
        return min(max(r, 0), self.rows - 1), min(max(c, 0), self.cols - 1)

    def cell_center(self, r: int, c: int) -> tuple[float, float]:
        return (
            self.origin[0] + (c + 0.5) * self.resolution,
            self.origin[1] + (r + 0.5) * self.resolution,
        )

    def cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (X, Y) arrays of shape ``self.shape`` with every cell centre."""
        xs = self.origin[0] + (np.arange(self.cols) + 0.5) * self.resolution
        ys = self.origin[1] + (np.arange(self.rows) + 0.5) * self.resolution
        return np.meshgrid(xs, ys)

    def disk_window(self, x: float, y: float, radius: float) -> tuple[int, int, np.ndarray]:
        """Cells whose centres lie within ``radius`` of (x, y).

        Returns ``(row0, col0, mask)`` where ``mask`` is a boolean window whose
        top-left element is cell ``(row0, col0)``. The window is clipped to the grid
        and may be empty.
        """
        res = self.resolution
        r_lo = max(int(np.floor((y - radius - self.origin[1]) / res)), 0)
        r_hi = min(int(np.floor((y + radius - self.origin[1]) / res)), self.rows - 1)
        c_lo = max(int(np.floor((x - radius - self.origin[0]) / res)), 0)
        c_hi = min(int(np.floor((x + radius - self.origin[0]) / res)), self.cols - 1)
        if r_hi < r_lo or c_hi < c_lo:
            return r_lo, c_lo, np.zeros((0, 0), dtype=bool)
        ys = self.origin[1] + (np.arange(r_lo, r_hi + 1) + 0.5) * res
        xs = self.origin[0] + (np.arange(c_lo, c_hi + 1) + 0.5) * res
        mask = (ys[:, None] - y) ** 2 + (xs[None, :] - x) ** 2 <= radius**2
        return r_lo, c_lo, mask
