"""Layered Bayesian belief over the target location (stationary target)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .grid import GridSpec
from .messages import LocalBeliefUpdate


class BeliefMap:
    """Unnormalised posterior mass ``P(target in layer l of cell, not yet detected)``.

    ``mass`` has shape ``(N_LAYERS, H, W)`` and starts as ``prior * layer_fractions``.
    Each negative observation multiplies the observed window by ``1 - p_detect``; the
    remaining total mass is ``1 - P(detected so far)``.
    """

    def __init__(self, grid: GridSpec, prior: np.ndarray, layer_fractions: np.ndarray):
        self.grid = grid
        self.initial_mass = prior[None, :, :] * layer_fractions
        self.mass = self.initial_mass.copy()
        self.initial_total = float(self.initial_mass.sum())
        self.detection_xy: Optional[tuple[float, float]] = None

    def apply(self, update: LocalBeliefUpdate) -> None:
        if update.detected:
            self.detection_xy = update.detection_xy
            return
        _, h, w = update.p_detect.shape
        if h == 0 or w == 0:
            return
        window = self.mass[:, update.row0 : update.row0 + h, update.col0 : update.col0 + w]
        window *= 1.0 - update.p_detect

    def cell_mass(self, layers=None) -> np.ndarray:
        """Remaining (unnormalised) mass per cell, optionally restricted to some layers."""
        if layers is None:
            return self.mass.sum(axis=0)
        return self.mass[list(layers)].sum(axis=0)

    def posterior(self) -> np.ndarray:
        """Normalised P(target in cell | all observations so far)."""
        m = self.cell_mass()
        total = m.sum()
        return m / total if total > 0 else m

    def searched_probability(self) -> float:
        """Prior probability mass already searched (cumulative detection probability)."""
        return 1.0 - float(self.mass.sum()) / self.initial_total

    def covered_cells(self, tol: float = 1e-6) -> np.ndarray:
        """Boolean (H, W): cells whose remaining mass is below ``tol`` of their initial mass."""
        init = self.initial_mass.sum(axis=0)
        return self.cell_mass() <= tol * np.maximum(init, 1e-300)
