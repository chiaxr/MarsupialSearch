"""Scenario container: map, graphs, prior, fleet and (hidden) ground-truth target."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from marsupial_search.core import (
    DockingSpec,
    GridSpec,
    Layer,
    MissionSetup,
    RobotSpec,
    RobotType,
    TraversabilityGraph,
)


@dataclass(frozen=True)
class RobotInit:
    node: Optional[int] = None  # start node on the robot's own graph (UGV)
    position: Optional[tuple[float, float]] = None  # start point (free-motion robots)
    docked_on: Optional[str] = None  # UAV: start docked on this UGV


@dataclass
class Scenario:
    name: str
    grid: GridSpec
    terrain: np.ndarray
    layer_fractions: np.ndarray
    prior: np.ndarray
    graphs: dict[RobotType, TraversabilityGraph]
    robots: dict[str, RobotSpec]
    init: dict[str, RobotInit]
    docking: DockingSpec
    target_xy: tuple[float, float]  # ground truth, never shown to controllers
    target_layer: Layer
    meta: dict = field(default_factory=dict)

    def mission_setup(self, dt: float) -> MissionSetup:
        return MissionSetup(
            grid=self.grid,
            terrain=self.terrain.copy(),
            layer_fractions=self.layer_fractions.copy(),
            prior=self.prior.copy(),
            graphs=self.graphs,
            robots=dict(self.robots),
            docking=self.docking,
            dt=dt,
        )

    def with_target(self, target_xy: tuple[float, float], target_layer: Layer) -> "Scenario":
        """Copy of this scenario with a different ground-truth target."""
        from dataclasses import replace

        return replace(self, target_xy=target_xy, target_layer=target_layer)
