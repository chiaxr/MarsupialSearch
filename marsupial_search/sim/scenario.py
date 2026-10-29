"""Scenario container: map, graphs, prior, fleet, (hidden) ground-truth target and road-map mismatches."""

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

from .visibility import Occlusion


@dataclass(frozen=True)
class MapMismatch:
    """Where the true road network differs from the prior map ``Scenario.graphs[UGV]`` (the one the
    robots are given). Hidden from controllers: they find it with their sensors."""

    blocked: tuple[int, ...] = ()  # prior road nodes that cannot be traversed
    # road nodes missing from the prior map: index (from its n_nodes on) -> position
    new_nodes: dict[int, tuple[float, float]] = field(default_factory=dict)
    new_edges: tuple[tuple[int, int], ...] = ()  # their edges

    def __bool__(self) -> bool:
        return bool(self.blocked or self.new_nodes)

    def apply(self, prior: TraversabilityGraph, blocked: bool = True) -> TraversabilityGraph:
        """The true road graph (``blocked=False``: with the missing roads, but every node passable)."""
        return prior.with_changes(self.blocked if blocked else (), self.new_nodes, self.new_edges)


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
    road_mismatch: MapMismatch = field(default_factory=MapMismatch)
    # The terrain as the robots know it (None: ``terrain``): roads missing from the map are vegetation.
    map_terrain: Optional[np.ndarray] = None
    # What hides the target and the roads from the sensors (the simulator's: never shown to the robots; None: nothing).
    occlusion: Optional[Occlusion] = None

    def true_road(self, blocked: bool = True) -> TraversabilityGraph:
        """The road network as it really is (see :meth:`MapMismatch.apply`)."""
        return self.road_mismatch.apply(self.graphs[RobotType.UGV], blocked)

    def mission_setup(self, dt: float) -> MissionSetup:
        return MissionSetup(
            grid=self.grid,
            terrain=(self.terrain if self.map_terrain is None else self.map_terrain).copy(),
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
