"""What occlusion lets a sensor see: per cell, the chance a look detects the target, and per road
node, the chance a look sees whether it is blocked.

A sensor looks from one of two *viewpoints*: ``"air"`` (a UAV flying high: it sees down onto
roofs and open ground, into a street only from near overhead, and nothing under the canopy)
or ``"ground"`` (the UGV, or a UAV flying at street level with a front-facing camera: it sees
street level and under the canopy, through the trees with some loss, but never onto a roof).

:class:`Occlusion` holds what hides things, per cell, as known from the prior map. Persistent
occlusion (the target under a tree, on a roof, at street level) is in the belief's layers: the
generator splits each cell's mass across them from the occlusion and the target class. What
varies from look to look is here:

* from the air, street level (``Layer.GROUND`` in a town) is seen on a pass with probability
  ``min(1, 1 / (2 · canyon))``: the share of a 45° camera's footprint from which the street
  floor clears the buildings either side (half a street width over the building height, per
  metre of altitude). It does not depend on the altitude, as the footprint grows with it.
  Outside towns, ``Layer.GROUND`` is what the canopy hides: never seen from the air;
* from the ground, the trees in a cell hide it with probability ``ground_veg_occlusion ·
  canopy``;
* the target class scales each look by ``air_p`` or ``ground_p`` (a person is harder to make
  out from the air than a car);
* a road node is seen from the air with probability ``1 - canopy`` (times the street term in a
  town), and always from the ground.

Each times the sensor's own per-look probability (:meth:`SensorSpec.layer_p`). With no
occlusion (``None``), a look detects with ``layer_p()`` alone, everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .grid import GridSpec
from .messages import RobotSpec
from .types import Layer, RobotType, Terrain

AIR, GROUND = "air", "ground"


@dataclass(frozen=True)
class Occlusion:
    """What hides the target and the roads, per cell (as the prior map shows it)."""

    canopy: np.ndarray  # (H, W) tree cover in [0, 1]
    canyon: np.ndarray  # (H, W) in towns: building height over street width (0 elsewhere)
    ground_veg_occlusion: float = 0.0  # from the ground, the trees hide this share of the canopy
    air_p: float = 1.0  # the target class: per-look factor from the air ...
    ground_p: float = 1.0  # ... and from the ground


def viewpoint(spec: RobotSpec) -> str:
    """The sensor's viewpoint: as set, else ``"ground"`` for a UGV and ``"air"`` for a UAV."""
    return spec.sensor.viewpoint or (GROUND if spec.robot_type == RobotType.UGV else AIR)


def street_visibility(canyon: np.ndarray) -> np.ndarray:
    """From the air: the chance a pass sees street level, per cell (1 where ``canyon`` is 0)."""
    return np.minimum(1.0, 1.0 / np.maximum(2.0 * np.asarray(canyon, dtype=float), 1e-12))


def detection_map(spec: RobotSpec, terrain: np.ndarray, occlusion: Optional[Occlusion]) -> np.ndarray:
    """Per layer and cell, shape (N_LAYERS, H, W): the chance a look by ``spec``'s sensor detects
    a target there (module docstring)."""
    p = np.broadcast_to(spec.sensor.layer_p()[:, None, None], (len(spec.sensor.layer_p()),) + terrain.shape).copy()
    if occlusion is None:
        return p
    urban = terrain == int(Terrain.URBAN)
    if viewpoint(spec) == AIR:
        p[Layer.GROUND] *= np.where(urban, street_visibility(occlusion.canyon), 0.0)
        p *= occlusion.air_p
    else:
        p *= 1.0 - occlusion.ground_veg_occlusion * occlusion.canopy
        p *= occlusion.ground_p
    return p


def road_observability(spec: RobotSpec, grid: GridSpec, terrain: np.ndarray, occlusion: Optional[Occlusion],
                       xy: np.ndarray) -> np.ndarray:
    """Per point of ``xy`` (n, 2), a road node: the chance a look by ``spec``'s sensor sees whether
    it is blocked (or that a road is there)."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    if occlusion is None or viewpoint(spec) == GROUND or not len(xy):
        return np.ones(len(xy))
    known = np.isfinite(xy).all(1)  # (a node not yet found has no position: never looked at)
    xy = np.where(known[:, None], xy, grid.origin)
    cols = np.clip(((xy[:, 0] - grid.origin[0]) // grid.resolution).astype(int), 0, grid.cols - 1)
    rows = np.clip(((xy[:, 1] - grid.origin[1]) // grid.resolution).astype(int), 0, grid.rows - 1)
    o = 1.0 - occlusion.canopy[rows, cols]
    urban = terrain[rows, cols] == int(Terrain.URBAN)
    return np.where(known, o * np.where(urban, street_visibility(occlusion.canyon[rows, cols]), 1.0), 0.0)
