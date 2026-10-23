"""Enumerations shared by the simulator, controllers and any deployment adapter."""

from __future__ import annotations

from enum import Enum, IntEnum


class RobotType(str, Enum):
    UGV = "ugv"
    UAV = "uav"


class Terrain(IntEnum):
    ROAD = 0
    VEGETATION = 1  # all off-road, non-urban ground; not drivable by UGVs
    URBAN = 2


class Layer(IntEnum):
    """Visibility layers that a cell's probability mass is split across.

    A target hidden in a cell sits in exactly one layer. A sensor can only clear the
    layers it can see, which is how "urban areas must be visited by both robot
    types" is modelled: urban mass is split between AERIAL (rooftops, courtyards)
    and GROUND (street level, under overhangs).
    """

    SHARED = 0  # visible to air and ground sensors
    AERIAL = 1  # visible from the air only
    GROUND = 2  # visible from the ground only


N_LAYERS = len(Layer)


class RobotMode(str, Enum):
    ACTIVE = "active"  # UGV: driving or parked. UAV: airborne.
    DOCKING = "docking"  # UAV landing on a stationary UGV
    DOCKED = "docked"  # UAV carried by the UGV, its battery swapped after landing
    UNDOCKING = "undocking"  # UAV taking off from a stationary UGV
    DEPLETED = "depleted"  # out of energy; permanently inactive
