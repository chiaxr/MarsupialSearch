"""Message types exchanged between a mission controller and its environment.

The environment is the simulator during research and a deployment adapter (e.g. a
ROS node translating topics to/from these dataclasses) on hardware. Controllers
must only depend on this module, :mod:`marsupial_search.core.graph`, :mod:`marsupial_search.core.grid`,
:mod:`marsupial_search.core.belief` and :mod:`marsupial_search.core.types`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Sequence, Union

import numpy as np

from .graph import TraversabilityGraph
from .grid import GridSpec
from .types import Layer, RobotMode, RobotType

# --------------------------------------------------------------------------- specs


@dataclass(frozen=True)
class EnergySpec:
    """Linear energy model: ``idle_power * duration + energy_per_m * distance``.

    ``idle_power`` is drawn whenever the robot is powered and not docked, so it
    covers UAV hover while waiting for a rendezvous.
    """

    capacity: float
    idle_power: float  # energy units per second
    energy_per_m: float  # additional energy units per metre travelled

    def travel_cost(self, distance: float, speed: float) -> float:
        return self.idle_power * distance / speed + self.energy_per_m * distance

    def hover_cost(self, duration: float) -> float:
        return self.idle_power * duration


@dataclass(frozen=True)
class SensorSpec:
    range_m: float
    visible_layers: tuple[Layer, ...]
    p_detect: float = 1.0  # detection probability inside the footprint (1 = perfect)


@dataclass(frozen=True)
class RobotSpec:
    robot_id: str
    robot_type: RobotType
    speed: float  # m/s
    energy: EnergySpec
    sensor: SensorSpec

    @property
    def free_motion(self) -> bool:
        """Free-motion robots may fly straight to any point; others must follow graph edges."""
        return self.robot_type == RobotType.UAV


@dataclass(frozen=True)
class DockingSpec:
    dock_duration_s: float = 30.0  # landing time; UGV must stay stationary throughout
    undock_duration_s: float = 15.0  # take-off time; UGV must stay stationary throughout
    swap_duration_s: float = 120.0  # battery swap after landing: the UAV is full once it is done
    swap_while_moving: bool = True  # False: the swap only progresses while the UGV is stationary
    dock_tolerance_m: float = 3.0  # max UAV-UGV separation to start docking


@dataclass(frozen=True)
class MissionSetup:
    """Static information available to the team at mission start."""

    grid: GridSpec
    terrain: np.ndarray  # (H, W) of Terrain values
    layer_fractions: np.ndarray  # (N_LAYERS, H, W): split of each cell's mass across layers
    prior: np.ndarray  # (H, W) global prior P(target in cell), sums to 1
    graphs: dict[RobotType, TraversabilityGraph]
    robots: dict[str, RobotSpec]
    docking: DockingSpec
    dt: float
    # Radio link range per robot type; a link needs both ends in range (None: no radio info).
    radio_range_m: Optional[dict[RobotType, float]] = None

    def robots_of(self, robot_type: RobotType) -> list[str]:
        return [rid for rid, spec in self.robots.items() if spec.robot_type == robot_type]


# -------------------------------------------------------------------- observations


@dataclass(frozen=True)
class RobotState:
    robot_id: str
    robot_type: RobotType
    position: tuple[float, float]
    mode: RobotMode
    energy: float  # current energy estimate
    energy_capacity: float
    moving: bool  # moved during the last step
    node: Optional[int]  # graph node the robot is currently at, if any
    last_node: Optional[int]  # most recently reached graph node
    nodes_reached: tuple[int, ...]  # graph nodes reached during the last step, in order
    path_remaining: tuple[int, ...]  # nodes left in the active FollowPath command
    goal_point: Optional[tuple[float, float]]  # active GoToPoint target, if any
    docked_uav: Optional[str] = None  # UGV only: UAV docked or docking/undocking on it
    host_ugv: Optional[str] = None  # UAV only: UGV it is docking/docked/undocking with
    mode_time_remaining: float = 0.0  # time left in a docking/undocking manoeuvre, or (docked) in the battery swap

    @property
    def energy_fraction(self) -> float:
        return self.energy / self.energy_capacity

    @property
    def idle(self) -> bool:
        return not self.path_remaining and self.goal_point is None


@dataclass(frozen=True)
class LocalBeliefUpdate:
    """Sensor measurement around one robot, expressed on the global grid.

    ``p_detect[l, i, j]`` is the probability that the sensor would have detected a
    target located in layer ``l`` of cell ``(row0 + i, col0 + j)``. If nothing was
    detected the Bayesian update is ``mass *= 1 - p_detect`` over the window.
    """

    robot_id: str
    robot_type: RobotType
    time: float
    row0: int
    col0: int
    p_detect: np.ndarray  # (N_LAYERS, h, w)
    detected: bool = False
    detection_xy: Optional[tuple[float, float]] = None


@dataclass(frozen=True)
class TraversabilityUpdate:
    """Mismatches between a graph in :class:`MissionSetup` (the prior map) and the world, found
    by one robot's sensor (only those it had not found before).

    ``blocked`` are prior nodes that cannot be traversed. ``new_nodes`` (index -> position, indices
    from the prior graph's ``n_nodes`` on) are nodes missing from the prior map, with
    ``new_edges``: each new node's edges, and the position of a new node at their far end even
    if it is out of sensor range. :class:`~marsupial_search.core.graph.GraphKnowledge` keeps track of them.
    """

    robot_id: str
    time: float
    graph: RobotType  # whose graph: MissionSetup.graphs[graph]
    blocked: tuple[int, ...] = ()
    new_nodes: dict[int, tuple[float, float]] = field(default_factory=dict)
    new_edges: tuple[tuple[int, int], ...] = ()


class EventType(str, Enum):
    DOCK_STARTED = "dock_started"
    DOCKED = "docked"
    DOCK_ABORTED = "dock_aborted"
    UNDOCK_STARTED = "undock_started"
    UNDOCKED = "undocked"
    UNDOCK_ABORTED = "undock_aborted"
    COMMAND_REJECTED = "command_rejected"
    PATH_BLOCKED = "path_blocked"  # the next node of a FollowPath cannot be traversed: the robot waits before it
    DEPLETED = "depleted"
    TARGET_SIGHTED = "target_sighted"  # detected by a robot that cannot end the mission
    TARGET_FOUND = "target_found"  # mission success


@dataclass(frozen=True)
class Event:
    time: float
    type: EventType
    robot_id: str
    other_id: Optional[str] = None
    detail: str = ""


@dataclass(frozen=True)
class RadioMessage:
    """A message received over the radio. ``payload`` is algorithm-defined."""

    sender: str
    topic: str  # algorithm-defined channel name, e.g. "claims", "belief", "rendezvous"
    payload: Any
    recipients: Optional[tuple[str, ...]]  # None: broadcast
    sent_time: float
    received_time: float
    size_bytes: int  # encoded payload size
    seq: int  # per-sender sequence number, for ordering and de-duplication


@dataclass(frozen=True)
class Observation:
    time: float
    robots: dict[str, RobotState]
    belief_updates: list[LocalBeliefUpdate] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    mission_complete: bool = False
    radio: dict[str, list[RadioMessage]] = field(default_factory=dict)  # inbox per receiving robot
    map_updates: list[TraversabilityUpdate] = field(default_factory=list)

    def inbox(self, robot_id: str, topic: Optional[str] = None) -> list[RadioMessage]:
        msgs = self.radio.get(robot_id, [])
        return msgs if topic is None else [m for m in msgs if m.topic == topic]

    def local_view(self, robot_id: str) -> "RobotObservation":
        """What ``robot_id`` can know without communication."""
        return RobotObservation(
            time=self.time,
            robot_id=robot_id,
            state=self.robots[robot_id],
            belief_updates=[u for u in self.belief_updates if u.robot_id == robot_id],
            events=[e for e in self.events if robot_id in (e.robot_id, e.other_id)],
            inbox=list(self.radio.get(robot_id, [])),
            mission_complete=self.mission_complete,
            map_updates=[u for u in self.map_updates if u.robot_id == robot_id],
        )


@dataclass(frozen=True)
class RobotObservation:
    """One robot's local view, the input of a decentralised :class:`RobotController`.

    It holds only the robot's own state, its own sensor updates (belief and map), events
    involving it (e.g. a dock abort on its dock), and the radio messages it received. Anything
    about other robots (positions, plans, what they have searched or found) must arrive by radio.
    """

    time: float
    robot_id: str
    state: RobotState
    belief_updates: list[LocalBeliefUpdate]
    events: list[Event]
    inbox: list[RadioMessage]
    mission_complete: bool = False
    map_updates: list[TraversabilityUpdate] = field(default_factory=list)

    def messages(self, topic: Optional[str] = None) -> list[RadioMessage]:
        return self.inbox if topic is None else [m for m in self.inbox if m.topic == topic]


# ------------------------------------------------------------------------ commands
# Motion commands persist until replaced. Dock/Undock are one-shot requests whose
# outcome is reported through events.


@dataclass(frozen=True)
class FollowPath:
    """Traverse graph nodes in order. For graph-bound robots the first node must be
    the current node, a neighbour of it, or an endpoint of the edge being driven."""

    nodes: tuple[int, ...]


@dataclass(frozen=True)
class GoToPoint:
    """Move in a straight line to a point (free-motion robots only)."""

    xy: tuple[float, float]


@dataclass(frozen=True)
class Hold:
    """Stop immediately and stay stationary (UAV: hover)."""


@dataclass(frozen=True)
class Dock:
    """UAV request to land on ``ugv_id``. Requires the UAV to be within
    ``dock_tolerance_m``, the UGV to be stationary and its dock to be free."""

    ugv_id: str


@dataclass(frozen=True)
class Undock:
    """UAV request to take off from its host UGV, which must be stationary."""


@dataclass(frozen=True)
class Transmit:
    """Send ``payload`` over the radio, to ``to`` (robot ids) or to everyone in range.

    Can be issued alongside a motion command: give the robot a list of commands.
    Delivery depends on link range, latency, loss and bandwidth; there is no ack,
    so algorithms that need reliability must re-send.
    """

    topic: str
    payload: Any
    to: Optional[tuple[str, ...]] = None


Command = Union[FollowPath, GoToPoint, Hold, Dock, Undock, Transmit]
# What a controller returns per robot: one command or several (e.g. a motion command plus Transmits).
RobotCommands = Union[Command, Sequence[Command]]
