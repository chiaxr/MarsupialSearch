"""Discrete-time 2D simulator for marsupial search: one UGV that carries UAVs on its dock.

Per step (``dt`` seconds), in order:

1. apply controller commands (invalid ones are rejected with an event);
2. move active robots along their commanded path/point at their nominal speed (a UGV
   stops before a road node that is blocked in reality);
3. carry docked UAVs with their host UGV;
4. start requested dock/undock manoeuvres (host UGV must not have moved this step);
5. advance manoeuvres in progress, aborting any whose host UGV moved;
6. drain energy and swap the batteries of docked UAVs, marking robots that run out as DEPLETED;
7. sense: emit a :class:`LocalBeliefUpdate` per sensing robot and check the target, and a
   :class:`TraversabilityUpdate` per robot that sees road-map mismatches it had not seen. Each
   pass is one look: for a layer seen with probability < 1 the update covers only the cells that
   entered the footprint this step, and a sighting is rolled once per pass over the target and
   holds while it stays in view. The probability is per layer and cell: the sensor's own, times
   what occlusion lets it see from its viewpoint (``core.visibility``). A robot of a
   ``terminate_on`` type ends the mission with the target in range (``confirm_test: "range"``) or
   on sighting it (``"sighting"``); one in ``confirm_by`` (a UAV flying low to verify a
   sighting) on sighting it. A road-map mismatch is seen once per pass too, with the chance
   occlusion leaves its node (its own random stream, so it does not depend on the target);
8. radio: queue this step's ``Transmit`` commands and deliver messages that are due.

Robots are given the prior road map (``Scenario.graphs``); the UGV drives on the true one
(``Scenario.true_road()``). A FollowPath may use any edge of either. A UGV whose next node is
blocked stops at the node before it (``PATH_BLOCKED``), finds it blocked, and waits there with
the rest of its path pending until it is given a new command.
"""

from __future__ import annotations

import dataclasses
import time as _time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.spatial import cKDTree

from marsupial_search.core import (
    BeliefMap,
    Command,
    Dock,
    Event,
    EventType,
    FollowPath,
    GoToPoint,
    Hold,
    LocalBeliefUpdate,
    MissionController,
    MissionSetup,
    Observation,
    RadioMessage,
    RobotCommands,
    RobotMode,
    RobotSpec,
    RobotState,
    RobotType,
    Terrain,
    Transmit,
    TraversabilityGraph,
    TraversabilityUpdate,
    Undock,
    detection_map,
    road_observability,
)

from .radio import _SIZE_KEYS as _RADIO_SIZE_KEYS
from .radio import _STAT_KEYS as _RADIO_STAT_KEYS
from .radio import RadioChannel, RadioConfig
from .scenario import Scenario

_EPS = 1e-9


@dataclass
class SimConfig:
    dt: float = 1.0
    max_time_s: float = 6 * 3600.0
    # Robot types whose sensor range ending on the target ends the mission (perfect,
    # distance-only test). Other types report TARGET_SIGHTED via their belief update.
    terminate_on: tuple[RobotType, ...] = (RobotType.UGV,)
    # Robots (by id) whose sighting also ends the mission, e.g. the near scout (uav0), flying low
    # to verify one. The controllers are told who ends it (MissionSetup.confirmers).
    confirm_by: tuple[str, ...] = ()
    # How a terminate_on robot ends the mission: "range" (the target in its sensor range, any layer: a
    # perfect test) or "sighting" (it sights the target, with its own detection probability there)
    confirm_test: str = "range"
    energy_noise_std: float = 0.0  # std of the energy estimate error, fraction of capacity
    log_interval_s: float = 10.0
    belief_snapshot_interval_s: Optional[float] = None  # store posterior snapshots for animation
    stop_when_controller_done: bool = True
    stop_when_ugv_depleted: bool = True  # end once the UGV has run out of energy
    target: bool = True  # False: nothing to find or sight (a target-free run, see marsupial_search.sim.evaluation)
    radio: RadioConfig = field(default_factory=RadioConfig)
    seed: int = 0


class _SimRobot:
    def __init__(self, spec: RobotSpec, graph: TraversabilityGraph, passable: Optional[TraversabilityGraph] = None):
        self.spec = spec
        self.graph = graph  # every node and edge a path may name
        self.passable = passable  # the edges that can really be driven (None: all of graph's)
        self.blocked_by: Optional[int] = None  # the node it stopped in front of this step
        self.energy = spec.energy.capacity
        self.pos = np.zeros(2)
        self.mode = RobotMode.ACTIVE
        self.path: list[int] = []
        self.goal_point: Optional[np.ndarray] = None
        self.at_node: Optional[int] = None
        self.last_node: Optional[int] = None
        self.edge: Optional[tuple[int, int]] = None  # (from, to) while between nodes
        self.moved = 0.0
        self.nodes_reached: list[int] = []
        self.host: Optional[str] = None  # UAV: UGV it is attached to
        self.dock_slot: Optional[str] = None  # UGV: UAV occupying its dock
        self.timer = 0.0
        self.swap_left = 0.0  # UAV: time left in its battery swap
        self.started_this_step = False
        self.odometer = 0.0
        self.energy_used = 0.0

    @property
    def rid(self) -> str:
        return self.spec.robot_id

    def place_at_node(self, node: int) -> None:
        self.pos = self.graph.positions[node].copy()
        self.at_node = self.last_node = node
        self.edge = None

    def move(self, dt: float) -> None:
        self.moved = 0.0
        self.nodes_reached = []
        self.blocked_by = None
        budget = self.spec.speed * dt
        while budget > _EPS and (self.path or self.goal_point is not None):
            if self.path:
                nxt = self.path[0]
                if self.passable is not None and self.at_node is not None and nxt != self.at_node \
                        and not self.passable.has_edge(self.at_node, nxt):
                    self.blocked_by = nxt  # wait here, the rest of the path pending
                    break
                target = self.graph.positions[nxt]
            else:
                target = self.goal_point
            delta = target - self.pos
            d = float(np.hypot(*delta))
            if d <= budget:
                self.pos = target.copy()
                budget -= d
                self.moved += d
                if self.path:
                    n = self.path.pop(0)
                    self.at_node = self.last_node = n
                    self.edge = None
                    self.nodes_reached.append(n)
                else:
                    self.goal_point = None
            else:
                self.pos = self.pos + delta * (budget / d)
                self.moved += budget
                budget = 0.0
                if self.path and not self.spec.free_motion:
                    self.edge = (self.last_node, self.path[0])
                self.at_node = None
        self.odometer += self.moved


@dataclass
class MissionResult:
    scenario_name: str
    controller_name: str
    outcome: str  # "found" | "timeout" | "controller_done" | "ugv_depleted" | "all_depleted"
    time_s: float
    found_by: Optional[str]
    log: dict[str, np.ndarray]
    traj_t: np.ndarray
    trajectories: dict[str, np.ndarray]  # (T, 2) positions per robot
    modes: dict[str, np.ndarray]  # (T,) RobotMode index per robot
    events: list[Event]
    robot_stats: dict[str, dict]
    final_mass: np.ndarray
    belief_snapshots: list[tuple[float, np.ndarray]] = field(default_factory=list)
    wall_time_s: float = 0.0
    # Set by Simulator.run: time spent inside the controller's reset and step calls.
    controller_timing: dict = field(default_factory=dict)

    MODE_ORDER = list(RobotMode)

    def summary(self) -> dict:
        out = {
            "scenario": self.scenario_name,
            "controller": self.controller_name,
            "outcome": self.outcome,
            "found": self.outcome == "found",
            "time_s": self.time_s,
            "found_by": self.found_by or "",
            "searched_prob": float(self.log["searched_prob"][-1]),
            "wall_time_s": self.wall_time_s,
        }
        for t in Terrain:
            key = f"coverage_{t.name.lower()}"
            if key in self.log:
                out[key] = float(self.log[key][-1])
        for k in ("sorties", "dock_aborts", "undock_aborts", "rejected", "path_blocked", "map_changes_found",
                  *_RADIO_STAT_KEYS):
            out[k] = sum(s[k] for s in self.robot_stats.values())
        senders = [s for s in self.robot_stats.values() if s["radio_sent"]]
        out["radio_payload_min_bytes"] = min((s["radio_payload_min_bytes"] for s in senders), default=float("nan"))
        out["radio_payload_avg_bytes"] = (out["radio_bytes_sent"] / out["radio_sent"] if out["radio_sent"]
                                          else float("nan"))
        out["radio_payload_max_bytes"] = max((s["radio_payload_max_bytes"] for s in senders), default=float("nan"))
        out["depleted"] = sum(int(s["depleted"]) for s in self.robot_stats.values())
        out["ugv_distance_m"] = sum(s["distance_m"] for s in self.robot_stats.values() if s["type"] == "ugv")
        out["uav_distance_m"] = sum(s["distance_m"] for s in self.robot_stats.values() if s["type"] == "uav")
        out.update(self.controller_timing)
        steps = self.controller_timing.get("controller_steps", 0)
        out["controller_ms_per_step"] = (1e3 * self.controller_timing["controller_s"] / steps if steps
                                         else float("nan"))
        # Decentralised teams: the busiest robot's onboard compute.
        timed = [s for s in self.robot_stats.values() if s.get("controller_steps")]
        out["robot_min_step_s"] = min((s["controller_min_step_s"] for s in timed), default=float("nan"))
        out["robot_ms_per_step"] = max((1e3 * s["controller_s"] / s["controller_steps"] for s in timed),
                                       default=float("nan"))
        out["robot_max_step_s"] = max((s["controller_max_step_s"] for s in timed), default=float("nan"))
        return out


class Simulator:
    def __init__(self, scenario: Scenario, config: Optional[SimConfig] = None):
        ugvs = [rid for rid, spec in scenario.robots.items() if spec.robot_type == RobotType.UGV]
        if len(ugvs) != 1:
            raise ValueError(f"scenario {scenario.name!r} has {len(ugvs)} UGVs; the simulator models exactly one")
        self.ugv = ugvs[0]
        self.scenario = scenario
        self.config = config or SimConfig()
        radio = self.config.radio
        if not radio.enabled:
            ranges = None
        elif radio.simulate_channel:
            ranges = dict(radio.range_m)
        else:
            ranges = {t: float("inf") for t in RobotType}  # ideal channel: unlimited range
        if self.config.confirm_test not in ("range", "sighting"):
            raise ValueError(f"confirm_test must be 'range' or 'sighting', got {self.config.confirm_test!r}")
        unknown = set(self.config.confirm_by) - set(scenario.robots)
        if unknown:
            raise ValueError(f"confirm_by: no robots {sorted(unknown)} in scenario {scenario.name!r}")
        confirmers = tuple(rid for rid, spec in scenario.robots.items()
                           if spec.robot_type in self.config.terminate_on or rid in self.config.confirm_by)
        self.setup: MissionSetup = dataclasses.replace(scenario.mission_setup(self.config.dt), radio_range_m=ranges,
                                                       confirmers=confirmers, confirm_test=self.config.confirm_test)
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self) -> Observation:
        sc, cfg = self.scenario, self.config
        self.rng = np.random.default_rng(cfg.seed)
        self.time = 0.0
        self.done = False
        self.outcome: Optional[str] = None
        self.found_by: Optional[str] = None
        self.events: list[Event] = []
        self._step_events: list[Event] = []
        self._requests: dict[str, Command] = {}
        self._outgoing: list[tuple[str, Transmit]] = []
        self.radio = RadioChannel(cfg.radio, np.random.default_rng([cfg.seed, 1]))
        self.truth_belief = BeliefMap(sc.grid, sc.prior, sc.layer_fractions)
        self.target_cell = sc.grid.world_to_cell(*sc.target_xy)
        self._target_center = np.array(sc.grid.cell_center(*self.target_cell)) if cfg.target else np.full(2, np.inf)

        mm = sc.road_mismatch
        road = sc.true_road(blocked=False) if mm else None
        self.robots: dict[str, _SimRobot] = {}
        for rid, spec in sc.robots.items():
            if spec.robot_type == RobotType.UGV and mm:
                self.robots[rid] = _SimRobot(spec, road, sc.true_road())
            else:
                self.robots[rid] = _SimRobot(spec, sc.graphs[spec.robot_type])
        # Road-map mismatches the robots can see: what each has seen, or found by running into it.
        self._road, self._n_prior = road, sc.graphs[RobotType.UGV].n_nodes
        self._mismatch_ids = np.array(sorted(mm.blocked) + sorted(mm.new_nodes), dtype=int)
        self._mismatch_tree = cKDTree(road.positions[self._mismatch_ids]) if mm else None
        self._blocked = set(mm.blocked)
        self._mismatch_seen: dict[str, set[int]] = {rid: set() for rid in self.robots}
        self._bumped: dict[str, set[int]] = {rid: set() for rid in self.robots}
        self._map_view: dict[str, set[int]] = {rid: set() for rid in self.robots}  # mismatches in range last step
        self._map_obs = {rid: dict(zip(self._mismatch_ids.tolist(), road_observability(
            spec, sc.grid, sc.terrain, sc.occlusion, road.positions[self._mismatch_ids]).tolist())) if mm else {}
            for rid, spec in sc.robots.items()}  # per mismatch node: the chance a look sees it
        self._map_rng = np.random.default_rng([cfg.seed, 3])
        for rid, init in sc.init.items():
            r = self.robots[rid]
            if init.node is not None:
                r.place_at_node(init.node)
            elif init.position is not None:
                r.pos = np.asarray(init.position, dtype=float)
                r.last_node = r.graph.nearest_node(r.pos)
        for rid, init in sc.init.items():
            if init.docked_on is not None:
                uav, ugv = self.robots[rid], self.robots[init.docked_on]
                if ugv.dock_slot is not None:
                    raise ValueError(f"UGV {ugv.rid} already carries {ugv.dock_slot}")
                uav.mode, uav.host, ugv.dock_slot = RobotMode.DOCKED, ugv.rid, uav.rid
                uav.pos = ugv.pos.copy()
                uav.last_node = uav.graph.nearest_node(uav.pos)

        # Per robot, layer and cell: the chance a look detects the target (true terrain and occlusion).
        self._layer_vis = {rid: detection_map(spec, sc.terrain, sc.occlusion) for rid, spec in sc.robots.items()}
        # One look per pass: a cell gets a look when it enters a footprint, and the target one
        # roll per pass, whose outcome holds while it stays in view. Layers seen for certain are
        # looked at every step, as repeating a certain look changes nothing: a robot that missed
        # another's update over the radio still hears the cells it keeps in view.
        self._detect_rng = np.random.default_rng([cfg.seed, 2])
        self._view: dict[str, Optional[tuple[int, int, np.ndarray]]] = {rid: None for rid in sc.robots}  # looked, in view
        self._target_in_view = {rid: False for rid in sc.robots}
        self._target_seen = {rid: False for rid in sc.robots}

        self._terrain_masks = {t: sc.terrain == int(t) for t in Terrain if (sc.terrain == int(t)).any()}
        self._log: dict[str, list] = {"t": [], "searched_prob": []}
        for t in self._terrain_masks:
            self._log[f"coverage_{t.name.lower()}"] = []
        for rid in self.robots:
            self._log[f"energy_{rid}"] = []
        self._traj_t: list[float] = []
        self._traj: dict[str, list] = {rid: [] for rid in self.robots}
        self._modes: dict[str, list] = {rid: [] for rid in self.robots}
        self._snapshots: list[tuple[float, np.ndarray]] = []
        self._next_log = 0.0
        self._next_snapshot = 0.0
        self._stats = {
            rid: {"type": r.spec.robot_type.value, "sorties": 0, "dock_aborts": 0, "undock_aborts": 0, "rejected": 0,
                  "path_blocked": 0, "map_changes_found": 0}
            for rid, r in self.robots.items()
        }

        for r in self.robots.values():
            r.moved, r.nodes_reached = 0.0, ([r.at_node] if r.at_node is not None else [])
        updates = self._sense()
        map_updates = self._sense_map()
        self._record()
        return self._observation(updates, map_updates=map_updates)

    # ------------------------------------------------------------------- loop
    def run(
        self,
        controller: MissionController,
        callback: Optional[Callable[["Simulator", Observation], None]] = None,
    ) -> MissionResult:
        wall0 = _time.perf_counter()
        obs = self.reset()
        t0 = _time.perf_counter()
        controller.reset(self.setup)
        reset_s = _time.perf_counter() - t0
        step_s, min_step_s, max_step_s, n_steps = 0.0, float("inf"), 0.0, 0
        while not self.done:
            t0 = _time.perf_counter()
            commands = controller.step(obs)
            dt = _time.perf_counter() - t0
            step_s, n_steps = step_s + dt, n_steps + 1
            min_step_s, max_step_s = min(min_step_s, dt), max(max_step_s, dt)
            if self.config.stop_when_controller_done and controller.is_done():
                self._finish("controller_done")
                break
            obs = self.step(commands)
            if callback is not None:
                callback(self, obs)
        result = self.result(getattr(controller, "name", type(controller).__name__))
        result.wall_time_s = _time.perf_counter() - wall0
        result.controller_timing = {"controller_reset_s": reset_s, "controller_s": step_s,
                                    "controller_steps": n_steps, "controller_min_step_s": min_step_s,
                                    "controller_max_step_s": max_step_s}
        if hasattr(controller, "robot_timing"):
            for rid, t in controller.robot_timing().items():
                result.robot_stats[rid].update(t)
        return result

    def step(self, commands: dict[str, RobotCommands]) -> Observation:
        if self.done:
            raise RuntimeError("mission already finished")
        dt = self.config.dt
        sent_time = self.time
        self._step_events = []
        self._requests = {}
        self._outgoing = []
        for rid, cmds in commands.items():
            for cmd in (cmds if isinstance(cmds, (list, tuple)) else (cmds,)):
                self._apply_command(rid, cmd)

        for r in self.robots.values():
            r.started_this_step = False
            stuck_on = r.blocked_by
            if r.mode == RobotMode.ACTIVE:
                r.move(dt)
            else:
                r.moved, r.nodes_reached, r.blocked_by = 0.0, [], None
            if r.blocked_by is not None and r.blocked_by != stuck_on:
                self._stats[r.rid]["path_blocked"] += 1
                self._bumped[r.rid].add(r.blocked_by)
                self._emit(EventType.PATH_BLOCKED, r.rid, detail=f"node {r.blocked_by} is blocked")
        for r in self.robots.values():
            if r.host is not None and r.mode in (RobotMode.DOCKED, RobotMode.DOCKING, RobotMode.UNDOCKING):
                r.pos = self.robots[r.host].pos.copy()

        for rid, req in self._requests.items():
            if isinstance(req, Dock):
                self._start_dock(self.robots[rid], req.ugv_id)
            else:
                self._start_undock(self.robots[rid])
        self._advance_manoeuvres(dt)
        self._update_energy(dt)

        self.time += dt
        updates = self._sense()
        map_updates = self._sense_map()
        inbox = self._radio(sent_time, dt)
        self._record()
        if not self.done:
            if all(r.mode == RobotMode.DEPLETED for r in self.robots.values()):
                self._finish("all_depleted")
            elif self.config.stop_when_ugv_depleted and self.robots[self.ugv].mode == RobotMode.DEPLETED:
                self._finish("ugv_depleted")
            elif self.time >= self.config.max_time_s - _EPS:
                self._finish("timeout")
        return self._observation(updates, inbox, map_updates)

    # --------------------------------------------------------------- commands
    def _emit(self, etype: EventType, rid: str, other: Optional[str] = None, detail: str = "") -> None:
        ev = Event(self.time, etype, rid, other, detail)
        self._step_events.append(ev)
        self.events.append(ev)

    def _reject(self, rid: str, why: str) -> None:
        self._stats[rid]["rejected"] += 1
        self._emit(EventType.COMMAND_REJECTED, rid, detail=why)

    def _apply_command(self, rid: str, cmd: Command) -> None:
        r = self.robots.get(rid)
        if r is None:
            return
        if r.mode == RobotMode.DEPLETED:
            return
        if isinstance(cmd, Transmit):
            self._outgoing.append((rid, cmd))
            return
        if isinstance(cmd, (Dock, Undock)):
            if r.spec.robot_type != RobotType.UAV:
                return self._reject(rid, f"{type(cmd).__name__} is UAV-only")
            self._requests[rid] = cmd
            return
        if r.mode != RobotMode.ACTIVE:
            return self._reject(rid, f"motion command while {r.mode.value}")

        if isinstance(cmd, Hold):
            r.path, r.goal_point = [], None
        elif isinstance(cmd, GoToPoint):
            if not r.spec.free_motion:
                return self._reject(rid, "GoToPoint requires free motion")
            r.path, r.goal_point = [], np.asarray(cmd.xy, dtype=float)
        elif isinstance(cmd, FollowPath):
            nodes = [int(n) for n in cmd.nodes]
            err = self._validate_path(r, nodes)
            if err:
                return self._reject(rid, err)
            if nodes and r.at_node is not None and nodes[0] == r.at_node:
                nodes = nodes[1:]
            r.path, r.goal_point = nodes, None
        else:
            self._reject(rid, f"unknown command {cmd!r}")

    @staticmethod
    def _validate_path(r: _SimRobot, nodes: list[int]) -> str:
        g = r.graph
        if any(n < 0 or n >= g.n_nodes for n in nodes):
            return "node index out of range"
        if r.spec.free_motion or not nodes:
            return ""
        for a, b in zip(nodes, nodes[1:]):
            if not g.has_edge(a, b):
                return f"no edge {a}-{b}"
        first = nodes[0]
        if r.at_node is not None:
            if first != r.at_node and not g.has_edge(r.at_node, first):
                return f"path start {first} not adjacent to current node {r.at_node}"
        elif r.edge is not None and first not in r.edge:
            return f"path start {first} not on current edge {r.edge}"
        return ""

    # --------------------------------------------------------------- docking
    def _start_dock(self, uav: _SimRobot, ugv_id: str) -> None:
        ugv = self.robots.get(ugv_id)
        why = ""
        if uav.mode != RobotMode.ACTIVE:
            why = f"cannot dock while {uav.mode.value}"
        elif ugv is None or ugv.spec.robot_type != RobotType.UGV:
            why = f"{ugv_id} is not a UGV"
        elif ugv.dock_slot is not None:
            why = f"{ugv_id} dock occupied by {ugv.dock_slot}"
        elif ugv.moved > _EPS:
            why = f"{ugv_id} is moving"
        elif np.hypot(*(uav.pos - ugv.pos)) > self.scenario.docking.dock_tolerance_m:
            why = f"too far from {ugv_id} ({np.hypot(*(uav.pos - ugv.pos)):.1f} m)"
        if why:
            return self._reject(uav.rid, why)
        uav.mode, uav.host, uav.timer = RobotMode.DOCKING, ugv_id, self.scenario.docking.dock_duration_s
        uav.path, uav.goal_point, uav.started_this_step = [], None, True
        uav.pos = ugv.pos.copy()
        ugv.dock_slot = uav.rid
        self._emit(EventType.DOCK_STARTED, uav.rid, ugv_id)

    def _start_undock(self, uav: _SimRobot) -> None:
        if uav.mode != RobotMode.DOCKED:
            return self._reject(uav.rid, f"cannot undock while {uav.mode.value}")
        if uav.swap_left > _EPS:
            return self._reject(uav.rid, "battery swap in progress")
        ugv = self.robots[uav.host]
        if ugv.moved > _EPS:
            return self._reject(uav.rid, f"{ugv.rid} is moving")
        uav.mode, uav.timer, uav.started_this_step = RobotMode.UNDOCKING, self.scenario.docking.undock_duration_s, True
        self._emit(EventType.UNDOCK_STARTED, uav.rid, ugv.rid)

    def _advance_manoeuvres(self, dt: float) -> None:
        for uav in self.robots.values():
            if uav.mode not in (RobotMode.DOCKING, RobotMode.UNDOCKING) or uav.started_this_step:
                continue
            ugv = self.robots[uav.host]
            if ugv.moved > _EPS:
                if uav.mode == RobotMode.DOCKING:
                    uav.mode, uav.host, ugv.dock_slot = RobotMode.ACTIVE, None, None
                    self._stats[uav.rid]["dock_aborts"] += 1
                    self._emit(EventType.DOCK_ABORTED, uav.rid, ugv.rid, "host moved during landing")
                else:
                    uav.mode = RobotMode.DOCKED
                    self._stats[uav.rid]["undock_aborts"] += 1
                    self._emit(EventType.UNDOCK_ABORTED, uav.rid, ugv.rid, "host moved during take-off")
                continue
            uav.timer -= dt
            if uav.timer <= _EPS:
                if uav.mode == RobotMode.DOCKING:
                    uav.mode, uav.swap_left = RobotMode.DOCKED, self.scenario.docking.swap_duration_s
                    self._emit(EventType.DOCKED, uav.rid, ugv.rid)
                else:
                    uav.mode, uav.host, ugv.dock_slot = RobotMode.ACTIVE, None, None
                    uav.last_node = uav.graph.nearest_node(uav.pos)
                    self._stats[uav.rid]["sorties"] += 1
                    self._emit(EventType.UNDOCKED, uav.rid, ugv.rid)

    # ----------------------------------------------------------------- energy
    def _update_energy(self, dt: float) -> None:
        for r in self.robots.values():
            if r.mode == RobotMode.DEPLETED:
                continue
            e = r.spec.energy
            if r.mode == RobotMode.DOCKED:  # docked UAVs draw no power; a battery swap ends with a full one
                paused = not self.scenario.docking.swap_while_moving and self.robots[r.host].moved > _EPS
                if r.swap_left > _EPS and not paused:
                    r.swap_left -= dt
                    if r.swap_left <= _EPS:
                        r.swap_left, r.energy = 0.0, e.capacity
                continue
            used = e.idle_power * dt + e.energy_per_m * r.moved
            r.energy -= used
            r.energy_used += used
        for r in self.robots.values():
            if r.mode != RobotMode.DEPLETED and r.energy <= 0.0:
                self._deplete(r)

    def _deplete(self, r: _SimRobot) -> None:
        r.energy = 0.0
        if r.mode == RobotMode.DOCKED:
            return  # docked UAVs draw no power
        if r.mode in (RobotMode.DOCKING, RobotMode.UNDOCKING):
            self.robots[r.host].dock_slot = None
            r.host = None
        # A flat UGV keeps its dock_slot: the carried UAV can still take off from it.
        r.mode, r.path, r.goal_point = RobotMode.DEPLETED, [], None
        self._emit(EventType.DEPLETED, r.rid)

    # ---------------------------------------------------------------- sensing
    def _sense(self) -> list[LocalBeliefUpdate]:
        sc, grid = self.scenario, self.scenario.grid
        updates = []
        for r in self.robots.values():
            if r.mode != RobotMode.ACTIVE:  # depleted, or a UAV that is not airborne
                self._view[r.rid], self._target_in_view[r.rid] = None, False  # back in the air: a new pass
                continue
            rng_m = r.spec.sensor.range_m
            row0, col0, mask = grid.disk_window(r.pos[0], r.pos[1], rng_m)
            new = mask & ~self._in_view_last_step(r.rid, row0, col0, mask)
            h, w = mask.shape
            vis = self._layer_vis[r.rid][:, row0:row0 + h, col0:col0 + w]
            p = vis * np.where(vis >= 1, mask, new)  # a certain look may repeat: it cannot double-count
            in_range = np.hypot(*(self._target_center - r.pos)) <= rng_m
            entering = in_range and not self._target_in_view[r.rid]
            self._target_in_view[r.rid] = in_range
            detected = False
            if in_range:
                if r.spec.robot_type in self.config.terminate_on and self.config.confirm_test == "range":
                    detected = True
                    if not self.done:
                        self.found_by = r.rid
                        self._emit(EventType.TARGET_FOUND, r.rid)
                        self._finish("found")
                else:
                    if entering:
                        pd = self._layer_vis[r.rid][(int(sc.target_layer),) + self.target_cell]
                        self._target_seen[r.rid] = pd >= 1 or (pd > 0 and self._detect_rng.random() < pd)
                    if self._target_seen[r.rid]:
                        detected = True
                        self._emit(EventType.TARGET_SIGHTED, r.rid)
                        ends = r.rid in self.config.confirm_by or r.spec.robot_type in self.config.terminate_on
                        if ends and not self.done:
                            self.found_by = r.rid
                            self._emit(EventType.TARGET_FOUND, r.rid)
                            self._finish("found")
            # A sighting's update carries no looks (BeliefMap.apply skips it): the cells that came
            # into view get theirs on the next step without one.
            self._view[r.rid] = (row0, col0, mask & ~new if detected else mask)
            upd = LocalBeliefUpdate(
                robot_id=r.rid,
                robot_type=r.spec.robot_type,
                time=self.time,
                row0=row0,
                col0=col0,
                p_detect=p,
                detected=detected,
                detection_xy=tuple(self._target_center) if detected else None,
            )
            self.truth_belief.apply(upd)
            updates.append(upd)
        return updates

    def _in_view_last_step(self, rid: str, row0: int, col0: int, mask: np.ndarray) -> np.ndarray:
        """Which cells of this footprint window were in robot ``rid``'s footprint last step and had
        their look."""
        out = np.zeros_like(mask)
        if self._view[rid] is None:
            return out
        r0, c0, m0 = self._view[rid]
        ra, rb = max(row0, r0), min(row0 + mask.shape[0], r0 + m0.shape[0])
        ca, cb = max(col0, c0), min(col0 + mask.shape[1], c0 + m0.shape[1])
        if ra < rb and ca < cb:
            out[ra - row0:rb - row0, ca - col0:cb - col0] = m0[ra - r0:rb - r0, ca - c0:cb - c0]
        return out

    def _sense_map(self) -> list[TraversabilityUpdate]:
        """Road-map mismatches each active robot sees (or ran into) for the first time: blocked nodes,
        and nodes missing from the map with their edges. A node coming into range is seen with the
        chance occlusion leaves it (``road_observability``), once per pass."""
        if self._mismatch_tree is None:
            return []
        out = []
        for r in self.robots.values():
            if r.mode != RobotMode.ACTIVE:
                self._map_view[r.rid] = set()
                continue
            near = set(self._mismatch_ids[self._mismatch_tree.query_ball_point(r.pos, r.spec.sensor.range_m)].tolist())
            obs = self._map_obs[r.rid]
            looked = {n for n in sorted(near - self._map_view[r.rid] - self._mismatch_seen[r.rid])
                      if obs[n] >= 1 or self._map_rng.random() < obs[n]}
            self._map_view[r.rid] = near
            new = (looked | self._bumped[r.rid]) - self._mismatch_seen[r.rid]
            self._bumped[r.rid].clear()
            if not new:
                continue
            self._mismatch_seen[r.rid] |= new
            self._stats[r.rid]["map_changes_found"] += len(new)
            added = sorted(new - self._blocked)
            edges = sorted({(min(n, m), max(n, m)) for n in added for m, _ in self._road.neighbors(n)})
            nodes = {m: tuple(map(float, self._road.positions[m])) for e in edges for m in e if m >= self._n_prior}
            out.append(TraversabilityUpdate(r.rid, self.time, RobotType.UGV, tuple(sorted(new & self._blocked)),
                                            nodes, tuple(edges)))
        return out

    # ------------------------------------------------------------ bookkeeping
    def _finish(self, outcome: str) -> None:
        if not self.done:
            self.done, self.outcome = True, outcome

    def _record(self) -> None:
        self._traj_t.append(self.time)
        for rid, r in self.robots.items():
            self._traj[rid].append(r.pos.copy())
            self._modes[rid].append(MissionResult.MODE_ORDER.index(r.mode))
        if self.time + _EPS >= self._next_log or self.done:
            self._next_log = self.time + self.config.log_interval_s
            self._log["t"].append(self.time)
            self._log["searched_prob"].append(self.truth_belief.searched_probability())
            covered = self.truth_belief.covered_cells()
            for t, m in self._terrain_masks.items():
                self._log[f"coverage_{t.name.lower()}"].append(float(covered[m].mean()))
            for rid, r in self.robots.items():
                self._log[f"energy_{rid}"].append(r.energy / r.spec.energy.capacity)
        si = self.config.belief_snapshot_interval_s
        if si is not None and self.time + _EPS >= self._next_snapshot:
            self._next_snapshot = self.time + si
            self._snapshots.append((self.time, self.truth_belief.cell_mass().astype(np.float32)))

    def _radio(self, sent_time: float, dt: float) -> dict[str, list[RadioMessage]]:
        alive = {rid: r.mode != RobotMode.DEPLETED for rid, r in self.robots.items()}
        positions = {rid: r.pos for rid, r in self.robots.items()}
        types = {rid: r.spec.robot_type for rid, r in self.robots.items()}
        for rid, why in self.radio.transmit(self._outgoing, sent_time, dt, positions, types, alive):
            self._reject(rid, why)
        return self.radio.collect(self.time, alive)

    def _observation(self, updates: list[LocalBeliefUpdate], inbox: Optional[dict[str, list[RadioMessage]]] = None,
                     map_updates: Optional[list[TraversabilityUpdate]] = None) -> Observation:
        states = {}
        noise = self.config.energy_noise_std
        for rid, r in self.robots.items():
            cap = r.spec.energy.capacity
            est = r.energy
            if noise > 0:
                est = float(np.clip(est + self.rng.normal(0, noise * cap), 0, cap))
            is_ugv = r.spec.robot_type == RobotType.UGV
            states[rid] = RobotState(
                robot_id=rid,
                robot_type=r.spec.robot_type,
                position=(float(r.pos[0]), float(r.pos[1])),
                mode=r.mode,
                energy=est,
                energy_capacity=cap,
                moving=r.moved > _EPS,
                node=r.at_node,
                last_node=r.last_node,
                nodes_reached=tuple(r.nodes_reached),
                path_remaining=tuple(r.path),
                goal_point=None if r.goal_point is None else (float(r.goal_point[0]), float(r.goal_point[1])),
                docked_uav=r.dock_slot if is_ugv else None,
                host_ugv=None if is_ugv else r.host,
                mode_time_remaining=max(r.timer, 0.0) if r.mode in (RobotMode.DOCKING, RobotMode.UNDOCKING)
                else max(r.swap_left, 0.0) if r.mode == RobotMode.DOCKED else 0.0,
            )
        return Observation(self.time, states, updates, list(self._step_events), mission_complete=self.done,
                           radio=inbox or {}, map_updates=map_updates or [])

    def result(self, controller_name: str = "") -> MissionResult:
        stats = {}
        for rid, r in self.robots.items():
            s = dict(self._stats[rid])
            s.update(self.radio.stats.get(rid, {**dict.fromkeys(_RADIO_STAT_KEYS, 0),
                                                **dict.fromkeys(_RADIO_SIZE_KEYS, float("nan"))}))
            s.update(distance_m=r.odometer, energy_used=r.energy_used, depleted=r.mode == RobotMode.DEPLETED,
                     final_energy_fraction=r.energy / r.spec.energy.capacity)
            stats[rid] = s
        return MissionResult(
            scenario_name=self.scenario.name,
            controller_name=controller_name,
            outcome=self.outcome or "running",
            time_s=self.time,
            found_by=self.found_by,
            log={k: np.asarray(v) for k, v in self._log.items()},
            traj_t=np.asarray(self._traj_t),
            trajectories={rid: np.asarray(v) for rid, v in self._traj.items()},
            modes={rid: np.asarray(v, dtype=np.int8) for rid, v in self._modes.items()},
            events=list(self.events),
            robot_stats=stats,
            final_mass=self.truth_belief.cell_mass(),
            belief_snapshots=self._snapshots,
        )
