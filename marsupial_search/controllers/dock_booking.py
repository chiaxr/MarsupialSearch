"""Dock booking: how the UAVs share the one dock on a UGV that drives a timed route.

The UGV (:mod:`.ugv_timed_route`) commits to a road route with an arrival time at each
node, and broadcasts it. Positions along it are *route indices*. A UAV books a landing at
a route index; the UGV adds a landing stop there and, once that UAV's battery is swapped,
a take-off stop further along. The landings take turns, so each sortie searches the stretch of route
between its take-off and its landing. The landing point and time are fixed before the UAV
takes off, so a sortie needs no messages in flight.

Messages
    ``"status"`` (every robot, every step, broadcast): position, mode, energy, its own new
        sensor updates, road nodes reached and the sighting it knows of.
        The UGV adds ``"holding"`` (the stop it is stationary for: UAV and kind) and, when it
        changes and every ``route_resend_s``, ``"plan"``: the route from the current index,
        its arrival times, the *pin* (last route index with a promised stop: the route is
        fixed up to here), the queue of accepted bookings with their tours, the take-off
        stop of each docked UAV, the UAVs it has written off and the road-map changes it knows
        (``"map"``: blocked nodes, and nodes and edges missing from the prior map).
        A UAV adds ``"map"``: the road-map changes it knows and the UGV's last plan did not,
        so it reports what it finds until the UGV has it; and in flight ``"free"``: where and
        when its tour ends and the energy it can spend after that, so a UGV running late can
        meet it elsewhere (``ugv_timed_route``).
    ``"book"`` (UAV → UGV): a :class:`Booking`, re-sent until it shows up in the queue.
        The UGV accepts a booking planned on its current route ``version`` that lands after
        the pin; accepting bumps the version, so bookings planned on the old route are
        refused and those UAVs plan again. Bookings only append: an accepted one never moves.
    ``"sync"`` (UAV → UGV after landing, and back): the UAV's belief map; the UGV fuses it
        (element-wise minimum) and returns the fused map, so the next tour is planned on
        what every UAV has reported.

:class:`BookingUav` is the UAV side of the protocol: booking, take-off, flight, landing and
sync. A UAV policy subclasses it and implements :meth:`BookingUav.plan_sortie`, which picks
the landing and the tour flown on the way to it.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from marsupial_search.core import (
    BeliefMap,
    Command,
    Dock,
    GoToPoint,
    GraphKnowledge,
    Hold,
    MissionSetup,
    RobotController,
    RobotMode,
    RobotObservation,
    RobotState,
    RobotType,
    Transmit,
    Undock,
)

from .common import dist
from .sortie_planner import SortieConfig

STATUS, BOOK, SYNC = "status", "book", "sync"
LAND, TAKEOFF = "land", "takeoff"


@dataclass
class DockBookingConfig(SortieConfig):
    """Settings the UGV and UAV policies of a team should agree on: ``slot_s``,
    ``stop_margin_s`` and ``uav_reserve_fraction`` set the landing spacing
    (:func:`slot_spacing`) on both sides."""
    slot_s: float = 0.0  # spacing of booked landings (0: automatic, see slot_spacing)
    stop_margin_s: float = 3.0  # planned on top of each landing and take-off


@dataclass
class BookingUavConfig(DockBookingConfig):
    partner_stale_s: float = 10.0  # the UGV's status or route older than this: do not plan or land on it
    sync_resend_s: float = 5.0  # docked UAV re-sends its belief until the UGV returns the fused map
    plan_lead_s: float = 30.0  # a docked UAV books its next sortie this long before its take-off
    abort_reserve_fraction: float = 0.05  # in flight: skip the rest of the tour below this margin
    dock_retry_s: float = 3.0  # wait after a rejected or aborted landing


@dataclass(frozen=True)
class Booking:
    """A UAV's request to land on the UGV, planned on route ``version``."""
    uav: str
    plan_id: int
    version: int
    land_idx: int  # route index of the landing node
    t_land: float  # the UAV is at the landing node by then
    e_land: float  # energy it expects to land with
    slack_s: float  # hover it can afford beyond t_land
    tour: tuple[int, ...]  # UAV-grid nodes (other UAVs skip them)


@dataclass(frozen=True)
class Sortie:
    """A UAV policy's choice for its next sortie (:meth:`BookingUav.plan_sortie`)."""
    land_idx: int  # route index of the landing node
    t_land: float  # when the UAV is at the landing node: the UGV's arrival time there
    e_land: float  # energy it expects to land with
    slack_s: float  # hover it can afford beyond t_land
    tour: tuple[int, ...]  # UAV-grid nodes to fly, in order, before the landing node


def slot_spacing(setup: MissionSetup, cfg: DockBookingConfig, n_uav: int) -> float:
    """Landing spacing: a full sortie plus a stay on the dock (landing, battery swap, take-off),
    shared by ``n_uav`` UAVs, and never shorter than one stay. ``cfg.slot_s`` overrides it."""
    if cfg.slot_s > 0:
        return cfg.slot_s
    uav = setup.robots[setup.robots_of(RobotType.UAV)[0]]
    e, dock = uav.energy, setup.docking
    sortie = (1.0 - cfg.uav_reserve_fraction) * e.capacity / (e.idle_power + e.energy_per_m * uav.speed)
    stay = dock.dock_duration_s + dock.swap_duration_s + dock.undock_duration_s + 2 * cfg.stop_margin_s
    return max(stay, (sortie + stay) / max(n_uav, 1))


# ------------------------------------------------------------------- both sides


class BookingRobot(RobotController):
    """Belief and ``"status"`` gossip shared by the UGV and UAV sides."""

    config_cls: type = DockBookingConfig

    def __init__(self, config: Optional[DockBookingConfig] = None):
        self.cfg = config or self.config_cls()

    def reset(self, robot_id: str, setup: MissionSetup) -> None:
        self.rid, self.setup = robot_id, setup
        self.spec = setup.robots[robot_id]
        self.type = self.spec.robot_type
        self.graph = setup.graphs[self.type]
        self.belief = BeliefMap(setup.grid, setup.prior, setup.layer_fractions)
        self.roads = GraphKnowledge(setup.graphs[RobotType.UGV])  # the road map and the changes found to it
        self.road = self.roads.graph  # the current road map
        self.road_visited = np.zeros(self.road.n_nodes, dtype=bool)
        self.peers: dict[str, dict[str, Any]] = {}
        self.started = False
        self._new_visited: list[int] = []
        self._new_updates: list = []
        self._done = False

    def is_done(self) -> bool:
        return self._done

    def step(self, obs: RobotObservation) -> list[Command]:
        s, now = obs.state, obs.time
        if not self.started:
            self.started = True
            self._start(s)
        for upd in obs.belief_updates:
            self.belief.apply(upd)
            self._new_updates.append(upd)
        for upd in obs.map_updates:
            self.roads.add(upd.blocked, upd.new_nodes, upd.new_edges)
        if self.type == RobotType.UGV:
            reached = list(s.nodes_reached) + ([s.node] if s.node is not None else [])
            self._mark_visited(reached)
            self._new_visited += reached
        for msg in obs.messages(STATUS):
            self._ingest(msg.sender, msg.payload, msg.sent_time)
        cmds: list[Command] = []
        self._step(s, now, obs, cmds)
        status = {"type": self.type.value, "pos": s.position, "mode": s.mode.value, "energy": s.energy,
                  "energy_fraction": s.energy_fraction, "visited": self._new_visited, "updates": self._new_updates,
                  "sighting": self.belief.detection_xy, "done": self._done, **self._status_extra(s, now)}
        report = self._map_report()
        if report is not None:
            status["map"] = report
        self._new_visited, self._new_updates = [], []
        cmds.append(Transmit(STATUS, status))
        return cmds

    def _ingest(self, sender: str, p: dict, t: float) -> None:
        self.peers[sender] = {**p, "t": t}
        for upd in p["updates"]:
            self.belief.apply(upd)
        if "map" in p:
            self.roads.add(**p["map"])
        if p["type"] == RobotType.UGV.value:
            self._mark_visited(p["visited"])
        if p["sighting"] is not None and self.belief.detection_xy is None:
            self.belief.detection_xy = tuple(p["sighting"])

    def _fuse(self, payload: dict) -> None:
        """Fuse a ``"sync"`` belief map and sighting into this robot's belief."""
        np.minimum(self.belief.mass, payload["mass"], out=self.belief.mass)
        if payload["sighting"] is not None and self.belief.detection_xy is None:
            self.belief.detection_xy = tuple(payload["sighting"])

    def _mark_visited(self, nodes: list[int]) -> None:
        if nodes and max(nodes) >= len(self.road_visited):
            self.road_visited = np.pad(self.road_visited, (0, max(nodes) + 1 - len(self.road_visited)))
        self.road_visited[nodes] = True

    def _check_map(self, s: RobotState, now: float) -> None:
        """Once the road map has changed: plan on the new one (``_map_changed``)."""
        if self.road is self.roads.graph:
            return
        self.road = self.roads.graph
        if self.road.n_nodes > len(self.road_visited):
            self.road_visited = np.pad(self.road_visited, (0, self.road.n_nodes - len(self.road_visited)))
        self._map_changed(s, now)

    def _map_changed(self, s: RobotState, now: float) -> None: ...

    def _map_report(self) -> Optional[dict]:
        """Road-map changes to put in the status (None: nothing to report)."""
        return None

    def _start(self, s: RobotState) -> None: ...

    def _step(self, s: RobotState, now: float, obs: RobotObservation, cmds: list) -> None: ...

    def _status_extra(self, s: RobotState, now: float) -> dict:
        return {}


# --------------------------------------------------------------------- UAV side


class BookingUav(BookingRobot):
    """The UAV side of dock booking. Books a landing on the UGV's route, takes off when the UGV
    holds for it, flies the booked tour and lands in turn. Subclasses choose the sortie."""

    config_cls = BookingUavConfig

    def reset(self, robot_id: str, setup: MissionSetup) -> None:
        super().reset(robot_id, setup)
        ugvs = setup.robots_of(RobotType.UGV)
        if len(ugvs) != 1:
            raise ValueError(f"{self.name} supports exactly one UGV")
        self.ugv = ugvs[0]
        self.n_uav = len(setup.robots_of(RobotType.UAV))
        self.phase = "airborne"  # docked | airborne (no booking) | sortie | landing | lost
        self.plan: Optional[dict] = None  # the UGV's latest route broadcast (+ "t")
        self.pending: Optional[Booking] = None
        self.booking: Optional[dict] = None  # accepted: the UGV's queue entry
        self.waypoints: list[tuple[float, float]] = []
        self.tour: list[int] = []
        self.plan_count = 0
        self.docked_at = -np.inf
        self.synced = True
        self.last_sync = -np.inf
        self.next_dock_try = 0.0
        self.dock_tries = 0
        self.t0 = 0.0
        self.ugv_roads = GraphKnowledge(setup.graphs[RobotType.UGV])  # the road-map changes the UGV's plan had

    @abstractmethod
    def plan_sortie(self, start, t_start: float, energy: float, now: float) -> Optional[Sortie]:
        """The next sortie on the UGV's route ``self.plan``, taking off from ``start`` at
        ``t_start`` with ``energy`` (or now from where the UAV hovers). None: nothing to book yet.

        ``self.plan`` holds (route indices are absolute; the lists start at index ``cur``):
        ``nodes`` and ``arrive`` (road nodes and the UGV's arrival at each), ``pin`` (land
        after it), ``last_land`` (latest booked landing time, or None), ``queue`` (accepted
        bookings: ``uav``, ``t_land``, ``tour``, ...) and ``lost`` (UAVs written off)."""

    def _start(self, s: RobotState) -> None:
        self.phase = "docked" if s.mode == RobotMode.DOCKED else "airborne"

    def _ingest(self, sender: str, p: dict, t: float) -> None:
        super()._ingest(sender, p, t)
        if sender == self.ugv and "plan" in p:
            self.plan = {**p["plan"], "t": t}
            if "map" in p["plan"]:
                self.roads.add(**p["plan"]["map"])
                self.ugv_roads.add(**p["plan"]["map"])

    def _map_report(self) -> Optional[dict]:
        news = self.roads.changes(exclude=self.ugv_roads)
        return news if any(news.values()) else None

    def _status_extra(self, s: RobotState, now: float) -> dict:
        out = {"home": self.ugv, "phase": self.phase}
        if self.phase == "sortie" and self.waypoints:  # when and where its tour ends, and the energy left to meet the UGV
            e, v = self.spec.energy, self.spec.speed
            pts = [s.position] + self.waypoints
            length = sum(dist(a, b) for a, b in zip(pts, pts[1:]))
            spare = s.energy - e.travel_cost(length, v) - e.hover_cost(self.setup.docking.dock_duration_s) \
                - self.cfg.uav_reserve_fraction * e.capacity
            out["free"] = (*self.waypoints[-1], now + length / v, spare)
        return out

    # ---------------------------------------------------------------- step
    def _step(self, s: RobotState, now: float, obs: RobotObservation, cmds: list) -> None:
        if s.mode == RobotMode.DEPLETED:
            self.phase = "lost"
        if self.phase == "lost":
            return
        self._check_map(s, now)
        gs = self.peers.get(self.ugv)
        self._done = gs is not None and gs["done"]
        self._check_booking(now)

        if self.phase == "landing":
            if s.mode == RobotMode.DOCKED:
                self.phase, self.docked_at, self.synced = "docked", now, False
                self.dock_tries, self.booking = 0, None  # spent: its take-off is in the UGV's "takeoffs"
            elif s.mode == RobotMode.ACTIVE:  # rejected or aborted: the UGV may be late
                self.dock_tries += 1
                self.phase = "sortie"
                self.next_dock_try = now + min(30.0, self.cfg.dock_retry_s * 2 ** (self.dock_tries - 1))
        if self.phase == "docked":
            if s.mode == RobotMode.ACTIVE:  # took off
                self._launched(now)
            else:
                self._step_docked(s, now, obs, gs, cmds)
                return
        if self.phase == "airborne":
            self._step_airborne(s, now, cmds)
        if self.phase == "sortie":
            self._step_sortie(s, now, gs, cmds)

    def _check_booking(self, now: float) -> None:
        """Pending booking: accepted once it is in the UGV's queue; dropped if the route moved on."""
        if self.plan is None:
            return
        mine = next((b for b in self.plan["queue"] if b["uav"] == self.rid), None)
        if self.pending is not None:
            if mine is not None and mine["plan_id"] == self.pending.plan_id:
                self.booking, self.pending = mine, None
            elif self.plan["version"] != self.pending.version:
                self.pending = None
        elif self.booking is not None and mine is not None and mine["plan_id"] == self.booking["plan_id"]:
            if self.phase == "sortie" and mine["land_node"] != self.booking["land_node"]:  # moved off a blocked node
                self.waypoints[-1] = tuple(self.road.positions[mine["land_node"]])
            self.booking = mine  # the UGV may have re-timed it (a detour)
        elif self.booking is not None and mine is None and self.phase == "sortie":
            self.booking, self.phase = None, "airborne"  # the UGV gave up on it: book again

    def _launched(self, now: float) -> None:
        self.t0 = now
        if self.booking is not None and self.booking["t_land"] > now:
            self._fly(self.booking)
        else:
            self.booking, self.phase = None, "airborne"

    def _fly(self, booking: dict) -> None:
        self.phase = "sortie"
        self.tour = list(booking["tour"])
        air = self.setup.graphs[RobotType.UAV]
        k = tuple(self.road.positions[booking["land_node"]])
        self.waypoints = [tuple(air.positions[n]) for n in self.tour] + [k]

    def _book(self, start, t_start: float, energy: float, now: float) -> None:
        """Plan the next sortie and make it the pending booking."""
        sortie = self.plan_sortie(start, t_start, energy, now)
        if sortie is None:
            return
        self.plan_count += 1
        self.pending = Booking(self.rid, self.plan_count, self.plan["version"], sortie.land_idx, sortie.t_land,
                               sortie.e_land, sortie.slack_s, sortie.tour)

    def _slot(self, p: dict) -> float:
        n = max(self.n_uav - len(p["lost"]), 1)
        return slot_spacing(self.setup, self.cfg, n)

    # --------------------------------------------------------------- docked
    def _step_docked(self, s: RobotState, now: float, obs: RobotObservation, gs: Optional[dict],
                     cmds: list) -> None:
        for msg in obs.messages(SYNC):
            if msg.sender == self.ugv and msg.payload["ack"] >= self.docked_at:
                self._fuse(msg.payload)
                self.synced = True
        if not self.synced and now - self.last_sync >= self.cfg.sync_resend_s:
            cmds.append(Transmit(SYNC, {"mass": self.belief.mass.astype(np.float32),
                                        "sighting": self.belief.detection_xy}, to=(self.ugv,)))
            self.last_sync = now
        p = self.plan
        if p is None or now - p["t"] > self.cfg.partner_stale_s:
            return
        off = p["takeoffs"].get(self.rid)
        if off is not None and self.pending is None and self.booking is None and self.synced:
            idx, ready = off
            local = idx - p["cur"]
            if 0 <= local < len(p["arrive"]):
                t_off = max(p["arrive"][local], ready, now) + self.setup.docking.undock_duration_s
                if t_off - now <= self.cfg.plan_lead_s + self.setup.docking.undock_duration_s:
                    swapped = t_off - self.setup.docking.undock_duration_s >= now + s.mode_time_remaining
                    energy = self.spec.energy.capacity if swapped else s.energy
                    pos = tuple(self.road.positions[p["nodes"][local]])
                    self._book(pos, t_off, energy, now)
        # The UGV holds at the take-off stop until this UAV has left, so a status a few seconds old is enough.
        h = gs.get("holding") if gs is not None and now - gs["t"] <= self.cfg.partner_stale_s else None
        if h is not None and h["uav"] == self.rid and h["kind"] == TAKEOFF and s.mode == RobotMode.DOCKED \
                and s.mode_time_remaining <= 0:  # swap done
            cmds.append(Undock())
        if self.pending is not None:
            cmds.append(Transmit(BOOK, self.pending, to=(self.ugv,)))

    # -------------------------------------------------------------- airborne
    def _step_airborne(self, s: RobotState, now: float, cmds: list) -> None:
        """No booking yet: hover and book from here (out of touch with the UGV: fly to where it was last heard)."""
        if self.booking is not None:
            self._fly(self.booking)
            return
        p = self.plan
        fresh = p is not None and now - p["t"] <= self.cfg.partner_stale_s
        gs = self.peers.get(self.ugv)
        if not fresh and gs is not None and dist(s.position, gs["pos"]) > 0.5 * self.setup.docking.dock_tolerance_m:
            if s.goal_point is None or dist(s.goal_point, gs["pos"]) > 0.5:  # out of touch: towards the UGV
                cmds.append(GoToPoint(tuple(gs["pos"])))
        elif s.goal_point is not None or s.moving:
            cmds.append(Hold())
        if self.pending is None and fresh:
            self._book(s.position, now, s.energy, now)
        if self.pending is not None:
            cmds.append(Transmit(BOOK, self.pending, to=(self.ugv,)))

    # --------------------------------------------------------------- flight
    def _step_sortie(self, s: RobotState, now: float, gs: Optional[dict], cmds: list) -> None:
        b = self.booking
        if not self._margin_ok(s, now):
            self.waypoints, self.tour = self.waypoints[-1:], []
        while len(self.waypoints) > 1 and dist(s.position, self.waypoints[0]) < 0.5:
            self.waypoints.pop(0)
            self.tour.pop(0)
        fresh = gs is not None and now - gs["t"] <= self.cfg.partner_stale_s
        h = gs.get("holding") if fresh else None
        waiting = h is not None and h["uav"] == self.rid and h["kind"] == LAND  # the UGV holds for this UAV
        # Tour done and the UGV waits for it: straight to the UGV, wherever it is.
        target = tuple(gs["pos"]) if waiting and len(self.waypoints) == 1 else self.waypoints[0]
        if dist(s.position, target) >= 0.5:
            if s.goal_point is None or dist(s.goal_point, target) > 0.5:
                cmds.append(GoToPoint(target))
            return
        # At the landing node: land once the UGV holds there for this UAV (or, with no radio, is due).
        # With a radio, a UAV out of touch waits: the UGV meets it within radio range (ugv_timed_route).
        here = waiting and dist(gs["pos"], s.position) <= 0.5 * self.setup.docking.dock_tolerance_m
        due = self.setup.radio_range_m is None and now >= b["t_land"] + 2 * self.setup.dt
        if (here or due) and now >= self.next_dock_try:
            cmds.append(Dock(self.ugv))
            self.phase = "landing"

    def _margin_ok(self, s: RobotState, now: float) -> bool:
        """Can the UAV still fly the rest of its tour, wait for the UGV and land?"""
        e = self.spec.energy
        pts = [s.position] + self.waypoints
        length = sum(dist(a, b) for a, b in zip(pts, pts[1:]))
        arrive = now + length / self.spec.speed
        hover = max(0.0, self.booking["t_land"] - arrive) + self.setup.docking.dock_duration_s
        need = e.travel_cost(length, self.spec.speed) + e.hover_cost(hover)
        return s.energy - need >= self.cfg.abort_reserve_fraction * e.capacity
