"""UAV policy ``uav_near_scout``: a standby UAV that stays on the dock until the UGV calls it.

Speaks :mod:`.dock_booking` as a *standby* UAV (it says so in its status). It is meant to fly
low, next to the UGV, helping it search road nodes nearby that a blockage has suddenly put far
away by road: give it a smaller sensor range than ``uav_far_scout``
(``FleetConfig.uav_sensor_ranges``). The UGV (``ugv_timed_route``) calls it in two cases:

Making room
    The other UAV is about to land and the dock takes one UAV: it takes off just before that
    landing, searches within ``nearby_m`` of the UGV's route, and lands back once the other
    UAV has left the dock.
A blockage
    The UGV's route runs into a blocked node that cuts off nearby road nodes with belief left
    round them (``ugv_timed_route``: ``call_reach_m``, ``call_gain_m``, ``call_min_belief``).
    It takes off where the UGV is and searches UAV-grid nodes within ``scout_radius_m`` of
    those road nodes, choosing them by the belief left (so not what any robot has searched:
    it syncs its belief with the UGV's before planning). It lands as soon as its tour covers
    them (``cover_fraction`` of what the longest tour, ``scout_max_s``, would sweep), after
    ``scout_min_s`` at least. With ``trace_roads``, it also follows any road it finds that is
    missing from the map (within ``trace_radius_m`` of the blocked node), as long as it still
    lands on time; it reports it like any UAV, and the UGV uses it from its next re-plan.

A messenger
    The other UAV is out of radio range and the UGV can no longer meet it in time where it
    waits (``ugv_timed_route``: ``messenger``). It flies to that point and waits there until it
    hears that UAV (relaying then gives it the UGV's plan, and brings its status back), or until
    it must leave to land on time; its landing leaves it ``deliver_wait_s`` for that.

Each time it lands at a route index where the dock is free (no other UAV on it), early
enough that its battery is swapped before the next UAV lands, and stays there.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from marsupial_search.core import Hold, MissionSetup, RobotState

from .common import dist
from .dock_booking import BookingUavConfig, Sortie
from .uav_greedy_tour import GreedyTourUav


@dataclass
class NearScoutUavConfig(BookingUavConfig):
    scout_radius_m: float = 75.0  # called to a blockage: search UAV-grid nodes this close to the road nodes it cuts off
    scout_min_s: float = 60.0  # called to a blockage: fly at least this long ...
    scout_max_s: float = 600.0  # ... and land by when it has covered the cut-off roads (within this), if the dock allows
    cover_fraction: float = 0.9  # covered: its tour sweeps this share of the belief the longest one would
    trace_roads: bool = True  # called to a blockage: follow the roads it finds missing from the map ...
    trace_radius_m: float = 200.0  # ... within this radius of the blocked node
    nearby_m: float = 300.0  # making room: search UAV-grid nodes this close to the UGV's route until it lands back
    deliver_wait_s: float = 180.0  # a messenger: the time its landing leaves it to wait for the other UAV


class NearScoutUav(GreedyTourUav):
    name = "uav_near_scout"
    config_cls = NearScoutUavConfig
    standby = True

    def reset(self, robot_id: str, setup: MissionSetup) -> None:
        super().reset(robot_id, setup)
        # The UGV's call while docked: {"for": UAV it makes room for, "xy": blocked node, "near": cut-off road nodes}.
        self.call: Optional[dict] = None
        self.task: Optional[dict] = None  # in the air: the call it answers (a blockage, or a messenger's)
        self.delivered = False  # a messenger: it has heard the UAV it was sent to
        self.seen: set[int] = set()  # road nodes missing from the map that it has been in sensor range of

    # ---------------------------------------------------------------- calls
    def _takeoff(self, p: dict) -> Optional[tuple[int, float]]:
        call = p["calls"].get(self.rid)  # its other take-offs are not calls: it stays
        if call is not None and call != self.call and self.pending is None and self.booking is None:
            self.synced, self.last_sync = False, -np.inf  # a new call: plan on all the UGV knows was searched
        self.call = call
        return p["takeoffs"][self.rid] if call is not None else None

    def _may_undock(self) -> bool:
        # Another UAV must land: leave in any case (and book from the air). A blocked node: once booked.
        return self.call is not None and (self.call["for"] is not None or self.booking is not None)

    def _launched(self, now: float) -> None:
        call = self.call
        self.task = call if call is not None and (call["near"] or call.get("deliver")) else None
        self.call, self.delivered = None, False
        super()._launched(now)

    def _check_booking(self, now: float) -> None:
        super()._check_booking(now)
        if self.phase == "docked" and self.booking is not None and self.plan is not None \
                and not any(b["uav"] == self.rid for b in self.plan["queue"]):
            self.booking = None  # dropped (the call was withdrawn): wait for the next one

    # -------------------------------------------------------------- sortie
    def plan_sortie(self, start, t_start: float, energy: float, now: float) -> Optional[Sortie]:
        p, P, cfg, d = self.plan, self.planner, self.cfg, self.setup.docking
        nodes, arr, base = np.asarray(p["nodes"], int), np.asarray(p["arrive"], float), p["cur"]
        ok = arr > t_start
        ok[0] = False
        for a, b in p["busy"]:  # another UAV (or this one, before it leaves) is on the dock
            ok[max(a - base, 0):len(nodes) if b is None else b - base + 1] = False
        # Its battery must be swapped before the next landing of another UAV: it takes off for it.
        stay = d.dock_duration_s + cfg.stop_margin_s + d.swap_duration_s
        lands = sorted((b["land_idx"] - base, max(arr[b["land_idx"] - base], b["t_land"]))
                       for b in p["queue"] if b["uav"] != self.rid and b["land_idx"] >= base)
        if lands:
            k = np.searchsorted([i for i, _ in lands], np.arange(len(nodes)), side="right")
            nxt = np.array([lands[j][1] if j < len(lands) else np.inf for j in k])  # when the next one lands
            ok &= arr + stay <= nxt
        task = self.call if self.phase == "docked" else self.task
        idx = np.flatnonzero(ok)
        if not len(idx):
            return None
        reach = self._reach(start, t_start, P.energy_budget(energy))
        if task is not None and task.get("deliver"):  # a messenger: to where the other UAV waits
            return self._deliver(idx, reach, start, t_start, energy, now, task)
        if task is not None and task["near"]:  # a blockage: the cut-off roads
            return self._scout(idx, reach, start, t_start, energy, now, task)
        pick = next((int(i) for i in idx if reach(i) >= 0), int(idx[0]))  # the first it can reach
        # Making room: near the UGV's route until it lands back.
        off = p["takeoffs"].get(self.rid)
        lo = max(off[0] - base, 0) if off is not None and self.phase == "docked" else 0
        return self._sortie(pick, start, t_start, energy, now, self._near(self.road.positions[nodes[lo:pick + 1]],
                                                                          cfg.nearby_m))

    def _scout(self, idx, reach, start, t_start: float, energy: float, now: float, task: dict) -> Sortie:
        """A sortie over the road nodes a blockage cut off: it lands at the first index (from
        ``scout_min_s``) by which its tour covers them, judged against the longest it may fly
        (``scout_max_s``). Its belief scores what has been searched already, by any robot."""
        cfg, arr = self.cfg, np.asarray(self.plan["arrive"], float)
        claim = self._near(np.asarray(task["near"]), cfg.scout_radius_m)
        cand = [int(i) for i in idx if reach(i) >= 0 and arr[i] >= t_start + cfg.scout_min_s]
        if not cand:
            return self._sortie(next((int(i) for i in idx if reach(i) >= 0), int(idx[0])), start, t_start, energy,
                                now, claim)
        within = [i for i in cand if arr[i] <= t_start + cfg.scout_max_s] or cand[:1]
        step = max(1, len(within) // 12)  # tours for at most a dozen landings
        tried = within[::step] + ([within[-1]] if (len(within) - 1) % step else [])
        tours = [(i, self._tour(i, start, t_start, energy, now, claim)) for i in tried]
        best = max((t[3] for _, t in tours if t is not None), default=0.0)
        i, tour = next(((i, t) for i, t in tours if t is not None and t[3] >= cfg.cover_fraction * best), tours[0])
        return self._sortie(i, start, t_start, energy, now, claim, tour)

    def _deliver(self, idx, reach, start, t_start: float, energy: float, now: float, task: dict) -> Optional[Sortie]:
        """A messenger's sortie: to the UAV-grid node nearest where the other UAV waits, then a
        landing that leaves it ``deliver_wait_s`` there (else the first it can reach). None if it
        could not land anywhere in time (the dock taken for longer than it can fly): it stays."""
        air, nodes = self.planner.air.positions, np.asarray(self.plan["nodes"], int)
        arr, v = np.asarray(self.plan["arrive"], float), self.spec.speed
        g = int(np.argmin(np.hypot(*(air - np.asarray(task["xy"])).T)))
        length = {int(i): dist(start, air[g]) + dist(air[g], self.road.positions[nodes[i]]) for i in idx}
        fits = [i for i in length if reach(i) >= length[i]]
        if not fits:
            return None
        pick = next((i for i in fits if arr[i] >= t_start + length[i] / v + self.cfg.deliver_wait_s), fits[0])
        return self._sortie(pick, start, t_start, energy, now, tour=(0.0, [g], length[pick], 0.0))

    def _near(self, pts: np.ndarray, radius: float) -> np.ndarray:
        """Claims (``chain_tour``) that keep a tour to UAV-grid nodes within ``radius`` of ``pts``."""
        air = self.planner.air.positions
        near = np.zeros(len(air), dtype=bool)
        for p in np.asarray(pts, float).reshape(-1, 2):
            near |= np.hypot(*(air - p).T) <= radius
        return np.where(near, np.inf, -np.inf)

    def _step_sortie(self, s: RobotState, now: float, gs: Optional[dict], cmds: list) -> None:
        if self.task is not None and self.task.get("deliver"):
            if self._wait_to_deliver(s, now, cmds):
                return
        elif self.task is not None and self.cfg.trace_roads and self.booking is not None:
            self._trace(s, now)
        super()._step_sortie(s, now, gs, cmds)

    def _wait_to_deliver(self, s: RobotState, now: float, cmds: list) -> bool:
        """A messenger at the point it was sent to: wait there (True) until it hears the UAV it was
        sent to (relaying does the rest), or until it must leave to land on time."""
        if self.delivered or self.booking is None or len(self.waypoints) < 2:
            return False
        heard = self.relay is not None and self.relay.neighbours.get(self.task["deliver"], ({}, -np.inf))[1] >= self.t0
        land = self.waypoints[-1]
        if heard or now + dist(self.waypoints[0], land) / self.spec.speed + 5.0 >= self.booking["t_land"]:
            self.delivered = True
            return False
        if dist(s.position, self.waypoints[0]) >= 0.5:
            return False  # on the way there
        if s.goal_point is not None or s.moving:
            cmds.append(Hold())
        return True

    def _trace(self, s: RobotState, now: float) -> None:
        """Follow a road missing from the map near the blocked node (``trace_radius_m``): fly to the far end of one it
        has found (a new node with one known edge that it has not seen), dropping tour nodes from
        the end as needed to land on time with its reserve."""
        new, r = self.roads.new_nodes, self.spec.sensor.range_m
        self.seen.update(i for i, xy in new.items() if dist(xy, s.position) <= r)
        if self.tour and self.tour[0] == -1:
            return  # on the way to one
        deg: dict[int, int] = {}
        for a, b in self.roads.new_edges:
            deg[a], deg[b] = deg.get(a, 0) + 1, deg.get(b, 0) + 1
        front = [i for i, xy in new.items() if i not in self.seen and deg.get(i, 0) == 1
                 and dist(xy, self.task["xy"]) <= self.cfg.trace_radius_m]
        if not front:
            return
        f = min(front, key=lambda i: dist(new[i], s.position))
        wps, tour = [new[f]] + self.waypoints, [-1] + self.tour
        e, v, t_land = self.spec.energy, self.spec.speed, self.booking["t_land"]

        def length():
            pts = [s.position] + wps
            return sum(dist(a, b) for a, b in zip(pts, pts[1:]))
        while len(wps) > 2 and length() > v * (t_land - now):
            del wps[-2]
            tour.pop()
        L = length()
        hover = max(0.0, t_land - now - L / v) + self.setup.docking.dock_duration_s
        if L > v * (t_land - now) or \
                s.energy - e.travel_cost(L, v) - e.hover_cost(hover) < self.cfg.uav_reserve_fraction * e.capacity:
            return
        self.waypoints, self.tour = wps, tour
