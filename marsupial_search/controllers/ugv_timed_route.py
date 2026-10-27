"""UGV policy ``ugv_timed_route``: the mobile dock of :mod:`.dock_booking`.

Route
    Greedy belief-orienteering over the road graph (``SortiePlanner.ugv_route``), kept
    planned ``route_horizon_s`` ahead. Arrival times include the stops it has promised. Up
    to the last promised stop (the *pin*) the route is fixed; after it, the route is
    re-planned on the current belief every ``route_replan_s``.
Dock queue
    A booking is a landing at a route index and the take-off that follows once that UAV's
    battery has been swapped. The UGV stops at the landing node until the UAV has docked, and at the take-off
    node until it has left. A landing must come after the last promised take-off, so
    bookings are append-only and never move one another. A UAV that has not landed
    ``max_late_s`` after its landing time and slack, or after its landing should have ended
    (it ran out of energy on the way down), is dropped.
Broadcast
    The route, its arrival times and the queue (with each booked tour) in its ``"status"``
    when they change, and every ``route_resend_s``.
Sightings
    The UGV drives past a sighting at once, from the node it is at or driving to, when the
    detour delays its next stops by no more than the hover slack of every UAV still to land.
    Otherwise it goes there straight after the pin, and re-checks the detour at every node it
    gets to (and once a stop there is done) while a stop still comes before the sighting.
Road-map changes
    Found by the UGV or reported by a UAV. New roads are used from the next re-plan. If the
    route runs into a blocked node, it goes round it at once: by the shortest way back to the
    first node of the route after it that keeps the next landing within the hover slack of the
    UAVs still to land (with none to come, that keeps the route no longer than planned). The
    landings stay at their nodes, put back to when the UGV now gets there; the take-offs move
    back to the first node the UGV reaches once the battery is swapped.
Recovery
    A landing the UAV can no longer afford to wait for, one on a blocked node, and (after a
    re-route) one the UGV now gets to more than ``recover_gain_s`` late, moves to where the UGV
    best meets the UAV: a node of the route between the stops around it, or of the shortest
    way there (skipping the route in between). The UAV flies there from where its tour ends,
    as it reports in its status (or as booked). A UAV out of touch (no status for
    ``uav_stale_s``) is only met within radio range of that point, so it hears the UGV there.
Road planner (``road_planner``)
    How the UGV gets round a blocked node, to a sighting and to a moved landing (and, with
    ``ctp_search_legs``, from one node of its search route to the next it chooses).
    ``"optimistic"``: the shortest way on the road map as known, as if every node not yet found
    blocked were passable. ``"ctp"``: a Canadian Traveller's Problem policy (:mod:`.ctp`) that
    weighs the risk of blockages not yet seen and the way back from them. A node is open once a robot has seen it (the UGV from where
    it drives, a UAV from where its status says it is); every other mapped node is blocked
    with the share found blocked so far (prior ``block_prior``). ``ugv_ctp_route`` is this
    policy with ``"ctp"``.
Standby UAVs
    A UAV on standby (``uav_near_scout``) leaves the dock only when called: just before another UAV
    lands while it is on the dock, when the route runs into a blocked node within
    ``call_radius_m`` that cuts off nearby road nodes with belief left round them, to search
    them (``_cut_off``), and (``messenger``) as a messenger to a UAV out of touch that can no
    longer be met in time where it waits: it flies there, relaying tells that UAV the UGV's plan,
    and the UGV plans to meet it as if in touch from then. It lands wherever the dock is free and
    stays (see ``dock_booking``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from marsupial_search.core import (
    EventType,
    FollowPath,
    Hold,
    MissionSetup,
    RobotMode,
    RobotObservation,
    RobotSpec,
    RobotState,
    RobotType,
    Transmit,
    TraversabilityGraph,
)

from .common import bypass, dist, pursuit_path
from .ctp import BlockageModel, CtpPlanner, clear_probabilities
from .dock_booking import BOOK, LAND, SYNC, TAKEOFF, Booking, BookingRobot, DockBookingConfig, slot_spacing
from .sortie_planner import SortiePlanner


@dataclass
class UgvTimedRouteConfig(DockBookingConfig):
    route_horizon_s: float = 0.0  # route kept planned this far ahead (0: one slot per UAV plus a full sortie)
    route_replan_s: float = 60.0  # re-plan the unpromised part of the route this often
    route_resend_s: float = 5.0  # re-broadcast an unchanged route this often
    max_late_s: float = 120.0  # beyond its landing time and slack: the UGV gives up on a UAV
    uav_stale_s: float = 10.0  # a UAV's status older than this: it is out of touch (recovery meets it within radio range)
    recover_gain_s: float = 60.0  # after a re-route: move a late landing if that lands the UAV this much sooner
    call_radius_m: float = 250.0  # the route runs into a blocked node this close: maybe call a standby UAV to scout it
    call_reach_m: float = 500.0  # cut off: road nodes this far from the UGV by road (before the blockage) ...
    call_gain_m: float = 200.0  # ... that the blockage puts this much further away by road (or out of reach)
    call_min_belief: float = 0.005  # call the standby UAV if the belief it can see around them is at least this
    pocket_weight: float = 2.0  # belief round nodes the blockage puts out of reach counts this many times
    call_timeout_s: float = 15.0  # a standby UAV called to a blocked node that has not booked its landing by then stays docked
    messenger: bool = True  # a UAV out of touch can no longer be met in time: send the standby UAV to tell it where (with relay)
    road_planner: str = "optimistic"  # optimistic: shortest paths on the map as known | ctp: Canadian Traveller's Problem
    ctp_samples: int = 16  # ctp: blockage scenarios per decision
    ctp_search_legs: bool = False  # ctp: also the ways between the search route's nodes (else: detours, pursuits, moved landings)
    ctp_belief_weight: float = 1.0  # ctp search legs: belief swept on the way, in metres at the greedy route's rate, times this (0: off)
    ctp_reuse_s: float = 60.0  # ctp: re-use a way planned between the same two nodes this long (until the map changes)
    block_prior: float = 0.05  # ctp: prior share of road nodes blocked ...
    block_prior_weight: float = 20.0  # ... worth this many road nodes seen
    max_cutoff_fraction: float = 0.0  # ctp: blockages cut off at most this share of the mapped road nodes (the generator's road_cutoff_fraction; 1: no limit)

    def __post_init__(self):
        if self.road_planner not in ("optimistic", "ctp"):
            raise ValueError(f"road_planner must be 'optimistic' or 'ctp', got {self.road_planner!r}")


@dataclass
class UgvCtpRouteConfig(UgvTimedRouteConfig):
    road_planner: str = "ctp"


class TimedRouteUgv(BookingRobot):
    """Drives a committed, timed route and stops for the landings and take-offs booked on it."""

    name = "ugv_timed_route"
    config_cls = UgvTimedRouteConfig

    def reset(self, robot_id: str, setup: MissionSetup) -> None:
        super().reset(robot_id, setup)
        if len(setup.robots_of(RobotType.UGV)) != 1:
            raise ValueError(f"{self.name} supports exactly one UGV")
        self.uavs = setup.robots_of(RobotType.UAV)
        self.planner = (SortiePlanner(setup, self.spec, setup.robots[self.uavs[0]], self.cfg) if self.uavs
                        else None)
        self.nodes: list[int] = []  # the route; an index into it is a route index
        self.cur = 0  # route index the UGV is at, or last passed
        self.target = 0  # route index the current FollowPath ends at
        self.arrived = 0.0  # when it reached nodes[cur]
        self.stops: dict[int, list[dict]] = {}  # route index -> stops still to do there, in order
        self.version = 0  # changes when route indices or promised times change: bookings planned on older ones are refused
        self.revision = 0  # changes with anything in the broadcast (appending to the route only changes this)
        self.sent_revision, self.sent_at = -1, -np.inf
        self.sent_path: Optional[tuple] = None
        self.last_replan = -np.inf
        self.last_land: Optional[float] = None  # latest booked landing time
        self.bookings: dict[str, dict] = {}  # UAV -> its booked landing (and take-off)
        self.lost: set[str] = set()
        self.pursuit = "none"  # none -> planned (route goes past the sighting) -> done (passed it)
        self.pursuit_end = 0  # route index where the pursuit path ends
        self.pursuit_check: Optional[tuple] = None  # pursuit planned after the pin: where the detour was last checked
        self.swap_until = -np.inf  # a docked UAV's battery swap ends then (held for, without swap_while_moving)
        self.swap_end: dict[str, float] = {}  # UAV -> when the battery swap after its last landing ends
        self.standby: set[str] = set()  # UAVs that stay on the dock until called (they say so in their status)
        self.free_from: Optional[float] = None  # a standby UAV is on the dock after the pin: it can leave from then
        self.extend_failed = False
        self.messengers: dict[str, dict] = {}  # UAV out of touch -> {"by": the standby UAV sent to tell it, "t": when it can}
        self.messenger_failed: set[tuple[str, int]] = set()  # (UAV, plan_id): a messenger called for that landing never left
        self.blockage = BlockageModel(self.roads, self.cfg.block_prior, self.cfg.block_prior_weight,
                                      self.cfg.max_cutoff_fraction) if self.cfg.road_planner == "ctp" else None
        self.ctp: Optional[CtpPlanner] = None
        self.ctp_paths: dict[tuple, tuple[float, list[int], float]] = {}  # (src, dst, search leg) -> (when, path, cost)
        self.detour_cache: dict = {}  # (prev, node, next) -> would its blockage send the UGV the long way round?
        self.detour_road: Optional[TraversabilityGraph] = None  # the road map detour_cache is for
        self.now = 0.0

    def _start(self, s: RobotState) -> None:
        self._look(s.position, self.spec.sensor.range_m)
        self.nodes, self.arrived = [s.node if s.node is not None else s.last_node], 0.0
        if s.docked_uav is not None:  # it takes off first, from where the UGV stands
            self.stops[0] = [{"kind": TAKEOFF, "uav": s.docked_uav, "ready": 0.0}]
        self._extend(0.0, s)

    # ------------------------------------------------------------ schedule
    def _pin(self) -> int:
        """Last route index with a promised stop (or the current one): fixed up to here."""
        return max([self.cur] + [i for i, st in self.stops.items() if st])

    def _stop_duration(self, stop: dict) -> float:
        d = self.setup.docking
        if stop["kind"] == LAND:  # the UGV also waits out the swap if it may not drive during it
            return d.dock_duration_s + (0.0 if d.swap_while_moving else d.swap_duration_s) + self.cfg.stop_margin_s
        return d.undock_duration_s + self.cfg.stop_margin_s

    def _swap_hold(self) -> float:
        """Until when the UGV must stand still for a battery swap (-inf: it may drive during swaps)."""
        return -np.inf if self.setup.docking.swap_while_moving else self.swap_until

    def _depart(self, i: int, arrive: float, now: float = -np.inf) -> float:
        """Departure from route index ``i``: after its stops (a stop not yet started starts now at the earliest)."""
        t = arrive
        for stop in self.stops.get(i, []):
            t = (stop["started"] if "started" in stop else max(t, stop["ready"], now)) + self._stop_duration(stop)
        return t

    def _times(self, s: RobotState, now: float) -> np.ndarray:
        """Expected arrival at each route index from ``cur`` on (nan before it)."""
        pos, v, n = self.road.positions, self.spec.speed, len(self.nodes)
        arr = np.full(n, np.nan)
        arr[self.cur] = self.arrived
        if self.cur + 1 < n:
            legs = np.hypot(*np.diff(pos[self.nodes[self.cur:]], axis=0).T) / v
            if s.node is not None and s.node == self.nodes[self.cur]:  # at nodes[cur]
                t0 = max(now, self._depart(self.cur, self.arrived, now), self._swap_hold())
            else:  # on the way to nodes[cur + 1]
                t0 = now + dist(s.position, pos[self.nodes[self.cur + 1]]) / v - legs[0]
            arr[self.cur + 1:] = t0 + np.cumsum(legs)
            for i in sorted(i for i, st in self.stops.items() if st and self.cur < i < n - 1):
                arr[i + 1:] += self._depart(i, arr[i]) - arr[i]  # the stops there hold up the rest
        return arr

    # --------------------------------------------------------------- route
    def _horizon(self) -> float:
        if self.cfg.route_horizon_s > 0:
            return self.cfg.route_horizon_s
        n = max(len(self.uavs) - len(self.lost | self.standby), 1)
        uav = self.setup.robots[self.uavs[0]]
        e = uav.energy
        sortie = e.capacity / (e.idle_power + e.energy_per_m * uav.speed)
        return (n + 1) * slot_spacing(self.setup, self.cfg, n) + sortie

    def _extend(self, now: float, s: RobotState, until: float = 0.0, replan: bool = False) -> None:
        """Once the route ends within the horizon (or before ``until``), plan it on from its end
        to 1.5 horizons. Once nothing is left to drive for, it goes there and back to a neighbour
        (so the UAVs can still book landings) and tries planning again only when ``replan``."""
        if self.planner is None:
            return
        horizon = self._horizon()
        arr = self._times(s, now)
        if arr[-1] >= max(now + horizon, until):
            return
        end = max(now + 1.5 * horizon, until)
        src = self.nodes[-1]
        if not self.extend_failed or replan:
            P = self.planner
            m = P.searchable_mass(self.belief)
            for n in self.nodes[self.cur:]:  # what the route will sweep anyway
                P.sweep(m, P.F_road, n, P.p_ugv)
            cand = ~self.road_visited[:self.road.n_nodes]
            cand[self.nodes[self.cur:]] = False
            claim = np.full(P.air.n_nodes, np.inf)
            for b in self.bookings.values():
                claim[list(b["tour"])] = -np.inf
            new, _ = P.ugv_route(m, src, tuple(self.road.positions[src]), cand, end - arr[-1], claim,
                                 self._ctp_path_fn() if self.cfg.ctp_search_legs else None, self.cfg.ctp_belief_weight)
            self.extend_failed = len(new) < 2
            if not self.extend_failed:
                self.nodes += [int(n) for n in new[1:]]
                self.revision += 1
                return
        nbs = self.road.neighbors(src)
        if nbs:
            nb, length = nbs[0]
            self.nodes += [int(nb), src] * int(np.ceil((end - arr[-1]) / (2 * length / self.spec.speed)))
            self.revision += 1

    def _replan_tail(self, now: float, s: RobotState) -> None:
        """Drop the route after the last promised stop (and the node being driven to), then extend."""
        keep = max(self._pin(), self.cur + 1, self.target, self.pursuit_end if self.pursuit == "planned" else 0)
        if keep + 1 < len(self.nodes):
            del self.nodes[keep + 1:]
            self.version += 1
            self.revision += 1
        self.last_replan = now
        self._extend(now, s, replan=True)

    # ------------------------------------------------------------- road map
    def _path(self, src: int, dst: int, reward: Optional[np.ndarray] = None) -> tuple[list[int], float]:
        """The way from ``src`` to ``dst`` (``road_planner``) and its cost: its length, or (ctp)
        the expected length with the detours round blockages not yet seen, less the ``reward``
        collected on the way (a search leg's belief, in metres)."""
        if self.blockage is None:
            return self.road.shortest_path(src, dst)
        if self.ctp is None or self.ctp.road is not self.road:
            self.ctp = CtpPlanner(self.road, self.spec.sensor.range_m, self.cfg.ctp_samples,
                                  seed=self.road.n_nodes + len(self.roads.blocked))
            self.ctp_paths = {}
        key = (src, dst, reward is not None)
        t, path, cost = self.ctp_paths.get(key, (-np.inf, None, 0.0))
        if self.now - t > self.cfg.ctp_reuse_s:
            path, cost = self.ctp.path(src, dst, self.blockage.probabilities(self.road), reward)
            self.ctp_paths[key] = (self.now, path, cost)
        return list(path), cost

    def _ctp_path_fn(self):
        """``_path`` for the shared planners when it is the CTP (None: their own shortest paths)."""
        return self._path if self.blockage is not None else None

    def _look(self, xy, radius: float) -> None:
        if self.blockage is not None:
            self.blockage.look(self.road, xy, radius - 1.0)  # (a margin for where it was when it looked)

    def _ingest(self, sender: str, p: dict, t: float) -> None:
        super()._ingest(sender, p, t)
        if p["type"] == RobotType.UAV.value and sender in self.setup.robots:  # what that UAV has seen
            self._look(p["pos"], self.setup.robots[sender].sensor.range_m)

    def _map_changed(self, s: RobotState, now: float) -> None:
        """Re-route at once if the route runs into a blocked node; new roads wait for the next re-plan."""
        if self.planner is None:
            return
        old = self.planner.road
        self.planner.set_road(self.road)
        ahead = self.nodes[self.cur:]
        if not all(self.road.has_edge(a, b) for a, b in zip(ahead, ahead[1:])):
            self._reroute(s, now, old)

    def _reroute(self, s: RobotState, now: float, old: Optional[TraversabilityGraph] = None) -> None:
        """Go round the blocked nodes on the route (``bypass``), delaying the landings still to
        come by no more than the hover slack those UAVs have left. The landings stay at their
        nodes, except one on a blocked node: it goes to wherever the UAV is best met
        (``_check_landings``), as does one the UGV now gets to too late. The take-offs follow
        (``_pull_takeoffs``). A blocked node near the UGV may call a standby UAV (``_call_to_blocked``),
        judged on the road map before the change (``old``)."""
        at_node = s.node is not None and s.node == self.nodes[self.cur]
        head = min(self.cur if at_node else self.cur + 1, len(self.nodes) - 1)  # the node it is at, or driving to
        hit = next((n for n in self.nodes[head:] if self.road.blocked[n]), None)
        ahead = [i - head for i, sts in self.stops.items() if i > head and any(st["kind"] == LAND for st in sts)]
        arr = self._times(s, now)
        spare = min((st["ready"] + st["slack"] - arr[i] for i, sts in self.stops.items() if i > head
                     for st in sts if st["kind"] == LAND), default=0.0)  # the UGV may come this much later
        tail, index = bypass(self.road, self.nodes[head:], ahead, max(spare, 0.0) * self.spec.speed, self._ctp_path_fn())
        index = np.concatenate([np.arange(head), head + np.where(index >= 0, index, -head - 1)])
        before = np.maximum.accumulate(index)  # the new index of the last kept old index up to each one
        self.nodes = self.nodes[:head] + tail
        stops = {}
        for i, sts in sorted(self.stops.items()):
            stops.setdefault(int(before[i]), []).extend(sts)
        self.stops = stops
        takeoffs = {st["uav"]: i for i, sts in self.stops.items() for st in sts if st["kind"] == TAKEOFF}
        for b in self.bookings.values():  # their landings keep their nodes (unless blocked), but may come later
            b["moved"] = index[b["land_idx"]] < 0
            b["land_idx"] = int(before[b["land_idx"]])
            b["off_idx"] = takeoffs[b["uav"]] if b["off_idx"] is not None else None  # (a standby UAV has none)
            b["land_node"] = self.nodes[b["land_idx"]]
        if self.pursuit == "planned":
            self.pursuit_end = int(index[self.pursuit_end]) if self.pursuit_end < len(index) else -1
            if self.pursuit_end < 0:
                self.pursuit = "none"  # plan it again
        self.version += 1
        self.revision += 1
        self.target, self.sent_path = head, None
        if hit is not None and old is not None and dist(s.position, self.road.positions[hit]) <= self.cfg.call_radius_m:
            self._call_to_blocked(s, now, head, tuple(self.road.positions[hit]), old)
        self._check_landings(s, now, retime=True)
        self._pull_takeoffs(s, now)
        self._extend(now, s)

    def _splice(self, a: int, j: int, path: list[int]) -> None:
        """Replace the route from index ``a`` to ``j`` (no stops in between) by ``path``, from
        ``nodes[a]`` to ``nodes[j]``; the route indices from ``j`` on shift."""
        shift = a + len(path) - 1 - j
        self.nodes[a:j + 1] = [int(n) for n in path]
        self.stops = {(i + shift if i >= j else i): sts for i, sts in self.stops.items() if sts}
        for b in self.bookings.values():
            b["land_idx"] += shift if b["land_idx"] >= j else 0
            if b["off_idx"] is not None and b["off_idx"] >= j:
                b["off_idx"] += shift
        if self.pursuit == "planned" and self.pursuit_end > a:
            if self.pursuit_end > j:
                self.pursuit_end += shift
            else:
                self.pursuit = "none"  # its path was skipped: plan it again

    # ------------------------------------------------------------- recovery
    def _free(self, b: dict, now: float) -> tuple[tuple[float, float], float, float, bool]:
        """Where and when the UAV of booking ``b`` is free to land, the energy it can spend after
        that (flying on, hovering, landing with its reserve), and whether it is in touch: from
        its status if fresh, else as booked."""
        p = self.peers.get(b["uav"])
        fresh = p is not None and now - p["t"] <= self.cfg.uav_stale_s
        x, y, t, spare = p["free"] if fresh and p.get("free") is not None else b["free"]
        return (x, y), t, spare, fresh

    def _meet(self, u: str, free, pos: np.ndarray, arr: np.ndarray):
        """Meeting UAV ``u`` (free at ``free`` = (xy, t)) at road positions ``pos``, where the UGV
        arrives at ``arr``: landing times, the UAV's energy for it, and a score (landing time
        plus the UGV's wait)."""
        uav = self.setup.robots[u]
        e = uav.energy
        d = np.hypot(*(pos - free[0]).T)
        ta = free[1] + d / uav.speed
        tm = np.maximum(arr, ta)
        return tm, e.energy_per_m * d + e.idle_power * (tm - free[1]), tm + np.maximum(ta - arr, 0.0)

    def _check_landings(self, s: RobotState, now: float, retime: bool = False) -> None:
        """Recovery: a landing the UAV can no longer afford to wait for (or moved off a blocked
        node) goes to where the UGV can best meet that UAV: a node of the route between the
        stops around it, or of the shortest way to the landing node (the UGV then skips the
        route in between). The UAV flies there from where its tour ends. A UAV out of touch is
        only met within radio range of that point, so it hears the UGV there. ``retime`` (after
        a re-route): also move a landing the UGV now gets to late if that lands the UAV
        ``recover_gain_s`` sooner, and otherwise put it back to when the UGV gets there."""
        if not self.bookings:
            return
        at_node = s.node is not None and s.node == self.nodes[self.cur]
        head = self.cur if at_node else self.cur + 1
        pos, link = self.road.positions, self._link_range()
        arr = self._times(s, now)
        n = len(self.nodes)
        for u, b in list(self.bookings.items()):
            j0, moved = b["land_idx"], b.pop("moved", False)
            land = next(st for st in self.stops[j0] if st["kind"] == LAND and st["uav"] == u)
            if "started" in land or j0 < head:
                continue
            xy, t, spare, fresh = self._free(b, now)
            if not fresh and u in self.messengers:  # it hears the new plan once the messenger gets there
                t_told = max(t, self.messengers[u]["t"])
                spare -= self.setup.robots[u].energy.idle_power * (t_told - t)
                t, fresh = t_told, True
            free = (np.asarray(xy), t)
            tm0, c0, score0 = self._meet(u, free, pos[[self.nodes[j0]]], arr[[j0]])
            late = retime and tm0[0] - t > self.cfg.recover_gain_s  # the UAV would wait this long for the UGV
            if not moved and c0[0] <= spare and not late:
                if retime and tm0[0] > b["t_land"]:
                    self._set_landing(b, land, j0, float(tm0[0]), spare - c0[0])
                continue
            others = [i for i, sts in self.stops.items() for st in sts if st["uav"] != u and st.get("call") != u]
            lo = max([head] + [i + 1 for i in others if i < j0] + [j0 for i in others if i == j0])
            off = n if b["off_idx"] is None else max(b["off_idx"], j0 + 1)  # (a re-route may put its take-off at j0)
            hi = min([off] + [i for i in others if i > j0]) - 1
            busy = self._busy(s, but=u) if u in self.standby else np.zeros(n, bool)  # (where another UAV is on the dock)
            # Candidates: the route from lo to hi, and the shortest way from the last fixed index to j0.
            cand = [(np.arange(lo, hi + 1), np.asarray(self.nodes[lo:hi + 1], dtype=int), arr[lo:hi + 1], None)]
            a = max(lo - 1, head)
            if a < j0:
                path, _ = self._path(self.nodes[a], self.nodes[j0])
                if path and self.road.path_length(path) < self.road.path_length(self.nodes[a:j0 + 1]) - 1.0:
                    dep = arr[a + 1] - dist(pos[self.nodes[a]], pos[self.nodes[a + 1]]) / self.spec.speed
                    legs = np.hypot(*np.diff(pos[path], axis=0).T)
                    t_path = dep + np.concatenate([[0.0], np.cumsum(legs)]) / self.spec.speed
                    k = np.arange(len(path))[a + np.arange(len(path)) >= lo]
                    cand.append((a + k, np.asarray(path)[k], t_path[k], path))
            best = None  # (infeasible, score or energy, ...)
            for idx, nodes, t_arr, path in cand:
                ok = ~self.road.blocked[nodes] & ~busy[idx]
                if not fresh:  # out of touch: where it will hear the UGV (with no radio: only at its landing node)
                    ok &= np.hypot(*(pos[nodes] - free[0]).T) <= link if link is not None else nodes == self.nodes[j0]
                if not ok.any():
                    continue
                tm, c, score = self._meet(u, free, pos[nodes[ok]], t_arr[ok])
                feas = c <= spare
                i = int(np.argmin(np.where(feas, score, np.inf))) if feas.any() else int(np.argmin(c))
                key = (not feas[i], score[i] if feas[i] else c[i])
                if best is None or key < best[0]:
                    best = (key, int(idx[ok][i]), float(tm[i]), float(c[i]), path)
            if not fresh and c0[0] > spare and (best is None or best[0][0]) \
                    and self._send_messenger(s, now, u, xy, head, arr):
                b["moved"] = moved  # re-planned next step, as if in touch from when the messenger gets there
                continue
            if c0[0] <= spare:  # affordable: only for a landing that much sooner
                better = best is not None and not best[0][0] and best[0][1] < score0[0] - self.cfg.recover_gain_s
            else:
                better = best is not None and best[3] < c0[0] - 1.0
            if not (moved and best is not None or better):
                if retime and tm0[0] > b["t_land"]:
                    self._set_landing(b, land, j0, float(tm0[0]), spare - c0[0])
                continue
            _, j, tm, c, path = best
            self.stops[j0].remove(land)
            if path is not None:  # take the shortest way to the landing node
                self._splice(a, j0, path)
                self.target, self.sent_path = head, None
            sts = self.stops.setdefault(j, [])  # (before its own take-off, which a re-route may have put here)
            sts.insert(next((k for k, st in enumerate(sts) if st["kind"] == TAKEOFF and st["uav"] == u), len(sts)), land)
            self._set_landing(b, land, j, tm, spare - c)
            self._pull_takeoffs(s, now)
            arr = self._times(s, now)
            n = len(self.nodes)

    def _send_messenger(self, s: RobotState, now: float, u: str, xy, head: int, arr: np.ndarray) -> bool:
        """UAV ``u`` is out of touch and can no longer be met in time where it waits (``xy``): call
        the standby UAV on the dock (its battery swapped within a minute) to fly there, at route
        index ``head``. Relaying tells ``u`` the UGV's plan once the messenger is in radio range; from
        then the UGV plans to meet ``u`` as if in touch (``_check_landings``). False if it cannot."""
        m, link = s.docked_uav, self._link_range()
        if not (self.cfg.messenger and self.cfg.relay) or link is None or u in self.messengers \
                or (u, self.bookings[u]["plan_id"]) in self.messenger_failed \
                or m is None or m == u or m not in self.standby or m in self.lost \
                or self.swap_end.get(m, 0.0) > now + 60.0 \
                or any(st["uav"] == m and "started" in st for sts in self.stops.values() for st in sts):
            return False
        for i in list(self.stops):  # it leaves from here: its other take-offs (e.g. to make room) go
            self.stops[i] = [st for st in self.stops[i] if not (st["uav"] == m and st["kind"] == TAKEOFF and i != head)]
        self._pull_takeoffs(s, now)  # the dock free again as early as it can be: the messenger books its landing at once
        arr = self._times(s, now)
        ready, dock = self.swap_end.get(m, 0.0), self.setup.docking
        t_off = max(arr[head], ready, now) + dock.undock_duration_s + self.cfg.stop_margin_s
        reach = max(dist(self.road.positions[self.nodes[head]], xy) - 0.5 * link, 0.0)
        call = {"deliver": u, "xy": tuple(map(float, xy)), "near": None}
        sts = self.stops.setdefault(head, [])
        own = next((st for st in sts if st["uav"] == m and st["kind"] == TAKEOFF), None)
        if own is not None:
            own.update(call)
        else:
            sts.insert(sum("started" in st for st in sts), {"kind": TAKEOFF, "uav": m, "ready": ready, "call": None, **call})
        self.messengers[u] = {"by": m, "t": t_off + reach / self.setup.robots[m].speed}
        self.version += 1
        self.revision += 1
        return True

    def _check_messengers(self, s: RobotState) -> None:
        """Forget a messenger once its UAV's landing is over, or if it never left (its call withdrawn,
        e.g. it could not land anywhere in time): then not again for that landing."""
        for u, info in list(self.messengers.items()):
            called = any(st.get("deliver") == u and st.get("xy") is not None for sts in self.stops.values() for st in sts)
            if u not in self.bookings:
                del self.messengers[u]
            elif s.docked_uav == info["by"] and not called:
                self.messenger_failed.add((u, self.bookings[u]["plan_id"]))
                del self.messengers[u]

    def _pull_takeoffs(self, s: RobotState, now: float) -> None:
        """Move each take-off still to come back to the first route node after the stops before
        it that the UGV reaches once that UAV's battery is swapped (after a detour or an earlier
        landing, it may be far later than it needs to be)."""
        at_node = s.node is not None and s.node == self.nodes[self.cur]
        head = self.cur if at_node else self.cur + 1
        arr = self._times(s, now)
        for i in sorted(i for i, sts in self.stops.items() if sts and sts[0]["kind"] == TAKEOFF):
            st = self.stops[i][0]
            if "started" in st or i <= head or "call" in st:  # a standby UAV leaves where it is called
                continue
            lo = max([head] + [k + 1 for k, sts in self.stops.items() if sts and k < i])
            sooner = [k for k in range(lo, i) if arr[k] >= st["ready"]]
            if not sooner:
                continue
            self.stops[i].pop(0)
            self.stops.setdefault(sooner[0], []).insert(0, st)
            if st["uav"] in self.bookings:
                self.bookings[st["uav"]]["off_idx"] = sooner[0]
            self.version += 1
            self.revision += 1
            arr = self._times(s, now)

    def _set_landing(self, b: dict, land: dict, j: int, t_land: float, spare: float) -> None:
        """Land UAV ``b["uav"]`` at route index ``j`` at ``t_land``, with ``spare`` energy left for waiting."""
        dock = self.setup.docking
        b["land_idx"], b["land_node"], b["t_land"] = j, self.nodes[j], t_land
        b["slack"] = land["slack"] = max(spare, 0.0) / self.setup.robots[b["uav"]].energy.idle_power
        land["ready"] = t_land
        b["t_ready"] = t_land + dock.dock_duration_s + dock.swap_duration_s
        for st in self.stops.get(b["off_idx"], []):
            if st["kind"] == TAKEOFF and st["uav"] == b["uav"]:
                st["ready"] = b["t_ready"]
        self.last_land = max(self.last_land, t_land)
        self.version += 1
        self.revision += 1

    def _link_range(self) -> Optional[float]:
        r = self.setup.radio_range_m
        return None if r is None else min(r[RobotType.UGV], r[RobotType.UAV])

    # ------------------------------------------------------------- bookings
    def _accept(self, b: Booking, s: RobotState, now: float) -> bool:
        """A landing after the pin, and the take-off once the battery is swapped. A standby UAV
        lands wherever the dock is free and stays on it (no take-off until it is called)."""
        if b.version != self.version or b.uav in self.bookings or b.uav in self.lost:
            return False
        standby = b.uav in self.standby
        if standby:
            if not self.cur < b.land_idx < len(self.nodes) or self._busy(s)[b.land_idx]:
                return False
        elif not self._pin() < b.land_idx < len(self.nodes) or self._unbooked_standby(s):
            return False
        dock = self.setup.docking
        land = {"kind": LAND, "uav": b.uav, "ready": b.t_land, "plan_id": b.plan_id, "slack": b.slack_s}
        self.stops.setdefault(b.land_idx, []).append(land)
        ready = b.t_land + dock.dock_duration_s + dock.swap_duration_s
        off = None
        if not standby:
            self._extend(now, s, until=ready)  # the take-off must be on the route
            arr = self._times(s, now)
            later = [i for i in range(b.land_idx + 1, len(self.nodes)) if arr[i] >= ready]
            off = later[0] if later else len(self.nodes) - 1
            self.stops.setdefault(off, []).append({"kind": TAKEOFF, "uav": b.uav, "ready": ready})
        self.bookings[b.uav] = {"uav": b.uav, "plan_id": b.plan_id, "land_idx": b.land_idx,
                                "land_node": self.nodes[b.land_idx], "t_land": b.t_land, "off_idx": off,
                                "t_ready": ready, "slack": b.slack_s, "tour": b.tour,
                                "free": (*self.road.positions[self.nodes[b.land_idx]], b.t_land,
                                         b.slack_s * self.setup.robots[b.uav].energy.idle_power)}
        self.last_land = b.t_land if self.last_land is None else max(self.last_land, b.t_land)
        self.version += 1
        self.revision += 1
        return True

    # ------------------------------------------------------------- standby
    def _spans(self, s: RobotState) -> list[tuple[str, int, Optional[int]]]:
        """(UAV, from, to): the route indices a UAV is on the dock, from its landing (or now) to its
        take-off (None: it is still on the dock at the end of the stops)."""
        on, out = ({s.docked_uav: self.cur} if s.docked_uav is not None else {}), []
        for i in sorted(i for i, sts in self.stops.items() if sts and i >= self.cur):
            for st in self.stops[i]:
                if st["kind"] == LAND:
                    on.setdefault(st["uav"], i)
                elif st["uav"] in on:
                    out.append((st["uav"], on.pop(st["uav"]), i))
        return out + [(u, i, None) for u, i in on.items()]

    def _busy(self, s: RobotState, but: Optional[str] = None) -> np.ndarray:
        """Per route index: is a UAV (other than ``but``) on the dock?"""
        busy = np.zeros(len(self.nodes), dtype=bool)
        for u, a, b in self._spans(s):
            if u != but:
                busy[a:len(self.nodes) if b is None else b + 1] = True
        return busy

    def _unbooked_standby(self, s: RobotState) -> bool:
        """Is a standby UAV in the air with no landing booked? (Other UAVs book after it.)"""
        on = {u for u, _, b in self._spans(s) if b is None}
        return any(u not in on and u not in self.bookings and self.peers.get(u, {}).get("mode") == RobotMode.ACTIVE.value
                   for u in self.standby - self.lost)

    def _call_standby(self, s: RobotState, now: float) -> None:
        """A standby UAV leaves the dock only when called: right before another UAV lands while it
        is on the dock, or to scout a blocked node (``_call_to_blocked``). Its other take-offs go
        (e.g. the one planned before the UGV knew it was on standby), as does a call to a blocked
        node it has not booked a landing for within ``call_timeout_s`` of the UGV holding for it."""
        changed, self.free_from = False, None
        at_node = s.node is not None and s.node == self.nodes[self.cur]
        lead = self.setup.docking.undock_duration_s + self.cfg.stop_margin_s  # making room: leave just before the landing
        for u in sorted(self.standby - self.lost):
            on, ready = s.docked_uav == u, self.swap_end.get(u, 0.0)
            for i in sorted(i for i, sts in self.stops.items() if sts and i >= self.cur):
                sts, k = self.stops[i], 0
                while k < len(sts):
                    st = sts[k]
                    if st["uav"] == u and st["kind"] == LAND:
                        if on and "started" not in st:  # it never left (its call was withdrawn)
                            del sts[k]
                            self.bookings.pop(u, None)
                            changed = True
                            continue
                        on, ready = True, self.bookings[u]["t_ready"]
                    elif st["uav"] == u and "started" in st:
                        on = False
                    elif st["uav"] == u:
                        makes_room = k + 1 < len(sts) and sts[k + 1]["kind"] == LAND and sts[k + 1]["uav"] != u
                        if at_node and i == self.cur and now >= st["ready"] and st.get("xy") is not None:
                            held = st.setdefault("held", now)
                            if now - held > self.cfg.call_timeout_s and u not in self.bookings:
                                st["xy"] = st["near"] = None  # never booked: no scouting
                        if not on or not (makes_room or st.get("xy") is not None):
                            del sts[k]
                            changed = True
                            continue
                        on = False
                        want = ready if st.get("xy") is not None else max(ready, sts[k + 1]["ready"] - lead)
                        if st["ready"] != want:
                            st["ready"], changed = want, True
                    elif st["kind"] == LAND and on and "started" not in st:  # make room for it
                        sts.insert(k, {"kind": TAKEOFF, "uav": u, "ready": max(ready, st["ready"] - lead),
                                       "call": st["uav"]})
                        on, changed, k = False, True, k + 1
                    k += 1
            if on:
                self.free_from = ready if self.free_from is None else max(self.free_from, ready)
        if changed:
            self.version += 1
            self.revision += 1

    def _call_to_blocked(self, s: RobotState, now: float, head: int, xy: tuple[float, float],
                         old: TraversabilityGraph) -> None:
        """Call a standby UAV on the dock (its battery swapped by the time the UGV gets to route index
        ``head``) to search the road nodes the blocked node at ``xy`` cuts off (``_cut_off``), if
        enough belief it can see is left round them: it takes off at ``head``."""
        arr = self._times(s, now)
        for u in sorted(self.standby - self.lost):
            if s.docked_uav != u or any(st["uav"] == u and "started" in st for sts in self.stops.values() for st in sts) \
                    or self.swap_end.get(u, 0.0) > arr[head]:
                continue
            cut = self._cut_off(old, self.nodes[head], self.setup.robots[u])
            if cut is None:
                return
            near = [tuple(map(float, self.road.positions[n])) for n in cut]
            sts = self.stops.setdefault(head, [])
            own = next((st for st in sts if st["uav"] == u and st["kind"] == TAKEOFF), None)
            if own is not None:  # it leaves here anyway
                own["xy"], own["near"] = xy, near
            else:
                k = sum("started" in st for st in sts)
                sts.insert(k, {"kind": TAKEOFF, "uav": u, "ready": self.swap_end.get(u, 0.0), "call": None, "xy": xy,
                               "near": near})
            self.version += 1
            self.revision += 1
            return

    def _cut_off(self, old: TraversabilityGraph, src: int, uav: RobotSpec) -> Optional[np.ndarray]:
        """The road nodes a blockage cuts off from the UGV at node ``src``: within ``call_reach_m`` of
        it by road on the map before (``old``), and now ``call_gain_m`` further away or out of
        reach. None if the belief ``uav`` can see round them (the UGV's footprint there; a node
        out of reach counts ``pocket_weight`` times) is below ``call_min_belief``: e.g. searched
        already, by any robot."""
        cfg, P = self.cfg, self.planner
        d0, _ = old.distances_from(src, limit=cfg.call_reach_m)
        d1, _ = self.road.distances_from(src)
        near = np.flatnonzero(np.isfinite(d0) & ~self.road.blocked[:old.n_nodes])
        cut = near[d1[near] - d0[near] >= cfg.call_gain_m]
        if not len(cut):
            return None
        m = P.seen(P.searchable_mass(self.belief), uav.sensor.layer_p())

        def mass(nodes):
            return float(m[np.unique(P.F_road[nodes].indices)].sum()) if len(nodes) else 0.0
        pocket = cut[~np.isfinite(d1[cut])]
        value = mass(cut) + (cfg.pocket_weight - 1.0) * mass(pocket)
        return cut if value >= cfg.call_min_belief else None

    def _drop(self, u: str) -> None:
        """Forget every stop for ``u`` (lost, or never came)."""
        for i in list(self.stops):
            self.stops[i] = [st for st in self.stops[i] if st["uav"] != u]
        self.bookings.pop(u, None)
        self.version += 1
        self.revision += 1

    def _plan_payload(self, s: RobotState, now: float) -> dict:
        arr = self._times(s, now)
        takeoffs, calls = {}, {}  # each UAV's next take-off
        for i in sorted(self.stops):
            for st in self.stops[i]:
                if st["kind"] == TAKEOFF and st["uav"] not in takeoffs:
                    takeoffs[st["uav"]] = (i, st["ready"])
                    if "call" in st:
                        calls[st["uav"]] = {"for": st["call"], "xy": st.get("xy"), "near": st.get("near"),
                                            "deliver": st.get("deliver")}
        out = {"version": self.version, "cur": self.cur, "nodes": self.nodes[self.cur:],
               "arrive": arr[self.cur:].tolist(), "pin": self._pin(), "last_land": self.last_land,
               "queue": list(self.bookings.values()), "takeoffs": takeoffs, "lost": sorted(self.lost),
               "standby": sorted(self.standby), "calls": calls, "free_from": self.free_from,
               "busy": [(a, b) for _, a, b in self._spans(s)]}
        if self.roads:  # every road-map change it knows: the route may use roads the UAVs have not seen
            out["map"] = self.roads.changes()
        if self.blockage is not None:  # (ctp) per route index: the chance of getting there with no long detour
            if self.detour_road is not self.road:
                self.detour_cache, self.detour_road = {}, self.road
            slack_m = (self.cfg.rendezvous_slack_s + self.cfg.hover_margin_s) * self.spec.speed
            out["clear"] = clear_probabilities(self.road, self.nodes[self.cur:], self.blockage.probabilities(self.road),
                                               slack_m, self.detour_cache).round(3).tolist()
        return out

    # ---------------------------------------------------------------- step
    def _step(self, s: RobotState, now: float, obs: RobotObservation, cmds: list) -> None:
        if self.planner is None:
            return
        self.now = now
        self._track(s, now)
        self._handle_events(obs, now)
        self._handle_messages(obs, s, now, cmds)
        self.standby |= {rid for rid, p in self.peers.items() if p.get("standby")}
        self._check_map(s, now)
        self._look(s.position, self.spec.sensor.range_m)
        for rid, p in self.peers.items():
            if p["type"] == RobotType.UAV.value and p["mode"] == RobotMode.DEPLETED.value and rid not in self.lost:
                self.lost.add(rid)
                self._drop(rid)
        self._give_up_late(s, now)
        self._check_messengers(s)
        self._check_landings(s, now)
        self._call_standby(s, now)
        if now - self.last_replan >= self.cfg.route_replan_s:
            self._replan_tail(now, s)
        else:
            self._extend(now, s)
        self._maybe_pursue(s, now)
        self._drive(s, now, cmds)
        if self.extend_failed and (self.belief.detection_xy is None or self.pursuit == "done"):
            # no searchable belief left for either type (UAVs may still be flying)
            P = self.planner
            m = P.searchable_mass(self.belief)
            self._done = not any((F @ P.seen(m, p) > 0).any() for F, p in ((P.F_road, P.p_ugv), (P.F_air, P.p_air)))
        else:
            self._done = False

    def _track(self, s: RobotState, now: float) -> None:
        """Advance ``cur`` along the FollowPath being driven (its node count left)."""
        cur = self.target - len(s.path_remaining) if self.sent_path is not None else self.cur
        if cur > self.cur:
            self.cur, self.arrived = cur, now

    def _handle_events(self, obs: RobotObservation, now: float) -> None:
        here = self.stops.get(self.cur, [])
        for ev in obs.events:
            if ev.other_id != self.rid or not here:
                continue
            stop = here[0]
            if ev.robot_id != stop["uav"]:
                continue
            if ev.type in (EventType.DOCK_STARTED, EventType.UNDOCK_STARTED):
                stop["started"] = now
            elif ev.type == EventType.DOCK_ABORTED or ev.type == EventType.UNDOCK_ABORTED:
                stop.pop("started", None)
            elif ev.type == EventType.DOCKED and stop["kind"] == LAND or \
                    ev.type == EventType.UNDOCKED and stop["kind"] == TAKEOFF:
                here.pop(0)
                if stop["kind"] == LAND:  # the booking is spent; its take-off stop remains
                    self.bookings.pop(stop["uav"], None)
                    self.swap_until = self.swap_end[stop["uav"]] = now + self.setup.docking.swap_duration_s
                self.revision += 1  # bookings stay valid: the pin only moves back

    def _handle_messages(self, obs: RobotObservation, s: RobotState, now: float, cmds: list) -> None:
        for msg in obs.messages(SYNC):
            self._fuse(msg.payload)
            cmds.append(Transmit(SYNC, {"mass": self.belief.mass.astype(np.float32), "sighting": self.belief.detection_xy,
                                        "ack": msg.sent_time}, to=(msg.sender,)))
        for msg in sorted(obs.messages(BOOK), key=lambda m: m.sender):
            self._accept(msg.payload, s, now)

    def _give_up_late(self, s: RobotState, now: float) -> None:
        """At a landing stop whose UAV is long overdue, or whose landing started long ago and never
        ended (the UAV ran out of energy on the way down, and a depleted UAV sends nothing): drop
        its booking and drive on."""
        here = self.stops.get(self.cur, [])
        if not here or here[0]["kind"] != LAND:
            return
        st = here[0]
        due = st["started"] + self.setup.docking.dock_duration_s if "started" in st else st["ready"] + st["slack"]
        if now > due + self.cfg.max_late_s:
            self._drop(st["uav"])

    def _maybe_pursue(self, s: RobotState, now: float) -> None:
        """Route the UGV past a sighting: right away if the detour keeps every UAV still to land
        within its slack, otherwise straight after the last promised stop (so no booking moves),
        re-checking the detour at every node it gets to while a stop still comes before it."""
        if self.pursuit == "planned" and self.cur >= self.pursuit_end:
            self.pursuit = "done"  # drove past it without confirming
        if not self.cfg.pursue_sightings or self.belief.detection_xy is None or self.pursuit == "done":
            return
        if self.pursuit == "planned" and self.pursuit_check is None:
            return
        xy = self.belief.detection_xy
        at_node = s.node is not None and s.node == self.nodes[self.cur]
        head = min(self.cur if at_node else self.cur + 1, len(self.nodes) - 1)  # the node it is at, or driving to
        free = not self.stops.get(head)
        if self.pursuit == "planned":  # planned after the pin: is the detour now worth it?
            if not any(st and head <= i < self.pursuit_end for i, st in self.stops.items()):
                self.pursuit_check = None  # no stop left before it: it is on its way there
                return
            if (head, free, self.version) == self.pursuit_check:
                return
            self.pursuit_check = (head, free, self.version)
            if not (free and self._detour_now(head, xy)):
                return
            del self.nodes[max(self._pin(), self.cur + 1, self.target, self.pursuit_end) + 1:]  # the pursuit after the pin
            self.pursuit_check = None
            self._extend(now, s)
        elif free and self._detour_now(head, xy):
            self.pursuit_check = None
        else:
            keep = max(self._pin(), self.cur + 1, self.target)
            del self.nodes[keep + 1:]
            self.nodes += [int(n) for n in pursuit_path(self.road, xy, self.nodes[keep], self._path)[1:]]
            self.pursuit_end = len(self.nodes) - 1
            self.pursuit_check = ()  # check the detour from the next step on
        self.pursuit = "planned"
        self.version += 1
        self.revision += 1

    def _detour_now(self, at: int, xy) -> bool:
        """Drive from route index ``at`` via the sighting to the next stop, if no UAV still to
        land waits past its slack."""
        path = [int(n) for n in pursuit_path(self.road, xy, self.nodes[at], self._path)]
        nxt = min((i for i, st in self.stops.items() if st and i > at), default=None)
        if nxt is None:  # no stops ahead: the detour replaces the route
            self.nodes = self.nodes[:at] + path
        else:
            back, _ = self._path(path[-1], self.nodes[nxt])
            detour = path + [int(n) for n in back[1:]]
            delay = (self.road.path_length(detour) - self.road.path_length(self.nodes[at:nxt + 1])) \
                / self.spec.speed
            slack = min((st["slack"] for sts in self.stops.values() for st in sts if st["kind"] == LAND),
                        default=np.inf)
            if delay > slack:
                return False
            shift = len(detour) - 1 - (nxt - at)
            self.nodes = self.nodes[:at] + detour + self.nodes[nxt + 1:]
            self.stops = {(i + shift if i >= nxt else i): st for i, st in self.stops.items()}
            for b in self.bookings.values():
                b["land_idx"] += shift if b["land_idx"] >= nxt else 0
                if b["off_idx"] is not None and b["off_idx"] >= nxt:
                    b["off_idx"] += shift
                b["t_land"] += max(delay, 0.0)
        self.pursuit_end = at + len(path) - 1
        return True

    def _drive(self, s: RobotState, now: float, cmds: list) -> None:
        """Hold at a stop until it is done; otherwise drive to the next stop (or the route end)."""
        here = self.stops.get(self.cur, [])
        at_node = s.node is not None and s.node == self.nodes[self.cur]
        if at_node and now < self._swap_hold():  # stand still until the battery swap is done
            if s.path_remaining or s.moving:
                cmds.append(Hold())
                self.sent_path = None
            return
        if here and at_node:
            if s.path_remaining or s.moving:
                cmds.append(Hold())
                self.sent_path = None
            self.target = self.cur
            return
        nxt = min((i for i, st in self.stops.items() if st and i > self.cur), default=len(self.nodes) - 1)
        if nxt <= self.cur:
            return
        path = tuple(self.nodes[self.cur:nxt + 1] if at_node else self.nodes[self.cur + 1:nxt + 1])
        if (nxt, path[-1], self.version) != self.sent_path:
            cmds.append(FollowPath(path))
            self.target, self.sent_path = nxt, (nxt, path[-1], self.version)

    # --------------------------------------------------------------- status
    def _status_extra(self, s: RobotState, now: float) -> dict:
        here = self.stops.get(self.cur, [])
        stationary = not s.path_remaining and not s.moving
        ready = here and (here[0]["kind"] == LAND or now >= here[0]["ready"] - 1e-9)  # a take-off waits for the battery swap
        holding = {"uav": here[0]["uav"], "kind": here[0]["kind"]} \
            if ready and stationary and s.node == self.nodes[self.cur] else None
        out = {"stationary": stationary, "holding": holding}
        if self.revision != self.sent_revision or now - self.sent_at >= self.cfg.route_resend_s:
            out["plan"] = self._plan_payload(s, now)
            self.sent_revision, self.sent_at = self.revision, now
        return out


class CtpRouteUgv(TimedRouteUgv):
    """``ugv_timed_route`` planning its way on the road with the Canadian Traveller's Problem policy."""

    name = "ugv_ctp_route"
    config_cls = UgvCtpRouteConfig
