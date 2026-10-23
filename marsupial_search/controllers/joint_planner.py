"""Centralised oracle ``joint_planner``: plans the UGV route and both UAVs' sorties jointly.

One controller sees every robot's state and sensor data instantly and ignores the radio.
Unlike the per-robot policies, it optimises one plan for the whole team over a horizon of
several sorties, and keeps improving it as the belief changes. It plans on the belief,
never on the true target.

Plan (:class:`JointPlan`)
    * the UGV's road route, from where it is now;
    * the dock schedule: the dock visits in order, each a landing and a take-off at route
      indices (one UAV docked at a time, so a visit's take-off comes before the next landing);
    * a tour of UAV-grid nodes for each sortie. A sortie starts at a take-off, or now for a
      UAV in the air, and ends at that UAV's next landing; the last one may stay open (after its
      tour the UAV must still be able to reach the UGV and land once the dock can be free; a
      later re-plan books the landing).

Timeline
    Built in one pass over the dock events: the UGV waits at a landing until the UAV is
    there, and at a take-off until that UAV's battery swap is done (it drives on during the
    swap, or waits it out at the landing without ``DockingSpec.swap_while_moving``). A plan is feasible if every sortie lands (or could land) with
    ``uav_reserve_fraction`` left.

Objective
    ``J = Σ_k p_k · (H − t_k)``: the belief ``p_k`` swept by each look (a robot at a node), in
    time order across all three robots, so shared-layer belief counts once, for whoever gets
    there first. Over the horizon ``H`` this is the reduction in expected time to find.
    A UAV can only sight the target and the UGV must drive there to confirm, so a UAV look
    counts at ``t + λ · c``, with ``c`` the UGV's drive from where it is at ``t``
    (``λ = sighting_delay_weight``). Looks after the UGV would run flat count nothing.

Search
    Large-neighbourhood search with simulated annealing, from the best of the previous plan
    (advanced to now) and a few greedy plans (timed route, the dock handed over in turn, chain
    tours; with a UAV docked, also launching it at once, with either UAV landing first). Moves edit tours (insert, remove,
    replace, reverse, rebuild), shift dock events along the route, add or remove a dock
    visit, and re-route the UGV between stops. An infeasible plan is repaired by trimming
    tours, or rejected. The number of iterations is fixed, so runs are reproducible.

Re-planning
    Every ``replan_s``, after every dock event and abort, on a new sighting, and when a robot
    finds a blocked node on the route. Dock manoeuvres in progress are kept; everything else
    may change, including the tour and landing of a UAV in flight. After a sighting the route
    starts with the pursuit path, and dock events go after it unless a UAV could not otherwise
    land. It plans on the road map as the robots have found it so far (roads missing from it
    are used from the next re-plan), never on the true one.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from marsupial_search.core import (
    BeliefMap,
    Command,
    Dock,
    EventType,
    FollowPath,
    GoToPoint,
    GraphKnowledge,
    Hold,
    MissionController,
    MissionSetup,
    Observation,
    RobotMode,
    RobotState,
    RobotType,
    Undock,
)

from .common import bypass, dist, pursuit_path
from .dock_booking import LAND, TAKEOFF, DockBookingConfig, slot_spacing
from .sortie_planner import SortiePlanner


@dataclass
class JointPlannerConfig(DockBookingConfig):
    horizon_s: float = 2700.0  # the plan is scored this far ahead
    replan_s: float = 120.0  # re-plan at least this often (and after every dock event)
    iterations: int = 1000  # search iterations per re-plan
    anneal_t0: float = 0.002  # initial annealing temperature, as a fraction of the starting plan's value
    sighting_delay_weight: float = 1.0  # λ: a UAV look counts at t + λ · (the UGV's drive to confirm)
    insert_radius_m: float = 400.0  # tour insertions pick nodes this close to the tour
    seed: int = 0


@dataclass
class Visit:
    """A stay on the dock: landing and take-off at route indices, and the tour flown after it."""
    uav: str
    land: Optional[int]  # None: on the dock when the plan starts
    off: Optional[int]  # None: still on the dock when the plan ends (the last visit only)
    tour: list[int] = field(default_factory=list)  # UAV-grid nodes flown after the take-off


@dataclass
class JointPlan:
    route: list[int]  # road nodes from the UGV's position
    visits: list[Visit]  # dock order
    first_tours: dict[str, list[int]]  # UAVs in the air when the plan starts: tour before their first landing

    def copy(self) -> "JointPlan":
        return JointPlan(list(self.route), [Visit(v.uav, v.land, v.off, list(v.tour)) for v in self.visits],
                         {u: list(t) for u, t in self.first_tours.items()})

    def events(self) -> list[tuple[int, str, int]]:
        """(route index, kind, visit index) of every dock event, in order."""
        out = []
        for k, v in enumerate(self.visits):
            if v.land is not None:
                out.append((v.land, LAND, k))
            if v.off is not None:
                out.append((v.off, TAKEOFF, k))
        return out

    def tour(self, key: tuple) -> list[int]:
        """A sortie's tour: ``("air", uav)`` for a UAV in the air at the start, ``("visit", k)`` after visit k."""
        return self.first_tours[key[1]] if key[0] == "air" else self.visits[key[1]].tour


@dataclass
class _Context:
    """What a re-plan starts from."""
    now: float
    h_end: float
    prefix: list[int]  # the route starts with these nodes (UGV position, pursuit path)
    t0: float  # the UGV is at prefix[0] (or may leave it) at this time
    lock: int  # no dock event before this route index
    min_lock: int  # the lock without a pursuit
    ugv_energy: float
    docked: dict[str, tuple[float, float]]  # UAV on the dock: (free to take off from, energy then)
    air: dict[str, tuple[float, tuple, float]]  # UAV in the air: (free from, position, energy)
    m0: np.ndarray  # searchable belief, (layers, cells)
    candidates: np.ndarray  # road nodes not yet visited
    air_value: dict[str, np.ndarray] = field(default_factory=dict)  # per UAV: belief each UAV-grid node would sweep (for insertions)


@dataclass
class _Sortie:
    uav: str
    key: tuple  # JointPlan.tour key
    tour: list[int]
    start: tuple
    t_start: float
    e_start: float
    land_visit: Optional[int]  # None: open at the end of the plan
    land_idx: int
    length: float = 0.0
    node_t: np.ndarray = field(default_factory=lambda: np.zeros(0))  # when it reaches each tour node
    t_arrive: float = 0.0  # at the landing node
    t_land: float = 0.0  # landing starts
    e_end: float = 0.0  # after landing
    ok: bool = True


@dataclass
class _Eval:
    value: float
    ok: bool
    arrive: np.ndarray  # UGV arrival at each route index
    sorties: list[_Sortie]
    t_dead: float = np.inf  # the UGV runs flat


class JointPlannerController(MissionController):
    name = "joint_planner"

    def __init__(self, config: Optional[JointPlannerConfig] = None):
        self.cfg = config or JointPlannerConfig()

    # ------------------------------------------------------------------ setup
    def reset(self, setup: MissionSetup) -> None:
        ugvs = setup.robots_of(RobotType.UGV)
        if len(ugvs) != 1:
            raise ValueError(f"{self.name} supports exactly one UGV")
        self.setup, self.ugv = setup, ugvs[0]
        self.uavs = setup.robots_of(RobotType.UAV)
        self.ugv_spec = setup.robots[self.ugv]
        self.roads = GraphKnowledge(setup.graphs[RobotType.UGV])  # the road map and the changes found to it
        self.road, self.air_graph = self.roads.graph, setup.graphs[RobotType.UAV]
        self.belief = BeliefMap(setup.grid, setup.prior, setup.layer_fractions)
        self.road_visited = np.zeros(self.road.n_nodes, dtype=bool)
        uav_spec = setup.robots[self.uavs[0]] if self.uavs else self.ugv_spec
        self.P = SortiePlanner(setup, self.ugv_spec, uav_spec, self.cfg)
        self.PU = {u: self.P.for_uav(setup.robots[u]) for u in self.uavs}  # each UAV's own sensor footprints
        self.n_cells = setup.grid.rows * setup.grid.cols
        self.near_road = self.road.nearest_nodes(self.air_graph.positions)  # road node nearest each UAV-grid node
        a = self.air_graph.positions
        self.grid_step = float(np.min(np.hypot(*(a[1:] - a[0]).T))) if len(a) > 1 else 1.0
        self._dist_cache: dict[int, np.ndarray] = {}
        self.plan = JointPlan([], [], {})
        self.events: list[tuple[int, str, int]] = []
        self.ev_ptr = 0
        self.cur = 0  # route index the UGV is at, or last passed
        self.sent: Optional[tuple] = None
        self.version = 0
        self.wp: dict[str, list[int]] = {}  # UAV in the air: tour nodes still to fly
        self.waiting: set[str] = set()
        self.last_plan = -np.inf
        self.replan_now = True
        self.sighting = None
        self.pursuit_goal: Optional[int] = None
        self.pursuit_done = False
        self.held: dict[str, float] = {}  # UAV -> when its dock manoeuvre ends
        self._done = False
        self.replans = 0

    def is_done(self) -> bool:
        return self._done

    # ------------------------------------------------------------------- step
    def step(self, obs: Observation) -> dict[str, list[Command]]:
        S, now = obs.robots, obs.time
        for upd in obs.belief_updates:
            self.belief.apply(upd)
        for upd in obs.map_updates:
            self.roads.add(upd.blocked, upd.new_nodes, upd.new_edges)
        if self.road is not self.roads.graph:
            self._map_changed()
        g = S[self.ugv]
        self.road_visited[list(g.nodes_reached) + ([g.node] if g.node is not None else [])] = True
        if self.sent is not None:
            self.cur = max(self.cur, self.sent[0] - len(g.path_remaining))
        self._handle_events(obs)
        if self.belief.detection_xy is not None and self.sighting is None:
            self.sighting, self.replan_now = self.belief.detection_xy, True
        if self.pursuit_goal is not None and not self.pursuit_done and g.node == self.pursuit_goal:
            self.pursuit_done, self.replan_now = True, True  # drove past it without confirming
        if g.mode == RobotMode.DEPLETED:
            return {}
        if self.replan_now or now - self.last_plan >= self.cfg.replan_s:
            self._replan(S, now)
        cmds: dict[str, list[Command]] = {}
        swapping = not self.setup.docking.swap_while_moving and any(
            S[u].mode == RobotMode.DOCKED and S[u].mode_time_remaining > 0 for u in self.uavs)
        self._drive(g, cmds, hold=swapping)
        for u in self.uavs:
            self._fly(u, S, cmds)
        return cmds

    def _map_changed(self) -> None:
        """Plan on the road map as now known: at once if the route runs into a blocked node, else
        (new roads) from the next re-plan."""
        self.road = self.roads.graph
        for P in {id(P): P for P in [self.P, *self.PU.values()]}.values():
            P.set_road(self.road)
        self.road_visited = np.pad(self.road_visited, (0, self.road.n_nodes - len(self.road_visited)))
        self.near_road = self.road.nearest_nodes(self.air_graph.positions)
        self._dist_cache.clear()
        ahead = self.plan.route[self.cur:]
        self.replan_now |= not all(self.road.has_edge(a, b) for a, b in zip(ahead, ahead[1:]))

    def _next_event(self) -> Optional[tuple[int, str, int]]:
        return self.events[self.ev_ptr] if self.ev_ptr < len(self.events) else None

    def _handle_events(self, obs: Observation) -> None:
        d = self.setup.docking
        for ev in obs.events:
            if ev.type in (EventType.DOCKED, EventType.UNDOCKED):
                nxt = self._next_event()
                kind = LAND if ev.type == EventType.DOCKED else TAKEOFF
                if nxt is not None and nxt[1] == kind and self.plan.visits[nxt[2]].uav == ev.robot_id:
                    self.ev_ptr += 1
                self.held.pop(ev.robot_id, None)
                self.replan_now = True
            elif ev.type in (EventType.DOCK_STARTED, EventType.UNDOCK_STARTED):
                self.held[ev.robot_id] = ev.time + (d.dock_duration_s if ev.type == EventType.DOCK_STARTED
                                                    else d.undock_duration_s)
            elif ev.type in (EventType.DOCK_ABORTED, EventType.UNDOCK_ABORTED, EventType.DEPLETED,
                             EventType.COMMAND_REJECTED, EventType.PATH_BLOCKED):
                self.held.pop(ev.robot_id, None)
                self.replan_now = True

    # ---------------------------------------------------------------- execute
    def _drive(self, g: RobotState, cmds: dict, hold: bool = False) -> None:
        """Drive to the next dock event's route index (or the route end) and hold there (and
        wherever it is while ``hold``)."""
        route = self.plan.route
        nxt = self._next_event()
        target = nxt[0] if nxt is not None else len(route) - 1
        at_node = g.node is not None and g.node == route[self.cur]
        if (self.cur >= target and at_node) or self.held or hold:
            if g.path_remaining or g.moving:
                cmds[self.ugv] = [Hold()]
                self.sent = None
            return
        if target <= self.cur:
            return
        path = tuple(route[self.cur:target + 1] if at_node else route[self.cur + 1:target + 1])
        if (target, self.version) != self.sent:
            cmds[self.ugv] = [FollowPath(path)]
            self.sent = (target, self.version)

    def _fly(self, u: str, S: dict[str, RobotState], cmds: dict) -> None:
        s, g = S[u], S[self.ugv]
        if s.mode not in (RobotMode.ACTIVE, RobotMode.DOCKED):
            return
        nxt = self._next_event()
        route = self.plan.route
        ugv_here = nxt is not None and self.plan.visits[nxt[2]].uav == u and self.cur == nxt[0] \
            and g.node == route[nxt[0]] and not g.moving and not g.path_remaining
        if s.mode == RobotMode.DOCKED:
            if ugv_here and nxt[1] == TAKEOFF and s.mode_time_remaining <= 0:  # battery swap done
                cmds[u] = [Undock()]
            return
        wp = self.wp.setdefault(u, [])
        while wp and dist(s.position, self.air_graph.positions[wp[0]]) < 0.5:
            wp.pop(0)
        if wp:
            target = tuple(self.air_graph.positions[wp[0]])
        else:
            land = next((v.land for v in self.plan.visits if v.uav == u and v.land is not None), None)
            if land is not None:
                target = tuple(self.road.positions[route[land]])
                if dist(s.position, target) < 0.5:
                    if ugv_here and nxt[1] == LAND and g.docked_uav is None:
                        cmds[u] = [Dock(self.ugv)]
                    elif s.goal_point is not None or s.moving:
                        cmds[u] = [Hold()]
                    return
            else:  # open sortie, tour done: wait over the UGV, and re-plan to book a landing
                if u not in self.waiting:
                    self.waiting.add(u)
                    self.replan_now = True
                target = tuple(g.position)
                if dist(s.position, target) <= 0.5 * self.setup.docking.dock_tolerance_m:
                    if s.goal_point is not None or s.moving:
                        cmds[u] = [Hold()]
                    return
        if s.goal_point is None or dist(s.goal_point, target) > 0.5:
            cmds[u] = [GoToPoint(target)]

    # ----------------------------------------------------------------- re-plan
    def _replan(self, S: dict[str, RobotState], now: float) -> None:
        self.replan_now, self.last_plan = False, now
        self.replans += 1
        ctx = self._context(S, now)
        P = self.P
        if not ((P.F_road @ ctx.m0[P.ugv_layers].sum(0) > 0).any()
                or (P.F_air @ ctx.m0[P.air_layers].sum(0) > 0).any()):
            self._done = self.sighting is None or self.pursuit_done  # no searchable belief left
        starts = self._starts(ctx)
        warm = self._advance(ctx)
        if warm is not None:
            self._ensure_length(ctx, warm)
            starts.append(warm)
        self._install(self._search(ctx, starts))

    def _context(self, S: dict[str, RobotState], now: float) -> _Context:
        g, v = S[self.ugv], self.ugv_spec.speed
        if g.node is not None:
            prefix, t0, lock = [g.node], now, 0
        else:  # between two nodes: passed prefix[0], heading for prefix[1]
            nxt = g.path_remaining[0] if g.path_remaining else self._edge_end(g)
            prefix = [g.last_node, nxt]
            t0, lock = now - dist(g.position, self.road.positions[g.last_node]) / v, 1
        docked, air, hold = {}, {}, now
        for u in self.uavs:
            s = S[u]
            e = self.setup.robots[u].energy
            swap = self.setup.docking.swap_duration_s
            if s.mode == RobotMode.DOCKED:  # mode_time_remaining: what is left of its battery swap
                docked[u] = (now + s.mode_time_remaining, e.capacity if s.mode_time_remaining > 0 else s.energy)
            elif s.mode in (RobotMode.DOCKING, RobotMode.UNDOCKING):
                done = max(self.held.get(u, now + s.mode_time_remaining), now)
                idle = e.idle_power
                if s.mode == RobotMode.DOCKING:
                    docked[u] = (done + swap, e.capacity)
                else:
                    air[u] = (done, tuple(g.position), s.energy - idle * (done - now))
                hold = max(hold, done)
            elif s.mode == RobotMode.ACTIVE:
                air[u] = (now, tuple(s.position), s.energy)
        if not self.setup.docking.swap_while_moving:
            hold = max([hold] + [t for t, _ in docked.values()])
        t0, min_lock = max(t0, hold), lock
        self.pursuit_goal = None
        if self.sighting is not None and self.cfg.pursue_sightings and not self.pursuit_done:
            path = pursuit_path(self.road, self.sighting, prefix[-1])
            prefix = prefix + [int(n) for n in path[1:]]
            lock, self.pursuit_goal = len(prefix) - 1, prefix[-1]
        return _Context(now, now + self.cfg.horizon_s, prefix, t0, lock, min_lock, g.energy, docked, air,
                        self.P.searchable_mass(self.belief), ~self.road_visited)

    def _edge_end(self, g: RobotState) -> int:
        """The node a UGV stopped between two nodes faces: the neighbour of its last node it is beside."""
        a = self.road.positions[g.last_node]
        return min((n for n, _ in self.road.neighbors(g.last_node)),
                   key=lambda n: abs(dist(a, g.position) + dist(g.position, self.road.positions[n])
                                     - dist(a, self.road.positions[n])))

    def _install(self, plan: JointPlan) -> None:
        self.plan, self.events, self.ev_ptr = plan, plan.events(), 0
        self.cur, self.sent = 0, None
        self.version += 1
        self.wp = {u: list(t) for u, t in plan.first_tours.items()}
        self.waiting: set[str] = set()  # UAVs whose open sortie's tour is done

    def _advance(self, ctx: _Context) -> Optional[JointPlan]:
        """The current plan from now: the route from the UGV's position (round any node found
        blocked since), without the dock events and tour nodes already done. None if the robots
        are not where it has them, or a dock event was on a blocked node."""
        old, cur = self.plan, self.cur
        if not old.route or old.route[cur:cur + len(ctx.prefix)] != ctx.prefix:
            return None
        done = set(self.events[:self.ev_ptr])
        visits = []
        first = {u: list(self.wp.get(u, [])) for u in ctx.air}
        for k, v in enumerate(old.visits):
            land_done = v.land is None or (v.land, LAND, k) in done
            off_done = v.off is not None and (v.off, TAKEOFF, k) in done
            if off_done and v.uav in ctx.air and not self.wp.get(v.uav):  # took off since: now its current sortie
                first[v.uav] = list(v.tour)
            if land_done and off_done:
                continue
            visits.append(Visit(v.uav, None if land_done else v.land - cur, None if v.off is None else v.off - cur,
                                list(v.tour)))
        plan = JointPlan(list(old.route[cur:]), visits, first)
        if self.roads:
            plan.route, index = bypass(self.road, plan.route, [i for i, _, _ in plan.events()])
            for v in plan.visits:
                v.land, v.off = (None if i is None else int(index[i]) for i in (v.land, v.off))
                if -1 in (v.land, v.off):
                    return None
        return plan if self._consistent(ctx, plan) else None

    def _consistent(self, ctx: _Context, plan: JointPlan) -> bool:
        """The plan's dock state matches the robots, its dock events are in route order, and its
        route can still be driven on the road map as now known."""
        if self.roads and not all(a == b or self.road.has_edge(a, b) for a, b in zip(plan.route, plan.route[1:])):
            return False  # it ends where bypass could not go on
        docked = [v.uav for v in plan.visits if v.land is None]
        if docked != list(ctx.docked) or (docked and plan.visits[0].land is not None):
            return False
        if set(plan.first_tours) != set(ctx.air):
            return False
        last = 0
        for idx, _, _ in plan.events():
            if idx < max(last, ctx.min_lock) or idx >= len(plan.route):
                return False
            last = idx
        return all(v.off is not None for v in plan.visits[:-1])

    # ---------------------------------------------------------------- timeline
    def _evaluate(self, ctx: _Context, plan: JointPlan, value: bool = True) -> _Eval:
        cfg, d = self.cfg, self.setup.docking
        R = np.asarray(plan.route, dtype=int)
        pos = self.road.positions[R]
        legs = np.hypot(*np.diff(pos, axis=0).T) if len(R) > 1 else np.zeros(0)
        cum = np.concatenate([[0.0], np.cumsum(legs) / self.ugv_spec.speed])
        arrive = ctx.t0 + cum
        sorties: list[_Sortie] = []
        dep = {u: (t, p, e, ("air", u)) for u, (t, p, e) in ctx.air.items()}
        docked = dict(ctx.docked)
        cur, t_free, ok = 0, ctx.t0, True
        for idx, kind, k in plan.events():
            if idx < cur or idx >= len(R):
                return _Eval(-np.inf, False, arrive, sorties)
            if idx > cur:
                t_free += cum[idx] - cum[cur]
                cur = idx
            u = plan.visits[k].uav
            spec = self.setup.robots[u]
            e = spec.energy
            if kind == LAND:
                if u not in dep:
                    return _Eval(-np.inf, False, arrive, sorties)
                s = self._sortie(plan, u, dep.pop(u), k, idx, pos[idx])
                s.t_land = max(t_free, s.t_arrive)
                s.e_end = s.e_start - e.travel_cost(s.length, spec.speed) \
                    - e.idle_power * (s.t_land - s.t_arrive + d.dock_duration_s)
                s.ok = s.e_end >= cfg.uav_reserve_fraction * e.capacity
                ok &= s.ok
                sorties.append(s)
                t_free = s.t_land + d.dock_duration_s + cfg.stop_margin_s
                docked[u] = (t_free + d.swap_duration_s, e.capacity)
                if not d.swap_while_moving:  # the UGV waits out the swap
                    t_free += d.swap_duration_s
            else:
                if u not in docked:
                    return _Eval(-np.inf, False, arrive, sorties)
                t_ready, e_off = docked.pop(u)
                start = max(t_free, t_ready)
                t_free = start + d.undock_duration_s + cfg.stop_margin_s
                dep[u] = (start + d.undock_duration_s, tuple(pos[idx]), e_off - e.idle_power * d.undock_duration_s,
                          ("visit", k))
            arrive[idx + 1:] = t_free + cum[idx + 1:] - cum[idx]
        # Open sorties: after the tour they must still reach the UGV and land, once the dock is free:
        # after the last take-off, or once a UAV still docked has been swapped and could leave.
        dock_free = t_free
        for t_ready, _ in docked.values():
            dock_free = max(dock_free, t_ready + d.undock_duration_s + cfg.stop_margin_s)
        opened = []
        for u, state in dep.items():
            s = self._sortie(plan, u, state, None, len(R) - 1, pos[-1])
            t_end = s.node_t[-1] if len(s.node_t) else s.t_start
            j = int(np.clip(np.searchsorted(arrive, t_end), 0, len(R) - 1))
            opened.append(self._sortie(plan, u, state, None, j, pos[j]))
        for s in sorted(opened, key=lambda s: s.t_arrive):
            spec = self.setup.robots[s.uav]
            e = spec.energy
            s.t_land = max(arrive[s.land_idx], s.t_arrive, dock_free) + cfg.hover_margin_s
            s.e_end = s.e_start - e.travel_cost(s.length, spec.speed) \
                - e.idle_power * (s.t_land - s.t_arrive + d.dock_duration_s)
            # A tour that ends before the next re-plan would leave the UAV waiting with no landing booked.
            t_end = s.node_t[-1] if len(s.node_t) else s.t_start
            s.ok = s.e_end >= cfg.uav_reserve_fraction * e.capacity and t_end >= ctx.now + cfg.replan_s
            ok &= s.ok
            sorties.append(s)
            dock_free = s.t_land + d.dock_duration_s + d.swap_duration_s + d.undock_duration_s + cfg.stop_margin_s
        # When the UGV runs flat.
        eu = self.ugv_spec.energy
        e_left = ctx.ugv_energy - eu.idle_power * (arrive - ctx.now) \
            - eu.energy_per_m * np.concatenate([[0.0], np.cumsum(legs)])
        flat = np.flatnonzero(e_left <= 0)
        t_dead = float(arrive[flat[0]]) if len(flat) else np.inf
        v = self._value(ctx, R, arrive, sorties, t_dead) if ok and value else (0.0 if ok else -np.inf)
        return _Eval(v, ok, arrive, sorties, t_dead)

    def _sortie(self, plan: JointPlan, u: str, state: tuple, land_visit: Optional[int], land_idx: int,
                land_xy) -> _Sortie:
        t, p, e, key = state
        tour = plan.tour(key)
        pts = np.vstack([np.asarray(p, float)[None], self.air_graph.positions[tour].reshape(-1, 2),
                         np.asarray(land_xy, float)[None]])
        seg = np.hypot(*np.diff(pts, axis=0).T)
        v = self.setup.robots[u].speed
        s = _Sortie(u, key, tour, tuple(p), t, e, land_visit, land_idx, length=float(seg.sum()))
        s.node_t = t + np.cumsum(seg[:-1]) / v
        s.t_arrive = t + s.length / v
        return s

    def _value(self, ctx: _Context, R: np.ndarray, arrive: np.ndarray, sorties: list[_Sortie], t_dead: float) -> float:
        """Σ swept belief × (horizon end − time found), looks applied in time order."""
        P, h_end = self.P, min(ctx.h_end, t_dead)
        keep = arrive <= h_end
        looks = [(P.F_road, R[keep], arrive[keep], arrive[keep], P.ugv_layers, P.p_ugv)]
        lam = self.cfg.sighting_delay_weight
        for s in sorties:
            if not s.tour:
                continue
            nodes, t = np.asarray(s.tour, dtype=int), s.node_t
            found = t + lam * self._confirm_delay(R, arrive, nodes, t) if lam > 0 else t
            keep = found <= h_end
            Pu = self.PU[s.uav]
            looks.append((Pu.F_air, nodes[keep], t[keep], found[keep], Pu.air_layers, Pu.p_air))
        keys, order_t, w, p = [], [], [], []
        for F, nodes, t, found, layers, pd in looks:
            if not len(nodes):
                continue
            rep, cells = _gather(F, nodes)
            for l in layers:
                keys.append(l * self.n_cells + cells)
                order_t.append(t[rep])
                w.append(h_end - found[rep])
                p.append(np.full(len(cells), min(pd, 1 - 1e-12)))
        if not keys:
            return 0.0
        keys, order_t, w, p = map(np.concatenate, (keys, order_t, w, p))
        o = np.lexsort((order_t, keys))
        keys, w, p = keys[o], w[o], p[o]
        lq = np.log1p(-p)
        excl = np.cumsum(lq) - lq
        first = np.r_[True, keys[1:] != keys[:-1]]
        base = excl[first][np.cumsum(first) - 1]
        gain = ctx.m0.reshape(-1)[keys] * np.exp(excl - base) * p
        return float((gain * w).sum())

    def _confirm_delay(self, R, arrive, nodes, t) -> np.ndarray:
        """The UGV's drive from where it is at time ``t`` to the road node nearest each UAV look."""
        src = R[np.clip(np.searchsorted(arrive, t, side="right") - 1, 0, len(R) - 1)]
        dst = self.near_road[nodes]
        out = np.empty(len(nodes))
        for n in np.unique(src):
            d = self._dist_cache.get(int(n))
            if d is None:
                d = self._dist_cache[int(n)] = self.road.distances_from(int(n))[0]
            sel = src == n
            out[sel] = d[dst[sel]]
        return out / self.ugv_spec.speed

    # ------------------------------------------------------------ construction
    def _starts(self, ctx: _Context) -> list[JointPlan]:
        """Greedy plans to start the search from. With one UAV docked and one in the air, also
        launch the docked one at once and try either UAV landing first."""
        plans = [self._construct(ctx)]
        if ctx.docked and ctx.air:
            d, a = next(iter(ctx.docked)), next(iter(ctx.air))
            plans += [self._construct(ctx, eager=True, first=d), self._construct(ctx, eager=True, first=a)]
        return plans

    def _construct(self, ctx: _Context, eager: bool = False, first: Optional[str] = None) -> JointPlan:
        """A greedy plan: the timed route, and the dock handed over in turn. The UAV in the air
        that must land first lands as late as its energy allows; the docked UAV takes off just
        early enough that its own flight lasts until the lander's battery is swapped. ``eager``:
        the UAV docked now takes off as soon as it can; ``first``: this UAV lands first, early
        enough that its battery is swapped before the other must land."""
        plan = JointPlan(list(ctx.prefix), [Visit(u, None, None) for u in ctx.docked],
                         {u: [] for u in ctx.air})
        self._ensure_length(ctx, plan)
        if not ctx.docked and not ctx.air:
            return plan
        if eager and plan.visits and not self._place_takeoff(ctx, plan, 0):
            return plan
        stay = slot_spacing(self.setup, dataclasses.replace(self.cfg, slot_s=0.0), 10 ** 6)  # one stay on the dock
        for _ in range(4 * len(self.uavs) + 2):
            ev = self._evaluate(ctx, plan, value=False)
            docked = plan.visits[-1] if plan.visits and plan.visits[-1].off is None else None
            open_ = [s for s in ev.sorties if s.land_visit is None]
            if not open_:
                if docked is None or not self._place_takeoff(ctx, plan, len(plan.visits) - 1):
                    break  # nobody left to fly, or not swapped before the horizon
                continue
            urgent = min(open_, key=self._deadline)
            s, target = urgent, self._deadline(urgent)
            forced = next((o for o in open_ if o.uav == first), None)
            if forced is not None and forced is not urgent:  # land first, swap, leave before the other lands
                s, target = forced, min(self._deadline(forced), self._deadline(urgent) - stay)
            first = None
            if s.t_start >= ctx.h_end:
                break
            last = plan.visits[-1] if plan.visits else None
            if docked is not None:  # it lands once the docked UAV has left
                lo = max(ctx.lock, docked.land if docked.land is not None else 0)
            elif last is not None:  # the dock is free from the last take-off
                lo = max(ctx.lock, last.off + (s.key == ("visit", len(plan.visits) - 1)))
            else:
                lo = ctx.lock
            pick = self._pick_landing(ctx, plan, ev.arrive, s, lo, target)
            if pick is None:
                break
            idx, tour = pick
            if docked is not None:
                self._hand_over(ctx, plan, ev, len(plan.visits) - 1, idx)
            plan.tour(s.key)[:] = tour
            plan.visits.append(Visit(s.uav, idx, None))
            k = len(plan.visits) - 1
            if not self._sortie_ok(ctx, plan, k):
                self._land_sooner(ctx, plan, k, lo if docked is None else max(lo, plan.visits[k - 1].off))
            ev = self._evaluate(ctx, plan, value=False)
            s = next(s for s in ev.sorties if s.land_visit == k)
            if s.ok and s.t_land > s.t_arrive + 1.0:  # it would hover until the dock is free: search instead
                before = list(plan.tour(s.key))
                plan.tour(s.key)[:] = self._chain(ctx, plan, ev.arrive, s, plan.visits[k].land, s.t_land)
                if not self._sortie_ok(ctx, plan, k):
                    plan.tour(s.key)[:] = before
            if s.t_land >= ctx.h_end:
                break
        self._ensure_length(ctx, plan)
        ev = self._evaluate(ctx, plan, value=False)
        for s in ev.sorties:  # open sorties: a chain tour to where the UGV is after most of a flight
            if s.land_visit is None and not s.tour:
                j = int(np.clip(np.searchsorted(ev.arrive, s.t_start + 0.8 * (self._deadline(s) - s.t_start)), 0,
                                len(plan.route) - 1))
                plan.tour(s.key)[:] = self._chain(ctx, plan, ev.arrive, s, j)
                self._repair(ctx, plan, self._evaluate(ctx, plan, value=False))
        return plan

    def _deadline(self, s: _Sortie) -> float:
        """The latest a sortie can land, searching all the way (``energy_budget`` keeps the reserve)."""
        spec = self.setup.robots[s.uav]
        e = spec.energy
        return s.t_start + max(self.PU[s.uav].energy_budget(s.e_start), 0.0) / (e.idle_power + e.energy_per_m * spec.speed)

    def _ready(self, ctx: _Context, plan: JointPlan, ev: _Eval, k: int) -> Optional[tuple[float, float]]:
        """(time, energy) from which the UAV of visit ``k`` can take off: once its battery is swapped."""
        v, d = plan.visits[k], self.setup.docking
        if v.land is None:
            return ctx.docked[v.uav]
        s = next((s for s in ev.sorties if s.land_visit == k), None)
        if s is None:
            return None
        return (s.t_land + d.dock_duration_s + self.cfg.stop_margin_s + d.swap_duration_s,
                self.setup.robots[v.uav].energy.capacity)

    def _hand_over(self, ctx: _Context, plan: JointPlan, ev: _Eval, k: int, land_idx: int) -> None:
        """Take-off of the docked UAV of visit ``k`` before a landing at ``land_idx``: as late as lets
        its flight last until the lander's battery is swapped, and not before its own is."""
        v, d, cfg = plan.visits[k], self.setup.docking, self.cfg
        ready = self._ready(ctx, plan, ev, k)
        if ready is None:
            return
        t_ready, e_ready = ready
        spec = self.setup.robots[v.uav]
        e = spec.energy
        stay = d.dock_duration_s + d.swap_duration_s + d.undock_duration_s + 2 * cfg.stop_margin_s
        flight = max(self.PU[v.uav].energy_budget(e_ready - e.idle_power * d.undock_duration_s), 0.0) \
            / (e.idle_power + e.energy_per_m * spec.speed)
        t_land = ev.arrive[land_idx]
        t_off = min(max(t_ready, t_land + stay - flight), t_land)
        lo = max(ctx.lock, v.land if v.land is not None else 0)
        later = [i for i in range(lo, land_idx + 1) if ev.arrive[i] >= t_off]
        v.off = later[0] if later else land_idx

    def _place_takeoff(self, ctx: _Context, plan: JointPlan, k: int, hi: Optional[int] = None) -> bool:
        """Put visit ``k``'s take-off at the first route index the UGV reaches once that UAV's
        battery is swapped (no later than ``hi``). False if that is beyond the horizon."""
        v = plan.visits[k]
        ev = self._evaluate(ctx, plan, value=False)
        ready = self._ready(ctx, plan, ev, k)
        if ready is None or ready[0] >= ctx.h_end:
            return False
        lo = max(ctx.lock, v.land if v.land is not None else 0)
        hi = len(plan.route) - 1 if hi is None else hi
        later = [i for i in range(lo, hi + 1) if ev.arrive[i] >= ready[0]]
        v.off = later[0] if later else hi
        return True

    def _land_sooner(self, ctx: _Context, plan: JointPlan, k: int, lo: int) -> bool:
        """Move visit ``k``'s landing to the latest index from ``lo`` on where its sortie (with a
        fresh chain tour) lands with its reserve. False (plan unchanged) if there is none."""
        v = plan.visits[k]
        key = next(s.key for s in self._evaluate(ctx, plan, value=False).sorties if s.land_visit == k)
        old_land, old_tour = v.land, list(plan.tour(key))
        plan.tour(key)[:] = []
        for i in range(old_land, lo - 1, -1):
            v.land = i
            if self._sortie_ok(ctx, plan, k):
                ev = self._evaluate(ctx, plan, value=False)
                s = next(s for s in ev.sorties if s.land_visit == k)
                plan.tour(key)[:] = self._chain(ctx, plan, ev.arrive, s, i, s.t_land)
                self._repair(ctx, plan, self._evaluate(ctx, plan, value=False))
                if self._sortie_ok(ctx, plan, k):
                    return True
                plan.tour(key)[:] = []
        v.land = old_land
        plan.tour(key)[:] = old_tour
        return False

    def _sortie_ok(self, ctx: _Context, plan: JointPlan, k: int) -> bool:
        """Does the sortie that lands at visit ``k`` land with its reserve?"""
        ev = self._evaluate(ctx, plan, value=False)
        return next((s.ok for s in ev.sorties if s.land_visit == k), False)

    def _pick_landing(self, ctx: _Context, plan: JointPlan, arr: np.ndarray, s: _Sortie, lo: int, target: float):
        """Landing index and tour for sortie ``s``: the first index at or after ``target`` it can
        reach, else the latest earlier one (as ``uav_greedy_tour`` picks)."""
        P, spec = self.PU[s.uav], self.setup.robots[s.uav]
        e, v = spec.energy, spec.speed
        budget = P.energy_budget(s.e_start)
        idx = np.arange(max(lo, 0), len(plan.route))
        idx = idx[arr[idx] > s.t_start]
        if not len(idx):
            return None
        flight = max(budget, 0.0) / (e.idle_power + e.energy_per_m * v)
        target = min(target, s.t_start + flight)

        def reach(i):
            tau = arr[i] - s.t_start
            return min(P.max_tour_length(budget, tau), v * tau) >= dist(s.start, self.road.positions[plan.route[i]])

        later = [i for i in idx if arr[i] >= target]
        order = ([later[0]] if later else []) + [i for i in idx[::-1] if not later or i < later[0]]
        pick = next((int(i) for i in order if reach(i)), int(idx[0]))
        return pick, self._chain(ctx, plan, arr, s, pick)

    def _chain(self, ctx: _Context, plan: JointPlan, arr: np.ndarray, s: _Sortie, land_idx: int,
               t_meet: Optional[float] = None) -> list[int]:
        """A chain tour for sortie ``s`` to ``land_idx``, ending by ``t_meet`` (default: when the UGV
        gets there), skipping the plan's other tours."""
        P, spec = self.PU[s.uav], self.setup.robots[s.uav]
        tau = (arr[land_idx] if t_meet is None else t_meet) - s.t_start
        if tau <= 0:
            return []
        L_max = min(P.max_tour_length(P.energy_budget(s.e_start), tau), spec.speed * tau)
        if L_max < 0:
            return []
        own = plan.tour(s.key)
        claim = np.full(P.air.n_nodes, np.inf)
        for t in [v.tour for v in plan.visits] + list(plan.first_tours.values()):
            if t is not own and t:
                claim[t] = -np.inf
        future = arr >= s.t_start
        t_ugv = P.ugv_sweep_times(np.asarray(plan.route)[future], arr[future] - s.t_start)
        tour = P.chain_tour(ctx.m0, t_ugv, s.start, tuple(self.road.positions[plan.route[land_idx]]), L_max, tau,
                            0.0, claim, s.t_start)
        return list(tour[1]) if tour is not None else []

    def _ensure_length(self, ctx: _Context, plan: JointPlan) -> None:
        """Extend the route greedily until the UGV reaches its end well after the horizon."""
        want = ctx.h_end + self.cfg.horizon_s / 3
        need = want - self._evaluate(ctx, plan, value=False).arrive[-1]
        if need <= 0:
            return
        P = self.P
        m = ctx.m0.copy()
        for n in plan.route:
            P.sweep(m, P.F_road, n, P.ugv_layers, P.p_ugv)
        cand = ctx.candidates.copy()
        cand[plan.route] = False
        src = plan.route[-1]
        new, _ = P.ugv_route(m, src, tuple(self.road.positions[src]), cand, need, np.full(P.air.n_nodes, np.inf))
        plan.route += [int(n) for n in new[1:]]
        short = want - self._evaluate(ctx, plan, value=False).arrive[-1]
        nbs = self.road.neighbors(plan.route[-1])
        if short > 0 and nbs:  # nothing left to drive for: there and back, so the UAVs can still land
            nb, length = nbs[0]
            end = plan.route[-1]
            plan.route += [int(nb), end] * int(np.ceil(short / (2 * length / self.ugv_spec.speed)))

    # ------------------------------------------------------------------ search
    def _search(self, ctx: _Context, starts: list[JointPlan]) -> JointPlan:
        tried = [self._repair(ctx, p, self._evaluate(ctx, p)) for p in starts]
        found = [(p, e) for p, e in tried if e.ok and all(idx >= ctx.lock for idx, _, _ in p.events())]
        if not found and ctx.lock > ctx.min_lock:  # no UAV can wait for the pursuit: dock on its way
            ctx.lock = ctx.min_lock
            tried += [self._repair(ctx, p, self._evaluate(ctx, p)) for p in self._starts(ctx)]
            found = [(p, e) for p, e in tried if e.ok]
        if not found:  # no plan lands everyone with the reserve: keep the emptiest UAV as full as possible
            return max(tried, key=lambda pe: min((s.e_end for s in pe[1].sorties), default=np.inf))[0]
        cur, cur_ev = max(found, key=lambda pe: pe[1].value)
        best, best_ev = cur, cur_ev
        n = self.cfg.iterations
        rng = np.random.default_rng([self.cfg.seed, int(ctx.now)])
        moves = self._moves()
        weights = np.array([w for w, _ in moves])
        weights = weights / weights.sum()
        t0 = self.cfg.anneal_t0 * max(abs(cur_ev.value), 1e-9)
        for it in range(n):
            move = moves[rng.choice(len(moves), p=weights)][1]
            cand = cur.copy()
            if not move(ctx, cand, cur_ev, rng):
                continue
            cand, ev = self._repair(ctx, cand, self._evaluate(ctx, cand))
            if not ev.ok:
                continue
            temp = t0 * (1.0 - it / n)
            delta = ev.value - cur_ev.value
            if delta >= 0 or (temp > 0 and rng.random() < np.exp(delta / temp)):
                cur, cur_ev = cand, ev
                if ev.value > best_ev.value:
                    best, best_ev = cand, ev
        return best

    def _repair(self, ctx: _Context, plan: JointPlan, ev: _Eval) -> tuple[JointPlan, _Eval]:
        """Trim the tours of infeasible sorties, the node with the longest detour first."""
        for _ in range(12):
            bad = next((s for s in ev.sorties if not s.ok and s.tour), None)
            if ev.ok or bad is None:
                break
            tour = plan.tour(bad.key)
            pts = np.vstack([np.asarray(bad.start, float)[None], self.air_graph.positions[tour],
                             self.road.positions[plan.route[bad.land_idx]][None]])
            detour = np.hypot(*(pts[1:-1] - pts[:-2]).T) + np.hypot(*(pts[2:] - pts[1:-1]).T) \
                - np.hypot(*(pts[2:] - pts[:-2]).T)
            tour.pop(int(np.argmax(detour)))
            ev = self._evaluate(ctx, plan)
        return plan, ev

    # ------------------------------------------------------------------- moves
    def _moves(self) -> list[tuple[float, Callable]]:
        return [(4.0, self._m_insert), (2.0, self._m_remove), (2.0, self._m_replace), (1.0, self._m_reverse),
                (3.0, self._m_shift), (1.0, self._m_detour), (0.5, self._m_shortcut), (0.5, self._m_add_visit),
                (0.5, self._m_remove_visit), (0.3, self._m_rebuild)]

    def _pick_sortie(self, ev: _Eval, rng, min_len: int = 0) -> Optional[_Sortie]:
        sorties = [s for s in ev.sorties if len(s.tour) >= min_len]
        return sorties[rng.integers(len(sorties))] if sorties else None

    def _m_insert(self, ctx, plan, ev, rng) -> bool:
        s = self._pick_sortie(ev, rng)
        if s is None:
            return False
        tour = plan.tour(s.key)
        pts = np.vstack([np.asarray(s.start, float)[None], self.air_graph.positions[tour].reshape(-1, 2),
                         self.road.positions[plan.route[s.land_idx]][None]])
        cand = np.asarray(self.air_graph.nodes_within(pts[rng.integers(len(pts))], self.cfg.insert_radius_m), int)
        if s.uav not in ctx.air_value:
            Pu = self.PU[s.uav]
            ctx.air_value[s.uav] = np.asarray(Pu.F_air @ ctx.m0[Pu.air_layers].sum(0)).ravel()
        val = ctx.air_value[s.uav][cand] if len(cand) else np.zeros(0)
        if val.sum() <= 0:
            return False
        node = int(cand[rng.choice(len(cand), p=val / val.sum())])
        if node in tour:
            return False
        q = self.air_graph.positions[node]
        cost = np.hypot(*(pts[:-1] - q).T) + np.hypot(*(pts[1:] - q).T) - np.hypot(*np.diff(pts, axis=0).T)
        tour.insert(int(np.argmin(cost)), node)
        return True

    def _m_remove(self, ctx, plan, ev, rng) -> bool:
        s = self._pick_sortie(ev, rng, 1)
        if s is None:
            return False
        tour = plan.tour(s.key)
        tour.pop(int(rng.integers(len(tour))))
        return True

    def _m_replace(self, ctx, plan, ev, rng) -> bool:
        s = self._pick_sortie(ev, rng, 1)
        if s is None:
            return False
        tour = plan.tour(s.key)
        i = int(rng.integers(len(tour)))
        near = np.asarray(self.air_graph.nodes_within(self.air_graph.positions[tour[i]], 2.2 * self.grid_step), int)
        near = near[~np.isin(near, tour)]
        if not len(near):
            return False
        tour[i] = int(near[rng.integers(len(near))])
        return True

    def _m_reverse(self, ctx, plan, ev, rng) -> bool:
        s = self._pick_sortie(ev, rng, 3)
        if s is None:
            return False
        tour = plan.tour(s.key)
        i, j = sorted(rng.choice(len(tour), 2, replace=False))
        tour[i:j + 1] = tour[i:j + 1][::-1]
        return True

    def _m_shift(self, ctx, plan, ev, rng) -> bool:
        """Move a landing or take-off along the route, within the dock schedule."""
        events = plan.events()
        if not events:
            return False
        idx, kind, k = events[rng.integers(len(events))]
        vs, n = plan.visits, len(plan.route)
        if kind == LAND:
            lo = max(ctx.lock, vs[k - 1].off if k > 0 else 0)
            hi = vs[k].off if vs[k].off is not None else n - 1
        else:
            lo = max(ctx.lock, vs[k].land if vs[k].land is not None else 0)
            hi = vs[k + 1].land if k + 1 < len(vs) else n - 1
        new = int(np.clip(idx + rng.choice([-1, 1]) * rng.integers(1, 9), lo, hi))
        if new == idx:
            return False
        if kind == LAND:
            vs[k].land = new
        else:
            vs[k].off = new
        return True

    def _free_span(self, ctx, plan, rng, max_len: int = 40) -> Optional[tuple[int, int]]:
        """Route indices i < j with no dock event strictly between them, after the fixed prefix."""
        n, lo = len(plan.route), max(ctx.lock, len(ctx.prefix) - 1)
        if n - 1 <= lo:
            return None
        i = int(rng.integers(lo, n - 1))
        nxt = min([idx for idx, _, _ in plan.events() if idx > i] + [n - 1])
        return i, int(rng.integers(i + 1, min(nxt, i + max_len) + 1))

    def _replace_span(self, plan: JointPlan, i: int, j: int, seg: list[int]) -> None:
        shift = len(seg) - (j - i + 1)
        plan.route[i:j + 1] = seg
        for v in plan.visits:
            if v.land is not None and v.land >= j:
                v.land += shift
            if v.off is not None and v.off >= j:
                v.off += shift

    def _m_detour(self, ctx, plan, ev, rng) -> bool:
        """Re-route the UGV between two stops through a road node with belief left."""
        span = self._free_span(ctx, plan, rng)
        if span is None:
            return False
        i, j = span
        near = np.asarray(self.road.nodes_within(self.road.positions[plan.route[(i + j) // 2]], 500.0), dtype=int)
        if not len(near):
            return False
        P = self.P
        val = np.asarray(P.F_road[near] @ ctx.m0[P.ugv_layers].sum(0)).ravel() * ctx.candidates[near]
        if val.sum() <= 0:
            return False
        w = int(near[rng.choice(len(near), p=val / val.sum())])
        a, _ = self.road.shortest_path(plan.route[i], w)
        b, _ = self.road.shortest_path(w, plan.route[j])
        if not a or not b:
            return False
        self._replace_span(plan, i, j, a + b[1:])
        self._ensure_length(ctx, plan)
        return True

    def _m_shortcut(self, ctx, plan, ev, rng) -> bool:
        span = self._free_span(ctx, plan, rng)
        if span is None:
            return False
        i, j = span
        seg, _ = self.road.shortest_path(plan.route[i], plan.route[j])
        if not seg or len(seg) >= j - i + 1:
            return False
        self._replace_span(plan, i, j, seg)
        self._ensure_length(ctx, plan)
        return True

    def _m_add_visit(self, ctx, plan, ev, rng) -> bool:
        """Split a sortie with an extra landing and take-off in a gap in the dock schedule."""
        s = self._pick_sortie(ev, rng, 2)
        if s is None:
            return False
        vs = plan.visits
        dep = s.key[1] if s.key[0] == "visit" else -1
        land = s.land_visit if s.land_visit is not None else len(vs)
        p = int(rng.integers(dep + 1, land + 1))
        if (p > 0 and vs[p - 1].off is None) or (p < len(vs) and vs[p].land is None):
            return False  # the dock is taken
        lo = max(ctx.lock, vs[p - 1].off if p > 0 else 0)
        hi = vs[p].land if p < len(vs) else len(plan.route) - 1
        if hi < lo:
            return False
        a, b = sorted(int(x) for x in rng.integers(lo, hi + 1, size=2))
        tour = plan.tour(s.key)
        r = int(rng.integers(1, len(tour)))
        rest = tour[r:]
        del tour[r:]
        vs.insert(p, Visit(s.uav, a, b, rest))
        return True

    def _m_remove_visit(self, ctx, plan, ev, rng) -> bool:
        """Drop a dock visit: the UAV flies on, its two sorties joined."""
        arriving = [s for s in ev.sorties if s.land_visit is not None]
        if not arriving:
            return False
        s = arriving[rng.integers(len(arriving))]
        k = s.land_visit
        plan.tour(s.key).extend(plan.visits[k].tour)
        del plan.visits[k]
        return True

    def _m_rebuild(self, ctx, plan, ev, rng) -> bool:
        """A fresh chain tour for one sortie, on the current timeline."""
        s = self._pick_sortie(ev, rng)
        if s is None:
            return False
        plan.tour(s.key)[:] = self._chain(ctx, plan, ev.arrive, s, s.land_idx, s.t_land)
        return True


def _gather(F, nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(look index, cell) for every cell in the footprints of ``nodes``."""
    starts = F.indptr[nodes]
    counts = F.indptr[nodes + 1] - starts
    rep = np.repeat(np.arange(len(nodes)), counts)
    offs = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    return rep, F.indices[starts[rep] + offs]
