"""UAV policy ``uav_greedy_tour``: book the next landing slot, fly a greedy chain tour to it.

Speaks :mod:`.dock_booking`; this module only chooses the sortie.

Landing (the split)
    After the pin (the last promised take-off), and ``slot_s`` after the last booked landing,
    so the UAVs land in turn and each gets the next stretch of the route. If it can't reach
    that landing node in time, the UAV takes the latest earlier one it can. Energy caps how
    late it can land.

Tour
    :meth:`~.sortie_planner.SortiePlanner.chain_tour` from the take-off point to the landing
    node, arriving by the UGV's arrival time there, skipping the tours other UAVs have
    booked. Shared-layer belief counts only where the UAV is ahead of the UGV's route.

The UAV assumes every UAV on the team has its specs (``slot_spacing``).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from marsupial_search.core import MissionSetup, RobotState

from .common import dist
from .dock_booking import BookingUav, Sortie
from .sortie_planner import SortiePlanner


class GreedyTourUav(BookingUav):
    name = "uav_greedy_tour"

    def reset(self, robot_id: str, setup: MissionSetup) -> None:
        super().reset(robot_id, setup)
        self.planner = SortiePlanner(setup, setup.robots[self.ugv], self.spec, self.cfg)

    def _map_changed(self, s: RobotState, now: float) -> None:
        self.planner.set_road(self.road)

    def plan_sortie(self, start, t_start: float, energy: float, now: float) -> Optional[Sortie]:
        p, P, cfg = self.plan, self.planner, self.cfg
        e, v = self.spec.energy, self.spec.speed
        road = self.road
        nodes, arr, base = np.asarray(p["nodes"], int), np.asarray(p["arrive"], float), p["cur"]
        first = p["pin"] - base + 1  # land after the last promised take-off
        idx = np.arange(first, len(nodes))
        idx = idx[arr[idx] > t_start]
        if not len(idx):
            return None
        budget = P.energy_budget(energy)
        flight = max(budget, 0.0) / (e.idle_power + e.energy_per_m * v)
        last = p["last_land"] if p["last_land"] is not None else now
        target = min(last + self._slot(p), t_start + flight)

        def reach(i):
            tau = arr[i] - t_start
            L = min(P.max_tour_length(budget, tau), v * tau)
            return L if L >= dist(start, road.positions[nodes[i]]) else -1.0

        later = [i for i in idx if arr[i] >= target]
        order = ([later[0]] if later else []) + [i for i in idx[::-1] if not later or i < later[0]]
        pick = next((i for i in order if reach(i) >= 0), int(idx[0]))  # none reachable: the earliest
        tau = arr[pick] - t_start
        L_max = reach(pick)
        k = tuple(road.positions[nodes[pick]])
        claim = np.full(P.air.n_nodes, np.inf)
        for b in p["queue"]:
            if b["uav"] != self.rid and b["t_land"] > now:
                claim[list(b["tour"])] = -np.inf
        future = arr >= t_start
        t_ugv = P.ugv_sweep_times(nodes[future], arr[future] - t_start)
        tour = P.chain_tour(P.searchable_mass(self.belief), t_ugv, tuple(start), k, L_max, tau, 0.0, claim, t_start) \
            if L_max >= 0 else None
        nodes_t, length = (list(tour[1]), tour[2]) if tour is not None else ([], dist(start, k))
        dock_s = self.setup.docking.dock_duration_s
        e_land = energy - e.energy_per_m * length - e.idle_power * (tau + dock_s)
        slack = (e_land - cfg.uav_reserve_fraction * e.capacity) / e.idle_power
        return Sortie(int(base + pick), float(arr[pick]), float(e_land), float(max(slack, 0.0)),
                      tuple(int(n) for n in nodes_t))
