"""UAV policy ``uav_far_scout``: a UAV flying high that searches the wider area round the UGV.

Speaks :mod:`.dock_booking` and books its own sorties. It is the counterpart of
``uav_near_scout``, which flies low and waits on the dock until the UGV calls it: give this one
the larger sensor range (``FleetConfig.uav_sensor_ranges``). With a standby UAV on the team, it
gets every landing slot.

It plans its sorties as ``uav_greedy_tour`` does (the landing slot, then a greedy chain tour
scoring each next UAV-grid node by ``belief / (distance + offset)``), on the belief weighed to
complement the UGV and the near scout. Each weighing is off at 0; only the first is on by
default (on 30 large maps with road mismatches it cut the expected time to find by 13.5 ± 8.0
min, while the other two showed no gain and, added to it, halved it):

The UGV's route here is the stretch it drives during a full sortie from take-off.

Near the route (``near_route_m``)
    Belief close to the UGV's route counts less (``near_route_weight`` on it, rising to
    full at ``near_route_m``): the UGV sweeps it, and the near scout searches there while this
    UAV is on the dock.
Road reconnaissance (``recon_weight``)
    Once the team has found a blocked node, each road node on the UGV's route ahead (from
    ``recon_from_s`` to ``recon_to_s`` after take-off) that no robot has looked at is worth
    ``recon_weight`` average footprints of belief: seeing it first finds blockages and missing
    roads before the UGV gets there. With ``recon_window: landing``, the window is set by when the
    UGV hears of a find instead: out of radio range on its sortie, this UAV reports what it
    finds when it lands. A road node then counts only if the UGV reaches the last junction before
    it on its route (where it can still turn off) from ``recon_react_s`` to ``recon_horizon_s``
    after the landing (0: one full sortie, after which the next sortie can look).
Confirmation delay (``confirm_scale_s``)
    Only the UGV can end the mission, so belief counts ``τ / (τ + d)``: ``d`` is the UGV's drive
    from its route to within sensor range of the cell (to confirm a sighting there), ``τ`` is
    ``confirm_scale_s``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from marsupial_search.core import MissionSetup

from .dock_booking import BookingUavConfig
from .uav_greedy_tour import GreedyTourUav


@dataclass
class FarScoutUavConfig(BookingUavConfig):
    near_route_m: float = 300.0  # belief this close to the UGV's route counts less (0: off) ...
    near_route_weight: float = 0.25  # ... this much on the route, rising to full at near_route_m
    recon_weight: float = 0.0  # once a blocked node is found: an unseen road node ahead is worth this many footprints (0: off) ...
    recon_from_s: float = 120.0  # ... on the UGV's route from this long after take-off ...
    recon_to_s: float = 1800.0  # ... to this long
    recon_window: str = "takeoff"  # takeoff: the window above | landing: by when the UGV hears of a find (at landing) ...
    recon_react_s: float = 60.0  # landing: ... the UGV reaches the junction before the node this long after the landing ...
    recon_horizon_s: float = 0.0  # ... to this long (0: one full sortie)
    confirm_scale_s: float = 0.0  # τ: belief counts τ / (τ + the UGV's drive to confirm a sighting there) (0: off)


class FarScoutUav(GreedyTourUav):
    name = "uav_far_scout"
    config_cls = FarScoutUavConfig

    def reset(self, robot_id: str, setup: MissionSetup) -> None:
        super().reset(robot_id, setup)
        X, Y = setup.grid.cell_centers()
        self.cells = np.stack([X.ravel(), Y.ravel()], axis=1)
        e = self.spec.energy  # a full sortie: how long the UGV drives while it is away
        self.sortie_s = max(self.planner.energy_budget(e.capacity), 0.0) / (e.idle_power + e.energy_per_m * self.spec.speed)
        if self.cfg.recon_window not in ("takeoff", "landing"):
            raise ValueError(f"recon_window must be takeoff or landing, not {self.cfg.recon_window!r}")

    def _tour_mass(self, m: np.ndarray, t_start: float, t_land: Optional[float] = None) -> np.ndarray:
        cfg, p = self.cfg, self.plan
        nodes, arr = np.asarray(p["nodes"], int), np.asarray(p["arrive"], float)
        ahead = nodes[(arr >= t_start) & (arr <= t_start + self.sortie_s)]  # the UGV's route while it is away
        if not len(ahead) or not (cfg.near_route_m > 0 or cfg.recon_weight > 0 or cfg.confirm_scale_s > 0):
            return m
        unit = None
        if cfg.recon_weight > 0:  # an average footprint's belief, before any weighing
            P = self.planner
            v = P.F_air @ P.seen(m, P.p_air)
            unit = float(v[v > 0].mean()) if (v > 0).any() else 0.0
        w = np.ones(m.shape[1])
        pos = self.road.positions
        if cfg.near_route_m > 0:
            d, _ = cKDTree(pos[ahead]).query(self.cells)
            w *= cfg.near_route_weight + (1.0 - cfg.near_route_weight) * np.clip(d / cfg.near_route_m, 0.0, 1.0)
        if cfg.confirm_scale_s > 0:
            w *= cfg.confirm_scale_s / (cfg.confirm_scale_s + self._confirm_s(ahead))
        m = m * w
        if unit:
            m = self._recon(m, nodes, arr, t_start, unit, t_land)
        return m

    def _decide_times(self, nodes: np.ndarray, arr: np.ndarray) -> np.ndarray:
        """Per route index: when the UGV reaches the last junction before it on the route (three
        or more neighbours not known blocked), where it can still turn off; with none before it on
        the route, the node before it."""
        csr, blocked = self.road.csr, self.road.blocked
        rows = np.repeat(np.arange(csr.shape[0]), np.diff(csr.indptr))
        deg = np.bincount(rows[~blocked[csr.indices]], minlength=csr.shape[0])
        idx = np.arange(len(nodes))
        last = np.maximum.accumulate(np.where(deg[nodes] >= 3, idx, -1))  # the last junction at or before i
        before = np.concatenate([[-1], last[:-1]])  # ... strictly before i
        return np.where(before >= 0, arr[np.maximum(before, 0)], arr[np.maximum(idx - 1, 0)])

    def _confirm_s(self, ahead: np.ndarray) -> np.ndarray:
        """Per cell: the UGV's drive from its route ``ahead`` to within sensor range of it."""
        road, ugv = self.road, self.setup.robots[self.ugv]
        ok = np.flatnonzero(~road.blocked & np.isfinite(road.positions).all(1))
        dr = dijkstra(road.csr, directed=False, indices=np.unique(ahead), min_only=True)  # from the route, by road
        de, j = cKDTree(road.positions[ok]).query(self.cells)  # the cell's nearest road node, as the crow flies
        return (dr[ok[j]] + np.maximum(de - ugv.sensor.range_m, 0.0)) / ugv.speed

    def _recon(self, m: np.ndarray, nodes: np.ndarray, arr: np.ndarray, t_start: float, unit: float,
               t_land: Optional[float] = None) -> np.ndarray:
        """Add ``recon_weight · unit`` of belief, seen from the air only, at each road node on the
        route ahead that no robot has looked at (its cell's belief is still at the prior)."""
        cfg = self.cfg
        if not self.roads.blocked:
            return m
        if cfg.recon_window == "landing" and t_land is not None:
            d = self._decide_times(nodes, arr)
            horizon = cfg.recon_horizon_s if cfg.recon_horizon_s > 0 else self.sortie_s
            win = (arr >= t_start) & (d >= t_land + cfg.recon_react_s) & (d <= t_land + horizon)
        else:
            win = (arr >= t_start + cfg.recon_from_s) & (arr <= t_start + cfg.recon_to_s)
        cand = np.unique(nodes[win])
        cand = cand[~self.road_visited[cand]]
        if not len(cand):
            return m
        g = self.setup.grid
        xy = self.road.positions[cand]
        cols = np.clip(((xy[:, 0] - g.origin[0]) // g.resolution).astype(int), 0, g.cols - 1)
        rows = np.clip(((xy[:, 1] - g.origin[1]) // g.resolution).astype(int), 0, g.rows - 1)
        b = self.belief
        init = b.initial_mass.sum(0)[rows, cols]
        unseen = b.mass.sum(0)[rows, cols] > 0.9 * init  # nothing has looked at it
        if not unseen.any():
            return m
        m = m.copy()
        np.add.at(m[self.planner.own_layers[0]], rows[unseen] * g.cols + cols[unseen], cfg.recon_weight * unit)
        return m
