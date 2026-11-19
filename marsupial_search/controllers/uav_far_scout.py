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

    With ``recon_mode: gated`` (after the gated rewards of Oerlemans, Shi and Popović, arXiv
    2606.12306, which beat additive ones), each road node in that window counts from the start,
    by how much seeing it first is worth: ``recon_weight · H · V · G`` footprints, where
    ``H = p / block_prior`` (``p`` the chance it is blocked, from its own ``ctp.BlockageModel``:
    0 once any robot has seen it), ``V`` the chance its pass makes the node out
    (``MissionSetup.road_detection``), and ``G = recon_floor + (1 - recon_floor) · max(L, B)``
    what a blockage there would cost: ``L`` 1 if it would send the UGV further round than the UAVs
    can wait at a landing (``rendezvous_slack_s + hover_margin_s`` of driving), or cut it off, as
    ``clear`` in ``ugv_ctp_route``'s plan; ``B`` the belief the UGV's route sweeps in the
    ``recon_belief_s`` after it, over the 90th percentile of that among the window's nodes, at
    most 1 (a blockage before much belief delays the search most).

    In both modes the tour scores this apart from the target's belief (``chain_tour``'s
    ``extra``): each node's worth counts by ``V``, the chance a look makes out the road, not the
    chance it sees a target there, and a pass leaves ``1 - V`` of it.
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

from marsupial_search.core import MissionSetup, RobotObservation, RobotState

from .ctp import BlockageModel, clear_probabilities
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
    recon_mode: str = "flat"  # flat: as above | gated: from the start, by the chance a node is blocked, seen, and what its blockage would cost
    recon_floor: float = 0.1  # gated: the weight of a node whose blockage breaks no landing and holds up little belief
    recon_belief_s: float = 600.0  # gated: the belief the UGV's route sweeps this long after a node counts toward its weight (0: off)
    block_prior: float = 0.05  # gated: prior share of road nodes blocked ...
    block_prior_weight: float = 20.0  # ... worth this many road nodes seen
    max_cutoff_fraction: float = 0.0  # gated: blockages cut off at most this share of the mapped road nodes (as the UGV's)
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
        cfg = self.cfg
        if cfg.recon_window not in ("takeoff", "landing"):
            raise ValueError(f"recon_window must be takeoff or landing, not {cfg.recon_window!r}")
        if cfg.recon_mode not in ("flat", "gated"):
            raise ValueError(f"recon_mode must be flat or gated, not {cfg.recon_mode!r}")
        self.blockage = BlockageModel(self.roads, cfg.block_prior, cfg.block_prior_weight, cfg.max_cutoff_fraction) \
            if cfg.recon_weight > 0 and cfg.recon_mode == "gated" else None
        self.detour_cache: dict = {}  # (prev, node, next) -> would its blockage send the UGV the long way round?
        self.detour_road = None  # the road map detour_cache is for

    def _look(self, xy, radius: float, hidden=()) -> None:
        """A robot at ``xy`` saw the road nodes within ``radius``, but those it could not make out."""
        if self.blockage is not None and xy is not None:
            self.blockage.look(self.road, xy, radius - 1.0, hidden)  # (a margin for where it was when it looked)

    def _step(self, s: RobotState, now: float, obs: RobotObservation, cmds: list) -> None:
        self._look(s.position, self.spec.sensor.range_m, self.hidden)
        super()._step(s, now, obs, cmds)

    def _ingest(self, sender: str, p: dict, t: float) -> None:
        super()._ingest(sender, p, t)
        if sender in self.setup.robots:  # what that robot has seen, from where its status says it was
            self._look(p.get("pos"), self.setup.robots[sender].sensor.range_m, p.get("hidden", ()))

    def _ahead(self, t_start: float) -> np.ndarray:
        """The road nodes the UGV's route passes during a full sortie from ``t_start``."""
        nodes, arr = np.asarray(self.plan["nodes"], int), np.asarray(self.plan["arrive"], float)
        return nodes[(arr >= t_start) & (arr <= t_start + self.sortie_s)]

    def _tour_mass(self, m: np.ndarray, t_start: float, t_land: Optional[float] = None) -> np.ndarray:
        cfg = self.cfg
        if not (cfg.near_route_m > 0 or cfg.confirm_scale_s > 0):
            return m
        ahead = self._ahead(t_start)
        if not len(ahead):
            return m
        w = np.ones(m.shape[1])
        if cfg.near_route_m > 0:
            d, _ = cKDTree(self.road.positions[ahead]).query(self.cells)
            w *= cfg.near_route_weight + (1.0 - cfg.near_route_weight) * np.clip(d / cfg.near_route_m, 0.0, 1.0)
        if cfg.confirm_scale_s > 0:
            w *= cfg.confirm_scale_s / (cfg.confirm_scale_s + self._confirm_s(ahead))
        return m * w

    def _tour_extra(self, m: np.ndarray, t_start: float, t_land: Optional[float] = None):
        """Road reconnaissance (module docstring), as ``chain_tour``'s ``extra``: a row of value per
        cell, seen with this UAV's chance of making out the road there
        (``MissionSetup.road_detection``) rather than the target's ``p_air``."""
        cfg = self.cfg
        if cfg.recon_weight <= 0 or not len(self._ahead(t_start)):
            return None
        P = self.planner
        v = P.F_air @ P.seen(m, P.p_air)  # an average footprint's belief, before any weighing
        unit = float(v[v > 0].mean()) if (v > 0).any() else 0.0
        if not unit:
            return None
        nodes, arr = np.asarray(self.plan["nodes"], int), np.asarray(self.plan["arrive"], float)
        found = self._recon_gated(m, nodes, arr, t_start, t_land) if self.blockage is not None \
            else self._recon(nodes, arr, t_start, t_land)
        if found is None:
            return None
        cell, worth = found
        g = self.setup.grid
        see = self.setup.road_detection.get(self.rid)
        p_see = np.ones(g.rows * g.cols) if see is None else np.asarray(see, float).ravel()
        mass = np.zeros(g.rows * g.cols)
        np.add.at(mass, cell, cfg.recon_weight * unit * worth)
        return mass, p_see

    def _cells(self, xy: np.ndarray):
        """Grid (rows, cols) of the points ``xy``."""
        g = self.setup.grid
        cols = np.clip(((xy[:, 0] - g.origin[0]) // g.resolution).astype(int), 0, g.cols - 1)
        rows = np.clip(((xy[:, 1] - g.origin[1]) // g.resolution).astype(int), 0, g.rows - 1)
        return rows, cols

    def _window(self, nodes: np.ndarray, arr: np.ndarray, t_start: float, t_land: Optional[float]) -> np.ndarray:
        """Per route index: is the node in the reconnaissance window (``recon_window``)?"""
        cfg = self.cfg
        if cfg.recon_window == "landing" and t_land is not None:
            d = self._decide_times(nodes, arr)
            horizon = cfg.recon_horizon_s if cfg.recon_horizon_s > 0 else self.sortie_s
            return (arr >= t_start) & (d >= t_land + cfg.recon_react_s) & (d <= t_land + horizon)
        return (arr >= t_start + cfg.recon_from_s) & (arr <= t_start + cfg.recon_to_s)

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

    def _recon(self, nodes: np.ndarray, arr: np.ndarray, t_start: float, t_land: Optional[float] = None):
        """The cells (and weight 1 each) of the road nodes in the window that no robot has looked
        at (their cell's belief is still at the prior), once a blocked node is found; None if
        there are none."""
        if not self.roads.blocked:
            return None
        cand = np.unique(nodes[self._window(nodes, arr, t_start, t_land)])
        cand = cand[~self.road_visited[cand]]
        if not len(cand):
            return None
        rows, cols = self._cells(self.road.positions[cand])
        b = self.belief
        init = b.initial_mass.sum(0)[rows, cols]
        unseen = b.mass.sum(0)[rows, cols] > 0.9 * init  # nothing has looked at it
        if not unseen.any():
            return None
        cell = rows[unseen] * self.setup.grid.cols + cols[unseen]
        return cell, np.ones(len(cell))

    def _recon_gated(self, m0: np.ndarray, nodes: np.ndarray, arr: np.ndarray, t_start: float,
                     t_land: Optional[float] = None):
        """The cells and weights ``H · G`` of the road nodes in the window (module docstring; ``V``
        is the chance of seeing them, which ``chain_tour`` applies); ``m0`` is the belief before
        any weighing. None if there are none."""
        cfg, g, P = self.cfg, self.setup.grid, self.planner
        idx = np.flatnonzero(self._window(nodes, arr, t_start, t_land))
        idx = idx[~self.road_visited[nodes[idx]]]
        xy = self.road.positions[nodes[idx]]
        idx, xy = idx[np.isfinite(xy).all(1)], xy[np.isfinite(xy).all(1)]
        p = self.blockage.probabilities(self.road)[nodes[idx]]
        keep = p > 0
        idx, xy, p = idx[keep], xy[keep], p[keep]
        if not len(idx):
            return None
        rows, cols = self._cells(xy)
        # L: would its blockage send the UGV further round than the UAVs can wait, or cut it off?
        if self.detour_road is not self.road:
            self.detour_cache, self.detour_road = {}, self.road
        slack_m = (cfg.rendezvous_slack_s + cfg.hover_margin_s) * self.setup.robots[self.ugv].speed
        q = np.zeros(self.road.n_nodes)
        q[nodes[idx]] = 1.0  # clear_probabilities with p = 1 here: 0 from the first long detour on
        L = np.zeros(len(idx))
        for k, i in enumerate(idx):
            if 0 < i < len(nodes) - 1:  # one at a time, each against its neighbours on the route
                L[k] = float(clear_probabilities(self.road, nodes[i - 1:i + 2], q, slack_m, self.detour_cache)[1] == 0.0)
        # B: the belief the UGV's route sweeps in the recon_belief_s after it, over the window's 90th percentile
        B = np.zeros(len(idx))
        if cfg.recon_belief_s > 0 and P.F_road.shape[0] == self.road.n_nodes:
            v = P.F_road @ P.seen(m0, P.p_ugv)  # per road node: the belief a UGV look there sweeps
            first = np.zeros(len(nodes), dtype=bool)
            first[np.unique(nodes, return_index=True)[1]] = True  # a node met twice on the route sweeps once
            s = np.where(first, v[nodes], 0.0)
            cs = np.concatenate([[0.0], np.cumsum(s)])
            end = np.searchsorted(np.maximum.accumulate(arr), arr[idx] + cfg.recon_belief_s, side="right")
            B = cs[end] - cs[idx + 1]
            b90 = float(np.percentile(B, 90))
            B = np.minimum(1.0, B / b90) if b90 > 0 else np.zeros(len(idx))
        G = cfg.recon_floor + (1.0 - cfg.recon_floor) * np.maximum(L, B)
        worth = (p / max(cfg.block_prior, 1e-9)) * G
        cell, k = np.unique(rows * g.cols + cols, return_index=True)  # one node per cell
        return cell, worth[k]
