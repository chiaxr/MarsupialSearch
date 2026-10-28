"""Sortie planning on the road graph and the UAV grid, after the flying sidekick TSP.

A sortie ⟨i, j..., k⟩ takes off from the UGV at road node i, visits UAV-grid nodes j...,
and lands on the UGV at a later road node k (Murray and Chu's FSTSP). The objective is
the orienteering one: collect the most belief, with several points per sortie.

* :meth:`SortiePlanner.ugv_route` – the UGV's greedy belief-orienteering road route (the
  baseline's ``belief / (distance + offset)`` rule, chained). When no road node has belief
  left, the route moves to the road node with the most UAV belief nearby.
* :meth:`SortiePlanner.chain_tour` – the UAV's greedy tour from its take-off point to k:
  the next node has the best ``belief / (distance + offset)`` among those that still leave
  enough range to reach k. Shared-layer belief counts for the UAV only in cells it reaches
  before the UGV's route sweeps them.
* The UAV energy model: what a sortie may spend after hovering until the UGV reaches k,
  landing, and keeping a reserve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from marsupial_search.core import BeliefMap, MissionSetup, RobotSpec, RobotType, TraversabilityGraph, detection_map

from .common import PathFn, SearchConfig, dist, footprint_matrix


@dataclass
class SortieConfig(SearchConfig):
    relocation_radius_m: float = 800.0  # roads covered: drive to the most UAV belief within this radius
    fill_tours: bool = True  # True: use the whole energy budget (OP-mD); False: keep the best-rate prefix
    rendezvous_slack_s: float = 60.0  # hover a UAV keeps spare at its landing: lets the UGV divert to a sighting


class SortiePlanner:
    """Road routes for the UGV and tours for a UAV that lands on it."""

    def __init__(self, setup: MissionSetup, ugv: RobotSpec, uav: RobotSpec, cfg: SortieConfig):
        self.setup, self.ugv, self.uav, self.cfg = setup, ugv, uav, cfg
        self.air = setup.graphs[RobotType.UAV]
        self._road_cells: dict[int, np.ndarray] = {}
        self.set_road(setup.graphs[RobotType.UGV])
        self.F_air = footprint_matrix(setup.grid, self.air, uav.sensor.range_m)
        self.p_ugv, self.p_air = self.detection(setup, ugv), self.detection(setup, uav)  # per look: (layers, cells)
        air = np.flatnonzero((self.p_air > 0).any(1))
        seen_ugv = (self.p_ugv > 0).any(1)
        self.common_layers = [int(l) for l in air if seen_ugv[l]]  # seen by either robot
        self.own_layers = [int(l) for l in air if not seen_ugv[l]]  # seen only by the UAV

    @staticmethod
    def detection(setup: MissionSetup, spec: RobotSpec) -> np.ndarray:
        """Per layer and cell (flat), the chance a look by ``spec`` detects the target
        (``core.detection_map``): shape (layers, cells)."""
        p = detection_map(spec, setup.terrain, setup.occlusion)
        return p.reshape(len(p), -1)

    def for_uav(self, uav: RobotSpec) -> "SortiePlanner":
        """This planner for another UAV (its own sensor footprints and energy model); itself if the
        UAV has the same specs."""
        if uav.speed == self.uav.speed and uav.energy == self.uav.energy and uav.sensor == self.uav.sensor:
            return self
        return SortiePlanner(self.setup, self.ugv, uav, self.cfg)

    def set_road(self, road: TraversabilityGraph) -> None:
        """Plan on this road map (the prior map, updated with the mismatches found)."""
        self.road = road
        self.F_road = footprint_matrix(self.setup.grid, road, self.ugv.sensor.range_m, self._road_cells)
        self._seers = np.maximum(np.asarray(self.F_road.sum(axis=0)).ravel(), 1.0)  # per cell: road nodes that see it

    # ------------------------------------------------------------- energy model
    def energy_budget(self, energy: float) -> float:
        """Energy the UAV can spend on flying and hovering before the landing."""
        e, d = self.uav.energy, self.setup.docking
        hover = d.dock_duration_s + self.cfg.hover_margin_s + self.cfg.rendezvous_slack_s
        return energy - self.cfg.uav_reserve_fraction * e.capacity - e.hover_cost(hover)

    def max_tour_length(self, budget: float, tau: float) -> float:
        """Longest flight that leaves the UAV able to hover until ``tau`` and land."""
        e, v = self.uav.energy, self.uav.speed
        if e.idle_power * tau > budget:
            return -1.0
        if e.energy_per_m <= 0:
            return v * budget / e.idle_power
        waiting = (budget - e.idle_power * tau) / e.energy_per_m  # the UAV arrives first and hovers
        return waiting if waiting <= v * tau else budget / (e.idle_power / v + e.energy_per_m)

    # ---------------------------------------------------------------- sweeping
    @staticmethod
    def sweep(m: np.ndarray, F, node: int, p: np.ndarray) -> float:
        """Apply one look from ``node`` (detection ``p`` per layer and cell) to ``m`` in place;
        returns the belief swept."""
        cells = F.indices[F.indptr[node]:F.indptr[node + 1]]
        layers = np.flatnonzero((p > 0).any(1))
        win = m[np.ix_(layers, cells)] * p[np.ix_(layers, cells)]
        m[np.ix_(layers, cells)] -= win
        return float(win.sum())

    @staticmethod
    def seen(m: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Belief per cell one look would sweep: ``Σ_l p[l] m[l]`` (``p`` per layer and cell)."""
        layers = np.flatnonzero((p > 0).any(1))
        return (m[layers] * p[layers]).sum(0)

    def searchable_mass(self, belief: BeliefMap) -> np.ndarray:
        """Belief per (layer, cell) a sortie may search: a sighting keeps its mass, but a
        pursuit, not a sortie, handles it."""
        H, W = belief.mass.shape[1:]
        m = belief.mass.reshape(len(belief.mass), H * W).copy()
        if belief.detection_xy is not None:
            r0, c0, mask = self.setup.grid.disk_window(*belief.detection_xy, self.uav.sensor.range_m)
            rr, cc = np.nonzero(mask)
            m[:, (rr + r0) * W + cc + c0] = 0.0
        return m

    def ugv_sweep_times(self, route, times) -> np.ndarray:
        """Per cell, when the UGV route first sweeps it (inf: never)."""
        t_ugv = np.full(self.F_road.shape[1], np.inf)
        for n, t in zip(route, times):
            cells = self.F_road.indices[self.F_road.indptr[n]:self.F_road.indptr[n + 1]]
            t_ugv[cells] = np.minimum(t_ugv[cells], t)
        return t_ugv

    # --------------------------------------------------------------- UGV route
    def ugv_route(self, m: np.ndarray, src: int, start_xy, candidates: np.ndarray, horizon_s: float,
                  claim_eta: np.ndarray, path_fn: Optional[PathFn] = None,
                  belief_weight: float = 0.0) -> tuple[list[int], list[float]]:
        """Greedy belief-orienteering road route of about ``horizon_s``. Sweeps ``m`` in place.
        ``claim_eta`` marks the UAV-grid nodes already taken (-inf), which relocation ignores.
        The next node is chosen by shortest distance; ``path_fn`` plans the way to it (default:
        the shortest way). With ``belief_weight`` > 0, ``path_fn`` also gets a reward per road
        node: its share of the belief left in its footprint (a cell seen from k road nodes
        gives each 1/k), in metres at ``belief_weight`` / the chosen node's score (belief per
        metre): the way may go further to sweep more on the way."""
        cfg, road, v = self.cfg, self.road, self.ugv.speed
        cand = candidates.copy()
        nodes, times = [src], [dist(start_xy, road.positions[src]) / v]
        self.sweep(m, self.F_road, src, self.p_ugv)
        cand[src] = False
        cur = src
        while times[-1] < horizon_s:
            value = self.F_road @ self.seen(m, self.p_ugv)
            d, pred = road.distances_from(cur, limit=cfg.belief_radius_m)
            ok = cand & np.isfinite(d) & (value > 0)
            if ok.any():
                best = int(np.argmax(np.where(ok, value / (d + cfg.belief_distance_offset_m), -np.inf)))
                path = road.path_from_predecessors(pred, cur, best)
            else:  # nothing in range: the nearest road node with belief left
                path, _ = road.nearest_matching(cur, cand & (value > 0))
                if not path:
                    break
            if path_fn is not None and len(path) > 2:
                if belief_weight > 0 and ok.any():
                    rate = value[best] / (d[best] + cfg.belief_distance_offset_m)
                    share = self.F_road @ (self.seen(m, self.p_ugv) / self._seers)
                    path = path_fn(cur, path[-1], belief_weight / rate * share)[0] or path
                else:
                    path = path_fn(cur, path[-1])[0] or path
            self._extend(m, nodes, times, path, cand)
            cur = path[-1]
        if len(nodes) == 1:  # roads covered: move towards UAV belief
            path = self._relocation_path(m, src, horizon_s, claim_eta)
            if path_fn is not None and len(path) > 2:
                path = path_fn(src, path[-1])[0] or path
            self._extend(m, nodes, times, path, cand)
        return nodes, times

    def _extend(self, m, nodes, times, path, cand) -> float:
        """Append ``path[1:]`` to the route, sweeping ``m``; returns the belief swept."""
        gained = 0.0
        for n in path[1:]:
            times.append(times[-1] + dist(self.road.positions[nodes[-1]], self.road.positions[n]) / self.ugv.speed)
            nodes.append(n)
            gained += self.sweep(m, self.F_road, n, self.p_ugv)
            cand[n] = False
        return gained

    def _relocation_path(self, m, src: int, horizon_s: float, claim_eta: np.ndarray) -> list[int]:
        a = np.where(claim_eta < np.inf, 0.0, self.F_air @ self.seen(m, self.p_air))
        if not (a > 0).any():
            return []
        d, pred = self.road.distances_from(src, limit=0.5 * horizon_s * self.ugv.speed)
        reach = np.flatnonzero(np.isfinite(d))
        if len(reach) > 200:
            reach = reach[np.linspace(0, len(reach) - 1, 200).astype(int)]
        near = self.air.nodes_within(self.road.positions[reach], self.cfg.relocation_radius_m)
        score = np.array([a[idx].sum() for idx in near]) / (d[reach] + self.cfg.belief_distance_offset_m)
        best = int(reach[int(np.argmax(score))])
        return self.road.path_from_predecessors(pred, src, best) if best != src else []

    # ---------------------------------------------------------------- UAV tour
    def _rate(self, tau: float, ugv_value: float):
        dock, v = self.setup.docking, self.uav.speed
        overhead = dock.undock_duration_s + dock.dock_duration_s
        return lambda length, value: (ugv_value + value) / (overhead + max(tau, length / v))

    def chain_tour(self, m, t_ugv: np.ndarray, start, end, L_max: float, tau: float, ugv_value: float,
                   claim_eta: np.ndarray, t_off: float):
        """Greedy tour from ``start`` to ``end``: the next node has the best ``belief /
        (distance + offset)`` among those that leave enough range to reach ``end``. Collects
        the most belief early, which matters for time to find. Belief in layers both robots see
        counts for the UAV only in cells it reaches before the UGV route sweeps them (``t_ugv``);
        the team value drops what both would sweep. ``claim_eta``: per UAV-grid node, when another UAV
        reaches it (inf: unclaimed, -inf: claimed outright). Returns (rate, nodes, length, value),
        or None if ``end`` is out of range."""
        cfg, v = self.cfg, self.uav.speed
        L0 = dist(start, end)
        if L_max < L0:
            return None
        rate = self._rate(tau, ugv_value)
        own, common = m[self.own_layers], m[self.common_layers]  # per layer: (layers, cells)
        p_own, p_com = self.p_air[self.own_layers], self.p_air[self.common_layers]
        p_both = p_com * self.p_ugv[self.common_layers]  # swept by both: counted twice
        t_ugv = np.where(t_ugv <= tau, t_ugv, np.inf)  # the UGV stops at k
        value = self.F_air @ ((own * p_own).sum(0) + (common * p_com).sum(0))
        pos = self.air.positions
        to_end = np.hypot(*(pos - end).T)
        direct = np.hypot(*(pos - start).T)
        cand = np.flatnonzero((direct + to_end <= L_max) & (value > 0) & (claim_eta > t_off + direct / v))
        Fc, pts, to_end, claim = self.F_air[cand], pos[cand], to_end[cand], claim_eta[cand]
        vals = value[cand]
        used = np.zeros(len(cand), dtype=bool)
        cur, flown, uav_value, overlap, order = np.asarray(start, float), 0.0, 0.0, 0.0, []
        best = (rate(L0, 0.0), [], L0, 0.0)
        while True:
            d = np.hypot(*(pts - cur).T)
            feas = ~used & (vals > 0) & (flown + d + to_end <= L_max) & (claim > t_off + (flown + d) / v)
            if not feas.any():
                break
            c = int(np.argmax(np.where(feas, vals / (d + cfg.belief_distance_offset_m), -np.inf)))
            used[c] = True
            order.append(int(cand[c]))
            flown += float(d[c])
            uav_value += float(vals[c])
            cur = pts[c]
            cells = Fc.indices[Fc.indptr[c]:Fc.indptr[c + 1]]
            later = t_ugv[cells] > flown / v
            both = cells[later & np.isfinite(t_ugv[cells])]
            overlap += float((common[:, both] * p_both[:, both]).sum())
            own[:, cells] *= 1.0 - p_own[:, cells]
            common[:, cells] = np.where(later, common[:, cells] * (1.0 - p_com[:, cells]), common[:, cells])
            vals = Fc @ ((own * p_own).sum(0) + (common * p_com).sum(0) * (t_ugv > flown / v))
            length = flown + float(to_end[c])
            r = rate(length, uav_value - overlap)
            if r > best[0] or cfg.fill_tours:
                best = (r, list(order), length, uav_value)
        return best
