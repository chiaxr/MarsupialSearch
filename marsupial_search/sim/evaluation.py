"""Expected time to find from one target-free run per map (a control variate).

Until a robot first detects the target, nothing any robot observes depends on where the
target is (with no energy noise): a look that misses it reads the same as a look at an empty
cell. So a mission follows the same trajectory as a run with no target at all, up to that first
detection. One target-free run per map therefore gives, for every target cell and layer:

* when the target would be **found** directly: the first time a robot whose type ends the
  mission (``SimConfig.terminate_on``, the UGV) is within range. This part is exact;
* or when it would be **sighted** first, by a robot that only reports it: then the find time
  is predicted as the sighting time plus the UGV's drive from where it is to the target.
  This part is an estimate: how quickly a team confirms a sighting depends on the team. A
  sensor that detects a layer with probability ``p < 1`` per look (per pass over the cell)
  sights it on its k-th look with probability ``p_k Π_{i<k} (1 − p_i)``; ``T̂`` is the
  expectation over which look, if any before the find, sights it.

The prediction ``T̂`` is averaged over the whole target distribution, ``E[T̂]``, and its error is
measured on a few real missions with sampled targets. The estimate of the expected time to
find, ``E[T̂] + mean(T − T̂)``, is unbiased whatever the quality of ``T̂``; a good ``T̂`` makes
``T − T̂`` vary far less than ``T``, so it needs far fewer missions than averaging ``T``.

Times count a mission that never finds the target at its end time.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from marsupial_search.core import N_LAYERS, RobotMode, RobotType

from .scenario import Scenario
from .simulator import MissionResult, SimConfig

_ACTIVE = list(RobotMode).index(RobotMode.ACTIVE)


def target_free(cfg: SimConfig) -> SimConfig:
    """``cfg`` without a target to find or sight."""
    return dataclasses.replace(cfg, target=False)


def target_weights(scenario: Scenario) -> np.ndarray:
    """P(target in layer l of cell), shape (layers, H, W): what ``resample_target`` draws from."""
    m = scenario.meta
    q = m["target_distribution"] * m["confirmable_mask"]
    lf = scenario.layer_fractions
    w = q[None] * lf / np.maximum(lf.sum(0, keepdims=True), 1e-300)
    return w / w.sum()


def first_contact(scenario: Scenario, result: MissionResult, rid: str) -> np.ndarray:
    """Per cell, the first time robot ``rid`` sensed it while active (inf: never)."""
    grid, spec = scenario.grid, scenario.robots[rid]
    out = np.full(grid.shape, np.inf)
    t, xy = result.traj_t, result.trajectories[rid]
    active = result.modes[rid] == _ACTIVE
    last = None
    for i in np.flatnonzero(active):
        p = (round(float(xy[i, 0]), 3), round(float(xy[i, 1]), 3))
        if p == last:
            continue
        last = p
        r0, c0, mask = grid.disk_window(p[0], p[1], spec.sensor.range_m)
        win = out[r0:r0 + mask.shape[0], c0:c0 + mask.shape[1]]
        win[mask & np.isinf(win)] = t[i]
    return out


def look_events(scenario: Scenario, result: MissionResult, rid: str) -> tuple[np.ndarray, np.ndarray]:
    """Every look robot ``rid`` took while active, in time order: (flat cell index, time). A look
    is a cell entering its footprint, as in the simulator: staying in view is one look."""
    grid, spec = scenario.grid, scenario.robots[rid]
    t, xy = result.traj_t, result.trajectories[rid]
    active = result.modes[rid] == _ACTIVE
    view = np.zeros(grid.shape, dtype=bool)  # the footprint last step
    prev, cells, times = None, [], []
    for i in range(len(t)):
        if not active[i]:
            if prev is not None:
                r0, c0, m0 = prev[1]
                view[r0:r0 + m0.shape[0], c0:c0 + m0.shape[1]] = False
                prev = None
            continue
        p = (float(xy[i, 0]), float(xy[i, 1]))
        if prev is not None and p == prev[0]:
            continue
        r0, c0, mask = grid.disk_window(p[0], p[1], spec.sensor.range_m)
        h, w = mask.shape
        new = mask & ~view[r0:r0 + h, c0:c0 + w]
        if prev is not None:
            pr, pc, pm = prev[1]
            view[pr:pr + pm.shape[0], pc:pc + pm.shape[1]] = False
        view[r0:r0 + h, c0:c0 + w] = mask
        prev = (p, (r0, c0, mask))
        rr, cc = np.nonzero(new)
        cells.append((rr + r0) * grid.cols + cc + c0)
        times.append(np.full(len(rr), t[i]))
    if not cells:
        return np.zeros(0, dtype=int), np.zeros(0)
    return np.concatenate(cells), np.concatenate(times)


def predicted_find_times(scenario: Scenario, result: MissionResult, cfg: SimConfig) -> np.ndarray:
    """``T̂`` per (layer, cell) from the target-free run ``result`` (see the module docstring)."""
    grid = scenario.grid
    ugv = next(r for r, s in scenario.robots.items() if s.robot_type == RobotType.UGV)
    contact = {rid: first_contact(scenario, result, rid) for rid in scenario.robots}
    found = np.full(grid.shape, np.inf)
    for rid, spec in scenario.robots.items():
        if spec.robot_type in cfg.terminate_on:
            found = np.minimum(found, contact[rid])
    end = result.time_s
    T = np.empty((N_LAYERS,) + grid.shape)
    road = scenario.true_road()  # the UGV drives the roads as they are, not as mapped
    X, Y = grid.cell_centers()
    cell_node = road.nearest_nodes(np.stack([X.ravel(), Y.ravel()], 1)).reshape(grid.shape)
    traj, speed = result.trajectories[ugv], scenario.robots[ugv].speed
    seers = [rid for rid, spec in scenario.robots.items() if spec.robot_type not in cfg.terminate_on]
    looks = {rid: look_events(scenario, result, rid) for rid in seers}
    found_flat, cell_node_flat = found.ravel(), cell_node.ravel()
    dist_cache: dict[int, np.ndarray] = {}
    for layer in range(N_LAYERS):
        Tl = np.where(np.isfinite(found), found, end).ravel()
        cells, times, pd = [], [], []
        for rid in seers:
            q = scenario.robots[rid].sensor.layer_p()[layer]
            if q > 0:
                c, t = looks[rid]
                cells.append(c)
                times.append(t)
                pd.append(np.full(len(c), q))
        if cells:
            cells, times, pd = map(np.concatenate, (cells, times, pd))
            keep = times < found_flat[cells]  # sighted before the UGV gets there
            cells, times, pd = cells[keep], times[keep], pd[keep]
            o = np.lexsort((times, cells))
            cells, times, pd = cells[o], times[o], pd[o]
            # P(this look is the first to sight it): p_k Π_{i<k} (1 − p_i) over the cell's looks so far
            start = np.r_[True, cells[1:] != cells[:-1]]
            first = np.flatnonzero(start)[np.cumsum(start) - 1]  # per look: its cell's first look
            certain = pd >= 1
            lq = np.log1p(-np.where(certain, 0.0, pd))
            lq_before, certain_before = np.cumsum(lq) - lq, np.cumsum(certain) - certain
            missed = np.exp(lq_before - lq_before[first]) * (certain_before == certain_before[first])
            prob = pd * missed
            live = prob > 0  # looks after a certain sighting cannot be the first
            cells, times, prob = cells[live], times[live], prob[live]
        if len(cells):  # sighted first: the UGV drives from where it is to the target
            steps = np.searchsorted(result.traj_t, times)
            src = np.array([road.nearest_node(traj[min(i, len(traj) - 1)]) for i in steps])
            dst = cell_node_flat[cells]
            drive = np.empty(len(src))
            for n in np.unique(src):
                if n not in dist_cache:
                    dist_cache[n] = road.distances_from(int(n))[0]
                sel = src == n
                drive[sel] = dist_cache[n][dst[sel]]
            Tk = np.minimum(np.minimum(times + drive / speed, found_flat[cells]), cfg.max_time_s)
            p_sight = np.bincount(cells, prob, minlength=Tl.size)
            Tl = np.bincount(cells, prob * Tk, minlength=Tl.size) + (1.0 - p_sight) * Tl
        T[layer] = Tl.reshape(grid.shape)
    return T


def predicted_time(scenario: Scenario, T: np.ndarray, target_xy, target_layer) -> float:
    r, c = scenario.grid.world_to_cell(*target_xy)
    return float(T[int(target_layer), r, c])


def expected(T: np.ndarray, weights: np.ndarray) -> float:
    """``E[T̂]`` over the target distribution."""
    return float((T * weights).sum())


def weighted_quantile(T: np.ndarray, weights: np.ndarray, q: float) -> float:
    t, w = T.ravel(), weights.ravel()
    keep = w > 0
    t, w = t[keep], w[keep]
    o = np.argsort(t)
    return float(t[o][np.searchsorted(np.cumsum(w[o]), q * w.sum())])
