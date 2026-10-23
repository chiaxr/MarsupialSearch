"""Expected time to find from one target-free run per map (a control variate).

Until a robot first gets within sensor range of the target, nothing any robot observes
depends on where the target is (with perfect sensors and no energy noise). So a mission
follows the same trajectory as a run with no target at all, up to that first contact. One
target-free run per map therefore gives, for every target cell and layer:

* when the target would be **found** directly: the first time a robot whose type ends the
  mission (``SimConfig.terminate_on``, the UGV) is within range. This part is exact;
* or when it would be **sighted** first, by a robot that only reports it: then the find time
  is predicted as the sighting time plus the UGV's drive from where it is to the target.
  This part is an estimate: how quickly a team confirms a sighting depends on the team.

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
    dist_cache: dict[int, np.ndarray] = {}
    for layer in range(N_LAYERS):
        sight = np.full(grid.shape, np.inf)
        for rid, spec in scenario.robots.items():
            if spec.robot_type not in cfg.terminate_on and layer in [int(l) for l in spec.sensor.visible_layers] \
                    and spec.sensor.p_detect > 0:
                sight = np.minimum(sight, contact[rid])
        Tl = np.where(np.isfinite(found), found, end)
        first = np.isfinite(sight) & (sight < found)
        if first.any():  # sighted first: the UGV drives from where it is to the target
            steps = np.searchsorted(result.traj_t, sight[first])
            src = np.array([road.nearest_node(traj[min(i, len(traj) - 1)]) for i in steps])
            dst = cell_node[first]
            drive = np.empty(len(src))
            for n in np.unique(src):
                if n not in dist_cache:
                    dist_cache[n] = road.distances_from(int(n))[0]
                sel = src == n
                drive[sel] = dist_cache[n][dst[sel]]
            Tl[first] = np.minimum(np.minimum(sight[first] + drive / speed, found[first]), cfg.max_time_s)
        T[layer] = Tl
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
