"""Building blocks shared by the controllers: search settings, footprints, distances, pursuit paths and detours."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.sparse import csr_matrix

from marsupial_search.core import GridSpec, TraversabilityGraph

# (src, dst) -> (path, cost), as TraversabilityGraph.shortest_path; a search leg may also pass a per-node reward (metres)
PathFn = Callable[..., tuple[list[int], float]]


@dataclass
class SearchConfig:
    uav_reserve_fraction: float = 0.10  # energy kept in hand on arrival at the rendezvous
    hover_margin_s: float = 30.0  # extra hover budget at the rendezvous
    belief_radius_m: float = 1500.0  # candidate search radius for the next node
    belief_distance_offset_m: float = 100.0  # score = belief in footprint / (distance + offset)
    pursue_sightings: bool = True  # send the UGV to confirm UAV sightings


def dist(a, b) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def footprint_matrix(grid: GridSpec, graph: TraversabilityGraph, radius: float,
                     cache: Optional[dict[int, np.ndarray]] = None) -> csr_matrix:
    """Sparse (n_nodes, n_cells) indicator of the cells a sensor sees from each node (none from a
    blocked node). ``cache`` (node -> cells) keeps the footprints for the next call on a changed graph."""
    rows, cols = [np.zeros(0, int)], [np.zeros(0, int)]
    for n in np.flatnonzero(~graph.blocked):
        cells = cache.get(n) if cache is not None else None
        if cells is None:
            r0, c0, mask = grid.disk_window(*graph.positions[n], radius)
            rr, cc = np.nonzero(mask)
            cells = (rr + r0) * grid.cols + (cc + c0)
            if cache is not None:
                cache[n] = cells
        rows.append(np.full(len(cells), n))
        cols.append(cells)
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    return csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(graph.n_nodes, grid.rows * grid.cols))


def pursuit_path(road: TraversabilityGraph, sighting_xy, src: int, path_fn: Optional[PathFn] = None) -> list[int]:
    """Path from ``src`` that drives along the road edge closest to a sighting, among those it can
    reach. ``path_fn`` plans the way to the edge (default: the shortest path)."""
    p = np.asarray(sighting_xy)
    edges = road.edges[np.isfinite(road.distances_from(src)[0])[road.edges[:, 0]]]
    a, b = road.positions[edges[:, 0]], road.positions[edges[:, 1]]
    ab = b - a
    t = np.clip(((p - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0, 1)
    k = int(np.argmin(np.linalg.norm(a + t[:, None] * ab - p, axis=1)))
    u, v = map(int, edges[k])
    path_fn = path_fn or road.shortest_path
    pu, du = path_fn(src, u)
    pv, dv = path_fn(src, v)
    return pu + [v] if du <= dv else pv + [u]


def bypass(road: TraversabilityGraph, route: Sequence[int], keep: Sequence[int] = (),
           budget: float = 0.0, path_fn: Optional[PathFn] = None) -> tuple[list[int], np.ndarray]:
    """``route`` (from a node that is not blocked) going round the blocked nodes on it. Where it
    runs into one, it takes the shortest way to the first node of the route after it that makes
    the route at most ``budget`` metres longer in all (if none does, the one that adds least),
    never skipping a route index in ``keep`` whose node is not blocked. ``path_fn`` plans the way
    there once the node to rejoin at is chosen (default: the shortest way), if it is no longer
    than the budget left allows. Returns the new route and, per index of the old one, its index
    in the new one (-1: skipped)."""
    nodes = np.asarray(route, dtype=int)
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(road.positions[nodes], axis=0).T))])
    keep = sorted(i for i in keep if not road.blocked[nodes[i]])
    out, index = [int(nodes[0])], np.full(len(nodes), -1)
    index[0], k = 0, 0
    while k + 1 < len(nodes):
        a = int(nodes[k])
        if a == nodes[k + 1] or road.has_edge(a, nodes[k + 1]):
            out.append(int(nodes[k + 1]))
            k += 1
        else:
            d, pred = road.distances_from(a)
            ms = np.arange(k + 1, next((i for i in keep if i > k), len(nodes) - 1) + 1)
            ms = ms[np.isfinite(d[nodes[ms]])]
            if not len(ms):
                break
            extra = d[nodes[ms]] - (cum[ms] - cum[k])
            within = np.flatnonzero(extra <= budget)
            j = int(within[0]) if len(within) else int(np.argmin(extra))
            budget -= extra[j]
            k = int(ms[j])
            way = path_fn(a, int(nodes[k]))[0] if path_fn is not None else []
            longer = road.path_length(way) - d[nodes[k]] if way else 0.0
            if longer > max(budget, 0.0):  # it would make the landings ahead later than they can wait
                way, longer = [], 0.0
            budget -= longer
            out += (way or road.path_from_predecessors(pred, a, int(nodes[k])))[1:]
        index[k] = len(out) - 1
    return out, index
