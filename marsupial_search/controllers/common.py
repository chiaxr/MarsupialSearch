"""Building blocks shared by the controllers: search settings, footprints, distances and pursuit paths."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix

from marsupial_search.core import GridSpec, TraversabilityGraph


@dataclass
class SearchConfig:
    uav_reserve_fraction: float = 0.10  # energy kept in hand on arrival at the rendezvous
    hover_margin_s: float = 30.0  # extra hover budget at the rendezvous
    belief_radius_m: float = 1500.0  # candidate search radius for the next node
    belief_distance_offset_m: float = 100.0  # score = belief in footprint / (distance + offset)
    pursue_sightings: bool = True  # send the UGV to confirm UAV sightings


def dist(a, b) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def footprint_matrix(grid: GridSpec, graph: TraversabilityGraph, radius: float) -> csr_matrix:
    """Sparse (n_nodes, n_cells) indicator of the cells a sensor sees from each node."""
    rows, cols = [], []
    for n, (x, y) in enumerate(graph.positions):
        r0, c0, mask = grid.disk_window(x, y, radius)
        rr, cc = np.nonzero(mask)
        rows.append(np.full(len(rr), n))
        cols.append((rr + r0) * grid.cols + (cc + c0))
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    return csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(graph.n_nodes, grid.rows * grid.cols))


def pursuit_path(road: TraversabilityGraph, sighting_xy, src: int) -> list[int]:
    """Path from ``src`` that drives along the road edge closest to a sighting."""
    p = np.asarray(sighting_xy)
    a, b = road.positions[road.edges[:, 0]], road.positions[road.edges[:, 1]]
    ab = b - a
    t = np.clip(((p - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0, 1)
    k = int(np.argmin(np.linalg.norm(a + t[:, None] * ab - p, axis=1)))
    u, v = map(int, road.edges[k])
    pu, du = road.shortest_path(src, u)
    pv, dv = road.shortest_path(src, v)
    return pu + [v] if du <= dv else pv + [u]
