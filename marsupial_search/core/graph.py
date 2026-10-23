"""Undirected, Euclidean-weighted traversability graph.

Used for both the UGV road network and the UAV flight grid. Depends only on
numpy/scipy so it can be reused unchanged in a deployment stack.
"""

from __future__ import annotations

import heapq
from typing import Sequence

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree


class TraversabilityGraph:
    def __init__(self, positions: np.ndarray, edges: np.ndarray, name: str = ""):
        positions = np.asarray(positions, dtype=float).reshape(-1, 2)
        edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        edges = edges[edges[:, 0] != edges[:, 1]]
        edges = np.unique(np.sort(edges, axis=1), axis=0)

        self.name = name
        self.positions = positions
        self.edges = edges
        self.lengths = np.linalg.norm(positions[edges[:, 0]] - positions[edges[:, 1]], axis=1)

        n = len(positions)
        rows = np.concatenate([edges[:, 0], edges[:, 1]])
        cols = np.concatenate([edges[:, 1], edges[:, 0]])
        data = np.concatenate([self.lengths, self.lengths])
        self.csr = csr_matrix((data, (rows, cols)), shape=(n, n))

        # Python-level adjacency for fast early-exit searches.
        indptr, indices, weights = self.csr.indptr, self.csr.indices, self.csr.data
        self._adj: list[list[tuple[int, float]]] = [
            list(zip(indices[indptr[i] : indptr[i + 1]].tolist(), weights[indptr[i] : indptr[i + 1]].tolist()))
            for i in range(n)
        ]
        self._kdtree = cKDTree(positions) if n else None

    # ------------------------------------------------------------------ queries
    @property
    def n_nodes(self) -> int:
        return len(self.positions)

    def neighbors(self, u: int) -> list[tuple[int, float]]:
        return self._adj[u]

    def has_edge(self, u: int, v: int) -> bool:
        return any(w == v for w, _ in self._adj[u])

    def nearest_node(self, xy: Sequence[float]) -> int:
        _, idx = self._kdtree.query(np.asarray(xy, dtype=float))
        return int(idx)

    def nodes_within(self, xy: Sequence[float], radius: float) -> list[int]:
        return self._kdtree.query_ball_point(np.asarray(xy, dtype=float), radius)

    def path_length(self, path: Sequence[int]) -> float:
        if len(path) < 2:
            return 0.0
        p = self.positions[np.asarray(path)]
        return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())

    def n_components(self) -> int:
        return int(connected_components(self.csr, directed=False)[0])

    # ----------------------------------------------------------------- searches
    def distances_from(self, src: int, limit: float = np.inf) -> tuple[np.ndarray, np.ndarray]:
        """Single-source shortest distances and predecessors (scipy, C speed)."""
        dist, pred = dijkstra(self.csr, directed=False, indices=src, limit=limit, return_predecessors=True)
        return dist, pred

    @staticmethod
    def path_from_predecessors(pred: np.ndarray, src: int, dst: int) -> list[int]:
        path = [dst]
        while path[-1] != src:
            p = int(pred[path[-1]])
            if p < 0:
                return []
            path.append(p)
        return path[::-1]

    def shortest_path(self, src: int, dst: int) -> tuple[list[int], float]:
        """A* shortest path; returns ``([], inf)`` if unreachable."""
        if src == dst:
            return [src], 0.0
        goal = self.positions[dst]
        pos = self.positions

        def h(u: int) -> float:
            return float(np.hypot(*(pos[u] - goal)))

        g = {src: 0.0}
        parent = {src: -1}
        heap = [(h(src), src)]
        closed = set()
        while heap:
            _, u = heapq.heappop(heap)
            if u in closed:
                continue
            if u == dst:
                break
            closed.add(u)
            gu = g[u]
            for v, w in self._adj[u]:
                nv = gu + w
                if nv < g.get(v, np.inf):
                    g[v] = nv
                    parent[v] = u
                    heapq.heappush(heap, (nv + h(v), v))
        if dst not in parent:
            return [], np.inf
        path = [dst]
        while parent[path[-1]] != -1:
            path.append(parent[path[-1]])
        return path[::-1], g[dst]

    def nearest_matching(self, src: int, candidates: np.ndarray) -> tuple[list[int], float]:
        """Path to the graph-nearest node with ``candidates[node] == True``.

        Early-exit Dijkstra: cheap when a candidate is close by, which is the
        common case for greedy coverage. Returns ``([], inf)`` if none reachable.
        """
        if candidates[src]:
            return [src], 0.0
        dist = {src: 0.0}
        parent = {src: -1}
        heap = [(0.0, src)]
        closed = set()
        while heap:
            d, u = heapq.heappop(heap)
            if u in closed:
                continue
            if candidates[u]:
                path = [u]
                while parent[path[-1]] != -1:
                    path.append(parent[path[-1]])
                return path[::-1], d
            closed.add(u)
            for v, w in self._adj[u]:
                nd = d + w
                if nd < dist.get(v, np.inf):
                    dist[v] = nd
                    parent[v] = u
                    heapq.heappush(heap, (nd, v))
        return [], np.inf
