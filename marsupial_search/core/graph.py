"""Undirected, Euclidean-weighted traversability graph, and what a robot knows of one.

Used for both the UGV road network and the UAV flight grid. Depends only on
numpy/scipy so it can be reused unchanged in a deployment stack.

A graph can mark nodes *blocked*: they keep their index (and position, if known) but
have no edges, and node queries skip them. :class:`GraphKnowledge` is a prior graph plus
the mismatches found so far (blocked nodes, and nodes and edges missing from the prior).
"""

from __future__ import annotations

import heapq
from typing import Mapping, Optional, Sequence

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree


class TraversabilityGraph:
    def __init__(self, positions: np.ndarray, edges: np.ndarray, name: str = "",
                 blocked: Optional[np.ndarray] = None):
        positions = np.asarray(positions, dtype=float).reshape(-1, 2)
        n = len(positions)
        edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        edges = edges[edges[:, 0] != edges[:, 1]]
        self.blocked = np.zeros(n, dtype=bool) if blocked is None else np.asarray(blocked, dtype=bool).copy()
        edges = edges[~self.blocked[edges].any(axis=1)]
        edges = np.unique(np.sort(edges, axis=1), axis=0)

        self.name = name
        self.positions = positions
        self.edges = edges
        self.lengths = np.linalg.norm(positions[edges[:, 0]] - positions[edges[:, 1]], axis=1)

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
        # Node queries skip blocked nodes: the k-d tree holds the others, _kd_ids maps back (None: all of them).
        open_ = np.flatnonzero(~self.blocked)
        self._kd_ids = open_ if len(open_) < n else None
        self._kdtree = cKDTree(positions[open_]) if len(open_) else None

    # ------------------------------------------------------------------ queries
    @property
    def n_nodes(self) -> int:
        return len(self.positions)

    def neighbors(self, u: int) -> list[tuple[int, float]]:
        return self._adj[u]

    def has_edge(self, u: int, v: int) -> bool:
        return any(w == v for w, _ in self._adj[u])

    def nearest_node(self, xy: Sequence[float]) -> int:
        """The nearest node that is not blocked."""
        return int(self.nearest_nodes(xy))

    def nearest_nodes(self, xy: np.ndarray) -> np.ndarray:
        """``nearest_node`` for each point of an (..., 2) array."""
        _, idx = self._kdtree.query(np.asarray(xy, dtype=float))
        return idx if self._kd_ids is None else self._kd_ids[idx]

    def nodes_within(self, xy: Sequence[float], radius: float) -> list[int]:
        """Nodes (not blocked) within ``radius`` of a point; for an (n, 2) array, a list per point."""
        xy = np.asarray(xy, dtype=float)
        found = self._kdtree.query_ball_point(xy, radius)
        if self._kd_ids is None:
            return found
        ids = self._kd_ids
        if xy.ndim == 1:
            return ids[found].tolist()
        out = np.empty(len(found), dtype=object)
        out[:] = [ids[f].tolist() for f in found]
        return out

    def path_length(self, path: Sequence[int]) -> float:
        if len(path) < 2:
            return 0.0
        p = self.positions[np.asarray(path)]
        return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())

    def n_components(self) -> int:
        return int(connected_components(self.csr, directed=False)[0])

    def with_changes(self, blocked: Sequence[int] = (), new_nodes: Optional[Mapping[int, Sequence[float]]] = None,
                     new_edges: Sequence[Sequence[int]] = ()) -> "TraversabilityGraph":
        """This graph with ``blocked`` nodes made impassable (they keep their index but lose their
        edges) and ``new_nodes`` (index -> position, indices from ``n_nodes`` on) and ``new_edges``
        added. An index past ``n_nodes`` whose position is not given is a blocked placeholder
        (position NaN), so indices always match the full graph's."""
        new_nodes = dict(new_nodes or {})
        new_edges = np.asarray(new_edges, dtype=np.int64).reshape(-1, 2)
        n = max([self.n_nodes - 1] + list(new_nodes) + new_edges.ravel().tolist()) + 1
        pos = np.full((n, 2), np.nan)
        pos[:self.n_nodes] = self.positions
        for i, xy in new_nodes.items():
            pos[i] = xy
        blk = np.isnan(pos[:, 0])
        blk[:self.n_nodes] |= self.blocked
        blk[np.asarray(blocked, dtype=np.int64)] = True
        return TraversabilityGraph(pos, np.concatenate([self.edges, new_edges]), self.name, blocked=blk)

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


class GraphKnowledge:
    """A prior graph plus the mismatches found so far: nodes found blocked, and nodes and edges
    missing from the prior. ``graph`` is the current map (indices of the prior are kept).

    Changes only accumulate. :meth:`changes` gives them in the form :meth:`add` takes, e.g. to
    send them over the radio.
    """

    def __init__(self, prior: TraversabilityGraph):
        self.prior = prior
        self.blocked: set[int] = set()
        self.new_nodes: dict[int, tuple[float, float]] = {}
        self.new_edges: set[tuple[int, int]] = set()
        self.version = 0  # bumped by every change
        self._graph, self._graph_version = prior, 0

    def __bool__(self) -> bool:
        return bool(self.blocked or self.new_nodes or self.new_edges)

    @property
    def graph(self) -> TraversabilityGraph:
        if self._graph_version != self.version:
            self._graph = self.prior.with_changes(sorted(self.blocked), self.new_nodes, sorted(self.new_edges))
            self._graph_version = self.version
        return self._graph

    def add(self, blocked: Sequence[int] = (), new_nodes: Optional[Mapping[int, Sequence[float]]] = None,
            new_edges: Sequence[Sequence[int]] = ()) -> bool:
        """Record changes; True if any was new."""
        n0 = len(self.blocked) + len(self.new_nodes) + len(self.new_edges)
        self.blocked.update(int(b) for b in blocked)
        for i, xy in (new_nodes or {}).items():
            self.new_nodes.setdefault(int(i), (float(xy[0]), float(xy[1])))
        self.new_edges.update((min(int(a), int(b)), max(int(a), int(b))) for a, b in new_edges)
        changed = len(self.blocked) + len(self.new_nodes) + len(self.new_edges) > n0
        self.version += changed
        return changed

    def changes(self, exclude: Optional["GraphKnowledge"] = None) -> dict:
        """The changes known here (``blocked``, ``new_nodes``, ``new_edges``), less those ``exclude`` knows."""
        if exclude is None:
            return {"blocked": sorted(self.blocked), "new_nodes": dict(self.new_nodes),
                    "new_edges": sorted(self.new_edges)}
        return {"blocked": sorted(self.blocked - exclude.blocked),
                "new_nodes": {i: xy for i, xy in self.new_nodes.items() if i not in exclude.new_nodes},
                "new_edges": sorted(self.new_edges - exclude.new_edges)}
