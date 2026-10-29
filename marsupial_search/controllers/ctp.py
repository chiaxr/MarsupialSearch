"""Canadian Traveller's Problem (CTP) on a road map with blocked nodes.

The UGV's road map may have blocked nodes it does not know of: it finds one only once it is
within sensor range of it, and must then go round. Planning a path on the map as if every
node were passable (the *optimistic* policy) ignores what that costs: a long stretch of
unseen road may end in a blockage and a long way back.

:class:`BlockageModel`
    Per road node, the probability that it is blocked: 0 once a robot has seen it (a robot
    within sensor range sees a blockage, unless it reports the node ``hidden``: it could not
    make it out, e.g. under trees) or if its blockage would cut off more of the road map than
    blockages ever cut off (``max_cutoff_fraction``: the generator's ``road_cutoff_fraction``);
    otherwise the blocked share of the nodes seen so far, under a Beta prior.
:class:`CtpPlanner`
    Optimistic rollout (ORO; Eyerich, Keller and Helmert, "High-quality policies for the
    Canadian Traveler's Problem", AAAI 2010). At a junction it takes the next node ``v`` with
    the least ``len(u, v) + E[cost of the optimistic policy from v]``, the expectation over
    ``samples`` blockage scenarios drawn from the model (the same ones for every candidate).
    A rollout drives the optimistic policy in one scenario: the shortest path on the map less
    the blockages found so far, finding each blocked node once within sensor range of it, and
    planning again when its path runs into one (a rollout that strays further from the goal
    than ``reach_factor`` times the shortest distance counts as cut off). Rollouts cost at least
    the shortest distance, so a candidate whose shortest distance is already worse than the best
    rollout cost is skipped, and when nothing within sensor range of the shortest path may be
    blocked, that path is the answer. The path returned is what the policy drives if it finds no
    blockage on the way: nodes it passes within sensor range of are open from then on.

    With a per-node ``reward`` (metres), the cost of driving into a node is its length less the
    node's reward, but at least ``MIN_COST_FRACTION`` of the length (so shortest paths stay well
    defined), and the full length when the path or rollout has been there already: the policy
    then minimises E[metres driven - reward collected], e.g. with the belief a search leg sweeps
    as the reward. A rollout then drives one of two optimistic policies, shortest ways at these
    costs or at length alone (each move counted at these costs either way), and a candidate
    scores the cheaper one's expectation (rollout with several base policies; Bertsekas): the
    first heads for the reward wherever it starts, so on its own a safe way without reward
    would never be valued.

Depends only on numpy/scipy.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from marsupial_search.core import GraphKnowledge, TraversabilityGraph

MIN_COST_FRACTION = 0.1  # with rewards: driving into a node costs at least this share of the way there


def cut_off_counts(graph: TraversabilityGraph) -> np.ndarray:
    """Per node: how many nodes of its connected piece blocking it would cut off from the largest
    part left (0: it is not an articulation point). Iterative Tarjan."""
    n = graph.n_nodes
    disc, low, size = np.full(n, -1), np.zeros(n, dtype=int), np.ones(n, dtype=int)
    split, biggest = np.zeros(n, dtype=int), np.zeros(n, dtype=int)  # per node: nodes in the subtrees it separates, the largest
    out = np.zeros(n, dtype=int)
    adj = [[v for v, _ in graph.neighbors(u)] for u in range(n)]
    t = 0
    for root in range(n):
        if disc[root] >= 0 or not adj[root]:
            continue
        disc[root] = low[root] = t
        t += 1
        piece, stack = [root], [(root, -1, iter(adj[root]))]
        while stack:
            u, parent, it = stack[-1]
            v = next(it, None)
            if v is None:
                stack.pop()
                if parent >= 0:
                    low[parent] = min(low[parent], low[u])
                    size[parent] += size[u]
                    if low[u] >= disc[parent]:  # u's subtree hangs on parent alone (always so for the root)
                        split[parent] += size[u]
                        biggest[parent] = max(biggest[parent], size[u])
                continue
            if disc[v] < 0:
                disc[v] = low[v] = t
                t += 1
                piece.append(v)
                stack.append((v, u, iter(adj[v])))
            elif v != parent:
                low[u] = min(low[u], disc[v])
        piece = np.asarray(piece)
        rest = size[root] - 1 - split[piece]  # the part still joined to the rest of the tree
        out[piece] = size[root] - 1 - np.maximum(biggest[piece], rest)
    return out


def cut_nodes(graph: TraversabilityGraph) -> np.ndarray:
    """Per node: is it an articulation point (would blocking it split its piece of the graph)?"""
    return cut_off_counts(graph) > 0


def clear_probabilities(road: TraversabilityGraph, route, p: np.ndarray, max_detour_m: float,
                        cache: Optional[dict] = None) -> np.ndarray:
    """Per index of ``route``: the chance the UGV gets there with no detour longer than
    ``max_detour_m``, i.e. no node on the way that may be blocked (``p``) and whose blockage would
    send it more than that much further round (or cut it off). ``cache`` keeps each node's answer
    for the next call on the same map."""
    route = np.asarray(route, dtype=int)
    cache = {} if cache is None else cache
    pos, csr = road.positions, road.csr
    ok = np.ones(len(route))
    for i in range(1, len(route) - 1):
        a, j, b = int(route[i - 1]), int(route[i]), int(route[i + 1])
        if p[j] <= 0 or a == b:
            continue
        key = (a, j, b)
        if key not in cache:
            data = csr.data.copy()
            data[csr.indptr[j]:csr.indptr[j + 1]] = np.inf
            data[np.flatnonzero(csr.indices == j)] = np.inf
            direct = float(np.hypot(*(pos[j] - pos[a])) + np.hypot(*(pos[b] - pos[j])))
            g = csr_matrix((data, csr.indices, csr.indptr), shape=csr.shape)
            d = dijkstra(g, directed=True, indices=a, limit=direct + max_detour_m)
            cache[key] = not np.isfinite(d[b])
        if cache[key]:
            ok[i] = 1.0 - p[j]
    return np.cumprod(ok)


class BlockageModel:
    """What a robot knows of which road nodes are blocked. ``look`` records the nodes a sensor
    has seen; ``probabilities`` gives, per node of the current map, the chance it is blocked."""

    def __init__(self, roads: GraphKnowledge, prior: float = 0.05, prior_weight: float = 20.0,
                 max_cutoff_fraction: float = 0.0):
        self.roads = roads
        self.alpha, self.beta = prior * prior_weight, (1.0 - prior) * prior_weight
        self.max_cutoff_fraction = max_cutoff_fraction  # blockages cut off at most this share of the mapped nodes (1: no limit)
        self.seen = np.zeros(roads.prior.n_nodes, dtype=bool)  # mapped nodes seen by a robot
        self.version = 0  # bumped whenever the probabilities may change
        self._cut, self._cut_version = None, -1
        self._p, self._p_key = None, None

    def look(self, graph: TraversabilityGraph, xy, radius: float, hidden=()) -> None:
        """A sensor at ``xy`` saw every road node within ``radius``, but those it reports
        ``hidden`` (it could not make them out: they stay as unknown as before)."""
        hidden = set(hidden)
        nodes = [n for n in graph.nodes_within(xy, radius) if n < len(self.seen) and n not in hidden]
        if nodes and not self.seen[nodes].all():
            self.seen[nodes] = True
            self.version += 1

    def block_rate(self) -> float:
        """Posterior mean of the blocked share of the mapped nodes that may be blocked."""
        blocked = np.zeros(len(self.seen), dtype=bool)
        blocked[sorted(self.roads.blocked)] = True
        open_ = self.seen & ~blocked & ~self._cuts()
        return (self.alpha + blocked.sum()) / (self.alpha + self.beta + blocked.sum() + open_.sum())

    def probabilities(self, graph: TraversabilityGraph) -> np.ndarray:
        """Per node of ``graph`` (the current map): the chance it is blocked. Known blocked nodes
        and nodes missing from the prior map (only mapped nodes are ever blocked) get 0."""
        key = (self.version, self.roads.version, graph.n_nodes)
        if key != self._p_key:
            p = np.zeros(graph.n_nodes)
            m = len(self.seen)
            p[:m] = np.where(self.seen | self._cuts() | graph.blocked[:m], 0.0, self.block_rate())
            self._p, self._p_key = p, key
        return self._p

    def _cuts(self) -> np.ndarray:
        """The mapped nodes that cannot be blocked: their blockage would cut off more of the map
        (less the known blockages) than the blockages may still cut off. With no limit, none."""
        if self.max_cutoff_fraction >= 1.0:
            return np.zeros(len(self.seen), dtype=bool)
        if self._cut_version != len(self.roads.blocked):
            g = self.roads.prior.with_changes(blocked=sorted(self.roads.blocked))
            _, comp = connected_components(g.csr, directed=False)
            main = np.bincount(comp[~g.blocked]).argmax() if (~g.blocked).any() else -1
            in_main = (comp == main) & ~g.blocked
            budget = self.max_cutoff_fraction * g.n_nodes - ((~g.blocked).sum() - in_main.sum())  # left to cut off
            self._cut = in_main & (cut_off_counts(g) > budget)
            self._cut_version = len(self.roads.blocked)
        return self._cut


class CtpPlanner:
    """Paths on ``road`` (the current map) that take the risk of unseen blockages into account
    (module docstring). ``sense_m``: the range within which the robot finds a blocked node."""

    def __init__(self, road: TraversabilityGraph, sense_m: float, samples: int = 16, seed: int = 0,
                 reach_factor: float = 3.0, reach_margin_m: float = 500.0):
        self.road, self.samples = road, samples
        self.reach_factor, self.reach_margin_m = reach_factor, reach_margin_m
        self.rng = np.random.default_rng(seed)
        n = road.n_nodes
        csr = road.csr.tocsr(copy=True)
        csr.sort_indices()
        self._indptr, self._indices, self._lengths = csr.indptr, csr.indices, csr.data
        self._data = self._lengths  # this query's costs (of driving from an entry's column into its row)
        rows = np.repeat(np.arange(n), np.diff(csr.indptr))
        self._rows = rows
        keys = rows.astype(np.int64) * n + csr.indices
        self._mirror = np.searchsorted(keys, csr.indices.astype(np.int64) * n + rows)  # entry (v, u) of (u, v)
        # Sensor reach: R[y] holds the nodes seen from y (and y's neighbours, even if farther apart).
        pos = np.where(np.isnan(road.positions), 1e12, road.positions)
        near = road.nodes_within(pos, sense_m) if road._kdtree is not None else [[] for _ in range(n)]
        r_rows, r_cols = [], []
        for y in range(n):
            cols = set(near[y]) | {v for v, _ in road.neighbors(y)}
            cols.discard(y)
            r_rows += [y] * len(cols)
            r_cols += sorted(cols)
        self.reach = csr_matrix((np.ones(len(r_rows)), (r_rows, r_cols)), shape=(n, n))
        self.reach = ((self.reach + self.reach.T) > 0).astype(float).tocsr()
        self._trees: dict[frozenset, tuple[list[float], list[int]]] = {}  # this query's, by the nodes taken as blocked
        self._goal, self._limit = -1, np.inf
        self._reward: Optional[np.ndarray] = None
        self._passed: set[int] = set()  # the nodes of the path planned so far

    # ------------------------------------------------------------------ trees
    def _tree(self, goal: int, blocked: frozenset, by_length: bool = False) -> tuple[list[float], list[int]]:
        """Shortest distance to ``goal`` (up to this query's limit) at this query's costs (or by
        length alone), and the next node towards it, on the map less ``blocked``."""
        if goal != self._goal:
            self._trees, self._goal, self._limit = {}, goal, np.inf
        tree = self._trees.get((by_length, blocked))
        if tree is not None:
            return tree
        data = self._lengths if by_length else self._data
        if blocked:
            data = data.copy()
            for b in blocked:
                pos = np.arange(self._indptr[b], self._indptr[b + 1])
                data[pos] = np.inf
                data[self._mirror[pos]] = np.inf
        g = csr_matrix((data, self._indices, self._indptr), shape=(self.road.n_nodes,) * 2)
        d, pred = dijkstra(g, directed=True, indices=goal, limit=self._limit, return_predecessors=True)
        tree = (d.tolist(), pred.tolist())
        self._trees[by_length, blocked] = tree
        return tree

    def _tree_path(self, src: int, goal: int, nxt: list[int]) -> list[int]:
        path = [src]
        while path[-1] != goal:
            path.append(nxt[path[-1]])
        return path

    # -------------------------------------------------------------- scenarios
    def _draw(self, p: np.ndarray) -> None:
        """Draw this query's blockage scenarios; ``_seen_blocked[y, k]``: how many of scenario k's
        blocked nodes are seen from y."""
        self._blk = self.rng.random((self.samples, len(p))) < p
        self._seen_blocked = np.asarray(self.reach @ self._blk.T.astype(float)).round().astype(int)
        self._left = int(self._blk.sum())

    def _reveal(self, y: int) -> None:
        """The path passes ``y``: the nodes seen from there are open in every scenario."""
        if not self._left:
            return
        cols = np.append(self.reach.indices[self.reach.indptr[y]:self.reach.indptr[y + 1]], y)
        for b in cols[self._blk[:, cols].any(axis=0)]:
            ks = np.flatnonzero(self._blk[:, b])
            seers = self.reach.indices[self.reach.indptr[b]:self.reach.indptr[b + 1]]
            self._seen_blocked[np.ix_(seers, ks)] -= 1
            self._blk[ks, b] = False
            self._left -= len(ks)

    def _rollout(self, k: int, x: int, goal: int, by_length: bool = False) -> float:
        """Cost of the optimistic policy (shortest ways at this query's costs, or by length) from
        ``x`` to ``goal`` in scenario ``k`` (inf: cut off)."""
        blk, seen_blocked = self._blk[k], self._seen_blocked[:, k]
        reach_ptr, reach_idx = self.reach.indptr, self.reach.indices
        pos, reward, been = self.road.positions, self._reward, {x}
        found: set[int] = set()  # scenario k's blocked nodes seen so far
        planned = frozenset()  # those its path has had to avoid
        d, nxt = self._tree(goal, planned, by_length)
        if not np.isfinite(d[x]):
            return np.inf
        y, cost = x, 0.0
        for _ in range(4 * self.road.n_nodes):
            if y == goal:
                return cost
            if seen_blocked[y]:
                new = [int(b) for b in reach_idx[reach_ptr[y]:reach_ptr[y + 1]] if blk[b] and b not in found]
                if new:
                    found.update(new)
                    while True:  # plan again until the path avoids what has been found
                        z = y
                        while z != goal and z not in found:
                            z = nxt[z]
                        if z == goal:
                            break
                        planned = planned | {z}
                        d, nxt = self._tree(goal, planned, by_length)
                        if not np.isfinite(d[y]):
                            return np.inf
            z = nxt[y]
            if reward is None:
                cost += d[y] - d[z]
            else:
                w = float(np.hypot(*(pos[z] - pos[y])))
                if z in been or z in self._passed:  # nothing left to collect there
                    cost += w
                else:
                    cost += max(w - float(reward[z]), MIN_COST_FRACTION * w)
                    been.add(z)
            y = z
        return np.inf

    def _expected(self, x: int, goal: int) -> float:
        out = np.inf
        for by_length in ((False, True) if self._reward is not None else (False,)):
            costs = np.array([self._rollout(k, x, goal, by_length) for k in range(self.samples)])
            finite = np.isfinite(costs)
            if finite.any():
                out = min(out, float(costs[finite].mean()))
        return out

    # ------------------------------------------------------------------ query
    def path(self, src: int, goal: int, p: np.ndarray,
             reward: Optional[np.ndarray] = None) -> tuple[list[int], float]:
        """The path from ``src`` to ``goal`` the ORO policy drives if it finds no blockage, and
        its expected cost (metres; less the reward collected) with the detours round the
        blockages it may find: that of its first move. ``p``: per node, the chance it is blocked;
        ``reward``: per node, what driving into it is worth (metres; module docstring).
        ``([], inf)`` if ``goal`` is out of reach."""
        if src == goal:
            return [src], 0.0
        self._goal, self._reward, self._passed = -1, reward, {src}
        self._data = self._lengths if reward is None else \
            np.maximum(self._lengths - reward[self._rows], MIN_COST_FRACTION * self._lengths)
        d0, nxt0 = self._tree(goal, frozenset())
        if not np.isfinite(d0[src]):
            return [], np.inf
        shortest = self._tree_path(src, goal, nxt0)
        if not (p[shortest] > 0).any() and not (self.reach[shortest] @ (p > 0)).any():
            return shortest, d0[src]  # no blockage to find on the shortest way: nothing does better
        self._limit = self.reach_factor * d0[src] + self.reach_margin_m  # for the rollouts' trees
        self._draw(p)
        self._reveal(src)
        expected = d0[src]
        path, visited, prev = [src], {src}, -1
        while path[-1] != goal:
            u = path[-1]
            if not self._left:  # nothing left that may be blocked: the shortest way
                path += self._tree_path(u, goal, nxt0)[1:]
                break
            cands = [(c + d0[v], v, c) for v, c in self._moves(u) if v != prev and np.isfinite(d0[v])]
            if len(cands) > 1 or u == src and cands:  # (from src also for its expected cost)
                best, best_q = None, np.inf
                for lb, v, w in sorted(cands):
                    if lb >= best_q:
                        break
                    q = w + self._expected(v, goal)
                    if q < best_q:
                        best, best_q = v, q
                v = best if best is not None else min(cands)[1]
                if u == src and best is not None:
                    expected = best_q
            else:
                v = cands[0][1] if cands else prev
            if v in visited or len(path) > 4 * self.road.n_nodes:  # (sampling noise) finish the shortest way
                path += self._tree_path(u, goal, nxt0)[1:]
                break
            path.append(v)
            visited.add(v)
            self._passed.add(v)
            self._reveal(v)
            prev = u
        return [int(n) for n in path], float(expected)

    def _moves(self, u: int) -> list[tuple[int, float]]:
        """(neighbour, cost of driving into it) from ``u``, at this query's costs."""
        if self._reward is None:
            return self.road.neighbors(u)
        return [(v, w if v in self._passed else max(w - float(self._reward[v]), MIN_COST_FRACTION * w))
                for v, w in self.road.neighbors(u)]
