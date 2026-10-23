"""Synthetic scenario generator: urban street grids, vegetation, rural roads, highways.

Terrain has three classes: URBAN, ROAD and VEGETATION (everything else, which the
UGV cannot drive on). Construction order:

1. Urban areas: rotated rectangles, each with a jittered street grid.
2. Roadless regions: smoothed noise thresholded to ``roadless_fraction`` of the map,
   kept clear of urban areas. Only highways cross them, so they become large
   stretches of vegetation that a UGV cannot get close to.
3. Highways: a minimum spanning tree between urban centres.
4. Rural roads: random points outside towns and roadless regions, joined by a
   relative-neighbourhood graph (plus some Gabriel edges for loops).
5. Merge near-duplicate nodes, subdivide long edges, keep the largest component,
   and rasterise the roads; every remaining non-urban cell is VEGETATION.
6. Optionally, a prior road map that differs from the true network (``MapMismatch``):
   stretches of road missing from the map (``road_unmapped_fraction``) and road nodes that
   cannot be traversed (``road_blocked_fraction``). The robots are given the prior map.

The UGV graph is the road network; the UAV graph is an 8-connected grid over the
whole map.

Prior (default ``prior_model="semantic"``): a lightweight proxy for LMPath
(Diller et al., 2026). LMPath asks a language model which context labels the target
object is found in (e.g. car -> road, parking lot), segments those labels in satellite
imagery, and normalises the summed masks. Here:

* the object -> labels lookup (``SEMANTIC_LABELS``) stands in for the language model;
  each label is a mask computed from the terrain and road graph (``label_mask``);
* the terrain map stands in for satellite segmentation, degraded by spatially
  correlated label errors (missed and spurious blobs) and a slight blur that mimics
  sliding-window mask averaging;
* ``H = sum of label masks / total``, mixed with an optional uniform floor (LMPath has
  none; a small floor keeps the Bayesian belief able to recover from missed labels).

The ground-truth target is placed uniformly at random on the *true* label cells
(not the degraded prior), restricted to cells a UGV can get within sensor range of
(otherwise the mission could never terminate). With a road-map mismatch, that is from a
mapped road the UGV can reach from its start without blocked nodes, so a mission can end
without the team finding any missing road. The prior (either model) is built from the road
map the robots are given: roads missing from it look like vegetation, there and in the
terrain the robots get (``Scenario.map_terrain``), though the target may be on one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from scipy.spatial import Delaunay, cKDTree

from marsupial_search.core import (
    N_LAYERS,
    DockingSpec,
    EnergySpec,
    GridSpec,
    Layer,
    RobotSpec,
    RobotType,
    SensorSpec,
    Terrain,
    TraversabilityGraph,
)

from .scenario import MapMismatch, RobotInit, Scenario


UAV_IDS = ("uav0", "uav1")


@dataclass
class FleetConfig:
    """The marsupial team: one UGV (``ugv0``) and two UAVs that share its dock. ``uav0``
    starts docked on the UGV; ``uav1`` starts airborne, hovering over it. The UAVs share a
    speed and battery; ``uav_sensor_ranges`` gives a UAV its own sensor range (e.g. a UAV
    flying lower sees a smaller footprint)."""
    ugv_speed: float = 5.0
    ugv_energy: EnergySpec = field(default_factory=lambda: EnergySpec(capacity=250_000, idle_power=2.0, energy_per_m=1.0))
    ugv_sensor_range: float = 55.0
    uav_speed: float = 10.0
    uav_energy: EnergySpec = field(default_factory=lambda: EnergySpec(capacity=2_000, idle_power=1.0, energy_per_m=0.04))
    uav_sensor_range: float = 60.0
    uav_sensor_ranges: dict[str, float] = field(default_factory=dict)  # per UAV (uav0, uav1): overrides uav_sensor_range

    def uav_range(self, rid: str) -> float:
        return self.uav_sensor_ranges.get(rid, self.uav_sensor_range)


@dataclass
class ScenarioConfig:
    name: str = "synthetic"
    size_m: tuple[float, float] = (5000.0, 5000.0)
    cell_m: float = 25.0
    # urban
    n_urban: int = 3
    urban_half_size_m: tuple[float, float] = (300.0, 600.0)
    urban_street_spacing_m: float = 100.0
    urban_street_drop_prob: float = 0.08
    urban_aerial_fraction: float = 0.5  # share of urban cell mass visible from the air
    # roadless regions (vegetation that only highways cross)
    roadless_fraction: float = 0.3
    roadless_smoothing_m: float = 250.0
    roadless_urban_buffer_m: float = 150.0
    # roads
    n_rural_nodes: int = 70
    rural_min_spacing_m: float = 250.0
    rural_max_edge_m: float = 1200.0
    rural_gabriel_extra_prob: float = 0.3
    highway_point_spacing_m: float = 100.0
    highway_wiggle_m: float = 150.0
    road_merge_m: float = 20.0
    road_node_spacing_m: Optional[float] = None  # long road edges are split into nodes this far apart (None: the UGV sensor range)
    # road-map mismatch: the robots are given a prior road map that differs from the true network
    road_unmapped_fraction: float = 0.0  # share of road nodes missing from the map (whole stretches of road)
    road_blocked_fraction: float = 0.0  # share of mapped road nodes that cannot be traversed
    # UAV grid
    uav_grid_spacing_m: Optional[float] = None  # None: √2 · the smallest UAV sensor range, the widest square grid whose footprints leave no gaps
    # prior
    prior_model: str = "semantic"  # "semantic" (LMPath-style) | "gaussian_mixture"
    target_object: str = "car"  # key into SEMANTIC_LABELS
    prior_labels: Optional[tuple[str, ...]] = None  # label names (see label_mask); overrides the object
    prior_segmentation_error: float = 0.1  # share of label area mislabelled (half missed, half spurious)
    prior_error_blob_m: float = 75.0  # spatial scale of segmentation errors
    prior_mask_blur_m: float = 25.0  # soft mask edges (sliding-window averaging)
    prior_background_weight: float = 0.02  # share of mass spread uniformly (LMPath: 0)
    # gaussian_mixture prior only
    n_prior_modes: int = 4
    prior_sigma_m: tuple[float, float] = (300.0, 900.0)
    prior_uniform_weight: float = 0.05
    fleet: FleetConfig = field(default_factory=FleetConfig)
    docking: DockingSpec = field(default_factory=DockingSpec)
    seed: int = 0
    target_seed: Optional[int] = None  # defaults to seed + 1


# Stand-in for LMPath's language-model step: the context labels each object is found in.
SEMANTIC_LABELS: dict[str, tuple[str, ...]] = {
    "car": ("road", "urban"),
    "building": ("urban",),
    # On a road next to vegetation, or in vegetation a UGV could see from the nearest road node.
    "hiker": ("road_near_vegetation", "vegetation_near_road_node"),
}
LABELS = ("road", "urban", "vegetation", "road_near_vegetation", "vegetation_near_road_node")


def label_mask(label: str, grid: GridSpec, terrain: np.ndarray, road: TraversabilityGraph,
               ugv_sensor_range: float) -> np.ndarray:
    """Boolean (H, W) mask of the cells carrying a semantic label."""
    veg = terrain == int(Terrain.VEGETATION)
    if label in ("road", "urban", "vegetation"):
        return terrain == int(Terrain[label.upper()])
    if label == "road_near_vegetation":
        return (terrain == int(Terrain.ROAD)) & binary_dilation(veg, structure=np.ones((3, 3), bool))
    if label == "vegetation_near_road_node":
        X, Y = grid.cell_centers()
        d, _ = cKDTree(road.positions).query(np.stack([X.ravel(), Y.ravel()], axis=1))
        return veg & (d.reshape(grid.shape) <= ugv_sensor_range)
    raise ValueError(f"unknown label {label!r}; choose from {LABELS}")

# ------------------------------------------------------------------- helpers


def _sample_segment(p: np.ndarray, q: np.ndarray, step: float) -> np.ndarray:
    n = max(int(np.ceil(np.linalg.norm(q - p) / step)), 1)
    t = np.linspace(0.0, 1.0, n + 1)[:, None]
    return p[None] * (1 - t) + q[None] * t


def _road_cells(grid: GridSpec, road: TraversabilityGraph, step: float) -> tuple[np.ndarray, np.ndarray]:
    """Row and column indices of the grid cells the edges of ``road`` pass through."""
    pts = [_sample_segment(road.positions[a], road.positions[b], step) for a, b in road.edges]
    return _cells_of(grid, np.concatenate(pts)) if pts else (np.zeros(0, int), np.zeros(0, int))


def _cells_of(grid: GridSpec, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = np.clip(((pts[:, 1] - grid.origin[1]) / grid.resolution).astype(int), 0, grid.rows - 1)
    c = np.clip(((pts[:, 0] - grid.origin[0]) / grid.resolution).astype(int), 0, grid.cols - 1)
    return r, c


class _RoadBuilder:
    def __init__(self):
        self.pos: list[np.ndarray] = []
        self.edges: list[tuple[int, int]] = []

    def add_node(self, p) -> int:
        self.pos.append(np.asarray(p, dtype=float))
        return len(self.pos) - 1

    def add_edge(self, a: int, b: int) -> None:
        if a != b:
            self.edges.append((a, b))


# ------------------------------------------------------------------ generator


def generate_scenario(cfg: Optional[ScenarioConfig] = None) -> Scenario:
    cfg = cfg or ScenarioConfig()
    unknown = set(cfg.fleet.uav_sensor_ranges) - set(UAV_IDS)
    if unknown:
        raise ValueError(f"fleet.uav_sensor_ranges: no UAV {sorted(unknown)}; the UAVs are {list(UAV_IDS)}")
    rng = np.random.default_rng(cfg.seed)
    W, H = cfg.size_m
    grid = GridSpec(origin=(0.0, 0.0), resolution=cfg.cell_m, shape=(int(round(H / cfg.cell_m)), int(round(W / cfg.cell_m))))
    X, Y = grid.cell_centers()
    terrain = np.full(grid.shape, int(Terrain.VEGETATION), dtype=np.int8)
    roads = _RoadBuilder()

    # 1. urban areas --------------------------------------------------------
    cities = []
    hmax = cfg.urban_half_size_m[1]
    margin = hmax + 200.0
    for _ in range(cfg.n_urban):
        for _attempt in range(500):
            c = rng.uniform([margin, margin], [W - margin, H - margin])
            if all(np.linalg.norm(c - o["center"]) > 2.2 * hmax for o in cities):
                break
        a, b = rng.uniform(*cfg.urban_half_size_m, size=2)
        th = rng.uniform(0, np.pi / 2)
        cities.append({"center": c, "half": (a, b), "theta": th})

    urban_mask = np.zeros(grid.shape, dtype=bool)
    street_nodes: list[list[int]] = []
    perimeter_nodes: list[int] = []
    s = cfg.urban_street_spacing_m
    for city in cities:
        (a, b), th, c = city["half"], city["theta"], city["center"]
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        du, dv = X - c[0], Y - c[1]
        u = du * np.cos(th) + dv * np.sin(th)
        v = -du * np.sin(th) + dv * np.cos(th)
        urban_mask |= (np.abs(u) <= a) & (np.abs(v) <= b)

        nu, nv = int(2 * a // s) + 1, int(2 * b // s) + 1
        us = np.linspace(-a, a, nu) if nu > 1 else np.array([0.0])
        vs = np.linspace(-b, b, nv) if nv > 1 else np.array([0.0])
        ids = np.zeros((nu, nv), dtype=int)
        mine = []
        for i, uu in enumerate(us):
            for j, vv in enumerate(vs):
                jitter = rng.normal(0, 0.06 * s, size=2)
                ids[i, j] = roads.add_node(c + R @ np.array([uu, vv]) + jitter)
                mine.append(ids[i, j])
                if i in (0, nu - 1) or j in (0, nv - 1):
                    perimeter_nodes.append(ids[i, j])
        for i in range(nu):
            for j in range(nv):
                for di, dj in ((1, 0), (0, 1)):
                    if i + di < nu and j + dj < nv and rng.random() >= cfg.urban_street_drop_prob:
                        roads.add_edge(ids[i, j], ids[i + di, j + dj])
        city["nodes"] = mine
        street_nodes.append(mine)
    terrain[urban_mask] = int(Terrain.URBAN)

    # 2. roadless regions ---------------------------------------------------
    noise = gaussian_filter(rng.standard_normal(grid.shape), sigma=cfg.roadless_smoothing_m / cfg.cell_m)
    buf = max(int(round(cfg.roadless_urban_buffer_m / cfg.cell_m)), 0)
    excluded = binary_dilation(urban_mask, iterations=buf) if buf else urban_mask.copy()
    avail = ~excluded
    roadless = np.zeros(grid.shape, dtype=bool)
    n_roadless = int(cfg.roadless_fraction * terrain.size)
    if n_roadless > 0 and avail.any():
        thresh = np.sort(noise[avail])[::-1][min(n_roadless, int(avail.sum())) - 1]
        roadless = avail & (noise >= thresh)

    # 3. highways (MST between city centres) --------------------------------
    highway_nodes: list[int] = []
    if len(cities) > 1:
        centers = np.array([ci["center"] for ci in cities])
        D = np.linalg.norm(centers[:, None] - centers[None], axis=-1)
        mst = minimum_spanning_tree(D).tocoo()
        street_tree = cKDTree(np.array([roads.pos[n] for n in perimeter_nodes]))
        for ia, ib in zip(mst.row, mst.col):
            p, q = centers[ia], centers[ib]
            L = np.linalg.norm(q - p)
            n = max(int(L // cfg.highway_point_spacing_m), 2)
            t = np.linspace(0, 1, n + 1)
            normal = np.array([-(q - p)[1], (q - p)[0]]) / L
            amp = rng.uniform(-1, 1) * cfg.highway_wiggle_m
            pts = p[None] * (1 - t[:, None]) + q[None] * t[:, None]
            pts += normal[None] * (amp * np.sin(np.pi * t) + rng.normal(0, 10, size=t.shape))[:, None]
            r, c = _cells_of(grid, pts)
            outside = ~urban_mask[r, c]
            # Split into runs outside urban areas; attach each run's ends to streets.
            run: list[int] = []
            for k in range(len(pts) + 1):
                if k < len(pts) and outside[k]:
                    run.append(k)
                    continue
                if len(run) >= 1:
                    ids = [roads.add_node(pts[k2]) for k2 in run]
                    highway_nodes.extend(ids)
                    for a_, b_ in zip(ids, ids[1:]):
                        roads.add_edge(a_, b_)
                    for end in (ids[0], ids[-1]):
                        d, j = street_tree.query(roads.pos[end])
                        if d < 3 * cfg.highway_point_spacing_m:
                            roads.add_edge(end, perimeter_nodes[j])
                run = []

    highway_set = set(highway_nodes)

    # 4. rural roads --------------------------------------------------------
    open_cells = np.argwhere(~urban_mask & ~roadless)
    rural: list[np.ndarray] = []
    for _ in range(cfg.n_rural_nodes * 30):
        if len(rural) >= cfg.n_rural_nodes or len(open_cells) == 0:
            break
        rr, cc = open_cells[rng.integers(len(open_cells))]
        p = np.array(grid.cell_center(rr, cc)) + rng.uniform(-0.5, 0.5, 2) * cfg.cell_m
        if all(np.linalg.norm(p - o) >= cfg.rural_min_spacing_m for o in rural):
            rural.append(p)
    rural_ids = [roads.add_node(p) for p in rural]
    hub_ids = rural_ids + highway_nodes[::3] + perimeter_nodes
    hub_ids = list(dict.fromkeys(hub_ids))
    perim_set = set(perimeter_nodes)
    if len(hub_ids) >= 4:
        P = np.array([roads.pos[i] for i in hub_ids])
        tri = Delaunay(P)
        cand = set()
        for simplex in tri.simplices:
            for k in range(3):
                i, j = sorted((simplex[k], simplex[(k + 1) % 3]))
                cand.add((i, j))
        tree = cKDTree(P)
        for i, j in cand:
            gi, gj = hub_ids[i], hub_ids[j]
            if (gi in perim_set and gj in perim_set) or (gi in highway_set and gj in highway_set):
                continue
            p, q = P[i], P[j]
            d = np.linalg.norm(q - p)
            if d > cfg.rural_max_edge_m or d < 1e-6:
                continue
            seg = _sample_segment(p, q, cfg.cell_m / 2)
            interior = seg[(np.linalg.norm(seg - p, axis=1) > 40) & (np.linalg.norm(seg - q, axis=1) > 40)]
            if len(interior):
                r, c = _cells_of(grid, interior)
                if roadless[r, c].any() or urban_mask[r, c].any():
                    continue
            # Relative neighbourhood graph test (sparse, road-like) ...
            near = tree.query_ball_point(p, d)
            rng_ok = not any(
                k not in (i, j) and max(np.linalg.norm(P[k] - p), np.linalg.norm(P[k] - q)) < d for k in near
            )
            if not rng_ok:
                # ... with some extra Gabriel edges to create loops.
                mid = (p + q) / 2
                gab = [k for k in tree.query_ball_point(mid, d / 2 - 1e-6) if k not in (i, j)]
                if gab or rng.random() >= cfg.rural_gabriel_extra_prob:
                    continue
            roads.add_edge(gi, gj)

    # 5. clean up -----------------------------------------------------------
    road_graph = _finalize_roads(roads, cfg)

    # Rasterise roads outside towns; everything else non-urban stays VEGETATION.
    r, c = _road_cells(grid, road_graph, cfg.cell_m / 4)
    keep = terrain[r, c] != int(Terrain.URBAN)
    terrain[r[keep], c[keep]] = int(Terrain.ROAD)

    # 6. the road map the robots get (its own random stream: the rest of the map is unchanged)
    full_road = road_graph
    road_graph, mismatch = _map_mismatch(full_road, cfg, np.random.default_rng([cfg.seed, 2]))
    reachable = road_graph.with_changes(blocked=mismatch.blocked)  # what a UGV can drive without the missing roads
    # The terrain as the robots know it, and the prior is built from: roads missing from the map look like vegetation.
    map_terrain = terrain
    if mismatch.new_nodes:
        mapped = np.zeros(grid.shape, dtype=bool)
        mapped[_road_cells(grid, road_graph, cfg.cell_m / 4)] = True
        map_terrain = terrain.copy()
        map_terrain[(terrain == int(Terrain.ROAD)) & ~mapped] = int(Terrain.VEGETATION)

    uav_graph = _grid_graph(cfg.size_m, uav_grid_spacing(cfg))

    # Layers, prior, target -------------------------------------------------
    layer_fractions = np.zeros((N_LAYERS,) + grid.shape)
    urban = terrain == int(Terrain.URBAN)
    layer_fractions[Layer.SHARED][~urban] = 1.0
    layer_fractions[Layer.AERIAL][urban] = cfg.urban_aerial_fraction
    layer_fractions[Layer.GROUND][urban] = 1.0 - cfg.urban_aerial_fraction

    if cfg.prior_model == "semantic":
        labels = cfg.prior_labels or SEMANTIC_LABELS[cfg.target_object]

        def labelled(terrain_, road_):
            return np.logical_or.reduce([label_mask(lab, grid, terrain_, road_, cfg.fleet.ugv_sensor_range)
                                         for lab in labels])

        truth = labelled(terrain, full_road)
        if not truth.any():
            raise ValueError(f"no cells carry any of the labels {labels} on this map")
        known = labelled(map_terrain, road_graph) if map_terrain is not terrain else truth  # from the road map
        prior = _semantic_prior(grid, known, cfg, rng)
        target_dist = truth.astype(float)  # uniform over true label cells
    elif cfg.prior_model == "gaussian_mixture":
        prior = _gaussian_mixture_prior(grid, map_terrain, cfg, rng)
        truth, target_dist = None, prior
    else:
        raise ValueError(f"unknown prior_model {cfg.prior_model!r}")

    robots, init = _make_fleet(cfg, reachable, terrain, grid, rng)

    confirm_mask = _confirmable_mask(grid, reachable, cfg.fleet.ugv_sensor_range, init["ugv0"].node)
    if not (target_dist * confirm_mask).any():
        raise ValueError("no cell is both a plausible target location and within UGV sensor range of a road")
    trng = np.random.default_rng(cfg.seed + 1 if cfg.target_seed is None else cfg.target_seed)
    target_xy, target_layer = sample_target(grid, target_dist, layer_fractions, confirm_mask, trng)

    meta = {
        "config": cfg,
        "confirmable_mask": confirm_mask,
        "confirmable_prior_mass": float(prior[confirm_mask].sum()),
        "target_distribution": target_dist,  # what the hidden target is drawn from (x confirmable)
        "label_mask": truth,  # true semantic label cells (semantic prior only)
        "road_length_m": float(full_road.lengths.sum()),
        "road_unmapped_nodes": len(mismatch.new_nodes),
        "road_blocked_nodes": len(mismatch.blocked),
        "cities": [{k: v for k, v in c.items() if k != "nodes"} for c in cities],
        "terrain_fractions": {t.name.lower(): float((terrain == int(t)).mean()) for t in Terrain},
    }
    return Scenario(
        name=f"{cfg.name}_s{cfg.seed}",
        grid=grid,
        terrain=terrain,
        layer_fractions=layer_fractions,
        prior=prior,
        graphs={RobotType.UGV: road_graph, RobotType.UAV: uav_graph},
        robots=robots,
        init=init,
        docking=cfg.docking,
        target_xy=target_xy,
        target_layer=target_layer,
        meta=meta,
        road_mismatch=mismatch,
        map_terrain=None if map_terrain is terrain else map_terrain,
    )


def road_node_spacing(cfg: ScenarioConfig) -> float:
    """Road node spacing: each node lies on the edge of the next one's UGV footprint."""
    return cfg.road_node_spacing_m or cfg.fleet.ugv_sensor_range


def uav_grid_spacing(cfg: ScenarioConfig) -> float:
    """UAV grid spacing: the footprints of a square grid's nodes just cover the plane (for every UAV)."""
    return cfg.uav_grid_spacing_m or np.sqrt(2.0) * min(cfg.fleet.uav_range(rid) for rid in UAV_IDS)


def _finalize_roads(roads: _RoadBuilder, cfg: ScenarioConfig) -> TraversabilityGraph:
    pos = np.array(roads.pos)
    edges = np.array(roads.edges, dtype=int).reshape(-1, 2)

    # Merge nodes closer than road_merge_m (union-find).
    parent = np.arange(len(pos))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in cKDTree(pos).query_pairs(cfg.road_merge_m):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    roots = np.array([find(i) for i in range(len(pos))])
    uniq, remap = np.unique(roots, return_inverse=True)
    pos = pos[uniq]
    edges = remap[edges]
    edges = edges[edges[:, 0] != edges[:, 1]]
    edges = np.unique(np.sort(edges, axis=1), axis=0)

    # Subdivide long edges.
    new_pos = list(pos)
    new_edges = []
    for a, b in edges:
        L = np.linalg.norm(pos[b] - pos[a])
        n = int(np.ceil(L / road_node_spacing(cfg)))
        prev = a
        for k in range(1, n):
            new_pos.append(pos[a] + (pos[b] - pos[a]) * k / n)
            cur = len(new_pos) - 1
            new_edges.append((prev, cur))
            prev = cur
        new_edges.append((prev, b))
    pos = np.array(new_pos)
    edges = np.array(new_edges, dtype=int)

    # Keep the largest connected component.
    g = TraversabilityGraph(pos, edges)
    ncomp, labels = connected_components(g.csr, directed=False)
    if ncomp > 1:
        big = np.argmax(np.bincount(labels))
        keep = np.flatnonzero(labels == big)
        idx = -np.ones(len(pos), dtype=int)
        idx[keep] = np.arange(len(keep))
        e = idx[edges]
        e = e[(e >= 0).all(axis=1)]
        pos = pos[keep]
        edges = e
    return TraversabilityGraph(pos, edges, name="road")


def _road_chains(road: TraversabilityGraph) -> list[tuple[list[int], bool]]:
    """Stretches of road between junctions: (inner nodes, dead end), where a dead end's last inner
    node is the leaf it ends at. Only stretches with an inner node."""
    deg = np.diff(road.csr.indptr)
    chains, seen = [], set()
    for end in np.flatnonzero(deg > 2):
        for nb, _ in road.neighbors(int(end)):
            prev, cur, inner = int(end), nb, []
            while deg[cur] == 2 and cur != end:
                inner.append(cur)
                prev, cur = cur, next(n for n, _ in road.neighbors(cur) if n != prev)
            dead = bool(deg[cur] == 1)
            if dead:
                inner.append(cur)
            if inner and frozenset(inner) not in seen:
                seen.add(frozenset(inner))
                chains.append((inner, dead))
    return chains


def _map_mismatch(road: TraversabilityGraph, cfg: ScenarioConfig, rng) -> tuple[TraversabilityGraph, MapMismatch]:
    """Split the true road network into the prior map the robots get and its mismatch with the world.

    Unmapped: whole stretches of road between junctions (or out to a dead end) are left off the
    map, but never one that the map needs to stay connected. Blocked: random mapped nodes whose
    loss leaves the map connected, so a blockage forces a detour but every other mapped node
    stays reachable. Mapped nodes keep their order; the unmapped ones get the indices after them."""
    n = road.n_nodes
    hidden = np.zeros(n, dtype=bool)
    target = int(round(cfg.road_unmapped_fraction * n))
    if target > 0:
        chains = _road_chains(road)
        for k in rng.permutation(len(chains)):
            if hidden.sum() >= target:
                break
            inner, dead = chains[k]
            if hidden[inner].any():
                continue
            hidden[inner] = True
            if not dead and road.with_changes(blocked=np.flatnonzero(hidden)).n_components() > 1 + hidden.sum():
                hidden[inner] = False  # it was the only road between two parts of the map
    mapped = np.flatnonzero(~hidden)
    n_blocked = int(round(cfg.road_blocked_fraction * len(mapped)))
    if not hidden.any() and n_blocked == 0:
        return road, MapMismatch()
    order = np.concatenate([mapped, np.flatnonzero(hidden)])
    index = np.empty(n, dtype=int)
    index[order] = np.arange(n)
    e = index[road.edges]
    prior = TraversabilityGraph(road.positions[mapped], e[(e < len(mapped)).all(axis=1)], name=road.name)
    full = TraversabilityGraph(road.positions[order], e).csr
    open_ = np.ones(n, dtype=bool)
    for b in rng.permutation(prior.n_nodes) if n_blocked else []:
        if open_.sum() <= n - n_blocked:
            break
        open_[b] = False
        m = open_[:prior.n_nodes]
        if connected_components(prior.csr[m][:, m], directed=False)[0] > 1 \
                or connected_components(full[open_][:, open_], directed=False)[0] > 1:
            open_[b] = True  # the only way past it, on the map or on the ground
    return prior, MapMismatch(
        blocked=tuple(int(b) for b in np.flatnonzero(~open_)),
        new_nodes={int(index[h]): tuple(map(float, road.positions[h])) for h in np.flatnonzero(hidden)},
        new_edges=tuple((int(a), int(b)) for a, b in np.sort(e[(e >= len(mapped)).any(axis=1)], axis=1)),
    )


def _grid_graph(size_m: tuple[float, float], spacing: float) -> TraversabilityGraph:
    W, H = size_m

    def axis(L):  # centred so both map borders get the same margin
        n = int(np.ceil(L / spacing))
        return (L - (n - 1) * spacing) / 2 + spacing * np.arange(n)

    xs, ys = axis(W), axis(H)
    nx_, ny_ = len(xs), len(ys)
    XX, YY = np.meshgrid(xs, ys)
    pos = np.stack([XX.ravel(), YY.ravel()], axis=1)
    idx = np.arange(nx_ * ny_).reshape(ny_, nx_)
    edges = []
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        r0, r1 = max(0, -dr), ny_ - max(0, dr)
        c0, c1 = max(0, -dc), nx_ - max(0, dc)
        a = idx[r0:r1, c0:c1]
        b = idx[r0 + dr : r1 + dr, c0 + dc : c1 + dc]
        edges.append(np.stack([a.ravel(), b.ravel()], axis=1))
    return TraversabilityGraph(pos, np.concatenate(edges), name="uav_grid")


def _semantic_prior(grid: GridSpec, truth: np.ndarray, cfg: ScenarioConfig, rng) -> np.ndarray:
    """LMPath-style prior: the true label mask, imperfectly "segmented", then normalised."""
    seg = truth.astype(float)
    n_err = int(0.5 * cfg.prior_segmentation_error * truth.sum())
    if n_err > 0:
        # Spatially correlated errors: low-noise label blobs are missed, high-noise
        # background blobs are spuriously labelled.
        noise = gaussian_filter(rng.standard_normal(grid.shape), sigma=cfg.prior_error_blob_m / cfg.cell_m)
        lab, bg = noise[truth], noise[~truth]
        if n_err < lab.size:
            seg[truth & (noise < np.partition(lab, n_err)[n_err])] = 0.0
        if 0 < n_err < bg.size:
            seg[~truth & (noise > -np.partition(-bg, n_err)[n_err])] = 1.0
    if cfg.prior_mask_blur_m > 0:
        seg = gaussian_filter(seg, sigma=cfg.prior_mask_blur_m / cfg.cell_m)

    H = seg / seg.sum()
    prior = (1 - cfg.prior_background_weight) * H + cfg.prior_background_weight / H.size
    return prior / prior.sum()


def _gaussian_mixture_prior(grid: GridSpec, terrain: np.ndarray, cfg: ScenarioConfig, rng) -> np.ndarray:
    X, Y = grid.cell_centers()
    k = max(cfg.n_prior_modes, 1)
    centers = []
    for t in (Terrain.URBAN, Terrain.VEGETATION):
        cells = np.argwhere(terrain == int(t))
        if len(cells) and len(centers) < k:
            r, c = cells[rng.integers(len(cells))]
            centers.append(grid.cell_center(r, c))
    while len(centers) < k:
        centers.append((rng.uniform(0, cfg.size_m[0]), rng.uniform(0, cfg.size_m[1])))
    weights = rng.dirichlet(np.ones(k) * 2.0)
    prior = np.zeros(grid.shape)
    for (cx, cy), w in zip(centers, weights):
        sx, sy = rng.uniform(*cfg.prior_sigma_m, size=2)
        g = np.exp(-0.5 * (((X - cx) / sx) ** 2 + ((Y - cy) / sy) ** 2))
        prior += w * g / g.sum()
    prior = (1 - cfg.prior_uniform_weight) * prior + cfg.prior_uniform_weight / prior.size
    return prior / prior.sum()


def _confirmable_mask(grid: GridSpec, road: TraversabilityGraph, sensor_range: float,
                      start: Optional[int] = None) -> np.ndarray:
    """Cells whose centre a UGV driving the road network (from ``start``, if given) can get in sensor range of."""
    keep = ~road.blocked
    if start is not None:
        _, comp = connected_components(road.csr, directed=False)
        keep &= comp == comp[start]
    pts = [road.positions[keep]]
    for a, b in road.edges[keep[road.edges[:, 0]]]:
        pts.append(_sample_segment(road.positions[a], road.positions[b], 10.0)[1:-1])
    tree = cKDTree(np.concatenate(pts))
    X, Y = grid.cell_centers()
    d, _ = tree.query(np.stack([X.ravel(), Y.ravel()], axis=1), distance_upper_bound=sensor_range + 1e-6)
    return (d <= sensor_range).reshape(grid.shape)


def sample_target(grid, dist, layer_fractions, confirm_mask, rng) -> tuple[tuple[float, float], Layer]:
    """Draw a target cell from ``dist`` restricted to ``confirm_mask``, then its visibility layer."""
    p = (dist * confirm_mask).ravel()
    idx = rng.choice(p.size, p=p / p.sum())
    r, c = divmod(int(idx), grid.cols)
    lf = layer_fractions[:, r, c]
    layer = Layer(int(rng.choice(N_LAYERS, p=lf / lf.sum())))
    return grid.cell_center(r, c), layer


def _make_fleet(cfg: ScenarioConfig, road: TraversabilityGraph, terrain, grid, rng):
    f = cfg.fleet
    # Depot: a random non-urban road node that is not blocked.
    r, c = _cells_of(grid, road.positions)
    rural = np.flatnonzero((terrain[r, c] != int(Terrain.URBAN)) & ~road.blocked)
    depot = int(rng.choice(rural)) if len(rural) else 0
    ugv_sensor = SensorSpec(range_m=f.ugv_sensor_range, visible_layers=(Layer.SHARED, Layer.GROUND))
    robots = {"ugv0": RobotSpec("ugv0", RobotType.UGV, f.ugv_speed, f.ugv_energy, ugv_sensor)}
    for rid in UAV_IDS:
        uav_sensor = SensorSpec(range_m=f.uav_range(rid), visible_layers=(Layer.SHARED, Layer.AERIAL))
        robots[rid] = RobotSpec(rid, RobotType.UAV, f.uav_speed, f.uav_energy, uav_sensor)
    init = {"ugv0": RobotInit(node=depot), "uav0": RobotInit(docked_on="ugv0"),
            "uav1": RobotInit(position=tuple(road.positions[depot]))}  # the dock takes one UAV at a time
    return robots, init


def resample_target(scenario: Scenario, rng) -> Scenario:
    """Copy of ``scenario`` with a new hidden target drawn the same way the generator does."""
    m = scenario.meta
    xy, layer = sample_target(scenario.grid, m["target_distribution"], scenario.layer_fractions,
                              m["confirmable_mask"], rng)
    return scenario.with_target(xy, layer)
