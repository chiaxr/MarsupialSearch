"""Road-map mismatch: the robots get a prior road map; the true roads have blocked nodes and roads missing from it."""

import dataclasses

import numpy as np
import pytest
from scipy.sparse.csgraph import connected_components

from conftest import make_line_scenario
from marsupial_search.controllers import DEFAULT_TEAM, make_team
from marsupial_search.controllers.common import bypass
from marsupial_search.core import EventType, FollowPath, GraphKnowledge, RobotType, Terrain, TraversabilityGraph
from marsupial_search.sim import MapMismatch, SimConfig, Simulator, generate_scenario, resample_target
from marsupial_search.sim.scenario_bases import small


def _ladder() -> TraversabilityGraph:
    # 0 - 1 - 2 - 3
    # |   |   |   |
    # 4 - 5 - 6 - 7
    pos = np.array([[x, y] for y in (1, 0) for x in range(4)], float)
    edges = [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    return TraversabilityGraph(pos, np.array(edges))


def test_blocked_nodes_keep_their_index_and_new_nodes_are_appended():
    g = _ladder().with_changes(blocked=[1], new_nodes={9: (4.0, 1.0)}, new_edges=[(3, 9)])
    assert g.n_nodes == 10 and g.neighbors(1) == [] and not g.has_edge(0, 1)
    assert g.blocked[1] and g.blocked[8] and np.isnan(g.positions[8]).all()  # 8: a placeholder, not yet known
    path, d = g.shortest_path(0, 9)
    assert g.has_edge(3, 9) and path[:3] == [0, 4, 5] and path[-2:] == [3, 9] and d == pytest.approx(6.0)
    assert g.nearest_node((1.0, 1.1)) != 1 and 1 not in g.nodes_within((1.0, 1.0), 0.5)
    assert list(g.nearest_nodes(np.array([[1.0, 1.0], [4.0, 1.0]]))) in ([0, 9], [2, 9])


def test_graph_knowledge_accumulates_and_reports_what_another_lacks():
    prior = _ladder()
    a, b = GraphKnowledge(prior), GraphKnowledge(prior)
    assert not a and a.graph is prior
    assert a.add(blocked=[1], new_nodes={8: (4.0, 1.0)}, new_edges=[(8, 3)])
    assert not a.add(blocked=[1])  # nothing new
    assert a.graph.n_nodes == 9 and a.graph.has_edge(3, 8) and not a.graph.has_edge(0, 1)
    b.add(blocked=[1])
    assert a.changes(exclude=b) == {"blocked": [], "new_nodes": {8: (4.0, 1.0)}, "new_edges": [(3, 8)]}
    b.add(**a.changes())
    assert b.changes() == a.changes()


def test_bypass_goes_round_blocked_nodes_and_keeps_stops():
    g = _ladder().with_changes(blocked=[2])
    route, index = bypass(g, [0, 1, 2, 3, 7])
    assert route == [0, 1, 5, 6, 7] and list(index) == [0, 1, -1, -1, 4]
    # Not skipping a kept index: it rejoins at 3 even though going straight to 7 would be shorter.
    route, index = bypass(g, [0, 1, 2, 3, 7], keep=[3], budget=0.0)
    assert route[-2:] == [3, 7] and index[3] >= 0


def test_bypass_takes_a_planned_way_only_if_the_budget_allows_it():
    g = _ladder().with_changes(blocked=[2])
    def roundabout(a, b):  # a planner's way from 1 that goes round by 0 and 4: 2 longer than the shortest
        rest, d = g.shortest_path(4, b)
        return [1, 0] + rest, 2.0 + d
    # Budget 0: it rejoins at 7 (no longer than planned), and the roundabout way would be 2 longer.
    assert bypass(g, [0, 1, 2, 3, 7], budget=0.0, path_fn=roundabout)[0] == [0, 1, 5, 6, 7]
    # Budget 4: it rejoins at 3 (2 longer), and the roundabout way (2 longer still) fits what is left.
    assert bypass(g, [0, 1, 2, 3, 7], budget=4.0, path_fn=roundabout)[0] == [0, 1, 0, 4, 5, 6, 7, 3, 7]
    assert bypass(g, [0, 1, 2, 3, 7], budget=3.0, path_fn=roundabout)[0] == [0, 1, 5, 6, 7, 3, 7]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_generated_mismatch_keeps_every_road_reachable_and_the_target_confirmable(seed):
    cfg = dataclasses.replace(small(seed), road_unmapped_fraction=0.1, road_blocked_fraction=0.1)
    sc = generate_scenario(cfg)
    prior, mm = sc.graphs[RobotType.UGV], sc.road_mismatch
    assert len(mm.blocked) == round(0.1 * prior.n_nodes) and len(mm.new_nodes) >= round(0.09 * sc.true_road().n_nodes)
    assert min(mm.new_nodes) == prior.n_nodes and set(mm.new_nodes) == set(range(prior.n_nodes, max(mm.new_nodes) + 1))
    depot = sc.init["ugv0"].node
    assert depot not in mm.blocked
    for g in (prior, prior.with_changes(blocked=mm.blocked), sc.true_road()):  # one piece, blocked nodes aside
        assert connected_components(g.csr[~g.blocked][:, ~g.blocked], directed=False)[0] == 1
    # The target is within UGV sensor range of a mapped, passable road.
    passable = prior.with_changes(blocked=mm.blocked)
    pts = np.concatenate([np.linspace(passable.positions[a], passable.positions[b], 20) for a, b in passable.edges])
    rng = sc.robots["ugv0"].sensor.range_m
    assert np.hypot(*(pts - sc.target_xy).T).min() <= rng
    # The robots are only given the prior map.
    assert Simulator(sc).setup.graphs[RobotType.UGV] is prior


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_blocked_nodes_cut_off_no_more_than_road_cutoff_fraction(seed):
    cfg = dataclasses.replace(small(seed), road_unmapped_fraction=0.1, road_blocked_fraction=0.1,
                              road_cutoff_fraction=0.1)
    sc = generate_scenario(cfg)
    prior, mm = sc.graphs[RobotType.UGV], sc.road_mismatch
    true_road = sc.true_road()
    for g in (prior.with_changes(blocked=mm.blocked), true_road):
        open_ = ~g.blocked
        _, comp = connected_components(g.csr[open_][:, open_], directed=False)
        assert open_.sum() - np.bincount(comp).max() <= 0.1 * g.n_nodes
    reach = prior.with_changes(blocked=mm.blocked)
    _, comp = connected_components(reach.csr, directed=False)
    open_ = ~reach.blocked
    main = np.bincount(comp[open_]).argmax()
    assert sc.meta["road_cutoff_nodes"] == int((open_ & (comp != main)).sum())
    assert comp[sc.init["ugv0"].node] == main  # the UGV starts in the main part, and the target is within its reach
    pts = np.concatenate([np.linspace(reach.positions[a], reach.positions[b], 20) for a, b in reach.edges
                          if comp[a] == main])
    assert np.hypot(*(pts - sc.target_xy).T).min() <= sc.robots["ugv0"].sensor.range_m


def test_some_maps_get_cut_off_regions():
    cut = [generate_scenario(dataclasses.replace(small(s), road_blocked_fraction=0.05, road_cutoff_fraction=0.1))
           .meta["road_cutoff_nodes"] for s in range(4)]
    assert max(cut) > 0
    assert [generate_scenario(dataclasses.replace(small(s), road_blocked_fraction=0.05)).meta["road_cutoff_nodes"]
            for s in range(4)] == [0] * 4


@pytest.mark.parametrize("seed", [0, 3])
def test_the_prior_and_the_robots_terrain_leave_out_missing_roads(seed):
    # No segmentation errors or blur: a car's prior is the road label plus the uniform background.
    cfg = dataclasses.replace(small(seed), target_object="car", road_unmapped_fraction=0.1,
                              prior_segmentation_error=0.0, prior_mask_blur_m=0.0)
    sc = generate_scenario(cfg)
    road = int(Terrain.ROAD)
    missing = (sc.terrain == road) & (sc.map_terrain != road)  # road cells only on missing roads
    assert missing.any() and (sc.map_terrain[sc.terrain != road] == sc.terrain[sc.terrain != road]).all()
    assert not (Simulator(sc).setup.terrain[missing] == road).any()
    background = sc.prior.min()
    assert np.allclose(sc.prior[missing], background) and (sc.prior[sc.map_terrain == road] > 2 * background).all()


def test_no_mismatch_by_default():
    sc = generate_scenario(small(0))
    assert not sc.road_mismatch and sc.true_road().n_nodes == sc.graphs[RobotType.UGV].n_nodes
    assert sc.map_terrain is None and Simulator(sc).setup.terrain.tolist() == sc.terrain.tolist()


def _mismatch_line():
    """The line scenario (road nodes 0-19, 50 m apart along y=500) with node 5 blocked and a road
    missing from the map: 20 at (125, 550) and 21 at (125, 600), off node 2."""
    sc = make_line_scenario()
    mm = MapMismatch(blocked=(5,), new_nodes={20: (125.0, 550.0), 21: (125.0, 600.0)},
                     new_edges=((2, 20), (20, 21)))
    return dataclasses.replace(sc, road_mismatch=mm)


def _run(sim, cmds, n):
    out = []
    for _ in range(n):
        out.append(sim.step(cmds))
        cmds = {}
    return out


def test_ugv_stops_before_a_blocked_node_and_finds_it():
    sim = Simulator(_mismatch_line(), SimConfig(target=False))
    obs = _run(sim, {"ugv0": FollowPath(tuple(range(9)))}, 60)
    blocked = [e for o in obs for e in o.events if e.type == EventType.PATH_BLOCKED]
    assert len(blocked) == 1 and blocked[0].robot_id == "ugv0"
    s = obs[-1].robots["ugv0"]
    assert s.node == 4 and not s.moving and s.path_remaining == (5, 6, 7, 8)  # waits there, path pending
    found = [u for o in obs for u in o.map_updates if u.robot_id == "ugv0" and u.blocked]
    assert [u.blocked for u in found] == [(5,)] and found[0].time <= blocked[0].time  # seen from 50 m away
    assert sim.result().robot_stats["ugv0"]["path_blocked"] == 1
    obs = _run(sim, {"ugv0": FollowPath((4, 3))}, 15)  # a new command replaces the pending path
    assert obs[-1].robots["ugv0"].node == 3


def test_robots_find_missing_roads_and_the_ugv_can_drive_them():
    sim = Simulator(_mismatch_line(), SimConfig(target=False))
    obs = sim.reset()
    assert sim.setup.graphs[RobotType.UGV].n_nodes == 20  # the map lacks 20 and 21
    obs = _run(sim, {"ugv0": FollowPath((0, 1, 2, 20, 21))}, 40)
    ups = [u for o in obs for u in o.map_updates if u.robot_id == "ugv0"]
    assert ups[0].new_nodes == {20: (125.0, 550.0), 21: (125.0, 600.0)} and ups[0].new_edges == ((2, 20), (20, 21))
    assert obs[-1].robots["ugv0"].node == 21
    assert sim.result().robot_stats["ugv0"]["map_changes_found"] == 2  # each node is found once per robot


@pytest.fixture(scope="module")
def mismatch_scenario():
    return generate_scenario(dataclasses.replace(small(5), road_unmapped_fraction=0.1, road_blocked_fraction=0.05))


def test_uavs_and_the_ugv_share_what_they_find(mismatch_scenario):
    team = make_team(DEFAULT_TEAM)
    found: dict[str, set] = {"ugv": set(), "uav": set()}
    ugv_found_at: dict[int, float] = {}

    def watch(sim, obs):
        for u in obs.map_updates:
            found["ugv" if u.robot_id == "ugv0" else "uav"].update(u.blocked)
            if u.robot_id == "ugv0":
                ugv_found_at.update({b: u.time for b in u.blocked})

    res = Simulator(mismatch_scenario, SimConfig()).run(team, callback=watch)
    assert res.outcome == "found"
    only_uav = found["uav"] - found["ugv"]
    assert only_uav and only_uav <= team.robots["ugv0"].roads.blocked  # reported to the UGV
    # The UGV's plan carries what it knows, so the UAVs plan on it (and stop reporting what it has).
    known = {b for b, t in ugv_found_at.items() if t < res.time_s - 10.0}
    assert known
    for uav in ("uav0", "uav1"):
        assert known <= team.robots[uav].roads.blocked and known <= team.robots[uav].ugv_roads.blocked


@pytest.mark.parametrize("team", ["joint_planner", "decentralised"])
def test_teams_find_the_target_despite_a_wrong_map(mismatch_scenario, team):
    res = Simulator(mismatch_scenario, SimConfig()).run(make_team(DEFAULT_TEAM if team == "decentralised" else team))
    s = res.summary()
    assert res.outcome == "found" and s["depleted"] == 0 and s["rejected"] == 0
    assert s["map_changes_found"] > 0


# --------------------------------------------------------------------------- recovery
def _ladder_scenario():
    """The line scenario with a second road 50 m above it (y=550, nodes 20-39), joined by a rung at every node."""
    sc = make_line_scenario()
    top = sc.graphs[RobotType.UGV]
    pos = np.concatenate([top.positions, top.positions + [0.0, 50.0]])
    rungs = np.stack([np.arange(20), np.arange(20, 40)], 1)
    edges = np.concatenate([top.edges, top.edges + 20, rungs])
    return dataclasses.replace(sc, graphs={**sc.graphs, RobotType.UGV: TraversabilityGraph(pos, edges, "road")})


def _ugv_on(route, radio_m, now=0.0):
    """A ``ugv_timed_route`` controller at the start of ``route``, with ``radio_m`` radio range."""
    from marsupial_search.controllers.ugv_timed_route import TimedRouteUgv
    from marsupial_search.core import RobotMode, RobotState
    from marsupial_search.sim.radio import RadioConfig
    sim = Simulator(_ladder_scenario(), SimConfig(radio=RadioConfig(range_m={t: radio_m for t in RobotType})))
    ugv = TimedRouteUgv()
    ugv.reset("ugv0", sim.setup)
    ugv.nodes, ugv.extend_failed = list(route), True  # keep the route as given
    pos = tuple(ugv.road.positions[route[0]])
    s = RobotState("ugv0", RobotType.UGV, pos, RobotMode.ACTIVE, 1e5, 1e5, False, route[0], route[0], (), (), None)
    return ugv, s


def _book(ugv, s, land_idx, t_land, slack_s, now=0.0):
    from marsupial_search.controllers.dock_booking import Booking
    assert ugv._accept(Booking("uav0", 1, ugv.version, land_idx, t_land, 500.0, slack_s, ()), s, now)
    return ugv.bookings["uav0"]


def _check_stops(ugv):
    b = ugv.bookings["uav0"]
    assert [st["kind"] for st in ugv.stops[b["land_idx"]]] == ["land"]
    assert [st["kind"] for st in ugv.stops[b["off_idx"]]] == ["takeoff"] and b["off_idx"] > b["land_idx"]
    assert all(ugv.road.has_edge(a, c) or a == c for a, c in zip(ugv.nodes, ugv.nodes[1:]))


@pytest.mark.parametrize("radio_m", [1000.0, 60.0])
def test_a_landing_on_a_blocked_node_moves_to_where_the_uav_is_met(radio_m):
    ugv, s = _ugv_on(range(20), radio_m)
    t_land = float(ugv._times(s, 0.0)[10])
    b = _book(ugv, s, 10, t_land, slack_s=200.0)
    ugv.roads.add(blocked=[10])
    ugv._check_map(s, 1.0)  # the UGV goes round node 10 on the other road
    assert b["land_node"] != 10 and not ugv.road.blocked[b["land_node"]]
    # Out of touch (no status from the UAV): met within radio range of node 10, where it will wait.
    assert np.hypot(*(ugv.road.positions[b["land_node"]] - ugv.road.positions[10])) <= radio_m
    assert b["t_land"] <= t_land + 200.0 and b["slack"] > 0
    _check_stops(ugv)
    # The take-off is at the first node the UGV reaches once the battery is swapped.
    arr = ugv._times(s, 1.0)
    assert arr[b["off_idx"]] >= b["t_ready"] and arr[b["off_idx"] - 1] < b["t_ready"]


def test_a_landing_the_uav_cannot_wait_for_moves_to_where_the_ugv_meets_it_in_time():
    # The route drives to the end of the road and back; the UAV waits at node 5 long before the
    # UGV gets back there, and cannot hover that long.
    route = list(range(20)) + list(range(18, 4, -1))
    ugv, s = _ugv_on(route, 1000.0)
    j0 = len(route) - 1
    b = _book(ugv, s, j0, 120.0, slack_s=30.0)
    ugv._check_landings(s, 1.0)
    assert b["land_idx"] < j0 and b["t_land"] <= 120.0 + 30.0  # met on the way out, in time
    _check_stops(ugv)


def test_the_ugv_takes_the_shortest_way_to_a_uav_it_would_reach_too_late():
    # Along the upper road to node 9, back along the lower one and up to land at node 12: the UGV
    # gets there long after the UAV can wait. Out of touch, with a 40 m radio, the UAV can only be
    # met at node 12 itself, so the UGV drives straight there.
    route = list(range(10)) + list(range(29, 40)) + list(range(19, 11, -1))
    ugv, s = _ugv_on(route, 40.0)
    b = _book(ugv, s, len(route) - 1, 150.0, slack_s=30.0)
    ugv._check_landings(s, 1.0)
    assert b["land_node"] == 12 and b["t_land"] <= 150.0 + 30.0
    assert ugv.nodes[:b["land_idx"] + 1] == list(range(13))
    _check_stops(ugv)


def test_a_uav_flies_straight_to_a_ugv_that_waits_for_it():
    from marsupial_search.controllers.uav_greedy_tour import GreedyTourUav
    from marsupial_search.core import GoToPoint, RobotMode, RobotState
    sim = Simulator(_ladder_scenario(), SimConfig())
    uav = GreedyTourUav()
    uav.reset("uav0", sim.setup)
    uav.phase, uav.booking, uav.waypoints, uav.tour = "sortie", {"t_land": 100.0, "land_node": 5}, [(275.0, 500.0)], []
    gs = {"pos": (325.0, 550.0), "holding": {"uav": "uav0", "kind": "land"}, "t": 50.0}
    s = RobotState("uav0", RobotType.UAV, (275.0, 500.0), RobotMode.ACTIVE, 1500.0, 2000.0, False, None, None, (), (),
                   None)
    cmds = []
    uav._step_sortie(s, 50.0, gs, cmds)
    assert cmds == [GoToPoint((325.0, 550.0))]


def test_out_of_touch_uavs_are_met_instead_of_landing_blind():
    # small map 0 with a 300 m radio: the UGV moves the landings of UAVs out of touch (detours
    # make it late). A UAV out of touch used to try landing blind, again and again.
    cfg = dataclasses.replace(small(0), target_object="car", road_unmapped_fraction=0.1, road_blocked_fraction=0.05)
    sc = generate_scenario(cfg)
    from marsupial_search.sim.radio import RadioConfig
    res = Simulator(sc, SimConfig(radio=RadioConfig(range_m={t: 300.0 for t in RobotType}))) \
        .run(make_team(DEFAULT_TEAM))
    s = res.summary()
    assert res.outcome == "found" and s["rejected"] == 0 and s["depleted"] == 0 and s["sorties"] >= 2


def test_a_landing_whose_take_off_a_re_route_put_at_the_same_node_can_still_move():
    # A re-route can put a take-off on its landing's index (the route between them skipped). The
    # UGV is there, and the landing node turns out blocked: the only place to meet is that index.
    ugv, s = _ugv_on(range(6), 1000.0)
    b = _book(ugv, s, 3, float(ugv._times(s, 0.0)[3]), slack_s=60.0)
    off = next(st for st in ugv.stops[b["off_idx"]] if st["kind"] == "takeoff")
    ugv.stops[b["off_idx"]].remove(off)
    ugv.stops[3].append(off)
    b["off_idx"], ugv.cur, b["moved"] = 3, 3, True
    from marsupial_search.core import RobotMode, RobotState
    pos = tuple(ugv.road.positions[3])
    s = RobotState("ugv0", RobotType.UGV, pos, RobotMode.ACTIVE, 1e5, 1e5, False, 3, 3, (), (), None)
    ugv._check_landings(s, 30.0)  # used to fail on an empty range of meeting points
    assert b["land_idx"] == 3
    # The landing comes first: the UGV used to wait there for the take-off of a UAV not yet on the dock.
    assert [st["kind"] for st in ugv.stops[3]] == ["land", "takeoff"]


def test_the_ugv_gives_up_on_a_landing_that_started_and_never_ended():
    # A UAV that runs out of energy while it lands sends nothing more: the UGV used to wait for it for good.
    ugv, s = _ugv_on(range(6), 1000.0)
    b = _book(ugv, s, 1, float(ugv._times(s, 0.0)[1]), slack_s=60.0)
    ugv.cur = 1
    ugv.stops[1][0]["started"] = 100.0
    dock = ugv.setup.docking.dock_duration_s
    ugv._give_up_late(s, 100.0 + dock + ugv.cfg.max_late_s - 1.0)
    assert "uav0" in ugv.bookings
    ugv._give_up_late(s, 100.0 + dock + ugv.cfg.max_late_s + 1.0)
    assert "uav0" not in ugv.bookings and not any(ugv.stops.values())
