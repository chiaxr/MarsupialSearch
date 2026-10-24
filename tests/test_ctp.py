"""Canadian Traveller's Problem road planning (controllers/ctp.py) and ugv_ctp_route."""

import dataclasses

import numpy as np
import pytest
from scipy.sparse.csgraph import connected_components

from marsupial_search.controllers import DEFAULT_TEAM, make_team
from marsupial_search.controllers.ctp import BlockageModel, CtpPlanner, cut_nodes, cut_off_counts
from marsupial_search.controllers.ugv_timed_route import UgvTimedRouteConfig
from marsupial_search.core import GraphKnowledge, TraversabilityGraph
from marsupial_search.sim import Simulator, generate_scenario, resample_target
from marsupial_search.sim.scenario_bases import small


def _two_roads():
    """S (0, 0) to G (1000, 0): a straight road (nodes 50 m apart, x = 50..950) and a road bowing
    out to y = 150 (x = 50..950), 1216 m long. Returns the graph, S, G, the straight and the bow nodes."""
    pos = [(0.0, 0.0), (1000.0, 0.0)]
    straight = list(range(2, 21))
    bow = list(range(21, 40))
    pos += [(x, 0.0) for x in range(50, 1000, 50)] + [(x, 150.0) for x in range(50, 1000, 50)]
    edges = []
    for road in (straight, bow):
        path = [0] + road + [1]
        edges += list(zip(path, path[1:]))
    return TraversabilityGraph(np.array(pos), np.array(edges)), 0, 1, straight, bow


@pytest.mark.parametrize("seed", range(5))
def test_cut_off_counts_are_the_nodes_a_blockage_parts_from_the_largest_piece(seed):
    rng = np.random.default_rng(seed)
    n = 40
    pos = rng.random((n, 2)) * 100
    edges = [(i, int(rng.integers(i))) for i in range(1, n)] + [tuple(rng.integers(n, size=2)) for _ in range(12)]
    g = TraversabilityGraph(pos, np.array(edges))
    counts, cut = cut_off_counts(g), cut_nodes(g)
    for v in range(n):
        keep = np.arange(n) != v
        k, comp = connected_components(g.csr[keep][:, keep], directed=False)
        assert counts[v] == n - 1 - np.bincount(comp).max() and cut[v] == (k > 1), v


def test_ctp_takes_a_known_road_round_a_risky_one_and_the_short_one_when_safe():
    g, S, G, straight, bow = _two_roads()
    p = np.zeros(g.n_nodes)
    p[straight] = 0.05  # 62 % that the straight road is blocked somewhere
    path, cost = CtpPlanner(g, 55.0, samples=32).path(S, G, p)
    assert path == [S] + bow + [G] and cost == pytest.approx(g.path_length(path))
    p[straight] = 0.002
    path, cost = CtpPlanner(g, 55.0, samples=32).path(S, G, p)
    assert path == [S] + straight + [G] and cost >= 1000.0
    # Nothing that may be blocked: the shortest way, at its length.
    assert CtpPlanner(g, 55.0).path(S, G, np.zeros(g.n_nodes)) == ([S] + straight + [G], 1000.0)


def test_a_rollout_finds_a_blockage_within_sensor_range_and_turns_back():
    g, S, G, straight, bow = _two_roads()
    p = np.zeros(g.n_nodes)
    p[straight[9]] = 1.0  # (500, 0): blocked in every scenario
    ctp = CtpPlanner(g, 55.0, samples=4)
    ctp._goal = -1
    ctp._tree(G, frozenset())
    ctp._draw(p)
    # From (50, 0) it sees the blockage at (450, 0), drives back to S and takes the bow.
    assert ctp._expected(straight[0], G) == pytest.approx(400.0 + 450.0 + g.path_length([S] + bow + [G]))
    path, cost = ctp.path(S, G, p)
    assert path == [S] + bow + [G] and cost == pytest.approx(g.path_length(path))


def test_rewards_pull_the_way_to_them_unless_the_risk_outweighs_them():
    g, S, G, straight, bow = _two_roads()
    none, reward = np.zeros(g.n_nodes), np.zeros(g.n_nodes)
    reward[bow] = 20.0  # 380 m in all, for 216 m more road
    path, cost = CtpPlanner(g, 55.0).path(S, G, none, reward)
    assert path == [S] + bow + [G] and cost == pytest.approx(g.path_length(path) - 380.0)
    reward[bow] = 5.0  # 95 m: not worth it
    assert CtpPlanner(g, 55.0).path(S, G, none, reward)[0] == [S] + straight + [G]
    # The bow may be blocked (62 %): backtracking over it collects nothing, so the straight road,
    # known to be clear, is cheaper. (Rolled out at these costs alone, every first move heads
    # for the bow, and the straight road is never valued.)
    risk = np.zeros(g.n_nodes)
    risk[bow] = 0.05
    reward[bow] = 20.0
    path, cost = CtpPlanner(g, 55.0, samples=32).path(S, G, risk, reward)
    assert path == [S] + straight + [G] and cost == pytest.approx(1000.0)


def test_blockage_model_rates_unseen_nodes_and_clears_seen_and_cut_ones():
    g, S, G, straight, bow = _two_roads()
    leaf = g.n_nodes  # a dead end off G: G becomes a cut node
    roads = GraphKnowledge(TraversabilityGraph(np.concatenate([g.positions, [[1050.0, 0.0]]]),
                                               np.concatenate([g.edges, [[G, leaf]]])))
    model = BlockageModel(roads, prior=0.1, prior_weight=10.0)
    p = model.probabilities(roads.graph)
    assert p[G] == 0 and p[leaf] == pytest.approx(0.1) and p[straight[0]] == pytest.approx(0.1)
    model.look(roads.graph, (0.0, 0.0), 55.0)  # sees S and straight[0], both open: (0 + 1) / (10 + 2)
    p = model.probabilities(roads.graph)
    assert p[S] == 0 and p[straight[0]] == 0 and p[straight[1]] == pytest.approx(1.0 / 12.0)
    # straight[5] blocked: the map is now a tree, and only its leaves may still be blocked. The
    # seen open nodes are cut nodes (they never count): the rate is (1 + 1) / (10 + 1).
    roads.add(blocked=[straight[5]])
    p = model.probabilities(roads.graph)
    assert p[straight[5]] == 0 and p[straight[:4] + bow + [G, S]].max() == 0
    assert p[straight[4]] == p[straight[6]] == p[leaf] == pytest.approx(2.0 / 11.0)
    assert BlockageModel(roads, 0.1, 10.0, max_cutoff_fraction=1.0).probabilities(roads.graph)[bow[3]] > 0


def test_blockage_model_rules_out_blockages_that_would_cut_off_too_much():
    g, S, G, straight, bow = _two_roads()
    tail = [g.n_nodes, g.n_nodes + 1, g.n_nodes + 2]  # a dead end off G: G -- t0 -- t1 -- t2
    pos = np.concatenate([g.positions, [[1050.0, 0.0], [1100.0, 0.0], [1150.0, 0.0]]])
    roads = GraphKnowledge(TraversabilityGraph(pos, np.concatenate([g.edges, [[G, tail[0]], tail[:2], tail[1:]]])))
    # Blocking G cuts off 3 of the 43 nodes, t0 2 and t1 1.
    p = BlockageModel(roads, 0.1, 10.0, max_cutoff_fraction=0.0).probabilities(roads.graph)
    assert p[[G] + tail[:2]].max() == 0 and p[tail[2]] > 0 and p[bow[3]] > 0
    p = BlockageModel(roads, 0.1, 10.0, max_cutoff_fraction=2.5 / 43).probabilities(roads.graph)
    assert p[G] == 0 and p[tail].min() > 0
    # Once t1 is found blocked (t2 cut off), 1.5 nodes are left to cut off: G and t0 now cut off one each.
    roads.add(blocked=[tail[1]])
    p = BlockageModel(roads, 0.1, 10.0, max_cutoff_fraction=2.5 / 43).probabilities(roads.graph)
    assert p[G] > 0 and p[tail[0]] > 0
    p = BlockageModel(roads, 0.1, 10.0, max_cutoff_fraction=1.0 / 43).probabilities(roads.graph)
    assert p[G] == 0 and p[tail[0]] > 0  # nothing left to cut off; t0, now a dead end, cuts off nothing


def test_road_planner_is_checked():
    with pytest.raises(ValueError):
        UgvTimedRouteConfig(road_planner="pessimistic")


@pytest.mark.parametrize("seed", [1, 2])
def test_ugv_ctp_route_finds_the_target_with_a_wrong_road_map(seed):
    cfg = dataclasses.replace(small(seed), road_unmapped_fraction=0.1, road_blocked_fraction=0.05)
    sc = resample_target(generate_scenario(cfg), np.random.default_rng(seed))
    res = Simulator(sc).run(make_team({**DEFAULT_TEAM, "ugv0": "ugv_ctp_route"}))
    assert res.outcome == "found"
    assert res.summary()["rejected"] == 0
