"""Relaying each robot's latest status (core.relay, dock_booking) and risk-aware landings."""

import numpy as np
import pytest

from conftest import make_line_scenario
from marsupial_search.controllers.ctp import clear_probabilities
from marsupial_search.controllers.dock_booking import BookingUavConfig
from marsupial_search.controllers.uav_greedy_tour import GreedyTourUav
from marsupial_search.core import Relay, TraversabilityGraph
from marsupial_search.sim import Simulator


def test_relay_sends_what_a_neighbour_lacks_and_merges_records():
    r = Relay("uav1", period_s=5.0)
    assert r.update("ugv0", 10.0, {"pos": (0, 0), "plan": {"v": 1}})
    assert r.update("ugv0", 20.0, {"pos": (5, 0)})  # no plan this time: the last one stays
    assert not r.update("ugv0", 15.0, {"pos": (9, 9)})  # older: ignored
    assert r.records["ugv0"] == (20.0, {"pos": (5, 0), "plan": {"v": 1}})
    r.update("uav1", 21.0, {"pos": (1, 1)})
    assert r.outgoing(21.0) == {}  # no neighbour heard
    r.heard("uav0", {"ugv0": 2.0}, 21.0)
    assert set(r.outgoing(21.0)) == {"ugv0"}  # uav0's copy is old (never its own record, nor uav1's)
    r.heard("uav0", {"ugv0": 17.0}, 21.0)
    assert r.outgoing(21.0) == {}  # less than period_s older: not worth sending again
    assert r.outgoing(30.0) == {}  # uav0 not heard lately
    r.heard("ugv0", {}, 21.0)
    assert r.outgoing(21.0) == {}  # the origin itself lacks nothing of its own


def _uav(seek=False):
    sim = Simulator(make_line_scenario())
    uav = GreedyTourUav(BookingUavConfig(seek_ugv=seek))
    uav.reset("uav0", sim.setup)
    return uav


def _status(t, **kw):
    return {"type": "ugv", "pos": (0.0, 500.0), "mode": "active", "energy": 1.0, "energy_fraction": 1.0,
            "visited": [], "updates": [], "sighting": None, "done": False, **kw}


def test_a_uav_learns_the_ugvs_plan_through_the_other_uav():
    uav = _uav()
    plan = {"version": 3, "cur": 0, "nodes": [0, 1], "arrive": [0.0, 10.0], "queue": [], "pin": 0}
    relayed = {"ugv0": (40.0, _status(40.0, plan=plan))}
    uav1 = {**_status(41.0), "type": "uav", "clock": {"ugv0": 40.0}, "relay": relayed}
    uav._receive("uav1", uav1, 41.0, 42.0)
    assert uav.plan["version"] == 3 and uav.plan["t"] == 40.0 and uav.peers["ugv0"]["t"] == 40.0
    # A direct status older than what came relayed keeps the newer state, but its belief updates still count.
    uav._receive("ugv0", {**_status(39.0), "pos": (99.0, 500.0)}, 39.0, 42.0)
    assert uav.peers["ugv0"]["t"] == 40.0 and uav.peers["ugv0"]["pos"] == (0.0, 500.0)
    # Its next broadcast tells the others what it holds.
    assert uav.relay.clock()["ugv0"] == 40.0


def test_clear_probabilities_count_the_nodes_with_a_long_way_round():
    # S - a - b - G along y = 0, 50 m apart, and a long way round (by y = 500) from S to G.
    pos = np.array([[0, 0], [50, 0], [100, 0], [150, 0], [0, 500], [150, 500]], float)
    g = TraversabilityGraph(pos, np.array([(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 3)]))
    p = np.array([0.0, 0.1, 0.2, 0.0, 0.0, 0.0])
    clear = clear_probabilities(g, [0, 1, 2, 3], p, max_detour_m=300.0)
    assert clear == pytest.approx([1.0, 0.9, 0.72, 0.72])
    assert clear_probabilities(g, [0, 1, 2, 3], p, max_detour_m=2000.0) == pytest.approx([1.0] * 4)


def test_a_uav_books_only_where_the_ugv_is_likely_to_get_without_a_long_detour():
    uav = _uav()
    uav.plan = {"clear": [1.0, 0.95, 0.8, 0.5, 0.3]}
    idx = np.array([2, 3, 4])
    assert list(uav._safe(idx)) == [2, 3, 4]  # off by default
    uav.cfg.max_landing_risk = 0.5
    assert list(uav._safe(idx)) == [2, 3]
    uav.cfg.max_landing_risk = 0.1
    assert list(uav._safe(idx)) == [2]  # none safe enough: the safest
