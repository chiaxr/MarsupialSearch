"""The default team: ``ugv_timed_route`` on the UGV, ``uav_greedy_tour`` on both UAVs."""

import dataclasses

import numpy as np
import pytest

from conftest import make_line_scenario
from marsupial_search.controllers import DEFAULT_TEAM, make_team, team_kind
from marsupial_search.controllers.common import dist
from marsupial_search.controllers.dock_booking import BookingUavConfig, slot_spacing
from marsupial_search.controllers.sortie_planner import SortieConfig, SortiePlanner
from marsupial_search.controllers.ugv_timed_route import TimedRouteUgv, UgvTimedRouteConfig
from marsupial_search.core import BeliefMap, EventType, RobotType
from marsupial_search.sim import RadioConfig, ScenarioConfig, SimConfig, Simulator, generate_scenario


@pytest.fixture(scope="module")
def scenario():
    cfg = ScenarioConfig(size_m=(2500.0, 2500.0), n_urban=2, urban_half_size_m=(200.0, 300.0), n_rural_nodes=25,
                         seed=3)
    return generate_scenario(cfg)


def test_default_team_is_decentralised():
    assert team_kind(DEFAULT_TEAM) == "decentralised"


def test_needs_exactly_one_ugv():
    sc = make_line_scenario(n_uav=1)
    spec = sc.robots["ugv0"]
    sc.robots["ugv1"] = dataclasses.replace(spec, robot_id="ugv1")
    with pytest.raises(ValueError, match="exactly one UGV"):
        TimedRouteUgv().reset("ugv0", sc.mission_setup(1.0))


def test_slot_spacing(scenario):
    setup = Simulator(scenario).setup
    cfg = BookingUavConfig()
    # A full sortie (90 % of the battery at 1.4 units/s, flying at 10 m/s) plus a stay on the dock, shared by both UAVs.
    stay = 30.0 + 120.0 + 15.0 + 2 * cfg.stop_margin_s
    assert slot_spacing(setup, cfg, 2) == pytest.approx((0.9 * 2000.0 / 1.4 + stay) / 2)
    assert slot_spacing(setup, cfg, 30) == pytest.approx(stay)  # never shorter than one stay
    assert slot_spacing(setup, BookingUavConfig(slot_s=123.0), 2) == 123.0


def test_chain_tour_fits_its_range_and_skips_claimed_nodes(scenario):
    setup = Simulator(scenario).setup
    P = SortiePlanner(setup, setup.robots["ugv0"], setup.robots["uav0"], SortieConfig())
    road = setup.graphs[RobotType.UGV]
    start, end = tuple(road.positions[scenario.init["ugv0"].node]), tuple(road.positions[road.n_nodes // 2])
    m = P.searchable_mass(BeliefMap(setup.grid, setup.prior, setup.layer_fractions))
    no_ugv = np.full(P.F_road.shape[1], np.inf)
    L_max = dist(start, end) + 3000.0
    free = np.full(P.air.n_nodes, np.inf)
    _, tour, length, value = P.chain_tour(m, no_ugv, start, end, L_max, 0.0, 0.0, free, 0.0)
    pts = [start, *map(tuple, P.air.positions[tour]), end]
    assert tour and value > 0 and len(set(tour)) == len(tour)
    assert length == pytest.approx(sum(dist(a, b) for a, b in zip(pts, pts[1:]))) and length <= L_max + 1e-6
    claimed = free.copy()
    claimed[tour[:3]] = -np.inf
    _, other, _, _ = P.chain_tour(m, no_ugv, start, end, L_max, 0.0, 0.0, claimed, 0.0)
    assert not set(other) & set(tour[:3])
    assert P.chain_tour(m, no_ugv, start, end, dist(start, end) - 1.0, 0.0, 0.0, free, 0.0) is None


def test_full_coverage_without_losing_uavs(scenario):
    res = Simulator(scenario, SimConfig(terminate_on=(), max_time_s=6 * 3600)).run(make_team(DEFAULT_TEAM))
    s = res.summary()
    assert res.outcome == "controller_done"
    assert s["depleted"] == 0 and s["sorties"] >= 4
    assert s["rejected"] == 0 and s["dock_aborts"] == 0 and s["undock_aborts"] == 0
    assert s["searched_prob"] > 0.99
    assert all(res.log[f"energy_{u}"].min() > 0.05 for u in ("uav0", "uav1"))


def test_landings_take_turns(scenario):
    res = Simulator(scenario, SimConfig(terminate_on=(), max_time_s=6 * 3600)).run(make_team(DEFAULT_TEAM))
    landers = [e.robot_id for e in res.events if e.type == EventType.DOCKED]
    assert len(landers) >= 6
    assert all(a != b for a, b in zip(landers, landers[1:]))  # the UAVs alternate
    times = [e.time for e in res.events if e.type == EventType.DOCKED]
    slot = slot_spacing(Simulator(scenario).setup, BookingUavConfig(), 2)
    assert np.median(np.diff(times)) > 0.8 * slot  # spread along the route (energy may bring one forward)


def test_finds_target(scenario):
    res = Simulator(scenario, SimConfig()).run(make_team(DEFAULT_TEAM))
    assert res.outcome == "found" and res.found_by == "ugv0"
    assert res.summary()["depleted"] == 0


def test_params_reach_each_robot_of_a_policy():
    team = make_team(DEFAULT_TEAM, {"uav_greedy_tour": {"plan_lead_s": 5.0}, "ugv_timed_route": {"route_replan_s": 30.0}})
    team.reset(Simulator(make_line_scenario(n_uav=2)).setup)
    assert all(team.robots[u].cfg.plan_lead_s == 5.0 for u in ("uav0", "uav1"))
    assert isinstance(team.robots["ugv0"].cfg, UgvTimedRouteConfig) and team.robots["ugv0"].cfg.route_replan_s == 30.0


POOR_RADIO = RadioConfig(range_m={RobotType.UGV: 400.0, RobotType.UAV: 400.0}, drop_prob=0.3, latency_s=3.0)


def test_keeps_uavs_safe_on_a_poor_radio(scenario):
    sim = Simulator(scenario, SimConfig(terminate_on=(), max_time_s=6 * 3600, radio=POOR_RADIO))
    res = sim.run(make_team(DEFAULT_TEAM))
    s = res.summary()
    assert s["depleted"] == 0 and s["sorties"] >= 4 and s["radio_dropped_range"] > 0
    assert s["rejected"] == 0 and s["dock_aborts"] == 0 and s["undock_aborts"] == 0
    assert all(res.log[f"energy_{u}"].min() > 0.05 for u in ("uav0", "uav1"))


def test_runs_are_reproducible(scenario):
    cfg = SimConfig(max_time_s=2 * 3600, radio=POOR_RADIO)
    a, b = (Simulator(scenario, cfg).run(make_team(DEFAULT_TEAM)) for _ in range(2))
    assert a.time_s == b.time_s and a.outcome == b.outcome
    assert all((a.trajectories[r] == b.trajectories[r]).all() for r in a.trajectories)


def test_reproducible_across_processes():
    # Iteration order of str-keyed sets changes with PYTHONHASHSEED; results must not.
    import os
    import subprocess
    import sys

    code = ("from marsupial_search.controllers import DEFAULT_TEAM, make_team; from marsupial_search.core import RobotType; "
            "from marsupial_search.sim import SCENARIO_BASES, RadioConfig, SimConfig, Simulator, generate_scenario; "
            "sc = generate_scenario(SCENARIO_BASES['small'](1)); c = SimConfig(); "
            "c.radio = RadioConfig(range_m={RobotType.UGV: 300.0, RobotType.UAV: 300.0}, drop_prob=0.3); "
            "r = Simulator(sc, c).run(make_team(DEFAULT_TEAM)); "
            "print(r.time_s, r.outcome, r.summary()['radio_received'])")
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                           env={**os.environ, "PYTHONHASHSEED": seed}).stdout for seed in ("1", "2")}
    assert len(outs) == 1, outs


def test_a_pursuit_put_after_the_pin_takes_the_detour_once_it_fits(scenario):
    sim = Simulator(scenario, SimConfig(terminate_on=()))
    team = make_team(DEFAULT_TEAM)
    obs = sim.reset()
    team.reset(sim.setup)
    ugv = team.robots["ugv0"]
    ahead = lambda: sorted(i for i, st in ugv.stops.items() if st and i > ugv.cur + 1)
    while not (len(ahead()) >= 2 and ahead()[0] > ugv.cur + 6 and obs.robots["ugv0"].moving):
        obs = sim.step(team.step(obs))
    nxt = ahead()[0]
    sighting = ugv.nodes[nxt - 2]  # on the route before the next stop: the detour costs nothing
    ugv.belief.detection_xy = tuple(ugv.road.positions[sighting])
    calls, detour_now = [], ugv._detour_now
    ugv._detour_now = lambda at, xy: calls.append(at) is None and len(calls) > 1 and detour_now(at, xy)
    obs = sim.step(team.step(obs))  # the first check fails: it goes after the pin
    assert ugv.pursuit == "planned" and ugv.pursuit_end > ugv._pin() and len(calls) == 1
    for _ in range(300):
        if ugv.pursuit_check is None:
            break
        obs = sim.step(team.step(obs))
    assert ugv.pursuit_check is None and len(calls) >= 2
    assert ugv.pursuit == "planned" and ugv.pursuit_end < ahead()[0]  # now before the next stop
    assert sighting not in ugv.nodes[ugv._pin():]  # the pursuit after the pin is gone
