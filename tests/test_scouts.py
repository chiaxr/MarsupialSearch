"""UAVs with their own sensors and policies: a low standby scout (``uav_near_scout``) that the UGV calls,
and a high one that books its own sorties (``uav_far_scout``)."""

import dataclasses

import numpy as np
import pytest

from conftest import make_line_scenario
from marsupial_search.controllers import DEFAULT_TEAM, make_team
from marsupial_search.controllers.dock_booking import LAND, TAKEOFF, Booking
from marsupial_search.controllers.uav_near_scout import NearScoutUav
from marsupial_search.controllers.ugv_timed_route import TimedRouteUgv
from marsupial_search.core import EventType, RobotMode, RobotState, RobotType, TraversabilityGraph
from marsupial_search.sim import SimConfig, Simulator, generate_scenario, resample_target
from marsupial_search.sim.generator import uav_grid_spacing
from marsupial_search.sim.radio import RadioConfig
from marsupial_search.sim.scenario_bases import small

SCOUT_TEAM = {"ugv0": "ugv_timed_route", "uav0": "uav_near_scout", "uav1": "uav_far_scout"}
RANGES = {"uav0": 20.0, "uav1": 50.0}


def test_each_uav_can_have_its_own_sensor_range():
    cfg = small(0)
    cfg = dataclasses.replace(cfg, fleet=dataclasses.replace(cfg.fleet, uav_sensor_ranges=RANGES))
    sc = generate_scenario(cfg)
    assert {u: sc.robots[u].sensor.range_m for u in RANGES} == RANGES
    assert uav_grid_spacing(cfg) == pytest.approx(np.sqrt(2) * 20.0)  # gap-free for the smaller footprint
    bad = dataclasses.replace(cfg, fleet=dataclasses.replace(cfg.fleet, uav_sensor_ranges={"uav2": 20.0}))
    with pytest.raises(ValueError, match="uav2"):
        generate_scenario(bad)


# ------------------------------------------------------------------- the UGV's side
def _ugv(radio_m=1000.0, ladder=False):
    """A ``ugv_timed_route`` controller at node 0 of a straight road (nodes 0-19, 50 m apart; with
    ``ladder``, and a parallel road 50 m above it), with ``uav0`` docked and on standby."""
    sc = make_line_scenario(n_uav=2)
    if ladder:
        top = sc.graphs[RobotType.UGV]
        pos = np.concatenate([top.positions, top.positions + [0.0, 50.0]])
        edges = np.concatenate([top.edges, top.edges + 20, np.stack([np.arange(20), np.arange(20, 40)], 1)])
        sc = dataclasses.replace(sc, graphs={**sc.graphs, RobotType.UGV: TraversabilityGraph(pos, edges, "road")})
    sim = Simulator(sc, SimConfig(radio=RadioConfig(range_m={t: radio_m for t in RobotType})))
    ugv = TimedRouteUgv()
    ugv.reset("ugv0", sim.setup)
    ugv.nodes, ugv.extend_failed = list(range(20)), True  # keep the route as given
    ugv.standby = {"uav0"}
    ugv.stops[0] = [{"kind": TAKEOFF, "uav": "uav0", "ready": 0.0}]  # as planned before it knew uav0 was on standby
    s = RobotState("ugv0", RobotType.UGV, tuple(ugv.road.positions[0]), RobotMode.ACTIVE, 1e5, 1e5, False, 0, 0, (),
                   (), None, docked_uav="uav0")
    return ugv, s


def _book(ugv, s, uav, land_idx, t_land, slack_s=100.0):
    return ugv._accept(Booking(uav, 1, ugv.version, land_idx, t_land, 500.0, slack_s, ()), s, 0.0)


def test_a_standby_uav_leaves_the_dock_just_before_another_uav_lands():
    ugv, s = _ugv()
    ugv._call_standby(s, 0.0)
    assert not ugv.stops.get(0)  # not called: it stays on the dock
    t_land = float(ugv._times(s, 0.0)[10]) + 100.0  # the UGV gets there first and waits
    assert _book(ugv, s, "uav1", 10, t_land)
    ugv._call_standby(s, 0.0)
    off, land = ugv.stops[10]
    assert (off["kind"], off["uav"], off["call"], land["kind"], land["uav"]) == (TAKEOFF, "uav0", "uav1", LAND, "uav1")
    lead = ugv.setup.docking.undock_duration_s + ugv.cfg.stop_margin_s
    assert off["ready"] == pytest.approx(t_land - lead)  # not as soon as the UGV gets there
    p = ugv._plan_payload(s, 0.0)
    assert p["calls"] == {"uav0": {"for": "uav1", "xy": None, "near": None, "deliver": None}} and p["takeoffs"]["uav0"][0] == 10
    o = ugv.bookings["uav1"]["off_idx"]
    assert p["busy"] == [(0, 10), (10, o)] and p["standby"] == ["uav0"]  # route indices each UAV is on the dock
    # uav0 lands where the dock is free: not while uav1 is on it, and it stays (no take-off).
    assert not _book(ugv, s, "uav0", o - 1, t_land + 100.0)
    assert _book(ugv, s, "uav0", o + 1, t_land + 400.0)
    assert ugv.bookings["uav0"]["off_idx"] is None
    assert [st["uav"] for st in ugv.stops[o + 1]] == ["uav0"]
    ugv._call_standby(s, 0.0)
    assert ugv.free_from == pytest.approx(ugv.bookings["uav0"]["t_ready"])  # the next landing waits for its swap
    assert ugv._pin() == o + 1  # uav1's next landing comes after it (append-only, as before)


@pytest.mark.parametrize("case", ["cut off", "searched", "detour", "far"])
def test_the_standby_uav_is_called_to_search_what_a_blockage_cuts_off(case):
    # On a single road, blocking node 3 puts nodes 4-19 out of the UGV's reach (a pocket).
    ugv, s = _ugv(ladder=case == "detour")  # with a second road, they are only 100 m further away
    ugv._call_standby(s, 0.0)
    node = 15 if case == "far" else 3  # 750 m away (call_radius_m = 250)
    if case == "searched":  # every robot's looks are in the UGV's belief
        ugv.belief.mass[:, :, 5:] = 0.0  # x >= 125 m
    ugv.roads.add(blocked=[node])
    ugv._check_map(s, 1.0)
    sts = ugv.stops.get(0, [])
    if case != "cut off":
        assert not sts
        return
    assert [(st["kind"], st["uav"], st["call"]) for st in sts] == [(TAKEOFF, "uav0", None)]  # where the UGV is
    near = ugv.road.positions[4:11]  # within call_reach_m = 500 m of node 0 by road
    assert sts[0]["xy"] == tuple(ugv.road.positions[3]) and np.allclose(sts[0]["near"], near)
    assert np.allclose(ugv._plan_payload(s, 1.0)["calls"]["uav0"]["near"], near)


def test_a_call_the_standby_uav_never_books_is_dropped():
    ugv, s = _ugv()
    ugv._call_standby(s, 0.0)
    ugv.roads.add(blocked=[3])
    ugv._check_map(s, 1.0)
    assert ugv.stops.get(0)
    ugv._call_standby(s, 1.0)  # the UGV holds for it from now
    ugv._call_standby(s, 1.0 + ugv.cfg.call_timeout_s + 1.0)
    assert not ugv.stops.get(0)


# ------------------------------------------------------------------------ missions
@pytest.fixture(scope="module")
def scout_mission():
    cfg = dataclasses.replace(small(4), target_object="car", road_unmapped_fraction=0.1, road_blocked_fraction=0.05)
    cfg = dataclasses.replace(cfg, fleet=dataclasses.replace(cfg.fleet, uav_sensor_ranges=RANGES))
    sc = resample_target(generate_scenario(cfg), np.random.default_rng(101))
    sorties = []  # (cut-off road nodes or None, tour) of each sortie the scout plans
    orig = NearScoutUav.plan_sortie

    def plan(self, *a):
        out = orig(self, *a)
        task = self.call if self.phase == "docked" else self.task
        if out is not None:
            sorties.append((task["near"] if task is not None else None, out.tour, self.planner.air.positions))
        return out
    NearScoutUav.plan_sortie = plan
    try:
        res = Simulator(sc, SimConfig(radio=RadioConfig(range_m={t: 300.0 for t in RobotType}))) \
            .run(make_team(SCOUT_TEAM))
    finally:
        NearScoutUav.plan_sortie = orig
    return res, sorties


def test_the_scout_leaves_the_dock_only_when_called(scout_mission):
    res, sorties = scout_mission
    s = res.summary()
    # Here the UGV runs flat before it finds the target (the small map's UGV has about 40 min); no UAV does.
    assert res.outcome in ("found", "ugv_depleted") and s["rejected"] == 0
    assert not any(res.robot_stats[u]["depleted"] for u in ("uav0", "uav1"))
    ev = [(e.time, e.type, e.robot_id) for e in res.events if e.type in (EventType.DOCKED, EventType.UNDOCKED)]
    blocked = [xy for xy, _, _ in sorties if xy is not None]
    making_room = [t for t, k, u in ev if k == EventType.DOCKED and u == "uav1"]
    scout_off = [t for t, k, u in ev if k == EventType.UNDOCKED and u == "uav0"]
    assert blocked and making_room and len(scout_off) >= 2
    for t in making_room:  # it left the dock shortly before uav1 landed
        assert any(t - 120.0 <= t_off < t for t_off in scout_off)


def test_the_scout_searches_what_the_blockage_cut_off(scout_mission):
    _, sorties = scout_mission
    radius = NearScoutUav.config_cls().scout_radius_m
    for near, tour, air in sorties:
        if near is not None and tour:
            d = np.hypot(*(air[list(tour)][:, None] - np.asarray(near)[None]).transpose(2, 0, 1))
            assert d.min(1).max() <= radius + 1e-6


def test_the_greedy_team_still_runs_with_different_sensors():
    cfg = dataclasses.replace(small(4), target_object="car")
    cfg = dataclasses.replace(cfg, fleet=dataclasses.replace(cfg.fleet, uav_sensor_ranges=RANGES))
    res = Simulator(generate_scenario(cfg), SimConfig()).run(make_team(DEFAULT_TEAM))
    s = res.summary()
    assert res.outcome == "found" and s["rejected"] == 0 and s["depleted"] == 0


# --------------------------------------------------------------------- the far scout
def _far(**cfg):
    """A ``uav_far_scout`` on the line map (road along y = 500 m), with the UGV's plan: the whole
    road, node i reached at 10 i s."""
    from marsupial_search.controllers.uav_far_scout import FarScoutUav, FarScoutUavConfig
    sim = Simulator(make_line_scenario(n_uav=2), SimConfig())
    uav = FarScoutUav(FarScoutUavConfig(**cfg))
    uav.reset("uav1", sim.setup)
    uav.plan = {"nodes": list(range(20)), "arrive": [10.0 * i for i in range(20)], "cur": 0}
    m = uav.planner.searchable_mass(uav.belief)
    return uav, m


def _cell(uav, x, y):
    return int(np.argmin(np.hypot(*(uav.cells - [x, y]).T)))


def test_the_far_scout_weighs_nothing_with_every_option_off():
    uav, m = _far(near_route_m=0.0)
    assert uav._tour_mass(m, 0.0) is m


def test_the_far_scout_leaves_belief_near_the_ugvs_route_to_the_others():
    uav, m = _far(near_route_m=300.0, near_route_weight=0.25)
    w = uav._tour_mass(m, 0.0).sum(0) / m.sum(0)
    assert w[_cell(uav, 500, 500)] == pytest.approx(0.25, abs=0.06)  # on the road (the cell centre is 18 m from a node)
    assert w[_cell(uav, 500, 650)] == pytest.approx(0.25 + 0.75 * 0.5, abs=0.06)  # 150 m off it
    assert w[_cell(uav, 500, 900)] == pytest.approx(1.0)  # 400 m off it


def test_the_far_scout_prefers_belief_the_ugv_can_confirm_soon():
    uav, m = _far(near_route_m=0.0, confirm_scale_s=600.0)
    w = uav._tour_mass(m, 0.0).sum(0) / m.sum(0)
    assert w[_cell(uav, 500, 525)] == pytest.approx(1.0)  # within the UGV's sensor range of its route
    far = w[_cell(uav, 500, 975)]  # 475 m off the road: 420 m beyond sensor range, 84 s at 5 m/s
    assert far == pytest.approx(600.0 / (600.0 + 420.0 / 5.0), abs=0.02)


def test_the_far_scout_checks_the_road_ahead_once_blockages_are_known():
    uav, m = _far(near_route_m=0.0, recon_weight=2.0, recon_from_s=50.0, recon_to_s=100.0)  # nodes 5-10
    assert np.array_equal(uav._tour_mass(m, 0.0), m)  # no blocked node found yet: nothing to check
    uav.roads.add(blocked=[15])
    out = uav._tour_mass(m, 0.0)
    extra = (out - m).sum(0)
    bumped = {int(np.argmin(np.hypot(*(uav.road.positions[:20] - uav.cells[c]).T))) for c in np.flatnonzero(extra)}
    assert bumped == set(range(5, 11))
    uav.belief.mass[:, :, :8] *= 0.0  # someone looked at x < 200 m (nodes 0-3) ...
    uav.belief.mass[:, 20, [11, 13]] *= 0.0  # ... and at nodes 5 and 6 (x = 275, 325)
    extra = (uav._tour_mass(m, 0.0) - m).sum(0)
    bumped = {int(np.argmin(np.hypot(*(uav.road.positions[:20] - uav.cells[c]).T))) for c in np.flatnonzero(extra)}
    assert bumped == set(range(7, 11))


# ----------------------------------------------------------------- the messenger
def _stranded(radio_m=300.0):
    """The UGV with ``uav1`` booked to land at node 10, out of touch and short of energy: the UGV
    gets there 200 s after the UAV, which can wait only 10 s."""
    ugv, s = _ugv(radio_m)
    ugv.stops.clear()
    t10 = float(ugv._times(s, 0.0)[10])
    assert _book(ugv, s, "uav1", 10, t10 - 200.0, slack_s=10.0)
    return ugv, s


def test_the_ugv_sends_its_standby_uav_to_a_uav_it_can_no_longer_meet_in_time():
    ugv, s = _stranded()
    ugv._check_landings(s, 1.0)
    call = next(st for st in ugv.stops[0] if st["uav"] == "uav0" and st["kind"] == TAKEOFF)
    assert call["deliver"] == "uav1" and call["xy"] == tuple(ugv.road.positions[10]) and call["call"] is None
    assert ugv.messengers["uav1"]["by"] == "uav0" and ugv.messengers["uav1"]["t"] > 1.0
    assert ugv._plan_payload(s, 1.0)["calls"]["uav0"]["deliver"] == "uav1"
    # Its call withdrawn (it never booked a landing): not again for that landing.
    ugv.stops[0] = [st for st in ugv.stops[0] if st.get("deliver") != "uav1"]
    ugv._check_messengers(s)
    assert "uav1" not in ugv.messengers and ("uav1", 1) in ugv.messenger_failed
    assert not ugv._send_messenger(s, 2.0, "uav1", ugv.road.positions[10], 0, ugv._times(s, 2.0))


def test_no_messenger_without_a_standby_uav_on_the_dock_or_without_relaying():
    ugv, s = _stranded()
    ugv.cfg.relay = False
    ugv._check_landings(s, 1.0)
    assert not ugv.messengers
    ugv, s = _stranded()
    s = dataclasses.replace(s, docked_uav=None)
    ugv._check_landings(s, 1.0)
    assert not ugv.messengers


def _messenger(plan):
    sim = Simulator(make_line_scenario(n_uav=2), SimConfig(radio=RadioConfig(range_m={t: 300.0 for t in RobotType})))
    uav = NearScoutUav()
    uav.reset("uav0", sim.setup)
    uav.plan, uav.phase = plan, "docked"
    return uav


def _plan(arrive_from=10.0):
    return {"version": 0, "cur": 0, "nodes": list(range(20)), "arrive": [arrive_from + 10.0 * i for i in range(20)],
            "pin": 0, "busy": [], "queue": [], "takeoffs": {"uav0": (0, 0.0)}, "free_from": None, "last_land": None,
            "lost": [], "standby": ["uav0"], "calls": {}}


def test_a_messenger_flies_to_where_the_other_uav_waits_and_lands_after():
    xy = (325.0, 800.0)  # 300 m off the road above node 6
    uav = _messenger(_plan())
    uav.call = {"for": None, "xy": xy, "near": None, "deliver": "uav1"}
    sortie = uav.plan_sortie((0.0, 500.0), 5.0, 2000.0, 0.0)
    air = uav.planner.air.positions
    assert len(sortie.tour) == 1 and np.hypot(*(air[sortie.tour[0]] - xy)) == pytest.approx(np.hypot(*(air - xy).T).min())
    # The dock taken for longer than it can fly: it does not go.
    uav = _messenger(_plan(arrive_from=1e5))
    uav.call = {"for": None, "xy": xy, "near": None, "deliver": "uav1"}
    assert uav.plan_sortie((0.0, 500.0), 5.0, 2000.0, 0.0) is None


def test_a_messenger_waits_there_until_it_hears_the_other_uav():
    from marsupial_search.core import Hold
    uav = _messenger(_plan())
    point, land = (325.0, 800.0), (500.0, 500.0)
    uav.phase, uav.t0 = "sortie", 10.0
    uav.task, uav.booking, uav.waypoints, uav.tour = {"deliver": "uav1"}, {"t_land": 1000.0}, [point, land], [7]
    s = RobotState("uav0", RobotType.UAV, point, RobotMode.ACTIVE, 1500.0, 2000.0, True, None, None, (), (), None)
    cmds = []
    assert uav._wait_to_deliver(s, 100.0, cmds) and cmds == [Hold()]
    uav.relay.heard("uav1", {}, 101.0)  # in radio range: relaying hands over the UGV's plan
    assert not uav._wait_to_deliver(s, 101.0, []) and uav.delivered
    # Not heard, but it must leave now to land on time.
    uav.delivered, uav.relay.neighbours = False, {}
    assert not uav._wait_to_deliver(s, 1000.0 - 30.0, []) and uav.delivered
