import pytest

from marsupial_search.controllers import make_team, team_kind
from marsupial_search.controllers.joint_planner import JointPlannerConfig, JointPlannerController
from marsupial_search.core import EventType
from marsupial_search.sim import ScenarioConfig, SimConfig, Simulator, generate_scenario

FAST = {"joint_planner": {"iterations": 100}}


@pytest.fixture(scope="module")
def scenario():
    cfg = ScenarioConfig(size_m=(2500.0, 2500.0), n_urban=2, urban_half_size_m=(200.0, 300.0), n_rural_nodes=25,
                         seed=3)
    return generate_scenario(cfg)


def _planner(scenario, **cfg):
    sim = Simulator(scenario)
    ctl = JointPlannerController(JointPlannerConfig(**cfg))
    ctl.reset(sim.setup)
    obs = sim.reset()
    return ctl, ctl._context(obs.robots, obs.time)


def test_registered_as_centralised():
    assert team_kind("joint_planner") == "centralised"


def test_search_never_returns_worse_than_its_best_start(scenario):
    ctl, ctx = _planner(scenario, iterations=200)
    starts = ctl._starts(ctx)
    start_values = [ctl._evaluate(ctx, p) for p in starts]
    assert all(e.ok for e in start_values)
    best = ctl._evaluate(ctx, ctl._search(ctx, starts))
    assert best.ok and best.value >= max(e.value for e in start_values)


def test_starting_plans_keep_every_sortie_above_the_reserve(scenario):
    ctl, ctx = _planner(scenario)
    for plan in ctl._starts(ctx):
        ev = ctl._evaluate(ctx, plan)
        reserve = ctl.cfg.uav_reserve_fraction * scenario.robots["uav0"].energy.capacity
        assert ev.ok and all(s.e_end >= reserve for s in ev.sorties)
        assert all(a <= b for a, b in zip(ev.arrive, ev.arrive[1:]))  # the UGV's timeline only moves forward


def test_sighting_delay_only_discounts_uav_looks(scenario):
    ctl, ctx = _planner(scenario)
    plan = ctl._starts(ctx)[0]
    with_delay = ctl._evaluate(ctx, plan).value
    ctl.cfg.sighting_delay_weight = 0.0
    assert ctl._evaluate(ctx, plan).value >= with_delay > 0


def test_full_coverage_without_losing_uavs(scenario):
    res = Simulator(scenario, SimConfig(terminate_on=(), max_time_s=6 * 3600)).run(make_team("joint_planner", FAST))
    s = res.summary()
    assert res.outcome == "controller_done" and s["searched_prob"] > 0.99
    assert s["depleted"] == 0 and s["sorties"] >= 4
    assert s["rejected"] == 0 and s["dock_aborts"] == 0 and s["undock_aborts"] == 0
    assert {e.robot_id for e in res.events if e.type == EventType.DOCKED} == {"uav0", "uav1"}
    assert all(res.log[f"energy_{u}"].min() > 0.09 for u in ("uav0", "uav1"))


def test_finds_target(scenario):
    res = Simulator(scenario, SimConfig()).run(make_team("joint_planner", FAST))
    assert res.outcome == "found" and res.found_by == "ugv0"
    assert res.summary()["depleted"] == 0


def test_runs_are_reproducible(scenario):
    cfg = SimConfig(max_time_s=3600)
    a, b = (Simulator(scenario, cfg).run(make_team("joint_planner", {"joint_planner": {"iterations": 50}}))
            for _ in range(2))
    assert a.time_s == b.time_s and a.outcome == b.outcome
    assert all((a.trajectories[r] == b.trajectories[r]).all() for r in a.trajectories)
