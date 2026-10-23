import numpy as np
import pytest

from conftest import make_line_scenario
from marsupial_search.controllers import DEFAULT_TEAM, make_team
from marsupial_search.core import EnergySpec, FollowPath, Layer, RobotMode, RobotSpec, RobotType, Terrain
from marsupial_search.sim import SCENARIO_BASES, SimConfig, Simulator, generate_scenario


@pytest.fixture(scope="module")
def small():
    return generate_scenario(SCENARIO_BASES["small"](0))


def test_small_base_is_one_ugv_two_uavs(small):
    types = [s.robot_type for s in small.robots.values()]
    assert types.count(RobotType.UGV) == 1 and types.count(RobotType.UAV) == 2
    assert small.grid.shape == (60, 60)
    assert small.graphs[RobotType.UGV].n_components() == 1
    for t in Terrain:
        assert (small.terrain == t).any(), f"no {t.name} cells"


@pytest.mark.parametrize("team", ["joint_planner", DEFAULT_TEAM])
def test_small_base_ends_on_target_or_ugv_depletion(small, team):
    sim_cfg = SimConfig()
    res = Simulator(small, sim_cfg).run(make_team(team))
    assert res.outcome in ("found", "ugv_depleted")
    if res.outcome == "found":
        assert res.found_by == "ugv0" and not res.robot_stats["ugv0"]["depleted"]
    else:
        assert res.robot_stats["ugv0"]["depleted"]
    assert res.time_s < sim_cfg.max_time_s


def test_small_base_ends_on_depletion_when_target_unreachable(small):
    # A target the UGV cannot get within sensor range of is never found, so the battery decides.
    unreachable = np.argwhere(~small.meta["confirmable_mask"])[0]
    sc = small.with_target(small.grid.cell_center(*unreachable), Layer.SHARED)
    res = Simulator(sc, SimConfig()).run(make_team(DEFAULT_TEAM))
    assert res.outcome == "ugv_depleted"
    assert res.robot_stats["ugv0"]["depleted"]


def _flat_ugv_scenario():
    sc = make_line_scenario()
    spec = sc.robots["ugv0"]
    sc.robots["ugv0"] = RobotSpec(spec.robot_id, spec.robot_type, spec.speed, EnergySpec(60.0, 1.0, 1.0), spec.sensor)
    return sc


def test_sim_stops_when_the_ugv_is_depleted_even_if_uav_alive():
    sim = Simulator(_flat_ugv_scenario(), SimConfig(terminate_on=()))
    obs = sim.step({"ugv0": FollowPath(tuple(range(10)))})
    while not sim.done:
        obs = sim.step({})
    assert sim.outcome == "ugv_depleted"
    assert obs.robots["ugv0"].mode == RobotMode.DEPLETED
    assert obs.robots["uav0"].mode == RobotMode.DOCKED


def test_ugv_depletion_stop_can_be_disabled():
    sim = Simulator(_flat_ugv_scenario(), SimConfig(terminate_on=(), stop_when_ugv_depleted=False, max_time_s=200))
    sim.step({"ugv0": FollowPath(tuple(range(10)))})
    while not sim.done:
        sim.step({})
    assert sim.outcome == "timeout"
