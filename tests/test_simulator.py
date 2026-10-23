import dataclasses

import numpy as np
import pytest

from conftest import make_line_scenario
from marsupial_search.core import Dock, DockingSpec, EventType, FollowPath, GoToPoint, Hold, RobotMode, RobotType, Undock
from marsupial_search.sim import SimConfig, Simulator


def _types(obs):
    return [e.type for e in obs.events]


def _run(sim, cmds, n):
    obs = None
    for _ in range(n):
        if sim.done:
            break
        obs = sim.step(cmds)
        cmds = {}
    return obs


def test_undock_requires_stationary_ugv(line_scenario):
    sim = Simulator(line_scenario)
    obs = sim.step({"ugv0": FollowPath((0, 1, 2)), "uav0": Undock()})
    assert EventType.COMMAND_REJECTED in _types(obs)
    assert obs.robots["uav0"].mode == RobotMode.DOCKED


def test_undock_completes_after_duration(line_scenario):
    sim = Simulator(line_scenario)
    obs = sim.step({"uav0": Undock()})
    assert obs.robots["uav0"].mode == RobotMode.UNDOCKING
    obs = _run(sim, {}, 5)
    assert obs.robots["uav0"].mode == RobotMode.ACTIVE
    assert obs.robots["ugv0"].docked_uav is None


def test_undock_aborts_if_ugv_moves(line_scenario):
    sim = Simulator(line_scenario)
    sim.step({"uav0": Undock()})
    obs = sim.step({"ugv0": FollowPath((0, 1))})
    assert EventType.UNDOCK_ABORTED in _types(obs)
    assert obs.robots["uav0"].mode == RobotMode.DOCKED


def test_docked_uav_rides_along_without_draining(line_scenario):
    sim = Simulator(line_scenario)
    obs = _run(sim, {"ugv0": FollowPath((0, 1, 2, 3))}, 20)
    u, g = obs.robots["uav0"], obs.robots["ugv0"]
    assert u.mode == RobotMode.DOCKED
    assert u.position == g.position
    assert u.energy == pytest.approx(u.energy_capacity)
    assert g.energy < g.energy_capacity


def _launch(sim):
    sim.step({"uav0": Undock()})
    return _run(sim, {}, 5)


def test_dock_aborts_if_ugv_moves_and_completes_when_still(line_scenario):
    sim = Simulator(line_scenario)
    _launch(sim)
    obs = sim.step({"uav0": Dock("ugv0")})
    assert obs.robots["uav0"].mode == RobotMode.DOCKING
    obs = sim.step({"ugv0": FollowPath((0, 1))})
    assert EventType.DOCK_ABORTED in _types(obs)
    assert obs.robots["uav0"].mode == RobotMode.ACTIVE

    obs = _run(sim, {"ugv0": Hold()}, 1)
    obs = _run(sim, {"uav0": GoToPoint(obs.robots["ugv0"].position)}, 5)
    obs = sim.step({"uav0": Dock("ugv0")})
    assert obs.robots["uav0"].mode == RobotMode.DOCKING
    obs = _run(sim, {}, 10)
    assert obs.robots["uav0"].mode == RobotMode.DOCKED
    assert EventType.DOCKED in [e.type for e in sim.events]


def test_dock_rejected_when_far_or_moving(line_scenario):
    sim = Simulator(line_scenario)
    _launch(sim)
    obs = _run(sim, {"uav0": GoToPoint((500.0, 900.0))}, 60)
    obs = sim.step({"uav0": Dock("ugv0")})
    assert EventType.COMMAND_REJECTED in _types(obs)


def test_the_dock_takes_one_uav_at_a_time():
    sim = Simulator(make_line_scenario(n_uav=2))  # uav0 docked, uav1 hovering 100 m away
    obs = _run(sim, {"uav1": GoToPoint(sim.setup.graphs[RobotType.UGV].positions[0])}, 20)
    obs = sim.step({"uav1": Dock("ugv0")})
    assert any(e.type == EventType.COMMAND_REJECTED and "occupied" in e.detail for e in obs.events)


def test_the_simulator_models_exactly_one_ugv(line_scenario):
    spec = line_scenario.robots["ugv0"]
    two = dataclasses.replace(line_scenario, robots={**line_scenario.robots, "ugv1": dataclasses.replace(
        spec, robot_id="ugv1")}, init={**line_scenario.init, "ugv1": line_scenario.init["ugv0"]})
    with pytest.raises(ValueError, match="exactly one"):
        Simulator(two)


def _fly_and_land(sim):
    _launch(sim)
    obs = _run(sim, {"uav0": GoToPoint((500.0, 900.0))}, 100)
    obs = _run(sim, {"uav0": GoToPoint(obs.robots["ugv0"].position)}, 80)
    sim.step({"uav0": Dock("ugv0")})
    obs = _run(sim, {}, 10)
    assert obs.robots["uav0"].mode == RobotMode.DOCKED
    return obs


def test_battery_swap_takes_its_duration_and_ends_full(line_scenario):
    sim = Simulator(line_scenario)
    obs = _fly_and_land(sim)
    e_landed, left = obs.robots["uav0"].energy, obs.robots["uav0"].mode_time_remaining
    assert e_landed < 0.9 * obs.robots["uav0"].energy_capacity and 90.0 < left <= 100.0  # 100 s swap, just started
    obs = sim.step({"uav0": Undock(), "ugv0": FollowPath((0, 1, 2))})  # the UGV may drive during the swap
    assert EventType.COMMAND_REJECTED in _types(obs) and obs.robots["uav0"].energy == e_landed
    assert obs.robots["uav0"].mode_time_remaining == pytest.approx(left - 1.0)
    obs = _run(sim, {}, int(left) - 1)
    assert obs.robots["uav0"].energy_fraction == pytest.approx(1.0) and obs.robots["uav0"].mode_time_remaining == 0
    obs = _run(sim, {"uav0": Undock()}, 6)
    assert obs.robots["uav0"].mode == RobotMode.ACTIVE


def test_battery_swap_pauses_while_the_ugv_moves_if_it_must_stand_still():
    sc = make_line_scenario(docking=DockingSpec(dock_duration_s=10, undock_duration_s=5, swap_duration_s=100,
                                                swap_while_moving=False))
    sim = Simulator(sc)
    left = _fly_and_land(sim).robots["uav0"].mode_time_remaining
    obs = _run(sim, {"ugv0": FollowPath(tuple(range(19)))}, 50)  # drives the whole time
    assert obs.robots["ugv0"].moving and obs.robots["uav0"].mode_time_remaining == pytest.approx(left)
    obs = _run(sim, {"ugv0": Hold()}, int(left) + 1)
    assert obs.robots["uav0"].energy_fraction == pytest.approx(1.0)


def test_energy_depletes_with_motion_and_uav_can_run_flat():
    sc = make_line_scenario(uav_capacity=50.0)
    sim = Simulator(sc)
    _launch(sim)
    obs = _run(sim, {"uav0": GoToPoint((500.0, 990.0))}, 60)
    assert obs.robots["uav0"].mode == RobotMode.DEPLETED
    assert EventType.DEPLETED in [e.type for e in sim.events]


def test_ugv_path_must_follow_edges(line_scenario):
    sim = Simulator(line_scenario)
    obs = sim.step({"ugv0": FollowPath((0, 5))})
    assert EventType.COMMAND_REJECTED in _types(obs)
    obs = sim.step({"ugv0": FollowPath((3, 4))})  # does not start at/next to node 0
    assert EventType.COMMAND_REJECTED in _types(obs)
    obs = sim.step({"ugv0": GoToPoint((10.0, 10.0))})
    assert EventType.COMMAND_REJECTED in _types(obs)


def test_ugv_can_reverse_mid_edge(line_scenario):
    sim = Simulator(line_scenario)
    obs = _run(sim, {"ugv0": FollowPath((0, 1))}, 3)  # 15 m along a 50 m edge
    assert obs.robots["ugv0"].node is None
    obs = sim.step({"ugv0": FollowPath((0,))})
    assert EventType.COMMAND_REJECTED not in _types(obs)


def test_ugv_in_range_ends_mission_uav_only_sights():
    sc = make_line_scenario(target_xy=(225.0, 625.0))
    sim = Simulator(sc)
    _launch(sim)
    obs = _run(sim, {"uav0": GoToPoint((225.0, 625.0))}, 25)
    assert EventType.TARGET_SIGHTED in [e.type for e in sim.events]
    assert not sim.done
    # The target is ~125 m off the road: the UGV (50 m range) never reaches it from the road.
    sc2 = make_line_scenario(target_xy=(225.0, 525.0))
    sim2 = Simulator(sc2)
    obs = _run(sim2, {"ugv0": FollowPath((0, 1, 2, 3, 4))}, 60)
    assert sim2.done and sim2.outcome == "found" and sim2.found_by == "ugv0"
    assert obs.mission_complete


def test_terminate_on_is_configurable():
    sc = make_line_scenario(target_xy=(225.0, 625.0))
    sim = Simulator(sc, SimConfig(terminate_on=(RobotType.UGV, RobotType.UAV)))
    _launch(sim)
    _run(sim, {"uav0": GoToPoint((225.0, 625.0))}, 25)
    assert sim.outcome == "found" and sim.found_by == "uav0"


def test_urban_needs_both_robot_types():
    sc = make_line_scenario(urban_right_half=True, target_xy=(25.0, 25.0))
    sim = Simulator(sc, SimConfig(terminate_on=()))
    sim.step({"uav0": Undock()})
    _run(sim, {}, 5)
    cell = sc.grid.world_to_cell(725.0, 525.0)  # urban cell next to the road
    init = sim.truth_belief.initial_mass[:, cell[0], cell[1]].sum()

    _run(sim, {"uav0": GoToPoint((725.0, 525.0))}, 80)
    after_uav = sim.truth_belief.mass[:, cell[0], cell[1]].sum()
    assert after_uav == pytest.approx(0.5 * init)

    _run(sim, {"ugv0": FollowPath(tuple(range(0, 16)))}, 200)
    assert sim.truth_belief.mass[:, cell[0], cell[1]].sum() == pytest.approx(0.0)
