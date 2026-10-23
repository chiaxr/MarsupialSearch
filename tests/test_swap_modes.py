import dataclasses

import numpy as np
import pytest

from marsupial_search.controllers import DEFAULT_TEAM, make_team
from marsupial_search.core import EventType
from marsupial_search.sim import ScenarioConfig, SimConfig, Simulator, generate_scenario


@pytest.fixture(scope="module")
def standstill():
    """The 2.5 km test map, with a dock that only swaps batteries while the UGV stands still."""
    sc = generate_scenario(ScenarioConfig(size_m=(2500.0, 2500.0), n_urban=2, urban_half_size_m=(200.0, 300.0),
                                          n_rural_nodes=25, seed=3))
    return dataclasses.replace(sc, docking=dataclasses.replace(sc.docking, swap_while_moving=False))


@pytest.mark.parametrize("team", ["joint_planner", DEFAULT_TEAM])
def test_controllers_hold_the_ugv_for_every_swap(standstill, team):
    res = Simulator(standstill, SimConfig(terminate_on=(), max_time_s=2 * 3600)).run(
        make_team(team, {"joint_planner": {"iterations": 100}}))
    s = res.summary()
    assert s["depleted"] == 0 and s["rejected"] == 0 and s["dock_aborts"] == 0 and s["undock_aborts"] == 0
    docked = [e.time for e in res.events if e.type == EventType.DOCKED]
    assert len(docked) >= 2
    swap, t, ugv = standstill.docking.swap_duration_s, res.traj_t, res.trajectories["ugv0"]
    for t_dock in docked:  # the UGV does not move until the swap is done
        during = (t >= t_dock) & (t <= t_dock + swap)
        if during.sum() > 1 and t_dock + swap <= t[-1]:
            assert np.ptp(ugv[during], axis=0).max() < 1e-6
