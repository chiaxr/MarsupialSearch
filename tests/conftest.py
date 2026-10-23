import numpy as np
import pytest

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
from marsupial_search.sim import RobotInit, Scenario
from marsupial_search.sim.generator import _grid_graph


def make_line_scenario(n_uav=1, docking=None, target_xy=(950.0, 975.0), uav_capacity=2000.0,
                       urban_right_half=False) -> Scenario:
    """1 km x 1 km map; a straight road along y=500 with nodes every 50 m. ugv0 starts at
    x=25 carrying uav0; uav{i} for i >= 1 hovers over the road at x = 25 + 100 i."""
    grid = GridSpec(origin=(0.0, 0.0), resolution=25.0, shape=(40, 40))
    terrain = np.full(grid.shape, int(Terrain.VEGETATION), dtype=np.int8)
    if urban_right_half:
        terrain[:, 20:] = int(Terrain.URBAN)
    lf = np.zeros((N_LAYERS,) + grid.shape)
    urban = terrain == int(Terrain.URBAN)
    lf[Layer.SHARED][~urban] = 1.0
    lf[Layer.AERIAL][urban] = 0.5
    lf[Layer.GROUND][urban] = 0.5
    xs = np.arange(25.0, 1000.0, 50.0)
    road = TraversabilityGraph(np.stack([xs, np.full_like(xs, 500.0)], 1),
                               np.stack([np.arange(len(xs) - 1), np.arange(1, len(xs))], 1), "road")
    uav_graph = _grid_graph((1000.0, 1000.0), 80.0)
    prior = np.full(grid.shape, 1.0 / terrain.size)
    robots = {"ugv0": RobotSpec("ugv0", RobotType.UGV, 5.0, EnergySpec(100_000, 1.0, 1.0),
                                SensorSpec(50.0, (Layer.SHARED, Layer.GROUND)))}
    init = {"ugv0": RobotInit(node=0)}
    for i in range(n_uav):
        rid = f"uav{i}"
        robots[rid] = RobotSpec(rid, RobotType.UAV, 10.0, EnergySpec(uav_capacity, 1.0, 0.05),
                                SensorSpec(60.0, (Layer.SHARED, Layer.AERIAL)))
        init[rid] = RobotInit(docked_on="ugv0") if i == 0 else RobotInit(position=tuple(road.positions[2 * i]))
    return Scenario(
        name="line", grid=grid, terrain=terrain, layer_fractions=lf, prior=prior,
        graphs={RobotType.UGV: road, RobotType.UAV: uav_graph}, robots=robots, init=init,
        docking=docking or DockingSpec(dock_duration_s=10, undock_duration_s=5, swap_duration_s=100),
        target_xy=target_xy, target_layer=Layer.SHARED,
    )


@pytest.fixture
def line_scenario():
    return make_line_scenario()
