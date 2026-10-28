import ast
import dataclasses
import pathlib

import numpy as np
import pytest

from marsupial_search.core import Layer, RobotType, SensorSpec, Terrain, TraversabilityGraph
from marsupial_search.sim import SCENARIO_BASES, ScenarioConfig, generate_scenario, resample_target
from marsupial_search.sim.generator import FleetConfig
from marsupial_search.sim.scenario_bases import small

ROOT = pathlib.Path(__file__).resolve().parents[1] / "marsupial_search"


@pytest.mark.parametrize("package", ["core", "controllers"])
def test_portable_packages_do_not_import_sim(package):
    banned = ("marsupial_search.sim", "marsupial_search.viz", "matplotlib")
    for path in (ROOT / package).rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
                if node.level and package == "controllers" and (node.module or "").startswith("sim"):
                    names = ["marsupial_search.sim"]
            for n in names:
                assert not n.startswith(banned), f"{path.name} imports {n}"


def test_graph_searches():
    # 0 - 1 - 2
    #     |
    #     3 - 4
    pos = np.array([[0, 0], [1, 0], [2, 0], [1, -1], [2, -1]], float)
    g = TraversabilityGraph(pos, np.array([[0, 1], [1, 2], [1, 3], [3, 4]]))
    path, d = g.shortest_path(0, 4)
    assert path == [0, 1, 3, 4] and d == pytest.approx(3.0)
    cand = np.array([False, False, True, False, True])
    path, d = g.nearest_matching(0, cand)
    assert path == [0, 1, 2] and d == pytest.approx(2.0)
    dist, pred = g.distances_from(0)
    assert g.path_from_predecessors(pred, 0, 4) == [0, 1, 3, 4]


@pytest.fixture(scope="module")
def small_scenario():
    cfg = ScenarioConfig(size_m=(2500.0, 2500.0), n_urban=2, urban_half_size_m=(200.0, 300.0), n_rural_nodes=25,
                         seed=3)
    return generate_scenario(cfg)


def test_generated_scenario_invariants(small_scenario):
    sc = small_scenario
    road, uav = sc.graphs[RobotType.UGV], sc.graphs[RobotType.UAV]
    assert road.n_components() == 1 and uav.n_components() == 1
    assert sc.prior.sum() == pytest.approx(1.0) and (sc.prior > 0).all()
    assert np.allclose(sc.layer_fractions.sum(0), 1.0)
    # Every UGV graph node sits on a drivable cell (road, or a street inside a town).
    for p in road.positions:
        assert sc.terrain[sc.grid.world_to_cell(*p)] in (Terrain.ROAD, Terrain.URBAN)
    assert set(np.unique(sc.terrain)) == {int(t) for t in Terrain}
    for t in Terrain:
        assert (sc.terrain == t).any(), f"no {t.name} cells"
    assert sc.meta["confirmable_mask"][sc.grid.world_to_cell(*sc.target_xy)]


def test_fleet_is_one_ugv_carrying_uav0_with_uav1_hovering_over_it(small_scenario):
    sc = small_scenario
    assert {r: spec.robot_type for r, spec in sc.robots.items()} == {
        "ugv0": RobotType.UGV, "uav0": RobotType.UAV, "uav1": RobotType.UAV}
    assert sc.init["uav0"].docked_on == "ugv0"
    road = sc.graphs[RobotType.UGV]
    assert sc.init["uav1"].docked_on is None and sc.init["uav1"].position == tuple(road.positions[sc.init["ugv0"].node])


def test_sensor_detection_per_layer():
    s = SensorSpec(60.0, (Layer.SHARED, Layer.AERIAL), 0.9, layer_p_detect=((Layer.GROUND, 0.3),))
    assert s.layer_p().tolist() == [0.9, 0.9, 0.3]
    assert s.seen_layers == (0, 1, 2)
    assert SensorSpec(60.0, (Layer.SHARED, Layer.AERIAL)).seen_layers == (int(Layer.SHARED), int(Layer.AERIAL))


def test_uavs_see_street_level_only_with_a_ground_detection_probability():
    cfg = SCENARIO_BASES["small"](0)
    sc = generate_scenario(cfg)
    assert all(sc.robots[r].sensor.layer_p()[Layer.GROUND] == 0 for r in ("uav0", "uav1"))
    cfg.fleet = FleetConfig(uav_ground_p_detect=0.3, uav_ground_p_detects={"uav0": 0.5})
    sc = generate_scenario(cfg)
    assert [sc.robots[r].sensor.layer_p()[Layer.GROUND] for r in ("ugv0", "uav0", "uav1")] == [1.0, 0.5, 0.3]
    with pytest.raises(ValueError, match="uav_ground_p_detects"):
        generate_scenario(ScenarioConfig(fleet=FleetConfig(uav_ground_p_detects={"uav7": 0.5})))


def test_semantic_prior_follows_labels_and_target_sits_on_a_label():
    from marsupial_search.sim import SEMANTIC_LABELS, resample_target

    cfg = ScenarioConfig(size_m=(2500.0, 2500.0), n_urban=2, urban_half_size_m=(200.0, 300.0), n_rural_nodes=25,
                         seed=3, target_object="car")
    sc = generate_scenario(cfg)
    lab = sc.meta["label_mask"]
    assert np.array_equal(lab, np.isin(sc.terrain, [int(Terrain.ROAD), int(Terrain.URBAN)]))
    # Labelled cells carry far more prior per cell than unlabelled ones.
    assert sc.prior[lab].mean() > 10 * sc.prior[~lab].mean()
    rng = np.random.default_rng(0)
    for _ in range(20):
        s2 = resample_target(sc, rng)
        cell = s2.grid.world_to_cell(*s2.target_xy)
        assert lab[cell] and s2.meta["confirmable_mask"][cell]


def test_semantic_prior_is_exact_without_noise_blur_or_floor():
    cfg = ScenarioConfig(size_m=(2500.0, 2500.0), n_urban=2, urban_half_size_m=(200.0, 300.0), n_rural_nodes=25,
                         seed=3, prior_segmentation_error=0.0,
                         prior_mask_blur_m=0.0, prior_background_weight=0.0)
    sc = generate_scenario(cfg)
    lab = sc.meta["label_mask"]
    assert np.allclose(sc.prior[lab], 1.0 / lab.sum()) and (sc.prior[~lab] == 0).all()


def test_hiker_is_on_roadside_or_in_vegetation_seen_from_a_road_node():
    from scipy.spatial import cKDTree

    from marsupial_search.sim import resample_target

    cfg = ScenarioConfig(size_m=(2500.0, 2500.0), n_urban=2, urban_half_size_m=(200.0, 300.0), n_rural_nodes=25,
                         seed=3, target_object="hiker")
    sc = generate_scenario(cfg)
    road = sc.graphs[RobotType.UGV]
    tree = cKDTree(road.positions)
    rng = np.random.default_rng(0)
    kinds = set()
    for _ in range(200):
        s2 = resample_target(sc, rng)
        r, c = s2.grid.world_to_cell(*s2.target_xy)
        t = sc.terrain[r, c]
        if t == Terrain.ROAD:
            nb = sc.terrain[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2]
            assert (nb == Terrain.VEGETATION).any()
        else:
            assert t == Terrain.VEGETATION
            assert tree.query(s2.target_xy)[0] <= cfg.fleet.ugv_sensor_range
        kinds.add(int(t))
    assert kinds == {int(Terrain.ROAD), int(Terrain.VEGETATION)}


def test_the_target_may_lie_where_the_ugv_cannot_confirm_it():
    """With ``target_ugv_confirmable`` off, the target is drawn from the whole target distribution,
    e.g. in parts of the road network that blockages cut off."""
    cfg = dataclasses.replace(small(3), road_blocked_fraction=0.1, road_cutoff_fraction=0.2)
    on = generate_scenario(cfg)
    off = generate_scenario(dataclasses.replace(cfg, target_ugv_confirmable=False))
    assert np.array_equal(on.meta["target_mask"], on.meta["confirmable_mask"]) and off.meta["target_mask"].all()
    assert off.meta["road_cutoff_nodes"] > 0
    rng = np.random.default_rng(0)
    cells = [off.grid.world_to_cell(*resample_target(off, rng).target_xy) for _ in range(300)]
    outside = [rc for rc in cells if not off.meta["confirmable_mask"][rc]]
    assert outside and len(outside) < len(cells)
    assert all(on.meta["confirmable_mask"][on.grid.world_to_cell(*resample_target(on, rng).target_xy)] for _ in range(50))
