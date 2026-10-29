"""Occlusion (``sim.visibility``): what the canopy and the street canyons let each viewpoint see, of
the target and of the road, and how the generator and simulator use it. The robots never get the
occlusion: the simulator gives each its detection probabilities (``MissionSetup.detection``) and
reports the road nodes a look could not make out (``TraversabilityUpdate.hidden``)."""

import dataclasses

import numpy as np
import pytest

from conftest import make_line_scenario
from marsupial_search.controllers import make_team
from marsupial_search.controllers.ctp import BlockageModel
from marsupial_search.controllers.sortie_planner import SortieConfig, SortiePlanner
from marsupial_search.controllers.ugv_timed_route import TimedRouteUgv
from marsupial_search.core import (
    EventType,
    GoToPoint,
    GraphKnowledge,
    Layer,
    RobotMode,
    RobotState,
    RobotType,
    SensorSpec,
    Terrain,
    Undock,
)
from marsupial_search.sim import SimConfig, Simulator, generate_scenario, resample_target
from marsupial_search.sim.scenario import MapMismatch
from marsupial_search.sim.visibility import Occlusion, detection_map, road_observability
from marsupial_search.sim.scenario_bases import small

S, A, G = int(Layer.SHARED), int(Layer.AERIAL), int(Layer.GROUND)


def _occluded(canopy=0.6, canyon=2.0, ground_veg=0.5, air_p=0.8, uav="air", **kw):
    """The line map with a town on its right half (canyon ``canyon``), trees on the left
    (``canopy``), and uav0 seeing street level from the air with per-look probability 1 (or, with
    ``uav="ground"``, flying at street level)."""
    sc = make_line_scenario(urban_right_half=True, **kw)
    urban = sc.terrain == int(Terrain.URBAN)
    occ = Occlusion(canopy=np.where(urban, 0.0, canopy), canyon=np.where(urban, canyon, 0.0),
                    ground_veg_occlusion=ground_veg, air_p=air_p, ground_p=1.0)
    sensor = SensorSpec(60.0, (Layer.SHARED, Layer.AERIAL), layer_p_detect=((Layer.GROUND, 1.0),)) if uav == "air" \
        else SensorSpec(60.0, (Layer.SHARED, Layer.GROUND), viewpoint="ground")
    robots = {**sc.robots, "uav0": dataclasses.replace(sc.robots["uav0"], sensor=sensor)}
    return dataclasses.replace(sc, occlusion=occ, robots=robots)


# ------------------------------------------------------------------- the model
def test_with_no_occlusion_a_look_detects_with_the_sensors_own_probability():
    sc = make_line_scenario(urban_right_half=True)
    for spec in sc.robots.values():
        p = detection_map(spec, sc.terrain, None)
        assert p.shape == (3,) + sc.grid.shape and np.array_equal(p, np.broadcast_to(
            spec.sensor.layer_p()[:, None, None], p.shape))
        assert (road_observability(spec, sc.grid, sc.terrain, None, sc.graphs[RobotType.UGV].positions) == 1).all()


def test_each_viewpoint_sees_what_occlusion_leaves_it():
    sc = _occluded()
    urban = sc.terrain == int(Terrain.URBAN)
    air = detection_map(sc.robots["uav0"], sc.terrain, sc.occlusion)
    # From the air: roofs and the open (times the class's 0.8); street level by the canyon (1 / (2 · 2));
    # under the canopy (the ground layer outside town), nothing.
    assert np.allclose(air[S], 0.8) and np.allclose(air[A], 0.8)
    assert np.allclose(air[G][urban], 0.8 * 0.25) and np.allclose(air[G][~urban], 0.0)
    # From the ground: never a roof; through the trees, 1 - 0.5 · 0.6 of what it sees.
    ugv = detection_map(sc.robots["ugv0"], sc.terrain, sc.occlusion)
    assert np.allclose(ugv[A], 0.0) and np.allclose(ugv[[S, G]][:, urban], 1.0)
    assert np.allclose(ugv[[S, G]][:, ~urban], 0.7)
    # A UAV at street level with a front-facing camera sees as the UGV does.
    low = _occluded(uav="ground")
    assert np.array_equal(detection_map(low.robots["uav0"], low.terrain, low.occlusion), ugv)
    # Road nodes: from the air, under the trees by 1 - canopy, in town by the canyon; from the ground, all.
    road = sc.graphs[RobotType.UGV].positions
    o = road_observability(sc.robots["uav0"], sc.grid, sc.terrain, sc.occlusion, road)
    assert np.allclose(o[road[:, 0] < 500], 0.4) and np.allclose(o[road[:, 0] > 500], 0.25)
    assert (road_observability(sc.robots["ugv0"], sc.grid, sc.terrain, sc.occlusion, road) == 1).all()
    assert road_observability(sc.robots["uav0"], sc.grid, sc.terrain, sc.occlusion, np.array([[np.nan, np.nan]])) == 0


# --------------------------------------------------------------- the generator
@pytest.mark.parametrize("obj", ["car", "hiker"])
def test_the_generator_splits_layers_by_canopy_and_target_class(obj):
    cfg = dataclasses.replace(small(1), occlusion=True, target_object=obj,
                              fleet=dataclasses.replace(small(1).fleet, uav_viewpoints={"uav0": "ground"}))
    sc = generate_scenario(cfg)
    occ, urban = sc.occlusion, sc.terrain == int(Terrain.URBAN)
    assert ((occ.canopy >= 0) & (occ.canopy <= 1)).all() and (occ.canopy[urban] == 0).all() and (occ.canyon[~urban] == 0).all()
    assert occ.canopy[sc.terrain == int(Terrain.VEGETATION)].mean() > occ.canopy[sc.terrain == int(Terrain.ROAD)].mean() > 0
    lf = sc.layer_fractions
    assert np.allclose(lf.sum(0), 1.0) and np.allclose(lf[G][~urban], occ.canopy[~urban])
    assert np.allclose(lf[A][urban], 0.2 if obj == "hiker" else 0.0)
    assert sc.robots["uav0"].sensor.viewpoint == "ground" and sc.robots["uav0"].sensor.seen_layers == (S, G)
    assert dict(sc.robots["uav1"].sensor.layer_p_detect) == {Layer.GROUND: 1.0}  # the canyon sets the chance
    # Confirmable targets: only where, and in a layer, the UGV's look can detect them (never on a roof).
    ugv = detection_map(sc.robots["ugv0"], sc.terrain, occ)
    rng = np.random.default_rng(0)
    for _ in range(100):
        t = resample_target(sc, rng)
        assert ugv[(int(t.target_layer),) + sc.grid.world_to_cell(*t.target_xy)] > 0


def test_without_occlusion_the_generator_is_as_before():
    sc = generate_scenario(small(1))
    assert sc.occlusion is None and not sc.robots["uav1"].sensor.layer_p_detect
    with pytest.raises(ValueError, match="viewpoint"):
        generate_scenario(dataclasses.replace(small(1), fleet=dataclasses.replace(
            small(1).fleet, uav_viewpoints={"uav0": "low"})))


def test_roads_under_trees_are_more_often_missing_from_the_map():
    under, all_ = [], []
    for seed in range(6):
        sc = generate_scenario(dataclasses.replace(small(seed), occlusion=True, road_unmapped_fraction=0.2))
        road = sc.true_road(blocked=False)
        r, c = (np.clip((road.positions[:, k] // sc.grid.resolution).astype(int), 0, n - 1)
                for k, n in ((1, sc.grid.rows), (0, sc.grid.cols)))
        canopy = sc.occlusion.canopy[r, c]
        under.append(canopy[sorted(sc.road_mismatch.new_nodes)].mean())
        all_.append(canopy.mean())
    assert np.mean(under) > np.mean(all_)


# --------------------------------------------------------------- the simulator
def _fly_over(sc, xy, seed, confirm_test="range"):
    """uav0 takes off and flies over ``xy`` once; the simulator after that."""
    sim = Simulator(sc, SimConfig(terminate_on=(), seed=seed, confirm_test=confirm_test))
    sim.step({"uav0": Undock()})
    for _ in range(5):
        sim.step({})
    cmds = {"uav0": GoToPoint(xy)}
    for _ in range(120):
        sim.step(cmds)
        cmds = {}
    return sim


@pytest.mark.parametrize("layer, uav, rate", [(G, "air", 0.5), (G, "ground", 1.0), (A, "air", 1.0), (A, "ground", 0.0)])
def test_street_level_shows_from_the_air_by_chance_and_roofs_never_from_street_level(layer, uav, rate):
    xy = (725.0, 725.0)  # in town, 225 m off the road
    sc = dataclasses.replace(_occluded(canyon=1.0, air_p=1.0, uav=uav, target_xy=xy), target_layer=Layer(layer))
    seen = [any(e.type == EventType.TARGET_SIGHTED for e in _fly_over(sc, xy, seed).events) for seed in range(40)]
    assert np.mean(seen) == pytest.approx(rate, abs=0.2)


def test_with_the_sighting_test_the_ugv_may_miss_a_target_under_trees():
    xy = (225.0, 525.0)  # in the trees, next to the road
    sc = _occluded(canopy=1.0, ground_veg=0.5, target_xy=xy)  # from the ground: 1 - 0.5 · 1
    from marsupial_search.core import FollowPath

    def found(seed, test):
        sim = Simulator(sc, SimConfig(seed=seed, confirm_test=test))
        cmds = {"ugv0": FollowPath(tuple(range(10)))}
        for _ in range(120):
            if sim.done:
                break
            sim.step(cmds)
            cmds = {}
        return sim.outcome == "found"
    assert all(found(seed, "range") for seed in range(5))
    assert np.mean([found(seed, "sighting") for seed in range(40)]) == pytest.approx(0.5, abs=0.2)


def test_a_blockage_under_trees_is_seen_from_the_air_only_by_chance():
    def reports(canopy, seed, rid="uav0"):
        sc = dataclasses.replace(_occluded(canopy=canopy), road_mismatch=MapMismatch(blocked=(6,)))  # x = 325 m
        sim = Simulator(sc, SimConfig(terminate_on=(), seed=seed))
        sim.step({"uav0": Undock()})
        for _ in range(5):
            sim.step({})
        cmds = {"uav0": GoToPoint((325.0, 500.0))}
        out = False
        for _ in range(60):
            obs = sim.step(cmds)
            cmds = {}
            out |= any(u.robot_id == rid and 6 in u.blocked for u in obs.map_updates)
        return out
    assert not any(reports(1.0, seed) for seed in range(10))  # fully under the canopy: never
    assert all(reports(0.0, seed) for seed in range(5))
    assert np.mean([reports(0.5, seed) for seed in range(40)]) == pytest.approx(0.5, abs=0.2)  # once per pass


# ------------------------------------------------------- planners and controllers
def test_the_planner_weighs_belief_by_what_each_robot_can_see():
    sc = _occluded()
    sim = Simulator(sc)
    P = SortiePlanner(sim.setup, sc.robots["ugv0"], sc.robots["uav0"], SortieConfig())
    n = sc.grid.rows * sc.grid.cols
    assert P.p_air.shape == P.p_ugv.shape == (3, n) and P.own_layers == [A] and P.common_layers == [S, G]
    low = _occluded(uav="ground")
    Q = SortiePlanner(Simulator(low).setup, low.robots["ugv0"], low.robots["uav0"], SortieConfig())
    assert Q.own_layers == [] and Q.common_layers == [S, G]  # nothing that only it sees
    m = np.ones((3, n))
    seen = P.seen(m, P.p_air).reshape(sc.grid.shape)
    urban = sc.terrain == int(Terrain.URBAN)
    assert np.allclose(seen[urban], 0.8 + 0.8 + 0.2) and np.allclose(seen[~urban], 1.6)


def test_a_look_counts_the_road_nodes_in_range_as_seen_but_those_it_could_not_make_out():
    road = make_line_scenario().graphs[RobotType.UGV]
    model = BlockageModel(GraphKnowledge(road), prior=0.1, prior_weight=10.0, max_cutoff_fraction=1.0)
    model.look(road, road.positions[5], 60.0, hidden=(5,))  # nodes 4-6 in range, 5 under trees
    p = model.probabilities(road)
    assert model.seen[[4, 6]].all() and not model.seen[5]
    assert p[4] == p[6] == 0.0 and p[5] == pytest.approx(model.block_rate())  # as unknown as before
    model.look(road, road.positions[5], 60.0)  # another pass makes it out
    assert model.seen[5] and model.probabilities(road)[5] == 0.0


def test_the_robots_get_detection_probabilities_and_hidden_nodes_but_not_the_occlusion():
    sc = dataclasses.replace(_occluded(canopy=1.0), road_mismatch=MapMismatch(blocked=(6,)))
    sim = Simulator(sc, SimConfig(terminate_on=()))
    assert not hasattr(sim.setup, "occlusion")
    for rid, spec in sc.robots.items():  # the simulator's own, per robot
        assert np.array_equal(sim.setup.detection[rid], detection_map(spec, sc.terrain, sc.occlusion))
    # uav0 over the trees (canopy 1: no road node made out from the air) is told which it could not make out.
    sim.step({"uav0": Undock()})
    for _ in range(5):
        sim.step({})
    obs, cmds, hidden = None, {"uav0": GoToPoint((325.0, 500.0))}, ()
    for _ in range(60):
        obs = sim.step(cmds)
        cmds = {}
        for u in obs.map_updates:
            if u.robot_id == "uav0":
                assert not u.blocked  # the blockage at node 6 stays hidden
                hidden = u.hidden
    assert hidden == (5, 6, 7)  # nodes 50 m apart, 60 m sensor
    assert not any(u.robot_id == "ugv0" and u.hidden for u in obs.map_updates)  # the ground sees them


def test_without_occlusion_nothing_is_hidden():
    sc = dataclasses.replace(make_line_scenario(), road_mismatch=MapMismatch(blocked=(6,)))
    sim = Simulator(sc, SimConfig(terminate_on=()))
    sim.step({"uav0": Undock()})
    for _ in range(5):
        sim.step({})
    cmds, ups = {"uav0": GoToPoint((325.0, 500.0))}, []
    for _ in range(60):
        ups += sim.step(cmds).map_updates
        cmds = {}
    assert ups and all(not u.hidden for u in ups) and any(6 in u.blocked for u in ups)


def test_the_ugv_counts_a_uavs_hidden_nodes_as_unknown():
    sim = Simulator(_occluded(canopy=1.0), SimConfig(terminate_on=()))
    from marsupial_search.controllers.ugv_timed_route import UgvCtpRouteConfig
    ugv = TimedRouteUgv(UgvCtpRouteConfig(max_cutoff_fraction=1.0))
    ugv.reset("ugv0", sim.setup)
    pos = (325.0, 500.0)
    ugv._ingest("uav0", {"type": "uav", "pos": pos, "updates": [], "visited": [], "sighting": None,
                         "hidden": [5, 6, 7]}, 1.0)
    assert not ugv.blockage.seen[[5, 6, 7]].any()
    ugv._ingest("uav0", {"type": "uav", "pos": pos, "updates": [], "visited": [], "sighting": None}, 2.0)
    assert ugv.blockage.seen[[6]].all()  # nothing hidden: all seen


@pytest.mark.parametrize("test, passes, again", [("sighting", 0, True), ("sighting", 3, False), ("range", 0, False)])
def test_the_ugv_drives_past_a_sighting_again_after_a_missed_pass(test, passes, again):
    sim = Simulator(make_line_scenario(n_uav=2), SimConfig(confirm_test=test))
    ugv = TimedRouteUgv()
    ugv.reset("ugv0", sim.setup)
    ugv.nodes, ugv.extend_failed = list(range(20)), True
    pos = ugv.road.positions

    def at(i):
        return RobotState("ugv0", RobotType.UGV, tuple(pos[ugv.nodes[i]]), RobotMode.ACTIVE, 1e5, 1e5, False,
                          ugv.nodes[i], ugv.nodes[i], (), (), None)
    ugv.belief.detection_xy = (475.0, 525.0)  # 25 m off node 9
    ugv._maybe_pursue(at(0), 1.0)
    assert ugv.pursuit == "planned"
    ugv.passes["ugv"] = passes
    ugv.cur = ugv.pursuit_end
    ugv._maybe_pursue(at(ugv.cur), 2.0)  # drove past: in range, no find
    assert ugv.pursuit == "done"
    ugv.nodes += [8, 7, 6]
    ugv.cur = len(ugv.nodes) - 1  # 125 m away: out of range
    ugv._maybe_pursue(at(ugv.cur), 3.0)
    assert (ugv.pursuit == "planned") is again


def test_the_scout_team_finds_targets_on_an_occluded_map():
    cfg = dataclasses.replace(small(0), occlusion=True, road_unmapped_fraction=0.1, road_blocked_fraction=0.05)
    cfg = dataclasses.replace(cfg, fleet=dataclasses.replace(cfg.fleet, uav_sensor_ranges={"uav0": 20.0, "uav1": 50.0},
                                                             uav_viewpoints={"uav0": "ground"}))
    res = Simulator(generate_scenario(cfg), SimConfig(confirm_by=("uav0",), confirm_test="sighting")).run(
        make_team({"ugv0": "ugv_timed_route", "uav0": "uav_near_scout", "uav1": "uav_far_scout"}))
    assert res.outcome == "found" and res.summary()["rejected"] == 0
