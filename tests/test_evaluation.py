import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from marsupial_search.controllers import DEFAULT_TEAM, make_team
import dataclasses

from marsupial_search.core import EventType, Layer
from marsupial_search.sim import SCENARIO_BASES, SimConfig, Simulator, generate_scenario, resample_target
from marsupial_search.sim import evaluation

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def small():
    sc = generate_scenario(SCENARIO_BASES["small"](0))
    cfg = SimConfig()
    free = Simulator(sc, evaluation.target_free(cfg)).run(make_team(DEFAULT_TEAM))
    return sc, cfg, free


def test_target_free_run_never_sees_the_target(small):
    sc, cfg, free = small
    assert not any(e.type in (EventType.TARGET_FOUND, EventType.TARGET_SIGHTED) for e in free.events)
    assert free.outcome != "found"


def test_target_weights_are_what_targets_are_drawn_from(small):
    sc, _, _ = small
    W = evaluation.target_weights(sc)
    assert W.sum() == pytest.approx(1.0)
    support = (W.sum(0) > 0)
    assert np.array_equal(support, (sc.meta["target_distribution"] * sc.meta["confirmable_mask"]) > 0)
    rng = np.random.default_rng(0)
    for _ in range(20):
        s2 = resample_target(sc, rng)
        r, c = sc.grid.world_to_cell(*s2.target_xy)
        assert W[int(s2.target_layer), r, c] > 0


def test_prediction_is_exact_when_the_ugv_gets_there_first(small):
    sc, cfg, free = small
    T = evaluation.predicted_find_times(sc, free, cfg)
    exact = 0
    for k in range(12):
        s2 = resample_target(sc, np.random.default_rng(k))
        res = Simulator(s2, cfg).run(make_team(DEFAULT_TEAM))
        pred = evaluation.predicted_time(sc, T, s2.target_xy, s2.target_layer)
        if not any(e.type == EventType.TARGET_SIGHTED for e in res.events):
            assert pred == pytest.approx(res.time_s)
            exact += 1
    assert exact >= 2
    assert evaluation.expected(T, evaluation.target_weights(sc)) <= free.time_s


def test_first_look_is_first_contact(small):
    sc, _, free = small
    for rid in sc.robots:
        cells, t = evaluation.look_events(sc, free, rid)
        first = np.full(sc.grid.rows * sc.grid.cols, np.inf)
        np.minimum.at(first, cells, t)
        assert np.array_equal(first, evaluation.first_contact(sc, free, rid).ravel())


def test_partial_ground_detection_predicts_between_never_and_always(small):
    sc, cfg, free = small

    def with_ground_p(p):
        robots = {r: dataclasses.replace(s, sensor=dataclasses.replace(s.sensor, layer_p_detect=((Layer.GROUND, p),)))
                  if s.robot_type.value == "uav" else s for r, s in sc.robots.items()}
        return evaluation.predicted_find_times(dataclasses.replace(sc, robots=robots), free, cfg)

    T0, Th, T1 = with_ground_p(0.0), with_ground_p(0.5), with_ground_p(1.0)
    assert np.array_equal(T0, evaluation.predicted_find_times(sc, free, cfg))
    g = int(Layer.GROUND)
    assert np.array_equal(np.delete(Th, g, 0), np.delete(T0, g, 0))
    # Per cell, the UAV looks before the UGV gets there: with one, sighted on it or never, half each.
    found = evaluation.first_contact(sc, free, "ugv0").ravel()
    n = np.zeros(found.size, dtype=int)
    for rid in ("uav0", "uav1"):
        cells, t = evaluation.look_events(sc, free, rid)
        np.add.at(n, cells[t < found[cells]], 1)
    n = n.reshape(T0[g].shape)
    assert np.array_equal(Th[g][n == 0], T0[g][n == 0])
    assert (n == 1).any() and (n > 1).any()
    assert np.allclose(Th[g][n == 1], 0.5 * T0[g][n == 1] + 0.5 * T1[g][n == 1])


def test_compare_cv_runs(tmp_path):
    team = ",".join(f"{r}={p}" for r, p in DEFAULT_TEAM.items())
    out = subprocess.run([sys.executable, str(ROOT / "scripts/compare.py"), "--scenario", "small", "--maps", "0", "1",
                          "--targets", "1", "--teams", team, "--workers", "2", "--out", str(tmp_path)],
                         check=True, capture_output=True, text=True, cwd=ROOT).stdout
    assert "expected time to find" in out
    assert (tmp_path / "maps.csv").exists() and (tmp_path / "runs.csv").exists()


def test_a_confirming_uavs_sighting_is_predicted_to_end_the_mission_there_and_then(small):
    sc, cfg, free = small
    T0 = evaluation.predicted_find_times(sc, free, cfg)
    T = evaluation.predicted_find_times(sc, free, dataclasses.replace(cfg, confirm_by=("uav0",)))
    assert (T <= T0).all() and (T < T0).any()
    # The shared layer, seen for certain: where uav0 looks first, it is found on that look.
    first = {rid: evaluation.first_contact(sc, free, rid) for rid in sc.robots}
    s = int(Layer.SHARED)
    mine = (first["uav0"] < first["ugv0"]) & (first["uav0"] < first["uav1"])
    assert mine.any() and np.array_equal(T[s][mine], first["uav0"][mine])
