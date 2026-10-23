import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from marsupial_search.controllers import DEFAULT_TEAM, make_team
from marsupial_search.core import EventType
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


def test_compare_cv_runs(tmp_path):
    team = ",".join(f"{r}={p}" for r, p in DEFAULT_TEAM.items())
    out = subprocess.run([sys.executable, str(ROOT / "scripts/compare.py"), "--scenario", "small", "--maps", "0", "1",
                          "--targets", "1", "--teams", team, "--workers", "2", "--out", str(tmp_path)],
                         check=True, capture_output=True, text=True, cwd=ROOT).stdout
    assert "expected time to find" in out
    assert (tmp_path / "maps.csv").exists() and (tmp_path / "runs.csv").exists()
