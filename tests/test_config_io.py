import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from marsupial_search.controllers import DEFAULT_TEAM, make_team
from marsupial_search.core import RobotType
from marsupial_search.sim import SCENARIO_BASES, SimConfig
from marsupial_search.sim.config_io import (
    SECTIONS,
    ConfigError,
    build_configs,
    dump_resolved,
    load_yaml,
    section,
    settable,
    to_plain,
)

ROOT = Path(__file__).resolve().parents[1]


def _write(tmp_path, doc):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(doc) if isinstance(doc, dict) else doc)
    return p


def test_partial_nested_overrides_keep_env_values(tmp_path):
    doc = load_yaml(_write(tmp_path, {
        "scenario": {"base": "small", "target_object": "hiker", "fleet": {"ugv_speed": 4.0, "ugv_energy": {"capacity": 9000}}},
        "sim": {"terminate_on": ["ugv", "uav"], "radio": {"range_m": {"uav": 500}}},
    }))
    scen, sim = build_configs(doc)
    base = SCENARIO_BASES["small"](0)
    assert scen.target_object == "hiker" and scen.size_m == base.size_m  # untouched base value
    assert scen.fleet.ugv_speed == 4.0 and scen.fleet.uav_speed == base.fleet.uav_speed
    assert scen.fleet.ugv_energy.capacity == 9000.0 and isinstance(scen.fleet.ugv_energy.capacity, float)
    assert scen.fleet.ugv_energy.idle_power == base.fleet.ugv_energy.idle_power
    assert sim.terminate_on == (RobotType.UGV, RobotType.UAV)
    assert sim.radio.range_m == {RobotType.UGV: 2000.0, RobotType.UAV: 500.0}  # dicts merge
    assert sim.stop_when_controller_done is SimConfig().stop_when_controller_done


def test_lists_become_tuples_and_enums_convert(tmp_path):
    doc = load_yaml(_write(tmp_path, {"scenario": {"size_m": [2000, 1500], "prior_labels": ["road", "urban"]}}))
    scen, _ = build_configs(doc)
    assert scen.size_m == (2000.0, 1500.0) and scen.prior_labels == ("road", "urban")


@pytest.mark.parametrize("doc, msg", [
    ({"scenarios": {}}, "unknown keys ['scenarios']"),
    ({"scenario": {"fleet": {"n_ugv": 2}}}, "scenario.fleet: unknown keys ['n_ugv']"),
    ({"scenario": {"cell_m": "big"}}, "scenario.cell_m: expected a number"),
    ({"scenario": {"n_urban": 1.5}}, "scenario.n_urban: expected an integer"),
    ({"sim": {"terminate_on": ["boat"]}}, "is not one of"),
    ({"scenario": {"seed": 3}}, "under `mission:`"),
    ({"scenario": {"base": "huge"}}, "unknown scenario.base"),
    ({"preset": "small"}, "`preset` replaced by `scenario.base`"),
    ({"env_size": "small"}, "`env_size` replaced by `scenario.base`"),
    ({"team": "joint_planner"}, "`team` replaced by `teams`"),
    ({"maps": [0]}, "`maps` moved under `compare:`"),
    ({"compare": {"seed": 0}}, "`compare.seed`; it belongs under `mission:`"),
    ({"mission": {"radius": 1}}, "unknown key `mission.radius`"),
    ({"teams": "joint_planner"}, "`teams` must be a list"),
    ({"compare": {"estimator": "bootstrap"}}, "compare.estimator must be one of"),
    ({"sim": {"radio": {"codec": "json"}}}, "cannot be set from YAML"),
])
def test_bad_files_fail_with_a_useful_message(tmp_path, doc, msg):
    with pytest.raises(ConfigError, match=msg.replace("[", r"\[").replace("]", r"\]")):
        build_configs(load_yaml(_write(tmp_path, doc)))


def test_resolved_dump_round_trips(tmp_path):
    doc = load_yaml(_write(tmp_path, {"scenario": {"base": "small", "target_object": "hiker"},
                                      "sim": {"radio": {"drop_prob": 0.2}}}))
    scen, sim = build_configs(doc)
    dump_resolved(tmp_path / "r.yaml", {"mission": {"seed": 4}}, scen, sim, "small")
    resolved = load_yaml(tmp_path / "r.yaml")
    assert resolved["scenario"]["base"] == "small"
    scen2, sim2 = build_configs(resolved)
    assert to_plain(scen2) == to_plain(scen) and to_plain(sim2) == to_plain(sim)


def test_controller_params():
    assert make_team("joint_planner", {"joint_planner": {"horizon_s": 900.0}}).cfg.horizon_s == 900.0
    with pytest.raises(ValueError, match="unknown parameters"):
        make_team("joint_planner", {"joint_planner": {"radius": 1}})
    with pytest.raises(ValueError, match="unknown parameters"):
        make_team("joint_planner", {"joint_planner": {"plan_lead_s": 5}})  # a uav_greedy_tour parameter
    with pytest.raises(ValueError, match="unknown controller"):
        make_team(DEFAULT_TEAM, {"uav_nope": {}})


def test_sections_fill_in_their_defaults(tmp_path):
    doc = load_yaml(_write(tmp_path, {"compare": {"maps": [3]}}))
    assert section(doc, "compare") == {**SECTIONS["compare"], "maps": [3]}
    assert section(doc, "mission") == SECTIONS["mission"]


def test_cli_flags_override_the_file(tmp_path):
    cfg = _write(tmp_path, {"teams": ["joint_planner"], "out": str(tmp_path / "o"),
                            "mission": {"seed": 1, "animate": False},
                            "scenario": {"base": "small", "target_object": "hiker"}})
    team = ",".join(f"{r}={p}" for r, p in DEFAULT_TEAM.items())
    subprocess.run([sys.executable, str(ROOT / "scripts/run_mission.py"), "--config", str(cfg), "--seed", "2",
                    "--teams", team], check=True, capture_output=True, cwd=ROOT)
    resolved = yaml.safe_load((tmp_path / "o" / "small_s2_config.yaml").read_text())
    assert resolved["mission"]["seed"] == 2 and resolved["teams"] == [DEFAULT_TEAM]
    assert resolved["scenario"]["target_object"] == "hiker" and resolved["mission"]["animate"] is False
    assert (tmp_path / "o" / "small_s2_ugv_timed_route+uav_greedy_tour+uav_greedy_tour.png").exists()
    assert build_configs(load_yaml(tmp_path / "o" / "small_s2_config.yaml"))  # the dump loads back


def _leaves(d, prefix=""):
    for k, v in d.items():
        path = f"{prefix}{k}"
        if isinstance(v, dict) and v and k not in ("range_m",):
            yield from _leaves(v, path + ".")
        else:
            yield path, v


def test_example_config_sets_every_key():
    """configs/example.yaml is the reference for what can be configured: it lists every key."""
    example = load_yaml(ROOT / "configs/example.yaml")
    assert set(dict(_leaves(example))) == set(dict(_leaves(settable({}))))
