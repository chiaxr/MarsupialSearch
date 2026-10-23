"""YAML experiment files: base scenario < YAML values < explicit CLI flags.

Settings shared by both scripts are at the top level; each script's own settings are in
its section::

    teams:                        # what runs, in both scripts
      - joint_planner             # a centralised controller's name
      - {ugv0: ugv_timed_route, uav0: uav_greedy_tour, uav1: uav_greedy_tour}   # or a policy per robot
    controller_params:            # per policy / controller: overrides of its config dataclass
      uav_greedy_tour: {belief_radius_m: 2000}
    out: outputs/my_experiment
    scenario:
      base: small                 # large (5 km x 5 km, 3 towns) | small (1.5 km x 1.5 km, 1 town)
      ...                         # any ScenarioConfig field, nested for fleet/docking
    sim: {...}                    # any SimConfig field, nested for radio
    mission:                      # run_mission.py: each team once, on the same map and target
      seed: 0                     # map seed
      target_seed: 7              # resample the hidden target (optional)
      animate: true               # write a GIF per team
      gif_speedup: 20             # mission seconds per second of GIF (default: about 150 frames per mission)
    compare:                      # compare.py: every team on the same maps and targets
      maps: [0, 1, 2]             # map seeds
      targets: 2                  # missions per team and map
      estimator: cv               # cv (target-free run + a few missions per map) | plain
      workers: 8                  # parallel processes

``scenario`` is applied on top of its base (``scenario_bases.py``) and ``sim`` on top of
``SimConfig()``, so a partial nested section (e.g. ``scenario.fleet.uav_speed``) keeps its
other values.
Values are converted from the dataclass annotations: lists become tuples, enum names
become enums (``terminate_on: [ugv]``, ``prior_labels: [road]``), and dicts are merged.
Unknown keys raise an error that lists the valid ones; keys from the older flat layout
raise an error that says where they went.
"""

from __future__ import annotations

import dataclasses
import enum
import types
import typing
from pathlib import Path
from typing import Any, Union

import yaml

from marsupial_search.controllers import DEFAULT_TEAM

from .generator import ScenarioConfig
from .scenario_bases import SCENARIO_BASES
from .simulator import SimConfig

DEFAULT_BASE = "large"
DEFAULT_TEAMS = ["joint_planner", DEFAULT_TEAM]
ESTIMATORS = ("cv", "plain")
# Defaults of the script sections. compare.targets defaults to 2 for cv and 10 for plain.
SECTIONS: dict[str, dict[str, Any]] = {
    "mission": {"seed": 0, "target_seed": None, "animate": True, "gif_speedup": None},
    "compare": {"maps": [0, 1, 2, 3, 4], "targets": None, "estimator": "cv", "workers": 4},
}
TOP_LEVEL_KEYS = {"teams", "controller_params", "out", "scenario", "sim", *SECTIONS}
_MOVED = {
    "preset": "replaced by `scenario.base` (and `default` is now `large`)",
    "env_size": "replaced by `scenario.base`",
    "team": "replaced by `teams`, a list that both scripts run",
    **{k: f"moved under `{name}:`" for name, keys in SECTIONS.items() for k in keys},
}


class ConfigError(ValueError):
    pass


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        doc = yaml.safe_load(f) or {}
    if not isinstance(doc, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    moved = [f"`{k}` {_MOVED[k]}" for k in doc if k in _MOVED]
    if moved:
        raise ConfigError(f"{path}: " + "; ".join(moved))
    unknown = set(doc) - TOP_LEVEL_KEYS
    if unknown:
        raise ConfigError(f"{path}: unknown keys {sorted(unknown)}; valid: {sorted(TOP_LEVEL_KEYS)}")
    for name, keys in SECTIONS.items():
        sec = doc.get(name) or {}
        if not isinstance(sec, dict):
            raise ConfigError(f"{path}: `{name}` must be a mapping")
        for k in set(sec) - set(keys):
            other = [o for o, ks in SECTIONS.items() if k in ks]
            where = f"; it belongs under `{other[0]}:`" if other else f"; valid: {sorted(keys)}"
            raise ConfigError(f"{path}: unknown key `{name}.{k}`{where}")
    if "teams" in doc and not isinstance(doc["teams"], list):
        raise ConfigError(f"{path}: `teams` must be a list, e.g. [joint_planner]")
    for key in ("seed", "target_seed"):
        if key in (doc.get("scenario") or {}):
            raise ConfigError(f"{path}: set `{key}` under `mission:`, not under `scenario`")
    estimator = section(doc, "compare")["estimator"]
    if estimator not in ESTIMATORS:
        raise ConfigError(f"{path}: compare.estimator must be one of {ESTIMATORS}, got {estimator!r}")
    return doc


def section(doc: dict[str, Any], name: str) -> dict[str, Any]:
    """A script section of ``doc`` with its defaults filled in."""
    return {**SECTIONS[name], **(doc.get(name) or {})}


def default_targets(estimator: str) -> int:
    return 2 if estimator == "cv" else 10


def scenario_base(doc: dict[str, Any], base: str | None = None) -> str:
    """The base scenario's name: ``base`` (a CLI flag) if given, else ``scenario.base``."""
    name = base or (doc.get("scenario") or {}).get("base", DEFAULT_BASE)
    if name not in SCENARIO_BASES:
        raise ConfigError(f"unknown scenario.base {name!r}; choose from {sorted(SCENARIO_BASES)}")
    return name


def build_configs(doc: dict[str, Any], base: str | None = None) -> tuple[ScenarioConfig, SimConfig]:
    """The base scenario with the YAML ``scenario`` section applied, and ``SimConfig()`` with ``sim``."""
    overrides = {k: v for k, v in (doc.get("scenario") or {}).items() if k != "base"}
    scenario = apply_overrides(SCENARIO_BASES[scenario_base(doc, base)](0), overrides, "scenario")
    sim = apply_overrides(SimConfig(), doc.get("sim") or {}, "sim")
    return scenario, sim


def apply_overrides(obj: Any, overrides: dict[str, Any], where: str = "") -> Any:
    """Copy of dataclass ``obj`` with ``overrides`` applied (recursively, type-converted)."""
    if not isinstance(overrides, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(overrides).__name__}")
    cls = type(obj)
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(overrides) - names
    if unknown:
        raise ConfigError(f"{where}: unknown keys {sorted(unknown)}; valid: {sorted(names)}")
    changes = {k: _convert(v, hints[k], getattr(obj, k), f"{where}.{k}") for k, v in overrides.items()}
    return dataclasses.replace(obj, **changes)


def _convert(value: Any, hint: Any, current: Any, where: str) -> Any:
    origin, args = typing.get_origin(hint), typing.get_args(hint)
    if hint is Any:
        return value
    if origin in (Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        errors = []
        for a in args:
            if a is type(None):
                continue
            try:
                return _convert(value, a, current, where)
            except ConfigError as e:
                errors.append(str(e))
        raise ConfigError("; ".join(errors))
    if dataclasses.is_dataclass(hint):
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a mapping")
        if current is not None:
            return apply_overrides(current, value, where)
        hints = typing.get_type_hints(hint)
        return hint(**{k: _convert(v, hints[k], None, f"{where}.{k}") for k, v in value.items()})
    if isinstance(hint, type) and issubclass(hint, enum.Enum):
        if isinstance(value, hint):
            return value
        for m in hint:
            if value == m.value or (isinstance(value, str) and value.upper() == m.name):
                return m
        raise ConfigError(f"{where}: {value!r} is not one of {[m.name.lower() for m in hint]}")
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{where}: expected a list")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_convert(v, args[0], None, f"{where}[{i}]") for i, v in enumerate(value))
        if len(value) != len(args):
            raise ConfigError(f"{where}: expected {len(args)} values, got {len(value)}")
        return tuple(_convert(v, a, None, f"{where}[{i}]") for i, (v, a) in enumerate(zip(value, args)))
    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected a mapping")
        kt, vt = args
        new = {_convert(k, kt, None, f"{where} key"): _convert(v, vt, None, f"{where}[{k}]") for k, v in value.items()}
        return {**(current or {}), **new}
    if origin is list:
        return [_convert(v, args[0], None, f"{where}[{i}]") for i, v in enumerate(value)]
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{where}: expected a number, got {value!r}")
        return float(value)
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where}: expected an integer, got {value!r}")
        return value
    if hint is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{where}: expected true/false, got {value!r}")
        return value
    if hint is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where}: expected a string, got {value!r}")
        return value
    raise ConfigError(f"{where}: cannot be set from YAML (type {hint})")


_SKIP = object()


def to_plain(obj: Any) -> Any:
    """YAML-friendly view of configs (enums by name, tuples as lists).

    Fields that cannot be expressed in YAML (e.g. the radio codec object) are omitted,
    so the output can be loaded back with :func:`load_yaml`.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        out = {f.name: to_plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
        return {k: v for k, v in out.items() if v is not _SKIP}
    if isinstance(obj, enum.Enum):
        return obj.name.lower()
    if isinstance(obj, dict):
        return {to_plain(k): to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    return _SKIP


def resolved_scenario(scenario: ScenarioConfig, base: str) -> dict[str, Any]:
    """Every ``scenario`` key of a run, its base first (the seeds are set elsewhere)."""
    scen = to_plain(scenario)
    scen.pop("seed", None)
    scen.pop("target_seed", None)
    return {"base": base, **scen}


def dump_resolved(path: str | Path, run: dict[str, Any], scenario: ScenarioConfig, sim: SimConfig,
                  base: str) -> None:
    """Write the fully resolved settings of a run, for reproducibility."""
    doc = {**run, "scenario": resolved_scenario(scenario, base), "sim": to_plain(sim)}
    with open(path, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False)


def settable(doc: dict[str, Any], base: str | None = None) -> dict[str, Any]:
    """Every key an experiment file can set, with its value after applying ``doc``."""
    from marsupial_search.controllers import CENTRALISED, POLICIES, controller_params

    base = scenario_base(doc, base)
    scenario, sim = build_configs(doc, base)
    compare = section(doc, "compare")
    compare["targets"] = compare["targets"] or default_targets(compare["estimator"])
    params = doc.get("controller_params") or {}
    return {"teams": doc.get("teams", DEFAULT_TEAMS),
            "controller_params": {name: {**controller_params(name), **(params.get(name) or {})}
                                  for name in [*sorted(POLICIES), *sorted(CENTRALISED)]},
            "out": doc.get("out"),
            "scenario": resolved_scenario(scenario, base), "sim": to_plain(sim),
            "mission": section(doc, "mission"), "compare": compare}


def main() -> None:
    """Print every settable key with its value for a base scenario (optionally after a YAML file)."""
    import argparse
    import sys

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--scenario", choices=sorted(SCENARIO_BASES), help="base scenario")
    ap.add_argument("--config", help="show the result of applying this YAML file")
    args = ap.parse_args()
    doc = load_yaml(args.config) if args.config else {}
    yaml.safe_dump(settable(doc, args.scenario), sys.stdout, sort_keys=False)


if __name__ == "__main__":
    main()
