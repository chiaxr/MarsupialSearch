"""Controllers. Portable: depend only on :mod:`marsupial_search.core` (+ numpy/scipy).

A *team* runs the mission. It is one of:

* **a policy per robot** – ``{"ugv0": "ugv_timed_route", "uav0": "uav_greedy_tour",
  "uav1": "uav_greedy_tour"}``. A policy is a :class:`~marsupial_search.core.RobotController`
  for one robot type, named with its prefix (``ugv_`` / ``uav_``). It sees only its own
  robot's view and radio inbox; :func:`make_team` runs the policies in a
  :class:`~marsupial_search.core.DecentralizedTeam`. Two UAVs with the same policy is the
  shared-policy case.
* **a centralised controller** – a :class:`~marsupial_search.core.MissionController` that sees
  every robot's state and sensor data instantly and ignores the radio. These serve as
  *oracles*: a reference with perfect, free communication.

A *team spec* is a centralised controller's name, a ``{robot_id: policy}`` mapping, or the
mapping written as ``"ugv0=ugv_timed_route,uav0=uav_greedy_tour,uav1=uav_greedy_tour"``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Union

from marsupial_search.core import DecentralizedTeam, MissionController, MissionSetup, RobotType

from .dock_booking import BookingUavConfig
from .joint_planner import JointPlannerConfig, JointPlannerController
from .uav_greedy_tour import GreedyTourUav
from .uav_far_scout import FarScoutUav, FarScoutUavConfig
from .uav_near_scout import NearScoutUav, NearScoutUavConfig
from .ugv_timed_route import CtpRouteUgv, TimedRouteUgv, UgvCtpRouteConfig, UgvTimedRouteConfig


@dataclass(frozen=True)
class PolicyEntry:
    """A per-robot policy: a RobotController class for robots of ``robot_type``."""
    cls: type
    robot_type: RobotType
    config_cls: Optional[type] = None  # config dataclass passed to the constructor (None: no-arg constructor)
    description: str = ""


@dataclass(frozen=True)
class CentralisedEntry:
    """A team-level MissionController class."""
    cls: type
    config_cls: Optional[type] = None
    description: str = ""


POLICIES: dict[str, PolicyEntry] = {
    "ugv_timed_route": PolicyEntry(TimedRouteUgv, RobotType.UGV, UgvTimedRouteConfig,
                                   "the mobile dock: drives a committed, timed route; UAVs book landings on it"),
    "ugv_ctp_route": PolicyEntry(CtpRouteUgv, RobotType.UGV, UgvCtpRouteConfig,
                                 "ugv_timed_route planning its way on the road as a Canadian Traveller's Problem "
                                 "(risk of blockages not yet seen)"),
    "uav_greedy_tour": PolicyEntry(GreedyTourUav, RobotType.UAV, BookingUavConfig,
                                   "books the next landing slot on the UGV's route, flies a greedy chain tour to it"),
    "uav_near_scout": PolicyEntry(NearScoutUav, RobotType.UAV, NearScoutUavConfig,
                                  "flies low, on standby on the dock: the UGV calls it to make room for a landing "
                                  "or to scout a blocked node"),
    "uav_far_scout": PolicyEntry(FarScoutUav, RobotType.UAV, FarScoutUavConfig,
                                 "flies high, searching the wider area: books its own sorties (for now as "
                                 "uav_greedy_tour)"),
}

CENTRALISED: dict[str, CentralisedEntry] = {
    "joint_planner": CentralisedEntry(JointPlannerController, JointPlannerConfig,
                                      "oracle: plans the UGV route and both UAVs' sorties jointly, ideal information"),
}

DEFAULT_TEAM: dict[str, str] = {"ugv0": "ugv_timed_route", "uav0": "uav_greedy_tour", "uav1": "uav_greedy_tour"}

TeamSpec = Union[str, Mapping[str, str]]


def register(name: str, entry: Union[PolicyEntry, CentralisedEntry]) -> None:
    if name in POLICIES or name in CENTRALISED:
        raise ValueError(f"controller {name!r} already registered")
    if isinstance(entry, PolicyEntry):
        if not name.startswith(f"{entry.robot_type.value}_"):
            raise ValueError(f"a {entry.robot_type.value.upper()} policy's name must start with "
                             f"'{entry.robot_type.value}_', got {name!r}")
        POLICIES[name] = entry
    else:
        CENTRALISED[name] = entry


def parse_team(spec: TeamSpec) -> Union[str, dict[str, str]]:
    """A centralised controller's name, or the ``{robot_id: policy}`` mapping."""
    if isinstance(spec, str) and "=" in spec:
        pairs = [p.split("=") for p in spec.split(",") if p.strip()]
        if any(len(p) != 2 for p in pairs):
            raise ValueError(f"team {spec!r}: expected robot=policy pairs separated by commas")
        spec = {rid.strip(): policy.strip() for rid, policy in pairs}
    if isinstance(spec, str):
        if spec not in CENTRALISED:
            hint = " (per-robot policies are given as robot=policy pairs)" if spec in POLICIES else ""
            raise ValueError(f"unknown centralised controller {spec!r}{hint}; choose from {sorted(CENTRALISED)}")
        return spec
    if not isinstance(spec, Mapping) or not all(isinstance(k, str) and isinstance(v, str) for k, v in spec.items()):
        raise ValueError(f"team {spec!r}: expected a controller name or a {{robot_id: policy}} mapping")
    for policy in spec.values():
        _policy(policy)
    return dict(spec)


def team_label(spec: TeamSpec) -> str:
    """Short name for a team: the centralised controller's name, or each robot's policy
    (UGV first, then by robot id) joined with ``+``."""
    team = parse_team(spec)
    if isinstance(team, str):
        return team
    order = sorted(team, key=lambda rid: (POLICIES[team[rid]].robot_type != RobotType.UGV, rid))
    return "+".join(team[rid] for rid in order)


def team_kind(spec: TeamSpec) -> str:
    return "centralised" if isinstance(parse_team(spec), str) else "decentralised"


def controller_params(name: str) -> dict[str, Any]:
    """Tunable parameters (and their defaults) of a registered policy or centralised controller."""
    e = _entry(name)
    if e.config_cls is None:
        return {}
    defaults = e.config_cls()
    return {f.name: getattr(defaults, f.name) for f in dataclasses.fields(e.config_cls)}


def make_team(spec: TeamSpec, params: Optional[Mapping[str, Mapping[str, Any]]] = None) -> MissionController:
    """Build a team. ``params`` maps a policy or controller name to overrides of its config
    dataclass (the YAML ``controller_params``); they apply wherever that name is used."""
    params = dict(params or {})
    for name, p in params.items():
        bad = set(p or {}) - set(controller_params(name))
        if bad:
            raise ValueError(f"{name}: unknown parameters {sorted(bad)}; tunable: {sorted(controller_params(name))}")

    def build(name: str):
        e = _entry(name)
        return e.cls(e.config_cls(**(params.get(name) or {}))) if e.config_cls else e.cls()

    team = parse_team(spec)
    if isinstance(team, str):
        ctl = build(team)
        ctl.name = team
        return ctl

    def factory(robot_id, robot_spec):
        return build(team[robot_id])  # a fresh instance (and config) per robot

    return _PolicyTeam(team, factory, name=team_label(team))


class _PolicyTeam(DecentralizedTeam):
    """A DecentralizedTeam that checks the assignment against the robots of the mission."""

    def __init__(self, assignment: dict[str, str], factory, name: str):
        super().__init__(factory, name=name)
        self.assignment = assignment

    def reset(self, setup: MissionSetup) -> None:
        missing, extra = set(setup.robots) - set(self.assignment), set(self.assignment) - set(setup.robots)
        problems = ([f"no policy for {sorted(missing)}"] if missing else []) + \
            ([f"no such robot {sorted(extra)}"] if extra else [])
        if problems:
            raise ValueError(f"team {self.name}: " + "; ".join(problems))
        for rid, policy in self.assignment.items():
            want, got = POLICIES[policy].robot_type, setup.robots[rid].robot_type
            if want != got:
                raise ValueError(f"{policy} is a {want.value.upper()} policy, but {rid} is a {got.value.upper()}")
        super().reset(setup)


def _policy(name: str) -> PolicyEntry:
    try:
        return POLICIES[name]
    except KeyError:
        raise ValueError(f"unknown policy {name!r}; choose from {sorted(POLICIES)}") from None


def _entry(name: str) -> Union[PolicyEntry, CentralisedEntry]:
    if name in POLICIES:
        return POLICIES[name]
    if name in CENTRALISED:
        return CENTRALISED[name]
    raise ValueError(f"unknown controller {name!r}; choose from {sorted(POLICIES) + sorted(CENTRALISED)}")
