import dataclasses

import numpy as np
import pytest

from conftest import make_line_scenario
from marsupial_search.controllers import (
    CENTRALISED,
    DEFAULT_TEAM,
    POLICIES,
    PolicyEntry,
    make_team,
    parse_team,
    register,
    team_kind,
    team_label,
)
from marsupial_search.core import (
    DecentralizedTeam,
    EnergySpec,
    FollowPath,
    GoToPoint,
    RobotController,
    RobotMode,
    RobotSpec,
    RobotType,
    Transmit,
    Undock,
)
from marsupial_search.sim import SCENARIO_BASES, RadioConfig, SimConfig, Simulator, generate_scenario


class GossipCoverage(RobotController):
    """Test algorithm: the UAVs cover the UAV grid nearest node first, sharing visited nodes
    and goals by radio. A docked UAV takes off first; the UGV stays put."""

    name = "gossip"

    def reset(self, robot_id, setup):
        self.rid, self.spec = robot_id, setup.robots[robot_id]
        self.graph = setup.graphs[RobotType.UAV]
        self.visited = np.zeros(self.graph.n_nodes, dtype=bool)
        self.others_goal: dict[str, int] = {}
        self.goal, self.done = None, False

    def step(self, obs):
        if self.spec.robot_type == RobotType.UGV:
            self.done = True
            return None
        s = obs.state
        if s.mode == RobotMode.DOCKED:
            return Undock()
        if s.mode != RobotMode.ACTIVE:
            return None
        for m in obs.messages("visited"):
            self.visited[m.payload["nodes"]] = True
            self.others_goal[m.sender] = m.payload["goal"]
        new = list(s.nodes_reached) + ([s.node] if s.node is not None else [])
        self.visited[new] = True
        cmds = []
        if s.idle:
            cand = ~self.visited
            for g in self.others_goal.values():
                if g is not None:
                    cand[g] = False
            src = s.node if s.node is not None else self.graph.nearest_node(s.position)
            path, _ = self.graph.nearest_matching(src, cand)
            self.goal = path[-1] if path else None
            self.done = not path and not (~self.visited).any()
            if path:
                cmds.append(FollowPath(tuple(path)))
        cmds.append(Transmit("visited", {"nodes": new, "goal": self.goal}))
        return cmds

    def is_done(self):
        return self.done


class GossipUgv(GossipCoverage):
    pass


class GossipUav(GossipCoverage):
    pass


@pytest.fixture
def gossip_registered():
    register("ugv_gossip", PolicyEntry(GossipUgv, RobotType.UGV, description="test"))
    register("uav_gossip", PolicyEntry(GossipUav, RobotType.UAV, description="test"))
    yield {"ugv0": "ugv_gossip", "uav0": "uav_gossip", "uav1": "uav_gossip"}
    POLICIES.pop("ugv_gossip")
    POLICIES.pop("uav_gossip")


def test_local_view_contains_only_what_the_robot_knows():
    sc = make_line_scenario(n_uav=2)
    sim = Simulator(sc, SimConfig(terminate_on=()))
    sim.step({"uav0": Undock()})
    obs = sim.step({"uav1": Transmit("hi", 1, to=("ugv0",))})
    for _ in range(5):
        obs = sim.step({"uav0": FollowPath((0,)), "uav1": GoToPoint((225.0, 500.0))})
    obs = sim.step({"uav1": Transmit("hi", 2, to=("ugv0",))})
    v0, vu = obs.local_view("ugv0"), obs.local_view("uav0")
    assert v0.state.robot_id == "ugv0" and [m.payload for m in v0.inbox] == [2] and vu.inbox == []
    assert {u.robot_id for u in v0.belief_updates} == {"ugv0"}
    assert {u.robot_id for u in vu.belief_updates} == {"uav0"}
    assert all("uav0" in (e.robot_id, e.other_id) for e in vu.events)


def test_team_gives_each_robot_its_own_controller_and_setup():
    seen = []

    class Probe(RobotController):
        def reset(self, robot_id, setup):
            self.setup = setup
            setup.prior[:] = 0  # must not leak to other robots
            seen.append((robot_id, id(self), id(setup)))

        def step(self, obs):
            return None

    sc = make_line_scenario(n_uav=2)
    team = DecentralizedTeam(lambda rid, spec: Probe())
    Simulator(sc, SimConfig(terminate_on=(), max_time_s=3)).run(team)
    assert len({s[1] for s in seen}) == 3 and len({s[2] for s in seen}) == 3
    assert sc.prior.sum() == pytest.approx(1.0)


def test_depleted_robots_are_no_longer_stepped():
    calls = {}

    class Count(RobotController):
        def reset(self, robot_id, setup):
            self.rid = robot_id

        def step(self, obs):
            calls[self.rid] = calls.get(self.rid, 0) + 1
            return GoToPoint((125.0, 600.0)) if self.rid == "uav1" and obs.time == 0 else None

    sc = make_line_scenario(n_uav=2)
    spec = sc.robots["uav1"]
    sc.robots["uav1"] = RobotSpec(spec.robot_id, spec.robot_type, spec.speed, EnergySpec(1.0, 1.0, 0.05), spec.sensor)
    Simulator(sc, SimConfig(terminate_on=(), max_time_s=10)).run(DecentralizedTeam(lambda rid, spec: Count()))
    assert calls["ugv0"] == 10 and calls["uav1"] == 1


def test_team_specs():
    assert team_kind("joint_planner") == "centralised" and "joint_planner" in CENTRALISED
    assert team_kind(DEFAULT_TEAM) == "decentralised"
    assert parse_team("ugv0=ugv_timed_route, uav0=uav_greedy_tour,uav1=uav_greedy_tour") == DEFAULT_TEAM
    assert team_label({"uav1": "uav_greedy_tour", "uav0": "uav_greedy_tour", "ugv0": "ugv_timed_route"}) == \
        "ugv_timed_route+uav_greedy_tour+uav_greedy_tour"
    with pytest.raises(ValueError, match="robot=policy pairs"):
        parse_team("uav_greedy_tour")  # a policy is not a team
    with pytest.raises(ValueError, match="unknown policy"):
        parse_team({**DEFAULT_TEAM, "uav1": "uav_nope"})


def test_registry_builds_a_policy_per_robot(gossip_registered):
    team = make_team(gossip_registered)
    assert isinstance(team, DecentralizedTeam) and team.name == "ugv_gossip+uav_gossip+uav_gossip"
    team.reset(Simulator(make_line_scenario(n_uav=2)).setup)
    assert type(team.robots["ugv0"]) is GossipUgv
    assert type(team.robots["uav0"]) is GossipUav and team.robots["uav0"] is not team.robots["uav1"]


def test_policies_must_match_their_robot(gossip_registered):
    setup = Simulator(make_line_scenario(n_uav=2)).setup
    with pytest.raises(ValueError, match="is a UAV policy, but ugv0 is a UGV"):
        make_team({**gossip_registered, "ugv0": "uav_gossip"}).reset(setup)
    with pytest.raises(ValueError, match=r"no policy for \['uav1'\]"):
        make_team({"ugv0": "ugv_gossip", "uav0": "uav_gossip"}).reset(setup)
    with pytest.raises(ValueError, match="must start with 'uav_'"):
        register("gossip", PolicyEntry(GossipUav, RobotType.UAV))


def test_sharing_over_radio_speeds_up_decentralised_coverage(gossip_registered):
    cfg = SCENARIO_BASES["small"](0)
    cfg.fleet = dataclasses.replace(cfg.fleet, uav_energy=EnergySpec(capacity=1e6, idle_power=1.0, energy_per_m=0.04))
    sc = generate_scenario(cfg)
    times = {}
    for label, radio in {"ideal": RadioConfig(simulate_channel=False), "off": RadioConfig(enabled=False)}.items():
        sim_cfg = SimConfig(terminate_on=(), max_time_s=4 * 3600, radio=radio, stop_when_ugv_depleted=False)
        res = Simulator(sc, sim_cfg).run(make_team(gossip_registered))
        assert res.outcome == "controller_done"
        times[label] = res.time_s
        if label == "ideal":
            assert res.summary()["radio_received"] > 0
    # Without radio each UAV covers the whole UAV grid on its own. (With it the gain is modest:
    # both start over the UGV, and nearest-first sweeps from one point interleave.)
    assert times["ideal"] < 0.85 * times["off"]


def test_controller_step_calls_are_timed(gossip_registered):
    sc = make_line_scenario(n_uav=2)
    res = Simulator(sc, SimConfig(terminate_on=(), max_time_s=20)).run(make_team(gossip_registered))
    s = res.summary()
    assert s["controller_steps"] == 20 and 0 < s["controller_max_step_s"] <= s["controller_s"] <= s["wall_time_s"]
    robots = [res.robot_stats[r] for r in ("ugv0", "uav0", "uav1")]
    assert all(r["controller_steps"] == 20 for r in robots)
    assert sum(r["controller_s"] for r in robots) <= s["controller_s"]
    assert s["robot_max_step_s"] == max(r["controller_max_step_s"] for r in robots)
    assert s["robot_min_step_s"] == min(r["controller_min_step_s"] for r in robots)
    assert 0 < s["controller_min_step_s"] <= s["controller_ms_per_step"] / 1e3 <= s["controller_max_step_s"]
