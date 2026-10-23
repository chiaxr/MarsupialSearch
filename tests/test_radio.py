from dataclasses import dataclass

import numpy as np
import pytest

from conftest import make_line_scenario
from marsupial_search.core import EnergySpec, EventType, GoToPoint, MissionController, RobotSpec, RobotType, Transmit
from marsupial_search.sim import RadioConfig, SimConfig, Simulator


@dataclass
class Claim:  # an algorithm-defined payload type
    node: int
    utility: float


def _sim(radio=None, n_uav=2, **kw):
    # ugv0 at x=25 carrying uav0, uav1 hovering at x=125 (100 m apart)
    sc = make_line_scenario(n_uav=n_uav, **kw)
    return Simulator(sc, SimConfig(terminate_on=(), radio=radio or RadioConfig()))


def test_broadcast_reaches_everyone_else_next_step():
    sim = _sim()
    obs = sim.step({"ugv0": Transmit("hello", {"n": 1})})
    assert set(obs.radio) == {"uav1", "uav0"}
    m = obs.radio["uav1"][0]
    assert (m.sender, m.topic, m.payload, m.recipients) == ("ugv0", "hello", {"n": 1}, None)
    assert m.sent_time == 0.0 and m.received_time == 1.0 and m.seq == 0 and m.size_bytes > 0
    assert sim.step({}).radio == {}  # delivered once


def test_unicast_only_reaches_recipient_and_inbox_filters_topics():
    sim = _sim()
    obs = sim.step({"ugv0": [Transmit("a", 1, to=("uav1",)), Transmit("b", 2, to=("uav1",))]})
    assert set(obs.radio) == {"uav1"}
    assert [m.payload for m in obs.inbox("uav1")] == [1, 2]
    assert [m.seq for m in obs.inbox("uav1")] == [0, 1]
    assert [m.payload for m in obs.inbox("uav1", topic="b")] == [2]


def test_out_of_range_is_dropped():
    sim = _sim(RadioConfig(range_m={RobotType.UGV: 50.0, RobotType.UAV: 50.0}))
    obs = sim.step({"ugv0": Transmit("x", 0)})
    assert "uav1" not in obs.radio and "uav0" in obs.radio  # uav0 is docked on ugv0
    stats = sim.result().robot_stats["ugv0"]
    assert stats["radio_sent"] == 1 and stats["radio_dropped_range"] == 1


def test_link_range_is_the_shorter_of_both_radios():
    sim = _sim(RadioConfig(range_m={RobotType.UGV: 150.0, RobotType.UAV: 5000.0}), n_uav=3)
    obs = sim.step({"uav2": Transmit("x", 0)})  # uav2 hovers 200 m from ugv0 and the uav0 it carries
    assert set(obs.radio) == {"uav0", "uav1"}  # ugv0's 150 m radio limits the link to it


@pytest.mark.parametrize("payload", [{"cells": [1, 2, 3]}, np.arange(6.0).reshape(2, 3), Claim(4, 0.5), b"\x00\x01"])
def test_generic_payloads_arrive_equal_but_not_shared(payload):
    sim = _sim()
    obs = sim.step({"ugv0": Transmit("p", payload)})
    a, b = obs.radio["uav1"][0].payload, obs.radio["uav0"][0].payload
    if isinstance(payload, np.ndarray):
        assert np.array_equal(a, payload) and a is not payload and a is not b
    else:
        assert a == payload
        if not isinstance(payload, bytes):
            assert a is not payload and a is not b


def test_receiver_mutation_does_not_leak_back():
    sim = _sim()
    sent = {"visited": [1, 2]}
    obs = sim.step({"ugv0": Transmit("v", sent)})
    obs.radio["uav1"][0].payload["visited"].append(99)
    assert sent == {"visited": [1, 2]} and obs.radio["uav0"][0].payload == {"visited": [1, 2]}


def test_latency_delays_delivery():
    sim = _sim(RadioConfig(latency_s=2.5))
    times = []
    obs = sim.step({"ugv0": Transmit("x", 0, to=("uav1",))})
    for _ in range(4):
        if obs.radio:
            times.append(obs.time)
        obs = sim.step({})
    assert times == [3.0]


def test_loss_and_bandwidth_and_payload_limit():
    sim = _sim(RadioConfig(drop_prob=1.0))
    assert sim.step({"ugv0": Transmit("x", 0)}).radio == {}
    assert sim.result().robot_stats["ugv0"]["radio_dropped_loss"] == 2

    sim = _sim(RadioConfig(max_bytes_per_s=100.0))
    obs = sim.step({"ugv0": [Transmit("x", b"a" * 60, to=("uav1",)), Transmit("x", b"b" * 60, to=("uav1",))]})
    assert [m.payload[:1] for m in obs.inbox("uav1")] == [b"a"]
    assert sim.result().robot_stats["ugv0"]["radio_dropped_bandwidth"] == 1

    sim = _sim(RadioConfig(max_payload_bytes=10))
    obs = sim.step({"ugv0": Transmit("x", b"z" * 100)})
    assert obs.radio == {} and EventType.COMMAND_REJECTED in [e.type for e in obs.events]


def test_transmit_alongside_motion_command():
    sim = _sim()
    obs = sim.step({"uav1": [GoToPoint((125.0, 600.0)), Transmit("pos", (125.0, 500.0))]})
    assert obs.robots["uav1"].moving
    assert obs.inbox("ugv0")[0].payload == (125.0, 500.0)


def test_depleted_robots_neither_send_nor_receive():
    sc = make_line_scenario(n_uav=2)
    spec = sc.robots["uav1"]
    sc.robots["uav1"] = RobotSpec(spec.robot_id, spec.robot_type, spec.speed, EnergySpec(1.0, 1.0, 0.05), spec.sensor)
    sim = Simulator(sc, SimConfig(terminate_on=()))
    sim.step({"uav1": GoToPoint((125.0, 600.0))})  # runs flat within the first step
    obs = sim.step({"ugv0": Transmit("x", 0), "uav1": Transmit("y", 1)})
    assert set(obs.radio) == {"uav0"} and [m.sender for m in obs.inbox("uav0")] == ["ugv0"]


def test_controller_round_trip_and_setup_range():
    class Chatter(MissionController):
        name = "chatter"

        def reset(self, setup):
            self.range = setup.radio_range_m
            self.heard = []

        def step(self, obs):
            self.heard += [(rid, m.sender, m.payload) for rid, msgs in obs.radio.items() for m in msgs]
            return {"ugv0": Transmit("tick", {"t": obs.time})} if obs.time < 3 else {}

    sc = make_line_scenario(n_uav=2)
    ctl = Chatter()
    Simulator(sc, SimConfig(terminate_on=(), max_time_s=5)).run(ctl)
    assert ctl.range == RadioConfig().range_m
    assert [p["t"] for rid, s, p in ctl.heard if rid == "uav1"] == [0.0, 1.0, 2.0]


def test_ideal_channel_ignores_range_loss_latency_and_limits():
    radio = RadioConfig(simulate_channel=False, range_m={RobotType.UGV: 1.0, RobotType.UAV: 1.0}, drop_prob=1.0,
                        latency_s=30.0, max_bytes_per_s=1.0, max_payload_bytes=1)
    sim = _sim(radio)
    assert sim.setup.radio_range_m == {RobotType.UGV: float("inf"), RobotType.UAV: float("inf")}
    payload = {"big": list(range(1000))}
    obs = sim.step({"ugv0": [Transmit("a", payload), Transmit("b", payload, to=("uav1",))]})
    assert [m.topic for m in obs.inbox("uav1")] == ["a", "b"] and [m.topic for m in obs.inbox("uav0")] == ["a"]
    assert obs.inbox("uav1")[0].payload == payload and obs.inbox("uav1")[0].payload is not payload
    s = sim.result().robot_stats["ugv0"]
    assert s["radio_sent"] == 2 and s["radio_dropped_range"] == s["radio_dropped_loss"] == 0


def test_disabled_radio_sends_nothing():
    sim = _sim(RadioConfig(enabled=False))
    assert sim.setup.radio_range_m is None
    assert sim.step({"ugv0": Transmit("x", 0)}).radio == {}


def test_payload_sizes_are_summarised():
    sim = _sim()
    sim.step({"ugv0": [Transmit("small", 1), Transmit("big", list(range(100)))]})
    sim.step({"uav1": Transmit("mid", list(range(10)))})
    sizes = [len(sim.radio.cfg.codec.encode(p)) for p in (1, list(range(100)), list(range(10)))]
    s = sim.result().summary()
    assert s["radio_payload_min_bytes"] == min(sizes) and s["radio_payload_max_bytes"] == max(sizes)
    assert s["radio_payload_avg_bytes"] == pytest.approx(sum(sizes) / 3)
    assert sim.result().robot_stats["uav1"]["radio_payload_min_bytes"] == sizes[2]
    assert np.isnan(sim.result().robot_stats["uav0"]["radio_payload_max_bytes"])  # sent nothing
