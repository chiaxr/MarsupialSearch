"""False sightings (``sim.false_alarms``): a UAV flying high mistakes clutter and decoys for the target
and sights it off by a position error. The belief weighs each sighting by its likelihood ratio
(``BeliefMap``), the controllers pursue one only while the belief makes it likely
(``common.Lead``), and the ground looks (the UGV, a UAV at street level, a confirmer) never sight
wrongly."""

import dataclasses

import numpy as np
import pytest

from conftest import make_line_scenario
from marsupial_search.controllers import make_team
from marsupial_search.controllers.common import Lead, SearchConfig
from marsupial_search.core import (
    BeliefMap,
    EventType,
    GoToPoint,
    Layer,
    LocalBeliefUpdate,
    RobotType,
    SensorSpec,
    SightingModel,
    Terrain,
    Undock,
)
from marsupial_search.sim import SimConfig, Simulator, generate_scenario
from marsupial_search.sim.evaluation import look_events, predicted_find_times, target_free
from marsupial_search.sim.false_alarms import FalseAlarms
from marsupial_search.sim.generator import TARGET_VISIBILITY
from marsupial_search.sim.scenario_bases import small
from marsupial_search.sim.simulator import robot_sensing

S = int(Layer.SHARED)


def _fa(sc, clutter=0.0, decoys=(), density=1e-7, decoy_p=1.0, sigma=5.0, max_p=0.9):
    """``sc`` with uniform clutter (per m² per look) and decoys at ``decoys`` (in the shared layer)."""
    xy = np.asarray(decoys, dtype=float).reshape(-1, 2)
    fa = FalseAlarms(clutter=np.full(sc.grid.shape, clutter), decoy_density=np.full(sc.grid.shape, density),
                     decoys=xy, decoy_layers=np.full(len(xy), S), decoy_p=decoy_p, sigma_m=sigma, max_p=max_p)
    return dataclasses.replace(sc, false_alarms=fa)


# ----------------------------------------------------------------- the belief
def _belief(lam=1e-6, sigma=10.0, decoy_share=None, decoy_p=1.0):
    sc = make_line_scenario()
    decoys = None if decoy_share is None else np.full(sc.grid.shape, decoy_share * lam)
    model = SightingModel(false_alarms=np.full(sc.grid.shape, lam), sigma_m=sigma, decoys=decoys, decoy_p=decoy_p)
    return sc, BeliefMap(sc.grid, sc.prior, sc.layer_fractions, {"uav0": model})


def _look(sc, xy, t, p=0.9, sightings=(), radius=100.0, rid="uav0"):
    r0, c0, mask = sc.grid.disk_window(*xy, radius)
    pd = np.zeros((3,) + mask.shape)
    pd[S] = p * mask
    return LocalBeliefUpdate(rid, RobotType.UAV, t, r0, c0, pd, sightings=tuple(sightings))


def test_a_sighting_multiplies_the_belief_by_its_likelihood_ratio():
    sc, b = _belief()
    z = (512.5, 712.5)
    m0 = b.mass.copy()
    before = b.p_near(z)
    b.apply(_look(sc, (500.0, 700.0), 1.0, sightings=[z]))
    X, Y = sc.grid.cell_centers()
    s2 = 10.0 ** 2
    g = np.exp(-((X - z[0]) ** 2 + (Y - z[1]) ** 2) / (2 * s2)) / (2 * np.pi * s2)
    r0, c0, mask = sc.grid.disk_window(500.0, 700.0, 100.0)
    win = np.zeros(sc.grid.shape, dtype=bool)
    win[r0:r0 + mask.shape[0], c0:c0 + mask.shape[1]] = mask
    ratio = np.where(win, 0.1 + 0.9 * g / 1e-6, 1.0)  # (1 - p) + p · g / λ
    assert np.allclose(b.mass[S], m0[S] * ratio)
    assert np.allclose(b.looked[S], m0[S] * np.where(win, 0.1, 1.0))  # the looks, as if it sighted nothing
    assert b.anchors == [z] and b.p_near(z) > 0.5 > 50 * before
    assert b.searched_probability() == pytest.approx(1 - b.looked.sum() / b.initial_total)


def test_a_pass_that_misses_it_lowers_it_less_where_decoys_are_and_a_repeat_counts_against_clutter():
    z, p2s = (512.5, 712.5), []
    for share in (0.0, 0.5, 1.0):
        sc, b = _belief(lam=1e-5, decoy_share=share, decoy_p=0.8)
        b.apply(_look(sc, (500.0, 700.0), 1.0, sightings=[z]))
        p1 = b.p_near(z)
        assert b.decoy_chance == [share]
        b.apply(_look(sc, (500.0, 700.0), 2.0))  # another pass, nothing there
        p2 = b.p_near(z)
        p2s.append(p2)
        assert p2 < p1 and np.allclose(b.mass, b.looked * b.evidence)
        dp = 0.8 * 0.9  # a decoy there would have been sighted with decoy_p times the look's chance
        assert b.decoy_chance[0] == pytest.approx(share * (1 - dp) / (1 - share * dp))  # less likely a decoy
        near = b.mass.copy()
        b.apply(_look(sc, (500.0, 700.0), 3.0, sightings=[(520.0, 700.0)]))  # sighted again, 15 m off: a repeat
        p3 = b.p_near(z)
        assert b.anchors == [z] and p3 > p2
        if share == 0.0:  # clutter does not come back: it was likely the target
            assert p3 > 10 * p2 and b.decoy_chance == [0.0]
        elif share == 0.5:  # the miss made a decoy unlikely, so the repeat counts
            assert p3 > 3 * p2 and b.decoy_chance[0] > 0.8
        else:  # only decoys: a repeat is little news
            assert p3 < 1.5 * p2 and b.decoy_chance == [1.0]
        far = sc.grid.world_to_cell(500.0, 610.0)  # in the look, outside the gate: missed again
        assert b.mass[(S,) + far] == pytest.approx(0.1 * near[(S,) + far])
    assert 3 * p2s[0] < p2s[2] and p2s[0] < p2s[1] < p2s[2]  # the miss lowers it less, the likelier a decoy


def test_a_pass_in_slivers_is_one_miss():
    sc, whole = _belief(lam=1e-5, decoy_share=1.0, decoy_p=0.8)
    _, parts = _belief(lam=1e-5, decoy_share=1.0, decoy_p=0.8)
    z = (512.5, 712.5)
    for b in (whole, parts):
        b.apply(_look(sc, (500.0, 700.0), 1.0, sightings=[z]))
    whole.apply(_look(sc, z, 2.0, radius=60.0))
    r0, c0, mask = sc.grid.disk_window(*z, 60.0)
    for k, rows in enumerate(np.array_split(np.arange(mask.shape[0]), 3)):  # three steps, each the rows newly in view
        pd = np.zeros((3,) + mask.shape)
        pd[S, rows] = 0.9 * mask[rows]
        parts.apply(LocalBeliefUpdate("uav0", RobotType.UAV, 2.0 + k / 10, r0, c0, pd))
    assert len(parts.events) == 4
    assert parts.decoy_chance[0] == pytest.approx(whole.decoy_chance[0], abs=0.02)
    assert parts.p_near(z) == pytest.approx(whole.p_near(z), rel=0.1)
    far = _look(sc, (100.0, 100.0), 5.0, radius=30.0)  # nowhere near it: not kept
    parts.apply(far)
    assert len(parts.events) == 4


def test_fused_beliefs_agree_whatever_order_the_sightings_come_in():
    sc, a = _belief()
    _, b = _belief()
    _, c = _belief()
    u1 = _look(sc, (500.0, 700.0), 1.0, sightings=[(512.5, 712.5)])
    u2 = _look(sc, (520.0, 700.0), 2.0, sightings=[(525.0, 705.0), (300.0, 300.0)], radius=500.0)  # a repeat, and new
    a.apply(u1)
    a.apply(u2)
    b.apply(u2)  # out of order: u2's first sighting is the anchor until u1 comes
    assert len(b.anchors) == 2
    b.apply(u1)
    assert b.anchors == a.anchors and np.allclose(b.mass, a.mass)
    c.fuse(a.sync_payload())  # by sync: the looks by minimum, the sightings by union
    assert c.anchors == a.anchors and np.allclose(c.mass, a.mass, rtol=1e-6)
    assert [z for z, _ in a.hypotheses()][0] in a.anchors


def test_without_a_sighting_model_the_belief_is_as_before():
    sc = make_line_scenario()
    b = BeliefMap(sc.grid, sc.prior, sc.layer_fractions)
    b.apply(_look(sc, (500.0, 700.0), 1.0, sightings=[(512.5, 712.5)]))  # from a robot it has no model for
    assert b.mass is b.looked and b.evidence is None and not b.anchors
    assert set(b.sync_payload()) == {"mass", "sighting"}


# ------------------------------------------------------------------- the lead
class _FakeBelief:
    def __init__(self):
        self.detection_xy, self.anchors, self.p = None, [], {}

    def hypotheses(self):
        return sorted(self.p.items(), key=lambda t: -t[1])

    def p_near(self, xy):
        return self.p.get(xy, 0.0)


def test_the_lead_is_pursued_while_the_belief_makes_it_likely():
    lead, b = Lead(SearchConfig(lead_pursue_p=0.5, lead_drop_p=0.1)), _FakeBelief()
    z, w = (100.0, 100.0), (300.0, 300.0)
    b.anchors, b.p = [z], {z: 0.4}
    assert not lead.update(b, 1.0) and lead.xy is None  # not likely enough: the search may get to it
    b.p[z] = 0.6
    assert lead.update(b, 2.0) and lead.xy == z and not lead.hard
    b.anchors, b.p = [z, w], {z: 0.2, w: 0.7}
    assert not lead.update(b, 3.0) and lead.xy == z  # still above the drop
    b.p[z] = 0.05
    assert lead.update(b, 4.0) and lead.xy == w  # dropped, and the next one taken up
    lead.verified()
    b.detection_xy = (5.0, 5.0)
    assert lead.update(b, 5.0) and lead.hard and lead.xy == (5.0, 5.0)
    b.p = {}
    assert not lead.update(b, 6.0) and lead.xy == (5.0, 5.0)  # a certain one is never dropped
    assert [(ld["t0"], ld["t1"], ld["xy"], ld["verify"]) for ld in lead.log] == [
        (2.0, 4.0, z, 0), (4.0, 5.0, w, 1), (5.0, None, (5.0, 5.0), 0)]


# -------------------------------------------------------------- the simulator
def test_only_a_high_uav_that_does_not_end_the_mission_sights_wrongly():
    sc = _fa(make_line_scenario(n_uav=2))
    sim = Simulator(sc, SimConfig())
    assert set(sim.setup.sightings) == {"uav0", "uav1"}
    m = sim.setup.sightings["uav0"]  # the decoys' share of its false sightings, and how they are sighted
    assert m.decoy_p == 1.0 and np.allclose(m.false_alarms, sc.false_alarms.clutter + m.decoys) and m.decoys.max() > 0
    assert sim.setup.detection["uav0"].max() == pytest.approx(0.9)  # nothing is certain from the air
    assert sim.setup.detection["ugv0"].max() == 1.0
    assert set(Simulator(sc, SimConfig(confirm_by=("uav0",))).setup.sightings) == {"uav1"}  # the check sights truly
    low = SensorSpec(60.0, (Layer.SHARED, Layer.GROUND), viewpoint="ground")
    sc2 = dataclasses.replace(sc, robots={**sc.robots, "uav1": dataclasses.replace(sc.robots["uav1"], sensor=low)})
    assert set(Simulator(sc2, SimConfig()).setup.sightings) == {"uav0"}  # at street level
    assert not Simulator(make_line_scenario(n_uav=2), SimConfig()).setup.sightings


def _fly(sc, path, seed=0, target=True, steps=60):
    """uav0 takes off and flies through ``path``; the simulator after that."""
    sim = Simulator(sc, SimConfig(terminate_on=(), seed=seed, target=target, max_time_s=1e5))
    sim.step({"uav0": Undock()})
    for _ in range(5):
        sim.step({})
    for xy in path:
        cmds = {"uav0": GoToPoint(xy)}
        for _ in range(steps):
            sim.step(cmds)
            cmds = {}
    return sim


def test_clutter_comes_at_its_rate_per_look():
    lam = 4e-4  # per m² per look: 0.25 per 25 m cell
    sc = _fa(make_line_scenario(), clutter=lam)
    path = [(100.0, 150.0), (900.0, 150.0), (900.0, 850.0), (100.0, 850.0)]
    sim = _fly(sc, path, target=False, steps=90)
    n = sum(k == "clutter" for *_, k in sim.sightings)
    cells, _ = look_events(sc, sim.result(), "uav0")
    expect = lam * sc.grid.resolution ** 2 * len(cells)
    assert expect > 100 and abs(n - expect) < 4 * np.sqrt(expect)
    assert sim.result().summary()["sightings_clutter"] == n


def test_a_decoy_is_sighted_with_its_chance_once_per_pass():
    decoy = (500.0, 800.0)
    sc = _fa(make_line_scenario(), decoys=[decoy], decoy_p=0.5)  # 0.5 · 0.9 per pass
    once = [sum(k == "decoy" for *_, k in _fly(sc, [decoy], seed=s, target=False).sightings) for s in range(60)]
    assert max(once) == 1 and np.mean(once) == pytest.approx(0.45, abs=0.15)  # hovering over it: one pass
    twice = [sum(k == "decoy" for *_, k in _fly(sc, [decoy, (500.0, 300.0), decoy], seed=s, target=False).sightings)
             for s in range(30)]
    assert max(twice) == 2


def test_a_true_sighting_from_the_air_is_off_by_its_error_and_ends_nothing():
    xy = (512.5, 812.5)
    sc = _fa(make_line_scenario(target_xy=xy), sigma=10.0)
    errs = []
    for seed in range(30):
        sim = _fly(sc, [xy], seed=seed)
        mine = [z for *_, z, k in sim.sightings if k == "target"]
        assert not sim.done and len(mine) <= 1
        assert (EventType.TARGET_SIGHTED in [e.type for e in sim.events]) == bool(mine)
        errs += [np.hypot(z[0] - xy[0], z[1] - xy[1]) for z in mine]
    assert 15 <= len(errs) and 0 < np.mean(errs) < 25  # Rayleigh: 10 · √(π/2) ≈ 12.5 m


def test_false_sightings_do_not_depend_on_the_target():
    sc = _fa(make_line_scenario(target_xy=(512.5, 812.5)), clutter=1e-4, decoys=[(300.0, 300.0), (700.0, 700.0)])
    path = [(300.0, 300.0), (512.5, 812.5), (700.0, 700.0)]
    with_target = [s for s in _fly(sc, path, seed=3).sightings if s[3] != "target"]
    without = _fly(sc, path, seed=3, target=False).sightings
    assert with_target == without and len(without) > 5


def test_a_sighting_comes_in_the_update_with_its_looks():
    xy = (512.5, 812.5)
    sc = _fa(make_line_scenario(target_xy=xy))
    sim = Simulator(sc, SimConfig(terminate_on=()))
    sim.step({"uav0": Undock()})
    for _ in range(5):
        sim.step({})
    cmds, got = {"uav0": GoToPoint(xy)}, []
    for _ in range(60):
        obs = sim.step(cmds)
        cmds = {}
        got += [u for u in obs.belief_updates if u.sightings]
    (u,) = got
    assert not u.detected and u.p_detect.max() == pytest.approx(0.9)  # it keeps its looks
    assert sim.truth_belief.anchors == [u.sightings[0]] and sim.truth_belief.p_near(xy) > 0.5
    assert 0 < sim.result().log["searched_prob"][-1] < 1  # the looks alone: a sighting does not count


# ------------------------------------------------------------------ the generator
def test_the_generator_places_decoys_by_terrain_on_their_own_stream():
    cfg = dataclasses.replace(small(2), decoys_per_km2=20.0)
    plain, fa = generate_scenario(cfg), generate_scenario(dataclasses.replace(cfg, false_alarms=True))
    assert plain.false_alarms is None
    assert np.array_equal(plain.prior, fa.prior) and plain.target_xy == fa.target_xy
    assert np.array_equal(plain.graphs[RobotType.UGV].positions, fa.graphs[RobotType.UGV].positions)
    f = fa.false_alarms
    w = np.asarray(TARGET_VISIBILITY["car"].clutter)[fa.terrain]
    assert np.allclose(f.clutter, cfg.clutter_per_km2 * 1e-6 * w) and np.allclose(f.decoy_density, 20e-6 * w)
    area = (fa.grid.resolution ** 2) * f.decoy_density.sum()
    assert len(f.decoys) == pytest.approx(area, abs=4 * np.sqrt(area))
    x0, x1, y0, y1 = fa.grid.extent
    assert ((f.decoys >= [x0, y0]) & (f.decoys <= [x1, y1])).all()
    cells = [fa.grid.world_to_cell(*d) for d in f.decoys]
    assert all(fa.layer_fractions[(int(l),) + c] > 0 for l, c in zip(f.decoy_layers, cells))
    urban = np.mean([fa.terrain[c] == int(Terrain.URBAN) for c in cells])
    assert urban > (fa.terrain == int(Terrain.URBAN)).mean()  # parked cars: more of them in town


def test_the_prediction_uses_what_a_high_uav_really_detects():
    cfg = dataclasses.replace(small(1), false_alarms=True)
    sc = generate_scenario(cfg)
    det, models = robot_sensing(sc, ("ugv0",))
    assert set(models) == {"uav0", "uav1"} and max(det[u].max() for u in models) == pytest.approx(cfg.air_max_p)
    sim_cfg = SimConfig(max_time_s=600)
    res = Simulator(sc, target_free(sim_cfg)).run(make_team({"ugv0": "ugv_timed_route", "uav0": "uav_greedy_tour",
                                                             "uav1": "uav_greedy_tour"}))
    T = predicted_find_times(sc, res, sim_cfg)
    assert np.isfinite(T).all() and (T <= 600).all()


# ------------------------------------------------------------------- missions
DECOY = (137.5, 512.5)  # 12.5 m off the road, under uav1 as it starts (over node 2)


@pytest.mark.parametrize("team", ["greedy", "joint"])
def test_a_decoy_lead_is_pursued_until_the_ugv_sees_nothing_there(team):
    sc = _fa(make_line_scenario(n_uav=2, target_xy=(937.5, 537.5)), decoys=[DECOY])
    spec = "joint_planner" if team == "joint" else {"ugv0": "ugv_timed_route", "uav0": "uav_greedy_tour",
                                                    "uav1": "uav_greedy_tour"}
    ctl = make_team(spec)
    sim = Simulator(sc, SimConfig(max_time_s=2000, seed=0))
    res = sim.run(ctl)
    assert any(k == "decoy" for *_, k in res.sightings)
    first = res.leads[0]
    assert np.hypot(first["xy"][0] - DECOY[0], first["xy"][1] - DECOY[1]) < 20 and first["t1"] is not None  # dropped
    assert first["by"] == "ugv0" and first["flights"] == []
    # One spot for the decoy, however often it is sighted, cleared by the pass that dropped the lead.
    (spot,) = res.false_spots
    assert spot["kind"] == "decoy" and spot["xy"] == first["xy"] and spot["n"] >= 2
    assert abs(spot["t_clear"] - first["t1"]) <= 5.0
    s = res.summary()
    assert s["false_leads"] >= 1 and s["false_lead_s"] > 0
    assert res.outcome == "found"
    belief = ctl.belief if team == "joint" else ctl.robots["ugv0"].belief
    assert belief.p_near(first["xy"]) < 0.1


def test_the_gif_draws_a_false_sighting_yellow_then_grey_and_a_line_to_the_lead(tmp_path):
    from PIL import Image
    from marsupial_search.viz.plot import CLEARED_COLOR, SPOT_COLOR, animate_result
    sc = _fa(make_line_scenario(n_uav=2, target_xy=(937.5, 537.5)), decoys=[DECOY])
    res = Simulator(sc, SimConfig(max_time_s=60, seed=0, belief_snapshot_interval_s=5)).run(
        make_team({"ugv0": "ugv_timed_route", "uav0": "uav_greedy_tour", "uav1": "uav_greedy_tour"}))
    t_clear = res.false_spots[0]["t_clear"]
    path = tmp_path / "m.gif"
    animate_result(sc, res, str(path), step_s=5)
    im = Image.open(path)

    def pixels(k, hexcol):  # (the legend shows both diamonds in every frame)
        im.seek(k)
        rgb = tuple(int(hexcol.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
        return int((np.asarray(im.convert("RGB")) == rgb).all(-1).sum())

    before, after = 1, int(t_clear // 5) + 1  # frames at 5 s and just after it is cleared
    assert pixels(before, SPOT_COLOR) > pixels(after, SPOT_COLOR)
    assert pixels(before, CLEARED_COLOR) < pixels(after, CLEARED_COLOR)


def test_the_teams_find_targets_with_false_alarms():
    cfg = dataclasses.replace(small(1), false_alarms=True)
    cfg = dataclasses.replace(cfg, fleet=dataclasses.replace(cfg.fleet, uav_viewpoints={"uav0": "ground"}))
    sc = generate_scenario(cfg)
    for team, by in [({"ugv0": "ugv_timed_route", "uav0": "uav_near_scout", "uav1": "uav_far_scout"}, ("uav0",)),
                     ({"ugv0": "ugv_timed_route", "uav0": "uav_greedy_tour", "uav1": "uav_greedy_tour"}, ())]:
        res = Simulator(sc, SimConfig(max_time_s=3000, confirm_by=by)).run(make_team(team))
        assert res.outcome == "found"
        assert res.sightings and all(rid == "uav1" for _, rid, *_ in res.sightings)  # uav0 flies low
