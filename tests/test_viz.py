import numpy as np
from PIL import Image

from conftest import make_line_scenario
from marsupial_search.controllers import DEFAULT_TEAM, make_team
from marsupial_search.sim import SimConfig, Simulator
from marsupial_search.viz.plot import _robot_color, animate_result


def test_gif_has_frames_and_exact_robot_colours(tmp_path):
    sc = make_line_scenario(n_uav=2)
    res = Simulator(sc, SimConfig(terminate_on=(), max_time_s=300, belief_snapshot_interval_s=30)).run(
        make_team(DEFAULT_TEAM))
    path = tmp_path / "m.gif"
    animate_result(sc, res, str(path), step_s=30)
    im = Image.open(path)
    assert im.n_frames == len(np.append(np.arange(0, res.traj_t[-1], 30), res.traj_t[-1]))
    colours = {tuple(c) for c in np.asarray(im.convert("RGB")).reshape(-1, 3)}
    for rid, spec in sc.robots.items():
        hexcol = _robot_color(rid, spec.robot_type).lstrip("#")
        assert tuple(int(hexcol[i:i + 2], 16) for i in (0, 2, 4)) in colours
