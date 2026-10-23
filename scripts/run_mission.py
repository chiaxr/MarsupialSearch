"""Run each team once on the same map and target, and save plots.

Settings come from the base scenario, then the YAML file (--config), then explicit flags.
The teams are the file's `teams` (the same list compare.py runs), or --teams. The resolved
settings are written next to the plots as <map>[_t<target seed>]_config.yaml.

Example:
    python scripts/run_mission.py --config configs/small_hiker.yaml
    python scripts/run_mission.py --teams joint_planner --seed 0 --target-seed 7
    python scripts/run_mission.py --teams ugv0=ugv_timed_route,uav0=uav_greedy_tour,uav1=uav_greedy_tour
    python scripts/run_mission.py --no-terminate                # full-coverage run
    python scripts/run_mission.py --no-animate                  # skip the GIFs (faster)
    python scripts/run_mission.py --scenario small              # the 1.5 km map
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import pathlib

import numpy as np

from marsupial_search.controllers import make_team, parse_team, team_kind, team_label
from marsupial_search.core import RobotType
from marsupial_search.sim import SCENARIO_BASES, SEMANTIC_LABELS, Simulator, generate_scenario, resample_target
from marsupial_search.sim.config_io import (
    DEFAULT_TEAMS,
    build_configs,
    dump_resolved,
    load_yaml,
    scenario_base,
    section,
)
from marsupial_search.viz.plot import animate_result, plot_result, plot_scenario


def _pick(*values, default=None):
    return next((v for v in values if v is not None), default)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="YAML experiment file (see configs/example.yaml)")
    ap.add_argument("--scenario", choices=sorted(SCENARIO_BASES),
                    help="base scenario: large (5 km x 5 km) or small (1.5 km x 1.5 km)")
    ap.add_argument("--teams", nargs="+", help="centralised controllers and/or robot=policy pairs "
                                               "(default: joint_planner and the default team)")
    ap.add_argument("--seed", type=int, help="map seed")
    ap.add_argument("--target-seed", type=int, help="resample the target with this seed")
    ap.add_argument("--object", choices=sorted(SEMANTIC_LABELS), help="target object for the semantic prior")
    ap.add_argument("--size", type=float, help="map side length [m]")
    ap.add_argument("--max-time-h", type=float)
    ap.add_argument("--dock-s", type=float, help="docking (landing) duration")
    ap.add_argument("--undock-s", type=float, help="undocking (take-off) duration")
    ap.add_argument("--swap-s", type=float, help="UAV battery swap time")
    ap.add_argument("--no-terminate", action="store_true", help="ignore the target; run to full coverage")
    ap.add_argument("--animate", action=argparse.BooleanOptionalAction, default=None,
                    help="write a GIF per team (default: on)")
    ap.add_argument("--out")
    args = ap.parse_args()

    doc = load_yaml(args.config) if args.config else {}
    mission = section(doc, "mission")
    base = scenario_base(doc, args.scenario)
    cfg, sim_cfg = build_configs(doc, base)
    seed = _pick(args.seed, mission["seed"])
    target_seed = _pick(args.target_seed, mission["target_seed"])
    teams = [parse_team(t) for t in _pick(args.teams, doc.get("teams"), default=DEFAULT_TEAMS)]
    params = doc.get("controller_params") or {}
    for team in teams:
        make_team(team, params)  # fail fast on bad names or parameters
    animate = _pick(args.animate, mission["animate"])
    out = pathlib.Path(_pick(args.out, doc.get("out"), default="outputs"))

    cfg.seed = seed
    if args.object is not None:
        cfg.target_object = args.object
    if args.size is not None:
        cfg.size_m = (args.size, args.size)
    dock = {k: v for k, v in (("dock_duration_s", args.dock_s), ("undock_duration_s", args.undock_s),
                              ("swap_duration_s", args.swap_s)) if v is not None}
    cfg.docking = dataclasses.replace(cfg.docking, **dock)
    if args.max_time_h is not None:
        sim_cfg.max_time_s = args.max_time_h * 3600
    if args.no_terminate:
        sim_cfg.terminate_on = ()

    sc = generate_scenario(cfg)
    if target_seed is not None:
        sc = resample_target(sc, np.random.default_rng(target_seed))
    if animate and sim_cfg.belief_snapshot_interval_s is None:
        # Snapshot as often as a ~100 MB budget allows over the time limit (10 s minimum).
        snapshot_bytes = 4 * sc.grid.rows * sc.grid.cols
        sim_cfg.belief_snapshot_interval_s = max(10.0, float(np.ceil(snapshot_bytes * sim_cfg.max_time_s / 100e6)))

    out.mkdir(parents=True, exist_ok=True)
    stem = sc.name + (f"_t{target_seed}" if target_seed is not None else "")
    run = {"teams": teams, "controller_params": params, "out": str(out),
           "mission": {"seed": seed, "target_seed": target_seed, "animate": animate}}
    dump_resolved(out / f"{stem}_config.yaml", run, cfg, sim_cfg, base)
    plot_scenario(sc, str(out / f"{sc.name}_map.png"))
    print(f"scenario {sc.name}: {sc.graphs[RobotType.UGV].n_nodes} road nodes, "
          f"{sc.meta['road_length_m'] / 1e3:.0f} km road, target {sc.target_xy} ({sc.target_layer.name})")

    for team in teams:
        label = team_label(team)
        res = Simulator(sc, sim_cfg).run(make_team(team, params))
        plot_result(sc, res, str(out / f"{stem}_{label}.png"))
        print(f"team {label} ({team_kind(team)})")
        for k, v in res.summary().items():
            print(f"  {k:>16}: {v:.3f}" if isinstance(v, float) else f"  {k:>16}: {v}")
        print("  events:", dict(collections.Counter(e.type.value for e in res.events)))
        if animate:
            animate_result(sc, res, str(out / f"{stem}_{label}.gif"))
            print(f"  animation: {out / f'{stem}_{label}.gif'}")
    print(f"outputs written to {out}/ (resolved settings: {stem}_config.yaml)")


if __name__ == "__main__":
    main()
