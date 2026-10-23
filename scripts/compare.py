"""Compare teams over map seeds x target seeds.

A team is a centralised controller (e.g. joint_planner) or a policy per robot
(robot=policy pairs, or a mapping in YAML). Every team sees exactly the same
(map, target) pairs. Settings come from the base scenario, then the YAML file
(--config: the shared settings and its `compare:` section), then explicit flags.

Two estimators of the expected time to find (a mission that never finds the target
counts at its end time):

* ``cv`` (default): per team and map, one target-free run predicts the find time of
  every possible target (marsupial_search/sim/evaluation.py), and ``targets`` real
  missions (default 2) correct the prediction's bias. Far fewer missions for the same
  precision. Writes maps.csv (one row per team and map) as well as runs.csv.
* ``plain``: the mean over ``targets`` missions per map (default 10).

Writes runs.csv (one row per mission), the resolved settings (config.yaml), prints a
summary table and saves a plot of the time-to-find distributions.

Example:
    python scripts/compare.py --config configs/small_hiker.yaml
    python scripts/compare.py --maps 0 1 2 3 4 --workers 8
    python scripts/compare.py --scenario small --maps 0 1 2 3 4 --estimator plain --targets 10
    python scripts/compare.py --teams joint_planner ugv0=ugv_timed_route,uav0=uav_greedy_tour,uav1=uav_greedy_tour
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import pathlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from marsupial_search.controllers import make_team, parse_team, team_kind, team_label
from marsupial_search.sim import SCENARIO_BASES, SEMANTIC_LABELS, Simulator, generate_scenario, resample_target
from marsupial_search.sim import evaluation
from marsupial_search.sim.config_io import (
    DEFAULT_TEAMS,
    ESTIMATORS,
    build_configs,
    default_targets,
    dump_resolved,
    load_yaml,
    scenario_base,
    section,
)


def _pick(*values, default=None):
    return next((v for v in values if v is not None), default)


def _mission(sc, sim_cfg, map_seed, target_seed, team, params) -> dict:
    s2 = resample_target(sc, np.random.default_rng(10_000 * map_seed + target_seed))
    res = Simulator(s2, sim_cfg).run(make_team(team, params))
    row = res.summary()
    row.update(kind=team_kind(team), map_seed=map_seed, target_seed=target_seed,
               target_terrain=int(s2.terrain[s2.grid.world_to_cell(*s2.target_xy)]))
    return row, s2


def _run(job):
    scen_cfg, sim_cfg, map_seed, target_seed, team, params = job
    sc = generate_scenario(dataclasses.replace(scen_cfg, seed=map_seed))
    return _mission(sc, sim_cfg, map_seed, target_seed, team, params)[0]


def _run_map(job):
    """cv: one team on one map: a target-free run, then the missions that correct its prediction."""
    scen_cfg, sim_cfg, map_seed, targets, team, params = job
    sc = generate_scenario(dataclasses.replace(scen_cfg, seed=map_seed))
    free = Simulator(sc, evaluation.target_free(sim_cfg)).run(make_team(team, params))
    T, W = evaluation.predicted_find_times(sc, free, sim_cfg), evaluation.target_weights(sc)
    rows = []
    for k in range(targets):
        row, s2 = _mission(sc, sim_cfg, map_seed, k, team, params)
        row["predicted_s"] = evaluation.predicted_time(sc, T, s2.target_xy, s2.target_layer)
        rows.append(row)
    expected = evaluation.expected(T, W)
    resid = [r["time_s"] - r["predicted_s"] for r in rows]
    summary = {"controller": team_label(team), "kind": team_kind(team), "map_seed": map_seed,
               "estimate_s": expected + (np.mean(resid) if resid else 0.0), "predicted_s": expected,
               "mean_residual_s": np.mean(resid) if resid else np.nan,
               "predicted_median_s": evaluation.weighted_quantile(T, W, 0.5),
               "predicted_p90_s": evaluation.weighted_quantile(T, W, 0.9),
               "target_free_end_s": free.time_s, "target_free_outcome": free.outcome}
    keep = W > 0
    return rows, summary, (T[keep], W[keep])


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="YAML experiment file (see configs/example.yaml)")
    ap.add_argument("--scenario", choices=sorted(SCENARIO_BASES),
                    help="base scenario: large (5 km x 5 km) or small (1.5 km x 1.5 km)")
    ap.add_argument("--teams", nargs="+", help="centralised controllers and/or robot=policy pairs "
                                               "(default: joint_planner and the default team)")
    ap.add_argument("--estimator", choices=ESTIMATORS, help="cv (default) or plain")
    ap.add_argument("--maps", nargs="+", type=int, help="map seeds (default: 0 1 2 3 4)")
    ap.add_argument("--targets", type=int, help="missions per team and map (default: 2 for cv, 10 for plain)")
    ap.add_argument("--object", choices=sorted(SEMANTIC_LABELS), help="target object for the semantic prior")
    ap.add_argument("--max-time-h", type=float, help="override the time limit")
    ap.add_argument("--workers", type=int, help="parallel processes (default: 4)")
    ap.add_argument("--out", help="default: outputs/compare_<base scenario>[_<object>]")
    args = ap.parse_args()

    doc = load_yaml(args.config) if args.config else {}
    compare = section(doc, "compare")
    base = scenario_base(doc, args.scenario)
    scen_cfg, sim_cfg = build_configs(doc, base)
    teams = [parse_team(t) for t in _pick(args.teams, doc.get("teams"), default=DEFAULT_TEAMS)]
    labels = [team_label(t) for t in teams]
    estimator = _pick(args.estimator, compare["estimator"])
    maps = _pick(args.maps, compare["maps"])
    targets = _pick(args.targets, compare["targets"], default=default_targets(estimator))
    workers = _pick(args.workers, compare["workers"])
    all_params = doc.get("controller_params") or {}
    for t in teams:
        make_team(t, all_params)  # fail fast on bad names or parameters

    if args.object is not None:
        scen_cfg.target_object = args.object
    if args.max_time_h is not None:
        sim_cfg.max_time_s = args.max_time_h * 3600

    map_rows, pooled = [], {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        if estimator == "cv":
            jobs = [(scen_cfg, sim_cfg, m, targets, team, all_params) for m in maps for team in teams]
            rows = []
            for rs, summary, dist in ex.map(_run_map, jobs):
                rows += rs
                map_rows.append(summary)
                pooled.setdefault(summary["controller"], []).append(dist)
        else:
            jobs = [(scen_cfg, sim_cfg, m, t, team, all_params) for m in maps for t in range(targets) for team in teams]
            rows = list(ex.map(_run, jobs))

    default_out = f"outputs/compare_{base}" + (f"_{scen_cfg.target_object}" if scen_cfg.target_object != "car" else "")
    out = pathlib.Path(_pick(args.out, doc.get("out"), default=default_out))
    out.mkdir(parents=True, exist_ok=True)
    if rows:
        _write_csv(out / "runs.csv", rows)
    if map_rows:
        _write_csv(out / "maps.csv", map_rows)
    run = {"teams": teams, "controller_params": all_params, "out": str(out),
           "compare": {"maps": list(maps), "targets": targets, "estimator": estimator, "workers": workers}}
    dump_resolved(out / "config.yaml", run, scen_cfg, sim_cfg, base)

    w = max(len(c) for c in labels)
    if estimator == "cv":
        print(f"expected time to find [min], {len(maps)} maps; ± is the standard error over maps; "
              f"median and p90 are predicted")
        print(f"{'team':>{w}} {'kind':>13} {'missions':>8} {'found':>6} {'estimate':>14} {'median':>7} {'p90':>7} "
              f"{'sd(T-T̂)':>9} {'sorties':>8} {'depleted':>8} {'ctl ms/step':>11}")
        est = {}
        for team, c in zip(teams, labels):
            ms = [r for r in map_rows if r["controller"] == c]
            rs = [r for r in rows if r["controller"] == c]
            est[c] = np.array([r["estimate_s"] for r in sorted(ms, key=lambda r: r["map_seed"])]) / 60
            se = est[c].std(ddof=1) / np.sqrt(len(est[c])) if len(est[c]) > 1 else np.nan
            T = np.concatenate([d[0] for d in pooled[c]])
            W = np.concatenate([d[1] / len(pooled[c]) for d in pooled[c]])
            resid = np.array([r["time_s"] - r["predicted_s"] for r in rs]) / 60
            print(f"{c:>{w}} {team_kind(team):>13} {len(rs):>8} "
                  f"{np.mean([r['found'] for r in rs]) if rs else np.nan:>6.0%} "
                  f"{est[c].mean():>7.1f} ± {se:<4.1f} {evaluation.weighted_quantile(T, W, 0.5) / 60:>7.1f} "
                  f"{evaluation.weighted_quantile(T, W, 0.9) / 60:>7.1f} "
                  f"{resid.std(ddof=1) if len(resid) > 1 else np.nan:>9.1f} "
                  f"{np.mean([r['sorties'] for r in rs]) if rs else np.nan:>8.1f} {sum(r['depleted'] for r in rs):>8} "
                  f"{1e3 * sum(r['controller_s'] for r in rs) / max(sum(r['controller_steps'] for r in rs), 1):>11.2f}")
        for c in labels[1:]:
            d = est[c] - est[labels[0]]
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
            print(f"  {c} − {labels[0]}: {d.mean():+.1f} ± {se:.1f} min (paired by map)")
    else:
        print(f"{'team':>{w}} {'kind':>13} {'runs':>5} {'found':>6} {'mean min':>9} {'median':>7} {'p90':>7} "
              f"{'searched':>9} {'end min':>8} {'sorties':>8} {'depleted':>8} {'ctl ms/step':>11}")
        for team, c in zip(teams, labels):
            rs = [r for r in rows if r["controller"] == c]
            t = np.array([r["time_s"] for r in rs]) / 60
            print(f"{c:>{w}} {team_kind(team):>13} {len(rs):>5} {np.mean([r['found'] for r in rs]):>6.0%} "
                  f"{np.mean(t):>9.1f} {np.median(t):>7.1f} {np.percentile(t, 90):>7.1f} "
                  f"{np.mean([r['searched_prob'] for r in rs]):>9.1%} {np.mean([r['time_s'] for r in rs]) / 60:>8.1f} "
                  f"{np.mean([r['sorties'] for r in rs]):>8.1f} {sum(r['depleted'] for r in rs):>8} "
                  f"{1e3 * sum(r['controller_s'] for r in rs) / sum(r['controller_steps'] for r in rs):>11.2f}")
    if rows:
        print("outcomes:", {o: sum(r["outcome"] == o for r in rows) for o in sorted({r["outcome"] for r in rows})})

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for c in labels:
        if estimator == "cv":  # predicted distribution over every possible target
            T = np.concatenate([d[0] for d in pooled[c]]) / 60
            W = np.concatenate([d[1] / len(pooled[c]) for d in pooled[c]])
            o = np.argsort(T)
            ax.step(np.concatenate([[0], T[o]]), np.concatenate([[0], np.cumsum(W[o])]), where="post", label=c)
        else:
            t = np.sort([r["time_s"] / 60 for r in rows if r["controller"] == c and r["found"]])
            n = sum(r["controller"] == c for r in rows)
            ax.step(np.concatenate([[0], t]), np.arange(len(t) + 1) / n, where="post", label=c)
    ax.set_xlabel("time [min]")
    ax.set_ylabel("predicted P(found by t)" if estimator == "cv" else "fraction of missions with target found")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "time_to_find.png", dpi=120)
    written = ["runs.csv"] + (["maps.csv"] if map_rows else []) + ["config.yaml", "time_to_find.png"]
    print(f"wrote {', '.join(str(out / f) for f in written)}")


if __name__ == "__main__":
    main()
