"""Matplotlib plotting for scenarios and mission results."""

from __future__ import annotations

from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from marsupial_search.core import EventType, RobotMode, RobotType, Terrain
from marsupial_search.sim import MissionResult, Scenario

TERRAIN_COLORS = {
    Terrain.ROAD: "#b5aea0",
    Terrain.VEGETATION: "#9cc28a",
    Terrain.URBAN: "#a3a9b8",
}
_TERRAIN_CMAP = ListedColormap([TERRAIN_COLORS[t] for t in Terrain])
ROAD_COLOR = "#4a4a4a"
NODE_FACE, NODE_EDGE = "white", "#333333"
TARGET_COLOR, SIGHTED_COLOR = "gold", "#00e5ff"  # the target star, before and after a robot sees it
UGV_COLORS = ["#c0392b", "#8e44ad", "#d35400", "#2c3e50", "#16a085"]
UAV_COLORS = ["#2980b9", "#27ae60", "#f39c12", "#e84393", "#00b8d4"]


def _robot_color(rid: str, rtype: RobotType) -> str:
    idx = int("".join(ch for ch in rid if ch.isdigit()) or 0)
    pal = UGV_COLORS if rtype == RobotType.UGV else UAV_COLORS
    return pal[idx % len(pal)]


def _node_marker_size(ax, scenario: Scenario) -> float:
    """Scatter size (pt^2) so road nodes read as a dotted line without merging."""
    road = scenario.graphs[RobotType.UGV]
    width_pt = ax.get_window_extent().width * 72.0 / ax.figure.dpi
    spacing_pt = float(np.median(road.lengths)) * width_pt / (scenario.grid.extent[1] - scenario.grid.extent[0])
    return float(np.clip(0.45 * spacing_pt, 1.3, 4.5)) ** 2


def draw_map(ax, scenario: Scenario, prior: bool = True, uav_grid: bool = False, road_nodes: bool = True) -> None:
    g = scenario.grid
    ax.imshow(scenario.terrain, origin="lower", extent=g.extent, cmap=_TERRAIN_CMAP, vmin=-0.5, vmax=len(Terrain) - 0.5,
              interpolation="nearest")
    road = scenario.graphs[RobotType.UGV]
    segs = road.positions[road.edges]
    ax.add_collection(LineCollection(segs, colors=ROAD_COLOR, linewidths=0.8, zorder=2))
    if road_nodes:
        ax.set_aspect("equal")
        ax.scatter(*road.positions.T, s=_node_marker_size(ax, scenario), c=NODE_FACE, edgecolors=NODE_EDGE,
                   linewidths=0.4, zorder=2.6)
    if uav_grid:
        uav = scenario.graphs[RobotType.UAV]
        ax.scatter(*uav.positions.T, s=0.5, c="#555", alpha=0.3, zorder=2)
    if prior:
        X, Y = g.cell_centers()
        p = scenario.prior
        levels = np.quantile(p, [0.5, 0.75, 0.9, 0.97])
        ax.contour(X, Y, p, levels=np.unique(levels), colors="#7a1f5c", linewidths=0.8, alpha=0.8, zorder=3)
    ax.set_xlim(g.extent[0], g.extent[1])
    ax.set_ylim(g.extent[2], g.extent[3])
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def _legend_handles():
    return [Patch(color=TERRAIN_COLORS[t], label=t.name.lower()) for t in Terrain]


def _log10(a: np.ndarray) -> np.ndarray:
    return np.log10(np.maximum(np.asarray(a, dtype=float), 1e-300))


def _log_mass(a: np.ndarray, decades: float = 4.0):
    vmax = float(_log10(a.max()))
    return _log10(a), vmax - decades, vmax


def draw_prior(ax, scenario: Scenario, cbar: bool = True) -> None:
    img, vmin, vmax = _log_mass(scenario.prior, 3.0)
    im = ax.imshow(img, origin="lower", extent=scenario.grid.extent, cmap="magma", vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    road = scenario.graphs[RobotType.UGV]
    ax.add_collection(LineCollection(road.positions[road.edges], colors="w", linewidths=0.3, alpha=0.35))
    ax.plot(*scenario.target_xy, "*", color=TARGET_COLOR, ms=14, mec="k", zorder=6)
    ax.set_aspect("equal")
    if cbar:
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, extend="min")
        cb.set_label(r"prior $P$(target in cell) ($\log_{10}$)", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    cfg = scenario.meta.get("config")
    what = f"semantic prior: '{cfg.target_object}'" if cfg is not None and cfg.prior_model == "semantic" else "prior"
    ax.set_title(what, fontsize=10)


def plot_scenario(scenario: Scenario, path: Optional[str] = None, show_target: bool = True):
    fig, (ax, axp) = plt.subplots(1, 2, figsize=(17, 8.5), gridspec_kw={"width_ratios": [1, 1.08]})
    draw_map(ax, scenario, prior=False)
    draw_prior(axp, scenario)
    for rid, init in scenario.init.items():
        if init.node is not None:
            p = scenario.graphs[RobotType.UGV].positions[init.node]
            ax.plot(*p, "s", color=_robot_color(rid, RobotType.UGV), ms=8, mec="k", zorder=5)
    if show_target:
        ax.plot(*scenario.target_xy, "*", color=TARGET_COLOR, ms=18, mec="k", zorder=6, label="target")
    handles = _legend_handles() + [
        plt.Line2D([], [], color=ROAD_COLOR, label="road graph edge"),
        plt.Line2D([], [], ls="", marker="o", ms=5, mfc=NODE_FACE, mec=NODE_EDGE, mew=0.6, label="UGV graph node"),
    ]
    if show_target:
        handles.append(plt.Line2D([], [], marker="*", ls="", color=TARGET_COLOR, mec="k", ms=12, label="target"))
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)
    m = scenario.meta
    ax.set_title(f"{scenario.name}: {scenario.graphs[RobotType.UGV].n_nodes} road nodes, "
                 f"{m.get('road_length_m', 0) / 1e3:.0f} km roads, "
                 f"{scenario.graphs[RobotType.UAV].n_nodes} UAV nodes")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
        plt.close(fig)
    return fig


def plot_result(scenario: Scenario, result: MissionResult, path: Optional[str] = None):
    fig = plt.figure(figsize=(16, 8.5))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.15, 1])
    ax = fig.add_subplot(gs[:, 0])
    draw_map(ax, scenario, prior=False)
    docked = MissionResult.MODE_ORDER.index(RobotMode.DOCKED)
    for rid, traj in result.trajectories.items():
        rtype = scenario.robots[rid].robot_type
        col = _robot_color(rid, rtype)
        if rtype == RobotType.UAV:
            airborne = result.modes[rid] != docked
            xy = np.where(airborne[:, None], traj, np.nan)
            ax.plot(xy[:, 0], xy[:, 1], color=col, lw=0.6, alpha=0.8, zorder=4, label=rid)
        else:
            ax.plot(traj[:, 0], traj[:, 1], color=col, lw=2.2, alpha=0.9, zorder=4, label=rid)
        ax.plot(*traj[-1], "o" if rtype == RobotType.UAV else "s", color=col, mec="k", ms=7, zorder=6)
    ax.plot(*scenario.target_xy, "*", color=TARGET_COLOR, ms=18, mec="k", zorder=7)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_title(f"{result.controller_name}: {result.outcome} at t={result.time_s / 60:.1f} min"
                 + (f" by {result.found_by}" if result.found_by else ""))

    t = result.log["t"] / 60
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(t, result.log["searched_prob"], color="k", label="searched prior mass")
    for terr in Terrain:
        key = f"coverage_{terr.name.lower()}"
        if key in result.log:
            ax1.plot(t, result.log[key], color=TERRAIN_COLORS[terr], lw=2, label=f"{terr.name.lower()} cells covered")
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("fraction")
    ax1.legend(fontsize=7, loc="upper left")
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)
    for rid, spec in scenario.robots.items():
        ax2.plot(t, result.log[f"energy_{rid}"], color=_robot_color(rid, spec.robot_type),
                 ls="-" if spec.robot_type == RobotType.UAV else "--", label=rid)
    ax2.set_ylim(0, 1.02)
    ax2.set_ylabel("energy fraction")
    ax2.legend(fontsize=7, ncol=3)
    ax2.grid(alpha=0.3)

    sub = gs[2, 1].subgridspec(1, 2, wspace=0.05)
    axp, ax3 = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
    draw_prior(axp, scenario, cbar=False)
    _, vmin, vmax = _log_mass(scenario.prior, 3.0)
    ax3.imshow(_log10(result.final_mass), origin="lower", extent=scenario.grid.extent, cmap="magma",
               vmin=vmin, vmax=vmax, interpolation="nearest")
    ax3.plot(*scenario.target_xy, "*", color=TARGET_COLOR, ms=10, mec="k")
    ax3.set_title("unsearched prior at end (same scale)", fontsize=9)
    for a in (axp, ax3):
        a.set_xticks([])
        a.set_yticks([])
    axp.title.set_fontsize(9)
    ax2.set_xlabel("time [min]")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=110)
        plt.close(fig)
    return fig


def _energy_color(frac: float) -> str:
    return "#2e9e4f" if frac > 0.5 else "#e09b1a" if frac > 0.2 else "#c0392b"


def animate_result(scenario: Scenario, result: MissionResult, path: str, step_s: Optional[float] = None,
                   fps: int = 10) -> None:
    """GIF of robot motion over the remaining belief (needs belief snapshots).

    Side panel: robot legend with live energy bars and modes, plus a target/terrain key.
    The colour bar explains the belief heatmap. The target star turns from gold to cyan once a
    robot has seen it, and a label next to it names the last robot to see it (bordered in that
    robot's colour): the first UAV sighting, then the UGV's find. ``step_s`` (mission time per
    frame) defaults to about 150 frames per mission, clipped to 10-60 s.
    """
    from matplotlib.patches import Rectangle

    if not result.belief_snapshots:
        raise ValueError("run with SimConfig(belief_snapshot_interval_s=...) to record snapshots")
    snaps = result.belief_snapshots
    robots = list(scenario.robots.items())

    fig = plt.figure(figsize=(11.5, 7.5))
    gs = fig.add_gridspec(1, 4, width_ratios=[7, 0.22, 0.75, 3.2], wspace=0.05, left=0.06, right=0.98, top=0.93,
                          bottom=0.08)
    ax, cax, side = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[3])

    # Map, belief heatmap and colour bar.
    draw_map(ax, scenario, prior=False)
    _, vmin, vmax = _log_mass(snaps[0][1], 3.0)
    img = ax.imshow(_log10(snaps[0][1]), origin="lower", extent=scenario.grid.extent, cmap="magma",
                    vmin=vmin, vmax=vmax, alpha=0.55, zorder=1.5)
    cb = fig.colorbar(img, cax=cax, extend="min")
    cb.solids.set_alpha(0.55)
    cb.set_label(r"unsearched probability of target per cell ($\log_{10}$)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    star = ax.plot(*scenario.target_xy, "*", color=TARGET_COLOR, ms=16, mec="k", zorder=7)[0]
    # "Seen by" label: the first sighting, then the find (it ends the mission).
    seen_by = [e for e in (next((e for e in result.events if e.type == t), None)
                           for t in (EventType.TARGET_SIGHTED, EventType.TARGET_FOUND)) if e is not None]
    seen = None
    if seen_by:  # next to the star, on the side towards the map centre
        (x0, x1, y0, y1), (tx, ty) = scenario.grid.extent, scenario.target_xy
        right, up = tx < (x0 + x1) / 2, ty < (y0 + y1) / 2
        seen = ax.annotate("", scenario.target_xy,
                           xytext=(12 if right else -12, 12 if up else -12), textcoords="offset points",
                           ha="left" if right else "right", va="bottom" if up else "top", fontsize=9, zorder=8,
                           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=SIGHTED_COLOR, lw=1.5),
                           visible=False)

    markers = {}
    for rid, spec in robots:
        is_ugv = spec.robot_type == RobotType.UGV
        markers[rid] = ax.plot([], [], "s" if is_ugv else "o", color=_robot_color(rid, spec.robot_type), mec="k",
                               ms=9 if is_ugv else 7, zorder=6 if is_ugv else 6.5)[0]
    title = ax.set_title("", fontsize=12)

    # Side panel: robot legend + energy bars.
    side.set_xlim(0, 1)
    side.set_ylim(0, 1)
    side.axis("off")
    row_h = min(0.11, 0.5 / max(len(robots), 1))
    y = 0.97
    side.text(0.0, y, "Robots", fontsize=11, fontweight="bold", va="top")
    y -= 0.06
    bar_x, bar_w, bar_h = 0.14, 0.62, 0.022
    rows = {}
    for rid, spec in robots:
        is_ugv = spec.robot_type == RobotType.UGV
        side.plot(0.05, y - 0.012, "s" if is_ugv else "o", color=_robot_color(rid, spec.robot_type), mec="k",
                  ms=11 if is_ugv else 9, clip_on=False)
        side.text(bar_x, y, f"{rid}  ({spec.robot_type.value.upper()})", fontsize=9.5, va="center")
        mode_txt = side.text(1.0, y, "", fontsize=8, va="center", ha="right", color="#555")
        by = y - 0.035
        side.add_patch(Rectangle((bar_x, by), bar_w, bar_h, facecolor="#e6e6e6", edgecolor="#888", lw=0.6))
        fill = side.add_patch(Rectangle((bar_x, by), bar_w, bar_h, facecolor="#2e9e4f", lw=0))
        pct = side.text(bar_x + bar_w + 0.03, by + bar_h / 2, "", fontsize=9, va="center")
        rows[rid] = (fill, pct, mode_txt)
        y -= row_h

    y -= 0.02
    side.text(0.0, y, "Map", fontsize=11, fontweight="bold", va="top")
    y -= 0.04
    side.text(0.0, y, "terrain is drawn beneath the heatmap", fontsize=8, color="#555", va="top")
    y -= 0.05
    side.plot(0.05, y, "*", color=TARGET_COLOR, mec="k", ms=14, clip_on=False)
    side.text(bar_x, y, "target (ground truth)", fontsize=9, va="center")
    y -= 0.05
    side.plot(0.05, y, "*", color=SIGHTED_COLOR, mec="k", ms=14, clip_on=False)
    side.text(bar_x, y, "target, sighted", fontsize=9, va="center")
    y -= 0.05
    side.plot([0.01, 0.09], [y, y], color=ROAD_COLOR, lw=1.2)
    side.text(bar_x, y, "road (UGV graph edge)", fontsize=9, va="center")
    y -= 0.04
    side.plot(0.05, y, "o", ms=5, mfc=NODE_FACE, mec=NODE_EDGE, mew=0.6, clip_on=False)
    side.text(bar_x, y, "UGV graph node", fontsize=9, va="center")
    for terr in (Terrain.VEGETATION, Terrain.URBAN):
        y -= 0.04
        side.add_patch(Rectangle((0.01, y - 0.013), 0.08, 0.026, facecolor=TERRAIN_COLORS[terr], edgecolor="#888",
                                 lw=0.5))
        side.text(bar_x, y, terr.name.lower(), fontsize=9, va="center")

    t_end = float(result.traj_t[-1])
    if step_s is None:
        step_s = float(np.clip(t_end / 150, 10.0, 60.0))
    times = np.append(np.arange(0.0, t_end, step_s), t_end)
    snap_t = np.array([s[0] for s in snaps])
    log_t = result.log["t"]

    def update(k):
        tk = times[k]
        i = min(np.searchsorted(result.traj_t, tk), len(result.traj_t) - 1)
        j = max(np.searchsorted(snap_t, tk, side="right") - 1, 0)
        e = max(np.searchsorted(log_t, tk, side="right") - 1, 0)
        img.set_data(_log10(snaps[j][1]))
        for rid, mk in markers.items():
            mk.set_data([result.trajectories[rid][i, 0]], [result.trajectories[rid][i, 1]])
            frac = float(np.clip(result.log[f"energy_{rid}"][e], 0.0, 1.0))
            fill, pct, mode_txt = rows[rid]
            fill.set_width(bar_w * frac)
            fill.set_facecolor(_energy_color(frac))
            pct.set_text(f"{frac:.0%}")
            mode_txt.set_text(MissionResult.MODE_ORDER[result.modes[rid][i]].value)
        title.set_text(f"t = {_clock(tk)}")
        last = [e for e in seen_by if e.time <= tk]
        star.set_color(SIGHTED_COLOR if last else TARGET_COLOR)
        if seen is not None:
            seen.set_visible(bool(last))
            if last:
                rid = last[-1].robot_id
                seen.set_text(f"Seen by {rid} at {_clock(last[-1].time)}")
                seen.get_bbox_patch().set_edgecolor(_robot_color(rid, scenario.robots[rid].robot_type))

    def render(k):
        update(k)
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()

    key_colors = [TARGET_COLOR, SIGHTED_COLOR, "black", "white", "#e6e6e6", "#888888", "#555555", ROAD_COLOR,
                  NODE_FACE, NODE_EDGE,
                  *TERRAIN_COLORS.values(), *(_energy_color(f) for f in (1.0, 0.3, 0.0)),
                  *(_robot_color(rid, spec.robot_type) for rid, spec in robots)]
    n = len(times)
    palette = _gif_palette([render(k) for k in sorted({0, n // 4, n // 2, 3 * n // 4, n - 1})], key_colors)
    _save_gif((render(k) for k in range(n)), palette, path, fps)
    plt.close(fig)


def _clock(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


def _gif_palette(samples: list[np.ndarray], key_colors: list) -> tuple["Image.Image", list[tuple[int, int, int]]]:
    """256-colour palette: exact entries for legend colours, the rest fitted to sample frames.

    GIF frames are limited to 256 colours. With a per-frame adaptive palette, small
    markers (robots, target) lose their exact colour, so the legend stops matching
    the map. Reserving palette entries for those colours keeps them exact.
    """
    from matplotlib.colors import to_rgb
    from PIL import Image

    keys = list(dict.fromkeys(tuple(int(round(255 * c)) for c in to_rgb(col)) for col in key_colors))
    fitted = Image.fromarray(np.concatenate(samples, axis=0)).quantize(
        colors=256 - len(keys), method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    pal = fitted.getpalette()[: 3 * (256 - len(keys))]
    pal += [0] * (3 * (256 - len(keys)) - len(pal)) + [v for rgb in keys for v in rgb]
    img = Image.new("P", (1, 1))
    img.putpalette(pal)
    return img, keys


def _save_gif(frames, palette, path: str, fps: int) -> None:
    from PIL import Image

    pal_img, keys = palette
    first_key = 256 - len(keys)

    def to_p(frame: np.ndarray) -> "Image.Image":
        idx = np.asarray(Image.fromarray(frame).quantize(palette=pal_img, dither=Image.Dither.NONE)).copy()
        # Pillow's palette lookup is approximate; pin reserved colours to their exact entries.
        for j, rgb in enumerate(keys):
            idx[(frame == rgb).all(axis=-1)] = first_key + j
        out = Image.fromarray(idx, mode="P")
        out.putpalette(pal_img.getpalette())
        return out

    imgs = (to_p(f) for f in frames)
    first = next(imgs)
    first.save(path, save_all=True, append_images=imgs, duration=int(round(1000 / fps)), loop=0)
