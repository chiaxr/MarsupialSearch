"""Layered Bayesian belief over the target location (stationary target)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .grid import GridSpec
from .messages import LocalBeliefUpdate, SightingModel

GATE_SIGMAS = 3.0  # a sighting within this many position errors (plus a cell) of another is the same thing


@dataclass(frozen=True)
class SightingEvent:
    """An update with uncertain sightings, kept so the belief can weigh them again (and pass them on)."""

    robot_id: str
    time: float
    row0: int
    col0: int
    p_detect: np.ndarray  # (N_LAYERS, h, w): the update's looks
    sightings: tuple[tuple[float, float], ...]

    @property
    def key(self) -> tuple[float, str]:
        return (self.time, self.robot_id)


class BeliefMap:
    """Unnormalised posterior mass ``P(target in layer l of cell, not yet detected)``.

    ``mass`` has shape ``(N_LAYERS, H, W)`` and starts as ``prior * layer_fractions``.
    Each negative observation multiplies the observed window by ``1 - p_detect``; the
    remaining total mass is ``1 - P(detected so far)``.

    Uncertain sightings (``LocalBeliefUpdate.sightings``, from a robot with a
    :class:`SightingModel` in ``sightings``): a look ``p`` over cells ``c`` that reports sightings
    ``z`` multiplies each cell by the likelihood ratio of what it saw, target there or not::

        (1 - p(c)) + p(c) · Σ_z g(z | c) / λ(z)

    with ``g`` the position error's Gaussian density and ``λ`` the false sightings per m² per look.
    With no sightings this is the plain look. A pass that sights nothing near a sighting later
    lowers it again. A sighting near an earlier one (within ``gate_m``) is a *repeat*: it says
    nothing new (a thing mistaken for the target once is mistaken again), so the cells within
    ``gate_m`` of the one it repeats get no look at all: a pass that sights it again neither
    raises nor lowers it, and only a pass that misses it (or a look that cannot sight wrongly,
    from the ground) clears it.

    The belief is kept in two parts, so two robots' beliefs can be fused: ``looked`` (every look as
    if it sighted nothing, which only ever goes down: fused by element-wise minimum) and the
    sighting events (fused by union), whose product ``evidence`` it is multiplied by. With no
    uncertain sighting, ``mass`` is ``looked`` itself.
    """

    def __init__(self, grid: GridSpec, prior: np.ndarray, layer_fractions: np.ndarray,
                 sightings: Optional[dict[str, SightingModel]] = None):
        self.grid = grid
        self.initial_mass = prior[None, :, :] * layer_fractions
        self.mass = self.initial_mass.copy()
        self.initial_total = float(self.initial_mass.sum())
        self.detection_xy: Optional[tuple[float, float]] = None
        self.models = dict(sightings or {})
        sigma = max((m.sigma_m for m in self.models.values()), default=0.0)
        self.gate_m = GATE_SIGMAS * sigma + grid.resolution
        self.looked = self.mass
        self.evidence: Optional[np.ndarray] = None  # None: all ones
        self.events: dict[tuple[float, str], SightingEvent] = {}
        self.anchors: list[tuple[float, float]] = []  # the uncertain sightings that are not repeats, in time order

    def apply(self, update: LocalBeliefUpdate) -> None:
        if update.detected:
            self.detection_xy = update.detection_xy
            return
        ev = None
        if update.sightings and update.robot_id in self.models and (update.time, update.robot_id) not in self.events:
            ev = SightingEvent(update.robot_id, update.time, update.row0, update.col0, update.p_detect,
                               tuple(tuple(map(float, z)) for z in update.sightings))
        _, h, w = update.p_detect.shape
        if h and w:
            win = (slice(None), slice(update.row0, update.row0 + h), slice(update.col0, update.col0 + w))
            self.looked[win] *= 1.0 - update.p_detect
            if self.evidence is not None and ev is None:
                self.mass[win] = self.looked[win] * self.evidence[win]
        if ev is not None:
            self.events[ev.key] = ev
            self._weigh()

    def _weigh(self) -> None:
        """``evidence`` from every sighting event, in time order (a repeat depends on what came before)."""
        if self.evidence is None:
            self.evidence = np.ones_like(self.looked)
            self.mass = self.looked * self.evidence  # its own array from now on
        else:
            self.evidence.fill(1.0)
        res, gate = self.grid.resolution, self.gate_m
        self.anchors = []
        for key in sorted(self.events):
            ev = self.events[key]
            model = self.models[ev.robot_id]
            _, h, w = ev.p_detect.shape
            if not (h and w):
                continue
            x0, y0 = self.grid.cell_center(ev.row0, ev.col0)
            X, Y = np.meshgrid(x0 + res * np.arange(w), y0 + res * np.arange(h))
            s2 = max(model.sigma_m, res / np.sqrt(12.0)) ** 2  # an exact sighting is still anywhere in its cell
            G, repeat = np.zeros((h, w)), np.zeros((h, w), dtype=bool)
            for z in ev.sightings:
                near = [a for a in self.anchors if np.hypot(z[0] - a[0], z[1] - a[1]) <= gate]
                for a in near:  # the cells the sighting it repeats is about
                    repeat |= (X - a[0]) ** 2 + (Y - a[1]) ** 2 <= gate ** 2
                if near:
                    continue
                d2 = (X - z[0]) ** 2 + (Y - z[1]) ** 2
                self.anchors.append(z)
                r, c = self.grid.world_to_cell(*z)
                lam = max(float(model.false_alarms[r, c]), 1e-12)
                G += np.exp(-d2 / (2.0 * s2)) / (2.0 * np.pi * s2 * lam)
            q = 1.0 - np.minimum(ev.p_detect, 1.0 - 1e-9)
            f = np.where(repeat, 1.0 / q, 1.0 + (1.0 - q) * G / q)  # times the look's 1 - p: the ratio above
            self.evidence[:, ev.row0:ev.row0 + h, ev.col0:ev.col0 + w] *= f
        np.multiply(self.looked, self.evidence, out=self.mass)

    # ------------------------------------------------------------------ sightings
    def p_near(self, xy, total: Optional[float] = None) -> float:
        """P(target within ``gate_m`` of ``xy``)."""
        total = float(self.mass.sum()) if total is None else total
        r0, c0, mask = self.grid.disk_window(xy[0], xy[1], self.gate_m)
        h, w = mask.shape
        if total <= 0 or not mask.any():
            return 0.0
        return float(self.mass[:, r0:r0 + h, c0:c0 + w].sum(0)[mask].sum()) / total

    def hypotheses(self) -> list[tuple[tuple[float, float], float]]:
        """Each uncertain sighting that is not a repeat, with the chance the target is near it
        (``p_near``): most likely first."""
        total = float(self.mass.sum())
        return sorted(((z, self.p_near(z, total)) for z in self.anchors), key=lambda t: -t[1])

    # ------------------------------------------------------------------- fusion
    def sync_payload(self) -> dict:
        """What :meth:`fuse` takes: ``looked``, the certain sighting and the sighting events."""
        out = {"mass": self.looked.astype(np.float32), "sighting": self.detection_xy}
        if self.events:
            out["events"] = list(self.events.values())
        return out

    def fuse(self, payload: dict) -> None:
        """Fuse another robot's :meth:`sync_payload` into this belief."""
        np.minimum(self.looked, payload["mass"], out=self.looked)
        if payload["sighting"] is not None and self.detection_xy is None:
            self.detection_xy = tuple(payload["sighting"])
        new = [e for e in payload.get("events", ()) if e.key not in self.events and e.robot_id in self.models]
        for e in new:
            self.events[e.key] = e
        if new:
            self._weigh()
        elif self.evidence is not None:
            np.multiply(self.looked, self.evidence, out=self.mass)

    # -------------------------------------------------------------------- views
    def cell_mass(self, layers=None) -> np.ndarray:
        """Remaining (unnormalised) mass per cell, optionally restricted to some layers."""
        if layers is None:
            return self.mass.sum(axis=0)
        return self.mass[list(layers)].sum(axis=0)

    def posterior(self) -> np.ndarray:
        """Normalised P(target in cell | all observations so far)."""
        m = self.cell_mass()
        total = m.sum()
        return m / total if total > 0 else m

    def searched_probability(self) -> float:
        """Prior probability mass already searched (cumulative detection probability; sightings aside)."""
        return 1.0 - float(self.looked.sum()) / self.initial_total

    def covered_cells(self, tol: float = 1e-6) -> np.ndarray:
        """Boolean (H, W): cells whose remaining mass (sightings aside) is below ``tol`` of their initial mass."""
        init = self.initial_mass.sum(axis=0)
        return self.looked.sum(axis=0) <= tol * np.maximum(init, 1e-300)
