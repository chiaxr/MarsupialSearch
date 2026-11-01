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
    """An update with uncertain sightings, or a look near one that sighted nothing there (a *miss*),
    kept so the belief can weigh them again (and pass them on)."""

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
    lowers it again. A sighting near an earlier one ``a`` (within ``gate_m``: its *anchor*) is a
    *repeat*. Clutter does not come back, but a decoy does: if the anchor was one, it is sighted
    again with the chance ``decoy_p · p(a)`` (``p(a)`` the look's chance of sighting the target
    there: the most it gives a cell near ``a``), at ``z`` off ``a`` by twice the position error's
    variance (``g₂``). So a repeat counts as a sighting with false sightings at
    ``λ(z) + π · decoy_p · p(a) · g₂(z - a)``, where ``π`` is the chance the anchor was a decoy
    were it not the target: the decoys' share of ``λ`` there (:attr:`SightingModel.decoys`) at
    first, higher after each repeat and lower after each miss. A repeat thus raises the belief
    where clutter is the likely false sighting, and little where decoys are (by about
    ``1 / decoy_p``).

    A look from such a robot that sights nothing near an anchor (a *miss*) lowers it by ``1 - p``
    as any look, but a decoy there would have gone unseen too, with the chance ``1 - x``,
    ``x = π · decoy_p · p̄(a)`` (``p̄(a)``: the look's chance of sighting it, averaged over where
    it may be round ``a``; a pass comes in slivers, the cells newly in view). So the anchor's own
    ratio is divided by ``1 - x``: a miss lowers it less where decoys are likely, and makes them
    less likely (``π`` falls), so later repeats count more. A look that cannot sight wrongly (from
    the ground) lowers it by ``1 - p`` alone.

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
        self.decoy_chance: list[float] = []  # per anchor: π, the chance it was a decoy were it not the target
        self._missed: dict[tuple, float] = {}  # (event key, anchor): its _miss_chance, which never changes
        lf = self.initial_mass.sum(0, keepdims=True)
        self._layer_share = np.divide(self.initial_mass, lf, out=np.zeros_like(self.initial_mass), where=lf > 0)

    def apply(self, update: LocalBeliefUpdate) -> None:
        if update.detected:
            self.detection_xy = update.detection_xy
            return
        ev = None
        if update.robot_id in self.models and (update.time, update.robot_id) not in self.events \
                and (update.sightings or self._misses(update)):
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
        """``evidence`` from every sighting event, in time order (a repeat or a miss depends on what
        came before; a miss also on the anchor's ratio, made before it, so the ratios come last)."""
        if self.evidence is None:
            self.evidence = np.ones_like(self.looked)
            self.mass = self.looked * self.evidence  # its own array from now on
        else:
            self.evidence.fill(1.0)
        res, gate = self.grid.resolution, self.gate_m
        self.anchors, self.decoy_chance = [], []
        missed: list[float] = []  # per anchor: ∏ 1 / (1 - x) over its misses
        terms = []  # per event: (event, s2, [(z, λ, anchor index or -1)])
        for key in sorted(self.events):
            ev = self.events[key]
            model = self.models[ev.robot_id]
            _, h, w = ev.p_detect.shape
            if not (h and w):
                continue
            s2 = self._s2(model)
            seen, zs = set(), []
            for z in ev.sightings:
                d = [np.hypot(z[0] - a[0], z[1] - a[1]) for a in self.anchors]
                i = int(np.argmin(d)) if d else -1
                if d and d[i] <= gate:
                    seen.add(i)
            for i, a in enumerate(self.anchors):  # the anchors this look sighted nothing near
                if i not in seen:
                    pi, dp = self.decoy_chance[i], min(model.decoy_p * self._miss_chance(ev, a), 1.0 - 1e-9)
                    if pi * dp > 0.0:  # a decoy there would have gone unseen with the chance 1 - x
                        self.decoy_chance[i] = pi * (1.0 - dp) / (1.0 - pi * dp)
                        missed[i] /= 1.0 - pi * dp
            for z in ev.sightings:
                r, c = self.grid.world_to_cell(*z)
                lam = max(float(model.false_alarms[r, c]), 1e-12)
                d = [np.hypot(z[0] - a[0], z[1] - a[1]) for a in self.anchors]
                i = int(np.argmin(d)) if d else -1
                if d and d[i] <= gate:  # a repeat: of a decoy, or the target
                    a, pi = self.anchors[i], self.decoy_chance[i]
                    again = model.decoy_p * self._sight_chance(ev, a) * np.exp(-d[i] ** 2 / (4.0 * s2)) / (4.0 * np.pi * s2)
                    self.decoy_chance[i] = pi * (lam + again) / (lam + pi * again)
                    zs.append((z, lam + pi * again, -1))
                else:
                    self.anchors.append(z)
                    share = 1.0 if model.decoys is None else float(model.decoys[r, c]) / lam
                    self.decoy_chance.append(min(max(share, 0.0), 1.0))
                    missed.append(1.0)
                    zs.append((z, lam, len(self.anchors) - 1))
            if zs:
                terms.append((ev, s2, zs))
        for ev, s2, zs in terms:
            _, h, w = ev.p_detect.shape
            x0, y0 = self.grid.cell_center(ev.row0, ev.col0)
            X, Y = np.meshgrid(x0 + res * np.arange(w), y0 + res * np.arange(h))
            G = np.zeros((h, w))
            for z, lam, i in zs:
                k = missed[i] if i >= 0 else 1.0
                G += k * np.exp(-((X - z[0]) ** 2 + (Y - z[1]) ** 2) / (2.0 * s2)) / (2.0 * np.pi * s2 * lam)
            q = 1.0 - np.minimum(ev.p_detect, 1.0 - 1e-9)
            f = 1.0 + (1.0 - q) * G / q  # times the look's 1 - p: the ratio above
            self.evidence[:, ev.row0:ev.row0 + h, ev.col0:ev.col0 + w] *= f
        np.multiply(self.looked, self.evidence, out=self.mass)

    def _s2(self, model: SightingModel) -> float:
        """The position error's variance (an exact sighting is still anywhere in its cell)."""
        return max(model.sigma_m, self.grid.resolution / np.sqrt(12.0)) ** 2

    def _misses(self, update: LocalBeliefUpdate) -> bool:
        """Whether a look with no sightings could have sighted one near an anchor (so it is kept)."""
        return any(self._miss_chance(update, a) > 0.0 for a in self.anchors)

    def _miss_chance(self, ev, xy) -> float:
        """The chance pass ``ev`` (a :class:`SightingEvent`, or the update it comes from) sights a
        decoy at ``xy``, averaged over where it may be (the position error round ``xy``, within
        ``gate_m``): a pass that comes in slivers adds up to one pass."""
        key = (ev.time, ev.robot_id, xy)
        if key in self._missed:
            return self._missed[key]
        _, h, w = ev.p_detect.shape
        r0, c0, mask = self.grid.disk_window(xy[0], xy[1], self.gate_m)
        mh, mw = mask.shape
        if not (h and w and r0 < ev.row0 + h and ev.row0 < r0 + mh and c0 < ev.col0 + w and ev.col0 < c0 + mw):
            return 0.0  # nowhere near: not kept, nor asked again
        rr, cc = np.nonzero(mask)
        rr, cc = rr + r0, cc + c0
        keep = (rr >= ev.row0) & (rr < ev.row0 + h) & (cc >= ev.col0) & (cc < ev.col0 + w)
        out = 0.0
        if keep.any():
            x, y = self.grid.cell_center(rr, cc)
            wt = np.exp(-((x - xy[0]) ** 2 + (y - xy[1]) ** 2) / (2.0 * self._s2(self.models[ev.robot_id])))
            rk, ck = rr[keep], cc[keep]
            p = (self._layer_share[:, rk, ck] * ev.p_detect[:, rk - ev.row0, ck - ev.col0]).sum(0)
            out = float((wt[keep] * p).sum() / wt.sum())
        self._missed[key] = out
        return out

    def _sight_chance(self, ev: SightingEvent, xy) -> float:
        """The chance pass ``ev`` sights the target near ``xy``: the most its look gives a cell within
        ``gate_m`` of it (over its layers, as the target's mass is split). A look covers only the
        cells that came into view, so not always the one at ``xy`` itself."""
        _, h, w = ev.p_detect.shape
        r0, c0, mask = self.grid.disk_window(xy[0], xy[1], self.gate_m)
        rr, cc = np.nonzero(mask)
        rr, cc = rr + r0, cc + c0
        keep = (rr >= ev.row0) & (rr < ev.row0 + h) & (cc >= ev.col0) & (cc < ev.col0 + w)
        rr, cc = rr[keep], cc[keep]
        if not len(rr):
            return 0.0
        return float((self._layer_share[:, rr, cc] * ev.p_detect[:, rr - ev.row0, cc - ev.col0]).sum(0).max())

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
