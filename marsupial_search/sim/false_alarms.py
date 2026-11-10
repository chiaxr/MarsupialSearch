"""What a UAV flying high mistakes for the target, how far off its sightings are, and how they score.

Simulator-side: the robots never get the decoys' positions or likenesses. The simulator gives each
robot that may sight wrongly a :class:`~marsupial_search.core.SightingModel`
(``MissionSetup.sightings``) to weigh its sightings with, and reports them, true or false, in
``LocalBeliefUpdate.sightings``, each with its detector score (``LocalBeliefUpdate.scores``).

Only the ``"air"`` viewpoint (:func:`.visibility.viewpoint`) sights wrongly: a ground look (the
UGV, or a UAV at street level) is close enough to tell, and sights only the target, exactly. So
does a robot whose sighting ends the mission (``MissionSetup.confirmers``): it is the check.

:class:`FalseAlarms` holds, per cell (ground truth):

* ``clutter``: transient false sightings per m² per look (a shadow, a glint: those that score
  above the threshold), drawn afresh on every pass over a cell;
* ``decoys``: things that look like the target from the air (a parked car like the one sought, a
  rock like a hiker), placed with density ``decoy_density``, each with its own mean score
  (``decoy_scores``: its likeness to the target). A pass over one that has it in view
  (occlusion, as for the target) scores it afresh and sights it if the score clears the
  threshold, once per pass like the target: the next pass over it may well sight it again;
* every air sighting, true or false, is off by Gaussian error ``sigma_m`` per axis, and scores
  as :class:`~marsupial_search.core.SightingScores` has it;
* nothing is certain from the air: an air look in view of the target sights it with its recall
  (``scores.recall(scores.target)``, 0.9 by default).

A robot's model of it (:func:`sighting_model`) has the position error, the score distributions,
and per cell the false sightings it should expect per look: the clutter, plus the decoy density
times the chance a look sights one (it does not know where they are), and which part of them are
decoys. By default the robots assume the truth; :class:`SightingBelief`
(``SimConfig.sighting_belief``) makes them assume other rates, ignore the scores, or weigh no
sighting at all (``cue``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.special import ndtr, ndtri

from marsupial_search.core import RobotSpec, SightingModel, SightingScores

from .visibility import AIR, viewpoint


@dataclass(frozen=True)
class FalseAlarms:
    """What an air look mistakes for the target (ground truth)."""

    clutter: np.ndarray  # (H, W) transient false sightings per m² per look
    decoy_density: np.ndarray  # (H, W) per m²: the density the decoys were placed with
    decoys: np.ndarray  # (n, 2) decoy positions
    decoy_layers: np.ndarray  # (n,) the layer each is in
    decoy_scores: np.ndarray  # (n,) each one's mean score (its likeness to the target)
    sigma_m: float = 15.0  # sighting position error, std per axis
    scores: SightingScores = SightingScores()


@dataclass(frozen=True)
class SightingBelief:
    """What the robots that sight wrongly assume of it (the true model by default)."""

    mode: str = "weigh"  # weigh: by their SightingModel | cue: a sighting is only a place to check from close up
    use_scores: bool = True  # weigh: each sighting's score too (else only that it was made)
    false_alarm_scale: float = 1.0  # weigh: they expect this times the true false sightings (clutter and decoys alike)


def sights_wrongly(spec: RobotSpec, false_alarms: Optional[FalseAlarms], confirmers=()) -> bool:
    """Does robot ``spec`` sight decoys and clutter (a high UAV that does not end the mission)?"""
    return false_alarms is not None and viewpoint(spec) == AIR and spec.robot_id not in confirmers


def air_detection(p: np.ndarray, false_alarms: FalseAlarms) -> np.ndarray:
    """A detection map (N_LAYERS, H, W) of a robot that sights wrongly: the chance the target is in
    view (``p``) times the detector's recall on it."""
    sc = false_alarms.scores
    return p * float(sc.recall(sc.target))


def draw_score(rng: np.random.Generator, mu: float, sd: float, threshold: float) -> float:
    """A score from ``N(mu, sd²)`` given it clears ``threshold`` (inverse CDF)."""
    lo = float(ndtr((threshold - mu) / sd))
    u = lo + rng.random() * (1.0 - lo)
    return max(mu + sd * float(ndtri(min(u, 1.0 - 1e-16))), threshold)


def sighting_model(detection: np.ndarray, layer_fractions: np.ndarray, false_alarms: FalseAlarms,
                   belief: SightingBelief = SightingBelief(), checkable: Optional[np.ndarray] = None) -> SightingModel:
    """The model a robot that sights wrongly is given: its position error, the score distributions,
    and per cell the false sightings to expect per look (clutter, plus decoys in view as the target
    would be, over its layers as the target's mass is split, times a decoy's chance to clear the
    threshold), as ``belief`` has it; with cues, where the robots that check from close up can
    detect the target (``checkable``, from their detection maps)."""
    if belief.mode not in ("weigh", "cue"):
        raise ValueError(f"sighting_belief.mode must be 'weigh' or 'cue', got {belief.mode!r}")
    fa, k = false_alarms, belief.false_alarm_scale
    sc = fa.scores
    lf = layer_fractions / np.maximum(layer_fractions.sum(0, keepdims=True), 1e-300)
    in_view = (lf * detection).sum(0) / float(sc.recall(sc.target))
    decoys = k * fa.decoy_density * in_view * sc.decoy_recall()
    return SightingModel(false_alarms=k * fa.clutter + decoys, sigma_m=fa.sigma_m, decoys=decoys, scores=sc,
                         use_scores=belief.use_scores, cue=belief.mode == "cue",
                         checkable=checkable if belief.mode == "cue" else None)
