"""What a UAV flying high mistakes for the target, and how far off its sightings are.

Simulator-side: the robots never get the decoys' positions. The simulator gives each robot that
may sight wrongly a :class:`~marsupial_search.core.SightingModel` (``MissionSetup.sightings``) to
weigh its sightings with, and reports them, true or false, in ``LocalBeliefUpdate.sightings``.

Only the ``"air"`` viewpoint (:func:`.visibility.viewpoint`) sights wrongly: a ground look (the
UGV, or a UAV at street level) is close enough to tell, and sights only the target, exactly. So
does a robot whose sighting ends the mission (``MissionSetup.confirmers``): it is the check.

:class:`FalseAlarms` holds, per cell (ground truth):

* ``clutter``: transient false sightings per m² per look (a shadow, a glint), drawn afresh on every
  pass over a cell;
* ``decoys``: things that look like the target from the air (a parked car like the one sought, a
  rock like a hiker), placed with density ``decoy_density``. A pass over one sights it with the
  chance ``decoy_p`` times what the look would detect the target with there, rolled once per
  pass like the target: the next pass over it may well sight it again;
* every air sighting, true or false, is off by Gaussian error ``sigma_m`` per axis;
* nothing is certain from the air: an air look detects the target with at most ``max_p``.

A robot's model of it (:func:`sighting_model`) has the position error, and per cell the false
sightings it should expect per look: the clutter, plus the decoy density times the chance a
look sights one (it does not know where they are), and which part of them are decoys.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from marsupial_search.core import RobotSpec, SightingModel

from .visibility import AIR, viewpoint


@dataclass(frozen=True)
class FalseAlarms:
    """What an air look mistakes for the target (ground truth)."""

    clutter: np.ndarray  # (H, W) transient false sightings per m² per look
    decoy_density: np.ndarray  # (H, W) per m²: the density the decoys were placed with
    decoys: np.ndarray  # (n, 2) decoy positions
    decoy_layers: np.ndarray  # (n,) the layer each is in
    decoy_p: float = 1.0  # a look sights a decoy with this times its chance of detecting the target there
    sigma_m: float = 15.0  # sighting position error, std per axis
    max_p: float = 0.9  # an air look detects the target with at most this


def sights_wrongly(spec: RobotSpec, false_alarms: Optional[FalseAlarms], confirmers=()) -> bool:
    """Does robot ``spec`` sight decoys and clutter (a high UAV that does not end the mission)?"""
    return false_alarms is not None and viewpoint(spec) == AIR and spec.robot_id not in confirmers


def capped_detection(p: np.ndarray, false_alarms: FalseAlarms) -> np.ndarray:
    """A detection map (N_LAYERS, H, W) of a robot that sights wrongly: at most ``max_p``."""
    return np.minimum(p, false_alarms.max_p)


def sighting_model(detection: np.ndarray, layer_fractions: np.ndarray, false_alarms: FalseAlarms) -> SightingModel:
    """The model a robot that sights wrongly is given: its position error, and per cell the false
    sightings to expect per look (clutter, plus decoys sighted as the target would be, over its
    layers as the target's mass is split)."""
    lf = layer_fractions / np.maximum(layer_fractions.sum(0, keepdims=True), 1e-300)
    sighted = (lf * detection).sum(0)
    fa = false_alarms
    decoys = fa.decoy_density * fa.decoy_p * sighted
    return SightingModel(false_alarms=fa.clutter + decoys, sigma_m=fa.sigma_m, decoys=decoys, decoy_p=fa.decoy_p)
