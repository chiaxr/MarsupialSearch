"""Base scenarios: named starting points for ``ScenarioConfig`` (YAML ``scenario.base``).

A base sets the map size and the generator parameters that go with it (towns, road density,
prior); any ``ScenarioConfig`` field can then be overridden on top. Both bases use the one
fleet (``FleetConfig``): one UGV, ``uav0`` docked on it and ``uav1`` hovering over it.

* ``large`` – 5 km x 5 km, 3 towns.
* ``small`` – 1.5 km x 1.5 km, 1 town, a denser road network for its size. The UGV battery
  is sized to run out mid-mission, so depletion is a realistic outcome.
"""

from __future__ import annotations

from typing import Callable

from marsupial_search.core import EnergySpec

from .generator import FleetConfig, ScenarioConfig


def large(seed: int = 0) -> ScenarioConfig:
    return ScenarioConfig(name="large", seed=seed)


def small(seed: int = 0) -> ScenarioConfig:
    return ScenarioConfig(
        name="small",
        size_m=(1500.0, 1500.0),
        n_urban=1,
        urban_half_size_m=(200.0, 300.0),
        n_rural_nodes=15,
        rural_min_spacing_m=200.0,
        roadless_smoothing_m=150.0,
        n_prior_modes=3,
        prior_sigma_m=(150.0, 400.0),
        fleet=FleetConfig(
            # About 40 min of driving.
            ugv_energy=EnergySpec(capacity=17_000, idle_power=2.0, energy_per_m=1.0),
        ),
        seed=seed,
    )


SCENARIO_BASES: dict[str, Callable[[int], ScenarioConfig]] = {"large": large, "small": small}
