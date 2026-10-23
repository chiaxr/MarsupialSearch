"""Controller interface. Implementations must not import :mod:`marsupial_search.sim`."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .messages import MissionSetup, Observation, RobotCommands


class MissionController(ABC):
    """Team-level controller: one observation in, commands per robot out.

    Each robot gets one command or a list of them (e.g. a motion command plus
    :class:`~marsupial_search.core.messages.Transmit` radio messages). Robots omitted from the
    returned dict keep executing their previous motion command. Received radio
    messages arrive in ``obs.radio[robot_id]``. The same object can be driven by the
    simulator or by a deployment adapter that builds :class:`Observation` messages
    from live robot data.
    """

    name: str = "controller"

    @abstractmethod
    def reset(self, setup: MissionSetup) -> None: ...

    @abstractmethod
    def step(self, obs: Observation) -> dict[str, RobotCommands]: ...

    def is_done(self) -> bool:
        """True once the controller has nothing left to do (e.g. full coverage)."""
        return False
