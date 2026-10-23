"""Decentralised control: one :class:`RobotController` per robot.

On hardware, each robot runs its own ``RobotController``, fed by a local adapter that
builds a :class:`RobotObservation` from its own sensors and radio. In simulation,
:class:`DecentralizedTeam` does the same for every robot from the global
:class:`Observation`, so a decentralised algorithm plugs into the simulator like any
centralised one.

Isolation: each robot gets its own controller instance and, by default, its own deep
copy of the :class:`MissionSetup`. Robots learn about each other only through
``RobotObservation.inbox``. Python cannot stop a controller from reading
module-level globals, so algorithms must not keep shared state there.
"""

from __future__ import annotations

import copy
import time
from abc import ABC, abstractmethod
from typing import Callable, Optional

from .controller import MissionController
from .messages import MissionSetup, Observation, RobotCommands, RobotObservation, RobotSpec
from .types import RobotMode


class RobotController(ABC):
    """Controller running on a single robot."""

    name: str = "robot_controller"

    @abstractmethod
    def reset(self, robot_id: str, setup: MissionSetup) -> None: ...

    @abstractmethod
    def step(self, obs: RobotObservation) -> Optional[RobotCommands]:
        """Commands for this robot (or None to keep the current motion command).

        Return a list to combine a motion command with ``Transmit`` messages.
        """

    def is_done(self) -> bool:
        """True once this robot has nothing left to do."""
        return False


RobotControllerFactory = Callable[[str, RobotSpec], RobotController]


class DecentralizedTeam(MissionController):
    """Runs one isolated :class:`RobotController` per robot behind the team interface.

    Depleted robots are treated as powered off: their controllers are no longer stepped.
    Each robot's ``step`` calls are timed (``robot_timing``), the compute it would need onboard.
    """

    def __init__(self, factory: RobotControllerFactory, name: str = "decentralized", isolate_setup: bool = True):
        self.factory = factory
        self.name = name
        self.isolate_setup = isolate_setup
        self.robots: dict[str, RobotController] = {}

    def reset(self, setup: MissionSetup) -> None:
        self.robots = {}
        for rid, spec in setup.robots.items():
            ctl = self.factory(rid, spec)
            ctl.reset(rid, copy.deepcopy(setup) if self.isolate_setup else setup)
            self.robots[rid] = ctl
        self._alive = list(self.robots)  # ordered: keeps runs reproducible
        self._timing = {rid: {"controller_s": 0.0, "controller_steps": 0, "controller_min_step_s": float("inf"),
                              "controller_max_step_s": 0.0} for rid in self.robots}

    def robot_timing(self) -> dict[str, dict]:
        """Per robot: total time in ``step``, number of steps, and the fastest and slowest step."""
        return {rid: dict(t) for rid, t in self._timing.items()}

    def step(self, obs: Observation) -> dict[str, RobotCommands]:
        out: dict[str, RobotCommands] = {}
        self._alive = [rid for rid in self.robots if obs.robots[rid].mode != RobotMode.DEPLETED]
        for rid in self._alive:
            view = obs.local_view(rid)
            t0 = time.perf_counter()
            cmds = self.robots[rid].step(view)
            dt = time.perf_counter() - t0
            t = self._timing[rid]
            t["controller_s"] += dt
            t["controller_steps"] += 1
            t["controller_min_step_s"] = min(t["controller_min_step_s"], dt)
            t["controller_max_step_s"] = max(t["controller_max_step_s"], dt)
            if cmds is not None and cmds != []:
                out[rid] = cmds
        return out

    def is_done(self) -> bool:
        return bool(self._alive) and all(self.robots[rid].is_done() for rid in self._alive)
