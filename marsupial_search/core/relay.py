"""Store-and-forward relaying of each robot's latest state over a radio without multi-hop routing.

Every robot keeps, per robot (its *origin*), the latest record it knows of that robot's state and
its version: the time the origin sent it. Each broadcast carries the sender's *clock* (the
version it holds per origin), and the records its neighbours lack: a record goes out when a
neighbour heard within ``neighbour_s`` holds an older version of it, by more than ``period_s``
(so a record that changes every step is not re-sent every step). Information then reaches a
robot out of range of its origin through any robot in between, at once or later (a robot carries
what it knows until it meets the next one).

Records are merged per key, so a key an origin sends only now and then (e.g. a plan) stays in its
record. Only state that may be applied more than once belongs in a record.
"""

from __future__ import annotations

from typing import Any


class Relay:
    def __init__(self, me: str, period_s: float = 5.0, neighbour_s: float = 5.0):
        self.me, self.period_s, self.neighbour_s = me, period_s, neighbour_s
        self.records: dict[str, tuple[float, dict[str, Any]]] = {}  # origin -> (version, record)
        self.neighbours: dict[str, tuple[dict[str, float], float]] = {}  # robot -> (its clock, when heard)

    def update(self, origin: str, version: float, record: dict) -> bool:
        """Merge a record of ``origin``'s state if it is newer than the one held; True if it was."""
        held = self.records.get(origin)
        if held is not None and held[0] >= version:
            return False
        self.records[origin] = (version, {**(held[1] if held else {}), **record})
        return True

    def clock(self) -> dict[str, float]:
        return {origin: v for origin, (v, _) in self.records.items()}

    def heard(self, robot: str, clock: dict[str, float], now: float) -> None:
        self.neighbours[robot] = (clock, now)

    def outgoing(self, now: float) -> dict[str, tuple[float, dict]]:
        """The records some neighbour heard lately lacks (not their own, nor this robot's: it sends that itself)."""
        out = {}
        for origin, (v, rec) in self.records.items():
            if origin == self.me:
                continue
            if any(n != origin and now - t <= self.neighbour_s and clock.get(origin, -float("inf")) < v - self.period_s
                   for n, (clock, t) in self.neighbours.items()):
                out[origin] = (v, rec)
        return out
