"""Simulated radio channel between robots.

Link model: a message from ``s`` reaches ``r`` if both robots are powered (not
DEPLETED) and their distance is within ``min(range[s.type], range[r.type])`` when it
is sent. It then survives random loss and arrives ``latency_s`` later, at the
earliest in the next observation. Each sender also has an optional
bytes-per-second budget; anything over the budget in a step is dropped. There is
no multi-hop routing: algorithms that want relaying must re-transmit themselves.

With ``simulate_channel=False`` the channel is ideal: range, loss, latency, bandwidth
and payload limits are ignored, and every message reaches every powered recipient
in the next observation. Payloads are still serialised, so recipients get
independent copies and sizes are still counted.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

from marsupial_search.core import PayloadCodec, PickleCodec, RadioMessage, RobotType, Transmit


@dataclass
class RadioConfig:
    enabled: bool = True  # False: no radio at all (Transmit commands are ignored)
    simulate_channel: bool = True  # False: ideal channel, every message goes through
    range_m: dict[RobotType, float] = field(
        default_factory=lambda: {RobotType.UGV: 2000.0, RobotType.UAV: 2000.0})
    latency_s: float = 0.0  # extra delay on top of the one-step minimum
    drop_prob: float = 0.0  # independent loss per (message, recipient)
    max_bytes_per_s: Optional[float] = None  # per-sender transmit budget
    max_payload_bytes: Optional[int] = None  # larger payloads are rejected
    codec: PayloadCodec = field(default_factory=PickleCodec)


_STAT_KEYS = ("radio_sent", "radio_bytes_sent", "radio_received", "radio_dropped_range", "radio_dropped_loss",
              "radio_dropped_bandwidth", "radio_dropped_offline", "radio_rejected")
_SIZE_KEYS = ("radio_payload_min_bytes", "radio_payload_max_bytes")  # per sender, over the messages it sent


class RadioChannel:
    def __init__(self, config: RadioConfig, rng: np.random.Generator):
        self.cfg = config
        self.rng = rng
        self._pending: list[tuple[float, int, str, RadioMessage]] = []  # (deliver_time, order, recipient, msg)
        self._order = itertools.count()
        self._seq: dict[str, int] = {}
        self.stats: dict[str, dict[str, int]] = {}

    def _stat(self, rid: str) -> dict[str, int]:
        if rid not in self.stats:
            self.stats[rid] = {**dict.fromkeys(_STAT_KEYS, 0), **dict.fromkeys(_SIZE_KEYS, float("nan"))}
        return self.stats[rid]

    def transmit(self, outgoing: list[tuple[str, Transmit]], sent_time: float, dt: float,
                 positions: dict[str, np.ndarray], types: dict[str, RobotType],
                 alive: dict[str, bool]) -> list[tuple[str, str]]:
        """Queue this step's transmissions. Returns (sender, reason) for rejected ones."""
        rejections = []
        budget: dict[str, float] = {}
        sim = self.cfg.simulate_channel
        latency = self.cfg.latency_s if sim else 0.0
        for sender, tx in outgoing:
            st = self._stat(sender)
            if not self.cfg.enabled or not alive.get(sender, False):
                continue
            data = self.cfg.codec.encode(tx.payload)
            size = len(data)
            if sim and self.cfg.max_payload_bytes is not None and size > self.cfg.max_payload_bytes:
                st["radio_rejected"] += 1
                rejections.append((sender, f"payload {size} B exceeds {self.cfg.max_payload_bytes} B"))
                continue
            if sim and self.cfg.max_bytes_per_s is not None:
                left = budget.setdefault(sender, self.cfg.max_bytes_per_s * dt)
                if size > left:
                    st["radio_dropped_bandwidth"] += 1
                    continue
                budget[sender] = left - size
            seq = self._seq[sender] = self._seq.get(sender, -1) + 1
            st["radio_sent"] += 1
            st["radio_bytes_sent"] += size
            st["radio_payload_min_bytes"] = float(np.fmin(size, st["radio_payload_min_bytes"]))  # fmin skips the nan
            st["radio_payload_max_bytes"] = float(np.fmax(size, st["radio_payload_max_bytes"]))
            targets = tx.to if tx.to is not None else [r for r in positions if r != sender]
            for r in targets:
                if r == sender or r not in positions:
                    continue
                if not alive.get(r, False):
                    st["radio_dropped_offline"] += 1
                    continue
                if sim:
                    link = min(self.cfg.range_m.get(types[sender], np.inf), self.cfg.range_m.get(types[r], np.inf))
                    if np.hypot(*(positions[r] - positions[sender])) > link:
                        st["radio_dropped_range"] += 1
                        continue
                if sim and self.cfg.drop_prob > 0 and self.rng.random() < self.cfg.drop_prob:
                    st["radio_dropped_loss"] += 1
                    continue
                msg = RadioMessage(
                    sender=sender, topic=tx.topic, payload=self.cfg.codec.decode(data),
                    recipients=None if tx.to is None else tuple(tx.to), sent_time=sent_time,
                    received_time=float("nan"), size_bytes=size, seq=seq,
                )
                heapq.heappush(self._pending, (sent_time + latency, next(self._order), r, msg))
        return rejections

    def collect(self, now: float, alive: dict[str, bool]) -> dict[str, list[RadioMessage]]:
        """Pop everything due by ``now``, grouped by recipient in send order."""
        inbox: dict[str, list[RadioMessage]] = {}
        while self._pending and self._pending[0][0] <= now + 1e-9:
            _, _, r, msg = heapq.heappop(self._pending)
            if not alive.get(r, False):
                continue
            inbox.setdefault(r, []).append(replace(msg, received_time=now))
            self._stat(r)["radio_received"] += 1
        return inbox
