"""Payload serialisation for robot-to-robot radio messages.

A radio payload is algorithm-defined: a dict, a dataclass, numpy arrays, raw bytes,
or anything else the codec can serialise. Each transmission is encoded once, which
gives its on-air size, and decoded separately for every recipient. Robots therefore
never share mutable objects. On hardware, swap :class:`PickleCodec` for a
cross-language codec (msgpack, protobuf, a ROS message) without touching
controller code.
"""

from __future__ import annotations

import pickle
from typing import Any, Protocol


class PayloadCodec(Protocol):
    def encode(self, payload: Any) -> bytes: ...

    def decode(self, data: bytes) -> Any: ...


class PickleCodec:
    """Default codec: any picklable Python object (Python-only, trusted peers)."""

    def encode(self, payload: Any) -> bytes:
        return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)

    def decode(self, data: bytes) -> Any:
        return pickle.loads(data)
