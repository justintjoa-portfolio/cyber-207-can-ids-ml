#!/usr/bin/env python3
"""Causal sliding-window features for the CAN ML detector.

This module deliberately reuses common.can_benchmark.Frame rather than
defining another CAN-frame representation or another CSV parser.
"""

from __future__ import annotations

from collections import Counter, deque

import numpy as np

from common.can_benchmark import Frame


FEATURE_COLUMNS = [
    "can_id",
    "dlc",
    "time_since_last_submission",
    "byte_0",
    "byte_1",
    "byte_2",
    "byte_3",
    "byte_4",
    "byte_5",
    "byte_6",
    "byte_7",
]


def target_from_frame(frame: Frame) -> int:
    """Return 1 for attack and 0 for normal."""
    return int(frame.label == "T")


# Get frame payload and pad up to 8 bytes if less than i bytes. 
def padded_payload(frame: Frame) -> tuple[int, ...]:
    payload = tuple(frame.payload[:8])
    return payload + (0,) * (8 - len(payload))


