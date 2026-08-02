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
    "delta_time",
    "id_count_window",
    "id_fraction_window",
    "unique_ids_window",
    "same_id_run_length",
    "mean_delta_time_window",
    "std_delta_time_window",
    "payload_changed",
    "payload_hamming_bytes",
    "payload_mean",
    "payload_std",
    "payload_min",
    "payload_max",
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


def padded_payload(frame: Frame) -> tuple[int, ...]:
    """Return exactly eight payload bytes for the classic CAN dataset."""

    payload = tuple(frame.payload[:8])
    return payload + (0,) * (8 - len(payload))


class SlidingWindowFeatureExtractor:
    """Generate one causal feature vector per CAN frame.

    Only the current frame and earlier frames are used. This makes the same
    extractor valid for both offline training and online inference.
    """

    def __init__(self, window_size: int = 50) -> None:
        if window_size < 2:
            raise ValueError("window_size must be at least 2")

        self.window_size = window_size
        self.history: deque[Frame] = deque(maxlen=window_size)
        self.id_counts: Counter[str] = Counter()
        self.last_frame_by_id: dict[str, Frame] = {}
        self.previous_id: str | None = None
        self.same_id_run_length = 0

    def reset(self) -> None:
        self.history.clear()
        self.id_counts.clear()
        self.last_frame_by_id.clear()
        self.previous_id = None
        self.same_id_run_length = 0

    def extract(self, frame: Frame) -> list[float]:
        previous_frame = self.history[-1] if self.history else None
        previous_same_id = self.last_frame_by_id.get(frame.can_id)

        delta_time = (
            max(0.0, frame.timestamp - previous_frame.timestamp)
            if previous_frame is not None
            else 0.0
        )

        if self.previous_id == frame.can_id:
            self.same_id_run_length += 1
        else:
            self.same_id_run_length = 1

        projected_length = len(self.history) + 1
        id_count = self.id_counts[frame.can_id] + 1

        deltas = [
            max(0.0, newer.timestamp - older.timestamp)
            for older, newer in zip(self.history, list(self.history)[1:])
        ]
        if previous_frame is not None:
            deltas.append(delta_time)

        current_payload = np.asarray(padded_payload(frame), dtype=np.uint8)

        if previous_same_id is None:
            payload_changed = 0
            payload_hamming_bytes = 0
        else:
            previous_payload = np.asarray(
                padded_payload(previous_same_id),
                dtype=np.uint8,
            )
            payload_hamming_bytes = int(
                np.count_nonzero(current_payload != previous_payload)
            )
            payload_changed = int(payload_hamming_bytes > 0)

        payload_float = current_payload.astype(float)

        features = [
            float(int(frame.can_id, 16)),
            float(frame.dlc),
            float(delta_time),
            float(id_count),
            float(id_count / projected_length),
            float(len(set(self.id_counts) | {frame.can_id})),
            float(self.same_id_run_length),
            float(np.mean(deltas)) if deltas else 0.0,
            float(np.std(deltas)) if deltas else 0.0,
            float(payload_changed),
            float(payload_hamming_bytes),
            float(payload_float.mean()),
            float(payload_float.std()),
            float(payload_float.min()),
            float(payload_float.max()),
            *payload_float.tolist(),
        ]

        self._append(frame)
        return features

    def _append(self, frame: Frame) -> None:
        if len(self.history) == self.history.maxlen:
            evicted = self.history[0]
            self.id_counts[evicted.can_id] -= 1

            if self.id_counts[evicted.can_id] <= 0:
                del self.id_counts[evicted.can_id]

        self.history.append(frame)
        self.id_counts[frame.can_id] += 1
        self.last_frame_by_id[frame.can_id] = frame
        self.previous_id = frame.can_id