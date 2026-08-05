#!/usr/bin/env python3
"""Conventional non-ML CAN IDS experiment.

This detector loads a fixed, inferred CAN frame specification from JSON.

A frame is flagged as malicious when:

1. Its CAN identifier is not present in the specification.
2. Its DLC does not match the specification.
3. Its payload length does not match its DLC.
4. Its payload length does not match the specification.
5. A payload byte falls outside its allowed range.

CSV parsing, scoring, reporting, and folder processing are provided by
common.can_benchmark.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

can_comspec_path = (Path(__file__).resolve().parent.parent
    / "original_input_data"
    / "can_dbc_comspec.json"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.can_benchmark import (
    CanDetector,
    Frame,
    run_folder_benchmark,
)


DEFAULT_SPEC_PATH = (
    PROJECT_ROOT
    / "original_input_data"
    / "can_dbc_comspec.json"
)


@dataclass(frozen=True)
class ByteRange:
    """Inclusive allowed range for one payload byte."""

    minimum: int
    maximum: int

    def contains(self, value: int) -> bool:
        """Return whether the value is inside the allowed range."""
        return self.minimum <= value <= self.maximum


@dataclass(frozen=True)
class MessageSpecification:
    """Structural rules for one CAN identifier."""

    can_id: int
    dlc: int
    byte_ranges: tuple[ByteRange, ...]


# Normalize can id if string into a int value. 
def normalize_can_id(
    can_id: str | int,
) -> int:
    """Convert a hexadecimal CAN identifier into an integer."""

    if isinstance(can_id, int):
        return can_id

    normalized = (
        can_id
        .strip()
        .lower()
        .removeprefix("0x")
    )

    if not normalized:
        raise ValueError("CAN ID is empty")

    return int(normalized, 16)


# Load CanComSpec DBC File. Please note that the CanComSpec is an inferred mock
# file because the original data set didn't have one. 
def load_can_specification() -> dict[int, MessageSpecification]:
    """Load and validate the inferred CAN specification."""
    with can_comspec_path.open(
        "r",
        encoding="utf-8",
    ) as stream:
        document: Any = json.load(stream)

    if not isinstance(document, dict):
        raise ValueError(
            "CAN specification must be a JSON object"
        )

    raw_messages = document.get("messages")

    if not isinstance(raw_messages, list):
        raise ValueError(
            "CAN specification must contain a 'messages' list"
        )

    specifications: dict[int, MessageSpecification] = {}

    for entry_number, raw_message in enumerate(
        raw_messages,
        start=1,
    ):
        if not isinstance(raw_message, dict):
            raise ValueError(
                f"Message entry {entry_number} is not an object"
            )

        try:
            can_id = normalize_can_id(
                raw_message["can_id"]
            )
            dlc = int(raw_message["dlc"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Message entry {entry_number} has an invalid "
                "CAN ID or DLC"
            ) from error

        if not 0 <= can_id <= 0x1FFFFFFF:
            raise ValueError(
                f"Invalid CAN ID in entry {entry_number}: "
                f"0x{can_id:X}"
            )

        if not 0 <= dlc <= 64:
            raise ValueError(
                f"Invalid DLC for CAN ID 0x{can_id:X}: "
                f"{dlc}"
            )

        raw_ranges = raw_message.get("byte_ranges")

        if not isinstance(raw_ranges, list):
            raise ValueError(
                f"CAN ID 0x{can_id:X} has no byte_ranges list"
            )

        if len(raw_ranges) != dlc:
            raise ValueError(
                f"CAN ID 0x{can_id:X} declares DLC {dlc}, "
                f"but has {len(raw_ranges)} byte ranges"
            )

        byte_ranges: list[ByteRange] = []

        for byte_index, raw_range in enumerate(
            raw_ranges
        ):
            if not isinstance(raw_range, dict):
                raise ValueError(
                    f"CAN ID 0x{can_id:X}, byte "
                    f"{byte_index}: range is not an object"
                )

            try:
                minimum = int(raw_range["min"])
                maximum = int(raw_range["max"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"CAN ID 0x{can_id:X}, byte "
                    f"{byte_index}: invalid range"
                ) from error

            if not 0 <= minimum <= maximum <= 255:
                raise ValueError(
                    f"CAN ID 0x{can_id:X}, byte "
                    f"{byte_index}: invalid range "
                    f"[{minimum}, {maximum}]"
                )

            byte_ranges.append(
                ByteRange(
                    minimum=minimum,
                    maximum=maximum,
                )
            )

        if can_id in specifications:
            raise ValueError(
                f"Duplicate CAN ID in specification: "
                f"0x{can_id:X}"
            )

        specifications[can_id] = MessageSpecification(
            can_id=can_id,
            dlc=dlc,
            byte_ranges=tuple(byte_ranges),
        )

    if not specifications:
        raise ValueError(
            "CAN specification contains no messages"
        )

    return specifications


class DbcSpecificationCanDetector(CanDetector):
    """Non-ML detector using a fixed CAN specification."""

    def __init__(
        self,
        specifications: dict[int, MessageSpecification],
    ) -> None:
        self.specifications = specifications
        self.ready = False

    def init(self) -> None:
        """Initialize the detector without learning from test data."""
        self.ready = True

    def predict_is_malicious(
        self,
        frame: Frame,
    ) -> bool:
        """Validate one frame against the fixed specification."""

        if not self.ready:
            raise RuntimeError(
                "init() must be called before predict_is_malicious()"
            )

        is_malicious = False

        try:
            can_id = normalize_can_id(frame.can_id)
        except ValueError:
            return True

        specification = self.specifications.get(can_id)

        if specification is None:
            return True

        if frame.dlc != specification.dlc:
            is_malicious = True

        if len(frame.payload) != frame.dlc:
            is_malicious = True

        if len(frame.payload) != specification.dlc:
            is_malicious = True

        bytes_to_check = min(
            len(frame.payload),
            len(specification.byte_ranges),
        )

        for byte_index in range(bytes_to_check):
            value = frame.payload[byte_index]
            allowed_range = (
                specification.byte_ranges[byte_index]
            )

            if not allowed_range.contains(value):
                is_malicious = True

        return is_malicious


def create_detector() -> CanDetector:
    """Create a fresh fixed-specification detector."""

    specifications = load_can_specification()

    return DbcSpecificationCanDetector(
        specifications=specifications,
    )


def main() -> None:

    run_folder_benchmark(
        detector=create_detector(),
        label="dbc-specification-conventional",
    )


if __name__ == "__main__":
    main()

