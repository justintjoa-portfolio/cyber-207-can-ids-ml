#!/usr/bin/env python3
"""Conventional non-ML CAN IDS experiment.

The detector is calibrated using a separate normal-run text file.

CSV parsing, scoring, reporting, and folder processing are provided by
common.can_benchmark.
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.can_benchmark import (
    build_parser,
    CanDetector,
    DetectionResult,
    Frame,
    add_common_arguments,
    run_folder_benchmark,
)


NORMAL_LABEL = "R"


NORMAL_LINE_PATTERN = re.compile(
    r"^Timestamp:\s*(?P<timestamp>\d+(?:\.\d+)?)\s+"
    r"ID:\s*(?P<can_id>[0-9A-Fa-f]+)\s+"
    r"\S+\s+"
    r"DLC:\s*(?P<dlc>\d+)\s+"
    r"(?P<payload>(?:[0-9A-Fa-f]{2}(?:\s+|$))+)$"
)




class RuleBasedCanDetector(CanDetector):
    """Conventional profile/rule-based CAN IDS.

    Detection rules:

    1. Unknown CAN identifier
    2. Unexpected DLC
    3. Excessive message frequency
    4. Unexpected payload-byte values or ranges
    """

    def __init__(
        self,
        categorical_limit: int = 16,
        range_margin: int = 0,
        burst_ratio: float = 0.35,
        minimum_timing_samples: int = 8,
    ) -> None:
        super().__init__(categorical_limit, range_margin, burst_ratio, minimum_timing_samples)

    def fit(self, calibration_frames: list[Frame]) -> None:
        """Build fixed normal-behavior rules from known-clean CAN traffic."""

        if self.ready:
            raise RuntimeError("Detector has already been fitted")

        for frame in calibration_frames:
            self.profiles[frame.can_id].observe(frame)

        if not self.profiles:
            raise ValueError("No calibration frames were provided")

        for profile in self.profiles.values():
            profile.freeze(
                categorical_limit=self.categorical_limit,
                range_margin=self.range_margin,
                burst_ratio=self.burst_ratio,
                minimum_timing_samples=self.minimum_timing_samples,
            )

        self.ready = True

    def predict(self, frame: Frame) -> DetectionResult:
        """Classify one CAN frame using the frozen conventional rules."""

        if not self.ready:
            raise RuntimeError(
                "fit() must be called before predict()"
            )

        reasons: list[str] = []
        profile = self.profiles.get(frame.can_id)

        if profile is None:
            reasons.append("unknown_can_id")
            self.last_timestamp[frame.can_id] = frame.timestamp

            return DetectionResult(
                malicious=True,
                reasons=tuple(reasons),
            )

        if frame.dlc not in profile.expected_dlcs:
            reasons.append("unexpected_dlc")

        previous = self.last_timestamp.get(frame.can_id)

        if previous is not None and profile.min_period is not None:
            period = frame.timestamp - previous

            if 0 <= period < profile.min_period:
                reasons.append("excessive_frequency")

        self.last_timestamp[frame.can_id] = frame.timestamp

        for index, value in enumerate(frame.payload):
            if index >= len(profile.byte_values):
                reasons.append(f"unexpected_byte_{index}")
                continue

            allowed = profile.byte_allowed_sets[index]
            value_range = profile.byte_ranges[index]

            if allowed is not None and value not in allowed:
                reasons.append(
                    f"invalid_byte_{index}_value"
                )

            elif value_range is not None and not (
                value_range[0] <= value <= value_range[1]
            ):
                reasons.append(
                    f"byte_{index}_out_of_range"
                )

        return DetectionResult(
            malicious=bool(reasons),
            reasons=tuple(reasons),
        )


def parse_normal_run(path: Path) -> Iterator[Frame]:
    """Parse the separate known-normal CAN text log."""

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as stream:
        for row_number, line in enumerate(stream, start=1):
            stripped = line.strip()

            if not stripped:
                continue

            match = NORMAL_LINE_PATTERN.fullmatch(stripped)

            if match is None:
                continue

            try:
                timestamp = float(match.group("timestamp"))

                can_id = (
                    match.group("can_id")
                    .lower()
                    .removeprefix("0x")
                    .zfill(4)
                )

                dlc = int(match.group("dlc"))

                if dlc < 0 or dlc > 64:
                    continue

                payload_text = match.group("payload").split()

                if len(payload_text) != dlc:
                    continue

                payload = tuple(
                    int(value, 16)
                    for value in payload_text
                )

            except ValueError:
                continue

            yield Frame(
                row_number=row_number,
                timestamp=timestamp,
                can_id=can_id,
                dlc=dlc,
                payload=payload,
                label=NORMAL_LABEL,
            )


def load_normal_frames(path: Path) -> list[Frame]:
    """Load and validate the complete normal calibration run."""

    if not path.exists():
        raise FileNotFoundError(
            f"Normal-run file does not exist: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Normal-run path is not a file: {path}"
        )

    frames = list(parse_normal_run(path))

    if not frames:
        raise ValueError(
            f"No valid CAN frames were parsed from {path}"
        )

    return frames


def create_detector(
    args: argparse.Namespace,
    calibration_frames: list[Frame],
) -> CanDetector:
    """Create and fit a fresh detector for one evaluation dataset."""

    detector = RuleBasedCanDetector(
        categorical_limit=args.categorical_limit,
        range_margin=args.range_margin,
        burst_ratio=args.burst_ratio,
        minimum_timing_samples=args.minimum_timing_samples,
    )

    return detector




def main() -> None:
    args = build_parser().parse_args()

    calibration_frames = load_normal_frames(
        Path("../original_input_data/normal_run_data.txt")
    )

    def detector_factory(
        current_args: argparse.Namespace,
    ) -> CanDetector:
        return create_detector(
            current_args,
            calibration_frames,
        )

    run_folder_benchmark(
        args=args,
        detector_factory=detector_factory,
        training_data=calibration_frames,
        label="conventional"
    )


if __name__ == "__main__":
    main()