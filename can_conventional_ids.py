#!/usr/bin/env python3
"""Conventional profile/rule-based IDS benchmark for CAN CSV datasets.

The script automatically processes every CSV file in the input_data/ folder.

Expected row format (variable length according to DLC):
    timestamp,can_id,dlc,data0,...,dataN,label

where:
    R = normal
    T = attack

The detector calibrates on the initial consecutive normal prefix, freezes its
profile, and evaluates every later frame. Labels are used only for scoring and
to verify that calibration data is clean; they are never used by detection.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

NORMAL_LABEL = "R"
ATTACK_LABEL = "T"


@dataclass(frozen=True)
class Frame:
    row_number: int
    timestamp: float
    can_id: str
    dlc: int
    payload: tuple[int, ...]
    label: str


@dataclass
class IdProfile:
    dlc_counts: Counter[int] = field(default_factory=Counter)
    byte_values: list[Counter[int]] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)

    expected_dlcs: set[int] = field(default_factory=set)
    byte_allowed_sets: list[set[int] | None] = field(default_factory=list)
    byte_ranges: list[tuple[int, int] | None] = field(default_factory=list)
    median_period: float | None = None
    min_period: float | None = None

    def observe(self, frame: Frame) -> None:
        self.dlc_counts[frame.dlc] += 1

        while len(self.byte_values) < frame.dlc:
            self.byte_values.append(Counter())

        for index, value in enumerate(frame.payload):
            self.byte_values[index][value] += 1

        self.timestamps.append(frame.timestamp)

    def freeze(
        self,
        categorical_limit: int,
        range_margin: int,
        burst_ratio: float,
        minimum_timing_samples: int,
    ) -> None:
        self.expected_dlcs = set(self.dlc_counts)

        for counts in self.byte_values:
            values = sorted(counts)

            if len(values) <= categorical_limit:
                self.byte_allowed_sets.append(set(values))
                self.byte_ranges.append(None)
            else:
                low = max(0, values[0] - range_margin)
                high = min(255, values[-1] + range_margin)
                self.byte_allowed_sets.append(None)
                self.byte_ranges.append((low, high))

        periods = [
            later - earlier
            for earlier, later in zip(self.timestamps, self.timestamps[1:])
            if later > earlier
        ]

        if len(periods) >= minimum_timing_samples:
            median = statistics.median(periods)
            self.median_period = median
            self.min_period = max(1e-6, median * burst_ratio)

        self.timestamps.clear()


class ConventionalCanIds:
    def __init__(
        self,
        categorical_limit: int = 16,
        range_margin: int = 0,
        burst_ratio: float = 0.35,
        minimum_timing_samples: int = 8,
    ) -> None:
        self.profiles: dict[str, IdProfile] = defaultdict(IdProfile)
        self.last_timestamp: dict[str, float] = {}
        self.categorical_limit = categorical_limit
        self.range_margin = range_margin
        self.burst_ratio = burst_ratio
        self.minimum_timing_samples = minimum_timing_samples
        self.frozen = False

    def observe_normal(self, frame: Frame) -> None:
        if self.frozen:
            raise RuntimeError("Cannot add calibration frames after freeze()")

        self.profiles[frame.can_id].observe(frame)

    def freeze(self) -> None:
        if not self.profiles:
            raise ValueError("No calibration frames were provided")

        for profile in self.profiles.values():
            profile.freeze(
                categorical_limit=self.categorical_limit,
                range_margin=self.range_margin,
                burst_ratio=self.burst_ratio,
                minimum_timing_samples=self.minimum_timing_samples,
            )

        self.frozen = True

    def detect(self, frame: Frame) -> list[str]:
        if not self.frozen:
            raise RuntimeError("freeze() must be called before detect()")

        reasons: list[str] = []
        profile = self.profiles.get(frame.can_id)

        if profile is None:
            reasons.append("unknown_can_id")
            self.last_timestamp[frame.can_id] = frame.timestamp
            return reasons

        if frame.dlc not in profile.expected_dlcs:
            reasons.append("unexpected_dlc")

        previous = self.last_timestamp.get(frame.can_id)

        if previous is not None and profile.min_period is not None:
            period = frame.timestamp - previous

            if period >= 0 and period < profile.min_period:
                reasons.append("excessive_frequency")

        self.last_timestamp[frame.can_id] = frame.timestamp

        for index, value in enumerate(frame.payload):
            if index >= len(profile.byte_values):
                reasons.append(f"unexpected_byte_{index}")
                continue

            allowed = profile.byte_allowed_sets[index]
            value_range = profile.byte_ranges[index]

            if allowed is not None and value not in allowed:
                reasons.append(f"invalid_byte_{index}_value")
            elif value_range is not None and not (
                value_range[0] <= value <= value_range[1]
            ):
                reasons.append(f"byte_{index}_out_of_range")

        return reasons


def parse_frames(path: Path) -> Iterator[Frame]:
    with path.open(
        "r",
        newline="",
        encoding="utf-8",
        errors="replace",
    ) as stream:
        reader = csv.reader(stream)

        for row_number, row in enumerate(reader, start=1):
            if len(row) < 4:
                continue

            try:
                timestamp = float(row[0].strip())
                can_id = (
                    row[1]
                    .strip()
                    .lower()
                    .removeprefix("0x")
                    .zfill(4)
                )
                dlc = int(row[2].strip())

                if dlc < 0 or dlc > 64:
                    raise ValueError("invalid DLC")

                if len(row) < dlc + 4:
                    raise ValueError("row shorter than DLC")

                payload = tuple(
                    int(value.strip(), 16)
                    for value in row[3 : 3 + dlc]
                )

                label = row[-1].strip().upper()

                if label not in {NORMAL_LABEL, ATTACK_LABEL}:
                    raise ValueError("unknown label")

            except (ValueError, IndexError):
                continue

            yield Frame(
                row_number=row_number,
                timestamp=timestamp,
                can_id=can_id,
                dlc=dlc,
                payload=payload,
                label=label,
            )


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else 0.0


def benchmark(path: Path, args: argparse.Namespace) -> dict[str, object]:
    detector = ConventionalCanIds(
        categorical_limit=args.categorical_limit,
        range_margin=args.range_margin,
        burst_ratio=args.burst_ratio,
        minimum_timing_samples=args.minimum_timing_samples,
    )

    frames = parse_frames(path)
    calibration_count = 0
    first_test_frame: Frame | None = None

    for frame in frames:
        if (
            frame.label != NORMAL_LABEL
            or calibration_count >= args.max_calibration_frames
        ):
            first_test_frame = frame
            break

        detector.observe_normal(frame)
        calibration_count += 1

    if calibration_count < args.minimum_calibration_frames:
        raise ValueError(
            f"Only {calibration_count} clean prefix frames in {path.name}; "
            f"need at least {args.minimum_calibration_frames}."
        )

    detector.freeze()

    counts = Counter(tp=0, fp=0, tn=0, fn=0)
    reason_counts: Counter[str] = Counter()
    alert_examples: list[dict[str, object]] = []
    evaluated = 0

    def evaluate(frame: Frame) -> None:
        nonlocal evaluated

        reasons = detector.detect(frame)
        predicted_attack = bool(reasons)
        actual_attack = frame.label == ATTACK_LABEL
        evaluated += 1

        if predicted_attack and actual_attack:
            counts["tp"] += 1
        elif predicted_attack and not actual_attack:
            counts["fp"] += 1
        elif not predicted_attack and actual_attack:
            counts["fn"] += 1
        else:
            counts["tn"] += 1

        reason_counts.update(reasons)

        if reasons and len(alert_examples) < args.example_alerts:
            alert_examples.append(
                {
                    "row": frame.row_number,
                    "timestamp": frame.timestamp,
                    "can_id": frame.can_id,
                    "label": frame.label,
                    "reasons": reasons,
                }
            )

    if first_test_frame is not None:
        evaluate(first_test_frame)

    for frame in frames:
        evaluate(frame)

    tp = counts["tp"]
    fp = counts["fp"]
    tn = counts["tn"]
    fn = counts["fn"]

    accuracy = safe_divide(tp + tn, tp + fp + tn + fn)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    false_positive_rate = safe_divide(fp, fp + tn)
    false_negative_rate = safe_divide(fn, fn + tp)

    return {
        "dataset": path.name,
        "calibration_frames": calibration_count,
        "evaluated_frames": evaluated,
        "profiled_can_ids": len(detector.profiles),
        "confusion_matrix": {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        },
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall_tpr": recall,
            "specificity_tnr": specificity,
            "f1": f1,
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
        },
        "alert_reason_counts": dict(reason_counts.most_common()),
        "alert_examples": alert_examples,
    }


def print_report(result: dict[str, object]) -> None:
    cm = result["confusion_matrix"]
    metrics = result["metrics"]

    assert isinstance(cm, dict)
    assert isinstance(metrics, dict)

    print(f"\nDataset: {result['dataset']}")
    print(f"Calibration frames: {result['calibration_frames']:,}")
    print(f"Evaluated frames:   {result['evaluated_frames']:,}")
    print(f"Profiled CAN IDs:   {result['profiled_can_ids']}")

    print("\nConfusion matrix (attack is the positive class)")
    print(f"  TP: {cm['tp']:,}    FP: {cm['fp']:,}")
    print(f"  FN: {cm['fn']:,}    TN: {cm['tn']:,}")

    print("\nMetrics")
    for name, value in metrics.items():
        print(f"  {name:24s} {value:.6f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("input_data"),
        help="Folder containing CAN CSV datasets.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("can_ids_results.json"),
        help="Combined JSON results file.",
    )
    parser.add_argument(
        "--max-calibration-frames",
        type=int,
        default=100_000,
    )
    parser.add_argument(
        "--minimum-calibration-frames",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--categorical-limit",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--range-margin",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--burst-ratio",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--minimum-timing-samples",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--example-alerts",
        type=int,
        default=10,
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_dir: Path = args.input_dir

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Could not find input directory: {input_dir}"
        )

    if not input_dir.is_dir():
        raise NotADirectoryError(
            f"Input path is not a directory: {input_dir}"
        )

    datasets = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".csv"
    )

    if not datasets:
        raise FileNotFoundError(
            f"No CSV files found inside {input_dir}"
        )

    print(f"Found {len(datasets)} CSV dataset(s) in {input_dir}:")

    for dataset in datasets:
        print(f"  - {dataset.name}")

    results: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []

    for dataset in datasets:
        print(f"\nProcessing {dataset.name}...")

        try:
            result = benchmark(dataset, args)
            results.append(result)
            print_report(result)
        except Exception as error:
            message = str(error)
            failures.append(
                {
                    "dataset": dataset.name,
                    "error": message,
                }
            )
            print(f"Failed to process {dataset.name}: {message}")

    report = {
        "input_directory": str(input_dir),
        "successful_datasets": len(results),
        "failed_datasets": len(failures),
        "results": results,
        "failures": failures,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved combined JSON report to {args.output}")

    if not results:
        raise RuntimeError(
            "No datasets were processed successfully. "
            "See the JSON report for failure details."
        )


if __name__ == "__main__":
    main()