#!/usr/bin/env python3
"""Reusable CAN IDS benchmarking library.

This module provides:

- Frame parsing from labeled CAN CSV files
- A generic detector interface
- Generic calibration and evaluation flow
- TP, FP, TN, FN and derived metrics
- Per-dataset elapsed time and throughput reporting
- Folder-wide CSV processing
- JSON and terminal reporting

Any conventional or machine-learning detector can use this library by
implementing the CanDetector interface.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator


NORMAL_LABEL = "R"
ATTACK_LABEL = "T"


@dataclass
class IdProfile:
    """Learned conventional profile for one CAN identifier."""

    dlc_counts: Counter[int] = field(default_factory=Counter)
    byte_values: list[Counter[int]] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)

    expected_dlcs: set[int] = field(default_factory=set)
    byte_allowed_sets: list[set[int] | None] = field(default_factory=list)
    byte_ranges: list[tuple[int, int] | None] = field(default_factory=list)

    median_period: float | None = None
    min_period: float | None = None

    def observe(self, frame: Frame) -> None:
        """Record one known-normal frame before rules are frozen."""

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
        """Convert collected normal observations into fixed IDS rules."""

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
            for earlier, later in zip(
                self.timestamps,
                self.timestamps[1:],
            )
            if later > earlier
        ]

        if len(periods) >= minimum_timing_samples:
            median = statistics.median(periods)

            self.median_period = median
            self.min_period = max(
                1e-6,
                median * burst_ratio,
            )

        self.timestamps.clear()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    add_common_arguments(parser)

    parser.add_argument(
        "--categorical-limit",
        type=int,
        default=16,
        help=(
            "Maximum distinct calibration values before a byte "
            "uses a numeric range instead of an allowed set."
        ),
    )

    parser.add_argument(
        "--range-margin",
        type=int,
        default=0,
        help="Margin added around learned byte ranges.",
    )

    parser.add_argument(
        "--burst-ratio",
        type=float,
        default=0.35,
        help=(
            "Minimum acceptable period as a fraction of the "
            "learned median period."
        ),
    )

    parser.add_argument(
        "--minimum-timing-samples",
        type=int,
        default=8,
        help=(
            "Minimum number of timing intervals required before "
            "enabling the timing rule."
        ),
    )

    return parser


@dataclass(frozen=True)
class Frame:
    """One normalized CAN frame from a CSV dataset."""

    row_number: int
    timestamp: float
    can_id: str
    dlc: int
    payload: tuple[int, ...]
    label: str


@dataclass(frozen=True)
class DetectionResult:
    """Detector output for one CAN frame."""

    malicious: bool
    reasons: tuple[str, ...] = ()


class CanDetector(ABC):
    """Interface implemented by conventional and ML detectors."""

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

        self.ready = False

    @abstractmethod
    def fit(self, calibration_frames: list[Frame]) -> None:
        """Prepare the detector using calibration or pre-trained state."""

    @abstractmethod
    def predict(self, frame: Frame) -> DetectionResult:
        """Classify one frame as malicious or normal."""


DetectorFactory = Callable[[argparse.Namespace], CanDetector]


@dataclass
class EvaluationState:
    """Mutable counters used while evaluating a dataset."""

    counts: Counter[str] = field(
        default_factory=lambda: Counter(tp=0, fp=0, tn=0, fn=0)
    )
    reason_counts: Counter[str] = field(default_factory=Counter)
    alert_examples: list[dict[str, object]] = field(default_factory=list)
    evaluated_frames: int = 0


def parse_frames(path: Path) -> Iterator[Frame]:
    """Yield valid CAN frames from a CSV file one row at a time."""

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


def safe_divide(
    numerator: int | float,
    denominator: int | float,
) -> float:
    return numerator / denominator if denominator else 0.0


def calculate_metrics(counts: Counter[str]) -> dict[str, float]:
    """Calculate standard binary-classification metrics."""

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
        "accuracy": accuracy,
        "precision": precision,
        "recall_tpr": recall,
        "specificity_tnr": specificity,
        "f1": f1,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
    }


def evaluate_frame(
    frame: Frame,
    detector: CanDetector,
    state: EvaluationState,
    maximum_examples: int,
) -> None:
    """Classify one frame and update generic scoring state."""

    result = detector.predict(frame)

    predicted_attack = result.malicious
    actual_attack = frame.label == ATTACK_LABEL

    state.evaluated_frames += 1

    if predicted_attack and actual_attack:
        state.counts["tp"] += 1
    elif predicted_attack and not actual_attack:
        state.counts["fp"] += 1
    elif not predicted_attack and actual_attack:
        state.counts["fn"] += 1
    else:
        state.counts["tn"] += 1

    state.reason_counts.update(result.reasons)

    if predicted_attack and len(state.alert_examples) < maximum_examples:
        state.alert_examples.append(
            {
                "row": frame.row_number,
                "timestamp": frame.timestamp,
                "can_id": frame.can_id,
                "dlc": frame.dlc,
                "payload": [
                    f"{value:02x}"
                    for value in frame.payload
                ],
                "actual_label": frame.label,
                "predicted_malicious": result.malicious,
                "reasons": list(result.reasons),
            }
        )


def benchmark_dataset(
    path: Path,
    detector: CanDetector,
    training_data: list[Frame],
    args: argparse.Namespace,
) -> dict[str, object]:
    """Evaluate one detector against one CSV dataset."""

    frames = parse_frames(path)

    detector.fit(training_data)

    state = EvaluationState()
    start_time = time.perf_counter()

    for frame in frames:
        evaluate_frame(
            frame=frame,
            detector=detector,
            state=state,
            maximum_examples=args.example_alerts,
        )

    elapsed_seconds = time.perf_counter() - start_time
    frames_per_second = safe_divide(
        state.evaluated_frames,
        elapsed_seconds,
    )
    average_seconds_per_frame = safe_divide(
        elapsed_seconds,
        state.evaluated_frames,
    )

    return {
        "dataset": path.name,
        "detector": type(detector).__name__,
        "evaluated_frames": state.evaluated_frames,
        "elapsed_seconds": elapsed_seconds,
        "frames_per_second": frames_per_second,
        "average_seconds_per_frame": average_seconds_per_frame,
        "average_microseconds_per_frame": (
            average_seconds_per_frame * 1_000_000
        ),
        "confusion_matrix": {
            "tp": state.counts["tp"],
            "fp": state.counts["fp"],
            "tn": state.counts["tn"],
            "fn": state.counts["fn"],
        },
        "metrics": calculate_metrics(state.counts),
        "alert_reason_counts": dict(
            state.reason_counts.most_common()
        ),
        "alert_examples": state.alert_examples,
    }


def print_report(result: dict[str, object]) -> None:
    """Print one dataset result to the terminal."""

    cm = result["confusion_matrix"]
    metrics = result["metrics"]

    assert isinstance(cm, dict)
    assert isinstance(metrics, dict)

    print(f"\nDataset: {result['dataset']}")
    print(f"Detector:           {result['detector']}")
    print(f"Evaluated frames:   {result['evaluated_frames']:,}")
    print(
        f"Elapsed time:       "
        f"{result['elapsed_seconds']:.6f} seconds"
    )
    print(
        f"Throughput:         "
        f"{result['frames_per_second']:,.2f} frames/second"
    )
    print(
        f"Average latency:    "
        f"{result['average_microseconds_per_frame']:,.2f} "
        f"microseconds/frame"
    )

    print("\nConfusion matrix (attack is the positive class)")
    print(f"  TP: {cm['tp']:,}    FP: {cm['fp']:,}")
    print(f"  FN: {cm['fn']:,}    TN: {cm['tn']:,}")

    print("\nMetrics")

    for name, value in metrics.items():
        print(f"  {name:24s} {value:.6f}")


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by all detector runners."""

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("../build/heldout_test_data"),
        help="Folder containing CAN CSV datasets.",
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
        "--example-alerts",
        type=int,
        default=10,
    )


def find_csv_datasets(input_dir: Path) -> list[Path]:
    """Return all CSV files directly inside the input directory."""

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

    return datasets


def run_folder_benchmark(
    args: argparse.Namespace,
    detector_factory: DetectorFactory,
    training_data: list[Frame],
    label: str,
) -> dict[str, object]:
    """Run a detector factory against every CSV in the input folder."""

    input_dir: Path = args.input_dir
    datasets = find_csv_datasets(input_dir)

    print(f"Found {len(datasets)} CSV dataset(s) in {input_dir}:")

    for dataset in datasets:
        print(f"  - {dataset.name}")

    results: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []

    folder_start_time = time.perf_counter()

    for dataset in datasets:
        print(f"\nProcessing {dataset.name}...")

        try:
            detector = detector_factory(args)

            result = benchmark_dataset(
                path=dataset,
                detector=detector,
                args=args,
                training_data=training_data,
            )

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

    folder_elapsed_seconds = time.perf_counter() - folder_start_time

    total_frames = sum(
        int(result["evaluated_frames"])
        for result in results
    )
    overall_frames_per_second = safe_divide(
        total_frames,
        folder_elapsed_seconds,
    )
    overall_average_seconds_per_frame = safe_divide(
        folder_elapsed_seconds,
        total_frames,
    )

    report = {
        "input_directory": str(input_dir),
        "successful_datasets": len(results),
        "failed_datasets": len(failures),
        "total_evaluated_frames": total_frames,
        "total_elapsed_seconds": folder_elapsed_seconds,
        "overall_frames_per_second": overall_frames_per_second,
        "overall_average_seconds_per_frame": (
            overall_average_seconds_per_frame
        ),
        "overall_average_microseconds_per_frame": (
            overall_average_seconds_per_frame * 1_000_000
        ),
        "results": results,
        "failures": failures,
    }

    args.output = Path(label + "_can_ids_results.json")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    args.output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nOverall benchmark performance")
    print(f"  Evaluated frames:   {total_frames:,}")
    print(
        f"  Total elapsed time: "
        f"{folder_elapsed_seconds:.6f} seconds"
    )
    print(
        f"  Overall throughput: "
        f"{overall_frames_per_second:,.2f} frames/second"
    )
    print(
        f"  Average latency:    "
        f"{overall_average_seconds_per_frame * 1_000_000:,.2f} "
        f"microseconds/frame"
    )

    print(f"\nSaved combined JSON report to {args.output}")

    if not results:
        raise RuntimeError(
            "No datasets were processed successfully. "
            "See the JSON report for failure details."
        )

    return report