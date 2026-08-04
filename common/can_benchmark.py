#!/usr/bin/env python3
"""Reusable CAN IDS benchmarking library.

This module provides:

- Frame parsing from labeled CAN CSV files
- A generic detector interface
- Generic detector initialization and evaluation flow
- TP, FP, TN, FN and derived metrics
- Per-dataset elapsed time and throughput reporting
- Folder-wide CSV processing
- JSON and terminal reporting

A conventional detector may load rules from a DBC file, while a
machine-learning detector may load a pre-trained model. Each detector
implements the CanDetector interface.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

HELDOUT_DATA_DIR = Path(__file__).resolve().parent.parent / "build" / "heldout_test_data"

NORMAL_LABEL = "R"
ATTACK_LABEL = "T"


@dataclass(frozen=True)
class Frame:
    """One normalized CAN frame from a CSV dataset."""

    row_number: int
    timestamp: float
    can_id: str
    dlc: int
    payload: tuple[int, ...]
    label: str


class CanDetector(ABC):
    """Interface implemented by conventional and ML detectors."""

    def __init__(self) -> None:
        self.ready = False

    @abstractmethod
    def init(self) -> None:
        """Initialize the detector from its configured rules or model."""

    @abstractmethod
    def predict_is_malicious(self, frame: Frame) -> bool:
        """Classify one frame as malicious or normal."""


@dataclass
class EvaluationState:
    """Mutable counters used while evaluating a dataset."""

    counts: Counter[str] = field(
        # tp is true positive, fp is false positive, 
        # tn is true negative, fn is false negative
        default_factory=lambda: Counter(
            tp=0,
            fp=0,
            tn=0,
            fn=0,
        )
    )
    reason_counts: Counter[str] = field(default_factory=Counter)
    evaluated_frames: int = 0


# Use this function to get frames from the input CSV files! 
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
    """Divide two values while safely handling a zero denominator."""

    return numerator / denominator if denominator else 0.0


def calculate_metrics(
    counts: Counter[str],
) -> dict[str, float]:
    """Calculate standard binary-classification metrics."""

    tp = counts["tp"]
    fp = counts["fp"]
    tn = counts["tn"]
    fn = counts["fn"]

    accuracy = safe_divide(
        tp + tn,
        tp + fp + tn + fn,
    )
    precision = safe_divide(
        tp,
        tp + fp,
    )
    recall = safe_divide(
        tp,
        tp + fn,
    )
    specificity = safe_divide(
        tn,
        tn + fp,
    )
    f1 = safe_divide(
        2 * precision * recall,
        precision + recall,
    )
    false_positive_rate = safe_divide(
        fp,
        fp + tn,
    )
    false_negative_rate = safe_divide(
        fn,
        fn + tp,
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall_tpr": recall,
        "specificity_tnr": specificity,
        "f1": f1,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
    }

# Check if a frame is malicious and update the evaluation state accordingly.
def evaluate_frame(
    frame: Frame,
    detector: CanDetector,
    state: EvaluationState
) -> None:
    """Classify one frame and update generic scoring state."""

    is_malicious = detector.predict_is_malicious(frame)

    actual_attack = frame.label == ATTACK_LABEL

    state.evaluated_frames += 1

    if is_malicious and actual_attack:
        state.counts["tp"] += 1
    elif is_malicious and not actual_attack:
        state.counts["fp"] += 1
    elif not is_malicious and actual_attack:
        state.counts["fn"] += 1
    else:
        state.counts["tn"] += 1


def benchmark_dataset(
    path: Path,
    detector: CanDetector
) -> dict[str, object]:
    """Evaluate one detector against one CSV dataset."""

    detector.init()

    state = EvaluationState()

    start_time = time.perf_counter()

    for frame in parse_frames(path):
        evaluate_frame(
            frame=frame,
            detector=detector,
            state=state
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
        )
    }


def print_report(
    result: dict[str, object],
) -> None:
    """Print one dataset result to the terminal."""

    confusion_matrix = result["confusion_matrix"]
    metrics = result["metrics"]

    assert isinstance(confusion_matrix, dict)
    assert isinstance(metrics, dict)

    print(f"\nDataset: {result['dataset']}")
    print(f"Detector:           {result['detector']}")
    print(
        f"Evaluated frames:   "
        f"{result['evaluated_frames']:,}"
    )
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
    print(
        f"  TP: {confusion_matrix['tp']:,}    "
        f"FP: {confusion_matrix['fp']:,}"
    )
    print(
        f"  FN: {confusion_matrix['fn']:,}    "
        f"TN: {confusion_matrix['tn']:,}"
    )

    print("\nMetrics")

    for name, value in metrics.items():
        print(f"  {name:24s} {value:.6f}")


def find_csv_datasets(
    input_dir: Path,
) -> list[Path]:
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
        if path.is_file()
        and path.suffix.lower() == ".csv"
    )

    if not datasets:
        raise FileNotFoundError(
            f"No CSV files found inside {input_dir}"
        )

    return datasets

# Run benchmark with a detector against all CSV datasets in the input folder.
# We explicitly run on the held out data as that is the data excluded from
# training set for ML. This gives a fair comparison between the ML and the 
# DBC methods. 
def run_folder_benchmark(
    detector: CanDetector, 
    label: str,
) -> dict[str, object]:
    """Run a detector factory against every CSV in the input folder."""

    input_dir: Path = Path(HELDOUT_DATA_DIR)
    datasets = find_csv_datasets(input_dir)

    print(
        f"Found {len(datasets)} CSV dataset(s) "
        f"in {input_dir}:"
    )

    for dataset in datasets:
        print(f"  - {dataset.name}")

    results: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []

    folder_start_time = time.perf_counter()

    for dataset in datasets:
        print(f"\nProcessing {dataset.name}...")

        try:

            result = benchmark_dataset(
                path=dataset,
                detector=detector
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

            print(
                f"Failed to process "
                f"{dataset.name}: {message}"
            )

    folder_elapsed_seconds = (
        time.perf_counter() - folder_start_time
    )

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

    output_path = Path(
        f"{label}_can_ids_results.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
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

    print(
        f"\nSaved combined JSON report to "
        f"{output_path}"
    )

    if not results:
        raise RuntimeError(
            "No datasets were processed successfully. "
            "See the JSON report for failure details."
        )

    return report