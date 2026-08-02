#!/usr/bin/env python3
"""Train a Random Forest CAN IDS using the shared benchmark parser.

The split is chronological within every CSV:

    first 60%  -> training
    next 20%   -> validation
    final 20%  -> held-out testing

A purge gap equal to the sliding-window size is removed at each boundary.
That prevents overlapping neighboring windows from appearing in two splits.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.can_benchmark import (
    ATTACK_LABEL,
    Frame,
    calculate_metrics,
    find_csv_datasets,
    parse_frames,
)
from ml.can_ml_features import (
    FEATURE_COLUMNS,
    SlidingWindowFeatureExtractor,
    target_from_frame,
)


def build_feature_rows(
    frames: list[Frame],
    window_size: int,
) -> list[dict[str, object]]:
    extractor = SlidingWindowFeatureExtractor(window_size)
    rows: list[dict[str, object]] = []

    for frame in frames:
        values = extractor.extract(frame)
        row = dict(zip(FEATURE_COLUMNS, values))
        row.update(
            {
                "target": target_from_frame(frame),
                "row_number": frame.row_number,
                "timestamp": frame.timestamp,
            }
        )
        rows.append(row)

    return rows


def split_rows(
    rows: list[dict[str, object]],
    purge_gap: int,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    n_rows = len(rows)
    train_boundary = int(n_rows * 0.60)
    validation_boundary = int(n_rows * 0.80)

    train_stop = max(0, train_boundary - purge_gap)
    validation_start = min(n_rows, train_boundary + purge_gap)
    validation_stop = max(
        validation_start,
        validation_boundary - purge_gap,
    )
    test_start = min(n_rows, validation_boundary + purge_gap)

    return (
        rows[:train_stop],
        rows[validation_start:validation_stop],
        rows[test_start:],
    )


def rows_to_xy(
    rows: list[dict[str, object]],
) -> tuple[pd.DataFrame, pd.Series]:
    table = pd.DataFrame(rows)

    if table.empty:
        raise ValueError("A data split is empty")

    return table[FEATURE_COLUMNS], table["target"].astype(int)


def attack_probability(
    model: RandomForestClassifier,
    features: pd.DataFrame,
) -> np.ndarray:
    attack_index = list(model.classes_).index(1)
    return model.predict_proba(features)[:, attack_index]


def choose_threshold(
    labels: pd.Series,
    probabilities: np.ndarray,
) -> float:
    precision, recall, thresholds = precision_recall_curve(
        labels,
        probabilities,
    )

    if len(thresholds) == 0:
        return 0.5

    f1_values = (
        2 * precision[:-1] * recall[:-1]
        / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    )
    return float(thresholds[int(np.nanargmax(f1_values))])


def confusion_counts(
    labels: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> Counter[str]:
    predictions = (probabilities >= threshold).astype(int)
    counts: Counter[str] = Counter(tp=0, fp=0, tn=0, fn=0)

    for actual, predicted in zip(labels, predictions):
        if predicted == 1 and actual == 1:
            counts["tp"] += 1
        elif predicted == 1 and actual == 0:
            counts["fp"] += 1
        elif predicted == 0 and actual == 1:
            counts["fn"] += 1
        else:
            counts["tn"] += 1

    return counts


def print_metrics(
    name: str,
    labels: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    counts = confusion_counts(labels, probabilities, threshold)
    metrics = calculate_metrics(counts)

    print(f"\n--- {name} ---")
    print(
        f"TP={counts['tp']:,}  FP={counts['fp']:,}  "
        f"FN={counts['fn']:,}  TN={counts['tn']:,}"
    )

    for metric_name, value in metrics.items():
        print(f"{metric_name:24s} {value:.6f}")

    return {
        "confusion_matrix": dict(counts),
        "metrics": metrics,
    }


def write_frames(path: Path, frames: list[Frame]) -> None:
    """Write normalized held-out rows in the format parse_frames accepts."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)

        for frame in frames:
            writer.writerow(
                [
                    f"{frame.timestamp:.6f}",
                    frame.can_id,
                    frame.dlc,
                    *[f"{value:02X}" for value in frame.payload],
                    frame.label,
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("../original_input_data"),
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("../build/can_ids_random_forest.joblib"),
    )
    parser.add_argument(
        "--heldout-output-dir",
        type=Path,
        default=Path("../build/heldout_test_data"),
        help=(
            "The final chronological test portions are written here so the "
            "normal common.run_folder_benchmark flow can evaluate them."
        ),
    )
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument(
        "--max-frames-per-file",
        type=int,
        default=50000,
        help=(
            "Optional chronological cap for a quicker experiment. "
            "Zero uses every frame. Frames are never randomly shuffled."
        ),
    )
    args = parser.parse_args()

    datasets = find_csv_datasets(args.input_dir)

    all_train_rows: list[dict[str, object]] = []
    all_validation_rows: list[dict[str, object]] = []
    all_test_rows: list[dict[str, object]] = []

    args.heldout_output_dir.mkdir(parents=True, exist_ok=True)

    for path in datasets:
        frames = list(parse_frames(path))

        if args.max_frames_per_file > 0:
            frames = frames[: args.max_frames_per_file]

        if len(frames) < args.window_size * 6:
            print(f"[!] Skipping {path.name}: only {len(frames):,} frames")
            continue

        feature_rows = build_feature_rows(frames, args.window_size)
        train_rows, validation_rows, test_rows = split_rows(
            feature_rows,
            purge_gap=args.window_size,
        )

        all_train_rows.extend(train_rows)
        all_validation_rows.extend(validation_rows)
        all_test_rows.extend(test_rows)

        test_row_numbers = {
            int(row["row_number"])
            for row in test_rows
        }
        heldout_frames = [
            frame
            for frame in frames
            if frame.row_number in test_row_numbers
        ]
        write_frames(
            args.heldout_output_dir / path.name,
            heldout_frames,
        )

        print(
            f"[*] {path.name}: total={len(frames):,}, "
            f"train={len(train_rows):,}, "
            f"validation={len(validation_rows):,}, "
            f"test={len(test_rows):,}"
        )

    X_train, y_train = rows_to_xy(all_train_rows)
    X_validation, y_validation = rows_to_xy(all_validation_rows)
    X_test, y_test = rows_to_xy(all_test_rows)

    if y_train.nunique() < 2:
        raise ValueError("Training split must contain normal and attack frames")

    candidates = [
    {
        "n_estimators": 10,
        "max_depth": 5,
        "min_samples_leaf": 20,
    },
    {
        "n_estimators": 25,
        "max_depth": 8,
        "min_samples_leaf": 10,
    },
]

    best_model: RandomForestClassifier | None = None
    best_threshold = 0.5
    best_validation_f1 = -1.0
    candidate_results: list[dict[str, object]] = []

    for parameters in candidates:
        model = RandomForestClassifier(
            **parameters,
            class_weight="balanced_subsample",
            max_features="sqrt",
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        probabilities = attack_probability(model, X_validation)
        threshold = choose_threshold(y_validation, probabilities)
        counts = confusion_counts(
            y_validation,
            probabilities,
            threshold,
        )
        metrics = calculate_metrics(counts)
        validation_f1 = metrics["f1"]

        result = {
            **parameters,
            "threshold": threshold,
            "validation_f1": validation_f1,
        }
        candidate_results.append(result)
        print(f"[*] Candidate: {result}")

        if validation_f1 > best_validation_f1:
            best_validation_f1 = validation_f1
            best_model = model
            best_threshold = threshold

    if best_model is None:
        raise RuntimeError("No model was trained")

    validation_result = print_metrics(
        "Validation",
        y_validation,
        attack_probability(best_model, X_validation),
        best_threshold,
    )
    test_result = print_metrics(
        "Held-out chronological test",
        y_test,
        attack_probability(best_model, X_test),
        best_threshold,
    )

    feature_importance = sorted(
        zip(FEATURE_COLUMNS, best_model.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    )

    print("\n--- Feature importance ---")
    for name, importance in feature_importance:
        print(f"{name:28s} {importance:.6f}")

    bundle = {
        "model": best_model,
        "threshold": best_threshold,
        "window_size": args.window_size,
        "feature_columns": FEATURE_COLUMNS,
        "split_strategy": "chronological_60_20_20_with_purge_gap",
    }

    joblib.dump(bundle, args.model_output)
    print(f"\n[+] Saved model to {args.model_output}")
    print(
        f"[+] Saved held-out CSV files to "
        f"{args.heldout_output_dir}"
    )

    summary_path = args.model_output.with_suffix(".metrics.json")
    summary_path.write_text(
        json.dumps(
            {
                "candidate_results": candidate_results,
                "selected_threshold": best_threshold,
                "validation": validation_result,
                "test": test_result,
                "feature_importance": feature_importance,
                "training_rows": len(X_train),
                "validation_rows": len(X_validation),
                "test_rows": len(X_test),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[+] Saved metrics to {summary_path}")


if __name__ == "__main__":
    main()